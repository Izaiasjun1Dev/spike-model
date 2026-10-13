//! Forward pass on MLX for Qwen3-MoE and DeepSeek-V3 / Kimi K2 models, with
//! the routed experts streamed from the SSD. Mirrors mlx-lm's qwen3_moe.py
//! and deepseek_v3.py (absorbed MLA attention, sigmoid router with the
//! correction bias, shared experts, leading dense layers, YaRN RoPE).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::fast::{self, ScaledDotProductAttentionMask};
use mlx_rs::ops::indexing::{argmax_axis, take_along_axis, take_axis};
use mlx_rs::ops::{self, argpartition_axis};
use mlx_rs::{Array, Dtype};

use crate::cbkernel::{hadamard, Kernels};
use crate::experts::{LayerCache, Pending, Res, Store};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arch {
    Qwen3,
    DeepseekV3,
}

/// YaRN RoPE parameters (DeepSeek-V3 / Kimi K2).
struct Yarn {
    factor: f32,
    original_max: f32,
    beta_fast: f32,
    beta_slow: f32,
    mscale: f32,
    mscale_all_dim: f32,
}

pub struct Config {
    pub arch: Arch,
    pub hidden: i32,
    /// All transformer layers.
    pub layers: usize,
    /// Layers that have routed experts, in order (their position is the
    /// expert-layer index used by the expert files and arenas).
    pub moe_layers: Vec<usize>,
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub experts: usize,
    pub top_k: usize,
    pub norm_topk: bool,
    pub eps: f32,
    pub rope_theta: f32,
    pub group_size: i32,
    pub bits: i32,
    pub eos: Vec<u32>,
    // DeepSeek-V3 / Kimi K2
    q_lora_rank: Option<i32>,
    kv_lora_rank: i32,
    qk_rope: i32,
    qk_nope: i32,
    v_head: i32,
    routed_scale: f32,
    yarn: Option<Yarn>,
}

impl Config {
    fn load(pack: &Path) -> Res<Self> {
        let c: serde_json::Value = serde_json::from_slice(&std::fs::read(pack.join("config.json"))?)?;
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(pack.join("experts.json"))?)?;
        let int = |k: &str| c[k].as_i64().ok_or_else(|| format!("config.json: {k}"));
        let eos = match &c["eos_token_id"] {
            serde_json::Value::Array(v) => v.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect(),
            v => v.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        };
        let arch = match c["model_type"].as_str().unwrap_or("") {
            "qwen3_moe" => Arch::Qwen3,
            "deepseek_v3" | "kimi_k2" => Arch::DeepseekV3,
            t => return Err(format!("unsupported model_type {t}").into()),
        };
        let layers = int("num_hidden_layers")? as usize;
        let moe_layers: Vec<usize> = match meta["moe_layers"].as_array() {
            Some(v) => v.iter().map(|x| x.as_u64().unwrap() as usize).collect(),
            None => (0..layers).collect(),
        };
        let yarn = match &c["rope_scaling"] {
            serde_json::Value::Object(r) if r.get("type").and_then(|t| t.as_str()) == Some("yarn") => {
                let f = |k: &str, d: f64| r.get(k).and_then(|v| v.as_f64()).unwrap_or(d) as f32;
                Some(Yarn {
                    factor: f("factor", 1.0),
                    original_max: f("original_max_position_embeddings", 4096.0),
                    beta_fast: f("beta_fast", 32.0),
                    beta_slow: f("beta_slow", 1.0),
                    mscale: f("mscale", 1.0),
                    mscale_all_dim: f("mscale_all_dim", 0.0),
                })
            }
            _ => None,
        };
        Ok(Self {
            arch,
            hidden: int("hidden_size")? as i32,
            layers,
            moe_layers,
            heads: int("num_attention_heads")? as i32,
            kv_heads: c["num_key_value_heads"].as_i64().unwrap_or(0) as i32,
            head_dim: c["head_dim"].as_i64().unwrap_or(0) as i32,
            experts: c["num_experts"].as_i64().or(c["n_routed_experts"].as_i64()).ok_or("config.json: num_experts")? as usize,
            top_k: int("num_experts_per_tok")? as usize,
            norm_topk: c["norm_topk_prob"].as_bool().unwrap_or(true),
            eps: c["rms_norm_eps"].as_f64().ok_or("rms_norm_eps")? as f32,
            rope_theta: c["rope_theta"].as_f64().ok_or("rope_theta")? as f32,
            group_size: c["quantization"]["group_size"].as_i64().ok_or("quantization")? as i32,
            bits: c["quantization"]["bits"].as_i64().ok_or("quantization")? as i32,
            eos,
            q_lora_rank: c["q_lora_rank"].as_i64().map(|v| v as i32),
            kv_lora_rank: c["kv_lora_rank"].as_i64().unwrap_or(0) as i32,
            qk_rope: c["qk_rope_head_dim"].as_i64().unwrap_or(0) as i32,
            qk_nope: c["qk_nope_head_dim"].as_i64().unwrap_or(0) as i32,
            v_head: c["v_head_dim"].as_i64().unwrap_or(0) as i32,
            routed_scale: c["routed_scaling_factor"].as_f64().unwrap_or(1.0) as f32,
            yarn,
        })
    }
}

/// A quantized linear layer (MLX affine format, weight stored [out, in/pack],
/// or [heads, out, in/pack] for the per-head MLA projections).
struct QLinear {
    w: Array,
    s: Array,
    b: Array,
}

impl QLinear {
    fn take(weights: &mut HashMap<String, Array>, name: &str) -> Res<Self> {
        let mut get = |t: &str| weights.remove(&format!("{name}.{t}")).ok_or_else(|| format!("missing weight {name}.{t}"));
        Ok(Self { w: get("weight")?, s: get("scales")?, b: get("biases")? })
    }

    fn quantize(w: &Array, cfg: &Config) -> Res<Self> {
        let (w, s, b) = ops::quantize(w, cfg.group_size, cfg.bits)?;
        mlx_rs::transforms::eval([&w, &s, &b])?;
        Ok(Self { w, s, b })
    }

    fn forward(&self, x: &Array, cfg: &Config) -> Res<Array> {
        self.matmul(x, true, cfg)
    }

    fn matmul(&self, x: &Array, transpose: bool, cfg: &Config) -> Res<Array> {
        Ok(ops::quantized_matmul(x, &self.w, &self.s, &self.b, transpose, cfg.group_size, cfg.bits)?)
    }
}

/// A linear layer that may be quantized (Qwen3's router) or plain bf16
/// (DeepSeek-V3's router, which mlx-lm leaves unquantized).
enum Linear {
    Q(QLinear),
    Dense(Array),
}

impl Linear {
    fn take(w: &mut HashMap<String, Array>, name: &str) -> Res<Self> {
        if w.contains_key(&format!("{name}.scales")) {
            Ok(Linear::Q(QLinear::take(w, name)?))
        } else {
            Ok(Linear::Dense(w.remove(&format!("{name}.weight")).ok_or_else(|| format!("missing weight {name}.weight"))?))
        }
    }

    fn forward(&self, x: &Array, cfg: &Config) -> Res<Array> {
        match self {
            Linear::Q(q) => q.forward(x, cfg),
            Linear::Dense(w) => Ok(ops::matmul(x, &ops::transpose_axes(w, &[1, 0])?)?),
        }
    }
}

struct Mlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl Mlp {
    fn take(w: &mut HashMap<String, Array>, p: &str) -> Res<Self> {
        Ok(Self {
            gate: QLinear::take(w, &format!("{p}.gate_proj"))?,
            up: QLinear::take(w, &format!("{p}.up_proj"))?,
            down: QLinear::take(w, &format!("{p}.down_proj"))?,
        })
    }

    fn forward(&self, x: &Array, cfg: &Config) -> Res<Array> {
        let act = ops::multiply(&mlx_rs::nn::silu(&self.gate.forward(x, cfg)?)?, &self.up.forward(x, cfg)?)?;
        self.down.forward(&act, cfg)
    }
}

enum Attn {
    Qwen3 {
        q: QLinear,
        k: QLinear,
        v: QLinear,
        o: QLinear,
        q_norm: Array,
        k_norm: Array,
    },
    /// Multi-head latent attention in mlx-lm's absorbed form: the cache
    /// holds the 512-d latent and the 64-d rope part of each token.
    Mla {
        q_a: Option<(QLinear, Array)>, // (q_a_proj, q_a_layernorm) when q_lora_rank is set
        q: QLinear,                    // q_proj or q_b_proj
        kv_a: QLinear,
        kv_a_norm: Array,
        embed_q: QLinear,     // [heads, kv_lora, qk_nope]: q_nope -> latent space
        unembed_out: QLinear, // [heads, v_head, kv_lora]: latent -> v space
        o: QLinear,
        freqs: Option<Array>,
        mscale: f32,
        scale: f32,
    },
}

enum Ffn {
    Dense(Mlp),
    Moe {
        gate: Linear,
        /// DeepSeek-V3's e_score_correction_bias (selection only).
        e_bias: Option<Array>,
        shared: Option<Mlp>,
    },
}

struct Layer {
    input_norm: Array,
    post_norm: Array,
    attn: Attn,
    ffn: Ffn,
    cache: LayerCache,
    /// Expert-layer index (position in cfg.moe_layers); usize::MAX if dense.
    xl: usize,
}

/// Per-layer attention state: (keys, values) for Qwen3, (latent, k_pe) for MLA.
#[derive(Default)]
pub struct KvCache {
    kv: Vec<Option<(Array, Array)>>,
    pub offset: i32,
}

pub struct Model {
    pub cfg: Config,
    embed: QLinear,
    layers: Vec<Layer>,
    norm: Array,
    lm_head: QLinear,
    pub store: Store,
    /// Added to the log of the selection score of resident experts when routing.
    pub bias: f32,
    /// Time blocked on the GPU at the per-layer routing sync.
    pub t_sync: std::time::Duration,
    /// While layer l runs, read the experts layer l+1's router picks on
    /// layer l's input (pre-gating), so the SSD works while the GPU does.
    pub prefetch: bool,
    /// Serve the lowest-weight half of a token's picks from the codebook
    /// copy when the q4 copy is not resident (needs experts_lo.bin).
    pub tiered: bool,
    /// Drop the lowest-weight pick when it is a miss (no read, no compute).
    pub skip_last_miss: bool,
    /// Every pick may use the codebook copy (quality x1.14, bytes x0.56).
    pub lo_all: bool,
    /// Prefill with codebook copies too (faster, costs more quality).
    pub lo_prefill: bool,
    pub skipped: u64,
    /// Prefetched as a codebook copy, then needed as q4.
    pub prefetch_lo_wrong: u64,
    /// Prefetches in flight, per expert layer, keyed by (expert, lo copy?).
    pending: Vec<HashMap<(usize, bool), Pending>>,
    /// Codebooks per expert layer: [gate, up, down], each [K, D] f16.
    codebooks: Vec<[Array; 3]>,
    kernels: Kernels,
}

pub struct Stats {
    pub hit: f64,
    pub gb_read: f64,
    /// q4 and codebook experts cached per layer.
    pub cap_hi: usize,
    pub cap_lo: usize,
    /// RAM held by the expert arenas.
    pub arena_gb: f64,
}

fn norm(w: &mut HashMap<String, Array>, p: &str, n: &str) -> Res<Array> {
    w.remove(&format!("{p}.{n}.weight")).ok_or_else(|| format!("missing {p}.{n}").into())
}

/// YaRN frequencies, as mlx-lm's YarnRoPE computes them.
fn yarn_freqs(dims: i32, base: f32, y: &Yarn) -> Array {
    let d = dims as f32;
    let corr = |rot: f32| d * (y.original_max / (rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln());
    let low = corr(y.beta_fast).floor().max(0.0);
    let high = corr(y.beta_slow).ceil().min(d - 1.0);
    let high = if low == high { high + 0.001 } else { high };
    let half = (dims / 2) as usize;
    let f: Vec<f32> = (0..half)
        .map(|i| {
            let extra = base.powf(2.0 * i as f32 / d);
            let inter = y.factor * extra;
            let ramp = ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let mask = 1.0 - ramp;
            inter * extra / (inter * mask + extra * (1.0 - mask))
        })
        .collect();
    Array::from_slice(&f, &[half as i32])
}

fn yarn_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 { 1.0 } else { 0.1 * mscale * scale.ln() + 1.0 }
}

impl Model {
    /// `cache_gb` is the RAM for experts; `lo_frac` of it holds codebook
    /// copies when the pack has them (0 disables the two-resolution mode).
    pub fn load(pack: &Path, cache_gb: f64, lo_frac: f64, lo_all: bool, bias: f32, threads: usize) -> Res<Self> {
        let cfg = Config::load(pack)?;
        let n_moe = cfg.moe_layers.len();
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(pack.join("experts.json"))?)?;
        let hi_bytes = meta["slot_bytes"].as_u64().ok_or("experts.json: slot_bytes")? as usize;
        let tiered = lo_frac > 0.0 && pack.join("experts_lo.bin").exists();
        let lo_bytes = if tiered {
            let m: serde_json::Value = serde_json::from_slice(&std::fs::read(pack.join("experts_lo.json"))?)?;
            m["slot_bytes"].as_u64().ok_or("experts_lo.json: slot_bytes")? as usize
        } else {
            0
        };
        let lo_frac = if tiered { lo_frac } else { 0.0 };
        let per_layer = cache_gb * 1e9 / n_moe as f64;
        let cap_hi = ((per_layer * (1.0 - lo_frac)) as usize / hi_bytes).min(cfg.experts);
        let cap_lo = if tiered { ((per_layer * lo_frac) as usize / lo_bytes).min(cfg.experts) } else { 0 };
        // Besides the caches, the arenas need room for one layer's misses in
        // a prefill (up to every expert, q4) and the next layer's prefetches.
        let store = Store::open(
            pack,
            n_moe * cap_hi + cfg.experts + cfg.top_k,
            // With every pick in codebook, a prefill may read all experts.
            if tiered { n_moe * cap_lo + 2 * cfg.top_k + if lo_all { cfg.experts } else { 0 } } else { 0 },
            threads,
        )?;
        let codebooks = if tiered {
            let cb = Array::load_safetensors(pack.join("codebooks.safetensors"))?;
            cfg.moe_layers
                .iter()
                .map(|l| -> Res<[Array; 3]> {
                    let get = |p: &str| cb.get(&format!("layers.{l}.{p}")).cloned().ok_or_else(|| format!("missing codebook layers.{l}.{p}"));
                    Ok([get("gate_proj")?, get("up_proj")?, get("down_proj")?])
                })
                .collect::<Res<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let mut w = Array::load_safetensors(pack.join("model.safetensors"))?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("model.layers.{i}");
            let (input_norm, post_norm) = (norm(&mut w, &p, "input_layernorm")?, norm(&mut w, &p, "post_attention_layernorm")?);
            let attn = match cfg.arch {
                Arch::Qwen3 => Attn::Qwen3 {
                    q_norm: norm(&mut w, &p, "self_attn.q_norm")?,
                    k_norm: norm(&mut w, &p, "self_attn.k_norm")?,
                    q: QLinear::take(&mut w, &format!("{p}.self_attn.q_proj"))?,
                    k: QLinear::take(&mut w, &format!("{p}.self_attn.k_proj"))?,
                    v: QLinear::take(&mut w, &format!("{p}.self_attn.v_proj"))?,
                    o: QLinear::take(&mut w, &format!("{p}.self_attn.o_proj"))?,
                },
                Arch::DeepseekV3 => {
                    // Split kv_b_proj into the absorbed per-head projections, as
                    // mlx-lm's sanitize does: [heads, nope+v, kv_lora] ->
                    // embed_q [heads, kv_lora, nope], unembed_out [heads, v, kv_lora].
                    let kv_b = QLinear::take(&mut w, &format!("{p}.self_attn.kv_b_proj"))?;
                    let full = ops::dequantize(&kv_b.w, &kv_b.s, &kv_b.b, cfg.group_size, cfg.bits)?;
                    let full = ops::reshape(&full, &[cfg.heads, cfg.qk_nope + cfg.v_head, cfg.kv_lora_rank])?;
                    let parts = ops::split_at_indices(&full, &[cfg.qk_nope], 1)?;
                    let wk = ops::transpose_axes(&parts[0], &[0, 2, 1])?;
                    let (yarn_freqs_a, mscale, scale) = match &cfg.yarn {
                        Some(y) => {
                            let ms = yarn_mscale(y.factor, y.mscale) / yarn_mscale(y.factor, y.mscale_all_dim);
                            let mut scale = ((cfg.qk_nope + cfg.qk_rope) as f32).powf(-0.5);
                            if y.mscale_all_dim > 0.0 && y.factor > 1.0 {
                                let s = 0.1 * y.mscale_all_dim * y.factor.ln() + 1.0;
                                scale *= s * s;
                            }
                            (Some(yarn_freqs(cfg.qk_rope, cfg.rope_theta, y)), ms, scale)
                        }
                        None => (None, 1.0, ((cfg.qk_nope + cfg.qk_rope) as f32).powf(-0.5)),
                    };
                    Attn::Mla {
                        q_a: match cfg.q_lora_rank {
                            Some(_) => Some((QLinear::take(&mut w, &format!("{p}.self_attn.q_a_proj"))?, norm(&mut w, &p, "self_attn.q_a_layernorm")?)),
                            None => None,
                        },
                        q: QLinear::take(&mut w, &format!("{p}.self_attn.{}", if cfg.q_lora_rank.is_some() { "q_b_proj" } else { "q_proj" }))?,
                        kv_a: QLinear::take(&mut w, &format!("{p}.self_attn.kv_a_proj_with_mqa"))?,
                        kv_a_norm: norm(&mut w, &p, "self_attn.kv_a_layernorm")?,
                        embed_q: QLinear::quantize(&wk, &cfg)?,
                        unembed_out: QLinear::quantize(&parts[1], &cfg)?,
                        o: QLinear::take(&mut w, &format!("{p}.self_attn.o_proj"))?,
                        freqs: yarn_freqs_a,
                        mscale,
                        scale,
                    }
                }
            };
            let xl = cfg.moe_layers.iter().position(|&l| l == i);
            let ffn = match xl {
                Some(_) => Ffn::Moe {
                    gate: Linear::take(&mut w, &format!("{p}.mlp.gate"))?,
                    e_bias: w.remove(&format!("{p}.mlp.gate.e_score_correction_bias")),
                    shared: if w.contains_key(&format!("{p}.mlp.shared_experts.gate_proj.weight")) {
                        Some(Mlp::take(&mut w, &format!("{p}.mlp.shared_experts"))?)
                    } else {
                        None
                    },
                },
                None => Ffn::Dense(Mlp::take(&mut w, &format!("{p}.mlp"))?),
            };
            layers.push(Layer {
                input_norm,
                post_norm,
                attn,
                ffn,
                cache: LayerCache::new(cfg.experts, cap_hi, cap_lo),
                xl: xl.unwrap_or(usize::MAX),
            });
        }
        let model = Self {
            embed: QLinear::take(&mut w, "model.embed_tokens")?,
            lm_head: QLinear::take(&mut w, "lm_head")?,
            norm: w.remove("model.norm.weight").ok_or("missing model.norm")?,
            pending: (0..n_moe).map(|_| HashMap::new()).collect(),
            layers,
            cfg,
            store,
            bias,
            t_sync: Default::default(),
            prefetch: true,
            tiered,
            skip_last_miss: tiered,
            lo_all: lo_all && tiered,
            lo_prefill: false,
            skipped: 0,
            prefetch_lo_wrong: 0,
            codebooks,
            kernels: Kernels::new(),
        };
        if let Some(k) = w.keys().next() {
            return Err(format!("unused weight {k}: not a pack of a supported model?").into());
        }
        Ok(model)
    }

    pub fn new_cache(&self) -> KvCache {
        KvCache { kv: (0..self.cfg.layers).map(|_| None).collect(), offset: 0 }
    }

    pub fn stats(&self) -> Stats {
        let (hits, lookups) = self.layers.iter().fold((0, 0), |(h, l), x| (h + x.cache.hits, l + x.cache.lookups));
        let lo_gb = self.store.lo.as_ref().map_or(0.0, |a| (a.slots * a.slot_bytes) as f64 / 1e9);
        Stats {
            hit: hits as f64 / lookups.max(1) as f64,
            gb_read: self.store.bytes_read as f64 / 1e9,
            cap_hi: self.layers[0].cache.cap_hi,
            cap_lo: self.layers[0].cache.cap_lo,
            arena_gb: (self.store.hi.slots * self.store.hi.slot_bytes) as f64 / 1e9 + lo_gb,
        }
    }

    /// Logits of the last position for `tokens` appended to the cache.
    pub fn forward(&mut self, tokens: &[u32], kv: &mut KvCache) -> Res<Array> {
        let ids = Array::from_slice(tokens, &[1, tokens.len() as i32]);
        let e = &self.embed;
        let mut h = ops::dequantize(
            take_axis(&e.w, &ids, 0)?,
            take_axis(&e.s, &ids, 0)?,
            &take_axis(&e.b, &ids, 0)?,
            self.cfg.group_size,
            self.cfg.bits,
        )?;
        for i in 0..self.layers.len() {
            let x = fast::rms_norm(&h, Some(&self.layers[i].input_norm), self.cfg.eps)?;
            let a = self.attention(i, &x, kv)?;
            h = ops::add(&h, &a)?;
            let x = fast::rms_norm(&h, Some(&self.layers[i].post_norm), self.cfg.eps)?;
            let m = match &self.layers[i].ffn {
                Ffn::Dense(mlp) => mlp.forward(&x, &self.cfg)?,
                Ffn::Moe { .. } => {
                    let routed = self.moe(i, &x)?;
                    match &self.layers[i].ffn {
                        Ffn::Moe { shared: Some(mlp), .. } => ops::add(&routed, &mlp.forward(&x, &self.cfg)?)?,
                        _ => routed,
                    }
                }
            };
            h = ops::add(&h, &m)?;
        }
        kv.offset += tokens.len() as i32;
        let last = take_axis(&h, &Array::from_int(tokens.len() as i32 - 1), 1)?;
        let x = fast::rms_norm(&last, Some(&self.norm), self.cfg.eps)?;
        self.lm_head.forward(&x, &self.cfg)
    }

    fn attention(&self, i: usize, x: &Array, kv: &mut KvCache) -> Res<Array> {
        let cfg = &self.cfg;
        let n = x.shape()[1];
        match &self.layers[i].attn {
            Attn::Qwen3 { q, k, v, o, q_norm, k_norm } => {
                let heads = |p: &QLinear, nh: i32, norm: Option<&Array>| -> Res<Array> {
                    let mut t = ops::reshape(p.forward(x, cfg)?, &[1, n, nh, cfg.head_dim])?;
                    if let Some(w) = norm {
                        t = fast::rms_norm(&t, Some(w), cfg.eps)?;
                    }
                    Ok(ops::transpose_axes(&t, &[0, 2, 1, 3])?)
                };
                let rope = |t: Array| fast::rope(&t, cfg.head_dim, false, cfg.rope_theta, 1.0, kv.offset, None);
                let qh = rope(heads(q, cfg.heads, Some(q_norm))?)?;
                let kh = rope(heads(k, cfg.kv_heads, Some(k_norm))?)?;
                let vh = heads(v, cfg.kv_heads, None)?;
                let (kh, vh) = match kv.kv[i].take() {
                    Some((pk, pv)) => (ops::concatenate(&[pk, kh], 2)?, ops::concatenate(&[pv, vh], 2)?),
                    None => (kh, vh),
                };
                let scale = (cfg.head_dim as f32).powf(-0.5);
                let mask = if n > 1 { Some(ScaledDotProductAttentionMask::Causal) } else { None };
                let out = fast::scaled_dot_product_attention(&qh, &kh, &vh, scale, mask, None)?;
                kv.kv[i] = Some((kh, vh));
                let out = ops::reshape(ops::transpose_axes(&out, &[0, 2, 1, 3])?, &[1, n, cfg.heads * cfg.head_dim])?;
                o.forward(&out, cfg)
            }
            Attn::Mla { q_a, q, kv_a, kv_a_norm, embed_q, unembed_out, o, freqs, mscale, scale } => {
                let (nope, pe, lora) = (cfg.qk_nope, cfg.qk_rope, cfg.kv_lora_rank);
                let qx = match q_a {
                    Some((a, norm)) => q.forward(&fast::rms_norm(&a.forward(x, cfg)?, Some(norm), 1e-6)?, cfg)?,
                    None => q.forward(x, cfg)?,
                };
                let qh = ops::transpose_axes(&ops::reshape(&qx, &[1, n, cfg.heads, nope + pe])?, &[0, 2, 1, 3])?; // [1, H, n, 192]
                let qs = ops::split_at_indices(&qh, &[nope], -1)?;
                let (q_nope, q_pe) = (&qs[0], &qs[1]);
                let ckv = kv_a.forward(x, cfg)?; // [1, n, lora + pe]
                let cs = ops::split_at_indices(&ckv, &[lora], -1)?;
                let latent = ops::expand_dims_axes(&fast::rms_norm(&cs[0], Some(kv_a_norm), 1e-6)?, &[1])?; // [1, 1, n, lora]
                let k_pe = ops::transpose_axes(&ops::reshape(&cs[1], &[1, n, 1, pe])?, &[0, 2, 1, 3])?; // [1, 1, n, pe]
                let rope = |t: &Array| -> Res<Array> {
                    let t = if *mscale != 1.0 { ops::multiply(t, &Array::from_f32(*mscale).as_dtype(t.dtype())?)? } else { t.clone() };
                    Ok(match freqs {
                        Some(f) => fast::rope(&t, pe, true, None, 1.0, kv.offset, Some(f))?,
                        None => fast::rope(&t, pe, true, cfg.rope_theta, 1.0, kv.offset, None)?,
                    })
                };
                let (q_pe, k_pe) = (rope(q_pe)?, rope(&k_pe)?);
                let (latent, k_pe) = match kv.kv[i].take() {
                    Some((pl, pk)) => (ops::concatenate(&[pl, latent], 2)?, ops::concatenate(&[pk, k_pe], 2)?),
                    None => (latent, k_pe),
                };
                // Rope part of the scores, used as the additive mask of the
                // attention over the latent part.
                let scale_a = Array::from_f32(*scale).as_dtype(q_pe.dtype())?;
                let mut pe_scores = ops::matmul(&ops::multiply(&q_pe, &scale_a)?, &ops::transpose_axes(&k_pe, &[0, 1, 3, 2])?)?; // [1, H, n, T]
                if n > 1 {
                    let t = k_pe.shape()[2];
                    let mut m = vec![0f32; (n * t) as usize];
                    for r in 0..n {
                        for c in 0..t {
                            if c > t - n + r {
                                m[(r * t + c) as usize] = f32::NEG_INFINITY;
                            }
                        }
                    }
                    let m = Array::from_slice(&m, &[1, 1, n, t]).as_dtype(pe_scores.dtype())?;
                    pe_scores = ops::add(&pe_scores, &m)?;
                }
                let out = if n == 1 {
                    let ql = embed_q.matmul(q_nope, true, cfg)?; // [1, H, 1, lora]
                    let a = fast::scaled_dot_product_attention(&ql, &latent, &latent, *scale, Some(ScaledDotProductAttentionMask::Array(&pe_scores)), None)?;
                    unembed_out.matmul(&a, true, cfg)? // [1, H, 1, v]
                } else {
                    let kh = embed_q.matmul(&latent, false, cfg)?; // [1, H, T, nope]
                    let vh = unembed_out.matmul(&latent, true, cfg)?; // [1, H, T, v]
                    fast::scaled_dot_product_attention(q_nope, &kh, &vh, *scale, Some(ScaledDotProductAttentionMask::Array(&pe_scores)), None)?
                };
                kv.kv[i] = Some((latent, k_pe));
                let out = ops::reshape(ops::transpose_axes(&out, &[0, 2, 1, 3])?, &[1, n, cfg.heads * cfg.v_head])?;
                o.forward(&out, cfg)
            }
        }
    }

    /// Router of `layer` on `x`, biased toward that layer's resident experts:
    /// (selection scores, top-k indices, their mixing weights).
    fn route(&self, layer: usize, x: &Array) -> Res<(Array, Array, Array)> {
        let k = self.cfg.top_k as i32;
        let n_exp = self.cfg.experts as i32;
        let Ffn::Moe { gate, e_bias, .. } = &self.layers[layer].ffn else { return Err("route on a dense layer".into()) };
        let logits = gate.forward(x, &self.cfg)?;
        // Qwen3 picks by softmax probability; DeepSeek-V3 by sigmoid score plus
        // a per-expert correction, and mixes with the uncorrected scores.
        let (probs, sel) = match self.cfg.arch {
            Arch::Qwen3 => {
                // Stock mlx-lm picks on the bf16 probabilities; selecting on
                // f32 logs would break its near-ties.
                let p = ops::softmax_axis(&logits, -1, true)?;
                let sel = if self.bias != 0.0 { ops::log(&p.as_type::<f32>()?)? } else { p.clone() };
                (p, sel)
            }
            Arch::DeepseekV3 => {
                let p = ops::sigmoid(&logits.as_type::<f32>()?)?;
                let corrected = match e_bias {
                    Some(b) => ops::add(&p, &b.as_type::<f32>()?)?,
                    None => p.clone(),
                };
                (p, ops::log(&ops::maximum(&corrected, &Array::from_f32(1e-6))?)?)
            }
        };
        let sel = if self.bias != 0.0 {
            let mut res = vec![0f32; self.cfg.experts];
            for e in 0..self.cfg.experts {
                if self.layers[layer].cache.resident(e) {
                    res[e] = self.bias;
                }
            }
            ops::add(&sel, &Array::from_slice(&res, &[n_exp]))?
        } else {
            sel
        };
        let top = Array::from_iter(n_exp - k..n_exp, &[k]);
        let inds = take_axis(argpartition_axis(&sel, -k, -1)?, &top, -1)?;
        let mut scores = take_along_axis(&probs, &inds, -1)?;
        if self.cfg.norm_topk {
            scores = ops::divide(&scores, &ops::sum_axis(&scores, -1, true)?)?;
        }
        if self.cfg.routed_scale != 1.0 {
            scores = ops::multiply(&scores, &Array::from_f32(self.cfg.routed_scale).as_dtype(scores.dtype())?)?;
        }
        Ok((probs, inds, scores.as_dtype(x.dtype())?))
    }

    /// Which copy a pick needs: the q4 one for the top half by weight, the
    /// codebook one for the rest (any resident copy serves those).
    fn needs_lo(&self, rank: usize) -> bool {
        self.tiered && (self.lo_all || rank >= self.cfg.top_k / 2)
    }

    /// Routed experts: route, read the misses, run the needed experts.
    fn moe(&mut self, i: usize, x: &Array) -> Res<Array> {
        let k = self.cfg.top_k;
        let xl = self.layers[i].xl;
        let next = if i + 1 < self.layers.len() && self.layers[i + 1].xl != usize::MAX { Some(i + 1) } else { None };
        let n_tok = x.shape()[1] as usize;
        let decode = n_tok == 1;
        let (_, inds, scores) = self.route(i, x)?;

        // The one sync per layer: which experts does this call need? In decode
        // it also yields the guess for the next layer.
        // take_axis leaves a strided view; flatten copies it to read on the CPU.
        let flat = ops::flatten(&inds, None, None)?;
        let flat_w = ops::flatten(&scores.as_type::<f32>()?, None, None)?;
        let guess = if self.prefetch && decode && next.is_some() {
            let (_, gi, gw) = self.route(next.unwrap(), x)?;
            Some((ops::flatten(&gi, None, None)?, ops::flatten(&gw.as_type::<f32>()?, None, None)?))
        } else {
            None
        };
        let t0 = std::time::Instant::now();
        let mut to_eval = vec![&flat, &flat_w];
        if let Some((gi, gw)) = &guess {
            to_eval.extend([gi, gw]);
        }
        mlx_rs::transforms::eval(to_eval)?;
        self.t_sync += t0.elapsed();
        let picks: Vec<usize> = flat.as_slice::<u32>().iter().map(|&p| p as usize).collect();
        let weights: Vec<f32> = flat_w.as_slice::<f32>().to_vec();

        // Rank of each pick within its token (0 = highest weight), and the
        // copy it needs. Prefills use q4 for everything.
        let mut rank = vec![0usize; picks.len()];
        for t in 0..n_tok {
            let mut order: Vec<usize> = (t * k..(t + 1) * k).collect();
            order.sort_by(|a, b| weights[*b].total_cmp(&weights[*a]));
            for (r, &j) in order.iter().enumerate() {
                rank[j] = r;
            }
        }
        let need_lo: Vec<bool> = rank.iter().map(|&r| if decode { self.needs_lo(r) } else { self.tiered && self.lo_all && self.lo_prefill }).collect();

        // What serves each pick: a resident copy, a read, or nothing.
        #[derive(Clone, Copy)]
        enum Serve {
            Hi(usize),
            Lo(usize),
            Read(bool),
            Skip,
        }
        let cache = &mut self.layers[i].cache;
        cache.lookups += picks.len() as u64;
        let mut serve = Vec::with_capacity(picks.len());
        let mut uses: HashMap<(usize, bool), usize> = HashMap::new();
        for (j, &e) in picks.iter().enumerate() {
            *uses.entry((e, !need_lo[j])).or_default() += 1;
            let s = match (cache.hi.get(&e), cache.lo.get(&e), need_lo[j]) {
                (Some(&s), _, _) => Serve::Hi(s),
                (None, Some(&s), true) => Serve::Lo(s),
                _ if decode && self.skip_last_miss && rank[j] == k - 1 => Serve::Skip,
                _ => Serve::Read(need_lo[j]),
            };
            if matches!(s, Serve::Hi(_) | Serve::Lo(_)) {
                cache.hits += 1;
            }
            serve.push(s);
        }
        let mut uses: Vec<(usize, bool, usize)> = uses.into_iter().map(|((e, high), c)| (e, high, c)).collect();
        uses.sort_unstable();

        // Misses: take over a prefetch already under way, else read on demand.
        let mut reads: HashMap<(usize, bool), Pending> = HashMap::new();
        for (j, &e) in picks.iter().enumerate() {
            let Serve::Read(lo) = serve[j] else { continue };
            if reads.contains_key(&(e, lo)) || (lo && reads.contains_key(&(e, false))) {
                continue;
            }
            // A pending q4 copy serves a low pick too; a pending codebook copy
            // cannot serve a high one (it is dropped as stale below).
            let key = if self.pending[xl].contains_key(&(e, lo)) { (e, lo) } else { (e, false) };
            if !lo && self.pending[xl].contains_key(&(e, true)) {
                self.prefetch_lo_wrong += 1;
            }
            match self.pending[xl].remove(&key) {
                Some(p) => {
                    self.store.promote(&p);
                    self.store.prefetch_used += 1;
                    reads.insert(key, p);
                }
                None => {
                    let p = self.store.start(xl, e, lo, true)?.ok_or("expert arena full")?;
                    reads.insert((e, lo), p);
                }
            }
        }
        let stale: Vec<Pending> = self.pending[xl].drain().map(|(_, p)| p).collect();
        // Queue the guesses for the next layer behind the demand reads.
        if let Some((gi, gw)) = &guess {
            let g: Vec<usize> = gi.as_slice::<u32>().iter().map(|&e| e as usize).collect();
            let w = gw.as_slice::<f32>();
            let mut order: Vec<usize> = (0..g.len()).collect();
            order.sort_by(|a, b| w[*b].total_cmp(&w[*a]));
            let (ni, nxl) = (next.unwrap(), self.layers[next.unwrap()].xl);
            let next = &self.layers[ni].cache;
            for (r, &j) in order.iter().enumerate() {
                if self.skip_last_miss && r == k - 1 {
                    continue; // a miss on the lowest pick is skipped, not read
                }
                let (e, lo) = (g[j], self.needs_lo(r));
                let served = next.hi.contains_key(&e) || (lo && next.lo.contains_key(&e));
                let queued = self.pending[nxl].contains_key(&(e, false)) || (lo && self.pending[nxl].contains_key(&(e, true)));
                if !served && !queued {
                    if let Some(p) = self.store.start(nxl, e, lo, false)? {
                        self.pending[nxl].insert((e, lo), p);
                    }
                }
            }
        }
        for p in stale {
            self.store.discard(p)?;
        }
        let mut reads: Vec<((usize, bool), Pending)> = reads.into_iter().collect();
        reads.sort_by_key(|(k, _)| *k);
        let mut fetched: Vec<(usize, bool, usize)> = Vec::new();
        for ((e, lo), p) in reads {
            let slot = self.store.finish(p)?;
            fetched.push((e, lo, slot));
        }
        for (j, &e) in picks.iter().enumerate() {
            if let Serve::Read(lo) = serve[j] {
                let f = fetched.iter().find(|f| f.0 == e && (!f.1 || lo)).unwrap();
                serve[j] = if f.1 { Serve::Lo(f.2) } else { Serve::Hi(f.2) };
            }
        }

        // Run the picks served by the q4 arena with one gather_qmm per
        // projection (same math as mlx-lm's SwitchGLU, including its sort for
        // many tokens), and the rest with the codebook kernel.
        let (gs, bits) = (self.store.group_size, self.store.bits);
        let hi = &self.store.hi;
        let qmm = |h: &Array, part: &str, idx: &Array, sorted: bool| -> Res<Array> {
            let t = |n: &str| hi.array(&format!("{part}.{n}"));
            Ok(ops::gather_qmm(h, t("weight"), t("scales"), t("biases"), None, idx, true, gs, bits, sorted)?)
        };
        let dtype = x.dtype();
        let hidden = x.shape()[2];
        // Rows served by the q4 arena run through one gather_qmm per
        // projection (mlx-lm's SwitchGLU math, rows sorted by slot); rows
        // served by codebook copies through the codebook kernels. Each row's
        // input is its token's x. Skipped rows contribute a zero vector.
        let n_rows = picks.len();
        let x_flat = ops::reshape(x, &[n_tok as i32, hidden])?;
        let mut hi_rows: Vec<(usize, u32)> = Vec::new();
        let mut lo_rows: Vec<(usize, u32)> = Vec::new();
        for (j, s) in serve.iter().enumerate() {
            match s {
                Serve::Hi(slot) => hi_rows.push((j, *slot as u32)),
                Serve::Lo(slot) => lo_rows.push((j, *slot as u32)),
                _ => {}
            }
        }
        if n_tok > 1 {
            // Sorting changes the order of the weighted sum, i.e. its
            // rounding; mlx-lm only sorts in prefills, so match it.
            hi_rows.sort_by_key(|&(_, slot)| slot);
        }
        let mut parts: Vec<Array> = Vec::new();
        let mut order: Vec<usize> = Vec::new(); // row j of each output row, in order
        if !hi_rows.is_empty() {
            let toks: Vec<u32> = hi_rows.iter().map(|&(j, _)| (j / k) as u32).collect();
            let slots: Vec<u32> = hi_rows.iter().map(|&(_, s)| s).collect();
            let hs = if n_tok == 1 {
                ops::expand_dims_axes(x, &[-2, -3])? // [1, 1, 1, 1, hidden], broadcast over the rows
            } else {
                ops::expand_dims_axes(&take_axis(&x_flat, &Array::from_slice(&toks, &[toks.len() as i32]), 0)?, &[1])? // [n, 1, hidden]
            };
            let is = if n_tok == 1 { Array::from_slice(&slots, &[1, 1, slots.len() as i32]) } else { Array::from_slice(&slots, &[slots.len() as i32]) };
            let sorted = n_tok > 1 && slots.len() >= 64;
            let up = qmm(&hs, "up_proj", &is, sorted)?;
            let gate = qmm(&hs, "gate_proj", &is, sorted)?;
            let act = ops::multiply(&mlx_rs::nn::silu(&gate)?, &up)?;
            parts.push(ops::reshape(&qmm(&act, "down_proj", &is, sorted)?, &[slots.len() as i32, hidden])?);
            order.extend(hi_rows.iter().map(|&(j, _)| j));
        }
        if !lo_rows.is_empty() {
            let lo = self.store.lo.as_ref().unwrap();
            let toks: Vec<u32> = lo_rows.iter().map(|&(j, _)| (j / k) as u32).collect();
            let slots: Vec<u32> = lo_rows.iter().map(|&(_, s)| s).collect();
            let eids = Array::from_slice(&slots, &[slots.len() as i32]);
            let xr = hadamard(&x_flat.as_type::<f32>()?)?; // [n_tok, hidden]
            let xr = if n_tok == 1 { xr } else { take_axis(&xr, &Array::from_slice(&toks, &[toks.len() as i32]), 0)? };
            let [cb_gate, cb_up, cb_down] = &self.codebooks[xl];
            let act = self.kernels.gate_up(
                &xr, &eids,
                lo.array("gate_proj.codes"), lo.array("up_proj.codes"),
                lo.array("gate_proj.scale"), lo.array("up_proj.scale"),
                cb_gate, cb_up,
            )?; // [n, inter] f32, silu(gate) * up
            let ar = hadamard(&act)?;
            parts.push(self.kernels.qmv(&ar, &eids, lo.array("down_proj.codes"), lo.array("down_proj.scale"), cb_down, dtype)?);
            order.extend(lo_rows.iter().map(|&(j, _)| j));
        }
        let n_skip = serve.iter().filter(|s| matches!(s, Serve::Skip)).count();
        self.skipped += n_skip as u64;
        let out = if n_tok == 1 {
            // One token: weighted sum of the output rows, in `order`.
            let y_all = if parts.len() == 1 { parts.pop().unwrap() } else { ops::concatenate(&parts, 0)? };
            let w: Vec<f32> = order.iter().map(|&j| weights[j]).collect();
            let w = Array::from_slice(&w, &[w.len() as i32, 1]).as_dtype(dtype)?;
            ops::reshape(&ops::sum_axis(&ops::multiply(&y_all, &w)?, 0, false)?, &[1, 1, hidden])?
        } else {
            // Several tokens: put the rows back in router order (skipped
            // picks point at a zero row), then sum each token's k rows.
            parts.push(ops::zeros_dtype(&[1, hidden], dtype)?);
            let y_all = ops::concatenate(&parts, 0)?;
            let zero_row = (n_rows - n_skip) as u32;
            let mut pos = vec![zero_row; n_rows];
            for (p, &j) in order.iter().enumerate() {
                pos[j] = p as u32;
            }
            let y = take_axis(&y_all, &Array::from_slice(&pos, &[n_rows as i32]), 0)?;
            let w: Vec<f32> = serve.iter().zip(&weights).map(|(s, w)| if matches!(s, Serve::Skip) { 0.0 } else { *w }).collect();
            let w = Array::from_slice(&w, &[n_tok as i32, k as i32, 1]).as_dtype(dtype)?;
            let y = ops::reshape(&y, &[n_tok as i32, k as i32, hidden])?;
            ops::reshape(&ops::sum_axis(&ops::multiply(&y, &w)?, 1, false)?, &[1, n_tok as i32, hidden])?
        };

        let gone = self.layers[i].cache.update(&uses, fetched);
        self.store.release(gone);
        Ok(out)
    }
}

/// Greedy next token.
pub fn argmax(logits: &Array) -> Res<u32> {
    let t = argmax_axis(logits, -1, false)?;
    t.eval()?;
    Ok(t.as_slice::<u32>()[0])
}
