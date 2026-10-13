//! Qwen3-MoE forward pass on MLX, with the routed experts streamed from the
//! SSD. Mirrors mlx-lm's qwen3_moe.py (and engine/engine.py for the MoE).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::fast::{self, ScaledDotProductAttentionMask};
use mlx_rs::ops::indexing::{argmax_axis, take_along_axis, take_axis};
use mlx_rs::ops::{self, argpartition_axis};
use mlx_rs::Array;

use crate::cbkernel::{hadamard, Kernels};
use crate::experts::{LayerCache, Pending, Res, Store};

pub struct Config {
    pub hidden: i32,
    pub layers: usize,
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
}

impl Config {
    fn load(pack: &Path) -> Res<Self> {
        let c: serde_json::Value = serde_json::from_slice(&std::fs::read(pack.join("config.json"))?)?;
        let int = |k: &str| c[k].as_i64().ok_or_else(|| format!("config.json: {k}"));
        let eos = match &c["eos_token_id"] {
            serde_json::Value::Array(v) => v.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect(),
            v => v.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        };
        Ok(Self {
            hidden: int("hidden_size")? as i32,
            layers: int("num_hidden_layers")? as usize,
            heads: int("num_attention_heads")? as i32,
            kv_heads: int("num_key_value_heads")? as i32,
            head_dim: int("head_dim")? as i32,
            experts: int("num_experts")? as usize,
            top_k: int("num_experts_per_tok")? as usize,
            norm_topk: c["norm_topk_prob"].as_bool().unwrap_or(true),
            eps: c["rms_norm_eps"].as_f64().ok_or("rms_norm_eps")? as f32,
            rope_theta: c["rope_theta"].as_f64().ok_or("rope_theta")? as f32,
            group_size: c["quantization"]["group_size"].as_i64().ok_or("quantization")? as i32,
            bits: c["quantization"]["bits"].as_i64().ok_or("quantization")? as i32,
            eos,
        })
    }
}

/// A quantized linear layer (MLX affine format, weight stored [out, in/pack]).
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

    fn forward(&self, x: &Array, cfg: &Config) -> Res<Array> {
        Ok(ops::quantized_matmul(x, &self.w, &self.s, &self.b, true, cfg.group_size, cfg.bits)?)
    }
}

struct Layer {
    input_norm: Array,
    post_norm: Array,
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    q_norm: Array,
    k_norm: Array,
    gate: QLinear,
    cache: LayerCache,
}

/// Keys and values seen so far, per layer: [1, kv_heads, tokens, head_dim].
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
    /// Added to the log-probability of resident experts when routing.
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
    /// Prefetches in flight, per layer, keyed by (expert, lo copy?).
    pending: Vec<HashMap<(usize, bool), Pending>>,
    /// Codebooks per layer: [gate, up, down], each [K, D] f16.
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

impl Model {
    /// `cache_gb` is the RAM for experts; `lo_frac` of it holds codebook
    /// copies when the pack has them (0 disables the two-resolution mode).
    pub fn load(pack: &Path, cache_gb: f64, lo_frac: f64, lo_all: bool, bias: f32, threads: usize) -> Res<Self> {
        let cfg = Config::load(pack)?;
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
        let per_layer = cache_gb * 1e9 / cfg.layers as f64;
        let cap_hi = ((per_layer * (1.0 - lo_frac)) as usize / hi_bytes).min(cfg.experts);
        let cap_lo = if tiered { ((per_layer * lo_frac) as usize / lo_bytes).min(cfg.experts) } else { 0 };
        // Besides the caches, the arenas need room for one layer's misses in
        // a prefill (up to every expert, q4) and the next layer's prefetches.
        let store = Store::open(
            pack,
            cfg.layers * cap_hi + cfg.experts + cfg.top_k,
            // With every pick in codebook, a prefill may read all experts.
            if tiered { cfg.layers * cap_lo + 2 * cfg.top_k + if lo_all { cfg.experts } else { 0 } } else { 0 },
            threads,
        )?;
        let codebooks = if tiered {
            let cb = Array::load_safetensors(pack.join("codebooks.safetensors"))?;
            (0..cfg.layers)
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
            let mut norm = |n: &str| w.remove(&format!("{p}.{n}.weight")).ok_or_else(|| format!("missing {p}.{n}"));
            let (input_norm, post_norm) = (norm("input_layernorm")?, norm("post_attention_layernorm")?);
            let (q_norm, k_norm) = (norm("self_attn.q_norm")?, norm("self_attn.k_norm")?);
            layers.push(Layer {
                input_norm,
                post_norm,
                q_norm,
                k_norm,
                q: QLinear::take(&mut w, &format!("{p}.self_attn.q_proj"))?,
                k: QLinear::take(&mut w, &format!("{p}.self_attn.k_proj"))?,
                v: QLinear::take(&mut w, &format!("{p}.self_attn.v_proj"))?,
                o: QLinear::take(&mut w, &format!("{p}.self_attn.o_proj"))?,
                gate: QLinear::take(&mut w, &format!("{p}.mlp.gate"))?,
                cache: LayerCache::new(cfg.experts, cap_hi, cap_lo),
            });
        }
        let model = Self {
            embed: QLinear::take(&mut w, "model.embed_tokens")?,
            lm_head: QLinear::take(&mut w, "lm_head")?,
            norm: w.remove("model.norm.weight").ok_or("missing model.norm")?,
            pending: (0..cfg.layers).map(|_| HashMap::new()).collect(),
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
            return Err(format!("unused weight {k}: model.safetensors is not a Qwen3-MoE pack?").into());
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
            let m = self.moe(i, &x)?;
            h = ops::add(&h, &m)?;
        }
        kv.offset += tokens.len() as i32;
        let last = take_axis(&h, &Array::from_int(tokens.len() as i32 - 1), 1)?;
        let x = fast::rms_norm(&last, Some(&self.norm), self.cfg.eps)?;
        self.lm_head.forward(&x, &self.cfg)
    }

    fn attention(&self, i: usize, x: &Array, kv: &mut KvCache) -> Res<Array> {
        let (cfg, l) = (&self.cfg, &self.layers[i]);
        let n = x.shape()[1];
        let heads = |p: &QLinear, nh: i32, norm: Option<&Array>| -> Res<Array> {
            let mut t = ops::reshape(p.forward(x, cfg)?, &[1, n, nh, cfg.head_dim])?;
            if let Some(w) = norm {
                t = fast::rms_norm(&t, Some(w), cfg.eps)?;
            }
            Ok(ops::transpose_axes(&t, &[0, 2, 1, 3])?)
        };
        let rope = |t: Array| fast::rope(&t, cfg.head_dim, false, cfg.rope_theta, 1.0, kv.offset, None);
        let q = rope(heads(&l.q, cfg.heads, Some(&l.q_norm))?)?;
        let k = rope(heads(&l.k, cfg.kv_heads, Some(&l.k_norm))?)?;
        let v = heads(&l.v, cfg.kv_heads, None)?;
        let (k, v) = match kv.kv[i].take() {
            Some((pk, pv)) => (ops::concatenate(&[pk, k], 2)?, ops::concatenate(&[pv, v], 2)?),
            None => (k, v),
        };
        let scale = (cfg.head_dim as f32).powf(-0.5);
        let mask = if n > 1 { Some(ScaledDotProductAttentionMask::Causal) } else { None };
        let o = fast::scaled_dot_product_attention(&q, &k, &v, scale, mask, None)?;
        kv.kv[i] = Some((k, v));
        let o = ops::reshape(ops::transpose_axes(&o, &[0, 2, 1, 3])?, &[1, n, cfg.heads * cfg.head_dim])?;
        l.o.forward(&o, cfg)
    }

    /// Router of `layer` on `x`, biased toward that layer's resident experts:
    /// (gates, top-k indices, their weights).
    fn route(&self, layer: usize, x: &Array) -> Res<(Array, Array, Array)> {
        let k = self.cfg.top_k as i32;
        let n_exp = self.cfg.experts as i32;
        let gates = ops::softmax_axis(self.layers[layer].gate.forward(x, &self.cfg)?, -1, true)?;
        let sel = if self.bias != 0.0 {
            let mut res = vec![0f32; self.cfg.experts];
            for e in 0..self.cfg.experts {
                if self.layers[layer].cache.resident(e) {
                    res[e] = self.bias;
                }
            }
            let res = Array::from_slice(&res, &[n_exp]);
            ops::add(&ops::log(&gates.as_type::<f32>()?)?, &res)?
        } else {
            gates.clone()
        };
        let top = Array::from_iter(n_exp - k..n_exp, &[k]);
        let inds = take_axis(argpartition_axis(&sel, -k, -1)?, &top, -1)?;
        let mut scores = take_along_axis(&gates, &inds, -1)?;
        if self.cfg.norm_topk {
            scores = ops::divide(&scores, &ops::sum_axis(&scores, -1, true)?)?;
        }
        Ok((gates, inds, scores))
    }

    /// Which copy a pick needs: the q4 one for the top half by weight, the
    /// codebook one for the rest (any resident copy serves those).
    fn needs_lo(&self, rank: usize) -> bool {
        self.tiered && (self.lo_all || rank >= self.cfg.top_k / 2)
    }

    /// Routed experts: route, read the misses, run the needed experts.
    fn moe(&mut self, i: usize, x: &Array) -> Res<Array> {
        let k = self.cfg.top_k;
        let n_tok = x.shape()[1] as usize;
        let decode = n_tok == 1;
        let (_, inds, scores) = self.route(i, x)?;

        // The one sync per layer: which experts does this call need? In decode
        // it also yields the guess for the next layer.
        // take_axis leaves a strided view; flatten copies it to read on the CPU.
        let flat = ops::flatten(&inds, None, None)?;
        let flat_w = ops::flatten(&scores.as_type::<f32>()?, None, None)?;
        let guess = if self.prefetch && decode && i + 1 < self.layers.len() {
            let (_, gi, gw) = self.route(i + 1, x)?;
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
            let key = if self.pending[i].contains_key(&(e, lo)) { (e, lo) } else { (e, false) };
            if !lo && self.pending[i].contains_key(&(e, true)) {
                self.prefetch_lo_wrong += 1;
            }
            match self.pending[i].remove(&key) {
                Some(p) => {
                    self.store.promote(&p);
                    self.store.prefetch_used += 1;
                    reads.insert(key, p);
                }
                None => {
                    let p = self.store.start(i, e, lo, true)?.ok_or("expert arena full")?;
                    reads.insert((e, lo), p);
                }
            }
        }
        let stale: Vec<Pending> = self.pending[i].drain().map(|(_, p)| p).collect();
        // Queue the guesses for the next layer behind the demand reads.
        if let Some((gi, gw)) = &guess {
            let g: Vec<usize> = gi.as_slice::<u32>().iter().map(|&e| e as usize).collect();
            let w = gw.as_slice::<f32>();
            let mut order: Vec<usize> = (0..g.len()).collect();
            order.sort_by(|a, b| w[*b].total_cmp(&w[*a]));
            let next = &self.layers[i + 1].cache;
            for (r, &j) in order.iter().enumerate() {
                if self.skip_last_miss && r == k - 1 {
                    continue; // a miss on the lowest pick is skipped, not read
                }
                let (e, lo) = (g[j], self.needs_lo(r));
                let served = next.hi.contains_key(&e) || (lo && next.lo.contains_key(&e));
                let queued = self.pending[i + 1].contains_key(&(e, false)) || (lo && self.pending[i + 1].contains_key(&(e, true)));
                if !served && !queued {
                    if let Some(p) = self.store.start(i + 1, e, lo, false)? {
                        self.pending[i + 1].insert((e, lo), p);
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
        hi_rows.sort_by_key(|&(_, slot)| slot);
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
            let [cb_gate, cb_up, cb_down] = &self.codebooks[i];
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
