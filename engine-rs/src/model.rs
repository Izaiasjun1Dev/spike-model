//! Qwen3-MoE forward pass on MLX, with the routed experts streamed from the
//! SSD. Mirrors mlx-lm's qwen3_moe.py (and engine/engine.py for the MoE).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::fast::{self, ScaledDotProductAttentionMask};
use mlx_rs::ops::indexing::{argmax_axis, take_along_axis, take_axis};
use mlx_rs::ops::{self, argpartition_axis, argsort_axis};
use mlx_rs::Array;

use crate::experts::{Expert, LayerCache, Res, Store};

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
}

pub struct Stats {
    pub hit: f64,
    pub gb_read: f64,
    pub slots: usize,
}

impl Model {
    pub fn load(pack: &Path, cache_gb: f64, bias: f32, threads: usize) -> Res<Self> {
        let cfg = Config::load(pack)?;
        let store = Store::open(pack, threads)?;
        let mut w = Array::load_safetensors(pack.join("model.safetensors"))?;
        let slots = ((cache_gb * 1e9) / (cfg.layers * store.slot) as f64) as usize;
        let slots = slots.min(cfg.experts);
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
                cache: LayerCache::new(cfg.experts, slots),
            });
        }
        let model = Self {
            embed: QLinear::take(&mut w, "model.embed_tokens")?,
            lm_head: QLinear::take(&mut w, "lm_head")?,
            norm: w.remove("model.norm.weight").ok_or("missing model.norm")?,
            layers,
            cfg,
            store,
            bias,
            t_sync: Default::default(),
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
        Stats {
            hit: hits as f64 / lookups.max(1) as f64,
            gb_read: self.store.bytes_read as f64 / 1e9,
            slots: self.layers[0].cache.slots,
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

    /// Routed experts: route, read the misses, run the needed experts.
    fn moe(&mut self, i: usize, x: &Array) -> Res<Array> {
        let k = self.cfg.top_k as i32;
        let n_exp = self.cfg.experts as i32;
        let gates = ops::softmax_axis(self.layers[i].gate.forward(x, &self.cfg)?, -1, true)?;
        let sel = if self.bias != 0.0 {
            let mut res = vec![0f32; self.cfg.experts];
            for e in self.layers[i].cache.experts.keys() {
                res[*e] = self.bias;
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

        // The one sync per layer: which experts does this call need?
        // take_axis leaves a strided view; flatten copies it to read on the CPU.
        let flat = ops::flatten(&inds, None, None)?;
        let t0 = std::time::Instant::now();
        flat.eval()?;
        self.t_sync += t0.elapsed();
        let picks: Vec<u32> = flat.as_slice::<u32>().to_vec();
        let mut counts: Vec<(usize, usize)> = Vec::new();
        let mut sorted = picks.clone();
        sorted.sort_unstable();
        for &p in &sorted {
            match counts.last_mut() {
                Some((e, c)) if *e == p as usize => *c += 1,
                _ => counts.push((p as usize, 1)),
            }
        }
        let cache = &mut self.layers[i].cache;
        cache.lookups += picks.len() as u64;
        let missing: Vec<usize> = counts.iter().map(|&(e, _)| e).filter(|e| !cache.experts.contains_key(e)).collect();
        cache.hits += counts.iter().filter(|(e, _)| cache.experts.contains_key(e)).map(|&(_, c)| c as u64).sum::<u64>();
        let fetched: HashMap<usize, Expert> = missing.iter().copied().zip(self.store.read(i, &missing)?).collect();
        let (gs, bits) = (self.store.group_size, self.store.bits);

        // One token (decode): run each expert on its own weights. Stacking them
        // for gather_qmm would copy every used expert (cached ones too) again.
        if picks.len() == self.cfg.top_k {
            let cache = &self.layers[i].cache;
            let q = |h: &Array, ex: &Expert, part: &str| -> Res<Array> {
                let t = |n: &str| &ex[&format!("{part}.{n}")];
                Ok(ops::quantized_matmul(h, t("weight"), t("scales"), t("biases"), true, gs, bits)?)
            };
            let ys = picks
                .iter()
                .map(|&p| -> Res<Array> {
                    let p = p as usize;
                    let ex = cache.experts.get(&p).or_else(|| fetched.get(&p)).unwrap();
                    let act = ops::multiply(&mlx_rs::nn::silu(&q(x, ex, "gate_proj")?)?, &q(x, ex, "up_proj")?)?;
                    q(&act, ex, "down_proj")
                })
                .collect::<Res<Vec<_>>>()?;
            let y = ops::stack(&ys, -2)?; // [1, 1, k, hidden], picks in router order
            let dtype = y.dtype();
            let out = ops::sum_axis(&ops::multiply(&y, &ops::expand_dims_axes(&scores, &[-1])?)?, -2, false)?.as_dtype(dtype)?;
            let gone = self.layers[i].cache.update(&counts, fetched);
            self.store.recycle(gone);
            return Ok(out);
        }

        // Several tokens (prefill): stack the needed experts and remap the
        // router's picks onto them.
        let cache = &self.layers[i].cache;
        let used: Vec<&Expert> = counts.iter().map(|(e, _)| cache.experts.get(e).or_else(|| fetched.get(e)).unwrap()).collect();
        let stack = |name: &str| ops::stack(&used.iter().map(|x| &x[name]).collect::<Vec<_>>(), 0);
        let mut pos = vec![0u32; self.cfg.experts];
        for (j, (e, _)) in counts.iter().enumerate() {
            pos[*e] = j as u32;
        }
        let remapped: Vec<u32> = picks.iter().map(|&p| pos[p as usize]).collect();
        let idx = Array::from_slice(&remapped, inds.shape());

        let qmm = |h: &Array, part: &str, idx: &Array, sorted: bool| -> Res<Array> {
            let (w, s, b) = (stack(&format!("{part}.weight"))?, stack(&format!("{part}.scales"))?, stack(&format!("{part}.biases"))?);
            Ok(ops::gather_qmm(h, &w, &s, &b, None, idx, true, gs, bits, sorted)?)
        };
        // Same math as mlx-lm's SwitchGLU, including its sort for many tokens.
        let h = ops::expand_dims_axes(x, &[-2, -3])?;
        let sort = idx.size() >= 64;
        let y = if sort {
            let flat = ops::flatten(&idx, None, None)?;
            let order = argsort_axis(&flat, 0)?;
            let inv = argsort_axis(&order, 0)?;
            let hs = ops::flatten(&h, 0, -3)?;
            let hs = take_axis(&hs, &ops::floor_divide(&order, &Array::from_int(k))?, 0)?;
            let is = take_axis(&flat, &order, 0)?;
            let up = qmm(&hs, "up_proj", &is, true)?;
            let gate = qmm(&hs, "gate_proj", &is, true)?;
            let act = ops::multiply(&mlx_rs::nn::silu(&gate)?, &up)?;
            let y = take_axis(&qmm(&act, "down_proj", &is, true)?, &inv, 0)?;
            let mut shape = idx.shape().to_vec();
            shape.extend_from_slice(&y.shape()[1..]);
            ops::reshape(&y, &shape)?
        } else {
            let up = qmm(&h, "up_proj", &idx, false)?;
            let gate = qmm(&h, "gate_proj", &idx, false)?;
            let act = ops::multiply(&mlx_rs::nn::silu(&gate)?, &up)?;
            qmm(&act, "down_proj", &idx, false)?
        };
        let y = ops::squeeze_axes(&y, &[-2])?;
        let dtype = y.dtype();
        let out = ops::sum_axis(&ops::multiply(&y, &ops::expand_dims_axes(&scores, &[-1])?)?, -2, false)?.as_dtype(dtype)?;

        let gone = self.layers[i].cache.update(&counts, fetched);
        self.store.recycle(gone);
        Ok(out)
    }
}

/// Greedy next token.
pub fn argmax(logits: &Array) -> Res<u32> {
    let t = argmax_axis(logits, -1, false)?;
    t.eval()?;
    Ok(t.as_slice::<u32>()[0])
}
