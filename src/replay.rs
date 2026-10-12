//! Trace replay: real routing decisions drive real SSD reads through an
//! in-memory LRU expert cache, optionally with pre-gated prefetch.
//!
//! The trace comes from `tools/routing-trace/capture.py`: per decode token and
//! MoE layer, the top-k experts the router picked (`<trace>.bin`) and the
//! pre-gated prediction for that layer made one layer earlier
//! (`<trace>.pred.bin`). Expert (layer, e) lives at a fixed offset of the test
//! file, sized like the target model's experts, so the bytes moved per miss
//! are the real ones even when the trace came from a smaller proxy model.
//!
//! I/O runs on a pread (F_NOCACHE) worker pool with two queues: demand reads
//! (the layer about to run is waiting on them) always go before prefetches,
//! and a demand on an expert still queued as a prefetch promotes that read.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::data::{self, AlignedBuf};

const NO_PRED: u8 = 255;

// ------------------------------------------------------------------ trace

pub struct Sequence {
    pub id: String,
    pub start: usize,
    pub tokens: usize,
}

pub struct Trace {
    pub layers: usize,
    pub experts: usize,
    pub top_k: usize,
    pub picks: Vec<u8>,
    pub preds: Option<Vec<u8>>,
    pub seqs: Vec<Sequence>,
}

/// Every integer that follows `"key":` in `json`, in order. The metadata is
/// written by our own capture script, so a scan is enough (no JSON crate).
fn json_ints(json: &str, key: &str) -> Vec<usize> {
    let pat = format!("\"{key}\":");
    json.match_indices(&pat)
        .filter_map(|(i, _)| {
            let rest = json[i + pat.len()..].trim_start();
            let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
            rest[..end].parse().ok()
        })
        .collect()
}

fn json_strs(json: &str, key: &str) -> Vec<String> {
    let pat = format!("\"{key}\":");
    json.match_indices(&pat)
        .filter_map(|(i, _)| {
            let rest = json[i + pat.len()..].trim_start().strip_prefix('"')?;
            Some(rest[..rest.find('"')?].to_string())
        })
        .collect()
}

impl Trace {
    pub fn load(base: &str) -> Result<Self, String> {
        let read = |ext: &str| std::fs::read(format!("{base}{ext}")).map_err(|e| format!("{base}{ext}: {e}"));
        let meta = String::from_utf8(read(".json")?).map_err(|e| e.to_string())?;
        let one = |key: &str| {
            json_ints(&meta, key)
                .first()
                .copied()
                .ok_or(format!("trace metadata has no {key}"))
        };
        let (layers, experts, top_k) = (one("num_layers")?, one("num_experts")?, one("top_k")?);
        let picks = read(".bin")?;
        let preds = read(".pred.bin").ok();
        let ids = json_strs(&meta, "id");
        let starts = json_ints(&meta, "start");
        let lens = json_ints(&meta, "decode_tokens");
        if ids.len() != starts.len() || starts.len() != lens.len() {
            return Err("trace metadata: sequence fields do not line up".into());
        }
        let row = layers * top_k;
        let seqs: Vec<Sequence> = ids
            .into_iter()
            .zip(starts)
            .zip(lens)
            .map(|((id, start), tokens)| Sequence { id, start, tokens })
            .collect();
        let total: usize = seqs.iter().map(|s| s.tokens).sum();
        if picks.len() != total * row || preds.as_ref().is_some_and(|p| p.len() != picks.len()) {
            return Err(format!("trace size mismatch: {} bytes for {total} tokens", picks.len()));
        }
        Ok(Self { layers, experts, top_k, picks, preds, seqs })
    }

    fn row<'a>(&self, buf: &'a [u8], token: usize, layer: usize) -> &'a [u8] {
        let i = (token * self.layers + layer) * self.top_k;
        &buf[i..i + self.top_k]
    }
}

// -------------------------------------------------------------- I/O pool

struct Job {
    slot: usize,
    epoch: u64,
    dst: *mut u8,
    len: usize,
    offset: u64,
}

// `dst` points into a cache slot that is neither read nor evicted while loading.
unsafe impl Send for Job {}

#[derive(Default)]
struct Queues {
    demand: VecDeque<Job>,
    prefetch: VecDeque<Job>,
    shutdown: bool,
}

struct Shared {
    queues: Mutex<Queues>,
    has_work: Condvar,
    /// Highest generation that has finished loading, per slot.
    done: Mutex<(Vec<u64>, Option<String>)>,
    landed: Condvar,
}

struct IoPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl IoPool {
    fn new(path: &Path, threads: usize, slots: usize) -> Result<Self, String> {
        let file = Arc::new(data::open_uncached(path, false).map_err(|e| e.to_string())?);
        let shared = Arc::new(Shared {
            queues: Mutex::new(Queues::default()),
            has_work: Condvar::new(),
            done: Mutex::new((vec![0; slots], None)),
            landed: Condvar::new(),
        });
        let workers = (0..threads)
            .map(|_| {
                let shared = Arc::clone(&shared);
                let file = Arc::clone(&file);
                std::thread::spawn(move || {
                    use std::os::unix::fs::FileExt;
                    loop {
                        let job = {
                            let mut q = shared.queues.lock().unwrap();
                            loop {
                                if let Some(j) = q.demand.pop_front().or_else(|| q.prefetch.pop_front()) {
                                    break j;
                                }
                                if q.shutdown {
                                    return;
                                }
                                q = shared.has_work.wait(q).unwrap();
                            }
                        };
                        let dst = unsafe { std::slice::from_raw_parts_mut(job.dst, job.len) };
                        let res = file.read_exact_at(dst, job.offset);
                        let mut done = shared.done.lock().unwrap();
                        if let Err(e) = res {
                            done.1.get_or_insert(format!("read at {:#x}: {e}", job.offset));
                        }
                        done.0[job.slot] = done.0[job.slot].max(job.epoch);
                        shared.landed.notify_all();
                    }
                })
            })
            .collect();
        Ok(Self { shared, workers })
    }

    fn submit(&self, job: Job, demand: bool) {
        let mut q = self.shared.queues.lock().unwrap();
        if demand {
            q.demand.push_back(job);
        } else {
            q.prefetch.push_back(job);
        }
        self.shared.has_work.notify_one();
    }

    /// Moves a still-queued prefetch for `slot` to the demand queue.
    fn promote(&self, slot: usize) -> bool {
        let mut q = self.shared.queues.lock().unwrap();
        match q.prefetch.iter().position(|j| j.slot == slot) {
            Some(i) => {
                let job = q.prefetch.remove(i).unwrap();
                q.demand.push_back(job);
                true
            }
            None => false,
        }
    }

    fn wait(&self, slot: usize, epoch: u64) -> Result<(), String> {
        let done = self
            .shared
            .landed
            .wait_while(self.shared.done.lock().unwrap(), |d| d.0[slot] < epoch && d.1.is_none())
            .unwrap();
        done.1.clone().map_or(Ok(()), Err)
    }
}

impl Drop for IoPool {
    fn drop(&mut self) {
        self.shared.queues.lock().unwrap().shutdown = true;
        self.shared.has_work.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

// ------------------------------------------------------------------ cache

#[derive(Clone, Default)]
struct Slot {
    expert: Option<usize>,
    last_used: u64,
    epoch: u64,
    pinned: bool,
    /// Loaded by a prefetch and not used yet (to count wasted prefetches).
    speculative: bool,
}

struct Cache {
    slots: Vec<Slot>,
    bufs: Vec<AlignedBuf>,
    /// Slot holding each global expert id (layer * experts + e).
    map: Vec<Option<usize>>,
    clock: u64,
}

impl Cache {
    fn new(slots: usize, expert_bytes: usize, total_experts: usize) -> Self {
        Self {
            slots: vec![Slot::default(); slots],
            bufs: (0..slots).map(|_| AlignedBuf::new(expert_bytes)).collect(),
            map: vec![None; total_experts],
            clock: 0,
        }
    }

    fn reset(&mut self) {
        for s in &mut self.slots {
            let epoch = s.epoch;
            *s = Slot { epoch, ..Slot::default() };
        }
        self.map.iter_mut().for_each(|m| *m = None);
    }

    /// Least recently used slot that is not pinned. Loading slots are fine to
    /// skip implicitly: they were just touched, so they are never the oldest
    /// unless the cache is tiny, and `pinned` covers the layer in flight.
    fn victim(&self, done: &[u64]) -> Option<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(i, s)| !s.pinned && done[*i] >= s.epoch)
            .min_by_key(|(_, s)| (s.expert.is_some(), s.last_used))
            .map(|(i, _)| i)
    }
}

// ------------------------------------------------------------------- run

pub struct ReplayConfig {
    pub cache_fraction: f64,
    pub prefetch: bool,
    pub threads: usize,
    pub compute: Duration,
    pub seqs: Option<Vec<usize>>,
    pub max_tokens: Option<usize>,
    pub warm: bool,
    pub verify: bool,
    pub target_layers: usize,
}

#[derive(Default)]
struct Stats {
    accesses: u64,
    hits: u64,
    hits_in_flight: u64,
    demand_reads: u64,
    prefetch_reads: u64,
    promoted: u64,
    wasted_prefetch: u64,
    stall: Duration,
    stalls: Vec<Duration>,
    tokens: u64,
}

pub fn replay(path: &Path, trace_base: &str, expert_bytes: usize, cfg: &ReplayConfig) -> Result<(), String> {
    let tr = Trace::load(trace_base)?;
    let (layers, experts, k) = (tr.layers, tr.experts, tr.top_k);
    let total = layers * experts;
    let file_len = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    if (total * expert_bytes) as u64 > file_len {
        return Err(format!(
            "file too small: {total} experts × {expert_bytes} B needs {:.1} GiB",
            (total * expert_bytes) as f64 / data::GIB
        ));
    }
    let preds = match (&tr.preds, cfg.prefetch) {
        (Some(p), true) => Some(p),
        (None, true) => return Err("--prefetch needs <trace>.pred.bin".into()),
        _ => None,
    };
    let cap = ((cfg.cache_fraction * total as f64).round() as usize).max(3 * k);
    let seq_ids: Vec<usize> = cfg.seqs.clone().unwrap_or_else(|| (0..tr.seqs.len()).collect());

    eprintln!(
        "trace {layers} layers × {experts} experts, top-{k} | cache {cap} experts ({:.0}%, {:.1} GiB) | expert {:.2} MiB | prefetch {} | threads {}",
        cap as f64 / total as f64 * 100.0,
        (cap * expert_bytes) as f64 / data::GIB,
        expert_bytes as f64 / (1 << 20) as f64,
        if cfg.prefetch { "pre-gate" } else { "off" },
        cfg.threads,
    );
    let resident = data::resident_pct(path).unwrap_or(-1.0);
    eprintln!("page cache  {resident:.2}% of file resident (start)");

    let pool = IoPool::new(path, cfg.threads, cap)?;
    let mut cache = Cache::new(cap, expert_bytes, total);
    let mut st = Stats::default();
    let mut current: Vec<usize> = Vec::with_capacity(k);

    // Assigns a slot to `expert` and queues its read. Returns the slot and
    // whether it evicted a prefetched expert that was never used.
    let load = |cache: &mut Cache, pool: &IoPool, expert: usize, demand: bool| -> Result<(usize, bool), String> {
        let slot = {
            let done = pool.shared.done.lock().unwrap();
            cache.victim(&done.0).ok_or("every cache slot is pinned or loading")?
        };
        let s = &mut cache.slots[slot];
        let wasted = s.expert.is_some() && s.speculative;
        if let Some(old) = s.expert.take() {
            cache.map[old] = None;
        }
        s.epoch += 1;
        s.expert = Some(expert);
        s.speculative = !demand;
        cache.map[expert] = Some(slot);
        pool.submit(
            Job {
                slot,
                epoch: s.epoch,
                dst: cache.bufs[slot].as_ptr(),
                len: expert_bytes,
                offset: (expert * expert_bytes) as u64,
            },
            demand,
        );
        Ok((slot, wasted))
    };

    let start = Instant::now();
    for (n, &si) in seq_ids.iter().enumerate() {
        let seq = tr.seqs.get(si).ok_or(format!("no sequence {si}"))?;
        if !cfg.warm || n == 0 {
            cache.reset();
        }
        let tokens = cfg.max_tokens.map_or(seq.tokens, |m| m.min(seq.tokens));
        let seq_start = Instant::now();
        let hits_before = st.hits;
        let acc_before = st.accesses;
        for t in seq.start..seq.start + tokens {
            for l in 0..layers {
                // 1. Resolve this layer's picks: hit (maybe still in flight) or demand read.
                current.clear();
                for &e in tr.row(&tr.picks, t, l) {
                    let id = l * experts + e as usize;
                    st.accesses += 1;
                    let slot = match cache.map[id] {
                        Some(slot) => {
                            st.hits += 1;
                            let s = &mut cache.slots[slot];
                            if s.speculative {
                                s.speculative = false;
                                if pool.shared.done.lock().unwrap().0[slot] < s.epoch {
                                    st.hits_in_flight += 1;
                                    if pool.promote(slot) {
                                        st.promoted += 1;
                                    }
                                }
                            }
                            slot
                        }
                        None => {
                            st.demand_reads += 1;
                            let (slot, wasted) = load(&mut cache, &pool, id, true)?;
                            st.wasted_prefetch += wasted as u64;
                            slot
                        }
                    };
                    cache.clock += 1;
                    cache.slots[slot].last_used = cache.clock;
                    cache.slots[slot].pinned = true;
                    current.push(slot);
                }

                // 2. Prefetch the next layer's predicted experts while this one runs.
                if let Some(p) = preds
                    && l + 1 < layers
                {
                    for &e in tr.row(p, t, l + 1) {
                        if e == NO_PRED {
                            continue;
                        }
                        let id = (l + 1) * experts + e as usize;
                        if cache.map[id].is_none() {
                            let (slot, wasted) = load(&mut cache, &pool, id, false)?;
                            st.wasted_prefetch += wasted as u64;
                            cache.clock += 1;
                            cache.slots[slot].last_used = cache.clock;
                            st.prefetch_reads += 1;
                        }
                    }
                }

                // 3. Wait for this layer's experts, then "compute".
                let w = Instant::now();
                for &slot in &current {
                    pool.wait(slot, cache.slots[slot].epoch)?;
                }
                let stall = w.elapsed();
                st.stall += stall;
                st.stalls.push(stall);
                if cfg.verify {
                    for &slot in &current {
                        let expert = cache.slots[slot].expert.unwrap();
                        data::verify(cache.bufs[slot].as_slice(), (expert * expert_bytes) as u64)
                            .map_err(|e| format!("verify failed (expert {expert}): {e}"))?;
                    }
                }
                let c = Instant::now();
                while c.elapsed() < cfg.compute {
                    std::hint::spin_loop();
                }
                for &slot in &current {
                    cache.slots[slot].pinned = false;
                }
            }
            st.tokens += 1;
        }
        let secs = seq_start.elapsed().as_secs_f64();
        eprintln!(
            "  {:<22} {tokens:>5} tok  hit {:5.1}%  {:6.1} ms/tok",
            seq.id,
            (st.hits - hits_before) as f64 / (st.accesses - acc_before) as f64 * 100.0,
            secs * 1e3 / tokens as f64
        );
    }
    let elapsed = start.elapsed().as_secs_f64();
    let resident_end = data::resident_pct(path).unwrap_or(-1.0);
    eprintln!("page cache  {resident_end:.2}% of file resident (end)");

    st.stalls.sort();
    let pct = |p: f64| st.stalls[((st.stalls.len() - 1) as f64 * p).round() as usize].as_secs_f64() * 1e3;
    let reads = st.demand_reads + st.prefetch_reads;
    let gb_read = (reads * expert_bytes as u64) as f64 / 1e9;
    let ms_tok = elapsed * 1e3 / st.tokens as f64;
    // Every per-layer cost scales with the number of MoE layers.
    let target_tok_s = 1e3 / (ms_tok * cfg.target_layers as f64 / layers as f64);
    let hit = st.hits as f64 / st.accesses as f64;

    println!("tokens      {} ({:.1} s)", st.tokens, elapsed);
    println!(
        "cache       hit {:.1}% ({} in flight, {} promoted) | {} demand + {} prefetch reads ({} wasted)",
        hit * 100.0, st.hits_in_flight, st.promoted, st.demand_reads, st.prefetch_reads, st.wasted_prefetch
    );
    println!(
        "ssd         {:.2} GB read, {:.2} GB/token, {:.2} GB/s",
        gb_read,
        gb_read / st.tokens as f64,
        gb_read / elapsed
    );
    println!(
        "layer stall p50 {:.2} ms | p95 {:.2} ms | p99 {:.2} ms | {:.1} ms/token total",
        pct(0.5),
        pct(0.95),
        pct(0.99),
        st.stall.as_secs_f64() * 1e3 / st.tokens as f64
    );
    println!(
        "decode      {:.1} ms/token on {layers} layers → {target_tok_s:.2} tok/s projected on {} layers",
        ms_tok, cfg.target_layers
    );
    if cfg.verify {
        println!("verify      ok ({} experts)", st.accesses);
    }
    println!(
        "RESULT mode=replay cache_pct={:.1} prefetch={} threads={} compute_us={} tokens={} hit_pct={:.2} gb_per_token={:.3} gbps={:.3} stall_ms_per_token={:.2} ms_per_token={:.2} target_tok_s={target_tok_s:.3} wasted_prefetch={} resident_pct={resident:.2} resident_end_pct={resident_end:.2}",
        cfg.cache_fraction * 100.0,
        cfg.prefetch as u8,
        cfg.threads,
        cfg.compute.as_micros(),
        st.tokens,
        hit * 100.0,
        gb_read / st.tokens as f64,
        gb_read / elapsed,
        st.stall.as_secs_f64() * 1e3 / st.tokens as f64,
        ms_tok,
        st.wasted_prefetch,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_scan() {
        let j = r#"{"num_layers": 48, "sequences": [{"id": "a", "start": 0, "decode_tokens": 5}, {"id": "b", "start": 5, "decode_tokens": 7}]}"#;
        assert_eq!(json_ints(j, "num_layers"), vec![48]);
        assert_eq!(json_ints(j, "start"), vec![0, 5]);
        assert_eq!(json_ints(j, "decode_tokens"), vec![5, 7]);
        assert_eq!(json_strs(j, "id"), vec!["a", "b"]);
    }

    #[test]
    fn victim_is_lru_and_skips_pinned() {
        let mut c = Cache::new(3, data::ALIGN, 10);
        for (i, s) in c.slots.iter_mut().enumerate() {
            s.expert = Some(i);
            s.last_used = 10 - i as u64;
        }
        let done = vec![0; 3];
        assert_eq!(c.victim(&done), Some(2));
        c.slots[2].pinned = true;
        assert_eq!(c.victim(&done), Some(1));
        c.slots[1].epoch = 1; // still loading
        assert_eq!(c.victim(&done), Some(0));
    }
}
