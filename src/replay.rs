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
//! The cache has a byte budget and either a global LRU or a per-layer LFU
//! with decay and bypass. With `tiered`, each expert also has a
//! low-precision copy (read as the first `low_bytes` of its slot in the test
//! file: same I/O pattern as a separate region) that serves the bottom half
//! of a token's picks by router weight; the top half always needs the full
//! copy. See tools/routing-trace/tiered.py for the simulated version.
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
    /// Router weight of each pick, f16 bits (`<trace>.scores.bin`).
    pub scores: Option<Vec<u16>>,
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
        let scores: Option<Vec<u16>> = read(".scores.bin")
            .ok()
            .map(|b| b.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect());
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
        if picks.len() != total * row
            || preds.as_ref().is_some_and(|p| p.len() != picks.len())
            || scores.as_ref().is_some_and(|s| s.len() != picks.len())
        {
            return Err(format!("trace size mismatch: {} bytes for {total} tokens", picks.len()));
        }
        Ok(Self { layers, experts, top_k, picks, preds, scores, seqs })
    }

    fn row<'a, T>(&self, buf: &'a [T], token: usize, layer: usize) -> &'a [T] {
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

/// LFU score half-life, in decode tokens (same as tools/routing-trace/tiered.py).
const LFU_HALF_LIFE: f64 = 32.0;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Global LRU over one byte budget.
    Lru,
    /// Per-layer budget; keep the entries with the most decayed recent use
    /// per byte, and bypass (read for this layer only) a newcomer worth less
    /// than everything it would evict.
    Lfu,
}

#[derive(Default)]
struct Slot {
    expert: Option<usize>,
    /// Holds the low-precision copy, which only serves low-weight picks.
    low: bool,
    /// Bypassed by the policy: read for the current layer, freed after it.
    transient: bool,
    last_used: u64,
    epoch: u64,
    pinned: bool,
    /// Loaded by a prefetch and not used yet (to count wasted prefetches).
    speculative: bool,
    buf: Option<AlignedBuf>,
}

/// Decayed use counts of one expert, split by router weight of the pick.
#[derive(Clone, Copy, Default)]
struct Score {
    hi: f64,
    lo: f64,
    at: u64,
}

/// Expert cache with a byte budget, so full and low-precision copies can
/// share it. Buffers are recycled through per-size free lists; `allocated`
/// reports what that costs on top of the budget.
struct Cache {
    slots: Vec<Slot>,
    /// Slot holding each global expert id (layer * experts + e).
    map: Vec<Option<usize>>,
    clock: u64,
    policy: Policy,
    experts: usize,
    /// Bytes per group: one group for LRU, one per layer for LFU.
    budget: usize,
    used: Vec<usize>,
    full_bytes: usize,
    low_bytes: usize,
    free_bufs: [Vec<AlignedBuf>; 2],
    allocated: usize,
    peak: usize,
    scores: Vec<Score>,
    decay: f64,
    wasted: u64,
}

impl Cache {
    fn new(policy: Policy, budget: usize, layers: usize, experts: usize, full_bytes: usize, low_bytes: usize) -> Self {
        let groups = if policy == Policy::Lfu { layers } else { 1 };
        let max_entries = budget / low_bytes.min(full_bytes) + 2 * layers + 64;
        Self {
            slots: (0..max_entries).map(|_| Slot::default()).collect(),
            map: vec![None; layers * experts],
            clock: 0,
            policy,
            experts,
            budget: budget / groups,
            used: vec![0; groups],
            full_bytes,
            low_bytes,
            free_bufs: [Vec::new(), Vec::new()],
            allocated: 0,
            peak: 0,
            scores: vec![Score::default(); layers * experts],
            decay: 0.5f64.powf(1.0 / LFU_HALF_LIFE),
            wasted: 0,
        }
    }

    fn group(&self, id: usize) -> usize {
        if self.used.len() == 1 { 0 } else { id / self.experts }
    }

    fn size(&self, low: bool) -> usize {
        if low { self.low_bytes } else { self.full_bytes }
    }

    fn reset(&mut self) {
        for i in 0..self.slots.len() {
            self.release(i);
        }
        self.scores.iter_mut().for_each(|s| *s = Score::default());
    }

    /// Unmaps a slot's expert and returns its buffer to the free list.
    fn release(&mut self, slot: usize) {
        let Some(id) = self.slots[slot].expert.take() else { return };
        let s = &mut self.slots[slot];
        let (low, transient, buf) = (s.low, s.transient, s.buf.take());
        *s = Slot { epoch: s.epoch, ..Slot::default() };
        if !transient {
            self.map[id] = None;
            let g = self.group(id);
            self.used[g] -= self.size(low);
        }
        if let Some(b) = buf {
            self.free_bufs[low as usize].push(b);
        }
    }

    /// Reuses a free buffer of the right size, else allocates one. Free
    /// buffers of the other size are dropped first when allocating would
    /// exceed the budget plus a few experts of room for transient reads.
    fn take_buf(&mut self, low: bool) -> AlignedBuf {
        let len = self.size(low);
        if let Some(b) = self.free_bufs[low as usize].pop() {
            return b;
        }
        let limit = self.budget * self.used.len() + 16 * self.full_bytes;
        while self.allocated + len > limit && self.free_bufs[!low as usize].pop().is_some() {
            self.allocated -= self.size(!low);
        }
        self.allocated += len;
        self.peak = self.peak.max(self.allocated);
        AlignedBuf::new(len)
    }

    fn decayed(&self, id: usize, now: u64) -> (f64, f64) {
        let s = self.scores[id];
        let f = self.decay.powi((now - s.at).min(100_000) as i32);
        (s.hi * f, s.lo * f)
    }

    fn bump(&mut self, id: usize, high: bool, now: u64) {
        let (mut hi, mut lo) = self.decayed(id, now);
        if high { hi += 1.0 } else { lo += 1.0 }
        self.scores[id] = Score { hi, lo, at: now };
    }

    /// Recent uses an entry could serve, per byte: a low copy can't serve
    /// high-weight picks.
    fn density(&self, id: usize, low: bool, now: u64) -> f64 {
        let (hi, lo) = self.decayed(id, now);
        (if low { lo } else { hi + lo }) / self.size(low) as f64
    }

    /// Evicts until `id` at this precision fits its group's budget. False
    /// means don't cache it: nothing evictable (LRU), or everything that
    /// would have to go is worth more per byte (LFU).
    fn make_room(&mut self, id: usize, low: bool, now: u64, done: &[u64]) -> bool {
        let g = self.group(id);
        let need = self.size(low);
        if need > self.budget {
            return false;
        }
        if self.used[g] + need <= self.budget {
            return true;
        }
        let newcomer = self.density(id, low, now);
        let mut cands: Vec<(f64, usize)> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(i, s)| {
                s.expert.is_some_and(|e| self.group(e) == g) && !s.pinned && !s.transient && done[*i] >= s.epoch
            })
            .map(|(i, s)| {
                let key = match self.policy {
                    Policy::Lru => s.last_used as f64,
                    Policy::Lfu => self.density(s.expert.unwrap(), s.low, now),
                };
                (key, i)
            })
            .collect();
        cands.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut free = self.budget - self.used[g];
        let mut take = 0;
        for &(key, i) in &cands {
            if free >= need {
                break;
            }
            if self.policy == Policy::Lfu && key >= newcomer {
                return false;
            }
            free += self.size(self.slots[i].low);
            take += 1;
        }
        if free < need {
            return false;
        }
        for &(_, i) in &cands[..take] {
            self.wasted += self.slots[i].speculative as u64;
            self.release(i);
        }
        true
    }

    fn free_slot(&self, done: &[u64]) -> Option<usize> {
        self.slots.iter().enumerate().position(|(i, s)| s.expert.is_none() && done[i] >= s.epoch)
    }
}

// ------------------------------------------------------------------- run

pub struct ReplayConfig {
    pub cache_fraction: f64,
    /// Cache budget in GB on the target model (overrides `cache_fraction`).
    pub cache_gb: Option<f64>,
    pub policy: Policy,
    /// Two resolutions: the bottom half of each token's picks by router
    /// weight may use (and on a miss, read) the low-precision copy.
    pub tiered: bool,
    pub low_bytes: usize,
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
    low_reads: u64,
    bypassed: u64,
    bytes_read: u64,
    promoted: u64,
    stall: Duration,
    stalls: Vec<Duration>,
    tokens: u64,
}

/// Which of a token's picks are in the top half by router weight. Weights
/// are positive f16, whose bit patterns sort like the values.
fn high_half(weights: &[u16]) -> Vec<bool> {
    (0..weights.len())
        .map(|j| {
            let above = (0..weights.len()).filter(|&i| weights[i] > weights[j] || (weights[i] == weights[j] && i < j)).count();
            above < weights.len() / 2
        })
        .collect()
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
        (Some(_), true) if cfg.tiered || cfg.policy != Policy::Lru => {
            return Err("--prefetch is only implemented for the single-format LRU cache".into());
        }
        (Some(p), true) => Some(p),
        (None, true) => return Err("--prefetch needs <trace>.pred.bin".into()),
        _ => None,
    };
    if cfg.tiered && tr.scores.is_none() {
        return Err("--tiered needs <trace>.scores.bin".into());
    }
    if cfg.low_bytes > expert_bytes {
        return Err("the low-precision copy is larger than the expert".into());
    }
    let fraction = cfg
        .cache_gb
        .map_or(cfg.cache_fraction, |gb| gb * 1e9 / (cfg.target_layers * experts * expert_bytes) as f64);
    let budget = ((fraction * (total * expert_bytes) as f64) as usize).max(3 * k * expert_bytes);
    let seq_ids: Vec<usize> = cfg.seqs.clone().unwrap_or_else(|| (0..tr.seqs.len()).collect());

    let mut cache = Cache::new(cfg.policy, budget, layers, experts, expert_bytes, cfg.low_bytes);
    eprintln!(
        "trace {layers} layers × {experts} experts, top-{k} | cache {:.1} GiB ({:.1}% of experts, ≈{:.1} GB on {} layers) | {} | expert {:.2} MiB{} | prefetch {} | threads {}",
        budget as f64 / data::GIB,
        fraction * 100.0,
        fraction * (cfg.target_layers * experts * expert_bytes) as f64 / 1e9,
        cfg.target_layers,
        if cfg.policy == Policy::Lru { "lru" } else { "lfu-decay, per layer" },
        expert_bytes as f64 / (1 << 20) as f64,
        if cfg.tiered { format!(", low copy {:.2} MiB for low-weight picks", cfg.low_bytes as f64 / (1 << 20) as f64) } else { String::new() },
        if cfg.prefetch { "pre-gate" } else { "off" },
        cfg.threads,
    );
    let resident = data::resident_pct(path).unwrap_or(-1.0);
    eprintln!("page cache  {resident:.2}% of file resident (start)");

    let pool = IoPool::new(path, cfg.threads, cache.slots.len())?;
    let mut st = Stats::default();
    let mut current: Vec<usize> = Vec::with_capacity(k);

    // Gives `id` a slot (low or full copy) and queues its read. Demand loads
    // the policy won't cache go to a transient slot; such prefetches are dropped.
    let load = |cache: &mut Cache, pool: &IoPool, id: usize, low: bool, demand: bool, now: u64| -> Result<Option<usize>, String> {
        if let Some(old) = cache.map[id] {
            cache.release(old); // a low copy being replaced by the full one
        }
        let (keep, slot) = {
            let done = pool.shared.done.lock().unwrap();
            let keep = cache.make_room(id, low, now, &done.0);
            if !keep && !demand {
                return Ok(None);
            }
            (keep, cache.free_slot(&done.0).ok_or("no free cache slot")?)
        };
        let buf = cache.take_buf(low);
        let (dst, len) = (buf.as_ptr(), cache.size(low));
        let s = &mut cache.slots[slot];
        s.epoch += 1;
        s.expert = Some(id);
        s.low = low;
        s.transient = !keep;
        s.speculative = !demand;
        s.buf = Some(buf);
        let epoch = s.epoch;
        if keep {
            cache.map[id] = Some(slot);
            let g = cache.group(id);
            cache.used[g] += len;
        }
        pool.submit(Job { slot, epoch, dst, len, offset: (id * expert_bytes) as u64 }, demand);
        Ok(Some(slot))
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
            let now = st.tokens;
            for l in 0..layers {
                // 1. Resolve this layer's picks: hit (maybe still in flight) or demand read.
                let picks = tr.row(&tr.picks, t, l);
                let high = match &tr.scores {
                    Some(sc) => high_half(tr.row(sc, t, l)),
                    None => vec![true; k],
                };
                if cfg.policy == Policy::Lfu {
                    for (j, &e) in picks.iter().enumerate() {
                        cache.bump(l * experts + e as usize, high[j], now);
                    }
                }
                current.clear();
                for (j, &e) in picks.iter().enumerate() {
                    let id = l * experts + e as usize;
                    let low = cfg.tiered && !high[j];
                    st.accesses += 1;
                    let slot = match cache.map[id] {
                        // A low copy can't stand in for a high-weight pick.
                        Some(slot) if !(cache.slots[slot].low && !low) => {
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
                        _ => {
                            st.demand_reads += 1;
                            st.low_reads += low as u64;
                            st.bytes_read += cache.size(low) as u64;
                            let slot = load(&mut cache, &pool, id, low, true, now)?.expect("demand loads always get a slot");
                            st.bypassed += cache.slots[slot].transient as u64;
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
                        if cache.map[id].is_none()
                            && let Some(slot) = load(&mut cache, &pool, id, false, false, now)?
                        {
                            cache.clock += 1;
                            cache.slots[slot].last_used = cache.clock;
                            st.prefetch_reads += 1;
                            st.bytes_read += expert_bytes as u64;
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
                        let s = &cache.slots[slot];
                        let expert = s.expert.unwrap();
                        let buf = &s.buf.as_ref().unwrap().as_slice()[..cache.size(s.low)];
                        data::verify(buf, (expert * expert_bytes) as u64)
                            .map_err(|e| format!("verify failed (expert {expert}): {e}"))?;
                    }
                }
                let c = Instant::now();
                while c.elapsed() < cfg.compute {
                    std::hint::spin_loop();
                }
                for &slot in &current {
                    cache.slots[slot].pinned = false;
                    if cache.slots[slot].transient {
                        cache.release(slot);
                    }
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
    let gb_read = st.bytes_read as f64 / 1e9;
    let ms_tok = elapsed * 1e3 / st.tokens as f64;
    // Every per-layer cost scales with the number of MoE layers.
    let target_tok_s = 1e3 / (ms_tok * cfg.target_layers as f64 / layers as f64);
    let hit = st.hits as f64 / st.accesses as f64;
    let buf_gib = cache.peak as f64 / data::GIB;

    println!("tokens      {} ({:.1} s)", st.tokens, elapsed);
    println!(
        "cache       hit {:.1}% ({} in flight, {} promoted) | {} demand ({} low copies, {} bypassed) + {} prefetch reads ({} wasted)",
        hit * 100.0, st.hits_in_flight, st.promoted, st.demand_reads, st.low_reads, st.bypassed, st.prefetch_reads, cache.wasted
    );
    println!("memory      budget {:.2} GiB, buffers peak {buf_gib:.2} GiB", budget as f64 / data::GIB);
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
        "RESULT mode=replay cache_pct={:.1} prefetch={} threads={} compute_us={} tokens={} hit_pct={:.2} gb_per_token={:.3} gbps={:.3} stall_ms_per_token={:.2} ms_per_token={:.2} target_tok_s={target_tok_s:.3} wasted_prefetch={} resident_pct={resident:.2} resident_end_pct={resident_end:.2} policy={} tiered={} low_reads={} bypassed={} buf_gib={buf_gib:.2}",
        fraction * 100.0,
        cfg.prefetch as u8,
        cfg.threads,
        cfg.compute.as_micros(),
        st.tokens,
        hit * 100.0,
        gb_read / st.tokens as f64,
        gb_read / elapsed,
        st.stall.as_secs_f64() * 1e3 / st.tokens as f64,
        ms_tok,
        cache.wasted,
        if cfg.policy == Policy::Lru { "lru" } else { "lfu" },
        cfg.tiered as u8,
        st.low_reads,
        st.bypassed,
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

    /// Fills a cache by hand: `entries` are (expert id, low copy?).
    fn filled(policy: Policy, budget_entries: usize, entries: &[(usize, bool)]) -> Cache {
        let full = data::ALIGN * 2;
        let mut c = Cache::new(policy, budget_entries * full, 1, 16, full, data::ALIGN);
        for (slot, &(id, low)) in entries.iter().enumerate() {
            let s = &mut c.slots[slot];
            s.expert = Some(id);
            s.low = low;
            s.last_used = slot as u64;
            c.map[id] = Some(slot);
            c.used[0] += c.size(low);
        }
        c
    }

    #[test]
    fn lru_evicts_oldest_and_skips_pinned() {
        let mut c = filled(Policy::Lru, 3, &[(0, false), (1, false), (2, false)]);
        let done = vec![0; c.slots.len()];
        c.slots[0].pinned = true;
        assert!(c.make_room(5, false, 0, &done));
        assert_eq!((c.map[0], c.map[1], c.map[2]), (Some(0), None, Some(2)));
        assert_eq!(c.used[0], 2 * c.full_bytes);
    }

    #[test]
    fn lfu_bypasses_cold_newcomer_and_low_copies_are_cheaper() {
        let mut c = filled(Policy::Lfu, 2, &[(0, false), (1, false)]);
        let done = vec![0; c.slots.len()];
        for _ in 0..3 {
            c.bump(0, true, 0);
            c.bump(1, true, 0);
        }
        c.bump(7, true, 0);
        assert!(!c.make_room(7, false, 0, &done), "one use should not evict three");
        // As a low copy (half the bytes) with low-weight uses, 7 is denser
        // per byte than a full copy used as often.
        for _ in 0..4 {
            c.bump(7, false, 0);
        }
        assert!(c.make_room(7, true, 0, &done));
        assert_eq!(c.used[0], c.full_bytes);
    }

    #[test]
    fn high_half_by_weight() {
        assert_eq!(high_half(&[10, 40, 20, 30]), vec![false, true, false, true]);
        assert_eq!(high_half(&[5, 5, 5, 5]), vec![true, true, false, false]);
    }
}
