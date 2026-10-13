//! Routed experts on the SSD, the arenas that hold some of them in RAM, and
//! the per-layer cache policy.
//!
//! Two copies of every expert may exist on disk: `experts.bin` (q4, written
//! by engine/pack.py) and `experts_lo.bin` (2.5-bit codebook, engine/packlo.py).
//! Each has an arena: one array per expert tensor, shaped `[slots, ...]`,
//! shared by every layer. An expert in RAM occupies the same slot index in
//! all of the arena's arrays, so a layer runs its experts with one kernel
//! per projection indexed by slot, with no stacking or copying. Reads go
//! straight from the file into the slot.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};

use mlx_rs::{Array, Dtype, ops};

pub type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// One tensor inside an expert slot of the file.
struct Part {
    name: String,
    offset: usize,
    nbytes: usize,
    shape: Vec<i32>,
    dtype: Dtype,
}

/// Size of one read request: experts are split so a few misses still keep
/// many requests in flight (one 10 MiB pread per thread does not).
const CHUNK: usize = 2 << 20;

/// Completion of one expert's chunks.
struct Ticket {
    left: Mutex<usize>,
    done: Condvar,
    err: Mutex<Option<String>>,
}

struct Job {
    file: Arc<File>,
    off: u64,
    addr: usize, // destination inside an arena array; usize so it can cross threads
    len: usize,
    ticket: Arc<Ticket>,
}

/// Demand reads (the current layer waits for them) always go before
/// prefetches (guesses for the next layer).
#[derive(Default)]
struct Queues {
    demand: VecDeque<Job>,
    prefetch: VecDeque<Job>,
    stop: bool,
}

struct Pool {
    q: Mutex<Queues>,
    ready: Condvar,
}

/// An expert being read into a slot.
pub struct Pending {
    pub lo: bool,
    pub slot: usize,
    ticket: Arc<Ticket>,
}

/// One copy format's file and the RAM arena for it.
pub struct Arena {
    file: Arc<File>,
    parts: Vec<Part>,
    /// One array per part, `[slots, ...part.shape]`.
    pub arrays: Vec<Array>,
    ptrs: Vec<usize>,
    pub slots: usize,
    free: Vec<usize>,
    pub slot_bytes: usize,
}

impl Arena {
    fn open(bin: &Path, meta: &serde_json::Value, slots: usize) -> Res<Self> {
        let parts = meta["tensors"]
            .as_array()
            .ok_or("no tensors in expert metadata")?
            .iter()
            .map(|t| -> Res<Part> {
                Ok(Part {
                    name: t["name"].as_str().ok_or("name")?.to_string(),
                    offset: t["offset"].as_u64().ok_or("offset")? as usize,
                    nbytes: t["nbytes"].as_u64().ok_or("nbytes")? as usize,
                    shape: t["shape"].as_array().ok_or("shape")?.iter().map(|d| d.as_i64().unwrap() as i32).collect(),
                    dtype: match t["dtype"].as_str().ok_or("dtype")? {
                        "uint8" => Dtype::Uint8,
                        "uint32" => Dtype::Uint32,
                        "bfloat16" => Dtype::Bfloat16,
                        "float16" => Dtype::Float16,
                        "float32" => Dtype::Float32,
                        d => return Err(format!("unsupported dtype {d}").into()),
                    },
                })
            })
            .collect::<Res<Vec<_>>>()?;
        let mut arrays = Vec::with_capacity(parts.len());
        let mut ptrs = Vec::with_capacity(parts.len());
        for p in &parts {
            let mut shape = vec![slots as i32];
            shape.extend_from_slice(&p.shape);
            let a = ops::zeros_dtype(&shape, p.dtype)?;
            a.eval()?;
            // data<T>() in mlx-c only casts the pointer, whatever the dtype.
            ptrs.push(unsafe { mlx_sys::mlx_array_data_uint8(a.as_ptr()) } as usize);
            arrays.push(a);
        }
        let file = File::open(bin)?;
        // Bypass the page cache: misses really come from the SSD, and the
        // OS does not grow a second copy of the experts in RAM.
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        Ok(Self {
            file: Arc::new(file),
            parts,
            arrays,
            ptrs,
            slots,
            free: (0..slots).rev().collect(),
            slot_bytes: meta["slot_bytes"].as_u64().ok_or("slot_bytes")? as usize,
        })
    }

    /// Index of the array holding `name` (e.g. "up_proj.scales").
    pub fn part(&self, name: &str) -> usize {
        self.parts.iter().position(|p| p.name == name).unwrap_or_else(|| panic!("no expert tensor {name}"))
    }

    pub fn array(&self, name: &str) -> &Array {
        &self.arrays[self.part(name)]
    }
}

pub struct Store {
    pub hi: Arena,
    pub lo: Option<Arena>,
    pool: Arc<Pool>,
    workers: Vec<std::thread::JoinHandle<()>>,
    pub experts: usize,
    pub group_size: i32,
    pub bits: i32,
    pub bytes_read: u64,
    pub prefetched: u64,
    pub prefetch_used: u64,
    pub lo_reads: u64,
    /// Time the caller spent blocked on reads.
    pub t_io: std::time::Duration,
}

impl Store {
    /// Opens the pack with arenas of `hi_slots` q4 experts and, when the pack
    /// has codebook copies and `lo_slots` > 0, `lo_slots` of those.
    pub fn open(pack: &Path, hi_slots: usize, lo_slots: usize, threads: usize) -> Res<Self> {
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(pack.join("experts.json"))?)?;
        let hi = Arena::open(&pack.join("experts.bin"), &meta, hi_slots)?;
        let lo = if lo_slots > 0 && pack.join("experts_lo.bin").exists() {
            let m: serde_json::Value = serde_json::from_slice(&std::fs::read(pack.join("experts_lo.json"))?)?;
            Some(Arena::open(&pack.join("experts_lo.bin"), &m, lo_slots)?)
        } else {
            None
        };
        let pool = Arc::new(Pool { q: Mutex::new(Queues::default()), ready: Condvar::new() });
        let workers = (0..threads.max(1))
            .map(|_| {
                let pool = pool.clone();
                std::thread::spawn(move || worker(&pool))
            })
            .collect();
        Ok(Self {
            hi,
            lo,
            pool,
            workers,
            experts: meta["experts"].as_u64().ok_or("experts")? as usize,
            group_size: meta["group_size"].as_i64().ok_or("group_size")? as i32,
            bits: meta["bits"].as_i64().ok_or("bits")? as i32,
            bytes_read: 0,
            prefetched: 0,
            prefetch_used: 0,
            lo_reads: 0,
            t_io: Default::default(),
        })
    }

    fn arena_mut(&mut self, lo: bool) -> &mut Arena {
        if lo { self.lo.as_mut().expect("no low-precision arena") } else { &mut self.hi }
    }

    pub fn arena(&self, lo: bool) -> &Arena {
        if lo { self.lo.as_ref().expect("no low-precision arena") } else { &self.hi }
    }

    /// Starts reading an expert's copy into a free slot of its arena. None
    /// when the arena is full (the caller must release slots first).
    pub fn start(&mut self, layer: usize, expert: usize, lo: bool, demand: bool) -> Res<Option<Pending>> {
        let n = self.experts;
        let a = self.arena_mut(lo);
        let Some(slot) = a.free.pop() else { return Ok(None) };
        let base = ((layer * n + expert) * a.slot_bytes) as u64;
        let mut jobs = Vec::new();
        for (p, &ptr) in a.parts.iter().zip(&a.ptrs) {
            let dst = ptr + slot * p.nbytes;
            for j in (0..p.nbytes).step_by(CHUNK) {
                jobs.push((base + (p.offset + j) as u64, dst + j, CHUNK.min(p.nbytes - j)));
            }
        }
        let ticket = Arc::new(Ticket { left: Mutex::new(jobs.len()), done: Condvar::new(), err: Mutex::new(None) });
        let (file, slot_bytes) = (a.file.clone(), a.slot_bytes);
        {
            let mut q = self.pool.q.lock().unwrap();
            let dst = if demand { &mut q.demand } else { &mut q.prefetch };
            for (off, addr, len) in jobs {
                dst.push_back(Job { file: file.clone(), off, addr, len, ticket: ticket.clone() });
            }
        }
        self.pool.ready.notify_all();
        self.bytes_read += slot_bytes as u64;
        if lo {
            self.lo_reads += 1;
        }
        if !demand {
            self.prefetched += 1;
        }
        Ok(Some(Pending { lo, slot, ticket }))
    }

    /// A prefetch is needed now: move its queued chunks ahead of the others.
    pub fn promote(&self, p: &Pending) {
        let mut q = self.pool.q.lock().unwrap();
        let (mine, rest): (VecDeque<Job>, VecDeque<Job>) =
            std::mem::take(&mut q.prefetch).into_iter().partition(|j| Arc::ptr_eq(&j.ticket, &p.ticket));
        q.prefetch = rest;
        for j in mine.into_iter().rev() {
            q.demand.push_front(j);
        }
    }

    fn wait_done(&mut self, p: &Pending) -> Res<()> {
        let t0 = std::time::Instant::now();
        let mut left = p.ticket.left.lock().unwrap();
        while *left > 0 {
            left = p.ticket.done.wait(left).unwrap();
        }
        drop(left);
        self.t_io += t0.elapsed();
        if let Some(e) = p.ticket.err.lock().unwrap().take() {
            return Err(e.into());
        }
        Ok(())
    }

    /// Waits for an expert; its slot is then ready to use.
    pub fn finish(&mut self, p: Pending) -> Res<usize> {
        self.wait_done(&p)?;
        Ok(p.slot)
    }

    /// Drops a prefetch that was not needed: its queued chunks are cancelled
    /// and the slot is freed once the chunks already in flight land.
    pub fn discard(&mut self, p: Pending) -> Res<()> {
        let (cancelled, bytes) = {
            let mut q = self.pool.q.lock().unwrap();
            let (mine, rest): (VecDeque<Job>, VecDeque<Job>) =
                std::mem::take(&mut q.prefetch).into_iter().partition(|j| Arc::ptr_eq(&j.ticket, &p.ticket));
            q.prefetch = rest;
            (mine.len(), mine.iter().map(|j| j.len as u64).sum::<u64>())
        };
        *p.ticket.left.lock().unwrap() -= cancelled;
        self.bytes_read -= bytes;
        self.wait_done(&p)?;
        self.arena_mut(p.lo).free.push(p.slot);
        Ok(())
    }

    /// Frees slots whose experts left a cache. Only safe once the work that
    /// read them has run: callers release at the end of a layer, and the
    /// next `start` happens after the next layer's routing sync, which forces
    /// that work.
    pub fn release(&mut self, slots: impl IntoIterator<Item = (bool, usize)>) {
        for (lo, s) in slots {
            self.arena_mut(lo).free.push(s);
        }
    }
}

fn worker(pool: &Pool) {
    loop {
        let job = {
            let mut q = pool.q.lock().unwrap();
            loop {
                if q.stop {
                    return;
                }
                if let Some(j) = q.demand.pop_front().or_else(|| q.prefetch.pop_front()) {
                    break j;
                }
                q = pool.ready.wait(q).unwrap();
            }
        };
        // Each chunk is written by exactly one worker, and nothing reads the
        // slot until its ticket is done.
        let dst = unsafe { std::slice::from_raw_parts_mut(job.addr as *mut u8, job.len) };
        if let Err(e) = job.file.read_exact_at(dst, job.off) {
            *job.ticket.err.lock().unwrap() = Some(format!("read at {}: {e}", job.off));
        }
        let mut left = job.ticket.left.lock().unwrap();
        *left -= 1;
        if *left == 0 {
            job.ticket.done.notify_all();
        }
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        self.pool.q.lock().unwrap().stop = true;
        self.pool.ready.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Per-layer LFU with decay and bypass (the design tools/routing-trace
/// settled on), kept separately for the two copy formats: `cap_hi` q4
/// experts and `cap_lo` codebook ones stay resident. Scores decay with a
/// half-life of 32 calls; a q4 entry is worth every recent use of the
/// expert, a codebook entry only its recent low-weight picks (it cannot
/// serve a high one). A newly read expert only enters by beating the
/// weakest resident of its format, otherwise it is used once and dropped.
pub struct LayerCache {
    pub cap_hi: usize,
    pub cap_lo: usize,
    decay: f64,
    hi_score: Vec<f64>,
    lo_score: Vec<f64>,
    at: Vec<u64>,
    clock: u64,
    /// Resident expert -> slot in the q4 arena.
    pub hi: HashMap<usize, usize>,
    /// Resident expert -> slot in the codebook arena.
    pub lo: HashMap<usize, usize>,
    pub hits: u64,
    pub lookups: u64,
}

impl LayerCache {
    pub fn new(n_experts: usize, cap_hi: usize, cap_lo: usize) -> Self {
        Self {
            cap_hi,
            cap_lo,
            decay: 0.5f64.powf(1.0 / 32.0),
            hi_score: vec![0.0; n_experts],
            lo_score: vec![0.0; n_experts],
            at: vec![0; n_experts],
            clock: 0,
            hi: HashMap::new(),
            lo: HashMap::new(),
            hits: 0,
            lookups: 0,
        }
    }

    pub fn resident(&self, e: usize) -> bool {
        self.hi.contains_key(&e) || self.lo.contains_key(&e)
    }

    fn scores(&self, e: usize) -> (f64, f64) {
        let f = self.decay.powi((self.clock - self.at[e]).min(100_000) as i32);
        (self.hi_score[e] * f, self.lo_score[e] * f)
    }

    /// uses: (expert, high-weight pick?, count) in this call; fetched:
    /// (expert, lo copy?, slot) read for it. Returns the slots that leave
    /// (evicted, replaced, or read and not kept) as (lo, slot).
    pub fn update(&mut self, uses: &[(usize, bool, usize)], fetched: Vec<(usize, bool, usize)>) -> Vec<(bool, usize)> {
        self.clock += 1;
        for &(e, high, c) in uses {
            let (h, l) = self.scores(e);
            self.hi_score[e] = h + if high { c as f64 } else { 0.0 };
            self.lo_score[e] = l + if high { 0.0 } else { c as f64 };
            self.at[e] = self.clock;
        }
        let mut gone = Vec::new();
        for lo in [false, true] {
            let new: Vec<(usize, usize)> = fetched.iter().filter(|f| f.1 == lo).map(|f| (f.0, f.2)).collect();
            if new.is_empty() {
                continue;
            }
            let cap = if lo { self.cap_lo } else { self.cap_hi };
            let score = |s: &Self, e: usize| {
                let (h, l) = s.scores(e);
                if lo { l } else { h + l }
            };
            let table = if lo { &self.lo } else { &self.hi };
            let mut cand: Vec<usize> = table.keys().chain(new.iter().map(|(e, _)| e)).copied().collect();
            // Ties broken by expert id, so runs are reproducible (HashMap
            // order is not).
            cand.sort_by(|a, b| score(self, *b).total_cmp(&score(self, *a)).then(a.cmp(b)));
            cand.truncate(cap);
            let table = if lo { &mut self.lo } else { &mut self.hi };
            let mut evicted: Vec<usize> = table.keys().filter(|e| !cand.contains(e)).copied().collect();
            evicted.sort_unstable();
            gone.extend(evicted.iter().filter_map(|e| table.remove(e)).map(|s| (lo, s)));
            for (e, s) in new {
                if cand.contains(&e) {
                    table.insert(e, s);
                } else {
                    gone.push((lo, s));
                }
            }
        }
        // A q4 copy serves everything: drop codebook copies it makes redundant.
        let mut redundant: Vec<usize> = self.lo.keys().filter(|e| self.hi.contains_key(e)).copied().collect();
        redundant.sort_unstable();
        gone.extend(redundant.iter().filter_map(|e| self.lo.remove(e)).map(|s| (true, s)));
        gone
    }
}
