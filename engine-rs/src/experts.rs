//! Routed experts on the SSD (`experts.bin`, written by engine/pack.py), the
//! arena that holds some of them in RAM, and the per-layer cache policy.
//!
//! The arena is one array per expert tensor (gate/up/down weight, scales,
//! biases), shaped `[slots, ...]`, shared by every layer. An expert in RAM
//! occupies the same slot index in all nine arrays, so a layer runs its
//! experts with one `gather_qmm` per projection, indexed by slot, with no
//! stacking or copying. Reads go straight from the file into the slot.

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

/// An expert being read into `slot`.
pub struct Pending {
    pub slot: usize,
    ticket: Arc<Ticket>,
}

pub struct Store {
    parts: Vec<Part>,
    pool: Arc<Pool>,
    workers: Vec<std::thread::JoinHandle<()>>,
    /// Arena arrays, one per part, `[slots, ...part.shape]`.
    pub arena: Vec<Array>,
    /// Base address of each arena array.
    arena_ptr: Vec<usize>,
    pub slots: usize,
    free: Vec<usize>,
    pub slot: usize,
    pub experts: usize,
    pub group_size: i32,
    pub bits: i32,
    pub bytes_read: u64,
    pub prefetched: u64,
    pub prefetch_used: u64,
    /// Time the caller spent blocked on reads.
    pub t_io: std::time::Duration,
}

impl Store {
    /// Opens the pack with an arena of `slots` experts.
    pub fn open(pack: &Path, slots: usize, threads: usize) -> Res<Self> {
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(pack.join("experts.json"))?)?;
        let parts = meta["tensors"]
            .as_array()
            .ok_or("experts.json: no tensors")?
            .iter()
            .map(|t| -> Res<Part> {
                Ok(Part {
                    name: t["name"].as_str().ok_or("name")?.to_string(),
                    offset: t["offset"].as_u64().ok_or("offset")? as usize,
                    nbytes: t["nbytes"].as_u64().ok_or("nbytes")? as usize,
                    shape: t["shape"].as_array().ok_or("shape")?.iter().map(|d| d.as_i64().unwrap() as i32).collect(),
                    dtype: match t["dtype"].as_str().ok_or("dtype")? {
                        "uint32" => Dtype::Uint32,
                        "bfloat16" => Dtype::Bfloat16,
                        "float16" => Dtype::Float16,
                        "float32" => Dtype::Float32,
                        d => return Err(format!("unsupported dtype {d}").into()),
                    },
                })
            })
            .collect::<Res<Vec<_>>>()?;
        let mut arena = Vec::with_capacity(parts.len());
        let mut arena_ptr = Vec::with_capacity(parts.len());
        for p in &parts {
            let mut shape = vec![slots as i32];
            shape.extend_from_slice(&p.shape);
            let a = ops::zeros_dtype(&shape, p.dtype)?;
            a.eval()?;
            // data<T>() in mlx-c only casts the pointer, whatever the dtype.
            arena_ptr.push(unsafe { mlx_sys::mlx_array_data_uint8(a.as_ptr()) } as usize);
            arena.push(a);
        }
        let file = File::open(pack.join("experts.bin"))?;
        // Bypass the page cache: misses really come from the SSD, and the
        // OS does not grow a second copy of the experts in RAM.
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        let file = Arc::new(file);
        let pool = Arc::new(Pool { q: Mutex::new(Queues::default()), ready: Condvar::new() });
        let workers = (0..threads.max(1))
            .map(|_| {
                let (pool, file) = (pool.clone(), file.clone());
                std::thread::spawn(move || worker(&pool, &file))
            })
            .collect();
        Ok(Self {
            parts,
            pool,
            workers,
            arena,
            arena_ptr,
            slots,
            free: (0..slots).rev().collect(),
            slot: meta["slot_bytes"].as_u64().ok_or("slot_bytes")? as usize,
            experts: meta["experts"].as_u64().ok_or("experts")? as usize,
            group_size: meta["group_size"].as_i64().ok_or("group_size")? as i32,
            bits: meta["bits"].as_i64().ok_or("bits")? as i32,
            bytes_read: 0,
            prefetched: 0,
            prefetch_used: 0,
            t_io: Default::default(),
        })
    }

    /// Index of the arena array holding `name` (e.g. "up_proj.scales").
    pub fn part(&self, name: &str) -> usize {
        self.parts.iter().position(|p| p.name == name).unwrap_or_else(|| panic!("no expert tensor {name}"))
    }

    /// Starts reading an expert into a free slot. Returns None when the arena
    /// is full (the caller must release slots first).
    pub fn start(&mut self, layer: usize, expert: usize, demand: bool) -> Res<Option<Pending>> {
        let Some(slot) = self.free.pop() else { return Ok(None) };
        let base = ((layer * self.experts + expert) * self.slot) as u64;
        let mut jobs = Vec::new();
        for (p, &ptr) in self.parts.iter().zip(&self.arena_ptr) {
            let dst = ptr + slot * p.nbytes;
            for j in (0..p.nbytes).step_by(CHUNK) {
                jobs.push((base + (p.offset + j) as u64, dst + j, CHUNK.min(p.nbytes - j)));
            }
        }
        let ticket = Arc::new(Ticket { left: Mutex::new(jobs.len()), done: Condvar::new(), err: Mutex::new(None) });
        {
            let mut q = self.pool.q.lock().unwrap();
            let dst = if demand { &mut q.demand } else { &mut q.prefetch };
            for (off, addr, len) in jobs {
                dst.push_back(Job { off, addr, len, ticket: ticket.clone() });
            }
        }
        self.pool.ready.notify_all();
        self.bytes_read += self.slot as u64;
        if !demand {
            self.prefetched += 1;
        }
        Ok(Some(Pending { slot, ticket }))
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
        self.free.push(p.slot);
        Ok(())
    }

    /// Frees slots whose experts left a cache. Only safe once the work that
    /// read them has run: callers release at the end of a layer, and the
    /// next `start` happens after the next layer's routing sync, which forces
    /// that work.
    pub fn release(&mut self, slots: impl IntoIterator<Item = usize>) {
        self.free.extend(slots);
    }

    pub fn free_slots(&self) -> usize {
        self.free.len()
    }
}

fn worker(pool: &Pool, file: &File) {
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
        if let Err(e) = file.read_exact_at(dst, job.off) {
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
/// settled on): `cap` experts stay; scores decay with a half-life of 32
/// calls, and a newly read expert only enters by beating the weakest
/// resident one, otherwise it is used for this call and dropped.
pub struct LayerCache {
    pub cap: usize,
    decay: f64,
    score: Vec<f64>,
    /// Resident expert -> arena slot.
    pub experts: HashMap<usize, usize>,
    pub hits: u64,
    pub lookups: u64,
}

impl LayerCache {
    pub fn new(n_experts: usize, cap: usize) -> Self {
        Self {
            cap,
            decay: 0.5f64.powf(1.0 / 32.0),
            score: vec![0.0; n_experts],
            experts: HashMap::new(),
            hits: 0,
            lookups: 0,
        }
    }

    /// counts: uses of each expert in this call; fetched: expert -> slot of
    /// the experts read for it. Returns the slots that leave (evicted, or
    /// read and not kept).
    pub fn update(&mut self, counts: &[(usize, usize)], mut fetched: HashMap<usize, usize>) -> Vec<usize> {
        for s in self.score.iter_mut() {
            *s *= self.decay;
        }
        for &(e, c) in counts {
            self.score[e] += c as f64;
        }
        if self.cap == 0 {
            return fetched.into_values().collect();
        }
        let mut cand: Vec<usize> = self.experts.keys().chain(fetched.keys()).copied().collect();
        cand.sort_by(|a, b| self.score[*b].total_cmp(&self.score[*a]));
        cand.truncate(self.cap);
        let evicted: Vec<usize> = self.experts.keys().filter(|e| !cand.contains(e)).copied().collect();
        let mut gone: Vec<usize> = evicted.iter().filter_map(|e| self.experts.remove(e)).collect();
        for e in cand {
            if let Some(s) = fetched.remove(&e) {
                self.experts.insert(e, s);
            }
        }
        gone.extend(fetched.into_values());
        gone
    }
}
