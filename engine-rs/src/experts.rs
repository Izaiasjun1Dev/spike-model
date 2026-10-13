//! Routed experts on the SSD (`experts.bin`, written by engine/pack.py) and
//! the per-layer cache that keeps some of them in RAM.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype, ops};

pub type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// One tensor inside an expert slot.
struct Part {
    name: String,
    offset: usize,
    nbytes: usize,
    shape: Vec<i32>,
    dtype: Dtype,
}

/// An expert's weights: gate/up/down weight, scales and biases, by name.
pub type Expert = HashMap<String, Array>;

pub struct Store {
    file: File,
    parts: Vec<Part>,
    pub slot: usize,
    pub experts: usize,
    pub group_size: i32,
    pub bits: i32,
    threads: usize,
    pub bytes_read: u64,
    /// Time spent in preads, and copying the bytes into MLX arrays.
    pub t_io: std::time::Duration,
    pub t_copy: std::time::Duration,
    /// Slot buffers of experts that left, reused by the next reads.
    free: Vec<Array>,
}

impl Store {
    pub fn open(pack: &Path, threads: usize) -> Res<Self> {
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
        let file = File::open(pack.join("experts.bin"))?;
        // Bypass the page cache: misses really come from the SSD, and the
        // OS does not grow a second copy of the experts in RAM.
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        Ok(Self {
            file,
            parts,
            slot: meta["slot_bytes"].as_u64().ok_or("slot_bytes")? as usize,
            experts: meta["experts"].as_u64().ok_or("experts")? as usize,
            group_size: meta["group_size"].as_i64().ok_or("group_size")? as i32,
            bits: meta["bits"].as_i64().ok_or("bits")? as i32,
            threads,
            bytes_read: 0,
            t_io: Default::default(),
            t_copy: Default::default(),
            free: Vec::new(),
        })
    }

    /// Reads the given experts of a layer. Each expert is split into CHUNK
    /// pieces spread over the threads, so a handful of misses still keeps
    /// many requests in flight (one 10 MiB pread per thread does not).
    pub fn read(&mut self, layer: usize, ids: &[usize]) -> Res<Vec<Expert>> {
        const CHUNK: usize = 2 << 20;
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        // Read straight into MLX-owned memory (unified with the GPU's), then
        // carve the expert's tensors out of it as views: no second copy.
        let t1 = std::time::Instant::now();
        let mut bases = Vec::with_capacity(ids.len());
        for _ in ids {
            bases.push(match self.free.pop() {
                Some(a) => a,
                None => {
                    let a = Array::zeros::<u8>(&[self.slot as i32])?;
                    a.eval()?;
                    a
                }
            });
        }
        self.t_copy += t1.elapsed();
        let (file, slot, n) = (&self.file, self.slot, self.experts);
        // (file offset, destination address, length) for every chunk of every
        // expert. Addresses as usize so the jobs can cross threads; each chunk
        // is written by exactly one thread and nothing reads it meanwhile.
        let mut jobs: Vec<(u64, usize, usize)> = Vec::new();
        for (a, &e) in bases.iter().zip(ids) {
            let base = ((layer * n + e) * slot) as u64;
            let ptr = a.as_slice::<u8>().as_ptr() as usize;
            for j in (0..slot).step_by(CHUNK) {
                jobs.push((base + j as u64, ptr + j, CHUNK.min(slot - j)));
            }
        }
        let threads = self.threads.min(jobs.len()).max(1);
        let per = jobs.len().div_ceil(threads);
        let t0 = std::time::Instant::now();
        std::thread::scope(|s| -> Res<()> {
            let handles: Vec<_> = jobs
                .chunks(per)
                .map(|js| {
                    s.spawn(move || -> std::io::Result<()> {
                        for &(off, addr, len) in js {
                            let dst = unsafe { std::slice::from_raw_parts_mut(addr as *mut u8, len) };
                            file.read_exact_at(dst, off)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            for h in handles {
                h.join().map_err(|_| "reader thread panicked")??;
            }
            Ok(())
        })?;
        self.bytes_read += (ids.len() * self.slot) as u64;
        self.t_io += t0.elapsed();
        let t1 = std::time::Instant::now();
        let out = bases
            .iter()
            .map(|a| -> Res<Expert> {
                let mut ex = self
                    .parts
                    .iter()
                    .map(|p| -> Res<(String, Array)> {
                        let t = a.index(p.offset as i32..(p.offset + p.nbytes) as i32).view_dtype(p.dtype)?;
                        Ok((p.name.clone(), ops::reshape(&t, &p.shape)?))
                    })
                    .collect::<Res<Expert>>()?;
                ex.insert(BASE.to_string(), a.clone());
                Ok(ex)
            })
            .collect::<Res<Vec<_>>>()?;
        self.t_copy += t1.elapsed();
        Ok(out)
    }
}

impl Store {
    /// Takes back the buffers of experts that left. Only safe once the work
    /// that read them has run: the caller recycles at the end of a layer, and
    /// the next read waits for the next layer's routing, which needs it.
    pub fn recycle(&mut self, gone: Vec<Expert>) {
        for mut ex in gone {
            if let Some(b) = ex.remove(BASE) {
                if self.free.len() < MAX_FREE {
                    self.free.push(b);
                }
            }
        }
    }
}

/// Key of an expert's whole slot buffer, which its tensors are views of.
const BASE: &str = "slot";
/// Spare buffers kept (a prefill can drop up to all experts of a layer).
const MAX_FREE: usize = 16;

/// Per-layer LFU with decay and bypass (the design tools/routing-trace
/// settled on): `slots` experts stay; scores decay with a half-life of 32
/// calls, and a newly read expert only enters by beating the weakest
/// resident one, otherwise it is used for this call and dropped.
pub struct LayerCache {
    pub slots: usize,
    decay: f64,
    score: Vec<f64>,
    pub experts: HashMap<usize, Expert>,
    pub hits: u64,
    pub lookups: u64,
}

impl LayerCache {
    pub fn new(n_experts: usize, slots: usize) -> Self {
        Self {
            slots,
            decay: 0.5f64.powf(1.0 / 32.0),
            score: vec![0.0; n_experts],
            experts: HashMap::new(),
            hits: 0,
            lookups: 0,
        }
    }

    /// counts: uses of each expert in this call; fetched: experts read for it.
    /// Returns the experts that leave (evicted, or read and not kept), so
    /// their buffers can be reused.
    pub fn update(&mut self, counts: &[(usize, usize)], mut fetched: HashMap<usize, Expert>) -> Vec<Expert> {
        for s in self.score.iter_mut() {
            *s *= self.decay;
        }
        for &(e, c) in counts {
            self.score[e] += c as f64;
        }
        if self.slots == 0 {
            return fetched.into_values().collect();
        }
        let mut cand: Vec<usize> = self.experts.keys().chain(fetched.keys()).copied().collect();
        cand.sort_by(|a, b| self.score[*b].total_cmp(&self.score[*a]));
        cand.truncate(self.slots);
        let evicted: Vec<usize> = self.experts.keys().filter(|e| !cand.contains(e)).copied().collect();
        let mut gone: Vec<Expert> = evicted.iter().filter_map(|e| self.experts.remove(e)).collect();
        for e in cand {
            if let Some(x) = fetched.remove(&e) {
                self.experts.insert(e, x);
            }
        }
        gone.extend(fetched.into_values());
        gone
    }
}
