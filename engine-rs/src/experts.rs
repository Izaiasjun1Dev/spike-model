//! Routed experts on the SSD (`experts.bin`, written by engine/pack.py) and
//! the per-layer cache that keeps some of them in RAM.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use mlx_rs::{Array, Dtype};

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
        })
    }

    /// Reads the given experts of a layer, in parallel, one pread each.
    pub fn read(&mut self, layer: usize, ids: &[usize]) -> Res<Vec<Expert>> {
        let mut bufs: Vec<Vec<u8>> = ids.iter().map(|_| vec![0u8; self.slot]).collect();
        let (file, slot, n) = (&self.file, self.slot, self.experts);
        let per = ids.len().div_ceil(self.threads.max(1)).max(1);
        std::thread::scope(|s| -> Res<()> {
            let jobs: Vec<_> = bufs
                .chunks_mut(per)
                .zip(ids.chunks(per))
                .map(|(bs, es)| {
                    s.spawn(move || -> std::io::Result<()> {
                        for (b, &e) in bs.iter_mut().zip(es) {
                            file.read_exact_at(b, ((layer * n + e) * slot) as u64)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            for j in jobs {
                j.join().map_err(|_| "reader thread panicked")??;
            }
            Ok(())
        })?;
        self.bytes_read += (ids.len() * self.slot) as u64;
        // MLX copies the bytes into its own buffers here.
        Ok(bufs
            .iter()
            .map(|b| {
                self.parts
                    .iter()
                    .map(|p| {
                        let a = unsafe { Array::from_raw_data(b[p.offset..p.offset + p.nbytes].as_ptr().cast(), &p.shape, p.dtype) };
                        (p.name.clone(), a)
                    })
                    .collect()
            })
            .collect())
    }
}

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
    pub fn update(&mut self, counts: &[(usize, usize)], mut fetched: HashMap<usize, Expert>) {
        for s in self.score.iter_mut() {
            *s *= self.decay;
        }
        for &(e, c) in counts {
            self.score[e] += c as f64;
        }
        if self.slots == 0 {
            return;
        }
        let mut cand: Vec<usize> = self.experts.keys().chain(fetched.keys()).copied().collect();
        cand.sort_by(|a, b| self.score[*b].total_cmp(&self.score[*a]));
        cand.truncate(self.slots);
        self.experts.retain(|e, _| cand.contains(e));
        for e in cand {
            if let Some(x) = fetched.remove(&e) {
                self.experts.insert(e, x);
            }
        }
    }
}
