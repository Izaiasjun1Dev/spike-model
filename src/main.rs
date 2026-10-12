//! expert-bench: how fast can MoE experts be pulled off the SSD?
//!
//! Simulates the decode loop "router picks top-k experts → load them → compute"
//! with no expert cache, so the numbers are the raw SSD-bound ceiling that the
//! cache and the "will it run" estimator have to work with.

mod backend;
mod data;
#[cfg(target_os = "macos")]
mod metal_io;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand, ValueEnum};

use backend::{Backend, Source, ThreadPool};
use data::GIB;

#[derive(Parser)]
#[command(about = "Benchmark random MoE expert reads from SSD")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write an incompressible test file (make it larger than RAM).
    Gen {
        #[arg(long)]
        path: PathBuf,
        #[arg(long)]
        size_gb: f64,
    },
    /// Run the simulated decode loop against a file written by `gen`.
    Run(RunArgs),
}

#[derive(Parser)]
struct RunArgs {
    #[arg(long)]
    path: PathBuf,
    #[arg(long, value_enum, default_value_t = Method::Pread)]
    method: Method,
    /// Take layer count, top-k and expert size from a known model.
    #[arg(long, value_enum)]
    preset: Option<Preset>,
    /// Bits per weight including quantization scales (4-bit + group scales ≈ 4.5).
    #[arg(long, default_value_t = 4.5)]
    bits: f64,
    /// Expert size in MiB (overrides the preset).
    #[arg(long)]
    expert_mb: Option<f64>,
    /// MoE layers per token (overrides the preset).
    #[arg(long)]
    layers: Option<usize>,
    /// Experts activated per layer (overrides the preset).
    #[arg(long)]
    topk: Option<usize>,
    #[arg(long, default_value_t = 10)]
    tokens: usize,
    /// Layers whose reads are issued ahead of the one being computed
    /// (0 = load on demand; >0 = ideal prefetch with a perfect router predictor).
    #[arg(long, default_value_t = 0)]
    lookahead: usize,
    /// Worker threads for pread/mmap (default: top-k).
    #[arg(long)]
    threads: Option<usize>,
    /// Busy-wait this long per layer after its experts land, to stand in for GPU work.
    #[arg(long, default_value_t = 0)]
    compute_us: u64,
    /// Check every loaded expert against the pattern written by `gen`.
    #[arg(long)]
    verify: bool,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Clone, Copy, ValueEnum)]
enum Method {
    Pread,
    Mmap,
    Metal,
}

#[derive(Clone, Copy, ValueEnum)]
enum Preset {
    Qwen3_30bA3b,
    Qwen3_235bA22b,
    DeepseekV3,
}

impl Preset {
    /// (hidden size, MoE intermediate size, MoE layers, active experts)
    fn dims(self) -> (u64, u64, usize, usize) {
        match self {
            Preset::Qwen3_30bA3b => (2048, 768, 48, 8),
            Preset::Qwen3_235bA22b => (4096, 1536, 94, 8),
            // 61 layers, the first 3 are dense.
            Preset::DeepseekV3 => (7168, 2048, 58, 8),
        }
    }
}

struct Shape {
    expert_bytes: u64,
    layers: usize,
    topk: usize,
}

fn shape(args: &RunArgs) -> Result<Shape, String> {
    let (preset_bytes, preset_layers, preset_topk) = match args.preset {
        Some(p) => {
            let (hidden, inter, layers, topk) = p.dims();
            // gate, up and down projections.
            let params = 3 * hidden * inter;
            (
                Some((params as f64 * args.bits / 8.0).ceil() as u64),
                Some(layers),
                Some(topk),
            )
        }
        None => (None, None, None),
    };
    let expert_bytes = args
        .expert_mb
        .map(|mb| (mb * 1024.0 * 1024.0) as u64)
        .or(preset_bytes)
        .ok_or("pass --preset or --expert-mb")?;
    Ok(Shape {
        expert_bytes: data::align_up(expert_bytes),
        layers: args
            .layers
            .or(preset_layers)
            .ok_or("pass --preset or --layers")?,
        topk: args.topk.or(preset_topk).ok_or("pass --preset or --topk")?,
    })
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `k` distinct values in 0..n.
    fn distinct(&mut self, k: usize, n: u64, out: &mut Vec<u64>) {
        out.clear();
        while out.len() < k {
            let v = self.next() % n;
            if !out.contains(&v) {
                out.push(v);
            }
        }
    }
}

fn percentile(sorted: &[Duration], p: f64) -> f64 {
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i].as_secs_f64() * 1e3
}

fn spin(d: Duration) {
    let start = Instant::now();
    while start.elapsed() < d {
        std::hint::spin_loop();
    }
}

/// Logs and returns how much of the test file sits in the page cache
/// (-1 if it could not be measured, so the RESULT line stays parseable).
fn residency(path: &std::path::Path, when: &str) -> f64 {
    match data::resident_pct(path) {
        Ok(pct) => {
            eprintln!("page cache  {pct:.2}% of file resident ({when})");
            pct
        }
        Err(e) => {
            eprintln!("warning: page cache residency check failed ({when}): {e}");
            -1.0
        }
    }
}

fn run(args: RunArgs) -> Result<(), String> {
    let shape = shape(&args)?;
    let file_len = std::fs::metadata(&args.path)
        .map_err(|e| format!("{}: {e}", args.path.display()))?
        .len();
    let pool = file_len / shape.expert_bytes;
    if pool < shape.topk as u64 {
        return Err(format!(
            "file holds only {pool} experts of this size; need at least {}",
            shape.topk
        ));
    }
    let slots = args.lookahead + 1;
    let threads = args.threads.unwrap_or(shape.topk).max(1);
    let eb = shape.expert_bytes as usize;

    let ram = data::physical_memory();
    if matches!(args.method, Method::Mmap) && file_len < 2 * ram {
        eprintln!(
            "warning: mmap goes through the page cache and the file ({:.1} GiB) is under 2× RAM ({:.1} GiB); \
             repeat reads will be served from memory. Use a bigger file or run `sudo purge` first.",
            file_len as f64 / GIB,
            ram as f64 / GIB
        );
    }

    let mut backend: Box<dyn Backend> = match args.method {
        Method::Pread => Box::new(ThreadPool::new(
            &args.path,
            Source::Pread,
            threads,
            slots,
            shape.topk,
            eb,
        )?),
        Method::Mmap => Box::new(ThreadPool::new(
            &args.path,
            Source::Mmap,
            threads,
            slots,
            shape.topk,
            eb,
        )?),
        #[cfg(target_os = "macos")]
        Method::Metal => Box::new(metal_io::MetalIo::new(&args.path, slots, shape.topk, eb)?),
        #[cfg(not(target_os = "macos"))]
        Method::Metal => return Err("the metal method needs macOS".into()),
    };

    let per_token = shape.layers as u64 * shape.topk as u64 * shape.expert_bytes;
    eprintln!(
        "expert {:.2} MiB × top-{} × {} layers = {:.2} GiB/token | pool {} experts ({:.1} GiB) | lookahead {} | threads {}",
        shape.expert_bytes as f64 / (1 << 20) as f64,
        shape.topk,
        shape.layers,
        per_token as f64 / GIB,
        pool,
        file_len as f64 / GIB,
        args.lookahead,
        threads,
    );

    let resident_start = residency(&args.path, "start");

    let mut rng = Rng(args.seed);
    let mut picks = Vec::with_capacity(shape.topk);
    let mut offsets: Vec<Vec<u64>> = vec![Vec::new(); slots];
    let total = args.tokens * shape.layers;
    let compute = Duration::from_micros(args.compute_us);
    let mut stalls = Vec::with_capacity(total);
    let mut token_times = Vec::with_capacity(args.tokens);

    let start = Instant::now();
    let mut token_start = start;
    for step in 0..total + args.lookahead {
        if step < total {
            let slot = step % slots;
            rng.distinct(shape.topk, pool, &mut picks);
            offsets[slot] = picks.iter().map(|e| e * shape.expert_bytes).collect();
            backend.submit(slot, &offsets[slot])?;
        }
        let Some(done) = step.checked_sub(args.lookahead) else {
            continue;
        };
        let slot = done % slots;
        let t = Instant::now();
        backend.wait(slot)?;
        stalls.push(t.elapsed());
        if args.verify {
            for (i, &off) in offsets[slot].iter().enumerate() {
                data::verify(backend.buffer(slot, i), off)
                    .map_err(|e| format!("verify failed: {e}"))?;
            }
        }
        spin(compute);
        if (done + 1) % shape.layers == 0 {
            token_times.push(token_start.elapsed());
            token_start = Instant::now();
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let resident_end = residency(&args.path, "end");

    stalls.sort();
    token_times.sort();
    let bytes = total as f64 * shape.topk as f64 * shape.expert_bytes as f64;
    let gbps = bytes / elapsed / 1e9;
    let tps = args.tokens as f64 / elapsed;
    println!(
        "method      {}",
        args.method.to_possible_value().unwrap().get_name()
    );
    println!("throughput  {gbps:.2} GB/s");
    println!(
        "decode      {tps:.2} tok/s ({:.0} ms/token, p50 {:.0} ms)",
        1e3 / tps,
        percentile(&token_times, 0.5)
    );
    println!(
        "layer stall p50 {:.2} ms | p95 {:.2} ms | p99 {:.2} ms | max {:.2} ms",
        percentile(&stalls, 0.5),
        percentile(&stalls, 0.95),
        percentile(&stalls, 0.99),
        percentile(&stalls, 1.0),
    );
    if args.verify {
        println!("verify      ok ({} experts)", total * shape.topk);
    }
    println!(
        "RESULT method={} expert_bytes={} topk={} layers={} lookahead={} threads={} compute_us={} gbps={gbps:.3} tok_s={tps:.3} resident_pct={resident_start:.2} resident_end_pct={resident_end:.2}",
        args.method.to_possible_value().unwrap().get_name(),
        shape.expert_bytes,
        shape.topk,
        shape.layers,
        args.lookahead,
        threads,
        args.compute_us,
    );
    Ok(())
}

fn main() -> ExitCode {
    let result = match Cli::parse().cmd {
        Cmd::Gen { path, size_gb } => {
            data::generate(&path, (size_gb * GIB) as u64).map_err(|e| e.to_string())
        }
        Cmd::Run(args) => run(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
