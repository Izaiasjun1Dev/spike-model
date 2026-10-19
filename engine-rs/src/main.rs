//! moe-run: chat with a large MoE model whose experts stay on the SSD.
//!
//!   moe-run <pack> [-p PROMPT | --chat] [--cache-gb 8] [--bias 0.5]
//!           [--max-tokens 512] [--threads 16] [--lo-frac 0.4 | --no-tiered] [--no-skip-last] [--lo-all [--lo-prefill]] [--no-prefetch] [--prefill-batch 32] [--dump-logits FILE]
//!           [--score IDS_FILE --n-prompt N --dump-logits FILE]   (teacher forcing, for tests)
//!
//! <pack> is a directory written by engine/pack.py.

mod cbkernel;
mod experts;
mod model;

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use experts::Res;
use mlx_rs::Array;
use model::{KvCache, Model};
use tokenizers::Tokenizer;

struct Args {
    pack: PathBuf,
    prompt: Option<String>,
    chat: bool,
    cache_gb: f64,
    lo_frac: f64,
    bias: f32,
    max_tokens: usize,
    threads: usize,
    dump_logits: Option<PathBuf>,
    prefetch: bool,
    tiered: bool,
    skip_last: bool,
    lo_all: bool,
    lo_prefill: bool,
    /// Prefill misses read and run this many experts at a time (0: a whole layer).
    prefill_batch: usize,
    /// Teacher forcing: a file of u32 token ids, and how many form the prompt.
    score: Option<PathBuf>,
    n_prompt: usize,
}

fn parse_args() -> Res<Args> {
    let usage = "usage: moe-run <pack> [-p PROMPT | --chat] [--cache-gb 8] [--bias 0.5] [--max-tokens 512] [--threads 16] [--lo-frac 0.4 | --no-tiered] [--no-skip-last] [--lo-all [--lo-prefill]] [--no-prefetch] [--prefill-batch 32] [--dump-logits FILE]\n       moe-run <pack> --score IDS_FILE --n-prompt N --dump-logits FILE   (teacher forcing, for tests)";
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        pack: PathBuf::new(),
        prompt: None,
        chat: false,
        cache_gb: 8.0,
        lo_frac: 0.4,
        bias: 0.5,
        max_tokens: 512,
        threads: 16,
        dump_logits: None,
        prefetch: true,
        tiered: true,
        skip_last: true,
        lo_all: false,
        lo_prefill: false,
        prefill_batch: 32,
        score: None,
        n_prompt: 0,
    };
    let mut pack = None;
    while let Some(arg) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{arg} needs a value\n{usage}"));
        match arg.as_str() {
            "-p" | "--prompt" => a.prompt = Some(val()?),
            "--chat" => a.chat = true,
            "--cache-gb" => a.cache_gb = val()?.parse()?,
            "--bias" => a.bias = val()?.parse()?,
            "--max-tokens" => a.max_tokens = val()?.parse()?,
            "--threads" => a.threads = val()?.parse()?,
            "--dump-logits" => a.dump_logits = Some(val()?.into()),
            "--no-prefetch" => a.prefetch = false,
            "--lo-frac" => a.lo_frac = val()?.parse()?,
            "--no-tiered" => a.tiered = false,
            "--no-skip-last" => a.skip_last = false,
            "--lo-all" => a.lo_all = true,
            "--lo-prefill" => a.lo_prefill = true,
            "--prefill-batch" => a.prefill_batch = val()?.parse()?,
            "--score" => a.score = Some(val()?.into()),
            "--n-prompt" => a.n_prompt = val()?.parse()?,
            "-h" | "--help" => return Err(usage.into()),
            p if pack.is_none() && !p.starts_with('-') => pack = Some(PathBuf::from(p)),
            _ => return Err(format!("unknown argument {arg}\n{usage}").into()),
        }
    }
    a.pack = pack.ok_or(usage)?;
    if a.score.is_some() {
        if a.dump_logits.is_none() || a.n_prompt == 0 {
            return Err(format!("--score needs --n-prompt N and --dump-logits FILE\n{usage}").into());
        }
        return Ok(a);
    }
    if a.prompt.is_none() && !a.chat {
        return Err(format!("pass -p PROMPT or --chat\n{usage}").into());
    }
    Ok(a)
}

/// One user turn in the model's chat template, with the assistant prompt.
/// Qwen3: thinking disabled (what enable_thinking=False renders). DeepSeek-V3
/// / Kimi: the Kimi template, with its default system prompt on the first turn.
fn turn(arch: model::Arch, user: &str, first: bool) -> String {
    match arch {
        model::Arch::Qwen3 => format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        model::Arch::DeepseekV3 => format!(
            "{}<|im_user|>user<|im_middle|>{user}<|im_end|><|im_assistant|>assistant<|im_middle|>",
            if first { "<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|>" } else { "" }
        ),
    }
}

fn peak_rss_gb() -> f64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    ru.ru_maxrss as f64 / 1e9 // bytes on macOS
}

/// Generates a reply to `text` (appended to the conversation in `kv`),
/// streaming it to stdout. Returns the reply's tokens.
fn reply(model: &mut Model, tok: &Tokenizer, kv: &mut KvCache, text: &str, args: &Args) -> Res<Vec<u32>> {
    let prompt = tok.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec();
    let (read0, t0) = (model.store.bytes_read, Instant::now());
    let logits = model.forward(&prompt, kv)?;
    if let Some(path) = &args.dump_logits {
        let l = logits.as_type::<f32>()?;
        l.eval()?;
        let bytes: Vec<u8> = l.as_slice::<f32>().iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
    }
    let mut next = model::argmax(&logits)?;
    let t_prefill = t0.elapsed().as_secs_f64();
    let (mut out, mut shown) = (Vec::new(), String::new());
    let t1 = Instant::now();
    let (io0, sync0) = (model.store.t_io, model.t_sync);
    let im_end = tok.token_to_id("<|im_end|>");
    while out.len() < args.max_tokens && !model.cfg.eos.contains(&next) && Some(next) != im_end {
        out.push(next);
        let text = tok.decode(&out, true).map_err(|e| e.to_string())?;
        if text.len() > shown.len() && !text.ends_with('\u{fffd}') {
            print!("{}", &text[shown.len()..]);
            std::io::stdout().flush()?;
            shown = text;
        }
        next = model::argmax(&model.forward(&[next], kv)?)?;
    }
    let t_gen = t1.elapsed().as_secs_f64();
    if std::env::var_os("MOE_PROFILE").is_some() {
        let ms = |d: std::time::Duration| 1e3 * d.as_secs_f64() / out.len().max(1) as f64;
        eprintln!(
            "[profile, decode only, ms per token: pread {:.0} | GPU sync {:.0} | other {:.0} | total {:.0} | codebook prefetches needed as q4: {}]",
            ms(model.store.t_io - io0), ms(model.t_sync - sync0),
            ms(t1.elapsed() - (model.store.t_io - io0) - (model.t_sync - sync0)),
            1e3 * t_gen / out.len().max(1) as f64,
            model.prefetch_lo_wrong
        );
    }
    let s = model.stats();
    println!(
        "\n\n[prompt {} tok, {:.1} tok/s | gen {} tok, {:.2} tok/s | hit {:.1}% | read {:.1} GB ({} codebook reads, {} skipped) | prefetch {}/{} used | peak MLX {:.1} GB, RSS {:.1} GB]",
        prompt.len(),
        prompt.len() as f64 / t_prefill,
        out.len(),
        out.len() as f64 / t_gen.max(1e-9),
        100.0 * s.hit,
        (model.store.bytes_read - read0) as f64 / 1e9,
        model.store.lo_reads,
        model.skipped,
        model.store.prefetch_used,
        model.store.prefetched,
        mlx_rs::memory::peak_memory()? as f64 / 1e9,
        peak_rss_gb(),
    );
    Ok(out)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn run() -> Res<()> {
    let args = parse_args()?;
    // Freed buffers would otherwise pile up in MLX's cache next to the arena.
    mlx_rs::memory::set_cache_limit(512 << 20)?;
    let t0 = Instant::now();
    let tok = Tokenizer::from_file(args.pack.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let mut model = Model::load(&args.pack, args.cache_gb, if args.tiered { args.lo_frac } else { 0.0 }, args.lo_all, args.bias, args.threads, args.prefill_batch)?;
    model.prefetch = args.prefetch;
    model.lo_prefill = args.lo_prefill;
    model.skip_last_miss = args.skip_last && model.tiered;
    let st = model.stats();
    eprintln!(
        "[loaded in {:.1}s, {} q4 + {} codebook experts cached per layer, arenas {:.1} GB, bias {}{}{}]\n",
        t0.elapsed().as_secs_f64(),
        st.cap_hi,
        st.cap_lo,
        st.arena_gb,
        args.bias,
        if model.tiered { if model.skip_last_miss { ", tiered + skip last miss" } else { ", tiered" } } else { "" },
        if model.prefill_batch > 0 { format!(", prefill batch {}", model.prefill_batch) } else { String::new() }
    );
    let mut kv = model.new_cache();
    if let Some(path) = &args.score {
        // Feed the given tokens as generation would (prompt as one prefill,
        // then one at a time) and dump every step's logits as f32 rows.
        let ids: Vec<u32> = std::fs::read(path)?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut rows = Vec::new();
        let mut step = |logits: Array| -> Res<()> {
            let l = logits.as_type::<f32>()?;
            l.eval()?;
            rows.extend(l.as_slice::<f32>().iter().flat_map(|x| x.to_le_bytes()));
            Ok(())
        };
        step(model.forward(&ids[..args.n_prompt], &mut kv)?)?;
        for &t in &ids[args.n_prompt..ids.len() - 1] {
            step(model.forward(&[t], &mut kv)?)?;
        }
        std::fs::write(args.dump_logits.as_ref().unwrap(), rows)?;
        return Ok(());
    }
    if let Some(p) = &args.prompt {
        let t = turn(model.cfg.arch, p, true);
        reply(&mut model, &tok, &mut kv, &t, &args)?;
        return Ok(());
    }
    let stdin = std::io::stdin();
    loop {
        print!("\n> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            return Ok(());
        }
        // The previous reply's <|im_end|> was never fed to the model.
        let first = kv.offset == 0;
        let prefix = match (first, model.cfg.arch) {
            (true, _) => "",
            (false, model::Arch::Qwen3) => "<|im_end|>\n",
            (false, model::Arch::DeepseekV3) => "<|im_end|>",
        };
        let t = format!("{prefix}{}", turn(model.cfg.arch, line.trim(), first));
        reply(&mut model, &tok, &mut kv, &t, &args)?;
    }
}
