//! moe-run: chat with a large MoE model whose experts stay on the SSD.
//!
//!   moe-run <pack> [-p PROMPT | --chat] [--cache-gb 8] [--bias 0.5]
//!           [--max-tokens 512] [--threads 8] [--dump-logits FILE]
//!
//! <pack> is a directory written by engine/pack.py.

mod experts;
mod model;

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use experts::Res;
use model::{KvCache, Model};
use tokenizers::Tokenizer;

struct Args {
    pack: PathBuf,
    prompt: Option<String>,
    chat: bool,
    cache_gb: f64,
    bias: f32,
    max_tokens: usize,
    threads: usize,
    dump_logits: Option<PathBuf>,
}

fn parse_args() -> Res<Args> {
    let usage = "usage: moe-run <pack> [-p PROMPT | --chat] [--cache-gb 8] [--bias 0.5] [--max-tokens 512] [--threads 8] [--dump-logits FILE]";
    let mut it = std::env::args().skip(1);
    let mut a = Args {
        pack: PathBuf::new(),
        prompt: None,
        chat: false,
        cache_gb: 8.0,
        bias: 0.5,
        max_tokens: 512,
        threads: 8,
        dump_logits: None,
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
            "-h" | "--help" => return Err(usage.into()),
            p if pack.is_none() && !p.starts_with('-') => pack = Some(PathBuf::from(p)),
            _ => return Err(format!("unknown argument {arg}\n{usage}").into()),
        }
    }
    a.pack = pack.ok_or(usage)?;
    if a.prompt.is_none() && !a.chat {
        return Err(format!("pass -p PROMPT or --chat\n{usage}").into());
    }
    Ok(a)
}

/// Qwen3 chat turn, thinking disabled (what enable_thinking=False renders).
fn turn(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
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
    let s = model.stats();
    println!(
        "\n\n[prompt {} tok, {:.1} tok/s | gen {} tok, {:.2} tok/s | hit {:.1}% | read {:.1} GB | peak RSS {:.1} GB]",
        prompt.len(),
        prompt.len() as f64 / t_prefill,
        out.len(),
        out.len() as f64 / t_gen.max(1e-9),
        100.0 * s.hit,
        (model.store.bytes_read - read0) as f64 / 1e9,
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
    let t0 = Instant::now();
    let tok = Tokenizer::from_file(args.pack.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let mut model = Model::load(&args.pack, args.cache_gb, args.bias, args.threads)?;
    eprintln!(
        "[loaded in {:.1}s, {} experts cached per layer, bias {}]\n",
        t0.elapsed().as_secs_f64(),
        model.stats().slots,
        args.bias
    );
    let mut kv = model.new_cache();
    if let Some(p) = &args.prompt {
        reply(&mut model, &tok, &mut kv, &turn(p), &args)?;
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
        let prefix = if kv.offset > 0 { "<|im_end|>\n" } else { "" };
        reply(&mut model, &tok, &mut kv, &format!("{prefix}{}", turn(line.trim())), &args)?;
    }
}
