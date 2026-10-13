"""engine-rs's moe-run must match stock mlx-lm the way test_engine.py checks
the Python engine: on each prompt plus the stock model's greedy continuation,
scored as generation runs, the KL to the stock model's incremental log-probs
must stay within twice the stock model's own noise (incremental vs one full
forward pass). Greedy text is reported but not required to match: 1-ulp
differences in gather_qmm flip a token now and then.

Usage: uv run test_rust.py ~/packs/qwen3-30b [--bin ../engine-rs/target/release/moe-run] [--cache-gb 0.5,100]
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import generate, load

from test_engine import NOISE_FACTOR, PROMPTS, logprobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--bin", default=str(Path(__file__).resolve().parent.parent / "engine-rs/target/release/moe-run"))
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--cache-gb", default="0.5,100")
    ap.add_argument("--extra", default="", help="extra moe-run flags, e.g. '--no-prefetch'")
    ap.add_argument("--max-kl", type=float, help="pass on KL below this instead of the stock-noise rule "
                    "(two-resolution mode is expected to cost ~0.02 nats, tools/routing-trace/codebook.py)")
    args = ap.parse_args()

    stock, tok = load(args.model)
    cases = []
    for p in PROMPTS:
        prompt = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                         enable_thinking=False)
        text = generate(stock, tok, prompt, max_tokens=args.tokens)
        ids = list(prompt) + tok.encode(text, add_special_tokens=False)
        ref = logprobs(stock, ids, len(prompt))
        full = stock(mx.array(ids)[None])[0, len(prompt) - 1:-1].astype(mx.float32)
        full = full - mx.logsumexp(full, axis=-1, keepdims=True)
        noise = (mx.exp(ref) * (ref - full)).sum(axis=-1).mean().item()
        cases.append((p, ids, len(prompt), np.array(ref), noise))
    del stock
    mx.clear_cache()

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for cache_gb in args.cache_gb.split(","):
            for p, ids, n, ref, noise in cases:
                ids_f, out_f = Path(tmp) / "ids.bin", Path(tmp) / "logits.bin"
                ids_f.write_bytes(np.array(ids, dtype=np.uint32).tobytes())
                cmd = [args.bin, args.pack, "--bias", "0", "--cache-gb", cache_gb, "--score", str(ids_f),
                       "--n-prompt", str(n), "--dump-logits", str(out_f)] + args.extra.split()
                subprocess.run(cmd, check=True, capture_output=True)
                got = np.fromfile(out_f, dtype=np.float32).reshape(len(ids) - n, -1)
                lp = got - got.max(-1, keepdims=True)
                lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
                kl = float((np.exp(ref) * (ref - lp)).sum(-1).mean())
                top1 = float((lp.argmax(-1) == ref.argmax(-1)).mean())
                passed = kl <= (args.max_kl if args.max_kl else NOISE_FACTOR * noise)
                ok &= passed
                print(f"cache {cache_gb} GB | {'ok  ' if passed else 'FAIL'} kl {kl:.2e} (stock noise {noise:.2e}) "
                      f"top1 {100 * top1:.1f}% | {p[:40]}")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
