"""The streaming engine must match stock mlx-lm when routing is unchanged
(bias 0), whatever the cache size.

Bit-exact output is not expected: mx.gather_qmm rounds differently (1 bf16
ulp) depending on how many experts the stacked weights hold, and the engine
stacks only the experts a call needs (in decode, always; in a prefill, until
the call needs all of them). So each prompt plus the stock model's greedy
continuation is scored by both models the way generation runs (prompt as one
prefill, then one token at a time with a KV cache). The yardstick is the stock
model's own noise: how far its incremental log-probs are from its full-
sequence forward pass, which is the same math in a different order. The engine
must stay within twice that.

Usage: uv run test_engine.py ~/packs/qwen3-30b [--model mlx-community/Qwen3-30B-A3B-4bit]
"""

import argparse

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache

import engine

PROMPTS = [
    "Write a Rust function that reverses a linked list.",
    "Explique em poucas frases por que o céu é azul.",
    "List three prime numbers greater than 100 and explain how you checked them.",
]
NOISE_FACTOR = 2.0


def logprobs(model, ids, n_prompt):
    """Log-probs of each next token, computed as generation does."""
    cache = make_prompt_cache(model)
    out = [model(mx.array(ids[:n_prompt])[None], cache=cache)[0, -1]]
    for t in ids[n_prompt:-1]:
        out.append(model(mx.array([[t]]), cache=cache)[0, -1])
    logits = mx.stack(out).astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--tokens", type=int, default=64)
    args = ap.parse_args()

    stock, tok = load(args.model)
    seqs, want = [], []
    for p in PROMPTS:
        prompt = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                         enable_thinking=False)
        text = generate(stock, tok, prompt, max_tokens=args.tokens)
        ids = list(prompt) + tok.encode(text, add_special_tokens=False)
        seqs.append((ids, len(prompt)))
        ref = logprobs(stock, ids, len(prompt))
        full = stock(mx.array(ids)[None])[0, len(prompt) - 1:-1].astype(mx.float32)
        full = full - mx.logsumexp(full, axis=-1, keepdims=True)
        noise = (mx.exp(ref) * (ref - full)).sum(axis=-1).mean().item()
        want.append((text, ref, noise))
        mx.eval(ref)
    del stock
    mx.clear_cache()

    ok = True
    for cache_gb in (0.5, 100.0):  # forced misses, then everything fits
        model, tok, store = engine.load(args.pack, cache_gb=cache_gb, bias=0.0)
        for p, (ids, n), (text, ref, noise) in zip(PROMPTS, seqs, want):
            lp = logprobs(model, ids, n)
            kl = (mx.exp(ref) * (ref - lp)).sum(axis=-1).mean().item()
            top1 = (mx.argmax(lp, -1) == mx.argmax(ref, -1)).mean().item()
            prompt = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                             enable_thinking=False)
            same = generate(model, tok, prompt, max_tokens=args.tokens) == text
            passed = kl <= NOISE_FACTOR * noise
            ok &= passed
            print(f"cache {cache_gb:g} GB | {'ok  ' if passed else 'FAIL'} kl {kl:.2e} (stock noise {noise:.2e}) "
                  f"top1 {100 * top1:.1f}% "
                  f"| greedy text {'identical' if same else 'diverges'} | {p[:40]}")
        print(f"  {engine.stats(model, store)}")
        del model
        mx.clear_cache()
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
