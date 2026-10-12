"""Generate text with a packed model, experts streamed from the SSD.

Usage:
  uv run run.py ~/packs/qwen3-235b --cache-gb 8 --bias 0.5 -p "Explique..." [--max-tokens 200]
  uv run run.py ~/packs/qwen3-235b --chat
"""

import argparse
import resource
import time

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

import engine


def generate(model, tokenizer, store, messages, args) -> str:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=args.think)
    read0, t0 = store.bytes_read, time.time()
    text, last = "", None
    for r in stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens,
                             sampler=make_sampler(temp=args.temp)):
        print(r.text, end="", flush=True)
        text += r.text
        last = r
    s = engine.stats(model, store)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    print(f"\n\n[prompt {last.prompt_tokens} tok, {last.prompt_tps:.1f} tok/s | "
          f"gen {last.generation_tokens} tok, {last.generation_tps:.2f} tok/s | "
          f"hit {100 * s['hit']:.1f}% | read {(store.bytes_read - read0) / 1e9:.1f} GB "
          f"in {time.time() - t0:.0f}s | MLX peak {mx.get_peak_memory() / 1e9:.1f} GB, RSS peak {rss:.1f} GB]")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("-p", "--prompt")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--cache-gb", type=float, default=8.0, help="RAM for cached experts")
    ap.add_argument("--bias", type=float, default=0.5, help="cache-aware routing (0 = stock routing)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--think", action="store_true", help="let Qwen3 think before answering")
    args = ap.parse_args()

    mx.set_cache_limit(512 * 2**20)  # don't let MLX hoard freed expert buffers
    t0 = time.time()
    model, tokenizer, store = engine.load(args.pack, args.cache_gb, args.bias)
    print(f"[loaded in {time.time() - t0:.0f}s, {mx.get_active_memory() / 1e9:.1f} GB resident, "
          f"{engine.stats(model, store)['slots_per_layer']} experts cached per layer]\n")

    if not args.chat:
        generate(model, tokenizer, store, [{"role": "user", "content": args.prompt}], args)
        return
    messages = []
    while True:
        try:
            q = input("\n> ")
        except EOFError:
            break
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": generate(model, tokenizer, store, messages, args)})


if __name__ == "__main__":
    main()
