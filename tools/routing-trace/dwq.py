"""Can distillation make the 2-bit expert copy good enough to use more?

The two-resolution cache serves the four lowest-weight picks of each token
from a 2-bit copy (perplexity x1.034, plain round-to-nearest). DWQ (MLX's
distilled weight quantization) keeps the 2-bit codes and tunes only the
per-group scales and biases so the model's output matches a reference.
Here the reference is the stock 4-bit model itself, the student is the same
model with its low-weight picks on the 2-bit copies (the rule the cache
uses), and only those copies' scales and biases train.

To keep the evaluation honest, calibration text comes only from the train
split of the prompt set (sampled continuations), and the before/after
numbers are measured on the test split plus human-written files, the same
metrics as quality.py.

Optimizer state for every expert's scales and biases would not fit next to
the model in 48 GB, so layers train in groups (coordinate descent).

Usage: uv run dwq.py [--lr 1e-5] [--epochs 2] [--group 12]
"""

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_map
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.losses import kl_div_loss

from prompts import PROMPTS
from quality import HUMAN_FILES, build_corpus, install, logprobs, low_bit_copy
from sim import split


def calibration(model, tokenizer, prompts, samples: int, gen_tokens: int, seq_len: int):
    seqs = []
    sampler = make_sampler(temp=0.7, top_p=0.8)
    for p in prompts:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": p["text"]}],
            add_generation_prompt=True, enable_thinking=p["thinking"])
        for i in range(samples):
            mx.random.seed(1000 + i)
            out = [r.token for r in stream_generate(model, tokenizer, prompt,
                                                    max_tokens=gen_tokens, sampler=sampler)]
            seqs.append((list(prompt) + out)[-seq_len:])
    return seqs


def set_rule(wraps, rule):
    for w in wraps:
        w.rule = rule


def evaluate(model, wraps, corpus, ref, rule) -> dict:
    set_rule(wraps, rule)
    ntok = sum(len(t) - 1 for t in corpus)
    kl = agree = dnll = 0.0
    for toks, (rlp, rtop, rnll) in zip(corpus, ref):
        lp = logprobs(model, toks)
        rlp32 = rlp.astype(mx.float32)
        kl += (mx.exp(rlp32) * (rlp32 - lp)).sum().item()
        agree += (mx.argmax(lp, axis=-1) == rtop).sum().item()
        tgt = mx.array(toks[1:])
        dnll += (-mx.take_along_axis(lp, tgt[:, None], axis=-1).sum() - rnll).item()
    return {"kl": kl / ntok, "top1": agree / ntok, "ppl_ratio": math.exp(dnll / ntok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--group", type=int, default=12, help="layers trained together")
    ap.add_argument("--samples", type=int, default=4, help="sampled continuations per train prompt")
    ap.add_argument("--gen-tokens", type=int, default=384)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--json", type=Path, default=Path("traces/dwq.json"))
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    train_p, test_p = split(PROMPTS)
    t0 = time.time()
    calib = calibration(model, tokenizer, train_p, args.samples, args.gen_tokens, args.seq_len)
    test_ids = {p["id"] for p in test_p} | set(HUMAN_FILES)
    corpus = [t for i, t in build_corpus(model, tokenizer, 160, 768) if i in test_ids]
    print(f"calibration {len(calib)} seqs / {sum(map(len, calib))} tokens (train prompts), "
          f"eval {len(corpus)} seqs (test prompts + human files), {time.time() - t0:.0f}s")

    wraps = install(model)
    ref = []
    for toks in corpus:
        lp = logprobs(model, toks)
        nll = -mx.take_along_axis(lp, mx.array(toks[1:])[:, None], axis=-1).sum()
        ref.append((lp.astype(mx.float16), mx.argmax(lp, axis=-1), nll))
        mx.eval(ref[-1])
    for w in wraps:
        w.lo = low_bit_copy(w.block.switch_mlp, 2)

    results = {"before": {r: evaluate(model, wraps, corpus, ref, r) for r in ("rank", "all")}}
    for r, m in results["before"].items():
        print(f"before  q2 {r:4s}  kl {m['kl']:.4f}  top1 {100 * m['top1']:.1f}%  ppl x{m['ppl_ratio']:.3f}")

    model.freeze()
    for g0 in range(0, len(wraps), args.group):
        group = wraps[g0: g0 + args.group]
        model.freeze()
        for w in group:
            for proj in (w.lo.gate_proj, w.lo.up_proj, w.lo.down_proj):
                proj.unfreeze(keys=["scales", "biases"], recurse=False)
        params = tree_map(lambda x: x.astype(mx.float32), model.trainable_parameters())
        opt = optim.Adam(learning_rate=args.lr)

        def loss_fn(params, x, target):
            model.update(tree_map(lambda p: p.astype(mx.bfloat16), params))
            return kl_div_loss(model(x), target).mean()

        step_loss = []
        for ep in range(args.epochs):
            for i in mx.random.permutation(len(calib)).tolist():
                x = mx.array(calib[i])[None]
                set_rule(wraps, "ref")
                target = mx.stop_gradient(model(x))
                set_rule(wraps, "rank")
                loss, grads = mx.value_and_grad(loss_fn)(params, x, target)
                params = opt.apply_gradients(grads, params)
                mx.eval(loss, params)
                step_loss.append(loss.item())
            n = len(calib)
            print(f"layers {g0}-{g0 + len(group) - 1} epoch {ep}: "
                  f"kl {sum(step_loss[-n:]) / n:.4f} (first steps {sum(step_loss[:8]) / 8:.4f})", flush=True)
        model.update(tree_map(lambda p: p.astype(mx.bfloat16), params))
        mx.eval(model.parameters())
        del params, opt
        mx.clear_cache()

    model.freeze()
    results["after"] = {r: evaluate(model, wraps, corpus, ref, r) for r in ("rank", "all")}
    for r, m in results["after"].items():
        print(f"after   q2 {r:4s}  kl {m['kl']:.4f}  top1 {100 * m['top1']:.1f}%  ppl x{m['ppl_ratio']:.3f}")
    results["args"] = vars(args) | {"json": str(args.json)}
    args.json.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
