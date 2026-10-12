"""How much answer quality do lower-precision experts cost?

budget.py shows q3/q2 experts make a small RAM budget go much further. This
measures the price, against the stock 4-bit model as reference:

  kl       mean KL(ref || variant) of the next-token distribution, nats
  top1     how often the variant's most likely token equals the reference's
  dnll     change in mean NLL of the text itself (positive = worse)

Only the routed experts change; attention, router and embeddings stay 4-bit,
since they live in RAM anyway. Variants:

  q4 re-quant floor  4-bit experts put through the same re-quantization:
                   the error the method itself adds, to subtract mentally
  q3, q2           every expert call at 3 / 2 bits
  q2 rank 5-8      the four lowest-weight picks of each token at 2 bits
  q2 on p% calls   a random p% of expert calls at 2 bits, the rest at 4: a
                   tiered cache where hits are 4-bit and misses are served
                   from a 2-bit copy (reads ~half the bytes)

Low-bit copies are re-quantized from the 4-bit weights (MLX affine, g64), so
they carry both quantization errors: a slightly pessimistic estimate.

Text: the prompt set with a greedy continuation from the reference model,
plus a few human-written files from this repo. Each sequence is cut to
--seq-len tokens.

Usage: uv run quality.py [--gen-tokens 160] [--seq-len 768]
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate

from prompts import PROMPTS, _REPO

HUMAN_FILES = ["README.md", "tools/routing-trace/README.md", "src/replay.rs"]


class TieredMoE(nn.Module):
    """Wraps a Qwen3 MoE block; routes some expert calls to a low-bit copy."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.lo = None
        self.rule = "ref"  # ref | all | rank | p<float>

    def __call__(self, x):
        b = self.block
        if self.rule == "ref" or self.lo is None:
            return b(x)
        gates = mx.softmax(b.gate(x), axis=-1, precise=True)
        k = b.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if b.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)
        y_lo = self.lo(x, inds)
        if self.rule == "all":
            y = y_lo
        else:
            if self.rule == "rank":
                rank = mx.argsort(mx.argsort(-scores, axis=-1), axis=-1)
                use_lo = rank >= k // 2
            else:
                use_lo = mx.random.uniform(shape=inds.shape) < float(self.rule[1:])
            y = mx.where(use_lo[..., None], y_lo, b.switch_mlp(x, inds))
        return (y * scores[..., None]).sum(axis=-2).astype(y.dtype)


def low_bit_copy(sw, bits: int):
    """Shallow copy of a quantized SwitchGLU with experts re-quantized."""
    lo = copy.copy(sw)
    for name in ("gate_proj", "up_proj", "down_proj"):
        src = getattr(sw, name)
        dst = copy.copy(src)
        w = mx.dequantize(src.weight, src.scales, src.biases,
                          group_size=src.group_size, bits=src.bits)
        dst.weight, dst.scales, dst.biases = mx.quantize(w, group_size=src.group_size, bits=bits)
        dst.bits = bits
        mx.eval(dst.weight, dst.scales, dst.biases)
        setattr(lo, name, dst)
    return lo


def install(model) -> list[TieredMoE]:
    wraps = []
    for layer in model.layers:
        w = TieredMoE(layer.mlp)
        layer.mlp = w
        wraps.append(w)
    return wraps


def build_corpus(model, tokenizer, gen_tokens: int, seq_len: int) -> list[tuple[str, list[int]]]:
    seqs = []
    for p in PROMPTS:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": p["text"]}],
            add_generation_prompt=True, enable_thinking=p["thinking"])
        out = [r.token for r in stream_generate(model, tokenizer, prompt, max_tokens=gen_tokens)]
        seqs.append((p["id"], (list(prompt) + out)[-seq_len:]))
    for rel in HUMAN_FILES:
        toks = tokenizer.encode((_REPO / rel).read_text())
        seqs.append((rel, toks[:seq_len]))
    return seqs


def logprobs(model, toks):
    logits = model(mx.array(toks)[None, :-1])[0].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--gen-tokens", type=int, default=160)
    ap.add_argument("--seq-len", type=int, default=768)
    ap.add_argument("--json", type=Path, default=Path("traces/quality.json"))
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    t0 = time.time()
    corpus = build_corpus(model, tokenizer, args.gen_tokens, args.seq_len)
    ntok = sum(len(t) - 1 for _, t in corpus)
    print(f"corpus: {len(corpus)} sequences, {ntok} scored tokens ({time.time() - t0:.0f}s)")

    wraps = install(model)
    ref = []  # per sequence: (log-probs f16, top-1, nll of text)
    for _, toks in corpus:
        lp = logprobs(model, toks)
        tgt = mx.array(toks[1:])
        nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1).sum()
        ref.append((lp.astype(mx.float16), mx.argmax(lp, axis=-1), nll))
        mx.eval(ref[-1])

    variants = [(4, "all", "q4 re-quant floor"), (3, "all", "q3"), (2, "all", "q2"), (2, "rank", "q2 rank 5-8"),
                (2, "p0.25", "q2 on 25% calls"), (2, "p0.5", "q2 on 50% calls")]
    results = {}
    built = None
    for bits, rule, name in variants:
        if built != bits:
            for w in wraps:
                w.lo = None
            mx.clear_cache()
            for w in wraps:
                w.lo = low_bit_copy(w.block.switch_mlp, bits)
            built = bits
        mx.random.seed(0)
        for w in wraps:
            w.rule = rule
        kl = agree = dnll = 0.0
        for (_, toks), (rlp, rtop, rnll) in zip(corpus, ref):
            lp = logprobs(model, toks)
            rlp32 = rlp.astype(mx.float32)
            kl += (mx.exp(rlp32) * (rlp32 - lp)).sum().item()
            agree += (mx.argmax(lp, axis=-1) == rtop).sum().item()
            tgt = mx.array(toks[1:])
            dnll += (-mx.take_along_axis(lp, tgt[:, None], axis=-1).sum() - rnll).item()
        results[name] = {"bits": bits, "rule": rule, "kl": kl / ntok,
                         "top1": agree / ntok, "dnll": dnll / ntok,
                         "ppl_ratio": math.exp(dnll / ntok)}
        r = results[name]
        print(f"{name:18s} kl {r['kl']:.4f}  top1 {100 * r['top1']:.1f}%  "
              f"dnll {r['dnll']:+.4f}  ppl x{r['ppl_ratio']:.3f}")
    args.json.write_text(json.dumps({"model": args.model, "tokens": ntok,
                                     "variants": results}, indent=1) + "\n")


if __name__ == "__main__":
    main()
