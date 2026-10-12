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
  q3/q2 by rank    same, with the four highest-weight picks at 3 bits
  q2 rank 5-7 + skip 8 [on 50%]  two-resolution cache that also drops the
                   lowest-weight pick (always, or on half the tokens as a
                   stand-in for "only when it misses")
  skip rank 8 / 7-8  drop the lowest-weight pick(s) of every token without
                   renormalizing (upper bound on skipping low-weight misses)
  ... hadamard     the low-bit copy quantizes W·H (H a Hadamard rotation of
                   the input dim) and rotates activations to match
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
        self.hi = None  # low-bit copy for the other calls; None = stock 4-bit
        self.rule = "ref"  # ref | all | rank | p<float>

    def __call__(self, x):
        b = self.block
        if self.rule == "ref" or self.lo is None:
            return b(x)
        gates = mx.softmax(b.gate(x), axis=-1, precise=True)
        k = b.top_k
        inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if b.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)
        if self.rule.startswith("skip"):
            # drop the n lowest-weight picks, as if their misses were skipped
            rank = mx.argsort(mx.argsort(-scores, axis=-1), axis=-1)
            keep = rank < k - int(self.rule[4:])
            y = b.switch_mlp(x, inds) * keep[..., None]
            return (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        y_lo = self.lo(x, inds)
        if self.rule == "all":
            y = y_lo
        else:
            if self.rule.startswith("rank"):
                rank = mx.argsort(mx.argsort(-scores, axis=-1), axis=-1)
                use_lo = rank >= k // 2
                if self.rule.startswith("rankskip"):
                    # also drop the lowest-weight pick, always or with prob p
                    p = float(self.rule[8:] or 1.0)
                    drop = (rank == k - 1) & (mx.random.uniform(shape=inds.shape) < p)
                    scores = mx.where(drop, 0.0, scores)
            else:
                use_lo = mx.random.uniform(shape=inds.shape) < float(self.rule[1:])
            hi = self.hi if self.hi is not None else b.switch_mlp
            y = mx.where(use_lo[..., None], y_lo, hi(x, inds))
        return (y * scores[..., None]).sum(axis=-2).astype(y.dtype)


class Rotated(nn.Module):
    """Quantized switch linear stored as W·H: rotates the input by the same
    orthonormal Hadamard H so the output is unchanged (H·H = I)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, x, indices, sorted_indices=False):
        return self.inner(mx.hadamard_transform(x), indices, sorted_indices=sorted_indices)


def low_bit_copy(sw, bits: int, rotate: bool = False):
    """Shallow copy of a quantized SwitchGLU with experts re-quantized.

    rotate: quantize W·H instead of W (incoherence processing, as in QuIP#),
    which spreads outlier weights across each group before rounding."""
    lo = copy.copy(sw)
    for name in ("gate_proj", "up_proj", "down_proj"):
        src = getattr(sw, name)
        dst = copy.copy(src)
        dst._no_grad = set(getattr(src, "_no_grad", ()))  # own freeze state, not src's
        w = mx.dequantize(src.weight, src.scales, src.biases,
                          group_size=src.group_size, bits=src.bits)
        if rotate:
            w = mx.hadamard_transform(w.astype(mx.float32)).astype(w.dtype)
        dst.weight, dst.scales, dst.biases = mx.quantize(w, group_size=src.group_size, bits=bits)
        dst.bits = bits
        mx.eval(dst.weight, dst.scales, dst.biases)
        setattr(lo, name, Rotated(dst) if rotate else dst)
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
    ap.add_argument("--only", help="comma-separated variant names; merges into --json")
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    t0 = time.time()
    corpus = build_corpus(model, tokenizer, args.gen_tokens, args.seq_len)
    ntok = sum(len(t) - 1 for _, t in corpus)
    print(f"corpus: {len(corpus)} sequences, {ntok} scored tokens ({time.time() - t0:.0f}s)")

    results = {}
    wraps = install(model)
    ref = []  # per sequence: (log-probs f16, top-1, nll of text)
    for _, toks in corpus:
        lp = logprobs(model, toks)
        tgt = mx.array(toks[1:])
        nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1).sum()
        ref.append((lp.astype(mx.float16), mx.argmax(lp, axis=-1), nll))
        mx.eval(ref[-1])

    variants = [(4, "all", "q4 re-quant floor"), (3, "all", "q3"), (2, "all", "q2"), (2, "rank", "q2 rank 5-8"),
                (2, "p0.25", "q2 on 25% calls"), (2, "p0.5", "q2 on 50% calls"),
                (2, "rank3", "q3/q2 by rank"),
                (3, "all", "q3 hadamard", True), (2, "all", "q2 hadamard", True),
                (2, "rank", "q2 hadamard rank 5-8", True),
                (2, "skip1", "skip rank 8"), (2, "skip2", "skip ranks 7-8"),
                (2, "rankskip", "q2 rank 5-7 + skip 8"),
                (2, "rankskip0.5", "q2 rank 5-7 + skip 8 on 50%")]
    if args.only:
        variants = [v for v in variants if v[2] in args.only.split(",")]
        if args.json.exists():
            results = json.loads(args.json.read_text())["variants"]
    built = None
    for bits, rule, name, *rot in variants:
        rotate = bool(rot and rot[0])
        if built != (bits, rotate):
            for w in wraps:
                w.lo = None
            mx.clear_cache()
            for w in wraps:
                w.lo = low_bit_copy(w.block.switch_mlp, bits, rotate)
            built = (bits, rotate)
        mx.random.seed(0)
        for w in wraps:
            w.hi = None
            w.rule = rule
        if rule == "rank3":
            for w in wraps:
                w.hi = low_bit_copy(w.block.switch_mlp, 3)
                w.rule = "rank"
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
