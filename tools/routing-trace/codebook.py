"""Codebook (vector) quantization of the experts: does ~2 bits keep quality?

The affine q2 tried in quality.py (2.5 bits/weight with its g64 scales) costs
ppl x1.345 on every expert call. Vector quantization methods (AQLM, QuIP#,
QTIP) store groups of d weights as an index into a learned codebook and
report near-4-bit quality at 2 bits. This checks that before any kernel work:

  vq d<d> k<K>   each output row is scaled to unit RMS (fp16 scale per row),
                 split into vectors of d weights, and every vector replaced by
                 its nearest of K centroids. One codebook per (layer, proj),
                 fitted by k-means on a sample. log2(K)/d bits per weight.
  ... had        quantize W·H instead (H an orthonormal Hadamard rotation of
                 the input dim, as QuIP# does), rotating activations to match.

Plain k-means, no activation weighting and no fine-tuning, so this is a
floor for what codebooks can do. The reconstructed weights are stored as
5-bit affine for the forward pass (error far below the codebook's), and,
like quality.py, start from the 4-bit weights.

Each variant is measured on every expert call ("all") and on the four
lowest-weight picks only ("rank 5-8", the two-resolution cache).

Usage: uv run codebook.py [--only "vq d4 k256,vq d8 k65536 had"]
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import mlx.core as mx

from quality import Rotated, build_corpus, install, logprobs
from mlx_lm import load

VARIANTS = [  # (name, d, K, rotate, down_bits: None = codebook, 4 = stock q4, else affine re-quant)
    ("vq d4 k256", 4, 256, False, None),
    ("vq d4 k256 had", 4, 256, True, None),
    ("vq d4 k1024 had", 4, 1024, True, None),  # 2.5 bits, same as affine q2 g64
    ("vq d8 k65536 had", 8, 65536, True, None),  # ~4 h of k-means on an M5 Max
    # Mixed precision inside the expert: the community's Kimi-K2 quants show
    # down_proj is the cheapest projection to keep a bit higher.
    ("vq d4 k1024 had down q3", 4, 1024, True, 3),  # 2.83 bits
    ("vq d4 k1024 had down q4", 4, 1024, True, 4),  # 3.17 bits
]
SAMPLE = 1 << 20  # vectors used to fit each codebook
ITERS = 10
CHUNK = 1 << 13  # vectors per distance block
# The reconstruction is stored as affine 5-bit for the forward pass: at 8 bits
# the copy of the 30B's ~29B expert weights alone is ~31 GB, which next to the
# 16 GB model does not fit in 48 GB. 5 bits adds ~0.05 relative error on top
# of the codebook's ~0.3 (a slightly pessimistic test).
STORE_BITS = 5


def assign(x: mx.array, c: mx.array) -> mx.array:
    """Index of the nearest centroid for each row of x."""
    cn = (c * c).sum(axis=1)
    out = []
    for i in range(0, x.shape[0], CHUNK):
        xb = x[i:i + CHUNK]
        out.append(mx.argmin(cn[None, :] - 2 * (xb @ c.T), axis=1).astype(mx.uint32))
        mx.eval(out[-1])
    return mx.concatenate(out)


def kmeans(x: mx.array, k: int) -> mx.array:
    n, d = x.shape
    sample = x[mx.random.randint(0, n, (min(SAMPLE, n),))]
    c = sample[mx.random.permutation(sample.shape[0])[:k]]
    for _ in range(ITERS):
        idx = assign(sample, c)
        sums = mx.zeros((k, d)).at[idx].add(sample)
        cnt = mx.zeros((k,)).at[idx].add(mx.ones(idx.shape))
        c = mx.where(cnt[:, None] > 0, sums / mx.maximum(cnt, 1)[:, None], c)  # empty clusters keep their centroid
        mx.eval(c)
    return c


def vq(w: mx.array, d: int, k: int) -> tuple[mx.array, float]:
    """Codebook-quantize w [experts, out, in]; returns the reconstruction and
    the relative error ||w - w'|| / ||w||."""
    w = w.astype(mx.float32)
    scale = mx.sqrt((w * w).mean(axis=-1, keepdims=True)) + 1e-8
    vecs = (w / scale).reshape(-1, d)
    c = kmeans(vecs, k)
    rec = c[assign(vecs, c)].reshape(w.shape) * scale.astype(mx.float16).astype(mx.float32)
    err = (mx.linalg.norm((w - rec).flatten()) / mx.linalg.norm(w.flatten())).item()
    return rec, err


def vq_copy(sw, d: int, k: int, rotate: bool, down_bits: int | None = None):
    """Shallow copy of a quantized SwitchGLU with codebook-quantized experts.

    down_bits: None keeps down_proj in the codebook too; 4 leaves the stock
    q4 down_proj; another value re-quantizes it to that many affine bits."""
    lo = copy.copy(sw)
    errs = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        src = getattr(sw, name)
        dst = copy.copy(src)
        dst._no_grad = set(getattr(src, "_no_grad", ()))
        w = mx.dequantize(src.weight, src.scales, src.biases, group_size=src.group_size, bits=src.bits)
        w = w.astype(mx.float32)
        if name == "down_proj" and down_bits is not None:
            if down_bits != src.bits:
                dst.weight, dst.scales, dst.biases = mx.quantize(w, group_size=src.group_size, bits=down_bits)
                dst.bits = down_bits
                mx.eval(dst.weight, dst.scales, dst.biases)
            setattr(lo, name, dst)
            continue
        if rotate:
            w = mx.hadamard_transform(w)
        rec, err = vq(w, d, k)
        errs.append(err)
        dst.weight, dst.scales, dst.biases = mx.quantize(rec.astype(mx.bfloat16), group_size=64, bits=STORE_BITS)
        dst.bits, dst.group_size = STORE_BITS, 64
        mx.eval(dst.weight, dst.scales, dst.biases)
        setattr(lo, name, Rotated(dst) if rotate else dst)
    return lo, errs


def score(model, corpus, ref, ntok):
    kl = agree = dnll = 0.0
    for (_, toks), (rlp, rtop, rnll) in zip(corpus, ref):
        lp = logprobs(model, toks)
        rlp32 = rlp.astype(mx.float32)
        kl += (mx.exp(rlp32) * (rlp32 - lp)).sum().item()
        agree += (mx.argmax(lp, axis=-1) == rtop).sum().item()
        tgt = mx.array(toks[1:])
        dnll += (-mx.take_along_axis(lp, tgt[:, None], axis=-1).sum() - rnll).item()
    return {"kl": kl / ntok, "top1": agree / ntok, "dnll": dnll / ntok, "ppl_ratio": math.exp(dnll / ntok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--gen-tokens", type=int, default=160)
    ap.add_argument("--seq-len", type=int, default=768)
    ap.add_argument("--json", type=Path, default=Path("traces/codebook.json"))
    ap.add_argument("--only", help="comma-separated variant names; merges into --json")
    ap.add_argument("--rules", default="all,rank", help="which rules to score: all (every call), rank (picks 5-8)")
    args = ap.parse_args()

    mx.random.seed(0)
    # The k-means temporaries are ~0.8 GB per projection; without a limit
    # MLX keeps the freed buffers cached and the run grows past 48 GB.
    mx.set_cache_limit(2 << 30)
    model, tokenizer = load(args.model)
    t0 = time.time()
    corpus = build_corpus(model, tokenizer, args.gen_tokens, args.seq_len)
    ntok = sum(len(t) - 1 for _, t in corpus)
    print(f"corpus: {len(corpus)} sequences, {ntok} scored tokens ({time.time() - t0:.0f}s)", flush=True)

    wraps = install(model)
    ref = []
    for _, toks in corpus:
        lp = logprobs(model, toks)
        tgt = mx.array(toks[1:])
        ref.append((lp.astype(mx.float16), mx.argmax(lp, axis=-1),
                    -mx.take_along_axis(lp, tgt[:, None], axis=-1).sum()))
        mx.eval(ref[-1])

    results = json.loads(args.json.read_text())["variants"] if args.only and args.json.exists() else {}
    variants = [v for v in VARIANTS if not args.only or v[0] in args.only.split(",")]
    for name, d, k, rotate, down_bits in variants:
        t0 = time.time()
        for w in wraps:
            w.lo = None
        mx.clear_cache()
        errs = []
        for i, w in enumerate(wraps):
            w.lo, e = vq_copy(w.block.switch_mlp, d, k, rotate, down_bits)
            errs += e
            print(f"  {name}: layer {i + 1}/{len(wraps)}, rel err {sum(e) / len(e):.3f} ({time.time() - t0:.0f}s)",
                  flush=True)
        # bits per weight over the whole expert (gate and up are 2/3 of it)
        bits = math.log2(k) / d if down_bits is None else (2 * math.log2(k) / d + (down_bits + 0.5)) / 3
        for rule, label in [r for r in (("all", "all"), ("rank", "rank 5-8")) if r[0] in args.rules.split(",")]:
            for w in wraps:
                w.rule = rule
            r = score(model, corpus, ref, ntok)
            r.update({"d": d, "k": k, "rotate": rotate, "down_bits": down_bits, "bits": bits, "rule": rule,
                      "rel_err": sum(errs) / len(errs)})
            results[f"{name} {label}"] = r
            print(f"{name + ' ' + label:28s} {bits:.2f} b  err {r['rel_err']:.3f}  kl {r['kl']:.4f}  "
                  f"top1 {100 * r['top1']:.1f}%  ppl x{r['ppl_ratio']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            args.json.write_text(json.dumps({"model": args.model, "tokens": ntok, "variants": results},
                                            indent=1) + "\n")


if __name__ == "__main__":
    main()
