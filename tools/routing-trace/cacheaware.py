"""Cache-aware routing: let the router prefer experts already in RAM.

When two experts score about the same, picking the resident one saves an
SSD read. Here the router's choice is biased toward each layer's cached
experts (in the spirit of "Mixture of Cache-Conditional Experts", Skliar et
al. 2024), with no training. The mixing weights stay the router's own
probabilities, renormalised over the chosen experts; only the choice moves.

Routing changes what gets cached and the cache changes routing, so this has
to run in a closed loop: the text is fed one token at a time with a KV cache
and every MoE layer keeps its own expert cache (per-layer LFU with decay and
bypass, as in budget.py, all in MLX arrays so nothing syncs per layer). The
budget is in GB of the 235B target (94 layers, 10.6 MB q4 experts), the same
per-layer slot count expert-bench replay uses for --cache-gb.

Rules (keep = top stock picks that never move, bias = added to the
log-probability of resident experts when choosing the rest):

  ref           stock routing, run through the same loop (the KL floor)
  bias <d>      keep 0, bias d
  swap <m>      keep k-m, bias 100: the m lowest picks go to the best
                resident experts outside the kept ones, if any

--tiered also serves the bottom half of each token's picks (by weight) from a
2-bit copy, as in quality.py's "q2 rank 5-8", to measure both together.

Each run writes the chosen experts as a trace (same format as capture.py)
under traces/cacheaware/, so `tiered.py` and `expert-bench replay` measure
the speed of exactly the routing whose quality is reported here.

Usage: uv run cacheaware.py [--budgets 8,16] [--rules ref,bias1,swap2] [--tiered]
"""

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from quality import build_corpus, logprobs, low_bit_copy
from prompts import PROMPTS

TARGET_LAYERS = 94
TARGET_EXPERT_BYTES = 10_616_832
HALF_LIFE = 32.0


def parse_rule(rule: str, k: int) -> tuple[int, float]:
    """rule -> (keep, bias)"""
    if rule == "ref":
        return k, 0.0
    if rule.startswith("bias"):
        return 0, float(rule[4:])
    if rule.startswith("swap"):
        return k - int(rule[4:]), 100.0
    raise SystemExit(f"unknown rule {rule}")


class CacheAwareMoE(nn.Module):
    """Wraps a Qwen3 MoE block with an expert cache and cache-biased routing."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.lo = None  # 2-bit copy for the bottom half, with --tiered
        self.configure(0, 0, 0.0)

    def configure(self, cap: int, keep: int, bias: float):
        E = self.block.num_experts
        self.cap, self.keep, self.bias = cap, keep, bias
        self.decay = 0.5 ** (1 / HALF_LIFE)
        self.score = mx.zeros((E,))
        self.resident = mx.zeros((E,), dtype=mx.bool_)
        self.hits = mx.array(0.0)
        self.picks, self.weights = [], []  # numpy, one [tokens, k] array per sequence
        self.seq_picks, self.seq_weights = [], []  # MLX, per token of the current sequence

    def __call__(self, x):
        b = self.block
        k, E = b.top_k, b.num_experts
        logp = nn.log_softmax(b.gate(x).astype(mx.float32), axis=-1)  # [1, 1, E]
        res = self.resident
        order = mx.argsort(-logp, axis=-1)
        if self.keep >= k:
            inds = order[..., :k]
        else:
            # Keep the stock top `keep`, then choose the rest with the bias.
            kept = order[..., : self.keep]
            sel = logp + self.bias * res.astype(mx.float32)
            if self.keep:
                is_kept = mx.zeros(logp.shape, dtype=mx.bool_)
                is_kept = mx.put_along_axis(is_kept, kept, mx.array(True), axis=-1)
                sel = mx.where(is_kept, -mx.inf, sel)
            rest = mx.argsort(-sel, axis=-1)[..., : k - self.keep]
            inds = mx.concatenate([kept, rest], axis=-1)
        p = mx.exp(mx.take_along_axis(logp, inds, axis=-1))
        w = p / p.sum(axis=-1, keepdims=True) if b.norm_topk_prob else p
        # sort the chosen experts by weight, as the trace format expects
        o = mx.argsort(-w, axis=-1)
        inds, w = mx.take_along_axis(inds, o, axis=-1), mx.take_along_axis(w, o, axis=-1)
        self.seq_picks.append(inds.reshape(k).astype(mx.uint8))
        self.seq_weights.append(w.reshape(k).astype(mx.float16))

        if self.lo is not None:
            low = (mx.arange(k) >= k // 2)[None, None, :, None]
            y = mx.where(low, self.lo(x, inds), b.switch_mlp(x, inds))
        else:
            y = b.switch_mlp(x, inds)
        out = (y * w[..., None].astype(y.dtype)).sum(axis=-2)

        # cache update: hits, decayed counts, then the best `cap` of
        # (resident ∪ chosen) stay; a new expert only enters by beating the
        # weakest resident one (bypass otherwise)
        chosen = mx.zeros((E,), dtype=mx.bool_)
        chosen = mx.put_along_axis(chosen, inds.reshape(k), mx.array(True), axis=0)
        self.hits = self.hits + (chosen & res).sum()
        self.score = self.score * self.decay + chosen
        if self.cap:
            cand = mx.where(res | chosen, self.score, -mx.inf)
            top = mx.argsort(-cand)[: self.cap]
            keep = mx.zeros((E,), dtype=mx.bool_)
            keep = mx.put_along_axis(keep, top, mx.array(True), axis=0)
            self.resident = keep & mx.isfinite(cand)
        return out.astype(x.dtype)


def run_sequence(model, wraps, toks):
    """Feed toks one at a time; returns log-probs [n-1, V] of each next token."""
    cache = make_prompt_cache(model)
    out = []
    for t in toks[:-1]:
        logits = model(mx.array([[t]]), cache=cache)[0, -1].astype(mx.float32)
        lp = logits - mx.logsumexp(logits)
        mx.eval(lp, [(w.score, w.resident, w.hits) for w in wraps])
        out.append(lp)
    # Move the sequence's picks to numpy: each per-token array pins its own
    # Metal buffers, and the live-buffer limit (~499k) is hit after ~5k tokens.
    for w in wraps:
        w.picks.append(np.array(mx.stack(w.seq_picks)))
        w.weights.append(np.array(mx.stack(w.seq_weights)))
        w.seq_picks, w.seq_weights = [], []
    return mx.stack(out)


def write_trace(base: Path, model_id, wraps, seqs_meta, k):
    picks = np.stack([np.concatenate(w.picks) for w in wraps], axis=1)
    weights = np.stack([np.concatenate(w.weights) for w in wraps], axis=1)
    base.parent.mkdir(parents=True, exist_ok=True)
    Path(str(base) + ".bin").write_bytes(picks.tobytes())
    Path(str(base) + ".scores.bin").write_bytes(weights.tobytes())
    meta = {"model": model_id, "num_layers": len(wraps), "num_experts": wraps[0].block.num_experts,
            "top_k": k, "dtype": "uint8", "layout": "token,layer,k (k sorted by descending router score)",
            "tokens": int(picks.shape[0]), "sequences": seqs_meta}
    Path(str(base) + ".json").write_text(json.dumps(meta, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--gen-tokens", type=int, default=160)
    ap.add_argument("--seq-len", type=int, default=768)
    ap.add_argument("--budgets", default="8,16", help="GB of experts on the 235B")
    ap.add_argument("--rules", default="ref,bias0.5,bias1,bias2,swap1,swap2,swap4")
    ap.add_argument("--tiered", action="store_true")
    ap.add_argument("--limit", type=int, help="first n corpus sequences (smoke test)")
    ap.add_argument("--json", type=Path, default=Path("traces/cacheaware.json"))
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    t0 = time.time()
    corpus = build_corpus(model, tokenizer, args.gen_tokens, args.seq_len)[: args.limit]
    ntok = sum(len(t) - 1 for _, t in corpus)
    print(f"corpus: {len(corpus)} sequences, {ntok} scored tokens ({time.time() - t0:.0f}s)")
    cats = {p["id"]: p["category"] for p in PROMPTS}

    ref = []  # stock model, one parallel forward per sequence
    for _, toks in corpus:
        lp = logprobs(model, toks)
        tgt = mx.array(toks[1:])
        ref.append((lp.astype(mx.float16), mx.argmax(lp, axis=-1),
                    -mx.take_along_axis(lp, tgt[:, None], axis=-1).sum()))
        mx.eval(ref[-1])

    wraps = []
    for layer in model.layers:
        w = CacheAwareMoE(layer.mlp)
        layer.mlp = w
        wraps.append(w)
    if args.tiered:
        for w in wraps:
            w.lo = low_bit_copy(w.block.switch_mlp, 2)
    k = wraps[0].block.top_k

    results = json.loads(args.json.read_text()) if args.json.exists() else {}
    budgets = [float(g) for g in args.budgets.split(",")]
    n_runs, run_i = len(budgets) * len(args.rules.split(",")), 0
    for gb in budgets:
        cap = int(gb * 1e9 / (TARGET_LAYERS * TARGET_EXPERT_BYTES))
        for rule in args.rules.split(","):
            name = f"{gb:g}GB {rule}" + (" tiered" if args.tiered else "")
            run_i += 1
            keep, bias = parse_rule(rule, k)
            t0 = time.time()
            for w in wraps:
                w.configure(cap, keep, bias)
            kl = agree = dnll = 0.0
            seqs_meta, start = [], 0
            for si, ((sid, toks), (rlp, rtop, rnll)) in enumerate(zip(corpus, ref)):
                for w in wraps:  # every sequence starts with a cold cache
                    w.score = mx.zeros_like(w.score)
                    w.resident = mx.zeros_like(w.resident)
                lp = run_sequence(model, wraps, toks)
                rlp32 = rlp.astype(mx.float32)
                kl += (mx.exp(rlp32) * (rlp32 - lp)).sum().item()
                agree += (mx.argmax(lp, axis=-1) == rtop).sum().item()
                tgt = mx.array(toks[1:])
                dnll += (-mx.take_along_axis(lp, tgt[:, None], axis=-1).sum() - rnll).item()
                n = len(toks) - 1
                seqs_meta.append({"id": sid, "category": cats.get(sid, "human"),
                                  "start": start, "decode_tokens": n})
                start += n
                print(f"  [{run_i}/{n_runs}] {name}: seq {si + 1}/{len(corpus)} {sid}, {start}/{ntok} tokens, "
                      f"{time.time() - t0:.0f}s", flush=True)
            hits = sum(w.hits.item() for w in wraps) / (start * len(wraps) * k)
            write_trace(Path("traces/cacheaware") / name.replace(" ", "-"), args.model, wraps, seqs_meta, k)
            results[name] = {"budget_gb": gb, "slots_per_layer": cap, "rule": rule, "keep": keep,
                             "bias": bias, "tiered": args.tiered, "hit": hits, "kl": kl / ntok,
                             "top1": agree / ntok, "dnll": dnll / ntok, "ppl_ratio": math.exp(dnll / ntok)}
            r = results[name]
            print(f"{name:24s} hit {100 * hits:5.1f}%  kl {r['kl']:.4f}  top1 {100 * r['top1']:.1f}%  "
                  f"ppl x{r['ppl_ratio']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            args.json.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
