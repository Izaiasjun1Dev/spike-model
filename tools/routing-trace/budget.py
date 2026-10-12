"""What a given amount of RAM buys: expert-cache policy × storage format.

sim.py sweeps cache size as a fraction of all experts and finds LRU good at
20–30%. Most Macs can't spare 30 GB, so this looks at small budgets, where
one token's working set (top_k × layers experts) no longer fits and global
LRU thrashes. Policies compared, all per layer unless named global:

  lru (global)     sim.py's baseline
  lru              cache split evenly across layers, LRU inside each
  bip              per-layer LRU with bimodal insertion: a miss enters at the
                   cold end (evicted next) except 1 in 16, so a stream larger
                   than the cache keeps a stable resident subset
  lfu-decay        per-layer, keep the experts with the highest decayed use
                   count (half-life ~ 32 tokens)
  belady           per-layer offline optimum (upper bound)

Formats change both how many experts fit in the budget and how many bytes a
miss reads: q4 = 4.5 bits/weight (MLX affine g64), q3 = 3.5, q2 = 2.5.
Quality of q3/q2 is measured separately by quality.py.

Usage: uv run budget.py traces/qwen3-30b-a3b [--json out.json]
"""

import argparse
import json
import random
from collections import OrderedDict

import numpy as np

from sim import TARGET, Trace, split

BUDGETS_GB = [4, 6, 8, 12, 16, 24, 32]
FORMATS = {"q4": 4.5, "q3": 3.5, "q2": 2.5}
EXPERT_PARAMS = TARGET["expert_bytes"] * 8 / 4.5  # weights per 235B expert


def expert_bytes(bpw: float) -> int:
    return int(EXPERT_PARAMS * bpw / 8)


def per_layer_streams(tr: Trace, seq) -> list[list[list[int]]]:
    """[layer][token] -> picked expert ids."""
    r = tr.rows(seq)
    return [r[:, li].tolist() for li in range(tr.L)]


def run_lru(steps, cap, bip=False, rng=None):
    lru: OrderedDict = OrderedDict()
    hits = 0
    for picks in steps:
        for e in picks:
            if e in lru:
                lru.move_to_end(e)
                hits += 1
                continue
            if cap == 0:
                continue
            if len(lru) >= cap:
                lru.popitem(last=False)
            lru[e] = None
            if bip and rng.random() >= 1 / 16:
                lru.move_to_end(e, last=False)
    return hits


def run_lfu_decay(steps, cap, half_life=32.0):
    decay = 0.5 ** (1 / half_life)
    score: dict[int, float] = {}
    cache: set[int] = set()
    hits = 0
    for picks in steps:
        for e in score:
            score[e] *= decay
        for e in picks:
            score[e] = score.get(e, 0.0) + 1.0
        for e in picks:
            if e in cache:
                hits += 1
                continue
            if cap == 0:
                continue
            if len(cache) >= cap:
                victim = min(cache - set(picks), key=score.__getitem__, default=None)
                if victim is None or score[victim] >= score[e]:
                    continue  # bypass: not worth evicting anything
                cache.remove(victim)
            cache.add(e)
    return hits


def run_belady(steps, cap):
    flat = [e for picks in steps for e in picks]
    n = len(flat)
    nxt = [0] * n
    last: dict[int, int] = {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(flat[i], n + 1)
        last[flat[i]] = i
    cache: dict[int, int] = {}
    hits = 0
    for i, e in enumerate(flat):
        if e in cache:
            hits += 1
            cache[e] = nxt[i]
            continue
        if cap == 0:
            continue
        if len(cache) >= cap:
            victim = max(cache, key=cache.__getitem__)
            if cache[victim] <= nxt[i]:
                continue  # optimal bypass
            del cache[victim]
        cache[e] = nxt[i]
    return hits


def run_global_lru(tr: Trace, seq, cap):
    r = tr.rows(seq).astype(np.int64)
    stream = (r + (np.arange(tr.L) * tr.E)[None, :, None]).reshape(-1).tolist()
    return run_lru([stream], cap)


def hit_rate(tr: Trace, test, policy: str, frac: float) -> float:
    per_layer_cap = int(frac * tr.E)
    hits = total = 0
    rng = random.Random(0)
    for s in test:
        total += tr.rows(s).size
        if policy == "lru (global)":
            hits += run_global_lru(tr, s, int(frac * tr.E * tr.L))
            continue
        for steps in per_layer_streams(tr, s):
            if policy == "lru":
                hits += run_lru(steps, per_layer_cap)
            elif policy == "bip":
                hits += run_lru(steps, per_layer_cap, bip=True, rng=rng)
            elif policy == "lfu-decay":
                hits += run_lfu_decay(steps, per_layer_cap)
            elif policy == "belady":
                hits += run_belady(steps, per_layer_cap)
    return hits / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json")
    args = ap.parse_args()
    tr = Trace(args.trace)
    _, test = split(tr.seqs)
    T = TARGET
    policies = ["lru (global)", "lru", "bip", "lfu-decay", "belady"]
    cache: dict[tuple[str, int], float] = {}

    def hit(policy, frac):
        key = (policy, int(frac * tr.E))  # per-layer slots decide the result
        if key not in cache:
            cache[key] = hit_rate(tr, test, policy, frac)
        return cache[key]

    rows = []
    print(f"target {T['name']}: {T['layers']} layers × {T['experts']} experts, "
          f"SSD {T['ssd_gbps']} GB/s; tok/s is the SSD-only bound")
    for fmt, bpw in FORMATS.items():
        eb = expert_bytes(bpw)
        all_gb = eb * T["layers"] * T["experts"] / 1e9
        print(f"\n== {fmt} ({bpw} bpw): expert {eb / 2**20:.1f} MiB, all experts {all_gb:.0f} GB")
        print(f"{'RAM GB':>6} {'slots/layer':>11} " + " ".join(f"{p:>13}" for p in policies)
              + f" {'tok/s best':>10}")
        for gb in BUDGETS_GB:
            frac = min(1.0, gb * 1e9 / (eb * T["layers"] * T["experts"]))
            hits = {p: hit(p, frac) for p in policies}
            best = max((h, p) for p, h in hits.items() if p != "belady")
            miss_gb = (1 - best[0]) * T["top_k"] * T["layers"] * eb / 1e9
            tok_s = T["ssd_gbps"] / miss_gb if miss_gb else float("inf")
            print(f"{gb:>6} {int(frac * tr.E):>11} "
                  + " ".join(f"{100 * hits[p]:>12.1f}%" for p in policies)
                  + f" {tok_s:>10.1f}")
            rows.append({"format": fmt, "bpw": bpw, "ram_gb": gb, "fraction": frac,
                         "slots_per_layer": int(frac * tr.E), "hit": hits,
                         "best_policy": best[1], "miss_gb_per_token": miss_gb,
                         "target_tok_s": tok_s})
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"trace": args.trace, "target": T, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
