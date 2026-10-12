"""Two-resolution expert cache, precision chosen by the router.

quality.py found that the four lowest-weight picks of each token can run at
2 bits for a perplexity cost of x1.034, less than 3 bits everywhere (x1.076).
This simulates the cache that exploits it, on the RAM budgets of budget.py:

  high pick (top half by router weight)  needs q4: hit only on a resident q4,
                                         otherwise read q4 from SSD
  low pick  (bottom half)                any resident copy serves it,
                                         otherwise read q2 (~0.56 the bytes)

The disk holds both copies of every expert. The RAM budget is in bytes, so
q2 entries take ~0.56 of a q4 slot. Replacement is per-layer LFU with decay
and bypass, scored per byte: a q4 entry is worth every recent use of the
expert, a q2 entry only its recent low picks (it can't serve a high one).
A newcomer evicts only entries worth less per byte than itself.

"+ skip 8" also drops the lowest-weight pick when it misses (no read).

Compared against single-format caches with the same policy (q4 and q3
everywhere). tok/s is the SSD-only bound on the 235B target.

Usage: uv run tiered.py traces/qwen3-30b-a3b [--json out.json]
"""

import argparse
import json

import numpy as np

from budget import BUDGETS_GB, expert_bytes
from sim import TARGET, Trace, split

BPW = {"q4": 4.5, "q3": 3.5, "q2": 2.5}
SIZE = {f: b / BPW["q4"] for f, b in BPW.items()}  # in q4-expert units
# Perplexity vs the 4-bit model, from quality.py (traces/quality.json).
# "+ skip 8" is the upper bound (pick 8 always dropped); skipping it only on
# a miss costs less.
PPL = {"q4": 1.0, "q3": 1.076, "q4/q2 by rank": 1.034, "q4/q2 + skip 8": None}
HALF_LIFE = 32.0


class LayerCache:
    def __init__(self, cap: float, decay: float):
        self.cap, self.decay = cap, decay
        self.entries: dict[int, str] = {}  # expert -> resident format
        self.used = 0.0
        self.hi: dict[int, tuple[float, int]] = {}  # decayed count, as of step
        self.lo: dict[int, tuple[float, int]] = {}

    def _count(self, table, e, t):
        s, t0 = table.get(e, (0.0, t))
        return s * self.decay ** (t - t0)

    def bump(self, table, e, t):
        table[e] = (self._count(table, e, t) + 1.0, t)

    def density(self, e, fmt, t):
        v = self._count(self.lo, e, t)
        if fmt != "q2":
            v += self._count(self.hi, e, t)
        return v / SIZE[fmt]

    def admit(self, e, fmt, t, protect):
        """Insert e at fmt, evicting cheaper entries; bypass if none qualify."""
        old = self.entries.get(e)
        free = self.cap - self.used + (SIZE[old] if old else 0.0)
        need = SIZE[fmt]
        if need > self.cap:
            return
        d = self.density(e, fmt, t)
        victims = []
        if free < need:
            pool = sorted((self.density(v, f, t), v) for v, f in self.entries.items()
                          if v != e and v not in protect)
            for dv, v in pool:
                if dv >= d:
                    return
                victims.append(v)
                free += SIZE[self.entries[v]]
                if free >= need:
                    break
            if free < need:
                return
        for v in victims:
            self.used -= SIZE[self.entries.pop(v)]
        if old:
            self.used -= SIZE[old]
        self.entries[e] = fmt
        self.used += need


def run_layer(picks: np.ndarray, weights: np.ndarray, cap: float, scheme: str):
    """picks, weights: [tokens, K]. Returns (hits, bytes read in q4 units)."""
    c = LayerCache(cap, 0.5 ** (1 / HALF_LIFE))
    K = picks.shape[1]
    skip_last = scheme == "q4/q2 + skip 8"
    if skip_last:
        scheme = "q4/q2 by rank"
    hits = 0
    read = 0.0
    for t in range(picks.shape[0]):
        order = np.argsort(-weights[t])
        high = set(picks[t, order[: K // 2]].tolist())
        last = int(picks[t, order[-1]])
        row = picks[t].tolist()
        for e in row:
            c.bump(c.hi if e in high else c.lo, e, t)
        protect = set(row)
        for e in row:
            if scheme == "q4/q2 by rank":
                need = "q4" if e in high else "q2"
            else:
                need = scheme
            have = c.entries.get(e)
            if have == need or have == "q4" or (have and need == "q2"):
                hits += 1
                continue
            if skip_last and e == last:
                continue  # lowest-weight pick missed: skip it, no read
            read += SIZE[need]
            c.admit(e, need, t, protect)
    return hits, read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json")
    args = ap.parse_args()
    tr = Trace(args.trace)
    _, test = split(tr.seqs)
    T = TARGET
    q4_bytes = expert_bytes(BPW["q4"])
    schemes = ["q4", "q3", "q4/q2 by rank", "q4/q2 + skip 8"]
    print(f"target {T['name']}: {T['layers']} layers, SSD {T['ssd_gbps']} GB/s, "
          f"tok/s is the SSD-only bound; ppl vs 4-bit model from quality.py")
    print(f"{'RAM GB':>6} " + " ".join(f"{s + ' hit':>17} {'GB/tok':>6} {'tok/s':>5}" for s in schemes))
    rows = []
    for gb in BUDGETS_GB:
        cap = gb * 1e9 / (T["layers"] * q4_bytes)  # q4 slots per layer
        line = f"{gb:>6} "
        for s in schemes:
            hits = total = 0
            read = 0.0
            for seq in test:
                p = tr.rows(seq)
                w = tr.weights(seq).reshape(p.shape)
                total += p.size
                for li in range(tr.L):
                    h, r = run_layer(p[:, li], w[:, li], cap, s)
                    hits += h
                    read += r
            tokens = sum(tr.rows(seq).shape[0] for seq in test)
            gb_tok = read / tokens / tr.L * T["layers"] * q4_bytes / 1e9
            tok_s = T["ssd_gbps"] / gb_tok
            line += f"{100 * hits / total:>16.1f}% {gb_tok:>6.2f} {tok_s:>5.1f} "
            rows.append({"ram_gb": gb, "scheme": s, "hit": hits / total,
                         "gb_per_token": gb_tok, "target_tok_s": tok_s, "ppl": PPL[s]})
        print(line, flush=True)
    print("ppl x: " + ", ".join(f"{s} {PPL[s]}" for s in schemes if PPL[s]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"trace": args.trace, "target": T, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
