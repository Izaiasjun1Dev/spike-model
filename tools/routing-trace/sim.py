"""Expert-cache and prefetch simulator driven by real routing traces.

Sequences are split into train / test: anything learned (expert popularity,
co-activation clusters, cross-layer tables) comes from train, every number
reported is measured on test. Categories with a single prompt go to test,
so part of the test set is a domain the "clusterer" never saw.

Analyses:
  1. predictors  how well each method guesses layer l's experts ahead of time
  2. cache       hit rate vs cache size, per policy, with and without
                 pre-gated prefetch and prefill warm-up
  3. layout      would storing co-activated experts next to each other on
                 disk (clusters of g) save reads, or waste bandwidth?
  4. per-seq     hit rate per test prompt, to see which tasks cache well

Cache size is a fraction of all experts (layers × experts) so it carries over
to other models; the 235B projection uses its expert size and layer count.

Usage: uv run sim.py traces/qwen3-30b-a3b [--json out.json]
"""

import argparse
import heapq
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

NO_PRED = 255
FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# Projection target: Qwen3-235B-A22B at 4.5 bits, measured SSD bandwidth.
TARGET = {"name": "qwen3-235b-a22b", "layers": 94, "experts": 128, "top_k": 8,
          "expert_bytes": 10_616_832, "ssd_gbps": 14.0}


# ---------------------------------------------------------------- loading

class Trace:
    def __init__(self, base: str):
        self.meta = json.loads(Path(base + ".json").read_text())
        m = self.meta
        self.L, self.E, self.K = m["num_layers"], m["num_experts"], m["top_k"]
        shape = (-1, self.L, self.K)
        self.picks = np.fromfile(base + ".bin", np.uint8).reshape(shape)
        pred_path = Path(base + ".pred.bin")
        self.preds = np.fromfile(pred_path, np.uint8).reshape(shape) if pred_path.exists() else None
        sc = Path(base + ".scores.bin")
        self.scores = (np.fromfile(sc, np.float16).reshape(shape).astype(np.float32)
                       if sc.exists() else None)
        pf = Path(base + ".prefill.npy")
        self.prefill = np.load(pf) if pf.exists() else None
        self.seqs = m["sequences"]
        for i, s in enumerate(self.seqs):
            s["index"] = i

    def rows(self, seq) -> np.ndarray:
        return self.picks[seq["start"]: seq["start"] + seq["decode_tokens"]]

    def weights(self, seq) -> np.ndarray:
        """Router weight per access, flat, aligned with `accesses`."""
        if self.scores is None:
            return np.full(seq["decode_tokens"] * self.L * self.K, 1.0 / self.K, np.float32)
        return self.scores[seq["start"]: seq["start"] + seq["decode_tokens"]].reshape(-1)

    def pred_rows(self, seq) -> np.ndarray:
        return self.preds[seq["start"]: seq["start"] + seq["decode_tokens"]]


def split(seqs):
    by_cat = defaultdict(list)
    for s in seqs:
        by_cat[s["category"]].append(s)
    train, test = [], []
    for group in by_cat.values():
        if len(group) == 1:
            test += group
            continue
        for i, s in enumerate(group):
            (train if i % 2 == 0 else test).append(s)
    return train, test


# ------------------------------------------------------------- learned tables

def popularity(tr: Trace, train) -> np.ndarray:
    """Activation counts [L, E] over the train decode tokens."""
    counts = np.zeros((tr.L, tr.E), np.int64)
    for s in train:
        r = tr.rows(s)
        for li in range(tr.L):
            np.add.at(counts[li], r[:, li].reshape(-1).astype(np.int64), 1)
    return counts


def transitions(tr: Trace, train) -> np.ndarray:
    """Cross-layer co-occurrence [L-1, E, E]: expert a at layer l with b at l+1."""
    t = np.zeros((tr.L - 1, tr.E, tr.E), np.int64)
    for s in train:
        r = tr.rows(s).astype(np.int64)
        for li in range(tr.L - 1):
            a, b = r[:, li], r[:, li + 1]
            np.add.at(t[li], (np.repeat(a, tr.K, 1).ravel(), np.tile(b, (1, tr.K)).ravel()), 1)
    return t


def coactivation(tr: Trace, train) -> np.ndarray:
    """Within-layer co-occurrence [L, E, E]."""
    c = np.zeros((tr.L, tr.E, tr.E), np.int64)
    for s in train:
        r = tr.rows(s).astype(np.int64)
        for li in range(tr.L):
            a = r[:, li]
            np.add.at(c[li], (np.repeat(a, tr.K, 1).ravel(), np.tile(a, (1, tr.K)).ravel()), 1)
    idx = np.arange(tr.E)
    c[:, idx, idx] = 0
    return c


# ------------------------------------------------------------ 1. predictors

def recall(pred_sets, actual) -> float:
    hit = sum(len(p & set(a)) for p, a in zip(pred_sets, actual))
    return hit / (len(actual) * len(actual[0]))


def predictors(tr: Trace, test, pop, trans):
    """Recall@top_k of each predictor for layers 1..L-1 on test tokens."""
    K = tr.K
    res = defaultdict(lambda: [0, 0])

    def add(name, pred, act):
        res[name][0] += len(set(pred) & set(act))
        res[name][1] += K

    hot = [set(np.argsort(-pop[li])[:K]) for li in range(tr.L)]
    for s in test:
        r = tr.rows(s)
        p = tr.pred_rows(s) if tr.preds is not None else None
        for t in range(r.shape[0]):
            for li in range(1, tr.L):
                act = r[t, li]
                if p is not None:
                    add("pre-gate (router l on hidden l-1)", p[t, li], act)
                score = trans[li - 1][r[t, li - 1]].sum(0)
                add("cross-layer table (train)", np.argsort(-score)[:K], act)
                add("hot list (train popularity)", hot[li], act)
                if t > 0:
                    add("same layer, previous token", r[t - 1, li], act)
    return {k: v[0] / v[1] for k, v in res.items()}


# ---------------------------------------------------------------- 2. cache

def accesses(tr: Trace, seq):
    """Flat stream of global expert ids, in decode order."""
    r = tr.rows(seq).astype(np.int64)
    return (r + (np.arange(tr.L) * tr.E)[None, :, None]).reshape(-1)


def sim_lru(stream, cap, pinned=(), warm=(), prefetch=None, K=8, weights=None, skip_below=0.0):
    """LRU over `cap` slots; `pinned` ids never leave and use their own slots.

    prefetch: per (token, layer) step, ids to load before the step's accesses.
    weights / skip_below: a miss whose router weight is under `skip_below`
    is dropped (not loaded, its expert skipped), trading quality for reads.
    Returns (hits, demand misses, prefetch loads, weight on misses, weight skipped).
    """
    pinned = set(pinned)
    cap -= len(pinned)
    lru: OrderedDict = OrderedDict()
    for e in warm:
        if e in pinned or e in lru:
            continue
        if len(lru) >= cap:
            break
        lru[e] = None
    hits = misses = loads = 0
    miss_w = skip_w = 0.0

    def touch(e):
        if e in lru:
            lru.move_to_end(e)
            return True
        if cap > 0:
            lru[e] = None
            if len(lru) > cap:
                lru.popitem(last=False)
        return False

    for step in range(len(stream) // K):
        if prefetch is not None:
            for e in prefetch[step]:
                if e not in pinned and e not in lru:
                    loads += 1
                    touch(e)
        for j in range(step * K, (step + 1) * K):
            e = stream[j]
            if e in pinned or e in lru:
                touch(e)
                hits += 1
                continue
            w = weights[j] if weights is not None else 0.0
            miss_w += w
            if w < skip_below:
                skip_w += w
                continue
            touch(e)
            misses += 1
    return hits, misses, loads, miss_w, skip_w


def sim_belady(stream, cap):
    """Optimal offline replacement (upper bound on any policy's hit rate)."""
    n = len(stream)
    nxt = np.empty(n, np.int64)
    last = {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(stream[i], n + i)
        last[stream[i]] = i
    cache, heap, hits = {}, [], 0
    for i in range(n):
        e = stream[i]
        if e in cache:
            hits += 1
        elif len(cache) >= cap:
            while True:
                negnext, v = heapq.heappop(heap)
                if cache.get(v) == -negnext:
                    del cache[v]
                    break
        cache[e] = nxt[i]
        heapq.heappush(heap, (-nxt[i], e))
    return hits, n - hits, 0, 0.0, 0.0


def cache_study(tr: Trace, test, pop):
    total = tr.L * tr.E
    hot_global = list(np.argsort(-pop.reshape(-1)))  # train popularity, all layers
    rows = []
    for f in FRACTIONS:
        cap = int(round(f * total))
        agg = defaultdict(lambda: np.zeros(5, np.float64))
        for s in test:
            stream = accesses(tr, s).tolist()
            wts = tr.weights(s).tolist()
            pf = None
            if tr.preds is not None:
                p = tr.pred_rows(s).astype(np.int64)
                p = p + (np.arange(tr.L) * tr.E)[None, :, None]
                p[:, 0, :] = -1
                pf = [[e for e in step if e >= 0] for step in p.reshape(-1, tr.K).tolist()]
            warm = []
            if tr.prefill is not None:
                h = tr.prefill[s["index"]].reshape(-1)
                warm = [int(e) for e in np.argsort(-h) if h[e] > 0]
            runs = {
                "lru": sim_lru(stream, cap, K=tr.K, weights=wts),
                "lru, skip misses < 5% weight": sim_lru(
                    stream, cap, K=tr.K, weights=wts, skip_below=0.05),
                "lru, skip misses < 10% weight": sim_lru(
                    stream, cap, K=tr.K, weights=wts, skip_below=0.10),
                "pinned hot (train)": sim_lru(stream, cap, pinned=hot_global[:cap], K=tr.K),
                "hybrid: 50% pinned + 50% lru": sim_lru(stream, cap, pinned=hot_global[: cap // 2], K=tr.K),
                "lru + prefill warm-up": sim_lru(stream, cap, warm=warm, K=tr.K),
                "belady (optimal bound)": sim_belady(stream, cap),
            }
            if pf is not None:
                runs["lru + pre-gate prefetch"] = sim_lru(stream, cap, prefetch=pf, K=tr.K)
                runs["warm-up + pre-gate prefetch"] = sim_lru(
                    stream, cap, warm=warm, prefetch=pf, K=tr.K)
                runs["pre-gate prefetch, skip < 5%"] = sim_lru(
                    stream, cap, prefetch=pf, K=tr.K, weights=wts, skip_below=0.05)
            for k, v in runs.items():
                agg[k] += np.array(v)
        steps = sum(s["decode_tokens"] for s in test) * tr.L
        for k, (h, m, ld, mw, sw) in agg.items():
            rows.append({"fraction": f, "policy": k, "hit": h / (steps * tr.K),
                         "demand_miss_per_step": m / steps,
                         "loads_per_step": (m + ld) / steps,
                         "miss_weight": mw / steps, "skipped_weight": sw / steps})
    return rows


def project(row):
    """SSD-bound tok/s on the target model, assuming prefetch overlaps compute
    and every load (demand miss or prefetch) costs SSD bandwidth."""
    T = TARGET
    gb = row["loads_per_step"] * T["layers"] * T["expert_bytes"] / 1e9
    stall_gb = row["demand_miss_per_step"] * T["layers"] * T["expert_bytes"] / 1e9
    return gb, T["ssd_gbps"] / gb if gb else float("inf"), stall_gb


# ---------------------------------------------------------------- 3. layout

def greedy_clusters(co: np.ndarray, g: int) -> list[list[int]]:
    """Pairs up experts with the strongest co-activation into groups of g."""
    E = co.shape[0]
    free = set(range(E))
    groups = []
    order = np.dstack(np.unravel_index(np.argsort(-co, axis=None), co.shape))[0]
    for a, b in order:
        if len(free) < g:
            break
        if a in free and b in free and a != b:
            grp = [int(a), int(b)]
            free -= {a, b}
            while len(grp) < g:
                cand = max(free, key=lambda x: co[grp, x].sum())
                grp.append(cand)
                free.discard(cand)
            groups.append(grp)
    if free:
        groups.append(sorted(free))
    return groups


def layout_study(tr: Trace, test, co, g):
    """Bytes read per byte needed if whole clusters are read, vs random groups."""
    rng = np.random.default_rng(0)
    out = {}
    for name in ("co-activation clusters", "random groups"):
        need = read = 0
        for li in range(tr.L):
            if name == "random groups":
                perm = rng.permutation(tr.E)
                groups = [perm[i:i + g].tolist() for i in range(0, tr.E, g)]
            else:
                groups = greedy_clusters(co[li], g)
            gid = np.empty(tr.E, np.int64)
            for i, grp in enumerate(groups):
                gid[grp] = i
            sizes = np.array([len(x) for x in groups])
            for s in test:
                r = tr.rows(s)[:, li].astype(np.int64)
                for row in r:
                    touched = np.unique(gid[row])
                    read += sizes[touched].sum()
                    need += len(row)
        out[name] = read / need
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="trace base path (without extension)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    tr = Trace(args.trace)
    train, test = split(tr.seqs)
    ntok = lambda ss: sum(s["decode_tokens"] for s in ss)  # noqa: E731
    print(f"{tr.meta['model']}: {tr.L} layers × {tr.E} experts, top-{tr.K}")
    print(f"train {len(train)} seqs / {ntok(train)} tokens | test {len(test)} seqs / {ntok(test)} tokens")
    print("test:", ", ".join(s["id"] for s in test))

    pop = popularity(tr, train)
    trans = transitions(tr, train)
    co = coactivation(tr, train)

    print("\n## 1. Predictors (recall of the actual top-%d, layers 1..%d)" % (tr.K, tr.L - 1))
    preds = predictors(tr, test, pop, trans)
    for k, v in sorted(preds.items(), key=lambda kv: -kv[1]):
        print(f"  {v * 100:5.1f}%  {k}")
    print(f"  {tr.K / tr.E * 100:5.1f}%  random")

    print("\n## 2. Cache (hit rate on test; 235B projection at %.0f GB/s)" % TARGET["ssd_gbps"])
    cache = cache_study(tr, test, pop)
    total_gib = TARGET["layers"] * TARGET["experts"] * TARGET["expert_bytes"] / 2**30
    for f in FRACTIONS:
        print(f"\n  cache = {f * 100:.0f}% of experts (≈ {f * total_gib:.0f} GiB on the 235B)")
        for row in [r for r in cache if r["fraction"] == f]:
            gb, tps, stall = project(row)
            row.update(target_gb_per_token=gb, target_tok_s=tps, target_stall_gb=stall)
            lost = f"  lost {row['skipped_weight'] * 100:4.1f}% weight" if row["skipped_weight"] else ""
            print(f"    {row['hit'] * 100:5.1f}% hit  {gb:5.2f} GB/tok  {tps:5.2f} tok/s"
                  f"  (stall {stall:4.2f} GB)  {row['policy']}{lost}")

    print("\n## 3. Disk layout (bytes read per byte needed when reading whole groups)")
    layout = {}
    for g in (2, 4):
        layout[g] = layout_study(tr, test, co, g)
        for k, v in layout[g].items():
            print(f"  g={g}  {v:4.2f}×  {k}")

    print("\n## 4. Per test prompt (LRU, cache = 25%)")
    cap = int(0.25 * tr.L * tr.E)
    per_seq = []
    for s in test:
        h, m, _, _, _ = sim_lru(accesses(tr, s).tolist(), cap, K=tr.K)
        per_seq.append({"id": s["id"], "category": s["category"], "hit": h / (h + m)})
        print(f"  {h / (h + m) * 100:5.1f}%  {s['category']:<18} {s['id']}")

    if tr.scores is not None:
        w = np.concatenate([tr.weights(s).reshape(-1, tr.K) for s in test])
        print("\n## 5. Router weight by rank (test): " + " ".join(f"{x * 100:.1f}%" for x in w.mean(0)))
        lru = [r for r in cache if r["policy"] == "lru"]
        for r in lru:
            print(f"  cache {r['fraction'] * 100:3.0f}%: misses carry {r['miss_weight'] * 100:4.1f}% "
                  f"of router weight ({(1 - r['hit']) * 100:4.1f}% of picks)")

    if args.json:
        args.json.write_text(json.dumps({
            "trace": args.trace, "train": [s["id"] for s in train],
            "test": [s["id"] for s in test], "predictors": preds, "cache": cache,
            "layout": {str(k): v for k, v in layout.items()}, "per_seq": per_seq,
            "target": TARGET}, indent=2) + "\n")


if __name__ == "__main__":
    main()
