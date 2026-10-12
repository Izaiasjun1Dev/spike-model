"""Can a small trained predictor see experts 2-3 layers ahead?

Pre-gating (layer l+1's router applied to layer l's router input) guesses
85% of the next layer's experts, but one layer of warning is shorter than an
expert read, so the prefetch barely helps. This trains, for each distance
d and layer l, a linear map from layer l's router input to layer l+d's
experts (in the spirit of Pre-gated MoE, Hwang et al. 2024) and compares it
with pre-gating at the same distance.

  pregate   router of l+d on h_l, no training
  linear    same shape (D x E plus bias): that router plus a trained
            correction, distilled toward l+d's full router distribution
            (softmax of router l+d on h_{l+d}). Cross-entropy against the
            top-k set instead makes recall *worse*, even on the train rows:
            it is dominated by the picks the router can't foresee and drags
            them into everyone's top-k.

Data comes from `capture.py --hidden`. Train on every row (prefill and
decode) of the train sequences of sim.py's split, pick the epoch on two
held-out train sequences, report on the decode rows of the test sequences:
recall@m is the share of the real top-k found among the m best predictions
(m > k means prefetching extra experts).

Writes <trace>.pred<d>.bin, uint8 [tokens, layers, 16]: the 16 best
predictions for each layer from the trained predictor, best first (255 for
layers < d), for every decode token of the trace, for expert-bench replay.

Usage: uv run predictor.py traces/qwen3-30b-a3b-h [--dists 1,2,3] [--json out.json]
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load

from sim import NO_PRED, split

WIDTHS = [8, 12, 16]
OUT_WIDTH = 16


def load_data(base: str):
    meta = json.loads(Path(base + ".json").read_text())
    L, k, D = meta["num_layers"], meta["top_k"], meta["hidden_size"]
    hidden = np.fromfile(base + ".hidden.bin", np.float16).reshape(-1, L, D)
    picks = np.fromfile(base + ".hpicks.bin", np.uint8).reshape(-1, L, k)
    return meta, hidden, picks


def rows_of(seqs, decode_only: bool) -> np.ndarray:
    out = []
    for s in seqs:
        a = s["hidden_start"] + (s["hidden_prefill"] if decode_only else 0)
        out.append(np.arange(a, s["hidden_start"] + s["hidden_prefill"] + s["hidden_decode"]))
    return np.concatenate(out)


def router_weights(model_id: str) -> list[mx.array]:
    """Each MoE layer's router as a float32 [E, D] matrix."""
    model, _ = load(model_id, lazy=True)
    ws = []
    for layer in model.layers:
        g = layer.mlp.gate
        if isinstance(g, nn.QuantizedLinear):
            w = mx.dequantize(g.weight, g.scales, g.biases, group_size=g.group_size, bits=g.bits)
        else:
            w = g.weight
        ws.append(w.astype(mx.float32))
    mx.eval(ws)
    return ws


class Linear(nn.Module):
    """One D x E map per source layer, applied to h[:, l] for l < L - d:
    the router of l+d (frozen) plus a trained correction, which weight decay
    pulls back toward the router."""

    def __init__(self, w0: mx.array):  # w0 [L-d, D, E]
        super().__init__()
        self.w0 = w0
        self.dw = mx.zeros_like(w0)
        self.b = mx.zeros((w0.shape[0], w0.shape[2]))
        self.freeze(keys=["w0"])

    def __call__(self, h):  # h [n, L-d, D] -> [n, L-d, E]
        return mx.einsum("nld,lde->nle", h.astype(mx.float32), self.w0 + self.dw) + self.b


def recall(logits: np.ndarray, picks: np.ndarray, m: int) -> np.ndarray:
    """Per-layer share of the true picks among the m best logits."""
    top = np.argpartition(-logits, m - 1, axis=-1)[..., :m]
    hit = (top[..., :, None] == picks[..., None, :].astype(np.int64)).any(axis=-2)
    return hit.mean(axis=(0, 2))


def predict(f, hidden: np.ndarray, rows: np.ndarray, d: int, batch: int = 512) -> np.ndarray:
    L = hidden.shape[1]
    out = []
    for i in range(0, len(rows), batch):
        h = mx.array(hidden[rows[i: i + batch], : L - d])
        out.append(np.array(f(h)))
    return np.concatenate(out)


def train(w0, hidden, picks, tr_rows, va_rows, d, epochs, lr, batch, wd):
    L = hidden.shape[1]
    model = Linear(w0)
    opt = optim.AdamW(learning_rate=lr, weight_decay=wd)

    def loss_fn(model, h, target):
        return -(target * nn.log_softmax(model(h), axis=-1)).sum(axis=-1).mean()

    step = nn.value_and_grad(model, loss_fn)
    va_y = picks[va_rows, d:]
    best = recall(predict(model, hidden, va_rows, d), va_y, picks.shape[-1]).mean()
    best_params, history = None, [{"epoch": 0, "val_recall8": float(best)}]
    print(f"  d={d} epoch 0: val recall@8 {best:.3f}")
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        order = rng.permutation(tr_rows)
        t0, tot, n = time.perf_counter(), 0.0, 0
        for i in range(0, len(order), batch):
            r = np.sort(order[i: i + batch])
            h = mx.array(hidden[r]).astype(mx.float32)
            # teacher: the real router of l+d on its own input
            target = mx.softmax(mx.einsum("nld,lde->nle", h[:, d:], w0), axis=-1)
            loss, grads = step(model, h[:, : L - d], target)
            opt.update(model, grads)
            mx.eval(model.trainable_parameters(), opt.state, loss)
            tot, n = tot + loss.item(), n + 1
        va = recall(predict(model, hidden, va_rows, d), va_y, picks.shape[-1]).mean()
        history.append({"epoch": ep + 1, "loss": tot / n, "val_recall8": float(va)})
        print(f"  d={d} epoch {ep + 1}: loss {tot / n:.4f}  val recall@8 {va:.3f}  ({time.perf_counter() - t0:.0f}s)")
        if va > best:
            best = va
            best_params = {k: mx.array(v) for k, v in nn.utils.tree_flatten(model.trainable_parameters())}
    if best_params is not None:
        model.update(nn.utils.tree_unflatten(list(best_params.items())))
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--model", default=None, help="default: the model named in the trace")
    ap.add_argument("--dists", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--json")
    args = ap.parse_args()

    meta, hidden, picks = load_data(args.trace)
    L = meta["num_layers"]
    seqs = meta["sequences"]
    train_seqs, test_seqs = split(seqs)
    val_seqs, fit_seqs = train_seqs[:2], train_seqs[2:]
    tr_rows = rows_of(fit_seqs, decode_only=False)
    va_rows = rows_of(val_seqs, decode_only=True)
    te_rows = rows_of(test_seqs, decode_only=True)
    print(f"rows: fit {len(tr_rows)} (prefill+decode), val {len(va_rows)}, test {len(te_rows)} (decode)")

    routers = router_weights(args.model or meta["model"])
    all_dec = rows_of(seqs, decode_only=True)
    results = {"args": vars(args), "dists": {}}
    for d in [int(x) for x in args.dists.split(",")]:
        # pre-gating baseline: router of layer l+d on h_l
        w0 = mx.stack([routers[l + d].T for l in range(L - d)])
        pre = Linear(w0)
        te_y = picks[te_rows, d:]
        base = {m: recall(predict(pre, hidden, te_rows, d), te_y, m) for m in WIDTHS}
        model, history = train(w0, hidden, picks, tr_rows, va_rows, d,
                               args.epochs, args.lr, args.batch, args.wd)
        logits = predict(model, hidden, te_rows, d)
        lin = {m: recall(logits, te_y, m) for m in WIDTHS}
        res = {
            "pregate": {f"recall@{m}": float(v.mean()) for m, v in base.items()},
            "linear": {f"recall@{m}": float(v.mean()) for m, v in lin.items()},
            "per_layer_recall@8": {"pregate": base[8].round(4).tolist(), "linear": lin[8].round(4).tolist()},
            "history": history,
        }
        results["dists"][str(d)] = res
        print(f"d={d}  " + "  ".join(
            f"recall@{m}: pregate {res['pregate'][f'recall@{m}']:.3f} linear {res['linear'][f'recall@{m}']:.3f}"
            for m in WIDTHS))

        # predictions for replay, every decode token of the trace
        logits = predict(model, hidden, all_dec, d)
        top = np.argsort(-logits, axis=-1)[..., :OUT_WIDTH].astype(np.uint8)
        out = np.full((len(all_dec), L, OUT_WIDTH), NO_PRED, np.uint8)
        out[:, d:] = top
        Path(f"{args.trace}.pred{d}.bin").write_bytes(out.tobytes())

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
