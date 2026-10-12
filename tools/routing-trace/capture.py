"""Capture real MoE routing traces from a Qwen3-MoE model running on MLX.

Wraps every layer's router (`mlp.gate`) so each forward pass records the
top-k experts it picked. Each wrapper also runs the *next* layer's router on
this layer's input ("pre-gating"): that is a prediction of the next layer's
experts available one layer early, which is what a prefetcher would use.

Decode steps (one position per call) go into the trace: that is the phase
where experts stream from the SSD token by token. Prefill calls (many
positions) are reduced to a per-sequence expert histogram, used to test
warming the cache from the prompt.

Output (for the simulator and `expert-bench`):
  <out>.bin       uint8 [tokens, layers, top_k], actual picks, each row
                  sorted by descending router score; sequences concatenated
  <out>.scores.bin  float16 [tokens, layers, top_k], router weight of each
                  pick (softmax renormalised over the top-k, as Qwen3 does)
  <out>.pred.bin  uint8 [tokens, layers, top_k], pre-gated prediction for
                  layer l made from layer l-1's input (layer 0 is 255)
  <out>.prefill.npy  uint32 [sequences, layers, experts], prefill counts
  <out>.json      model shape plus one entry per sequence

With --hidden, also the router input of every MoE layer (the training data
for predictor.py), for all decode tokens plus every --hidden-stride-th
prefill position:
  <out>.hidden.bin  float16 [rows, layers, hidden]
  <out>.hpicks.bin  uint8 [rows, layers, top_k], picks at those rows
  rows per sequence go in <out>.json as hidden_start / hidden_prefill /
  hidden_decode (prefill rows first, then the decode rows of the trace)

Usage: uv run capture.py --out traces/qwen3-30b-a3b [--max-tokens 1024]
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from prompts import PROMPTS

NO_PRED = 255


class RouterRecorder(nn.Module):
    """Drop-in replacement for a router Linear that logs its top-k picks."""

    def __init__(self, inner: nn.Module, top_k: int):
        super().__init__()
        self.inner = inner
        self.top_k = top_k
        self.next_inner = None  # next MoE layer's router, for pre-gating
        self.picks: list = []
        self.scores: list = []
        self.preds: list = []  # predictions for the next layer
        self.hidden: list | None = None  # router inputs, with --hidden

    def _top(self, logits):
        # Softmax is monotonic, so ranking logits gives the router's top-k.
        return mx.argsort(-logits, axis=-1)[..., : self.top_k].astype(mx.uint8)

    def __call__(self, x):
        logits = self.inner(x)
        if self.hidden is not None:
            self.hidden.append(x.astype(mx.float16))
        top = self._top(logits)
        self.picks.append(top)
        w = mx.take_along_axis(mx.softmax(logits, axis=-1, precise=True), top.astype(mx.int32), axis=-1)
        self.scores.append((w / w.sum(axis=-1, keepdims=True)).astype(mx.float16))
        if self.next_inner is not None:
            self.preds.append(self._top(self.next_inner(x)))
        return logits


def install_recorders(model) -> tuple[list[RouterRecorder], int, int]:
    recorders = []
    for layer in model.layers:
        moe = layer.mlp
        if not hasattr(moe, "gate") or not hasattr(moe, "top_k"):
            continue  # dense layer
        rec = RouterRecorder(moe.gate, moe.top_k)
        moe.gate = rec
        recorders.append((rec, moe))
    if not recorders:
        raise SystemExit("no MoE router found; is this a Qwen3-MoE model?")
    for (rec, _), (nxt, _) in zip(recorders, recorders[1:]):
        rec.next_inner = nxt.inner
    moe = recorders[0][1]
    return [r for r, _ in recorders], moe.num_experts, moe.top_k


def split_calls(calls: list) -> tuple[np.ndarray, np.ndarray]:
    """Recorded calls → (decode rows [n, k], prefill rows [m, k])."""
    mx.eval(calls)
    decode, prefill = [], []
    for top in calls:
        rows = np.array(top).reshape(-1, top.shape[-1])
        (decode if rows.shape[0] == 1 else prefill).append(rows)
    k = calls[0].shape[-1] if calls else 0
    dtype = np.array(calls[0]).dtype if calls else np.uint8
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros((0, k), dtype)  # noqa: E731
    return cat(decode), cat(prefill)


def drain_hidden(recs: list[RouterRecorder], stride: int):
    """Router inputs and picks of one sequence, before drain() clears the picks:
    (hidden [rows, L, D], picks [rows, L, k], prefill rows). Prefill rows are
    every stride-th prompt position; all decode rows follow."""
    hs, ps = [], []
    for rec in recs:
        mx.eval(rec.hidden)
        dec_h, pre_h, dec_p, pre_p = [], [], [], []
        for h, p in zip(rec.hidden, rec.picks):
            h = np.array(h).reshape(-1, h.shape[-1])
            p = np.array(p).reshape(-1, p.shape[-1])
            if h.shape[0] == 1:
                dec_h.append(h)
                dec_p.append(p)
            else:
                pre_h.append(h)
                pre_p.append(p)
        rec.hidden.clear()
        pre_h = np.concatenate(pre_h)[::stride] if pre_h else np.zeros((0, dec_h[0].shape[1]), np.float16)
        pre_p = np.concatenate(pre_p)[::stride] if pre_p else np.zeros((0, dec_p[0].shape[1]), np.uint8)
        hs.append(np.concatenate([pre_h] + dec_h))
        ps.append(np.concatenate([pre_p] + dec_p))
    return np.stack(hs, axis=1), np.stack(ps, axis=1), len(pre_h)


def drain(recs: list[RouterRecorder], num_experts: int):
    """One sequence: (picks [T, L, k], scores [T, L, k], preds [T, L, k], prefill hist [L, E])."""
    picks, scores, preds = [], [], []
    hist = np.zeros((len(recs), num_experts), np.uint32)
    top_k = recs[0].top_k
    for li, rec in enumerate(recs):
        dec, pre = split_calls(rec.picks)
        picks.append(dec)
        np.add.at(hist[li], pre.reshape(-1).astype(np.int64), 1)
        rec.picks.clear()
        scores.append(split_calls(rec.scores)[0])
        rec.scores.clear()
    for li, rec in enumerate(recs):
        if rec.next_inner is None:
            continue
        dec, _ = split_calls(rec.preds)
        preds.append(dec)  # prediction for layer li + 1
        rec.preds.clear()
    counts = {a.shape[0] for a in picks + scores + preds}
    if len(counts) != 1:
        raise SystemExit(f"layers saw different decode counts: {sorted(counts)}")
    t = picks[0].shape[0]
    none = np.full((t, top_k), NO_PRED, np.uint8)
    return (np.stack(picks, axis=1), np.stack(scores, axis=1),
            np.stack([none] + preds, axis=1), hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", help="comma-separated prompt ids (default: all)")
    ap.add_argument("--hidden", action="store_true", help="also save router inputs for predictor.py")
    ap.add_argument("--hidden-stride", type=int, default=4, help="keep every n-th prefill position")
    args = ap.parse_args()

    prompts = PROMPTS
    if args.only:
        wanted = set(args.only.split(","))
        prompts = [p for p in PROMPTS if p["id"] in wanted]

    mx.random.seed(args.seed)
    model, tokenizer = load(args.model)
    recs, num_experts, top_k = install_recorders(model)
    print(f"{len(recs)} MoE layers, {num_experts} experts, top-{top_k}")
    base = str(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.hidden:
        for rec in recs:
            rec.hidden = []
        hidden_f = open(base + ".hidden.bin", "wb")
        hpicks_f = open(base + ".hpicks.bin", "wb")
    hidden_rows = 0

    sampler = make_sampler(temp=args.temp, top_p=0.8)
    all_picks, all_scores, all_preds, hists, sequences, start = [], [], [], [], [], 0
    for i, p in enumerate(prompts):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": p["text"]}],
            add_generation_prompt=True,
            enable_thinking=p["thinking"],
        )
        t0 = time.perf_counter()
        n_out = 0
        for _ in stream_generate(
            model, tokenizer, prompt, max_tokens=args.max_tokens, sampler=sampler
        ):
            n_out += 1
        if args.hidden:
            h, hp, n_pre = drain_hidden(recs, args.hidden_stride)
            hidden_f.write(h.tobytes())
            hpicks_f.write(hp.tobytes())
        picks, scores, preds, hist = drain(recs, num_experts)
        secs = time.perf_counter() - t0
        all_picks.append(picks)
        all_scores.append(scores)
        all_preds.append(preds)
        hists.append(hist)
        sequences.append(
            {
                "id": p["id"],
                "category": p["category"],
                "thinking": p["thinking"],
                "prompt_tokens": len(prompt),
                "prefill_positions": int(hist[0].sum()) // top_k,
                "generated_tokens": n_out,
                "decode_tokens": int(picks.shape[0]),
                "start": start,
            }
        )
        if args.hidden:
            if h.shape[0] - n_pre != picks.shape[0]:
                raise SystemExit("hidden decode rows do not match the trace")
            sequences[-1].update(hidden_start=hidden_rows, hidden_prefill=n_pre,
                                 hidden_decode=h.shape[0] - n_pre)
            hidden_rows += h.shape[0]
        start += picks.shape[0]
        print(
            f"[{i + 1}/{len(prompts)}] {p['id']:<22} prompt {len(prompt):>5} "
            f"decode {picks.shape[0]:>5}  {n_out / secs:5.1f} tok/s"
        )

    if args.hidden:
        hidden_f.close()
        hpicks_f.close()
    Path(base + ".bin").write_bytes(np.concatenate(all_picks).tobytes())
    Path(base + ".scores.bin").write_bytes(np.concatenate(all_scores).tobytes())
    Path(base + ".pred.bin").write_bytes(np.concatenate(all_preds).tobytes())
    np.save(base + ".prefill.npy", np.stack(hists))
    meta = {
        "model": args.model,
        "num_layers": len(recs),
        "num_experts": num_experts,
        "top_k": top_k,
        "dtype": "uint8",
        "layout": "token,layer,k (k sorted by descending router score)",
        "tokens": start,
        "temp": args.temp,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "sequences": sequences,
    }
    if args.hidden:
        meta.update(hidden_size=int(h.shape[-1]), hidden_rows=hidden_rows,
                    hidden_dtype="float16", hidden_stride=args.hidden_stride)
    Path(base + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved {start} decode tokens to {base}.bin")


if __name__ == "__main__":
    main()
