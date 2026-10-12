"""Do a layer's experts share structure a compressor could keep in RAM?

Idea under test (D2-MoE / MoBE style): store per layer a shared input basis
V (in × r) that stays resident, and per expert only its coefficients
U_e = W_e·V (out × r), streamed from SSD. The expert costs r/in of its full
size. Whether that works depends on how much of each expert's weight energy
lives in a subspace the experts share, versus what each expert alone would
need, versus what an unstructured matrix of the same shape gives.

For sampled layers and each projection (gate, up: in = hidden; down: in =
moe_intermediate), reports the fraction of total weight energy (squared
Frobenius norm) captured at rank r by:

  shared   one basis for all experts of the layer (top-r eigenvectors of
           Σ_e W_eᵀW_e)
  own      each expert's own top-r right singular vectors (upper bound for
           any shared basis of the same rank)
  random   a Gaussian matrix of the same shape (no structure: about r/in)

plus the energy left after subtracting the mean expert (a shared base with a
full-rank delta, D2-MoE's first step).

Energy is a proxy, not quality: quality.py is the judge before anything ships.

Usage: uv run basis.py [--layers 0,12,24,36,47] [--ranks 64,128,256,512,1024]
"""

import argparse
import glob
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

SNAP = "~/.cache/huggingface/hub/models--mlx-community--Qwen3-30B-A3B-4bit/snapshots/*/"


def load_layer(layer: int) -> dict[str, np.ndarray]:
    """Dequantized expert weights [E, out, in] for one layer, float32."""
    want = f"model.layers.{layer}.mlp.switch_mlp."
    out = {}
    for f in sorted(glob.glob(os.path.expanduser(SNAP) + "model-*.safetensors")):
        t = mx.load(f)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            k = want + proj
            if k + ".weight" in t:
                w = mx.dequantize(t[k + ".weight"], t[k + ".scales"], t[k + ".biases"],
                                  group_size=64, bits=4)
                out[proj] = np.array(w.astype(mx.float32))
        del t
    return out


def energy_curves(w: np.ndarray, ranks: list[int], own_sample: int, rng) -> dict:
    E, o, n = w.shape
    total = float((w ** 2).sum())
    x = w.reshape(E * o, n)
    gram = x.T @ x  # Σ W_eᵀ W_e, n × n
    ev = np.linalg.eigvalsh(gram)[::-1]
    shared = {r: float(ev[:r].sum() / total) for r in ranks}

    own = {r: 0.0 for r in ranks}
    picks = rng.choice(E, size=min(own_sample, E), replace=False)
    for e in picks:
        s = np.linalg.svd(w[e], compute_uv=False) ** 2
        for r in ranks:
            own[r] += float(s[:r].sum() / s.sum()) / len(picks)

    g = rng.standard_normal((o, n)).astype(np.float32)
    s = np.linalg.svd(g, compute_uv=False) ** 2
    rand = {r: float(s[:r].sum() / s.sum()) for r in ranks}

    mean = w.mean(axis=0, keepdims=True)
    delta = float(((w - mean) ** 2).sum() / total)
    return {"shared": shared, "own": own, "random": rand, "delta_after_mean": delta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,12,24,36,47")
    ap.add_argument("--ranks", default="64,128,256,512,1024")
    ap.add_argument("--own-sample", type=int, default=16)
    ap.add_argument("--json", type=Path, default=Path("traces/basis.json"))
    args = ap.parse_args()
    ranks = [int(r) for r in args.ranks.split(",")]
    rng = np.random.default_rng(0)
    results = {}
    for layer in [int(x) for x in args.layers.split(",")]:
        ws = load_layer(layer)
        for proj, w in ws.items():
            rs = [r for r in ranks if r < w.shape[2]]
            c = energy_curves(w, rs, args.own_sample, rng)
            results[f"{layer}.{proj}"] = c
            print(f"layer {layer:2d} {proj:9s} in={w.shape[2]:4d}  delta after mean {100 * c['delta_after_mean']:5.1f}%")
            for r in rs:
                print(f"   r={r:4d} ({100 * r / w.shape[2]:4.1f}% size)  shared {100 * c['shared'][r]:5.1f}%"
                      f"  own {100 * c['own'][r]:5.1f}%  random {100 * c['random'][r]:5.1f}%")
    args.json.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
