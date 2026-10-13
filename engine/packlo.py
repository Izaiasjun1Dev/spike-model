"""Add codebook (2.5-bit) copies of the experts to a pack made by pack.py.

Writes, into the pack directory:

  experts_lo.bin         one slot per (layer, expert), 16 KiB aligned:
                         codes and row scales of gate/up/down (cbq.py)
  experts_lo.json        the slot layout
  codebooks.safetensors  layers.<l>.<proj>: [K, D] f16

The engine serves the lowest-weight half of a token's picks from these
copies (0.56 of the q4 bytes) when the q4 copy is not resident.

Usage: uv run packlo.py ~/packs/qwen3-30b [--layers 0-3]   (subset for tests)
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import cbq

ALIGN = 16 * 1024
PROJS = ("gate_proj", "up_proj", "down_proj")
FIT_EXPERTS = 16  # experts sampled to fit each codebook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack", type=Path)
    ap.add_argument("--layers", help="range a-b (inclusive) for a partial pack, e.g. 0-3")
    args = ap.parse_args()
    mx.random.seed(0)
    mx.set_cache_limit(2 << 30)

    meta = json.loads((args.pack / "experts.json").read_text())
    L, E, slot = meta["layers"], meta["experts"], meta["slot_bytes"]
    gs, bits = meta["group_size"], meta["bits"]
    parts = {t["name"]: t for t in meta["tensors"]}
    layers = range(L)
    if args.layers:
        a, b = map(int, args.layers.split("-"))
        layers = range(a, b + 1)

    def tensor(buf: bytes, name: str) -> mx.array:
        t = parts[name]
        raw = np.frombuffer(buf, dtype=np.uint8, count=t["nbytes"], offset=t["offset"])
        if t["dtype"] == "bfloat16":
            return mx.array(raw.view(np.uint16).reshape(t["shape"])).view(mx.bfloat16)
        return mx.array(raw.view(np.dtype(t["dtype"])).reshape(t["shape"]))

    def weight(buf: bytes, proj: str) -> mx.array:
        """Dequantized [out, in] of one expert's projection."""
        return mx.dequantize(tensor(buf, f"{proj}.weight"), tensor(buf, f"{proj}.scales"),
                             tensor(buf, f"{proj}.biases"), group_size=gs, bits=bits)

    # Slot layout of the low copy.
    layout, off = [], 0
    for proj in PROJS:
        out, in8 = parts[f"{proj}.weight"]["shape"]
        in_dim = in8 * 32 // bits
        for name, nbytes, shape, dtype in ((f"{proj}.codes", out * in_dim * 5 // 16, [out, in_dim * 5 // 16], "uint8"),
                                           (f"{proj}.scale", out * 2, [out], "float16")):
            layout.append({"name": name, "offset": off, "nbytes": nbytes, "shape": shape, "dtype": dtype})
            off += nbytes
    lo_slot = -(-off // ALIGN) * ALIGN

    codebooks = {}
    t0 = time.time()
    with open(args.pack / "experts.bin", "rb") as src, open(args.pack / "experts_lo.bin", "r+b" if (args.pack / "experts_lo.bin").exists() else "wb") as dst:
        for layer in layers:
            src.seek(layer * E * slot)
            bufs = [src.read(slot) for _ in range(E)]
            out = bytearray(lo_slot * E)
            for proj in PROJS:
                # Fit the codebook on a sample of experts, then encode all.
                sample = mx.random.permutation(E)[:FIT_EXPERTS].tolist()
                vecs = mx.concatenate([cbq.normalise(weight(bufs[e], proj))[0].reshape(-1, cbq.D) for e in sample])
                cb = cbq.kmeans(vecs).astype(mx.float16)
                mx.eval(cb)
                codebooks[f"layers.{layer}.{proj}"] = cb
                del vecs
                c, s = layout[[t["name"] for t in layout].index(f"{proj}.codes")], layout[[t["name"] for t in layout].index(f"{proj}.scale")]
                for e in range(E):
                    codes, scale = cbq.encode(weight(bufs[e], proj), cb)
                    out[e * lo_slot + c["offset"]:e * lo_slot + c["offset"] + c["nbytes"]] = codes.tobytes()
                    out[e * lo_slot + s["offset"]:e * lo_slot + s["offset"] + s["nbytes"]] = scale.tobytes()
                mx.clear_cache()
            dst.seek(layer * E * lo_slot)
            dst.write(out)
            print(f"layer {layer + 1}/{L} ({time.time() - t0:.0f}s)", flush=True)

    cb_path = args.pack / "codebooks.safetensors"
    if cb_path.exists():
        old = mx.load(str(cb_path))
        old.update(codebooks)
        codebooks = old
    mx.save_safetensors(str(cb_path), codebooks, metadata={"format": "mlx"})
    lo_meta = {"layers": L, "experts": E, "slot_bytes": lo_slot, "tensors": layout,
               "codebook": {"d": cbq.D, "k": cbq.K, "bits": cbq.BITS, "hadamard": True}}
    (args.pack / "experts_lo.json").write_text(json.dumps(lo_meta, indent=1) + "\n")
    print(f"experts_lo.bin: {L} layers × {E} experts × {lo_slot / 2**20:.2f} MiB ({lo_slot / slot:.2f} of q4)")


if __name__ == "__main__":
    main()
