"""Add low-precision copies of the experts to a pack made by pack.py.

Writes, into the pack directory:

  experts_lo.bin         one slot per (layer, expert), 16 KiB aligned:
                         gate/up as codebook codes and row scales (cbq.py,
                         2.5 bits); down_proj as affine q3 by default
                         (--down 3), or q4 copied from the pack (--down 4),
                         or codebook too (--down cb)
  experts_lo.json        the slot layout ("down_bits": null = codebook)
  codebooks.safetensors  layers.<l>.<proj>: [K, D] f16

The engine serves the lowest-weight half of a token's picks (or every pick,
--lo-all) from these copies when the q4 copy is not resident. Keeping
down_proj in affine q3 costs 0.63 of the q4 bytes instead of 0.56 and buys
most of the quality back: it is the cheapest bit of the expert, as the
community's Kimi-K2 quants also found.

Usage: uv run packlo.py ~/packs/qwen3-30b [--down 3|4|cb] [--layers 0-3]   (subset for tests)
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
    ap.add_argument("--down", default="3", help="down_proj format: affine bits (3, 4 = copy of the pack's q4) or cb (codebook)")
    args = ap.parse_args()
    down_bits = None if args.down == "cb" else int(args.down)
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
        if proj == "down_proj" and down_bits is not None:
            tensors = ((f"{proj}.weight", out * in_dim * down_bits // 8, [out, in_dim * down_bits // 32], "uint32"),
                       (f"{proj}.scales", out * (in_dim // gs) * 2, [out, in_dim // gs], "bfloat16"),
                       (f"{proj}.biases", out * (in_dim // gs) * 2, [out, in_dim // gs], "bfloat16"))
        else:
            tensors = ((f"{proj}.codes", out * in_dim * 5 // 16, [out, in_dim * 5 // 16], "uint8"),
                       (f"{proj}.scale", out * 2, [out], "float16"))
        for name, nbytes, shape, dtype in tensors:
            layout.append({"name": name, "offset": off, "nbytes": nbytes, "shape": shape, "dtype": dtype})
            off += nbytes
    lo_slot = -(-off // ALIGN) * ALIGN
    part_of = {t["name"]: t for t in layout}

    def put(out: bytearray, e: int, name: str, data: bytes):
        t = part_of[name]
        assert len(data) == t["nbytes"], (name, len(data), t["nbytes"])
        out[e * lo_slot + t["offset"]:e * lo_slot + t["offset"] + t["nbytes"]] = data

    def affine_down(buf: bytes) -> tuple[bytes, bytes, bytes]:
        """down_proj of one expert as (weight, scales, biases) bytes in down_bits."""
        if down_bits == bits:  # the pack's own q4: copy the bytes, no re-quantization
            return tuple(buf[parts[f"down_proj.{n}"]["offset"]:][:parts[f"down_proj.{n}"]["nbytes"]] for n in ("weight", "scales", "biases"))
        w, sc, bi = mx.quantize(weight(buf, "down_proj"), group_size=gs, bits=down_bits)
        mx.eval(w, sc, bi)
        return (np.array(w).tobytes(), np.array(sc.view(mx.uint16)).tobytes(), np.array(bi.view(mx.uint16)).tobytes())

    codebooks = {}
    t0 = time.time()
    with open(args.pack / "experts.bin", "rb") as src, open(args.pack / "experts_lo.bin", "r+b" if (args.pack / "experts_lo.bin").exists() else "wb") as dst:
        for layer in layers:
            src.seek(layer * E * slot)
            bufs = [src.read(slot) for _ in range(E)]
            out = bytearray(lo_slot * E)
            for proj in PROJS:
                if proj == "down_proj" and down_bits is not None:
                    for e in range(E):
                        for name, data in zip(("weight", "scales", "biases"), affine_down(bufs[e])):
                            put(out, e, f"down_proj.{name}", data)
                    mx.clear_cache()
                    continue
                # Fit the codebook on a sample of experts, then encode all.
                sample = mx.random.permutation(E)[:FIT_EXPERTS].tolist()
                vecs = mx.concatenate([cbq.normalise(weight(bufs[e], proj))[0].reshape(-1, cbq.D) for e in sample])
                cb = cbq.kmeans(vecs).astype(mx.float16)
                mx.eval(cb)
                codebooks[f"layers.{layer}.{proj}"] = cb
                del vecs
                for e in range(E):
                    codes, scale = cbq.encode(weight(bufs[e], proj), cb)
                    put(out, e, f"{proj}.codes", codes.tobytes())
                    put(out, e, f"{proj}.scale", scale.tobytes())
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
    lo_meta = {"layers": L, "moe_layers": meta.get("moe_layers", list(range(L))), "experts": E, "slot_bytes": lo_slot, "tensors": layout,
               "codebook": {"d": cbq.D, "k": cbq.K, "bits": cbq.BITS, "hadamard": True},
               "down_bits": down_bits, "group_size": gs}
    (args.pack / "experts_lo.json").write_text(json.dumps(lo_meta, indent=1) + "\n")
    print(f"experts_lo.bin: {L} layers × {E} experts × {lo_slot / 2**20:.2f} MiB ({lo_slot / slot:.2f} of q4)")


if __name__ == "__main__":
    main()
