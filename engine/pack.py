"""Pack an MLX MoE model for SSD streaming.

Writes, into the output directory:

  model.safetensors  every weight except the routed experts (stays in RAM;
                     the name mlx-lm's load_model looks for)
  experts.bin        one slot per (layer, expert), 16 KiB aligned, holding that
                     expert's gate/up/down weight, scales and biases back to
                     back, so a miss is a single contiguous read
  experts.json       the slot layout
  config.json, tokenizer files   copied from the source model

Usage: uv run pack.py mlx-community/Qwen3-30B-A3B-4bit ~/packs/qwen3-30b
"""

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

ALIGN = 16 * 1024
EXPERT_KEY = re.compile(r"model\.layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$")
PARTS = [f"{p}.{t}" for p in ("gate_proj", "up_proj", "down_proj") for t in ("weight", "scales", "biases")]


def to_numpy(a: mx.array) -> tuple[np.ndarray, str]:
    """numpy has no bfloat16: keep its bits as uint16 and remember the dtype."""
    if a.dtype == mx.bfloat16:
        return np.array(a.view(mx.uint16)), "bfloat16"
    return np.array(a), str(np.array(a[:0]).dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Hugging Face repo id or local path of an MLX MoE model")
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    src = Path(args.model) if Path(args.model).exists() else Path(snapshot_download(args.model))
    config = json.loads((src / "config.json").read_text())
    n_layers, n_experts = config["num_hidden_layers"], config["num_experts"]
    args.out.mkdir(parents=True, exist_ok=True)

    index = src / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        weight_map = {k: "model.safetensors" for k in mx.load(str(src / "model.safetensors"))}
    files: dict[str, dict] = {}

    def tensor(name):
        f = weight_map[name]
        if f not in files:
            files[f] = mx.load(str(src / f))
        return files[f][name]

    # Everything but the routed experts goes to model.safetensors.
    base = {k: tensor(k) for k in weight_map if not EXPERT_KEY.match(k)}
    mx.save_safetensors(str(args.out / "model.safetensors"), base, metadata={"format": "mlx"})
    del base
    print(f"model.safetensors: {(args.out / 'model.safetensors').stat().st_size / 1e9:.2f} GB")

    layout, slot = None, None
    t0 = time.time()
    with open(args.out / "experts.bin", "wb") as f:
        for layer in range(n_layers):
            arrays = {}
            for part in PARTS:
                name = f"model.layers.{layer}.mlp.switch_mlp.{part}"
                arrays[part] = to_numpy(tensor(name))
            if layout is None:
                layout, off = [], 0
                for part in PARTS:
                    a, dtype = arrays[part]
                    nbytes = a[0].nbytes
                    layout.append({"name": part, "offset": off, "nbytes": nbytes,
                                   "shape": list(a.shape[1:]), "dtype": dtype})
                    off += nbytes
                slot = -(-off // ALIGN) * ALIGN
            buf = bytearray(slot * n_experts)
            for e in range(n_experts):
                for t in layout:
                    a = arrays[t["name"]][0][e]
                    start = e * slot + t["offset"]
                    buf[start:start + t["nbytes"]] = a.tobytes()
            f.write(buf)
            files.clear()  # drop the mapped shards so memory stays at one layer
            print(f"layer {layer + 1}/{n_layers} ({time.time() - t0:.0f}s)", flush=True)

    meta = {"layers": n_layers, "experts": n_experts, "slot_bytes": slot, "tensors": layout,
            "group_size": config["quantization"]["group_size"], "bits": config["quantization"]["bits"]}
    (args.out / "experts.json").write_text(json.dumps(meta, indent=1) + "\n")
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name != "model.safetensors.index.json":
            shutil.copy(p, args.out / p.name)
    print(f"experts.bin: {n_layers} layers × {n_experts} experts × {slot / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
