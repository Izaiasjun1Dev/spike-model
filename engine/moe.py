"""One command to run a big MoE model from the SSD.

  uv run moe.py build                                   # compile the engine (once)
  uv run moe.py get mlx-community/Qwen3-235B-A22B-4bit  # download + pack (+ codebook copies with --fast)
  uv run moe.py chat qwen3-235b-a22b-4bit [--ram 16]    # chat; --fast uses the codebook copies
  uv run moe.py list

`--ram` is the memory the engine may use (default: 70% of the machine's).
The expert cache gets what is left after the resident weights, the arenas'
spare slots and a margin for the KV cache.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE_RS = HERE.parent / "engine-rs"
BIN = ENGINE_RS / "target/release/moe-run"
PACKS = Path(os.environ.get("MOE_PACKS", Path.home() / "packs"))


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def mlx_dir() -> Path:
    import mlx

    return Path(mlx.__file__).resolve().parent


def build(args):
    env = dict(os.environ, MLX_SYSTEM_DIR=str(mlx_dir()))
    run(["cargo", "build", "--release"], cwd=ENGINE_RS, env=env)
    print(f"engine: {BIN}")


def pack_dir(name: str) -> Path:
    p = Path(name).expanduser()
    return p if p.is_dir() else PACKS / name


def get(args):
    from huggingface_hub import snapshot_download

    name = args.name or args.repo.split("/")[-1].lower()
    out = PACKS / name
    if (out / "experts.json").exists() and not args.force:
        print(f"{out}: already packed")
    else:
        src = snapshot_download(args.repo, max_workers=16)
        run([sys.executable, HERE / "pack.py", src, out])
        if args.clean:
            shutil.rmtree(Path(src).resolve().parent.parent)  # the whole models--... cache dir
            print("download removed")
    if args.fast and not (out / "experts_lo.json").exists():
        run([sys.executable, HERE / "packlo.py", out])
    print(f"ready: uv run moe.py chat {name}" + (" --fast" if args.fast else ""))


def budget(pack: Path, ram_gb: float, fast: bool) -> float:
    """Expert cache size in GB for a RAM budget."""
    fixed = (pack / "model.safetensors").stat().st_size
    cfg = json.loads((pack / "config.json").read_text())
    hi = json.loads((pack / "experts.json").read_text())
    experts, top_k = hi["experts"], cfg["num_experts_per_tok"]
    spare = (experts + top_k) * hi["slot_bytes"]
    if fast:
        lo = json.loads((pack / "experts_lo.json").read_text())
        spare += (experts + 2 * top_k) * lo["slot_bytes"]
    margin = 1.5e9  # KV cache, activations, MLX's buffer cache
    cache = ram_gb * 1e9 - fixed - spare - margin
    if cache < 1e9:
        raise SystemExit(f"{ram_gb} GB is not enough: resident weights {fixed / 1e9:.1f} GB + spare slots "
                         f"{spare / 1e9:.1f} GB + margin 1.5 GB leave {cache / 1e9:.1f} GB for the cache")
    print(f"[ram {ram_gb:g} GB: weights {fixed / 1e9:.1f} + spare {spare / 1e9:.1f} + margin 1.5 + "
          f"expert cache {cache / 1e9:.1f} GB]")
    return cache / 1e9


def chat(args):
    pack = pack_dir(args.pack)
    if not (pack / "experts.json").exists():
        raise SystemExit(f"no pack at {pack}; run: uv run moe.py get <repo>")
    if not BIN.exists():
        build(args)
    fast = args.fast
    if fast and not (pack / "experts_lo.json").exists():
        raise SystemExit(f"{pack} has no codebook copies; run: uv run moe.py get <repo> --fast")
    ram = args.ram or os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9 * 0.7
    cmd = [BIN, pack, "--cache-gb", f"{budget(pack, ram, fast):.2f}", "--bias", str(args.bias)]
    cmd += ["--lo-all", "--lo-frac", "1"] if fast else ["--no-tiered"]
    cmd += ["-p", args.prompt] if args.prompt else ["--chat"]
    if args.max_tokens:
        cmd += ["--max-tokens", str(args.max_tokens)]
    os.execv(str(BIN), [str(c) for c in cmd])


def list_packs(args):
    for p in sorted(PACKS.glob("*/experts.json")):
        d = p.parent
        gb = sum(f.stat().st_size for f in d.iterdir()) / 1e9
        print(f"{d.name:40s} {gb:7.0f} GB" + ("  (+ codebook copies)" if (d / "experts_lo.json").exists() else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=build)
    g = sub.add_parser("get")
    g.add_argument("repo")
    g.add_argument("--name")
    g.add_argument("--fast", action="store_true", help="also write the 2.5-bit codebook copies (slow, once)")
    g.add_argument("--clean", action="store_true", help="delete the Hugging Face download after packing")
    g.add_argument("--force", action="store_true")
    g.set_defaults(fn=get)
    c = sub.add_parser("chat")
    c.add_argument("pack")
    c.add_argument("--ram", type=float, help="GB the engine may use (default 70%% of the machine)")
    c.add_argument("--fast", action="store_true", help="every expert from the codebook copy (~2x faster, ppl x1.14)")
    c.add_argument("--bias", type=float, default=1.0, help="cache-aware routing strength (0 = stock routing)")
    c.add_argument("-p", "--prompt")
    c.add_argument("--max-tokens", type=int)
    c.set_defaults(fn=chat)
    sub.add_parser("list").set_defaults(fn=list_packs)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
