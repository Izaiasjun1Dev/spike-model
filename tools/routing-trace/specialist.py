"""Does a code "specialist" keep code quality after shrinking the experts?

Three cuts, each calibrated on code (evol-codealpaca), alone and combined:

  E<p>  REAP expert pruning: drop the p% experts per layer with the lowest
        mean (router weight x ||expert output||) on code; the router then
        picks its top-8 among the survivors (same as REAP removing them).
  N<p>  FlexMoE-style neuron pruning: inside every expert, drop the p% of
        the 768 intermediate neurons with the lowest
        mean(router weight x |silu(gate)*up|) x ||down column||.
  Q     mixed precision inside the expert: gate/up in the d4 k1024 Hadamard
        codebook (2.5 bits), down_proj in q3 affine (codebook.py's
        "vq d4 k1024 had down q3", 2.83 bits/weight).

Pruning is applied as masks, so this measures quality; the size of each
variant is arithmetic (printed as GB for Kimi-K2's 570.8 GB of experts).

Measured per variant, against the unmodified 4-bit model:
  code     HumanEval (164) + MBPP sanitized (257), greedy pass@1, no thinking
  general  MMLU, 1000 random questions, zero-shot letter likelihood
  ppl      held-out code (codeparrot valid), English (fineweb-edu), Portuguese (Wikipedia)

Generated programs are executed locally in a subprocess (temp dir, 10 s
timeout), as every HumanEval/MBPP harness does.

Usage: uv run specialist.py [--only "base,E50,E50+N50+Q"] [--limit 40]
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np
import mlx.nn as nn
from datasets import load_dataset
from mlx_lm import load
from mlx_lm.generate import batch_generate
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

from codebook import vq_copy

MODEL = "mlx-community/Qwen3-30B-A3B-4bit"
VARIANTS = ["base", "E25", "E50", "N30", "N50", "E50+N50", "Q", "E50+Q", "N50+Q", "E50+N50+Q"]
KIMI_EXPERTS_GB, KIMI_FIXED_GB = 570.8, 6.6
Q_BYTES = 2.83 / 4.5  # mixed precision vs q4 g64 (4.5 bits/weight)
VQ_CACHE = Path.home() / "packs/qwen3-30b/specialist_vq_d4k1024had_down3.safetensors"


class NeuronMasked(nn.Module):
    """SwitchGLU with a per-expert mask on the intermediate neurons."""

    def __init__(self, sw, mask: mx.array):
        super().__init__()
        self.sw = sw
        self.mask = mask  # [experts, hidden], 0/1 in the activation dtype
        self.freeze()

    def __call__(self, x, indices):
        sw = self.sw
        x = mx.expand_dims(x, (-2, -3))
        indices = mx.stop_gradient(indices)
        do_sort = indices.size >= 64
        idx, inv_order = indices, None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        h = sw.activation(sw.up_proj(x, idx, sorted_indices=do_sort), sw.gate_proj(x, idx, sorted_indices=do_sort))
        h = h * mx.expand_dims(self.mask[idx], -2).astype(h.dtype)
        x = sw.down_proj(h, idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class SpecialistMoE(nn.Module):
    """Wraps a Qwen3 MoE block: optional expert mask, calibration statistics."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.mlp = block.switch_mlp  # what runs the experts (may be wrapped/replaced)
        self.allowed = None  # bool [experts] or None
        self.stats = None  # dict of accumulators while calibrating

    def __call__(self, x):
        b = self.block
        gates = mx.softmax(b.gate(x), axis=-1, precise=True)
        if self.allowed is not None:
            gates = mx.where(self.allowed, gates, mx.zeros_like(gates))
        k = b.top_k
        inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if b.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)
        if self.stats is not None:
            y = self._calibrate(x, inds, scores)
        else:
            y = self.mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

    def _calibrate(self, x, inds, scores):
        # Unsorted SwitchGLU path, keeping the intermediate activations.
        sw, s = self.block.switch_mlp, self.stats
        xe = mx.expand_dims(x, (-2, -3))
        h = sw.activation(sw.up_proj(xe, inds), sw.gate_proj(xe, inds))  # [B, T, k, 1, H]
        y = sw.down_proj(h, inds)  # [B, T, k, 1, D]
        flat = inds.reshape(-1)
        g = scores.reshape(-1, 1).astype(mx.float32)
        hn = mx.abs(h.reshape(flat.size, -1).astype(mx.float32)) * g
        yn = mx.linalg.norm(y.reshape(flat.size, -1).astype(mx.float32), axis=-1) * g[:, 0]
        s["neuron"] = s["neuron"].at[flat].add(hn)
        s["reap"] = s["reap"].at[flat].add(yn)
        s["count"] = s["count"].at[flat].add(mx.ones(flat.shape))
        mx.eval(s["neuron"], s["reap"], s["count"])
        return y.squeeze(-2)


def install(model) -> list[SpecialistMoE]:
    wraps = []
    for layer in model.layers:
        w = SpecialistMoE(layer.mlp)
        layer.mlp = w
        wraps.append(w)
    return wraps


# ---------------------------------------------------------------- data

def chat(tok, user: str) -> list[int]:
    return tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                   enable_thinking=False)


def calibration(tok, n_samples: int, max_len: int) -> list[list[int]]:
    ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train", streaming=True).shuffle(seed=0, buffer_size=2000)
    out = []
    for ex in ds:
        msgs = [{"role": "user", "content": ex["instruction"]}, {"role": "assistant", "content": ex["output"]}]
        ids = tok.apply_chat_template(msgs, enable_thinking=False)[:max_len]
        if len(ids) >= 128:
            out.append(ids)
        if len(out) >= n_samples:
            return out
    return out


def ppl_sets(tok, n: int, max_len: int) -> dict[str, list[list[int]]]:
    srcs = {
        "code": ("codeparrot/codeparrot-clean-valid", None, "content"),
        "en": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
        "pt": ("wikimedia/wikipedia", "20231101.pt", "text"),
    }
    sets = {}
    for name, (ds_name, cfg, field) in srcs.items():
        docs = []
        for ex in load_dataset(ds_name, cfg, split="train", streaming=True):
            ids = tok.encode(ex[field])[:max_len]
            if len(ids) >= 256:
                docs.append(ids)
            if len(docs) >= n:
                break
        sets[name] = docs
    return sets


def code_tasks(limit: int | None) -> list[dict]:
    tasks = []
    for ex in load_dataset("openai/openai_humaneval", split="test"):
        imports = "\n".join(l for l in ex["prompt"].splitlines() if l.startswith(("import ", "from ")))
        tasks.append({
            "id": ex["task_id"],
            "prompt": "Complete the following Python function. Reply with the whole function in a single "
                      f"```python code block and no explanation.\n\n```python\n{ex['prompt']}```",
            "harness": lambda code, ex=ex, imports=imports: f"{imports}\n{code}\n\n{ex['test']}\ncheck({ex['entry_point']})\n",
        })
    for ex in load_dataset("google-research-datasets/mbpp", "sanitized", split="test"):
        tests = "\n".join(ex["test_list"])
        imports = "\n".join(ex["test_imports"])
        tasks.append({
            "id": f"mbpp/{ex['task_id']}",
            "prompt": f"{ex['prompt']}\nYour code should pass these tests:\n```python\n{tests}\n```\n"
                      "Reply with the code in a single ```python code block and no explanation.",
            "harness": lambda code, tests=tests, imports=imports: f"{imports}\n{code}\n\n{tests}\n",
        })
    return tasks[:limit] if limit else tasks


def mmlu(n: int) -> list[dict]:
    ds = list(load_dataset("cais/mmlu", "all", split="test"))
    return random.Random(0).sample(ds, n)


# ---------------------------------------------------------------- metrics

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    blocks = CODE_RE.findall(text)
    return blocks[-1] if blocks else text


def run_program(src: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "prog.py"
        path.write_text(src)
        try:
            r = subprocess.run([sys.executable, str(path)], cwd=d, capture_output=True, timeout=10,
                               env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"})
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def eval_code(model, tok, tasks, batch: int, max_tokens: int) -> dict:
    prompts = [chat(tok, t["prompt"]) for t in tasks]
    texts = []
    for i in range(0, len(prompts), batch):
        texts += batch_generate(model, tok, prompts[i:i + batch], max_tokens=max_tokens).texts
    with ThreadPoolExecutor(8) as pool:
        passed = list(pool.map(lambda tt: run_program(tt[0]["harness"](extract_code(tt[1]))), zip(tasks, texts)))
    return {"pass": {t["id"]: p for t, p in zip(tasks, passed)}, "texts": {t["id"]: x for t, x in zip(tasks, texts)}}


def eval_mmlu(model, tok, qs) -> float:
    letters = [tok.encode(l, add_special_tokens=False)[0] for l in "ABCD"]
    correct = 0
    for q in qs:
        body = q["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", q["choices"]))
        ids = chat(tok, body + "\nAnswer with the letter of the correct option only.")
        logits = model(mx.array(ids)[None])[0, -1]
        correct += int(mx.argmax(logits[mx.array(letters)]).item() == q["answer"])
    return correct / len(qs)


def eval_ppl(model, docs) -> float:
    nll, n = 0.0, 0
    for ids in docs:
        x = mx.array(ids)[None]
        logits = model(x[:, :-1]).astype(mx.float32)
        loss = mx.logsumexp(logits, -1) - mx.take_along_axis(logits, x[:, 1:, None], -1)[..., 0]
        nll += loss.sum().item()
        n += len(ids) - 1
    return math.exp(nll / n)


# ---------------------------------------------------------------- variants

def calibrate(model, wraps, samples):
    E = wraps[0].block.num_experts
    H = wraps[0].block.switch_mlp.down_proj.weight.shape[-1] * 32 // wraps[0].block.switch_mlp.down_proj.bits
    for w in wraps:
        w.stats = {"neuron": mx.zeros((E, H)), "reap": mx.zeros((E,)), "count": mx.zeros((E,))}
    t0 = time.time()
    for i, ids in enumerate(samples):
        model(mx.array(ids)[None])
        if i % 64 == 0:
            print(f"  calibration {i}/{len(samples)}  {time.time() - t0:.0f}s", flush=True)
    stats = []
    for w in wraps:
        s, w.stats = w.stats, None
        dp = w.block.switch_mlp.down_proj
        down = mx.dequantize(dp.weight, dp.scales, dp.biases, group_size=dp.group_size, bits=dp.bits)  # [E, D, H]
        col = mx.linalg.norm(down.astype(mx.float32), axis=1)  # [E, H]
        cnt = mx.maximum(s["count"], 1)
        stats.append({"reap": s["reap"] / cnt, "neuron": s["neuron"] / cnt[:, None] * col, "count": s["count"]})
        mx.eval(stats[-1])
    return stats


def expert_mask(stat, frac: float) -> mx.array:
    reap = np.array(stat["reap"])
    keep = np.zeros(reap.shape, bool)
    keep[np.argsort(-reap)[: len(reap) - round(frac * len(reap))]] = True
    return mx.array(keep)


def neuron_mask(stat, frac: float) -> mx.array:
    score = np.array(stat["neuron"])
    mask = np.ones(score.shape, np.float32)
    np.put_along_axis(mask, np.argsort(score, axis=1)[:, : round(frac * score.shape[1])], 0.0, axis=1)
    return mx.array(mask).astype(mx.bfloat16)


def build_quant(wraps):
    """Swap every block's experts for the mixed-precision copy (cached on disk)."""
    mx.random.seed(0)
    if VQ_CACHE.exists():
        saved = mx.load(str(VQ_CACHE))
    else:
        saved = None
    out = {}
    for i, w in enumerate(wraps):
        sw = w.block.switch_mlp
        if saved is None:
            lo, errs = vq_copy(sw, 4, 1024, True, down_bits=3)
            print(f"  vq layer {i}: rel err {', '.join(f'{e:.3f}' for e in errs)}", flush=True)
        else:
            lo, _ = _vq_from_cache(sw, saved, i)
        w.block.switch_mlp = w.mlp = lo  # drop the q4 experts of this layer now
        del sw
        for p in ("gate_proj", "up_proj", "down_proj"):
            inner = getattr(lo, p)
            inner = getattr(inner, "inner", inner)
            for a in ("weight", "scales", "biases"):
                out[f"{i}.{p}.{a}"] = getattr(inner, a)
        mx.eval(out)
    if saved is None:
        mx.save_safetensors(str(VQ_CACHE), out)
        print(f"  cached -> {VQ_CACHE}", flush=True)


def _vq_from_cache(sw, saved, i):
    import copy
    from quality import Rotated
    lo = copy.copy(sw)
    for p in ("gate_proj", "up_proj", "down_proj"):
        dst = copy.copy(getattr(sw, p))
        dst._no_grad = set(getattr(dst, "_no_grad", ()))
        dst.weight, dst.scales, dst.biases = (saved[f"{i}.{p}.{a}"] for a in ("weight", "scales", "biases"))
        dst.bits = 3 if p == "down_proj" else 5
        dst.group_size = 64
        setattr(lo, p, dst if p == "down_proj" else Rotated(dst))
    return lo, []


def apply(wraps, stats, name: str):
    parts = {p[0]: p for p in name.split("+")} if name != "base" else {}
    for w, st in zip(wraps, stats):
        w.allowed = expert_mask(st, int(parts["E"][1:]) / 100) if "E" in parts else None
        sw = w.block.switch_mlp
        w.mlp = NeuronMasked(sw, neuron_mask(st, int(parts["N"][1:]) / 100)) if "N" in parts else sw


def kimi_gb(name: str) -> float:
    f = 1.0
    for p in name.split("+"):
        if p[0] in "EN":
            f *= 1 - int(p[1:]) / 100
        elif p == "Q":
            f *= Q_BYTES
    return KIMI_EXPERTS_GB * f + KIMI_FIXED_GB


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated variants (default: all)")
    ap.add_argument("--limit", type=int, help="code tasks to run (smoke test)")
    ap.add_argument("--calib", type=int, default=512, help="calibration samples")
    ap.add_argument("--mmlu", type=int, default=1000)
    ap.add_argument("--ppl-docs", type=int, default=48)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--json", type=Path, default=Path("traces/specialist.json"))
    args = ap.parse_args()

    names = args.only.split(",") if args.only else VARIANTS
    names = sorted(names, key=lambda n: "Q" in n.split("+"))  # every Q variant after the swap
    args.json.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.json.read_text()) if args.json.exists() else {}

    model, tok = load(MODEL)
    wraps = install(model)
    tasks = code_tasks(args.limit)
    qs = mmlu(args.mmlu)
    ppl_docs = ppl_sets(tok, args.ppl_docs, 1024)

    print(f"calibrating on {args.calib} code samples", flush=True)
    stats = calibrate(model, wraps, calibration(tok, args.calib, 2048))

    quantized = False
    for name in names:
        if "Q" in name.split("+") and not quantized:
            print("building mixed-precision experts", flush=True)
            build_quant(wraps)
            quantized = True
        apply(wraps, stats, name)
        t0 = time.time()
        code = eval_code(model, tok, tasks, args.batch, args.max_tokens)
        res = {
            "kimi_gb": round(kimi_gb(name), 1),
            "code_pass": sum(code["pass"].values()) / len(tasks),
            "humaneval": _rate(code["pass"], lambda k: not k.startswith("mbpp")),
            "mbpp": _rate(code["pass"], lambda k: k.startswith("mbpp")),
            "mmlu": eval_mmlu(model, tok, qs),
            "ppl": {k: eval_ppl(model, v) for k, v in ppl_docs.items()},
            "tasks": len(tasks),
            "pass_by_task": code["pass"],
            "seconds": round(time.time() - t0),
        }
        results[name] = res
        args.json.write_text(json.dumps(results, indent=1))
        (args.json.parent / f"specialist-{name}.texts.json").write_text(json.dumps(code["texts"]))
        print(f"{name:12s} kimi {res['kimi_gb']:6.1f} GB | code {res['code_pass']:.1%} "
              f"(HE {res['humaneval']:.1%}, MBPP {res['mbpp']:.1%}) | MMLU {res['mmlu']:.1%} | "
              + " ".join(f"ppl-{k} {v:.3f}" for k, v in res["ppl"].items()) + f" | {res['seconds']}s", flush=True)


def _rate(passed: dict, pick) -> float:
    vals = [v for k, v in passed.items() if pick(k)]
    return sum(vals) / len(vals) if vals else float("nan")


if __name__ == "__main__":
    main()
