"""Streaming MoE engine: routed experts live on the SSD, a small cache in RAM.

`load(pack_dir, cache_gb, bias)` returns an mlx-lm model whose MoE blocks read
experts from `experts.bin` on demand. Everything else (attention, embeddings,
routers) is loaded from `model.safetensors` and stays resident.

Per layer and forward call: route, find which experts are missing, read the
misses in parallel (one contiguous pread each, F_NOCACHE so the page cache
does not grow), stack the needed experts and run them with gather_qmm.

Cache: per-layer LFU with decay and bypass, as in tools/routing-trace (the
design that won the low-RAM study): a layer holds `slots` experts, scores
decay with a half-life of 32 calls, and a new expert only enters by beating
the weakest resident one; otherwise it is read just for this call.

`bias` > 0 adds that much to the log-probability of resident experts when
choosing the top-k (cache-aware routing; bias 0 = stock routing). Mixing
weights stay the router's own probabilities, renormalised over the chosen.
"""

import fcntl
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwiGLU
from mlx_lm.utils import load_model, load_tokenizer

F_NOCACHE = 48  # macOS
HALF_LIFE = 32.0


class ExpertStore:
    """Reads expert slots from experts.bin and tracks I/O."""

    def __init__(self, pack: Path, threads: int = 8):
        self.meta = json.loads((pack / "experts.json").read_text())
        self.slot = self.meta["slot_bytes"]
        self.n_experts = self.meta["experts"]
        self.fd = os.open(pack / "experts.bin", os.O_RDONLY)
        if hasattr(fcntl, "F_NOCACHE"):
            fcntl.fcntl(self.fd, fcntl.F_NOCACHE, 1)
        else:
            try:
                fcntl.fcntl(self.fd, F_NOCACHE, 1)
            except OSError:
                pass
        self.pool = ThreadPoolExecutor(threads)
        self.local = threading.local()
        self.bytes_read = 0
        self.reads = 0

    def _read(self, layer: int, expert: int) -> dict[str, mx.array]:
        buf = getattr(self.local, "buf", None)
        if buf is None:
            buf = self.local.buf = bytearray(self.slot)
        off = (layer * self.n_experts + expert) * self.slot
        n = os.preadv(self.fd, [buf], off)
        if n != self.slot:
            raise IOError(f"short read at layer {layer} expert {expert}: {n} of {self.slot}")
        raw = np.frombuffer(buf, dtype=np.uint8)
        out = {}
        for t in self.meta["tensors"]:
            dtype = np.uint16 if t["dtype"] == "bfloat16" else np.dtype(t["dtype"])
            a = mx.array(raw[t["offset"]:t["offset"] + t["nbytes"]].view(dtype).reshape(t["shape"]))
            out[t["name"]] = a.view(mx.bfloat16) if t["dtype"] == "bfloat16" else a
        mx.eval(out)  # copy out of the thread's buffer before it is reused
        return out

    def read_many(self, layer: int, experts: list[int]) -> list[dict[str, mx.array]]:
        self.reads += len(experts)
        self.bytes_read += len(experts) * self.slot
        return list(self.pool.map(lambda e: self._read(layer, e), experts))


class LayerCache:
    """Per-layer LFU with decay and bypass, `slots` experts."""

    def __init__(self, n_experts: int, slots: int):
        self.slots = slots
        self.decay = 0.5 ** (1 / HALF_LIFE)
        self.score = np.zeros(n_experts)
        self.experts: dict[int, dict[str, mx.array]] = {}
        self.hits = self.lookups = 0

    def resident_mask(self, n_experts: int) -> np.ndarray:
        m = np.zeros(n_experts, dtype=bool)
        m[list(self.experts)] = True
        return m

    def update(self, counts: dict[int, int], fetched: dict[int, dict[str, mx.array]]):
        """counts: uses per expert in this call; fetched: experts read for it."""
        self.score *= self.decay
        for e, c in counts.items():
            self.score[e] += c
        if not self.slots:
            return
        cand = set(self.experts) | set(fetched)
        keep = sorted(cand, key=lambda e: -self.score[e])[: self.slots]
        keep = set(keep)
        for e in list(self.experts):
            if e not in keep:
                del self.experts[e]
        for e in keep:
            if e not in self.experts:
                self.experts[e] = fetched[e]


class StreamingMoE(nn.Module):
    """Drop-in for Qwen3MoeSparseMoeBlock with experts streamed from the SSD."""

    def __init__(self, block, layer: int, store: ExpertStore, slots: int, bias: float):
        super().__init__()
        self.gate = block.gate
        self.top_k = block.top_k
        self.num_experts = block.num_experts
        self.norm_topk_prob = block.norm_topk_prob
        self.layer = layer
        self.bias = bias
        self.group_size = store.meta["group_size"]
        self.bits = store.meta["bits"]
        self.activation = SwiGLU()
        # Not module attributes: keep them out of parameters().
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_cache", LayerCache(block.num_experts, slots))

    def _qmm(self, x, w, name, idx, sort):
        return mx.gather_qmm(x, w[f"{name}.weight"], w[f"{name}.scales"], w[f"{name}.biases"],
                             rhs_indices=idx, transpose=True, group_size=self.group_size,
                             bits=self.bits, sorted_indices=sort)

    def __call__(self, x):
        store, cache, k = self._store, self._cache, self.top_k
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        if self.bias:
            sel = mx.log(gates) + self.bias * mx.array(cache.resident_mask(self.num_experts), dtype=gates.dtype)
            inds = mx.argpartition(sel, kth=-k, axis=-1)[..., -k:]
        else:
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)

        # The only sync per layer: which experts does this call need?
        np_inds = np.array(inds)
        used, counts = np.unique(np_inds, return_counts=True)
        cache.lookups += int(counts.sum())
        missing = [int(e) for e in used if int(e) not in cache.experts]
        cache.hits += int(counts.sum()) - int(sum(c for e, c in zip(used, counts) if int(e) in missing))
        fetched = dict(zip(missing, store.read_many(self.layer, missing)))
        experts = [cache.experts.get(int(e)) or fetched[int(e)] for e in used]

        # Stack the needed experts and remap the router's indices onto them.
        w = {name: mx.stack([ex[name] for ex in experts]) for name in experts[0]}
        remap = np.zeros(self.num_experts, dtype=np.uint32)
        remap[used] = np.arange(len(used), dtype=np.uint32)
        idx = mx.array(remap[np_inds])

        # Same math as mlx-lm's SwitchGLU.
        h = mx.expand_dims(x, (-2, -3))
        sort = idx.size >= 64
        inv = None
        if sort:
            flat = idx.flatten()
            order = mx.argsort(flat)
            inv = mx.argsort(order)
            h, idx_s = h.flatten(0, -3)[order // k], flat[order]
        else:
            idx_s = idx
        up = self._qmm(h, w, "up_proj", idx_s, sort)
        gate = self._qmm(h, w, "gate_proj", idx_s, sort)
        y = self._qmm(self.activation(up, gate), w, "down_proj", idx_s, sort)
        if sort:
            y = mx.unflatten(y[inv], 0, idx.shape)
        y = y.squeeze(-2)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

        cache.update({int(e): int(c) for e, c in zip(used, counts)}, fetched)
        return y


def load(pack: str | Path, cache_gb: float = 8.0, bias: float = 0.0, threads: int = 8):
    """Load a packed model. cache_gb is the RAM budget for cached experts."""
    pack = Path(pack).expanduser()
    # strict=False below would otherwise leave the model with random weights.
    if not (pack / "model.safetensors").exists():
        raise FileNotFoundError(f"{pack}/model.safetensors missing: not a packed model?")
    store = ExpertStore(pack, threads)
    model, _ = load_model(pack, lazy=True, strict=False)
    layers = model.layers
    slots = min(store.n_experts, int(cache_gb * 1e9 / (len(layers) * store.slot)))
    for i, layer in enumerate(layers):
        layer.mlp = StreamingMoE(layer.mlp, i, store, slots, bias)
    mx.eval(model.parameters())
    tokenizer = load_tokenizer(pack)
    return model, tokenizer, store


def stats(model, store) -> dict:
    caches = [layer.mlp._cache for layer in model.layers]
    lookups = sum(c.lookups for c in caches)
    return {
        "hit": sum(c.hits for c in caches) / max(lookups, 1),
        "gb_read": store.bytes_read / 1e9,
        "slots_per_layer": caches[0].slots,
        "cached_gb": sum(len(c.experts) for c in caches) * store.slot / 1e9,
    }
