"""The specialist wrappers must be transparent when nothing is cut.

  uv run python -m pytest -q test_specialist.py   (loads the 16 GB model once)
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm import load

from specialist import MODEL, NeuronMasked, expert_mask, install, neuron_mask

PROMPT = "def fibonacci(n):\n    # return the n-th Fibonacci number\n" * 12  # > 64 picks: sorted path too


@pytest.fixture(scope="module")
def setup():
    model, tok = load(MODEL)
    ids = mx.array(tok.encode(PROMPT))[None]
    short = ids[:, :5]  # < 64 picks: unsorted path
    ref = {"long": model(ids).astype(mx.float32), "short": model(short).astype(mx.float32)}
    mx.eval(ref)
    return model, install(model), {"long": ids, "short": short}, ref


@pytest.mark.parametrize("which", ["long", "short"])
def test_wrapper_is_transparent(setup, which):
    model, wraps, ids, ref = setup
    assert mx.allclose(model(ids[which]).astype(mx.float32), ref[which], atol=1e-4).item()


@pytest.mark.parametrize("which", ["long", "short"])
def test_all_ones_neuron_mask_is_transparent(setup, which):
    model, wraps, ids, ref = setup
    for w in wraps:
        sw = w.block.switch_mlp
        w.mlp = NeuronMasked(sw, mx.ones((128, 768), mx.bfloat16))
    out = model(ids[which]).astype(mx.float32)
    for w in wraps:
        w.mlp = w.block.switch_mlp
    assert mx.allclose(out, ref[which], atol=1e-4).item()


def test_calibration_path_matches_and_accumulates(setup):
    model, wraps, ids, ref = setup
    for w in wraps:
        w.stats = {"neuron": mx.zeros((128, 768)), "reap": mx.zeros((128,)), "count": mx.zeros((128,))}
    out = model(ids["long"]).astype(mx.float32)
    counts = [w.stats["count"].sum().item() for w in wraps]
    for w in wraps:
        w.stats = None
    # The unsorted gather_qmm rounds bf16 sums in another order than the sorted
    # path mlx-lm takes for long inputs: same distribution, not the same bits.
    lp = out - mx.logsumexp(out, -1, keepdims=True)
    rlp = ref["long"] - mx.logsumexp(ref["long"], -1, keepdims=True)
    kl = (mx.exp(rlp) * (rlp - lp)).sum(-1).mean().item()
    top1 = (mx.argmax(out, -1) == mx.argmax(ref["long"], -1)).mean().item()
    assert kl < 1e-2 and top1 >= 0.95, (kl, top1)
    assert all(c == ids["long"].shape[1] * 8 for c in counts)


def test_masks():
    stat = {"reap": mx.arange(128).astype(mx.float32), "neuron": mx.array(np.tile(np.arange(768), (128, 1)), mx.float32)}
    keep = np.array(expert_mask(stat, 0.5))
    assert keep.sum() == 64 and keep[64:].all()  # highest REAP scores survive
    m = np.array(neuron_mask(stat, 0.3).astype(mx.float32))
    assert (m[:, :230] == 0).all() and (m[:, 231:] == 1).all()
