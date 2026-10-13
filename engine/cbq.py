"""Codebook (vector) quantization of expert weights, as tools/routing-trace's
codebook.py found it to work: rows normalised to unit RMS, a Hadamard
rotation of the input dimension, vectors of D=4 weights replaced by the
nearest of K=1024 centroids fitted by k-means (2.5 bits per weight).

Stored form of one projection W [out, in]:
  cb     [K, D] f16     one codebook per (layer, projection)
  codes  [out, in*5/16] uint8 bitstream, 10 bits per group of D weights
  scale  [out] f16      RMS of the row
  W·H[r, 4g:4g+4] ~= scale[r] * cb[index(r, g)]
"""

import mlx.core as mx
import numpy as np

D, K, BITS = 4, 1024, 10
SAMPLE = 1 << 20
ITERS = 10
CHUNK = 1 << 13


def assign(x: mx.array, c: mx.array) -> mx.array:
    """Index of the nearest centroid for each row of x."""
    cn = (c * c).sum(axis=1)
    out = []
    for i in range(0, x.shape[0], CHUNK):
        xb = x[i:i + CHUNK]
        out.append(mx.argmin(cn[None, :] - 2 * (xb @ c.T), axis=1).astype(mx.uint32))
        mx.eval(out[-1])
    return mx.concatenate(out)


def kmeans(x: mx.array, k: int = K) -> mx.array:
    n, d = x.shape
    sample = x[mx.random.randint(0, n, (min(SAMPLE, n),))]
    c = sample[mx.random.permutation(sample.shape[0])[:k]]
    for _ in range(ITERS):
        idx = assign(sample, c)
        sums = mx.zeros((k, d)).at[idx].add(sample)
        cnt = mx.zeros((k,)).at[idx].add(mx.ones(idx.shape))
        c = mx.where(cnt[:, None] > 0, sums / mx.maximum(cnt, 1)[:, None], c)
        mx.eval(c)
    return c


def normalise(w: mx.array) -> tuple[mx.array, mx.array]:
    """Rotated rows of unit RMS, and the scales: w [.., out, in] f32."""
    w = mx.hadamard_transform(w.astype(mx.float32))
    scale = mx.sqrt((w * w).mean(axis=-1, keepdims=True)) + 1e-8
    return w / scale, scale.squeeze(-1)


def pack_codes(idx: np.ndarray) -> np.ndarray:
    """idx [rows, in/4] uint32 < 1024 -> uint8 [rows, in*5/16] bitstream."""
    rows, g = idx.shape
    assert g % 8 == 0
    i = idx.astype(np.uint64).reshape(rows, g // 8, 8)
    lo = np.zeros((rows, g // 8), dtype=np.uint64)
    for j in range(6):
        lo |= i[..., j] << np.uint64(10 * j)
    lo |= (i[..., 6] & np.uint64(0xF)) << np.uint64(60)
    hi = ((i[..., 6] >> np.uint64(4)) | (i[..., 7] << np.uint64(6))).astype(np.uint16)
    out = np.empty((rows, g // 8, 10), dtype=np.uint8)
    out[..., :8] = lo.view(np.uint8).reshape(rows, g // 8, 8)
    out[..., 8:] = hi.view(np.uint8).reshape(rows, g // 8, 2)
    return out.reshape(rows, g * 10 // 8)


def unpack_codes(codes: np.ndarray, groups: int) -> np.ndarray:
    rows = codes.shape[0]
    bits = np.unpackbits(codes, axis=1, bitorder="little")[:, : groups * BITS].reshape(rows, groups, BITS)
    return (bits.astype(np.uint32) << np.arange(BITS, dtype=np.uint32)).sum(-1)


def encode(w: mx.array, cb: mx.array) -> tuple[np.ndarray, np.ndarray]:
    """One expert's projection w [out, in] (unrotated) -> (codes uint8, scale f16)."""
    v, scale = normalise(w)
    idx = np.array(assign(v.reshape(-1, D), cb.astype(mx.float32))).reshape(w.shape[0], w.shape[1] // D)
    return pack_codes(idx), np.array(scale.astype(mx.float16))


def decode(codes: np.ndarray, scale: np.ndarray, cb: mx.array, in_dim: int) -> mx.array:
    """Float reconstruction of W·H [out, in] from the stored form."""
    idx = unpack_codes(codes, in_dim // D)
    w = cb.astype(mx.float32)[mx.array(idx)].reshape(codes.shape[0], in_dim)
    return w * mx.array(scale.astype(np.float32))[:, None]
