"""Metal kernel for codebook-quantized experts (decode: one token per expert).

Format of one projection W [out, in] (codebook.py, "vq d4 k1024 had"):

  cb      [K, d] f16      one codebook per (layer, projection), shared by
                          every expert of the layer; d = 4, K = 1024
  codes   [out, in*5/16]  uint8 bitstream, 10 bits per group of d weights
  scale   [out] f16       RMS of each row (the row was normalised before
                          the codebook was fitted)

  W[r, 4g:4g+4] ~= scale[r] * cb[index(r, g)]

With the Hadamard rotation, the stored W is W·H and the kernel must get
x' = hadamard_transform(x) (H orthonormal, so W·H·H·x = W·x).

gather_cb_qmv(x', eids, codes, scales, cb): x' [B, in], one expert id per
row; codes/scales carry a leading expert axis. Returns [B, out] in x's dtype.
One simdgroup per output row; the codebook sits in threadgroup memory.

Usage: uv run cbkernel.py   (correctness against a float reference, and
timing against gather_qmm q4 for the 235B's expert shapes)
"""

import time

import mlx.core as mx
import numpy as np

D, K = 4, 1024
BITS = 10
ROWS_PER_TG = 4  # simdgroups per threadgroup = rows per threadgroup


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


def encode(w: mx.array, cb: mx.array) -> tuple[mx.array, mx.array]:
    """w [E, out, in] (already rotated if the codebook was fitted on W·H) ->
    (codes uint8 [E, out, in*5/16], scale f16 [E, out]) for a fitted cb."""
    from codebook import assign

    w = w.astype(mx.float32)
    scale = mx.sqrt((w * w).mean(axis=-1, keepdims=True)) + 1e-8
    vecs = (w / scale).reshape(-1, D)
    idx = np.array(assign(vecs, cb.astype(mx.float32))).reshape(w.shape[0], w.shape[1], w.shape[2] // D)
    codes = np.stack([pack_codes(e) for e in idx])
    return mx.array(codes), scale.squeeze(-1).astype(mx.float16)


def decode(codes: mx.array, scale: mx.array, cb: mx.array, in_dim: int) -> mx.array:
    """Float reference of the stored weights, [E, out, in]."""
    idx = np.stack([unpack_codes(np.array(c), in_dim // D) for c in codes])
    w = cb.astype(mx.float32)[mx.array(idx)].reshape(codes.shape[0], codes.shape[1], in_dim)
    return w * scale.astype(mx.float32)[..., None]


_kernels = {}


def _kernel(in_dim: int, out_dim: int):
    key = (in_dim, out_dim)
    if key not in _kernels:
        src = f"""
    const uint IN = {in_dim}, OUT = {out_dim}, ROWB = {in_dim * 5 // 16}, GROUPS = {in_dim // D};
    threadgroup float cbs[{K * D}];
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tid = thread_position_in_threadgroup.x;
    for (uint i = tid; i < {K * D}; i += threads_per_threadgroup.x) cbs[i] = (float)cb[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = threadgroup_position_in_grid.x * {ROWS_PER_TG} + sg;
    uint b = threadgroup_position_in_grid.y;
    if (row >= OUT) return;
    uint e = eids[b];
    const device uchar* c = codes + ((size_t)e * OUT + row) * ROWB;
    const device T* xr = x + (size_t)b * IN;
    float acc = 0.0f;
    for (uint g = lane; g < GROUPS; g += 32) {{
        uint bit = g * {BITS};
        uint byte = bit >> 3;
        uint idx = ((uint)c[byte] | ((uint)c[byte + 1] << 8)) >> (bit & 7);
        idx &= {K - 1};
        threadgroup const float* v = cbs + idx * {D};
        uint xi = g * {D};
        acc += v[0] * (float)xr[xi] + v[1] * (float)xr[xi + 1] + v[2] * (float)xr[xi + 2] + v[3] * (float)xr[xi + 3];
    }}
    acc = simd_sum(acc);
    if (lane == 0) y[(size_t)b * OUT + row] = (T)(acc * (float)scales[(size_t)e * OUT + row]);
"""
        _kernels[key] = mx.fast.metal_kernel(
            name=f"cb_qmv_{in_dim}_{out_dim}",
            input_names=["x", "eids", "codes", "scales", "cb"],
            output_names=["y"],
            source=src,
        )
    return _kernels[key]


def gather_cb_qmv(x: mx.array, eids: mx.array, codes: mx.array, scales: mx.array, cb: mx.array) -> mx.array:
    """x [B, in] (rotated), eids [B] uint32 -> [B, out]."""
    B, in_dim = x.shape
    out_dim = codes.shape[1]
    k = _kernel(in_dim, out_dim)
    return k(
        inputs=[x, eids, codes, scales, cb],
        template=[("T", x.dtype)],
        grid=(-(-out_dim // ROWS_PER_TG) * 32 * ROWS_PER_TG, B, 1),
        threadgroup=(32 * ROWS_PER_TG, 1, 1),
        output_shapes=[(B, out_dim)],
        output_dtypes=[x.dtype],
    )[0]


def main():
    from codebook import kmeans

    mx.random.seed(0)
    for name, (in_dim, out_dim) in {"up/gate": (4096, 1536), "down": (1536, 4096)}.items():
        E, B = 16, 8
        w = mx.random.normal((E, out_dim, in_dim)) * 0.02
        wr = mx.hadamard_transform(w)
        scale = mx.sqrt((wr * wr).mean(axis=-1, keepdims=True))
        cb = kmeans((wr / scale).reshape(-1, D), K).astype(mx.float16)
        codes, scales = encode(wr, cb)
        mx.eval(codes, scales)
        eids = mx.array(np.random.randint(0, E, B).astype(np.uint32))
        x = mx.random.normal((B, in_dim)).astype(mx.bfloat16)
        xr = mx.hadamard_transform(x.astype(mx.float32)).astype(mx.bfloat16)

        ref_w = decode(codes, scales, cb, in_dim)  # [E, out, in], W·H reconstructed
        ref = mx.stack([ref_w[int(e)] @ xr[i].astype(mx.float32) for i, e in enumerate(np.array(eids))])
        got = gather_cb_qmv(xr, eids, codes, scales, cb).astype(mx.float32)
        err = (mx.abs(got - ref).max() / mx.abs(ref).max()).item()
        rel = (mx.linalg.norm((ref_w - wr).flatten()) / mx.linalg.norm(wr.flatten())).item()
        print(f"{name}: kernel vs float reference, max abs err / max |ref| = {err:.2e}; "
              f"codebook weight error {rel:.3f}; codes {codes.nbytes / E / 2**20:.2f} MiB per expert")

        # timing against the q4 kernel, 8 experts, one token each (decode)
        wq, sq, bq = mx.quantize(w.astype(mx.bfloat16), group_size=64, bits=4)
        xq = x[:, None, None, :]
        def q4():
            return mx.gather_qmm(xq, wq, sq, bq, rhs_indices=eids[:, None, None], transpose=True, group_size=64, bits=4)
        def cbk():
            return gather_cb_qmv(xr, eids, codes, scales, cb)
        for label, f in (("q4 gather_qmm", q4), ("codebook kernel", cbk)):
            mx.eval(f())
            t = time.perf_counter()
            for _ in range(50):
                mx.eval(f())
            print(f"  {label}: {(time.perf_counter() - t) / 50 * 1e6:.0f} us")


if __name__ == "__main__":
    main()
