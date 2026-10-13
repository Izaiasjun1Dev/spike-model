//! Metal kernel for codebook-quantized experts (engine/cbq.py's format), one
//! token per expert row. Same kernel as tools/routing-trace/cbkernel.py:
//!
//!   y[b, r] = scale[e_b, r] * sum_g cb[index(e_b, r, g)] . x[b, 4g:4g+4]
//!
//! x must already be rotated (`hadamard`), since the codebook was fitted on
//! W·H. One simdgroup per output row; the codebook sits in threadgroup memory.

use std::collections::HashMap;
use std::ffi::CString;

use mlx_rs::{Array, Dtype, Stream};

use crate::experts::Res;

pub const D: usize = 4;
pub const K: usize = 1024;
const BITS: usize = 10;
const ROWS_PER_TG: usize = 4;

pub struct Kernels {
    kernels: HashMap<(usize, usize), mlx_sys::mlx_fast_metal_kernel>,
}

impl Kernels {
    pub fn new() -> Self {
        Self { kernels: HashMap::new() }
    }

    fn kernel(&mut self, in_dim: usize, out_dim: usize) -> mlx_sys::mlx_fast_metal_kernel {
        *self.kernels.entry((in_dim, out_dim)).or_insert_with(|| {
            let src = format!(
                r#"
    const uint IN = {in_dim}, OUT = {out_dim}, ROWB = {rowb}, GROUPS = {groups};
    threadgroup float cbs[{kd}];
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tid = thread_position_in_threadgroup.x;
    for (uint i = tid; i < {kd}; i += threads_per_threadgroup.x) cbs[i] = (float)cb[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = threadgroup_position_in_grid.x * {rows_per_tg} + sg;
    uint b = threadgroup_position_in_grid.y;
    if (row >= OUT) return;
    uint e = eids[b];
    const device uchar* c = codes + ((size_t)e * OUT + row) * ROWB;
    const device float* xr = x + (size_t)b * IN;
    float acc = 0.0f;
    for (uint g = lane; g < GROUPS; g += 32) {{
        uint bit = g * {bits};
        uint byte = bit >> 3;
        uint idx = ((uint)c[byte] | ((uint)c[byte + 1] << 8)) >> (bit & 7);
        idx &= {kmask};
        threadgroup const float* v = cbs + idx * {d};
        uint xi = g * {d};
        acc += v[0] * xr[xi] + v[1] * xr[xi + 1] + v[2] * xr[xi + 2] + v[3] * xr[xi + 3];
    }}
    acc = simd_sum(acc);
    if (lane == 0) y[(size_t)b * OUT + row] = (T)(acc * (float)scales[(size_t)e * OUT + row]);
"#,
                rowb = in_dim * 5 / 16,
                groups = in_dim / D,
                kd = K * D,
                rows_per_tg = ROWS_PER_TG,
                bits = BITS,
                kmask = K - 1,
                d = D,
            );
            let name = CString::new(format!("cb_qmv_{in_dim}_{out_dim}")).unwrap();
            let src = CString::new(src).unwrap();
            let header = CString::new("").unwrap();
            let inputs: Vec<CString> = ["x", "eids", "codes", "scales", "cb"].iter().map(|s| CString::new(*s).unwrap()).collect();
            let outputs = [CString::new("y").unwrap()];
            let mut in_ptrs: Vec<*const std::os::raw::c_char> = inputs.iter().map(|s| s.as_ptr()).collect();
            let mut out_ptrs: Vec<*const std::os::raw::c_char> = outputs.iter().map(|s| s.as_ptr()).collect();
            unsafe {
                let in_names = mlx_sys::mlx_vector_string_new_data(in_ptrs.as_mut_ptr(), in_ptrs.len());
                let out_names = mlx_sys::mlx_vector_string_new_data(out_ptrs.as_mut_ptr(), out_ptrs.len());
                let k = mlx_sys::mlx_fast_metal_kernel_new(name.as_ptr(), in_names, out_names, src.as_ptr(), header.as_ptr(), true, false);
                mlx_sys::mlx_vector_string_free(in_names);
                mlx_sys::mlx_vector_string_free(out_names);
                k
            }
        })
    }

    /// x [B, in] float32 (rotated), eids [B] uint32 slot ids, codes
    /// [slots, out, in*5/16] uint8, scales [slots, out] f16, cb [K, D] f16.
    /// Returns [B, out] in `out_dtype`.
    pub fn qmv(&mut self, x: &Array, eids: &Array, codes: &Array, scales: &Array, cb: &Array, out_dtype: Dtype) -> Res<Array> {
        let (b, in_dim) = (x.shape()[0], x.shape()[1] as usize);
        let out_dim = codes.shape()[1] as usize;
        let kernel = self.kernel(in_dim, out_dim);
        let shape = [b, out_dim as i32];
        let grid_x = (out_dim.div_ceil(ROWS_PER_TG) * 32 * ROWS_PER_TG) as i32;
        let tname = CString::new("T").unwrap();
        let stream = Stream::thread_local_or_default();
        unsafe {
            let cfg = mlx_sys::mlx_fast_metal_kernel_config_new();
            mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(cfg, shape.as_ptr(), shape.len(), out_dtype.into());
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, tname.as_ptr(), out_dtype.into());
            mlx_sys::mlx_fast_metal_kernel_config_set_grid(cfg, grid_x, b, 1);
            mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(cfg, (32 * ROWS_PER_TG) as i32, 1, 1);
            let ins = [x.as_ptr(), eids.as_ptr(), codes.as_ptr(), scales.as_ptr(), cb.as_ptr()];
            let inputs = mlx_sys::mlx_vector_array_new_data(ins.as_ptr(), ins.len());
            let mut outs = mlx_sys::mlx_vector_array_new();
            let status = mlx_sys::mlx_fast_metal_kernel_apply(&mut outs, kernel, inputs, cfg, stream.as_ptr());
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_fast_metal_kernel_config_free(cfg);
            if status != 0 {
                mlx_sys::mlx_vector_array_free(outs);
                return Err("codebook kernel failed".into());
            }
            let mut res = mlx_sys::mlx_array_new();
            mlx_sys::mlx_vector_array_get(&mut res, outs, 0);
            mlx_sys::mlx_vector_array_free(outs);
            Ok(Array::from_ptr(res))
        }
    }
}

/// Orthonormal Walsh-Hadamard transform along the last axis (MLX's default
/// scale, the one engine/cbq.py used on the weights).
pub fn hadamard(x: &Array) -> Res<Array> {
    let stream = Stream::thread_local_or_default();
    unsafe {
        let mut res = mlx_sys::mlx_array_new();
        let scale = mlx_sys::mlx_optional_float { value: 0.0, has_value: false };
        if mlx_sys::mlx_hadamard_transform(&mut res, x.as_ptr(), scale, stream.as_ptr()) != 0 {
            return Err("hadamard_transform failed".into());
        }
        Ok(Array::from_ptr(res))
    }
}
