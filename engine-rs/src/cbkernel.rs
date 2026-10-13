//! Metal kernels for codebook-quantized experts (engine/cbq.py's format), one
//! token per expert row (same kernels as tools/routing-trace/cbkernel.py):
//!
//!   y[b, r] = scale[e_b, r] * sum_g cb[index(e_b, r, g)] . x[b, 4g:4g+4]
//!
//! x must already be rotated (`hadamard`), since the codebook was fitted on
//! W·H. One simdgroup per output row, 8 rows per threadgroup, the codebook in
//! threadgroup memory as half4; each lane decodes 4 groups (5 bytes) per
//! iteration with float4 loads of x. `gate_up` does both projections in one
//! pass and applies silu(gate) * up.

use std::collections::HashMap;
use std::ffi::CString;

use mlx_rs::{Array, Dtype, Stream};

use crate::experts::Res;

pub const D: usize = 4;
pub const K: usize = 1024;
const ROWS_PER_TG: usize = 8;

pub struct Kernels {
    single: HashMap<(usize, usize), mlx_sys::mlx_fast_metal_kernel>,
    fused: HashMap<(usize, usize), mlx_sys::mlx_fast_metal_kernel>,
}

/// Body shared by both kernels: one lane decodes 4 groups (5 bytes of
/// indices, 16 weights) per iteration, with float4 loads of x.
fn decode_body(in_dim: usize, out_dim: usize) -> String {
    format!(
        "const uint OUT = {out_dim}, ROWB = {rowb}, UNITS = {units};
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup, tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x * {rows} + sg;
    uint b = threadgroup_position_in_grid.y;
    const device float4* xr = (const device float4*)(x + (size_t)(XROWS > 1 ? b : 0) * {in_dim});",
        rowb = in_dim * 5 / 16,
        units = in_dim / D / 4,
        rows = ROWS_PER_TG,
    )
}

fn new_kernel(name: &str, inputs: &[&str], src: &str) -> mlx_sys::mlx_fast_metal_kernel {
    let name = CString::new(name).unwrap();
    let src = CString::new(src).unwrap();
    let header = CString::new("").unwrap();
    let inputs: Vec<CString> = inputs.iter().map(|s| CString::new(*s).unwrap()).collect();
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
}

/// Runs a kernel on `inputs`, output [b, out_dim] of `out_dtype`.
fn apply(kernel: mlx_sys::mlx_fast_metal_kernel, inputs: &[&Array], b: i32, out_dim: usize, x_rows: i32, out_dtype: Dtype) -> Res<Array> {
    let shape = [b, out_dim as i32];
    let grid_x = (out_dim.div_ceil(ROWS_PER_TG) * 32 * ROWS_PER_TG) as i32;
    let tname = CString::new("T").unwrap();
    let xname = CString::new("XROWS").unwrap();
    let stream = Stream::thread_local_or_default();
    unsafe {
        let cfg = mlx_sys::mlx_fast_metal_kernel_config_new();
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(cfg, shape.as_ptr(), shape.len(), out_dtype.into());
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, tname.as_ptr(), out_dtype.into());
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(cfg, xname.as_ptr(), x_rows);
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(cfg, grid_x, b, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(cfg, (32 * ROWS_PER_TG) as i32, 1, 1);
        let ins: Vec<mlx_sys::mlx_array> = inputs.iter().map(|a| a.as_ptr()).collect();
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

impl Kernels {
    pub fn new() -> Self {
        Self { single: HashMap::new(), fused: HashMap::new() }
    }

    /// One projection: x [1 or B, in] float32 (rotated), eids [B] uint32 slot
    /// ids, codes [slots, out, in*5/16] uint8, scales [slots, out] f16, cb
    /// [K, D] f16. Returns [B, out] in `out_dtype`.
    pub fn qmv(&mut self, x: &Array, eids: &Array, codes: &Array, scales: &Array, cb: &Array, out_dtype: Dtype) -> Res<Array> {
        let (x_rows, in_dim) = (x.shape()[0], x.shape()[1] as usize);
        let out_dim = codes.shape()[1] as usize;
        let kernel = *self.single.entry((in_dim, out_dim)).or_insert_with(|| {
            let src = format!(
                r#"
    {body}
    threadgroup half4 cbs[{k}];
    for (uint i = tid; i < {k}; i += threads_per_threadgroup.x) cbs[i] = ((device const half4*)cb)[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= OUT) return;
    uint e = eids[b];
    const device uchar* c = codes + ((size_t)e * OUT + row) * ROWB;
    float acc = 0.0f;
    for (uint u = lane; u < UNITS; u += 32) {{
        const device uchar* p = c + u * 5;
        uint b0 = p[0], b1 = p[1], b2 = p[2], b3 = p[3], b4 = p[4];
        const device float4* xv = xr + u * 4;
        acc += dot(float4(cbs[(b0 | (b1 << 8)) & 1023u]), xv[0]) + dot(float4(cbs[((b1 >> 2) | (b2 << 6)) & 1023u]), xv[1])
             + dot(float4(cbs[((b2 >> 4) | (b3 << 4)) & 1023u]), xv[2]) + dot(float4(cbs[((b3 >> 6) | (b4 << 2)) & 1023u]), xv[3]);
    }}
    acc = simd_sum(acc);
    if (lane == 0) y[(size_t)b * OUT + row] = (T)(acc * (float)scales[(size_t)e * OUT + row]);
"#,
                body = decode_body(in_dim, out_dim),
                k = K,
            );
            new_kernel(&format!("cb_qmv_{in_dim}_{out_dim}"), &["x", "eids", "codes", "scales", "cb"], &src)
        });
        apply(kernel, &[x, eids, codes, scales, cb], eids.shape()[0], out_dim, x_rows, out_dtype)
    }

    /// gate and up projections of the same rows in one pass, returning
    /// silu(gate) * up as float32 [B, out].
    #[allow(clippy::too_many_arguments)]
    pub fn gate_up(&mut self, x: &Array, eids: &Array, codes_g: &Array, codes_u: &Array, sc_g: &Array, sc_u: &Array, cb_g: &Array, cb_u: &Array) -> Res<Array> {
        let (x_rows, in_dim) = (x.shape()[0], x.shape()[1] as usize);
        let out_dim = codes_g.shape()[1] as usize;
        let kernel = *self.fused.entry((in_dim, out_dim)).or_insert_with(|| {
            let src = format!(
                r#"
    {body}
    threadgroup half4 cg[{k}]; threadgroup half4 cu[{k}];
    for (uint i = tid; i < {k}; i += threads_per_threadgroup.x) {{ cg[i] = ((device const half4*)cb_gate)[i]; cu[i] = ((device const half4*)cb_up)[i]; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= OUT) return;
    uint e = eids[b];
    const device uchar* pg = codes_gate + ((size_t)e * OUT + row) * ROWB;
    const device uchar* pu = codes_up + ((size_t)e * OUT + row) * ROWB;
    float ag = 0.0f, au = 0.0f;
    for (uint u = lane; u < UNITS; u += 32) {{
        const device float4* xv = xr + u * 4;
        float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
        const device uchar* p = pg + u * 5;
        uint b0 = p[0], b1 = p[1], b2 = p[2], b3 = p[3], b4 = p[4];
        ag += dot(float4(cg[(b0 | (b1 << 8)) & 1023u]), x0) + dot(float4(cg[((b1 >> 2) | (b2 << 6)) & 1023u]), x1)
            + dot(float4(cg[((b2 >> 4) | (b3 << 4)) & 1023u]), x2) + dot(float4(cg[((b3 >> 6) | (b4 << 2)) & 1023u]), x3);
        p = pu + u * 5;
        b0 = p[0]; b1 = p[1]; b2 = p[2]; b3 = p[3]; b4 = p[4];
        au += dot(float4(cu[(b0 | (b1 << 8)) & 1023u]), x0) + dot(float4(cu[((b1 >> 2) | (b2 << 6)) & 1023u]), x1)
            + dot(float4(cu[((b2 >> 4) | (b3 << 4)) & 1023u]), x2) + dot(float4(cu[((b3 >> 6) | (b4 << 2)) & 1023u]), x3);
    }}
    ag = simd_sum(ag); au = simd_sum(au);
    if (lane == 0) {{
        float g = ag * (float)sc_gate[(size_t)e * OUT + row];
        float v = au * (float)sc_up[(size_t)e * OUT + row];
        y[(size_t)b * OUT + row] = (T)(g / (1.0f + exp(-g)) * v);
    }}
"#,
                body = decode_body(in_dim, out_dim),
                k = K,
            );
            new_kernel(&format!("cb_gate_up_{in_dim}_{out_dim}"), &["x", "eids", "codes_gate", "codes_up", "sc_gate", "sc_up", "cb_gate", "cb_up"], &src)
        });
        apply(kernel, &[x, eids, codes_g, codes_u, sc_g, sc_u, cb_g, cb_u], eids.shape()[0], out_dim, x_rows, Dtype::Float32)
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
