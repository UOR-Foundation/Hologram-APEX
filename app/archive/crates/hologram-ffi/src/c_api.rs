/**
 * C-Compatible FFI API
 *
 * This module provides C-compatible exports for direct linkage from C++.
 * Unlike the main UniFFI exports, these are simple C functions without
 * the UniFFI wrapper layer.
 */
use crate::*;

// Re-export all FFI functions with C linkage for direct C++ access
// This avoids the UniFFI wrapping and provides direct function calls

#[no_mangle]
pub extern "C" fn hologram_new_executor() -> u64 {
    new_executor()
}

#[no_mangle]
pub extern "C" fn hologram_new_executor_with_backend(backend: *const std::os::raw::c_char) -> u64 {
    let backend_str = unsafe { std::ffi::CStr::from_ptr(backend).to_str().unwrap_or("cpu") };
    new_executor_with_backend(backend_str.to_string())
}

#[no_mangle]
pub extern "C" fn hologram_executor_cleanup(handle: u64) {
    executor_cleanup(handle)
}

#[no_mangle]
pub extern "C" fn hologram_executor_allocate_buffer(exec: u64, len: u32) -> u64 {
    executor_allocate_buffer(exec, len)
}

#[no_mangle]
pub extern "C" fn hologram_buffer_cleanup(handle: u64) {
    buffer_cleanup(handle)
}

#[no_mangle]
pub extern "C" fn hologram_buffer_length(handle: u64) -> u32 {
    buffer_length(handle)
}

#[no_mangle]
pub extern "C" fn hologram_buffer_as_ptr(exec: u64, buf: u64) -> u64 {
    buffer_as_ptr(exec, buf)
}

#[no_mangle]
pub extern "C" fn hologram_buffer_as_mut_ptr(exec: u64, buf: u64) -> u64 {
    buffer_as_mut_ptr(exec, buf)
}

// Math operations
#[no_mangle]
pub extern "C" fn hologram_vector_add_f32(exec: u64, a: u64, b: u64, c: u64, len: u32) {
    vector_add_f32(exec, a, b, c, len)
}

#[no_mangle]
pub extern "C" fn hologram_vector_sub_f32(exec: u64, a: u64, b: u64, c: u64, len: u32) {
    vector_sub_f32(exec, a, b, c, len)
}

#[no_mangle]
pub extern "C" fn hologram_vector_mul_f32(exec: u64, a: u64, b: u64, c: u64, len: u32) {
    vector_mul_f32(exec, a, b, c, len)
}

#[no_mangle]
pub extern "C" fn hologram_vector_div_f32(exec: u64, a: u64, b: u64, c: u64, len: u32) {
    vector_div_f32(exec, a, b, c, len)
}

#[no_mangle]
pub extern "C" fn hologram_vector_min_f32(exec: u64, a: u64, b: u64, c: u64, len: u32) {
    vector_min_f32(exec, a, b, c, len)
}

#[no_mangle]
pub extern "C" fn hologram_vector_max_f32(exec: u64, a: u64, b: u64, c: u64, len: u32) {
    vector_max_f32(exec, a, b, c, len)
}

#[no_mangle]
pub extern "C" fn hologram_vector_abs_f32(exec: u64, a: u64, c: u64, len: u32) {
    vector_abs_f32(exec, a, c, len)
}

#[no_mangle]
pub extern "C" fn hologram_vector_neg_f32(exec: u64, a: u64, c: u64, len: u32) {
    vector_neg_f32(exec, a, c, len)
}

#[no_mangle]
pub extern "C" fn hologram_vector_relu_f32(exec: u64, a: u64, c: u64, len: u32) {
    vector_relu_f32(exec, a, c, len)
}

#[no_mangle]
pub extern "C" fn hologram_vector_clip_f32(exec: u64, a: u64, c: u64, len: u32, min: f32, max: f32) {
    vector_clip_f32(exec, a, c, len, min, max)
}

#[no_mangle]
pub extern "C" fn hologram_scalar_add_f32(exec: u64, a: u64, c: u64, len: u32, scalar: f32) {
    scalar_add_f32(exec, a, c, len, scalar)
}

#[no_mangle]
pub extern "C" fn hologram_scalar_mul_f32(exec: u64, a: u64, c: u64, len: u32, scalar: f32) {
    scalar_mul_f32(exec, a, c, len, scalar)
}

// Activations
#[no_mangle]
pub extern "C" fn hologram_sigmoid_f32(exec: u64, input: u64, output: u64, len: u32) {
    sigmoid_f32(exec, input, output, len)
}

#[no_mangle]
pub extern "C" fn hologram_tanh_f32(exec: u64, input: u64, output: u64, len: u32) {
    tanh_f32(exec, input, output, len)
}

#[no_mangle]
pub extern "C" fn hologram_gelu_f32(exec: u64, input: u64, output: u64, len: u32) {
    gelu_f32(exec, input, output, len)
}

#[no_mangle]
pub extern "C" fn hologram_softmax_f32(exec: u64, input: u64, output: u64, len: u32) {
    softmax_f32(exec, input, output, len)
}

// Reductions (these return f32)
#[no_mangle]
pub extern "C" fn hologram_reduce_sum_f32(exec: u64, input: u64, output: u64, len: u32) -> f32 {
    reduce_sum_f32(exec, input, output, len)
}

#[no_mangle]
pub extern "C" fn hologram_reduce_min_f32(exec: u64, input: u64, output: u64, len: u32) -> f32 {
    reduce_min_f32(exec, input, output, len)
}

#[no_mangle]
pub extern "C" fn hologram_reduce_max_f32(exec: u64, input: u64, output: u64, len: u32) -> f32 {
    reduce_max_f32(exec, input, output, len)
}

// Linear algebra
#[no_mangle]
pub extern "C" fn hologram_gemm_f32(exec: u64, a: u64, b: u64, c: u64, m: u32, k: u32, n: u32) {
    gemm_f32(exec, a, b, c, m, k, n)
}

#[no_mangle]
pub extern "C" fn hologram_matvec_f32(exec: u64, mat: u64, vec: u64, out: u64, rows: u32, cols: u32) {
    matvec_f32(exec, mat, vec, out, rows, cols)
}

// Loss functions (these return f32)
#[no_mangle]
pub extern "C" fn hologram_mse_loss_f32(exec: u64, pred: u64, target: u64, output: u64, len: u32) -> f32 {
    mse_loss_f32(exec, pred, target, output, len)
}

#[no_mangle]
pub extern "C" fn hologram_cross_entropy_loss_f32(exec: u64, logits: u64, targets: u64, output: u64, len: u32) -> f32 {
    cross_entropy_loss_f32(exec, logits, targets, output, len)
}

#[no_mangle]
pub extern "C" fn hologram_binary_cross_entropy_loss_f32(
    exec: u64,
    pred: u64,
    target: u64,
    output: u64,
    len: u32,
) -> f32 {
    binary_cross_entropy_loss_f32(exec, pred, target, output, len)
}

// Tensor operations
#[no_mangle]
pub extern "C" fn hologram_tensor_from_buffer(buf_handle: u64, shape_json: *const std::os::raw::c_char) -> u64 {
    let shape_str = unsafe { std::ffi::CStr::from_ptr(shape_json).to_str().unwrap_or("[]") };
    tensor_from_buffer(buf_handle, shape_str.to_string())
}

#[no_mangle]
pub extern "C" fn hologram_tensor_cleanup(handle: u64) {
    tensor_cleanup(handle)
}
