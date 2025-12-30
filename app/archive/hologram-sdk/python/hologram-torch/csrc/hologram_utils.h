#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <string>
#include <vector>
#include "hologram_storage.h"

namespace hologram {

/**
 * Hologram FFI function declarations
 *
 * These are C-compatible functions from hologram-ffi (Rust library).
 * All operations use handle-based API for safety across FFI boundary.
 */
extern "C" {
    // Executor management
    uint64_t hologram_new_executor();
    uint64_t hologram_new_executor_with_backend(const char* backend);
    void hologram_executor_cleanup(uint64_t handle);

    // Buffer management
    uint64_t hologram_executor_allocate_buffer(uint64_t exec_handle, uint32_t len);
    void hologram_buffer_cleanup(uint64_t buf_handle);
    uint32_t hologram_buffer_length(uint64_t buf_handle);
    uint64_t hologram_buffer_as_ptr(uint64_t exec_handle, uint64_t buf_handle);
    uint64_t hologram_buffer_as_mut_ptr(uint64_t exec_handle, uint64_t buf_handle);

    // Math operations (f32)
    void hologram_vector_add_f32(uint64_t exec, uint64_t a, uint64_t b, uint64_t c, uint32_t n);
    void hologram_vector_sub_f32(uint64_t exec, uint64_t a, uint64_t b, uint64_t c, uint32_t n);
    void hologram_vector_mul_f32(uint64_t exec, uint64_t a, uint64_t b, uint64_t c, uint32_t n);
    void hologram_vector_div_f32(uint64_t exec, uint64_t a, uint64_t b, uint64_t c, uint32_t n);
    void hologram_vector_min_f32(uint64_t exec, uint64_t a, uint64_t b, uint64_t c, uint32_t n);
    void hologram_vector_max_f32(uint64_t exec, uint64_t a, uint64_t b, uint64_t c, uint32_t n);
    void hologram_vector_abs_f32(uint64_t exec, uint64_t a, uint64_t c, uint32_t n);
    void hologram_vector_neg_f32(uint64_t exec, uint64_t a, uint64_t c, uint32_t n);
    void hologram_vector_relu_f32(uint64_t exec, uint64_t a, uint64_t c, uint32_t n);
    void hologram_vector_clip_f32(uint64_t exec, uint64_t a, uint64_t c, uint32_t n, float min, float max);
    void hologram_scalar_add_f32(uint64_t exec, uint64_t a, uint64_t c, uint32_t n, float scalar);
    void hologram_scalar_mul_f32(uint64_t exec, uint64_t a, uint64_t c, uint32_t n, float scalar);

    // Activation functions
    void hologram_sigmoid_f32(uint64_t exec, uint64_t input, uint64_t output, uint32_t n);
    void hologram_tanh_f32(uint64_t exec, uint64_t input, uint64_t output, uint32_t n);
    void hologram_gelu_f32(uint64_t exec, uint64_t input, uint64_t output, uint32_t n);
    void hologram_softmax_f32(uint64_t exec, uint64_t input, uint64_t output, uint32_t n);

    // Reduction operations (return f32 result)
    float hologram_reduce_sum_f32(uint64_t exec, uint64_t input, uint64_t output, uint32_t n);
    float hologram_reduce_min_f32(uint64_t exec, uint64_t input, uint64_t output, uint32_t n);
    float hologram_reduce_max_f32(uint64_t exec, uint64_t input, uint64_t output, uint32_t n);

    // Linear algebra
    void hologram_gemm_f32(uint64_t exec, uint64_t a, uint64_t b, uint64_t c, uint32_t m, uint32_t k, uint32_t n);
    void hologram_matvec_f32(uint64_t exec, uint64_t mat, uint64_t vec, uint64_t out, uint32_t rows, uint32_t cols);

    // Loss functions (return f32 loss value)
    float hologram_mse_loss_f32(uint64_t exec, uint64_t pred, uint64_t target, uint64_t output, uint32_t n);
    float hologram_cross_entropy_loss_f32(uint64_t exec, uint64_t logits, uint64_t targets, uint64_t output, uint32_t n);
    float hologram_binary_cross_entropy_loss_f32(uint64_t exec, uint64_t pred, uint64_t target, uint64_t output, uint32_t n);

    // Tensor operations
    uint64_t hologram_tensor_from_buffer(uint64_t buf_handle, const char* shape_json);
    void hologram_tensor_cleanup(uint64_t tensor_handle);
}

/**
 * Utility functions
 */

// Convert std::vector<int64_t> to JSON string for FFI
inline std::string sizes_to_json(const std::vector<int64_t>& sizes) {
    std::string json = "[";
    for (size_t i = 0; i < sizes.size(); ++i) {
        json += std::to_string(sizes[i]);
        if (i < sizes.size() - 1) {
            json += ",";
        }
    }
    json += "]";
    return json;
}

// Compute total number of elements from shape
inline int64_t compute_numel(const std::vector<int64_t>& sizes) {
    int64_t numel = 1;
    for (auto s : sizes) {
        numel *= s;
    }
    return numel;
}

// Compute total number of elements from IntArrayRef
inline int64_t compute_numel(at::IntArrayRef sizes) {
    int64_t numel = 1;
    for (auto s : sizes) {
        numel *= s;
    }
    return numel;
}

// Convert IntArrayRef to vector
inline std::vector<int64_t> to_vector(at::IntArrayRef arr) {
    return std::vector<int64_t>(arr.begin(), arr.end());
}

} // namespace hologram
