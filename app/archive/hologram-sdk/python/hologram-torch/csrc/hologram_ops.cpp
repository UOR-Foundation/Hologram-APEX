/**
 * Hologram Operations - Forward Pass Implementations
 *
 * This file implements all PyTorch operations for Hologram backend by
 * delegating to hologram-ffi functions. No reimplementation needed!
 */

#include <torch/extension.h>
#include <ATen/ATen.h>

#include "hologram_storage.h"
#include "hologram_utils.h"

namespace hologram {

// Forward declarations
void register_tensor_storage(void* data_ptr, std::unique_ptr<HologramStorage> storage);
HologramStorage* get_tensor_storage(void* data_ptr);
at::Tensor allocate_hologram_tensor(at::IntArrayRef sizes, const at::TensorOptions& options);

/**
 * Helper: Get storage from tensor
 */
HologramStorage* get_storage(const at::Tensor& tensor) {
    return get_tensor_storage(tensor.data_ptr());
}

/**
 * Helper: Create output tensor with same shape as input
 */
at::Tensor create_output_like(const at::Tensor& input) {
    return allocate_hologram_tensor(
        input.sizes(),
        input.options()
    );
}

//==============================================================================
// Element-wise Binary Operations
//==============================================================================

at::Tensor add_hologram(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match for addition");

    auto* a_storage = get_storage(a);
    auto* b_storage = get_storage(b);

    auto c = create_output_like(a);
    auto* c_storage = get_storage(c);

    // TODO: Handle alpha scaling
    hologram_vector_add_f32(
        a_storage->executor_handle,
        a_storage->buffer_handle,
        b_storage->buffer_handle,
        c_storage->buffer_handle,
        static_cast<uint32_t>(a.numel())
    );

    return c;
}

at::Tensor sub_hologram(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match for subtraction");

    auto* a_storage = get_storage(a);
    auto* b_storage = get_storage(b);

    auto c = create_output_like(a);
    auto* c_storage = get_storage(c);

    hologram_vector_sub_f32(
        a_storage->executor_handle,
        a_storage->buffer_handle,
        b_storage->buffer_handle,
        c_storage->buffer_handle,
        static_cast<uint32_t>(a.numel())
    );

    return c;
}

at::Tensor mul_hologram(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match for multiplication");

    auto* a_storage = get_storage(a);
    auto* b_storage = get_storage(b);

    auto c = create_output_like(a);
    auto* c_storage = get_storage(c);

    hologram_vector_mul_f32(
        a_storage->executor_handle,
        a_storage->buffer_handle,
        b_storage->buffer_handle,
        c_storage->buffer_handle,
        static_cast<uint32_t>(a.numel())
    );

    return c;
}

at::Tensor div_hologram(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match for division");

    auto* a_storage = get_storage(a);
    auto* b_storage = get_storage(b);

    auto c = create_output_like(a);
    auto* c_storage = get_storage(c);

    hologram_vector_div_f32(
        a_storage->executor_handle,
        a_storage->buffer_handle,
        b_storage->buffer_handle,
        c_storage->buffer_handle,
        static_cast<uint32_t>(a.numel())
    );

    return c;
}

at::Tensor min_elementwise_hologram(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match");

    auto* a_storage = get_storage(a);
    auto* b_storage = get_storage(b);

    auto c = create_output_like(a);
    auto* c_storage = get_storage(c);

    hologram_vector_min_f32(
        a_storage->executor_handle,
        a_storage->buffer_handle,
        b_storage->buffer_handle,
        c_storage->buffer_handle,
        static_cast<uint32_t>(a.numel())
    );

    return c;
}

at::Tensor max_elementwise_hologram(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match");

    auto* a_storage = get_storage(a);
    auto* b_storage = get_storage(b);

    auto c = create_output_like(a);
    auto* c_storage = get_storage(c);

    hologram_vector_max_f32(
        a_storage->executor_handle,
        a_storage->buffer_handle,
        b_storage->buffer_handle,
        c_storage->buffer_handle,
        static_cast<uint32_t>(a.numel())
    );

    return c;
}

//==============================================================================
// Element-wise Unary Operations
//==============================================================================

at::Tensor abs_hologram(const at::Tensor& input) {
    auto* input_storage = get_storage(input);

    auto output = create_output_like(input);
    auto* output_storage = get_storage(output);

    hologram_vector_abs_f32(
        input_storage->executor_handle,
        input_storage->buffer_handle,
        output_storage->buffer_handle,
        static_cast<uint32_t>(input.numel())
    );

    return output;
}

at::Tensor neg_hologram(const at::Tensor& input) {
    auto* input_storage = get_storage(input);

    auto output = create_output_like(input);
    auto* output_storage = get_storage(output);

    hologram_vector_neg_f32(
        input_storage->executor_handle,
        input_storage->buffer_handle,
        output_storage->buffer_handle,
        static_cast<uint32_t>(input.numel())
    );

    return output;
}

at::Tensor relu_hologram(const at::Tensor& input) {
    auto* input_storage = get_storage(input);

    auto output = create_output_like(input);
    auto* output_storage = get_storage(output);

    hologram_vector_relu_f32(
        input_storage->executor_handle,
        input_storage->buffer_handle,
        output_storage->buffer_handle,
        static_cast<uint32_t>(input.numel())
    );

    return output;
}

at::Tensor clamp_hologram(const at::Tensor& input, const std::optional<at::Scalar>& min, const std::optional<at::Scalar>& max) {
    auto* input_storage = get_storage(input);

    auto output = create_output_like(input);
    auto* output_storage = get_storage(output);

    float min_val = min.has_value() ? min.value().toFloat() : -std::numeric_limits<float>::infinity();
    float max_val = max.has_value() ? max.value().toFloat() : std::numeric_limits<float>::infinity();

    hologram_vector_clip_f32(
        input_storage->executor_handle,
        input_storage->buffer_handle,
        output_storage->buffer_handle,
        static_cast<uint32_t>(input.numel()),
        min_val,
        max_val
    );

    return output;
}

//==============================================================================
// Scalar Operations
//==============================================================================

at::Tensor add_scalar_hologram(const at::Tensor& input, const at::Scalar& scalar, const at::Scalar& alpha) {
    auto* input_storage = get_storage(input);

    auto output = create_output_like(input);
    auto* output_storage = get_storage(output);

    float value = scalar.toFloat();
    // TODO: Handle alpha scaling

    hologram_scalar_add_f32(
        input_storage->executor_handle,
        input_storage->buffer_handle,
        output_storage->buffer_handle,
        static_cast<uint32_t>(input.numel()),
        value
    );

    return output;
}

at::Tensor mul_scalar_hologram(const at::Tensor& input, const at::Scalar& scalar) {
    auto* input_storage = get_storage(input);

    auto output = create_output_like(input);
    auto* output_storage = get_storage(output);

    float value = scalar.toFloat();

    hologram_scalar_mul_f32(
        input_storage->executor_handle,
        input_storage->buffer_handle,
        output_storage->buffer_handle,
        static_cast<uint32_t>(input.numel()),
        value
    );

    return output;
}

//==============================================================================
// Activation Functions
//==============================================================================

at::Tensor sigmoid_hologram(const at::Tensor& input) {
    auto* input_storage = get_storage(input);

    auto output = create_output_like(input);
    auto* output_storage = get_storage(output);

    hologram_sigmoid_f32(
        input_storage->executor_handle,
        input_storage->buffer_handle,
        output_storage->buffer_handle,
        static_cast<uint32_t>(input.numel())
    );

    return output;
}

at::Tensor tanh_hologram(const at::Tensor& input) {
    auto* input_storage = get_storage(input);

    auto output = create_output_like(input);
    auto* output_storage = get_storage(output);

    hologram_tanh_f32(
        input_storage->executor_handle,
        input_storage->buffer_handle,
        output_storage->buffer_handle,
        static_cast<uint32_t>(input.numel())
    );

    return output;
}

at::Tensor gelu_hologram(const at::Tensor& input, c10::string_view approximate) {
    auto* input_storage = get_storage(input);

    auto output = create_output_like(input);
    auto* output_storage = get_storage(output);

    hologram_gelu_f32(
        input_storage->executor_handle,
        input_storage->buffer_handle,
        output_storage->buffer_handle,
        static_cast<uint32_t>(input.numel())
    );

    return output;
}

at::Tensor softmax_hologram(const at::Tensor& input, int64_t dim, bool half_to_float) {
    // TODO: Handle multi-dimensional softmax properly
    // For now, assume 1D or flattened input
    auto* input_storage = get_storage(input);

    auto output = create_output_like(input);
    auto* output_storage = get_storage(output);

    hologram_softmax_f32(
        input_storage->executor_handle,
        input_storage->buffer_handle,
        output_storage->buffer_handle,
        static_cast<uint32_t>(input.numel())
    );

    return output;
}

//==============================================================================
// Reduction Operations
//==============================================================================

at::Tensor sum_hologram(const at::Tensor& input, std::optional<at::ScalarType> dtype) {
    auto* input_storage = get_storage(input);

    // Reductions need 3-element output buffer for temporaries
    auto output = allocate_hologram_tensor({3}, input.options());
    auto* output_storage = get_storage(output);

    hologram_reduce_sum_f32(
        input_storage->executor_handle,
        input_storage->buffer_handle,
        output_storage->buffer_handle,
        static_cast<uint32_t>(input.numel())
    );

    // Return scalar (first element)
    return output.select(0, 0);
}

at::Tensor min_reduction_hologram(const at::Tensor& input) {
    auto* input_storage = get_storage(input);

    // Reductions need 3-element output buffer
    auto output = allocate_hologram_tensor({3}, input.options());
    auto* output_storage = get_storage(output);

    hologram_reduce_min_f32(
        input_storage->executor_handle,
        input_storage->buffer_handle,
        output_storage->buffer_handle,
        static_cast<uint32_t>(input.numel())
    );

    return output.select(0, 0);
}

at::Tensor max_reduction_hologram(const at::Tensor& input) {
    auto* input_storage = get_storage(input);

    // Reductions need 3-element output buffer
    auto output = allocate_hologram_tensor({3}, input.options());
    auto* output_storage = get_storage(output);

    hologram_reduce_max_f32(
        input_storage->executor_handle,
        input_storage->buffer_handle,
        output_storage->buffer_handle,
        static_cast<uint32_t>(input.numel())
    );

    return output.select(0, 0);
}

at::Tensor mean_hologram(const at::Tensor& input, std::optional<at::ScalarType> dtype) {
    // mean = sum / numel
    auto sum_result = sum_hologram(input, dtype);
    return sum_result / static_cast<float>(input.numel());
}

//==============================================================================
// Linear Algebra
//==============================================================================

at::Tensor matmul_hologram(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "matmul currently only supports 2D matrices");
    TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimensions incompatible for multiplication");

    uint32_t m = static_cast<uint32_t>(a.size(0));
    uint32_t k = static_cast<uint32_t>(a.size(1));
    uint32_t n = static_cast<uint32_t>(b.size(1));

    auto* a_storage = get_storage(a);
    auto* b_storage = get_storage(b);

    auto c = allocate_hologram_tensor({m, n}, a.options());
    auto* c_storage = get_storage(c);

    hologram_gemm_f32(
        a_storage->executor_handle,
        a_storage->buffer_handle,
        b_storage->buffer_handle,
        c_storage->buffer_handle,
        m, k, n
    );

    return c;
}

//==============================================================================
// Loss Functions
//==============================================================================

at::Tensor mse_loss_hologram(
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction
) {
    TORCH_CHECK(input.sizes() == target.sizes(), "Input and target must have same shape");

    auto* input_storage = get_storage(input);
    auto* target_storage = get_storage(target);

    // Loss functions need 3-element output buffer
    auto output = allocate_hologram_tensor({3}, input.options());
    auto* output_storage = get_storage(output);

    hologram_mse_loss_f32(
        input_storage->executor_handle,
        input_storage->buffer_handle,
        target_storage->buffer_handle,
        output_storage->buffer_handle,
        static_cast<uint32_t>(input.numel())
    );

    // Return scalar (first element)
    auto loss = output.select(0, 0);

    // Handle reduction
    if (reduction == at::Reduction::Mean) {
        return loss / static_cast<float>(input.numel());
    } else if (reduction == at::Reduction::Sum) {
        return loss;
    } else {
        // No reduction
        return loss;
    }
}

//==============================================================================
// Register Operations with PyTorch
//==============================================================================

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // Element-wise binary operations
    m.impl("add.Tensor", add_hologram);
    m.impl("sub.Tensor", sub_hologram);
    m.impl("mul.Tensor", mul_hologram);
    m.impl("div.Tensor", div_hologram);
    m.impl("minimum", min_elementwise_hologram);
    m.impl("maximum", max_elementwise_hologram);

    // Element-wise unary operations
    m.impl("abs", abs_hologram);
    m.impl("neg", neg_hologram);
    m.impl("relu", relu_hologram);
    m.impl("clamp", clamp_hologram);

    // Scalar operations
    m.impl("add.Scalar", add_scalar_hologram);
    m.impl("mul.Scalar", mul_scalar_hologram);

    // Activation functions
    m.impl("sigmoid", sigmoid_hologram);
    m.impl("tanh", tanh_hologram);
    m.impl("gelu", gelu_hologram);
    m.impl("_softmax", softmax_hologram);

    // Reductions
    m.impl("sum", sum_hologram);
    m.impl("min", min_reduction_hologram);
    m.impl("max", max_reduction_hologram);
    m.impl("mean", mean_hologram);

    // Linear algebra
    m.impl("matmul", matmul_hologram);

    // Loss functions
    m.impl("mse_loss", mse_loss_hologram);
}

} // namespace hologram
