/**
 * Hologram Tensor - Tensor Allocation and Management
 *
 * This file implements tensor factory functions for Hologram backend:
 * - CPU backend: Uses raw pointers from buffer_as_mut_ptr()
 * - GPU backends (Metal, CUDA): Uses DLPack protocol
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/core/TensorOptions.h>

#include <vector>
#include <memory>
#include <stdexcept>

#include "hologram_storage.h"
#include "hologram_utils.h"

namespace hologram {

// Forward declarations from hologram_backend.cpp
void register_tensor_storage(void* data_ptr, std::unique_ptr<HologramStorage> storage);
void unregister_tensor_storage(void* data_ptr);
HologramStorage* get_tensor_storage(void* data_ptr);
uint64_t get_current_executor();
std::string get_backend();

/**
 * Tensor deleter for CPU backend
 *
 * Called when PyTorch tensor is destroyed. Cleans up Hologram buffer.
 */
void cpu_tensor_deleter(void* ctx) {
    auto* storage = static_cast<HologramStorage*>(ctx);

    // Clean up Hologram buffer
    hologram_buffer_cleanup(storage->buffer_handle);

    // Unregister from global map (this will delete storage via unique_ptr)
    unregister_tensor_storage(storage->data_ptr);

    // Note: Do NOT delete storage manually - the unique_ptr in the map owns it
}

/**
 * Allocate tensor on CPU backend (raw pointer approach)
 *
 * Steps:
 * 1. Allocate Hologram buffer via FFI
 * 2. Get raw pointer via buffer_as_mut_ptr()
 * 3. Wrap pointer in PyTorch tensor with custom deleter
 * 4. Register storage in global map
 */
at::Tensor allocate_cpu_tensor(
    at::IntArrayRef sizes,
    const at::TensorOptions& options
) {
    // Get CPU executor
    uint64_t exec = get_current_executor();

    // Compute total elements
    int64_t numel = compute_numel(sizes);
    if (numel <= 0) {
        throw std::runtime_error("Cannot allocate tensor with zero elements");
    }

    // Allocate Hologram buffer
    uint64_t buf_handle = hologram_executor_allocate_buffer(exec, static_cast<uint32_t>(numel));

    // Get raw mutable pointer (CPU only)
    uint64_t raw_ptr_u64 = hologram_buffer_as_mut_ptr(exec, buf_handle);
    void* raw_ptr = reinterpret_cast<void*>(raw_ptr_u64);

    // Create storage info
    auto* storage = new HologramStorage{
        .executor_handle = exec,
        .buffer_handle = buf_handle,
        .tensor_handle = 0,
        .backend = "cpu",
        .data_ptr = raw_ptr,
        .numel = static_cast<size_t>(numel),
        .from_dlpack = false
    };

    // Register in global map first
    register_tensor_storage(raw_ptr, std::unique_ptr<HologramStorage>(storage));

    // Create deleter that captures storage pointer
    auto deleter = [storage](void* ptr) {
        cpu_tensor_deleter(storage);
    };

    // Create PyTorch tensor from raw pointer
    auto tensor = at::from_blob(
        raw_ptr,
        sizes,
        deleter,
        options.device(c10::Device(c10::DeviceType::PrivateUse1, 0))
               .dtype(at::kFloat) // f32 only for now
    );

    return tensor;
}

/**
 * Tensor deleter for GPU backends
 */
void gpu_tensor_deleter(void* ctx) {
    auto* storage = static_cast<HologramStorage*>(ctx);

    // Clean up Hologram tensor
    if (storage->tensor_handle != 0) {
        hologram_tensor_cleanup(storage->tensor_handle);
    }

    // Clean up Hologram buffer
    hologram_buffer_cleanup(storage->buffer_handle);

    // Unregister from global map (this will delete storage via unique_ptr)
    if (storage->data_ptr != nullptr) {
        unregister_tensor_storage(storage->data_ptr);
    }

    // Note: Do NOT delete storage manually - the unique_ptr in the map owns it
}

/**
 * Allocate tensor on GPU backend (DLPack approach)
 *
 * Steps:
 * 1. Allocate Hologram buffer
 * 2. Create Hologram tensor from buffer
 * 3. Export as DLPack
 * 4. Import to PyTorch via torch::from_dlpack
 * 5. Mark as PrivateUse1 device
 */
at::Tensor allocate_gpu_tensor(
    at::IntArrayRef sizes,
    const at::TensorOptions& options,
    const std::string& backend
) {
    // Get executor for backend
    uint64_t exec = get_current_executor();

    // Compute total elements
    int64_t numel = compute_numel(sizes);
    if (numel <= 0) {
        throw std::runtime_error("Cannot allocate tensor with zero elements");
    }

    // Allocate Hologram buffer
    uint64_t buf_handle = hologram_executor_allocate_buffer(exec, static_cast<uint32_t>(numel));

    // Create Hologram tensor from buffer
    std::string shape_json = sizes_to_json(to_vector(sizes));
    uint64_t tensor_handle = hologram_tensor_from_buffer(buf_handle, shape_json.c_str());

    // TODO: Export as DLPack (requires DLPack FFI functions)
    // For now, we'll use a placeholder approach
    // In practice, we need to call tensor_to_dlpack() and torch::from_dlpack()

    // PLACEHOLDER: For GPU backends, we'll fall back to CPU approach for now
    // This needs to be fixed with proper DLPack integration
    throw std::runtime_error(
        "GPU backend tensor allocation not yet implemented. "
        "DLPack export/import needed. Use CPU backend for now."
    );
}

/**
 * Main tensor allocation function
 *
 * Dispatches to CPU or GPU allocation based on current backend.
 */
at::Tensor allocate_hologram_tensor(
    at::IntArrayRef sizes,
    const at::TensorOptions& options
) {
    std::string backend = get_backend();

    if (backend == "cpu") {
        return allocate_cpu_tensor(sizes, options);
    } else {
        return allocate_gpu_tensor(sizes, options, backend);
    }
}

/**
 * Tensor factory functions
 *
 * These implement PyTorch's ATen operations for tensor creation.
 */

// empty: Create uninitialized tensor
at::Tensor empty_hologram(
    at::IntArrayRef sizes,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    std::optional<at::MemoryFormat> memory_format
) {
    auto options = at::TensorOptions()
        .dtype(dtype.value_or(at::kFloat))
        .layout(layout.value_or(at::kStrided))
        .device(device.value_or(at::Device(c10::DeviceType::PrivateUse1, 0)))
        .pinned_memory(pin_memory.value_or(false));

    return allocate_hologram_tensor(sizes, options);
}

// zeros: Create tensor filled with zeros
at::Tensor zeros_hologram(
    at::IntArrayRef sizes,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory
) {
    auto tensor = empty_hologram(sizes, dtype, layout, device, pin_memory, std::nullopt);
    tensor.zero_();
    return tensor;
}

// ones: Create tensor filled with ones
at::Tensor ones_hologram(
    at::IntArrayRef sizes,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory
) {
    auto tensor = empty_hologram(sizes, dtype, layout, device, pin_memory, std::nullopt);
    tensor.fill_(1.0);
    return tensor;
}

/**
 * Copy operations - Required for .to('hologram') to work
 */

// Copy from CPU to Hologram
at::Tensor& copy_hologram_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking
) {
    TORCH_CHECK(self.numel() == src.numel(), "Tensor sizes must match for copy");

    auto* self_storage = get_tensor_storage(self.data_ptr());

    // If source is CPU tensor, copy data directly
    if (src.device().type() == c10::DeviceType::CPU) {
        // Get source data as contiguous f32
        auto src_cont = src.contiguous().to(at::kFloat);
        const float* src_data = src_cont.data_ptr<float>();

        // Copy to hologram buffer via raw pointer
        float* dst_data = static_cast<float*>(self_storage->data_ptr);
        std::memcpy(dst_data, src_data, self.numel() * sizeof(float));

        return self;
    } else if (src.device().type() == c10::DeviceType::PrivateUse1) {
        // Copy from Hologram to Hologram
        auto* src_storage = get_tensor_storage(const_cast<at::Tensor&>(src).data_ptr());
        float* src_data = static_cast<float*>(src_storage->data_ptr);
        float* dst_data = static_cast<float*>(self_storage->data_ptr);
        std::memcpy(dst_data, src_data, self.numel() * sizeof(float));

        return self;
    } else {
        throw std::runtime_error("Copy from " + c10::toString(src.device().type()) + " to Hologram not supported");
    }
}

// Fill with scalar value
at::Tensor& fill_scalar_hologram_(at::Tensor& self, const at::Scalar& value) {
    auto* storage = get_tensor_storage(self.data_ptr());
    float* data = static_cast<float*>(storage->data_ptr);
    float fill_value = value.toFloat();

    for (int64_t i = 0; i < self.numel(); i++) {
        data[i] = fill_value;
    }

    return self;
}

// Zero fill
at::Tensor& zero_hologram_(at::Tensor& self) {
    auto* storage = get_tensor_storage(self.data_ptr());
    float* data = static_cast<float*>(storage->data_ptr);
    std::memset(data, 0, self.numel() * sizeof(float));
    return self;
}

// _to_copy - Used by .to('hologram') and .cpu() for device transfers
at::Tensor _to_copy_hologram(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<at::MemoryFormat> memory_format
) {
    auto target_device = device.value_or(at::Device(c10::DeviceType::PrivateUse1, 0));

    // Handle hologram → CPU transfer
    if (target_device.type() == c10::DeviceType::CPU) {
        // Create CPU tensor
        auto options = at::TensorOptions()
            .dtype(dtype.value_or(self.scalar_type()))
            .layout(layout.value_or(at::kStrided))
            .device(c10::DeviceType::CPU)
            .pinned_memory(pin_memory.value_or(false));

        auto result = at::empty(self.sizes(), options);

        // Copy from hologram to CPU
        auto* src_storage = get_tensor_storage(const_cast<at::Tensor&>(self).data_ptr());
        const float* src_data = static_cast<const float*>(src_storage->data_ptr);
        float* dst_data = result.data_ptr<float>();
        std::memcpy(dst_data, src_data, self.numel() * sizeof(float));

        return result;
    }

    // Handle CPU/hologram → hologram transfer (original behavior)
    auto options = at::TensorOptions()
        .dtype(dtype.value_or(self.scalar_type()))
        .layout(layout.value_or(at::kStrided))
        .device(c10::DeviceType::PrivateUse1, target_device.index())
        .pinned_memory(pin_memory.value_or(false));

    auto result = allocate_hologram_tensor(self.sizes(), options);

    // Copy data from source
    result.copy_(self, non_blocking);

    return result;
}

/**
 * Register tensor factory functions with PyTorch
 *
 * This makes torch.empty(device='hologram') work.
 */
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // Tensor creation
    m.impl("empty.memory_format", empty_hologram);
    m.impl("zeros", zeros_hologram);
    m.impl("ones", ones_hologram);

    // Tensor operations
    m.impl("copy_", copy_hologram_);
    m.impl("fill_.Scalar", fill_scalar_hologram_);
    m.impl("zero_", zero_hologram_);
    m.impl("_to_copy", _to_copy_hologram);
}

// Copy from Hologram to CPU (still inside namespace)
at::Tensor& copy_from_hologram_to_cpu_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking
) {
    TORCH_CHECK(self.numel() == src.numel(), "Tensor sizes must match for copy");

    // Only handle case where self is CPU and src is hologram
    if (self.device().type() != c10::DeviceType::CPU ||
        src.device().type() != c10::DeviceType::PrivateUse1) {
        // Fallback to default
        return self;
    }

    // Get source storage
    auto* src_storage = get_tensor_storage(const_cast<at::Tensor&>(src).data_ptr());
    const float* src_data = static_cast<const float*>(src_storage->data_ptr);

    // Get destination CPU pointer
    float* dst_data = self.data_ptr<float>();

    // Copy data
    std::memcpy(dst_data, src_data, self.numel() * sizeof(float));

    return self;
}

} // namespace hologram

// Register CPU-side operations for copying FROM hologram
TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("copy_", hologram::copy_from_hologram_to_cpu_);
}
