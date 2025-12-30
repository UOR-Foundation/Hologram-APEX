/**
 * CPU-side copy operations for Hologram backend
 * This handles copying FROM hologram TO CPU
 */

#include <torch/extension.h>
#include <ATen/ATen.h>

#include "hologram_storage.h"

namespace hologram {

// Forward declaration
HologramStorage* get_tensor_storage(void* data_ptr);

// Copy from Hologram to CPU
at::Tensor& copy_from_hologram_to_cpu_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking
) {
    TORCH_CHECK(self.numel() == src.numel(), "Tensor sizes must match for copy");
    
    // Only handle case where self is CPU and src is hologram
    if (self.device().type() != c10::DeviceType::CPU ||
        src.device().type() != c10::DeviceType::PrivateUse1) {
        // Let default implementation handle other cases
        return self.copy_(src, non_blocking);
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
