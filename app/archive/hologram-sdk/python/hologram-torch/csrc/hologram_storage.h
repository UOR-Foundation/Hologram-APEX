#pragma once

#include <cstdint>
#include <string>

namespace hologram {

/**
 * Storage information for Hologram tensors
 *
 * Maps PyTorch tensor data to Hologram buffer handles.
 * Different strategies for CPU (raw pointers) vs GPU (DLPack).
 */
struct HologramStorage {
    // Executor handle (from hologram-ffi)
    uint64_t executor_handle;

    // Buffer handle (from hologram-ffi)
    uint64_t buffer_handle;

    // Tensor handle (for GPU backends using DLPack)
    uint64_t tensor_handle = 0;

    // Backend type: "cpu", "metal", or "cuda"
    std::string backend;

    // Raw data pointer (for CPU backend only)
    void* data_ptr = nullptr;

    // Number of elements
    size_t numel = 0;

    // Whether this tensor was created from DLPack
    bool from_dlpack = false;
};

} // namespace hologram
