/**
 * Hologram Backend - Device Registration and Management
 *
 * This file implements PyTorch's PrivateUse1 device registration for Hologram,
 * enabling native torch.device('hologram') support with all backends.
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>

#include <unordered_map>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

#include "hologram_storage.h"
#include "hologram_utils.h"

namespace hologram {

/**
 * Global context for Hologram backend
 *
 * Manages executors for all backends and tracks tensor storage.
 */
struct HologramContext {
    // Executors for each backend
    uint64_t cpu_executor = 0;
    uint64_t metal_executor = 0;
    uint64_t cuda_executor = 0;

    // Current active backend
    std::string current_backend = "cpu";

    // Tensor storage map: data_ptr → HologramStorage
    std::unordered_map<void*, std::unique_ptr<HologramStorage>> tensor_map;

    // Mutex for thread-safety
    std::mutex mutex;

    // Backend availability flags
    bool cpu_available = false;
    bool metal_available = false;
    bool cuda_available = false;

    // Initialization flag
    bool initialized = false;
};

// Global context instance
static HologramContext g_ctx;

/**
 * Initialize Hologram backend
 *
 * Creates executors for all available backends and selects the best one.
 */
void init_hologram() {
    std::lock_guard<std::mutex> lock(g_ctx.mutex);

    if (g_ctx.initialized) {
        return; // Already initialized
    }

    // CPU backend is always available
    g_ctx.cpu_executor = hologram_new_executor();
    if (g_ctx.cpu_executor == 0) {
        throw std::runtime_error("Failed to create Hologram CPU executor");
    }
    g_ctx.cpu_available = true;
    std::cout << "✅ Hologram CPU backend initialized" << std::endl;

    // Try to initialize Metal backend (macOS only)
    // hologram_new_executor_with_backend returns 0 on failure
    g_ctx.metal_executor = hologram_new_executor_with_backend("metal");
    if (g_ctx.metal_executor != 0) {
        g_ctx.metal_available = true;
        std::cout << "✅ Hologram Metal backend initialized" << std::endl;
    } else {
        g_ctx.metal_available = false;
    }

    // Try to initialize CUDA backend
    // hologram_new_executor_with_backend returns 0 on failure
    g_ctx.cuda_executor = hologram_new_executor_with_backend("cuda");
    if (g_ctx.cuda_executor != 0) {
        g_ctx.cuda_available = true;
        std::cout << "✅ Hologram CUDA backend initialized" << std::endl;
    } else {
        g_ctx.cuda_available = false;
    }

    // Select best backend (preference: CUDA > Metal > CPU)
    if (g_ctx.cuda_available) {
        g_ctx.current_backend = "cuda";
    } else if (g_ctx.metal_available) {
        g_ctx.current_backend = "metal";
    } else {
        g_ctx.current_backend = "cpu";
    }

    g_ctx.initialized = true;

    // Register PrivateUse1 device name
    // Note: PrivateUse1 backend name registration is done in Python via
    // torch.utils.rename_privateuse1_backend('hologram') in __init__.py
}

/**
 * Get current executor handle based on active backend
 */
uint64_t get_current_executor() {
    std::lock_guard<std::mutex> lock(g_ctx.mutex);

    if (!g_ctx.initialized) {
        throw std::runtime_error("Hologram backend not initialized");
    }

    if (g_ctx.current_backend == "cpu") {
        return g_ctx.cpu_executor;
    } else if (g_ctx.current_backend == "metal") {
        return g_ctx.metal_executor;
    } else if (g_ctx.current_backend == "cuda") {
        return g_ctx.cuda_executor;
    } else {
        throw std::runtime_error("Unknown backend: " + g_ctx.current_backend);
    }
}

/**
 * Get executor for specific backend
 */
uint64_t get_executor_for_backend(const std::string& backend) {
    std::lock_guard<std::mutex> lock(g_ctx.mutex);

    if (backend == "cpu") {
        return g_ctx.cpu_executor;
    } else if (backend == "metal") {
        if (!g_ctx.metal_available) {
            throw std::runtime_error("Metal backend not available");
        }
        return g_ctx.metal_executor;
    } else if (backend == "cuda") {
        if (!g_ctx.cuda_available) {
            throw std::runtime_error("CUDA backend not available");
        }
        return g_ctx.cuda_executor;
    } else {
        throw std::runtime_error("Unknown backend: " + backend);
    }
}

/**
 * Register tensor storage
 */
void register_tensor_storage(void* data_ptr, std::unique_ptr<HologramStorage> storage) {
    std::lock_guard<std::mutex> lock(g_ctx.mutex);
    g_ctx.tensor_map[data_ptr] = std::move(storage);
}

/**
 * Get tensor storage
 */
HologramStorage* get_tensor_storage(void* data_ptr) {
    std::lock_guard<std::mutex> lock(g_ctx.mutex);
    auto it = g_ctx.tensor_map.find(data_ptr);
    if (it == g_ctx.tensor_map.end()) {
        throw std::runtime_error("Tensor storage not found");
    }
    return it->second.get();
}

/**
 * Unregister tensor storage
 */
void unregister_tensor_storage(void* data_ptr) {
    std::lock_guard<std::mutex> lock(g_ctx.mutex);
    g_ctx.tensor_map.erase(data_ptr);
}

/**
 * Check if backend is available
 */
bool is_backend_available(const std::string& backend) {
    std::lock_guard<std::mutex> lock(g_ctx.mutex);

    if (backend == "cpu") {
        return g_ctx.cpu_available;
    } else if (backend == "metal") {
        return g_ctx.metal_available;
    } else if (backend == "cuda") {
        return g_ctx.cuda_available;
    }
    return false;
}

/**
 * Python API Functions
 */

bool is_available() {
    return g_ctx.initialized;
}

std::string get_backend() {
    std::lock_guard<std::mutex> lock(g_ctx.mutex);
    return g_ctx.current_backend;
}

void set_backend(const std::string& backend) {
    if (!is_backend_available(backend)) {
        throw std::runtime_error("Backend '" + backend + "' not available");
    }

    std::lock_guard<std::mutex> lock(g_ctx.mutex);
    g_ctx.current_backend = backend;
}

std::vector<std::string> list_backends() {
    std::lock_guard<std::mutex> lock(g_ctx.mutex);

    std::vector<std::string> backends;
    if (g_ctx.cpu_available) {
        backends.push_back("cpu");
    }
    if (g_ctx.metal_available) {
        backends.push_back("metal");
    }
    if (g_ctx.cuda_available) {
        backends.push_back("cuda");
    }
    return backends;
}

uint64_t get_executor_handle() {
    return get_current_executor();
}

/**
 * Device Guard Implementation
 *
 * PyTorch uses device guards to ensure operations run on the correct device.
 */
struct HologramGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

    HologramGuardImpl() {}

    c10::DeviceType type() const override {
        return c10::DeviceType::PrivateUse1;
    }

    c10::Device getDevice() const override {
        // Hologram currently uses single device (index 0)
        return c10::Device(c10::DeviceType::PrivateUse1, 0);
    }

    void setDevice(c10::Device device) const override {
        TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1,
                    "Expected Hologram device, got ", device.type());
        TORCH_CHECK(device.index() == 0,
                    "Hologram currently only supports device index 0");
        // Device switching is a no-op for now (single device)
    }

    void uncheckedSetDevice(c10::Device device) const noexcept override {
        // Device switching is a no-op
    }

    c10::Stream getStream(c10::Device device) const noexcept override {
        // Hologram doesn't use streams yet
        return c10::Stream(c10::Stream::DEFAULT, device);
    }

    c10::Stream exchangeStream(c10::Stream stream) const noexcept override {
        // No stream management yet
        return c10::Stream(c10::Stream::DEFAULT, stream.device());
    }

    c10::DeviceIndex deviceCount() const noexcept override {
        // Single device for now
        return 1;
    }

    void record(
        void** event,
        const c10::Stream& stream,
        const c10::DeviceIndex device_index,
        const c10::EventFlag flag
    ) const override {
        // No event recording yet
    }

    void block(void* event, const c10::Stream& stream) const override {
        // No event blocking yet
    }

    bool queryEvent(void* event) const override {
        // No event queries yet
        return true;
    }

    void destroyEvent(void* event, const c10::DeviceIndex device_index) const noexcept override {
        // No event destruction needed
    }
};

/**
 * PrivateUse1 Hooks Implementation
 *
 * PyTorch requires registering hooks for custom backends.
 */
struct HologramHooksInterface : public at::PrivateUse1HooksInterface {
    ~HologramHooksInterface() override = default;

    // Required: Get device from data pointer
    c10::Device getDeviceFromPtr(void* data) const override {
        // All Hologram tensors are on device 0
        return c10::Device(c10::DeviceType::PrivateUse1, 0);
    }
};

/**
 * Register device guard implementation
 *
 * Note: Device guard registration API changed in PyTorch 2.9+.
 * For now, we skip this as it's optional for basic functionality.
 * PyTorch will use default device guard behavior.
 */
void register_device_guard() {
    // TODO: Update to PyTorch 2.9+ device guard API
    // The registerDeviceGuardImpl API is no longer available
    // Device guards are now registered via TORCH_LIBRARY_IMPL if needed
}

/**
 * Register PrivateUse1 hooks
 */
void register_hooks() {
    static HologramHooksInterface hologram_hooks;
    at::RegisterPrivateUse1HooksInterface(&hologram_hooks);
}

/**
 * Python bindings
 */
PYBIND11_MODULE(_hologram_torch, m) {
    m.doc() = "Hologram Torch backend - Native torch.device('hologram') support";

    // Initialize backend
    m.def("init_hologram", []() {
        init_hologram();
        register_hooks();
        // Device guard registration skipped for PyTorch 2.9+ compatibility
        // register_device_guard();
    }, "Initialize Hologram backend");

    // Backend management
    m.def("is_available", &is_available, "Check if Hologram is available");
    m.def("get_backend", &get_backend, "Get current backend");
    m.def("set_backend", &set_backend, "Set current backend");
    m.def("list_backends", &list_backends, "List available backends");
    m.def("get_executor_handle", &get_executor_handle, "Get current executor handle");
}

} // namespace hologram
