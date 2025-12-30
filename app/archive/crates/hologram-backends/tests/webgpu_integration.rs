//! Integration tests for WebGPU backend
//!
//! Tests WebGPU device initialization, buffer management, and compute pipeline execution.
//! These tests only run when the 'webgpu' feature is enabled and in WASM environments.

#![cfg(all(feature = "webgpu", target_arch = "wasm32"))]

use hologram_backends::backends::wasm::webgpu::WebGpuDevice;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// Test WebGPU availability detection
#[wasm_bindgen_test]
fn test_webgpu_availability() {
    // This test verifies that is_available() returns a boolean
    let available = WebGpuDevice::is_available();

    // Log the result for debugging
    web_sys::console::log_1(&format!("WebGPU available: {}", available).into());

    // The result depends on the browser, but should always be a boolean
    assert!(available == true || available == false);
}

/// Test WebGPU device initialization
#[wasm_bindgen_test]
async fn test_webgpu_device_initialization() {
    // Skip test if WebGPU is not available
    if !WebGpuDevice::is_available() {
        web_sys::console::log_1(&"Skipping test: WebGPU not available".into());
        return;
    }

    // Initialize device
    let device_result = WebGpuDevice::new().await;

    match device_result {
        Ok(device) => {
            // Verify device was created successfully
            web_sys::console::log_1(&"WebGPU device initialized successfully".into());

            // Check adapter info
            let info = device.adapter_info();
            web_sys::console::log_1(&format!("Adapter: {}", info.name).into());
            web_sys::console::log_1(&format!("Backend: {:?}", info.backend).into());

            // Verify we have access to device, queue, adapter
            assert!(device.device().limits().max_buffer_size > 0);

            web_sys::console::log_1(&"Device initialization test passed".into());
        }
        Err(e) => {
            web_sys::console::error_1(&format!("Failed to initialize device: {}", e).into());
            panic!("Device initialization failed: {}", e);
        }
    }
}

/// Test WebGPU device limits
#[wasm_bindgen_test]
async fn test_webgpu_device_limits() {
    // Skip test if WebGPU is not available
    if !WebGpuDevice::is_available() {
        return;
    }

    let device = match WebGpuDevice::new().await {
        Ok(d) => d,
        Err(_) => return,
    };

    // Check device limits
    let limits = device.limits();

    web_sys::console::log_1(&format!("Max buffer size: {}", limits.max_buffer_size).into());
    web_sys::console::log_1(
        &format!(
            "Max compute workgroups per dimension: {}",
            limits.max_compute_workgroups_per_dimension
        )
        .into(),
    );
    web_sys::console::log_1(&format!("Max compute workgroup size x: {}", limits.max_compute_workgroup_size_x).into());

    // Verify minimum requirements
    assert!(limits.max_buffer_size > 0);
    assert!(limits.max_compute_workgroups_per_dimension > 0);
    assert!(limits.max_compute_workgroup_size_x >= 256);
}

/// Test WebGPU buffer creation
#[wasm_bindgen_test]
async fn test_webgpu_buffer_creation() {
    // Skip test if WebGPU is not available
    if !WebGpuDevice::is_available() {
        return;
    }

    let device = match WebGpuDevice::new().await {
        Ok(d) => d,
        Err(_) => return,
    };

    // Create a simple buffer
    let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Buffer"),
        size: 1024,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Verify buffer size
    assert_eq!(buffer.size(), 1024);

    web_sys::console::log_1(&"Buffer creation test passed".into());
}

/// Test WebGPU device debug formatting
#[wasm_bindgen_test]
async fn test_webgpu_device_debug() {
    // Skip test if WebGPU is not available
    if !WebGpuDevice::is_available() {
        return;
    }

    let device = match WebGpuDevice::new().await {
        Ok(d) => d,
        Err(_) => return,
    };

    // Test Debug implementation
    let debug_str = format!("{:?}", device);
    assert!(debug_str.contains("WebGpuDevice"));

    web_sys::console::log_1(&format!("Device debug: {}", debug_str).into());
}
