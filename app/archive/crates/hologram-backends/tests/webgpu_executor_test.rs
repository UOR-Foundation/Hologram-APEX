//! Integration tests for WebGPU executor
//!
//! Tests compute operations (vector add, mul) with WebGPU backend.

#![cfg(all(feature = "webgpu", target_arch = "wasm32"))]

use hologram_backends::backends::wasm::webgpu::{WebGpuDevice, WebGpuExecutor};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// Test vector addition with WebGPU compute
#[wasm_bindgen_test]
async fn test_webgpu_vector_add() {
    // Skip if WebGPU not available
    if !WebGpuDevice::is_available() {
        web_sys::console::log_1(&"Skipping: WebGPU not available".into());
        return;
    }

    // Initialize device and executor
    let device = match WebGpuDevice::new().await {
        Ok(d) => d,
        Err(e) => {
            web_sys::console::error_1(&format!("Device init failed: {}", e).into());
            return;
        }
    };

    let mut executor = match WebGpuExecutor::new(device) {
        Ok(e) => e,
        Err(e) => {
            web_sys::console::error_1(&format!("Executor creation failed: {}", e).into());
            return;
        }
    };

    // Test data
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

    web_sys::console::log_1(&"Executing vector addition on GPU...".into());

    // Execute on GPU
    let result = match executor.vector_add(&a, &b).await {
        Ok(r) => r,
        Err(e) => {
            web_sys::console::error_1(&format!("Vector add failed: {}", e).into());
            panic!("Vector add failed: {}", e);
        }
    };

    // Verify results
    let expected = vec![11.0f32, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0];

    web_sys::console::log_1(&format!("Result: {:?}", result).into());
    web_sys::console::log_1(&format!("Expected: {:?}", expected).into());

    assert_eq!(result.len(), expected.len());
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            r,
            e
        );
    }

    web_sys::console::log_1(&"Vector addition test PASSED".into());
}

/// Test vector multiplication with WebGPU compute
#[wasm_bindgen_test]
async fn test_webgpu_vector_mul() {
    // Skip if WebGPU not available
    if !WebGpuDevice::is_available() {
        return;
    }

    let device = match WebGpuDevice::new().await {
        Ok(d) => d,
        Err(_) => return,
    };

    let mut executor = match WebGpuExecutor::new(device) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Test data
    let a = vec![2.0f32, 3.0, 4.0, 5.0];
    let b = vec![10.0f32, 10.0, 10.0, 10.0];

    web_sys::console::log_1(&"Executing vector multiplication on GPU...".into());

    let result = match executor.vector_mul(&a, &b).await {
        Ok(r) => r,
        Err(e) => {
            web_sys::console::error_1(&format!("Vector mul failed: {}", e).into());
            panic!("Vector mul failed: {}", e);
        }
    };

    // Verify results
    let expected = vec![20.0f32, 30.0, 40.0, 50.0];

    assert_eq!(result.len(), expected.len());
    for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            r,
            e
        );
    }

    web_sys::console::log_1(&"Vector multiplication test PASSED".into());
}

/// Test large vector operations
#[wasm_bindgen_test]
async fn test_webgpu_large_vectors() {
    if !WebGpuDevice::is_available() {
        return;
    }

    let device = match WebGpuDevice::new().await {
        Ok(d) => d,
        Err(_) => return,
    };

    let mut executor = match WebGpuExecutor::new(device) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Large vectors (1024 elements - 4 workgroups)
    let n = 1024;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();

    web_sys::console::log_1(&format!("Testing {} element vectors...", n).into());

    let result = match executor.vector_add(&a, &b).await {
        Ok(r) => r,
        Err(e) => {
            panic!("Large vector add failed: {}", e);
        }
    };

    // Verify a few elements
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 3.0); // 1 + 2
    assert_eq!(result[10], 30.0); // 10 + 20
    assert_eq!(result[100], 300.0); // 100 + 200
    assert_eq!(result[1023], 3069.0); // 1023 + 2046

    web_sys::console::log_1(&"Large vector test PASSED".into());
}

/// Test cache statistics
#[wasm_bindgen_test]
async fn test_webgpu_cache_stats() {
    if !WebGpuDevice::is_available() {
        return;
    }

    let device = match WebGpuDevice::new().await {
        Ok(d) => d,
        Err(_) => return,
    };

    let mut executor = match WebGpuExecutor::new(device) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Initial cache should be empty
    let stats = executor.cache_stats();
    web_sys::console::log_1(&format!("Initial cache stats: {}", stats).into());

    // Execute vector add (should compile shader)
    let a = vec![1.0f32, 2.0];
    let b = vec![3.0f32, 4.0];
    let _ = executor.vector_add(&a, &b).await;

    // Cache should have 1 pipeline and 1 shader module
    let stats = executor.cache_stats();
    web_sys::console::log_1(&format!("After vector_add: {}", stats).into());
    assert!(stats.num_pipelines >= 1);
    assert!(stats.num_shader_modules >= 1);

    // Execute vector mul (should compile another shader)
    let _ = executor.vector_mul(&a, &b).await;

    let stats = executor.cache_stats();
    web_sys::console::log_1(&format!("After vector_mul: {}", stats).into());
    assert!(stats.num_pipelines >= 2);

    web_sys::console::log_1(&"Cache stats test PASSED".into());
}
