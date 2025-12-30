//! WebGPU End-to-End Demo
//!
//! Demonstrates WebGPU acceleration from hologram-core operations.
//!
//! This example shows:
//! - Creating an Executor with WebGPU backend
//! - Performing GPU-accelerated vector operations
//! - Using activation functions on the GPU
//! - Reading results back from GPU memory
//!
//! # Running
//!
//! This example requires a WebGPU-capable browser environment:
//!
//! ```bash
//! wasm-pack test --chrome --headless --features webgpu
//! ```

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

use hologram_core::{ops, BackendType, Executor};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// Basic WebGPU vector addition example
#[wasm_bindgen_test]
async fn webgpu_vector_add() {
    // Create executor with WebGPU backend
    let mut exec = Executor::new_with_backend_async(BackendType::WebGpu)
        .await
        .expect("Failed to create WebGPU executor");

    // Allocate buffers
    let mut a = exec.allocate::<f32>(1024).expect("Failed to allocate buffer a");
    let mut b = exec.allocate::<f32>(1024).expect("Failed to allocate buffer b");
    let mut c = exec.allocate::<f32>(1024).expect("Failed to allocate buffer c");

    // Initialize data
    let data_a = vec![1.0f32; 1024];
    let data_b = vec![2.0f32; 1024];

    a.copy_from_slice(&mut exec, &data_a).expect("Failed to copy data to buffer a");
    b.copy_from_slice(&mut exec, &data_b).expect("Failed to copy data to buffer b");

    // GPU-accelerated addition (via WebGPU fast-path)
    ops::math::vector_add(&mut exec, &a, &b, &mut c, 1024).expect("vector_add failed");

    // Read result back from GPU
    let result = c.to_vec(&exec).expect("Failed to read result");

    // Verify: c[i] should be a[i] + b[i] = 1.0 + 2.0 = 3.0
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - 3.0).abs() < 1e-6,
            "Element {} incorrect: expected 3.0, got {}",
            i,
            val
        );
    }

    wasm_bindgen_test::console_log!("✅ WebGPU vector_add test passed!");
}

/// WebGPU activation function example
#[wasm_bindgen_test]
async fn webgpu_sigmoid() {
    // Create executor with WebGPU backend
    let mut exec = Executor::new_with_backend_async(BackendType::WebGpu)
        .await
        .expect("Failed to create WebGPU executor");

    // Allocate buffers
    let mut input = exec.allocate::<f32>(1024).expect("Failed to allocate input buffer");
    let mut output = exec.allocate::<f32>(1024).expect("Failed to allocate output buffer");

    // Initialize with test values
    let data = vec![0.0f32; 1024]; // sigmoid(0) = 0.5

    input
        .copy_from_slice(&mut exec, &data)
        .expect("Failed to copy data to input");

    // GPU-accelerated sigmoid activation
    ops::activation::sigmoid(&mut exec, &input, &mut output, 1024).expect("sigmoid failed");

    // Read result
    let result = output.to_vec(&exec).expect("Failed to read result");

    // Verify: sigmoid(0.0) ≈ 0.5
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - 0.5).abs() < 0.01,
            "Element {} incorrect: expected 0.5, got {}",
            i,
            val
        );
    }

    wasm_bindgen_test::console_log!("✅ WebGPU sigmoid test passed!");
}

/// Chained WebGPU operations example
#[wasm_bindgen_test]
async fn webgpu_operation_chain() {
    // Create executor with WebGPU backend
    let mut exec = Executor::new_with_backend_async(BackendType::WebGpu)
        .await
        .expect("Failed to create WebGPU executor");

    // Allocate buffers
    let mut a = exec.allocate::<f32>(1024).expect("Failed to allocate buffer a");
    let mut b = exec.allocate::<f32>(1024).expect("Failed to allocate buffer b");
    let mut temp = exec.allocate::<f32>(1024).expect("Failed to allocate temp buffer");
    let mut result = exec.allocate::<f32>(1024).expect("Failed to allocate result buffer");

    // Initialize data
    let data_a = vec![1.0f32; 1024];
    let data_b = vec![2.0f32; 1024];

    a.copy_from_slice(&mut exec, &data_a).expect("Failed to copy data to buffer a");
    b.copy_from_slice(&mut exec, &data_b).expect("Failed to copy data to buffer b");

    // Chain of GPU operations:
    // 1. temp = a + b      (1.0 + 2.0 = 3.0)
    // 2. result = temp * temp  (3.0 * 3.0 = 9.0)

    ops::math::vector_add(&mut exec, &a, &b, &mut temp, 1024).expect("vector_add failed");

    ops::math::vector_mul(&mut exec, &temp, &temp, &mut result, 1024).expect("vector_mul failed");

    // Read final result
    let output = result.to_vec(&exec).expect("Failed to read result");

    // Verify: (1 + 2)² = 9
    for (i, &val) in output.iter().enumerate() {
        assert!(
            (val - 9.0).abs() < 1e-5,
            "Element {} incorrect: expected 9.0, got {}",
            i,
            val
        );
    }

    wasm_bindgen_test::console_log!("✅ WebGPU operation chain test passed!");
}

/// Multiple data type test
#[wasm_bindgen_test]
async fn webgpu_different_sizes() {
    let mut exec = Executor::new_with_backend_async(BackendType::WebGpu)
        .await
        .expect("Failed to create WebGPU executor");

    // Test with different buffer sizes
    let sizes = vec![256, 512, 1024, 2048];

    for size in sizes {
        let mut a = exec.allocate::<f32>(size).expect(&format!("Failed to allocate buffer of size {}", size));
        let mut b = exec.allocate::<f32>(size).expect(&format!("Failed to allocate buffer of size {}", size));
        let mut c = exec.allocate::<f32>(size).expect(&format!("Failed to allocate buffer of size {}", size));

        let data_a = vec![5.0f32; size];
        let data_b = vec![10.0f32; size];

        a.copy_from_slice(&mut exec, &data_a).unwrap();
        b.copy_from_slice(&mut exec, &data_b).unwrap();

        ops::math::vector_add(&mut exec, &a, &b, &mut c, size).expect(&format!("vector_add failed for size {}", size));

        let result = c.to_vec(&exec).unwrap();

        for &val in &result {
            assert!((val - 15.0).abs() < 1e-5);
        }

        wasm_bindgen_test::console_log!("✅ Size {} test passed", size);
    }
}

/// Performance comparison test (WebGPU should be faster for large buffers)
#[wasm_bindgen_test]
async fn webgpu_performance_check() {
    let mut exec = Executor::new_with_backend_async(BackendType::WebGpu)
        .await
        .expect("Failed to create WebGPU executor");

    let size = 100_000; // Large buffer for performance test

    let mut a = exec.allocate::<f32>(size).expect("Failed to allocate buffer a");
    let mut b = exec.allocate::<f32>(size).expect("Failed to allocate buffer b");
    let mut c = exec.allocate::<f32>(size).expect("Failed to allocate buffer c");

    let data_a = vec![1.0f32; size];
    let data_b = vec![2.0f32; size];

    a.copy_from_slice(&mut exec, &data_a).unwrap();
    b.copy_from_slice(&mut exec, &data_b).unwrap();

    // Perform operation (WebGPU fast-path should be used)
    let start = web_sys::window()
        .expect("no window")
        .performance()
        .expect("no performance")
        .now();

    ops::math::vector_add(&mut exec, &a, &b, &mut c, size).expect("vector_add failed");

    let duration = web_sys::window()
        .expect("no window")
        .performance()
        .expect("no performance")
        .now()
        - start;

    wasm_bindgen_test::console_log!(
        "✅ WebGPU vector_add on {} elements took {:.2}ms",
        size,
        duration
    );

    // Verify correctness
    let result = c.to_vec(&exec).unwrap();
    assert!((result[0] - 3.0).abs() < 1e-6);
    assert!((result[size - 1] - 3.0).abs() < 1e-6);
}
