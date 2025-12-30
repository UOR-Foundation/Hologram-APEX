//! Comprehensive tests for all WebGPU operations
//!
//! Tests all MergeVariant, SplitVariant, and reduction operations

#![cfg(all(feature = "webgpu", target_arch = "wasm32"))]

use hologram_backends::backends::wasm::webgpu::{WebGpuDevice, WebGpuExecutor};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// Helper to create executor
async fn create_executor() -> Option<WebGpuExecutor> {
    if !WebGpuDevice::is_available() {
        return None;
    }

    match WebGpuDevice::new().await {
        Ok(device) => WebGpuExecutor::new(device).ok(),
        Err(_) => None,
    }
}

/// Helper to assert vectors are approximately equal
fn assert_vec_approx_eq(actual: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(actual.len(), expected.len(), "Vector lengths differ");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < epsilon,
            "Mismatch at index {}: got {}, expected {} (diff: {})",
            i,
            a,
            e,
            (a - e).abs()
        );
    }
}

// ============================================================================
// Binary Operations (MergeVariant)
// ============================================================================

#[wasm_bindgen_test]
async fn test_vector_min() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => {
            web_sys::console::log_1(&"Skipping: WebGPU not available".into());
            return;
        }
    };

    let a = vec![1.0f32, 5.0, 3.0, 9.0, 2.0];
    let b = vec![3.0f32, 2.0, 7.0, 1.0, 8.0];

    let result = executor.vector_min(&a, &b).await.unwrap();
    let expected = vec![1.0f32, 2.0, 3.0, 1.0, 2.0];

    assert_vec_approx_eq(&result, &expected, 1e-5);
    web_sys::console::log_1(&"vector_min test PASSED".into());
}

#[wasm_bindgen_test]
async fn test_vector_max() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    let a = vec![1.0f32, 5.0, 3.0, 9.0, 2.0];
    let b = vec![3.0f32, 2.0, 7.0, 1.0, 8.0];

    let result = executor.vector_max(&a, &b).await.unwrap();
    let expected = vec![3.0f32, 5.0, 7.0, 9.0, 8.0];

    assert_vec_approx_eq(&result, &expected, 1e-5);
    web_sys::console::log_1(&"vector_max test PASSED".into());
}

// ============================================================================
// Split Operations
// ============================================================================

#[wasm_bindgen_test]
async fn test_vector_sub() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    let a = vec![10.0f32, 20.0, 30.0, 40.0];
    let b = vec![1.0f32, 2.0, 3.0, 4.0];

    let result = executor.vector_sub(&a, &b).await.unwrap();
    let expected = vec![9.0f32, 18.0, 27.0, 36.0];

    assert_vec_approx_eq(&result, &expected, 1e-5);
    web_sys::console::log_1(&"vector_sub test PASSED".into());
}

#[wasm_bindgen_test]
async fn test_vector_div() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    let a = vec![10.0f32, 20.0, 30.0, 40.0];
    let b = vec![2.0f32, 4.0, 5.0, 8.0];

    let result = executor.vector_div(&a, &b).await.unwrap();
    let expected = vec![5.0f32, 5.0, 6.0, 5.0];

    assert_vec_approx_eq(&result, &expected, 1e-5);
    web_sys::console::log_1(&"vector_div test PASSED".into());
}

// ============================================================================
// Unary Operations
// ============================================================================

#[wasm_bindgen_test]
async fn test_vector_abs() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    let input = vec![-1.0f32, 2.0, -3.0, 4.0, -5.0];
    let result = executor.vector_abs(&input).await.unwrap();
    let expected = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

    assert_vec_approx_eq(&result, &expected, 1e-5);
    web_sys::console::log_1(&"vector_abs test PASSED".into());
}

#[wasm_bindgen_test]
async fn test_vector_exp() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    let input = vec![0.0f32, 1.0, 2.0];
    let result = executor.vector_exp(&input).await.unwrap();
    let expected = vec![1.0f32, 2.71828, 7.38906]; // e^0, e^1, e^2

    assert_vec_approx_eq(&result, &expected, 1e-3); // Slightly larger epsilon for exp
    web_sys::console::log_1(&"vector_exp test PASSED".into());
}

#[wasm_bindgen_test]
async fn test_vector_log() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    let input = vec![1.0f32, 2.71828, 7.38906];
    let result = executor.vector_log(&input).await.unwrap();
    let expected = vec![0.0f32, 1.0, 2.0]; // ln(1), ln(e), ln(e^2)

    assert_vec_approx_eq(&result, &expected, 1e-3);
    web_sys::console::log_1(&"vector_log test PASSED".into());
}

#[wasm_bindgen_test]
async fn test_vector_sqrt() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    let input = vec![1.0f32, 4.0, 9.0, 16.0, 25.0];
    let result = executor.vector_sqrt(&input).await.unwrap();
    let expected = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

    assert_vec_approx_eq(&result, &expected, 1e-5);
    web_sys::console::log_1(&"vector_sqrt test PASSED".into());
}

#[wasm_bindgen_test]
async fn test_vector_sigmoid() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    let input = vec![0.0f32, 1.0, -1.0];
    let result = executor.vector_sigmoid(&input).await.unwrap();

    // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269
    let expected = vec![0.5f32, 0.7310586, 0.26894143];

    assert_vec_approx_eq(&result, &expected, 1e-5);
    web_sys::console::log_1(&"vector_sigmoid test PASSED".into());
}

#[wasm_bindgen_test]
async fn test_vector_tanh() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    let input = vec![0.0f32, 1.0, -1.0, 2.0];
    let result = executor.vector_tanh(&input).await.unwrap();

    // tanh(0) = 0, tanh(1) ≈ 0.762, tanh(-1) ≈ -0.762, tanh(2) ≈ 0.964
    let expected = vec![0.0f32, 0.7615942, -0.7615942, 0.9640276];

    assert_vec_approx_eq(&result, &expected, 1e-5);
    web_sys::console::log_1(&"vector_tanh test PASSED".into());
}

// ============================================================================
// Large Vector Tests
// ============================================================================

#[wasm_bindgen_test]
async fn test_large_vector_operations() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    // Test with 1024 elements (4 workgroups)
    let n = 1024;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();

    web_sys::console::log_1(&format!("Testing {} elements...", n).into());

    // Test min
    let result = executor.vector_min(&a, &b).await.unwrap();
    assert_eq!(result[0], 0.0); // min(0, 0) = 0
    assert_eq!(result[1], 1.0); // min(1, 2) = 1
    assert_eq!(result[100], 100.0); // min(100, 200) = 100

    // Test max
    let result = executor.vector_max(&a, &b).await.unwrap();
    assert_eq!(result[0], 0.0); // max(0, 0) = 0
    assert_eq!(result[1], 2.0); // max(1, 2) = 2
    assert_eq!(result[100], 200.0); // max(100, 200) = 200

    // Test sub
    let result = executor.vector_sub(&b, &a).await.unwrap();
    assert_eq!(result[0], 0.0); // 0 - 0 = 0
    assert_eq!(result[1], 1.0); // 2 - 1 = 1
    assert_eq!(result[100], 100.0); // 200 - 100 = 100

    web_sys::console::log_1(&"Large vector operations test PASSED".into());
}

// ============================================================================
// Combined Operations
// ============================================================================

#[wasm_bindgen_test]
async fn test_combined_operations() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    // Test: abs(a - b) for element-wise distance
    let a = vec![1.0f32, 5.0, 10.0];
    let b = vec![3.0f32, 2.0, 8.0];

    let diff = executor.vector_sub(&a, &b).await.unwrap();
    let distance = executor.vector_abs(&diff).await.unwrap();

    let expected = vec![2.0f32, 3.0, 2.0];
    assert_vec_approx_eq(&distance, &expected, 1e-5);

    web_sys::console::log_1(&"Combined operations test PASSED".into());
}

// ============================================================================
// Edge Cases
// ============================================================================

#[wasm_bindgen_test]
async fn test_edge_cases() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    // Test with single element
    let result = executor.vector_abs(&[-1.0]).await.unwrap();
    assert_eq!(result[0], 1.0);

    // Test with zeros
    let zeros = vec![0.0f32; 100];
    let result = executor.vector_exp(&zeros).await.unwrap();
    for &val in &result {
        assert!((val - 1.0).abs() < 1e-5); // exp(0) = 1
    }

    web_sys::console::log_1(&"Edge cases test PASSED".into());
}

// ============================================================================
// Pipeline Cache Test
// ============================================================================

#[wasm_bindgen_test]
async fn test_pipeline_caching() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    let input = vec![1.0f32, 2.0, 3.0];

    // First call should compile shader
    let _ = executor.vector_abs(&input).await.unwrap();
    let stats1 = executor.cache_stats();

    // Second call should reuse cached shader
    let _ = executor.vector_abs(&input).await.unwrap();
    let stats2 = executor.cache_stats();

    // Cache size should not increase (same shader)
    assert_eq!(stats1.num_pipelines, stats2.num_pipelines);
    assert_eq!(stats1.num_shader_modules, stats2.num_shader_modules);

    // Use different operation (should compile new shader)
    let _ = executor.vector_exp(&input).await.unwrap();
    let stats3 = executor.cache_stats();

    // Cache should have grown
    assert!(stats3.num_pipelines > stats2.num_pipelines);

    web_sys::console::log_1(&format!("Pipeline cache test PASSED (cache: {})", stats3).into());
}
