//! WebGPU Backend Integration Tests
//!
//! Tests buffer lifecycle, pool operations, and error handling.
//! These tests require a WebGPU-capable browser environment.

#![cfg(all(target_arch = "wasm32", feature = "webgpu"))]

use hologram_backends::{Backend, WebGpuBackend};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// Test basic buffer allocation, copy operations, and deallocation
#[wasm_bindgen_test]
async fn test_backend_buffer_lifecycle() {
    let mut backend = WebGpuBackend::new().await.expect("Failed to create WebGPU backend");

    // Allocate buffer
    let buffer = backend.allocate_buffer(1024).expect("Failed to allocate buffer");

    // Verify size
    assert_eq!(backend.buffer_size(buffer).unwrap(), 1024);

    // Copy data to GPU
    let data = vec![1.0f32; 256];
    backend
        .copy_to_buffer(buffer, bytemuck::cast_slice(&data))
        .expect("Failed to copy to buffer");

    // Copy data from GPU
    let mut result = vec![0.0f32; 256];
    backend
        .copy_from_buffer(buffer, bytemuck::cast_slice_mut(&mut result))
        .expect("Failed to copy from buffer");

    // Verify data
    assert_eq!(result, data);

    // Free buffer
    backend.free_buffer(buffer).expect("Failed to free buffer");
}

/// Test pool allocation and operations
#[wasm_bindgen_test]
async fn test_pool_operations() {
    let mut backend = WebGpuBackend::new().await.expect("Failed to create WebGPU backend");

    // Allocate pool
    let pool = backend.allocate_pool(4096).expect("Failed to allocate pool");

    // Verify size
    assert_eq!(backend.pool_size(pool).unwrap(), 4096);

    // Copy data to pool at different offsets
    let data1 = vec![1.0f32; 256];
    let data2 = vec![2.0f32; 256];

    backend
        .copy_to_pool(pool, 0, bytemuck::cast_slice(&data1))
        .expect("Failed to copy to pool at offset 0");

    backend
        .copy_to_pool(pool, 1024, bytemuck::cast_slice(&data2))
        .expect("Failed to copy to pool at offset 1024");

    // Read back and verify
    let mut result1 = vec![0.0f32; 256];
    let mut result2 = vec![0.0f32; 256];

    backend
        .copy_from_pool(pool, 0, bytemuck::cast_slice_mut(&mut result1))
        .expect("Failed to copy from pool at offset 0");

    backend
        .copy_from_pool(pool, 1024, bytemuck::cast_slice_mut(&mut result2))
        .expect("Failed to copy from pool at offset 1024");

    assert_eq!(result1, data1);
    assert_eq!(result2, data2);

    // Free pool
    backend.free_pool(pool).expect("Failed to free pool");
}

/// Test buffer pool hit rate
///
/// Allocate and free many buffers to verify pool reuse
#[wasm_bindgen_test]
async fn test_buffer_pool_hit_rate() {
    let mut backend = WebGpuBackend::new().await.expect("Failed to create WebGPU backend");

    // First allocation will be a miss
    let buffer1 = backend.allocate_buffer(1024).expect("Failed to allocate buffer 1");
    backend.free_buffer(buffer1).expect("Failed to free buffer 1");

    // Subsequent allocations of the same size should hit the pool
    for _ in 0..100 {
        let buffer = backend.allocate_buffer(1024).expect("Failed to allocate buffer");
        backend.free_buffer(buffer).expect("Failed to free buffer");
    }

    // Check pool statistics
    let stats = backend.buffer_pool().stats();

    // We expect high hit rate (>95%) after the first allocation
    // First alloc is a miss, next 100 are hits → 100/101 ≈ 99%
    assert!(
        stats.hit_rate() > 95.0,
        "Pool hit rate should be >95%, got {}%",
        stats.hit_rate()
    );
}

/// Test error handling for invalid buffer handle
#[wasm_bindgen_test]
async fn test_invalid_buffer_handle() {
    let mut backend = WebGpuBackend::new().await.expect("Failed to create WebGPU backend");

    // Create an invalid handle (ID 999999 that doesn't exist)
    let invalid_handle = hologram_backends::BufferHandle::new(999999);

    // Attempting to get buffer size should fail
    let result = backend.buffer_size(invalid_handle);
    assert!(result.is_err(), "Expected error for invalid buffer handle");

    // Attempting to copy to invalid buffer should fail
    let data = vec![1.0f32; 256];
    let result = backend.copy_to_buffer(invalid_handle, bytemuck::cast_slice(&data));
    assert!(result.is_err(), "Expected error for copy to invalid buffer");

    // Attempting to copy from invalid buffer should fail
    let mut result_data = vec![0.0f32; 256];
    let result = backend.copy_from_buffer(invalid_handle, bytemuck::cast_slice_mut(&mut result_data));
    assert!(result.is_err(), "Expected error for copy from invalid buffer");
}

/// Test error handling for out-of-bounds copy operations
#[wasm_bindgen_test]
async fn test_out_of_bounds_operations() {
    let mut backend = WebGpuBackend::new().await.expect("Failed to create WebGPU backend");

    // Allocate a small buffer
    let buffer = backend.allocate_buffer(256).expect("Failed to allocate buffer");

    // Attempt to copy more data than buffer size
    let oversized_data = vec![1.0f32; 256]; // 1024 bytes
    let result = backend.copy_to_buffer(buffer, bytemuck::cast_slice(&oversized_data));
    assert!(result.is_err(), "Expected error for oversized copy");

    backend.free_buffer(buffer).expect("Failed to free buffer");
}

/// Test error handling for invalid pool operations
#[wasm_bindgen_test]
async fn test_invalid_pool_operations() {
    let mut backend = WebGpuBackend::new().await.expect("Failed to create WebGPU backend");

    let pool = backend.allocate_pool(1024).expect("Failed to allocate pool");

    // Attempt to copy beyond pool bounds
    let data = vec![1.0f32; 256]; // 1024 bytes
    let result = backend.copy_to_pool(pool, 512, bytemuck::cast_slice(&data)); // offset 512 + 1024 = 1536 > 1024
    assert!(result.is_err(), "Expected error for out-of-bounds pool copy");

    backend.free_pool(pool).expect("Failed to free pool");
}

/// Test multiple buffer allocations and concurrent operations
#[wasm_bindgen_test]
async fn test_multiple_buffers() {
    let mut backend = WebGpuBackend::new().await.expect("Failed to create WebGPU backend");

    // Allocate multiple buffers
    let buffer1 = backend.allocate_buffer(1024).expect("Failed to allocate buffer 1");
    let buffer2 = backend.allocate_buffer(2048).expect("Failed to allocate buffer 2");
    let buffer3 = backend.allocate_buffer(512).expect("Failed to allocate buffer 3");

    // Write different data to each buffer
    let data1 = vec![1.0f32; 256];
    let data2 = vec![2.0f32; 512];
    let data3 = vec![3.0f32; 128];

    backend.copy_to_buffer(buffer1, bytemuck::cast_slice(&data1)).unwrap();
    backend.copy_to_buffer(buffer2, bytemuck::cast_slice(&data2)).unwrap();
    backend.copy_to_buffer(buffer3, bytemuck::cast_slice(&data3)).unwrap();

    // Read back and verify data is correct for each buffer
    let mut result1 = vec![0.0f32; 256];
    let mut result2 = vec![0.0f32; 512];
    let mut result3 = vec![0.0f32; 128];

    backend
        .copy_from_buffer(buffer1, bytemuck::cast_slice_mut(&mut result1))
        .unwrap();
    backend
        .copy_from_buffer(buffer2, bytemuck::cast_slice_mut(&mut result2))
        .unwrap();
    backend
        .copy_from_buffer(buffer3, bytemuck::cast_slice_mut(&mut result3))
        .unwrap();

    assert_eq!(result1, data1);
    assert_eq!(result2, data2);
    assert_eq!(result3, data3);

    // Clean up
    backend.free_buffer(buffer1).unwrap();
    backend.free_buffer(buffer2).unwrap();
    backend.free_buffer(buffer3).unwrap();
}

/// Test pool reuse with different sizes
#[wasm_bindgen_test]
async fn test_pool_size_variance() {
    let mut backend = WebGpuBackend::new().await.expect("Failed to create WebGPU backend");

    // Allocate and free buffers of different sizes
    // Pool should handle this gracefully
    let sizes = vec![512, 1024, 2048, 1024, 512, 4096];

    for size in sizes {
        let buffer = backend
            .allocate_buffer(size)
            .expect(&format!("Failed to allocate buffer of size {}", size));

        // Write and read to verify functionality
        let elem_count = size / 4; // f32 is 4 bytes
        let data = vec![42.0f32; elem_count];
        backend.copy_to_buffer(buffer, bytemuck::cast_slice(&data)).unwrap();

        let mut result = vec![0.0f32; elem_count];
        backend
            .copy_from_buffer(buffer, bytemuck::cast_slice_mut(&mut result))
            .unwrap();
        assert_eq!(result, data);

        backend.free_buffer(buffer).unwrap();
    }

    // Pool should have accumulated buffers of various sizes
    let stats = backend.buffer_pool().stats();
    assert!(stats.total_allocations() > 0, "Pool should have tracked allocations");
}
