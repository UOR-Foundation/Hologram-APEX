//! Hologram FFI Integration Tests
//!
//! Tests the FFI API exposed to other languages

use hologram_ffi::*;
use serial_test::serial;

#[test]
fn test_version() {
    let version = get_version();
    assert!(!version.is_empty(), "Version should not be empty");
    assert!(
        version.chars().any(|c| c.is_numeric()),
        "Version should contain numbers"
    );
}

#[test]
#[serial]
fn test_executor_creation_and_cleanup() {
    // Create executor
    let exec_handle = new_executor();
    assert_ne!(exec_handle, 0, "Executor handle should not be zero");

    // Cleanup executor
    executor_cleanup(exec_handle);
}

#[test]
#[serial]
fn test_buffer_allocation_and_cleanup() {
    // Create executor
    let exec = new_executor();

    // Allocate buffer for 256 f32 elements
    let buffer = executor_allocate_buffer(exec, 256);
    assert_ne!(buffer, 0, "Buffer handle should not be zero");

    // Check buffer properties
    let length = buffer_length(buffer);
    assert_eq!(length, 256, "Buffer length should be 256");

    let size_bytes = buffer_size_bytes(buffer);
    assert_eq!(size_bytes, 1024, "Buffer size should be 1024 bytes (256 * 4)");

    let elem_size = buffer_element_size(buffer);
    assert_eq!(elem_size, 4, "Element size should be 4 bytes");

    // Cleanup
    buffer_cleanup(buffer);
    executor_cleanup(exec);
}

#[test]
#[serial]
fn test_buffer_fill_and_read() {
    let exec = new_executor();
    let buffer = executor_allocate_buffer(exec, 10);

    // Fill buffer with value 42.0
    buffer_fill(exec, buffer, 42.0, 10);

    // Read back values as JSON
    let json_result = buffer_to_vec(exec, buffer);
    let values: Vec<f32> = serde_json::from_str(&json_result).expect("Failed to parse JSON");
    assert_eq!(values.len(), 10, "Should have 10 elements");

    // Verify all values are 42.0
    for value in values {
        assert!((value - 42.0).abs() < 0.001, "Value should be 42.0");
    }

    buffer_cleanup(buffer);
    executor_cleanup(exec);
}

#[test]
#[serial]
fn test_buffer_copy_from_slice() {
    let exec = new_executor();
    let buffer = executor_allocate_buffer(exec, 5);

    // Create test data as JSON: [1.0, 2.0, 3.0, 4.0, 5.0]
    let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let json_data = serde_json::to_string(&test_data).expect("Failed to serialize test data");

    // Copy data to buffer
    buffer_copy_from_slice(exec, buffer, json_data);

    // Read back and verify
    let json_result = buffer_to_vec(exec, buffer);
    let result: Vec<f32> = serde_json::from_str(&json_result).expect("Failed to parse result");

    for (i, &value) in result.iter().enumerate() {
        assert!(
            (value - test_data[i]).abs() < 0.001,
            "Value at index {} should be {}, got {}",
            i,
            test_data[i],
            value
        );
    }

    buffer_cleanup(buffer);
    executor_cleanup(exec);
}

#[test]
#[serial]
fn test_vector_add() {
    let exec = new_executor();

    // Allocate buffers
    let buf_a = executor_allocate_buffer(exec, 4);
    let buf_b = executor_allocate_buffer(exec, 4);
    let buf_c = executor_allocate_buffer(exec, 4);

    // Initialize data: a = [1, 2, 3, 4], b = [5, 6, 7, 8]
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let data_b = vec![5.0, 6.0, 7.0, 8.0];
    let json_a = serde_json::to_string(&data_a).unwrap();
    let json_b = serde_json::to_string(&data_b).unwrap();

    buffer_copy_from_slice(exec, buf_a, json_a);
    buffer_copy_from_slice(exec, buf_b, json_b);

    // Perform addition: c = a + b
    vector_add_f32(exec, buf_a, buf_b, buf_c, 4);

    // Verify result: [6, 8, 10, 12]
    let json_result = buffer_to_vec(exec, buf_c);
    let result: Vec<f32> = serde_json::from_str(&json_result).unwrap();
    let expected = [6.0, 8.0, 10.0, 12.0];

    for (i, &value) in result.iter().enumerate() {
        assert!(
            (value - expected[i]).abs() < 0.001,
            "Result at index {} should be {}, got {}",
            i,
            expected[i],
            value
        );
    }

    buffer_cleanup(buf_a);
    buffer_cleanup(buf_b);
    buffer_cleanup(buf_c);
    executor_cleanup(exec);
}

#[test]
#[serial]
fn test_vector_neg() {
    let exec = new_executor();

    // Allocate buffers
    let buf_in = executor_allocate_buffer(exec, 4);
    let buf_out = executor_allocate_buffer(exec, 4);

    // Initialize data: [1.0, -2.0, 3.0, -4.0]
    let data_in = vec![1.0, -2.0, 3.0, -4.0];
    let json_in = serde_json::to_string(&data_in).unwrap();
    buffer_copy_from_slice(exec, buf_in, json_in);

    // Perform negation: out = -in
    vector_neg_f32(exec, buf_in, buf_out, 4);

    // Verify result: [-1.0, 2.0, -3.0, 4.0]
    let json_result = buffer_to_vec(exec, buf_out);
    let result: Vec<f32> = serde_json::from_str(&json_result).unwrap();
    let expected = [-1.0, 2.0, -3.0, 4.0];

    for (i, &value) in result.iter().enumerate() {
        assert!(
            (value - expected[i]).abs() < 0.001,
            "Result at index {} should be {}, got {}",
            i,
            expected[i],
            value
        );
    }

    buffer_cleanup(buf_in);
    buffer_cleanup(buf_out);
    executor_cleanup(exec);
}

#[test]
#[serial]
fn test_reduce_sum() {
    let exec = new_executor();

    // Allocate buffers (sum needs 3 elements for temporaries)
    let buf_in = executor_allocate_buffer(exec, 5);
    let buf_out = executor_allocate_buffer(exec, 3);

    // Initialize data: [1.0, 2.0, 3.0, 4.0, 5.0]
    let data_in = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let json_in = serde_json::to_string(&data_in).unwrap();
    buffer_copy_from_slice(exec, buf_in, json_in);

    // Perform sum reduction
    reduce_sum_f32(exec, buf_in, buf_out, 5);

    // Verify result: sum = 15.0 (result is in first element)
    let json_result = buffer_to_vec(exec, buf_out);
    let result: Vec<f32> = serde_json::from_str(&json_result).unwrap();
    let sum = result[0];
    assert!((sum - 15.0).abs() < 0.001, "Sum should be 15.0, got {}", sum);

    buffer_cleanup(buf_in);
    buffer_cleanup(buf_out);
    executor_cleanup(exec);
}

#[test]
#[serial]
fn test_tensor_creation_and_cleanup() {
    let exec = new_executor();
    let buffer = executor_allocate_buffer(exec, 12);

    // Create tensor with shape [3, 4]
    let shape = vec![3u64, 4u64];
    let shape_json = serde_json::to_string(&shape).unwrap();
    let tensor = tensor_from_buffer(buffer, shape_json.clone());
    assert_ne!(tensor, 0, "Tensor handle should not be zero");

    // Verify tensor properties
    let ndim = tensor_ndim(tensor);
    assert_eq!(ndim, 2, "Tensor should be 2D");

    let numel = tensor_numel(tensor);
    assert_eq!(numel, 12, "Tensor should have 12 elements");

    let tensor_shape_json = tensor_shape(tensor);
    let tensor_shape: Vec<u64> = serde_json::from_str(&tensor_shape_json).unwrap();
    assert_eq!(tensor_shape, shape, "Shape should match");

    // Cleanup
    tensor_cleanup(tensor);
    buffer_cleanup(buffer);
    executor_cleanup(exec);
}

#[test]
#[serial]
fn test_clear_all_registries() {
    // Create some resources
    let exec1 = new_executor();
    let exec2 = new_executor();
    let buffer = executor_allocate_buffer(exec1, 10);

    // Verify they exist
    assert_ne!(exec1, 0);
    assert_ne!(exec2, 0);
    assert_ne!(buffer, 0);

    // Clear all registries (for cleanup in tests)
    clear_all_registries();

    // After clearing, these handles are invalid, but the function should not panic
    // This is mainly for testing the cleanup mechanism
}
