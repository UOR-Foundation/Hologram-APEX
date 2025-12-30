//! Tensor operation wrappers for FFI
//!
//! Provides handle-based API for tensor operations.

use crate::handles::{generate_handle, lock_registry, BUFFER_REGISTRY, EXECUTOR_REGISTRY, TENSOR_REGISTRY};
use hologram_core::Tensor as CoreTensor;

/// Create tensor from buffer with given shape (JSON-encoded array)
///
/// # Arguments
///
/// * `buffer_handle` - Handle to the buffer
/// * `shape_json` - JSON-encoded shape array, e.g., "[4, 6]" for 4x6 matrix
///
/// # Returns
///
/// Handle to the created tensor
///
/// # Example
///
/// ```json
/// "[4, 6]"  // 4x6 matrix
/// "[2, 3, 4]"  // 2x3x4 tensor
/// ```
pub fn tensor_from_buffer(buffer_handle: u64, shape_json: String) -> u64 {
    let tensor_handle = generate_handle();

    tracing::debug!(
        buffer_handle = buffer_handle,
        tensor_handle = tensor_handle,
        shape_json = shape_json,
        "Creating tensor from buffer"
    );

    // Parse shape JSON
    let shape: Vec<usize> = serde_json::from_str(&shape_json).expect("Failed to parse shape JSON");

    // Get buffer from registry
    let buf_registry = lock_registry(&BUFFER_REGISTRY);
    let buffer = buf_registry
        .get(&buffer_handle)
        .unwrap_or_else(|| panic!("Buffer handle {} not found", buffer_handle));

    // Create tensor
    // We need to clone the buffer since Tensor takes ownership
    // For the FFI, we'll work around this by creating a new buffer reference
    let tensor = CoreTensor::from_buffer(buffer.clone(), shape).expect("Failed to create tensor");

    // Store tensor in registry
    lock_registry(&TENSOR_REGISTRY).insert(tensor_handle, tensor);

    tracing::info!(
        buffer_handle = buffer_handle,
        tensor_handle = tensor_handle,
        "Tensor created successfully"
    );

    tensor_handle
}

/// Get tensor shape as JSON-encoded array
///
/// # Arguments
///
/// * `tensor_handle` - Handle to the tensor
///
/// # Returns
///
/// JSON-encoded shape array, e.g., "[4, 6]"
pub fn tensor_shape(tensor_handle: u64) -> String {
    let registry = lock_registry(&TENSOR_REGISTRY);
    let tensor = registry
        .get(&tensor_handle)
        .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

    serde_json::to_string(tensor.shape()).expect("Failed to serialize shape")
}

/// Get number of dimensions
///
/// # Arguments
///
/// * `tensor_handle` - Handle to the tensor
///
/// # Returns
///
/// Number of dimensions
pub fn tensor_ndim(tensor_handle: u64) -> u32 {
    let registry = lock_registry(&TENSOR_REGISTRY);
    let tensor = registry
        .get(&tensor_handle)
        .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

    tensor.ndim() as u32
}

/// Get total number of elements
///
/// # Arguments
///
/// * `tensor_handle` - Handle to the tensor
///
/// # Returns
///
/// Total number of elements in the tensor
pub fn tensor_numel(tensor_handle: u64) -> u32 {
    let registry = lock_registry(&TENSOR_REGISTRY);
    let tensor = registry
        .get(&tensor_handle)
        .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

    tensor.numel() as u32
}

/// Create tensor from buffer with given shape and strides (JSON-encoded arrays)
pub fn tensor_from_buffer_with_strides(buffer_handle: u64, shape_json: String, strides_json: String) -> u64 {
    let tensor_handle = generate_handle();

    // Parse shape and strides JSON
    let shape: Vec<usize> = serde_json::from_str(&shape_json).expect("Failed to parse shape JSON");
    let strides: Vec<usize> = serde_json::from_str(&strides_json).expect("Failed to parse strides JSON");

    // Get buffer from registry
    let buf_registry = lock_registry(&BUFFER_REGISTRY);
    let buffer = buf_registry
        .get(&buffer_handle)
        .unwrap_or_else(|| panic!("Buffer handle {} not found", buffer_handle));

    // Create tensor with explicit strides
    let tensor = CoreTensor::from_buffer_with_strides(buffer.clone(), shape, strides)
        .expect("Failed to create tensor with strides");

    // Store tensor in registry
    lock_registry(&TENSOR_REGISTRY).insert(tensor_handle, tensor);

    tensor_handle
}

/// Get tensor strides as JSON-encoded array
pub fn tensor_strides(tensor_handle: u64) -> String {
    let registry = lock_registry(&TENSOR_REGISTRY);
    let tensor = registry
        .get(&tensor_handle)
        .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

    serde_json::to_string(tensor.strides()).expect("Failed to serialize strides")
}

/// Get tensor offset
pub fn tensor_offset(tensor_handle: u64) -> u32 {
    let registry = lock_registry(&TENSOR_REGISTRY);
    let tensor = registry
        .get(&tensor_handle)
        .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

    tensor.offset() as u32
}

/// Check if tensor is contiguous (row-major)
pub fn tensor_is_contiguous(tensor_handle: u64) -> u8 {
    let registry = lock_registry(&TENSOR_REGISTRY);
    let tensor = registry
        .get(&tensor_handle)
        .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

    if tensor.is_contiguous() {
        1
    } else {
        0
    }
}

/// Create a contiguous copy of the tensor
pub fn tensor_contiguous(executor_handle: u64, tensor_handle: u64) -> u64 {
    use crate::handles::EXECUTOR_REGISTRY;

    let new_tensor_handle = generate_handle();

    // Get executor
    let mut exec_registry = lock_registry(&EXECUTOR_REGISTRY);
    let executor = exec_registry
        .get_mut(&executor_handle)
        .unwrap_or_else(|| panic!("Executor handle {} not found", executor_handle));

    // Get tensor and create contiguous copy
    let contiguous_tensor = {
        let tensor_registry = lock_registry(&TENSOR_REGISTRY);
        let tensor = tensor_registry
            .get(&tensor_handle)
            .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

        // Create contiguous copy
        futures::executor::block_on(tensor.contiguous(executor)).expect("Failed to make tensor contiguous")
    }; // Lock dropped here

    // Store new tensor (safe to lock again now)
    lock_registry(&TENSOR_REGISTRY).insert(new_tensor_handle, contiguous_tensor);

    new_tensor_handle
}

/// Transpose 2D tensor (swap dimensions 0 and 1)
pub fn tensor_transpose(tensor_handle: u64) -> u64 {
    let new_tensor_handle = generate_handle();

    // Get tensor and transpose
    let transposed = {
        let tensor_registry = lock_registry(&TENSOR_REGISTRY);
        let tensor = tensor_registry
            .get(&tensor_handle)
            .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

        // Transpose
        tensor.transpose().expect("Failed to transpose tensor")
    }; // Lock dropped here

    // Store new tensor (safe to lock again now)
    lock_registry(&TENSOR_REGISTRY).insert(new_tensor_handle, transposed);

    new_tensor_handle
}

/// Reshape tensor (must have same number of elements)
pub fn tensor_reshape(executor_handle: u64, tensor_handle: u64, new_shape_json: String) -> u64 {
    use crate::handles::EXECUTOR_REGISTRY;

    let new_tensor_handle = generate_handle();

    // Parse new shape
    let new_shape: Vec<usize> = serde_json::from_str(&new_shape_json).expect("Failed to parse new shape JSON");

    // Get executor
    let mut exec_registry = lock_registry(&EXECUTOR_REGISTRY);
    let executor = exec_registry
        .get_mut(&executor_handle)
        .unwrap_or_else(|| panic!("Executor handle {} not found", executor_handle));

    // Get tensor and reshape
    let reshaped = {
        let tensor_registry = lock_registry(&TENSOR_REGISTRY);
        let tensor = tensor_registry
            .get(&tensor_handle)
            .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

        // Reshape
        futures::executor::block_on(tensor.reshape(executor, new_shape)).expect("Failed to reshape tensor")
    }; // Lock dropped here

    // Store new tensor (safe to lock again now)
    lock_registry(&TENSOR_REGISTRY).insert(new_tensor_handle, reshaped);

    new_tensor_handle
}

/// Permute dimensions according to given order
pub fn tensor_permute(tensor_handle: u64, dims_json: String) -> u64 {
    let new_tensor_handle = generate_handle();

    // Parse dims
    let dims: Vec<usize> = serde_json::from_str(&dims_json).expect("Failed to parse dims JSON");

    // Get tensor and permute
    let permuted = {
        let tensor_registry = lock_registry(&TENSOR_REGISTRY);
        let tensor = tensor_registry
            .get(&tensor_handle)
            .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

        // Permute
        tensor.permute(dims).expect("Failed to permute tensor")
    }; // Lock dropped here

    // Store new tensor (safe to lock again now)
    lock_registry(&TENSOR_REGISTRY).insert(new_tensor_handle, permuted);

    new_tensor_handle
}

/// View tensor as 1D (flattened)
pub fn tensor_view_1d(tensor_handle: u64) -> u64 {
    let new_tensor_handle = generate_handle();

    // Get tensor and flatten
    let flattened = {
        let tensor_registry = lock_registry(&TENSOR_REGISTRY);
        let tensor = tensor_registry
            .get(&tensor_handle)
            .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

        // View as 1D
        tensor.view_1d().expect("Failed to view tensor as 1D")
    }; // Lock dropped here

    // Store new tensor (safe to lock again now)
    lock_registry(&TENSOR_REGISTRY).insert(new_tensor_handle, flattened);

    new_tensor_handle
}

/// Select a single index along a dimension (reduces dimensionality by 1)
pub fn tensor_select(tensor_handle: u64, dim: u32, index: u32) -> u64 {
    let new_tensor_handle = generate_handle();

    // Get tensor and select
    let selected = {
        let tensor_registry = lock_registry(&TENSOR_REGISTRY);
        let tensor = tensor_registry
            .get(&tensor_handle)
            .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

        // Select
        tensor
            .select(dim as usize, index as usize)
            .expect("Failed to select from tensor")
    }; // Lock dropped here

    // Store new tensor (safe to lock again now)
    lock_registry(&TENSOR_REGISTRY).insert(new_tensor_handle, selected);

    new_tensor_handle
}

/// Narrow a dimension to a range [start, start+length)
pub fn tensor_narrow(tensor_handle: u64, dim: u32, start: u32, length: u32) -> u64 {
    let new_tensor_handle = generate_handle();

    // Get tensor and narrow
    let narrowed = {
        let tensor_registry = lock_registry(&TENSOR_REGISTRY);
        let tensor = tensor_registry
            .get(&tensor_handle)
            .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

        // Narrow
        tensor
            .narrow(dim as usize, start as usize, length as usize)
            .expect("Failed to narrow tensor")
    }; // Lock dropped here

    // Store new tensor (safe to lock again now)
    lock_registry(&TENSOR_REGISTRY).insert(new_tensor_handle, narrowed);

    new_tensor_handle
}

/// Slice a dimension with start, end, and step
/// Pass -1 for None values
pub fn tensor_slice(tensor_handle: u64, dim: u32, start: i32, end: i32, step: i32) -> u64 {
    let new_tensor_handle = generate_handle();

    // Convert -1 to None
    let start_opt = if start < 0 { None } else { Some(start as usize) };
    let end_opt = if end < 0 { None } else { Some(end as usize) };
    let step_opt = if step < 0 { None } else { Some(step as usize) };

    // Get tensor and slice
    let sliced = {
        let tensor_registry = lock_registry(&TENSOR_REGISTRY);
        let tensor = tensor_registry
            .get(&tensor_handle)
            .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

        // Slice
        tensor
            .slice(dim as usize, start_opt, end_opt, step_opt)
            .expect("Failed to slice tensor")
    }; // Lock dropped here

    // Store new tensor (safe to lock again now)
    lock_registry(&TENSOR_REGISTRY).insert(new_tensor_handle, sliced);

    new_tensor_handle
}

/// Matrix multiplication: C = A @ B (for 2D tensors)
pub fn tensor_matmul(executor_handle: u64, a_handle: u64, b_handle: u64) -> u64 {
    use crate::handles::EXECUTOR_REGISTRY;

    let new_tensor_handle = generate_handle();

    // Get executor
    let mut exec_registry = lock_registry(&EXECUTOR_REGISTRY);
    let executor = exec_registry
        .get_mut(&executor_handle)
        .unwrap_or_else(|| panic!("Executor handle {} not found", executor_handle));

    // Get tensors and perform matmul
    let result = {
        let tensor_registry = lock_registry(&TENSOR_REGISTRY);
        let tensor_a = tensor_registry
            .get(&a_handle)
            .unwrap_or_else(|| panic!("Tensor A handle {} not found", a_handle));
        let tensor_b = tensor_registry
            .get(&b_handle)
            .unwrap_or_else(|| panic!("Tensor B handle {} not found", b_handle));

        // Matrix multiply (block on async for FFI compatibility)
        futures::executor::block_on(tensor_a.matmul(executor, tensor_b)).expect("Failed to perform matmul")
    }; // Lock dropped here

    // Store result tensor (safe to lock again now)
    lock_registry(&TENSOR_REGISTRY).insert(new_tensor_handle, result);

    new_tensor_handle
}

/// Check if two tensors are broadcast-compatible
pub fn tensor_is_broadcast_compatible_with(a_handle: u64, b_handle: u64) -> u8 {
    let tensor_registry = lock_registry(&TENSOR_REGISTRY);
    let tensor_a = tensor_registry
        .get(&a_handle)
        .unwrap_or_else(|| panic!("Tensor A handle {} not found", a_handle));
    let tensor_b = tensor_registry
        .get(&b_handle)
        .unwrap_or_else(|| panic!("Tensor B handle {} not found", b_handle));

    if tensor_a.is_broadcast_compatible_with(tensor_b) {
        1
    } else {
        0
    }
}

/// Compute broadcast result shape for two shapes (JSON-encoded arrays)
pub fn tensor_broadcast_shapes(shape_a_json: String, shape_b_json: String) -> String {
    // Parse shapes
    let shape_a: Vec<usize> = serde_json::from_str(&shape_a_json).expect("Failed to parse shape A JSON");
    let shape_b: Vec<usize> = serde_json::from_str(&shape_b_json).expect("Failed to parse shape B JSON");

    // Compute broadcast shape
    let result_shape =
        CoreTensor::<f32>::broadcast_shapes(&shape_a, &shape_b).expect("Shapes are not broadcast-compatible");

    // Serialize result
    serde_json::to_string(&result_shape).expect("Failed to serialize result shape")
}

/// Get underlying buffer handle from tensor
pub fn tensor_buffer(tensor_handle: u64) -> u64 {
    let tensor_registry = lock_registry(&TENSOR_REGISTRY);
    let tensor = tensor_registry
        .get(&tensor_handle)
        .unwrap_or_else(|| panic!("Tensor handle {} not found", tensor_handle));

    // Get the buffer from the tensor
    let buffer = tensor.buffer();

    // Find the buffer handle by matching the buffer's class index
    // This is a bit of a workaround since we don't have a direct buffer -> handle mapping
    let buf_registry = lock_registry(&BUFFER_REGISTRY);
    for (handle, buf) in buf_registry.iter() {
        if buf.class_index() == buffer.class_index() {
            return *handle;
        }
    }

    panic!("Could not find buffer handle for tensor's buffer");
}

/// Cleanup tensor and free resources
///
/// # Arguments
///
/// * `tensor_handle` - Handle to the tensor to cleanup
pub fn tensor_cleanup(tensor_handle: u64) {
    tracing::debug!(tensor_handle = tensor_handle, "Cleaning up tensor");

    let mut registry = lock_registry(&TENSOR_REGISTRY);
    if registry.remove(&tensor_handle).is_some() {
        tracing::info!(tensor_handle = tensor_handle, "Tensor cleaned up successfully");
    } else {
        tracing::warn!(tensor_handle = tensor_handle, "Tensor handle not found during cleanup");
    }
}

// ====================================================================================
// DLPack Interoperability
// ====================================================================================

/// Export tensor to DLPack format for zero-copy sharing with PyTorch, JAX, etc.
///
/// Returns a raw pointer to DLManagedTensor as u64. This pointer must be wrapped
/// in a PyCapsule with name "dltensor" on the Python side.
///
/// # Arguments
///
/// * `executor_handle` - Handle to the executor
/// * `tensor_handle` - Handle to the tensor (must be f32 type and contiguous)
///
/// # Returns
///
/// Raw pointer to DLManagedTensor as u64, or 0 on error
///
/// # Safety
///
/// The returned pointer must be:
/// - Wrapped in a PyCapsule with name "dltensor"
/// - Passed to torch.from_dlpack() or similar
/// - The deleter will be called automatically by the consuming framework
///
/// # Example (Python)
///
/// ```python
/// # Get DLPack pointer from Hologram
/// ptr = hologram.tensor_to_dlpack(exec_handle, tensor_handle)
///
/// # Wrap in PyCapsule
/// capsule = PyCapsule(ptr, "dltensor", dlpack_deleter)
///
/// # Convert to PyTorch (zero-copy)
/// pytorch_tensor = torch.from_dlpack(capsule)
/// ```
pub fn tensor_to_dlpack(executor_handle: u64, tensor_handle: u64) -> u64 {
    use crate::handles::EXECUTOR_REGISTRY;

    tracing::debug!(
        executor_handle = executor_handle,
        tensor_handle = tensor_handle,
        "Exporting tensor to DLPack"
    );

    // Get executor
    let exec_registry = lock_registry(&EXECUTOR_REGISTRY);
    let executor = match exec_registry.get(&executor_handle) {
        Some(exec) => exec,
        None => {
            tracing::error!(
                executor_handle = executor_handle,
                "Executor handle not found for DLPack export"
            );
            return 0;
        }
    };

    // Get tensor
    let tensor_registry = lock_registry(&TENSOR_REGISTRY);
    let tensor = match tensor_registry.get(&tensor_handle) {
        Some(t) => t,
        None => {
            tracing::error!(
                tensor_handle = tensor_handle,
                "Tensor handle not found for DLPack export"
            );
            return 0;
        }
    };

    // Export to DLPack
    match executor.tensor_to_dlpack(tensor) {
        Ok(dlpack_tensor) => {
            let ptr = Box::into_raw(dlpack_tensor) as u64;
            tracing::info!(
                executor_handle = executor_handle,
                tensor_handle = tensor_handle,
                dlpack_ptr = ptr,
                "Tensor exported to DLPack successfully"
            );
            ptr
        }
        Err(e) => {
            tracing::error!(
                executor_handle = executor_handle,
                tensor_handle = tensor_handle,
                error = ?e,
                "Failed to export tensor to DLPack"
            );
            0
        }
    }
}

/// Get DLPack device type for the current executor backend
///
/// Returns device type as u32:
/// - 1: CPU
/// - 2: CUDA
/// - 0: Unknown/error
///
/// # Arguments
///
/// * `executor_handle` - Handle to the executor
///
/// # Returns
///
/// Device type code
pub fn tensor_dlpack_device_type(executor_handle: u64) -> u32 {
    use crate::handles::EXECUTOR_REGISTRY;

    let exec_registry = lock_registry(&EXECUTOR_REGISTRY);
    let executor = match exec_registry.get(&executor_handle) {
        Some(exec) => exec,
        None => return 0,
    };

    let device = executor.get_device_type();
    device.device_type as u32
}

/// Get DLPack device ID for the current executor backend
///
/// Returns device ID (e.g., GPU 0, GPU 1, etc.)
///
/// # Arguments
///
/// * `executor_handle` - Handle to the executor
///
/// # Returns
///
/// Device ID
pub fn tensor_dlpack_device_id(executor_handle: u64) -> i32 {
    use crate::handles::EXECUTOR_REGISTRY;

    let exec_registry = lock_registry(&EXECUTOR_REGISTRY);
    let executor = match exec_registry.get(&executor_handle) {
        Some(exec) => exec,
        None => return -1,
    };

    let device = executor.get_device_type();
    device.device_id
}

/// Import f32 tensor from DLPack capsule (universal framework support)
///
/// Creates a Hologram tensor from any DLPack-compatible framework (PyTorch, JAX,
/// TensorFlow, CuPy, etc.) via DLPack protocol.
///
/// # Arguments
///
/// * `executor_handle` - Executor handle
/// * `dlpack_ptr` - Pointer to DLManagedTensor (from PyCapsule)
///
/// # Returns
///
/// Returns JSON string with {"tensor_handle": u64, "buffer_handle": u64} on success,
/// empty string on error
///
/// # Notes
///
/// - Currently supports f32 tensors only (most common use case)
/// - Data is copied from external framework to Hologram
/// - Export (via tensor_to_dlpack) is zero-copy
/// - Import is copy-based for safety with Hologram's class-based memory
pub fn tensor_from_dlpack_capsule(executor_handle: u64, dlpack_ptr: u64) -> String {
    let mut exec_registry = lock_registry(&EXECUTOR_REGISTRY);
    let executor = match exec_registry.get_mut(&executor_handle) {
        Some(exec) => exec,
        None => {
            tracing::error!("Executor handle not found for DLPack import");
            return String::new();
        }
    };

    // Import f32 tensor
    match executor.tensor_from_dlpack::<f32>(dlpack_ptr) {
        Ok(tensor) => {
            // Register the tensor
            let mut tensor_registry = lock_registry(&TENSOR_REGISTRY);
            let tensor_handle = generate_handle();
            tensor_registry.insert(tensor_handle, tensor);
            tracing::info!("Imported f32 tensor from DLPack: tensor_handle={}", tensor_handle);

            // Return tensor handle and 0 for buffer handle (tensor owns the buffer)
            // We don't separately register the buffer to avoid ownership issues
            serde_json::json!({
                "tensor_handle": tensor_handle,
                "buffer_handle": 0
            })
            .to_string()
        }
        Err(e) => {
            tracing::error!("Failed to import f32 tensor from DLPack: {:?}", e);
            String::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{executor_allocate_buffer, new_executor};
    use crate::handles::clear_all_registries;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_tensor_creation() {
        clear_all_registries();

        let exec = new_executor();
        let buf = executor_allocate_buffer(exec, 24); // 4x6 = 24 elements

        let shape_json = "[4, 6]".to_string();
        let tensor = tensor_from_buffer(buf, shape_json);

        assert!(tensor > 0);

        // Verify tensor properties
        assert_eq!(tensor_ndim(tensor), 2);
        assert_eq!(tensor_numel(tensor), 24);

        let shape = tensor_shape(tensor);
        let parsed_shape: Vec<usize> = serde_json::from_str(&shape).unwrap();
        assert_eq!(parsed_shape, vec![4, 6]);
    }

    #[test]
    #[serial]
    fn test_tensor_cleanup() {
        clear_all_registries();

        let exec = new_executor();
        let buf = executor_allocate_buffer(exec, 12);
        let tensor = tensor_from_buffer(buf, "[3, 4]".to_string());

        assert!(lock_registry(&TENSOR_REGISTRY).contains_key(&tensor));

        tensor_cleanup(tensor);

        assert!(!lock_registry(&TENSOR_REGISTRY).contains_key(&tensor));
    }
}
