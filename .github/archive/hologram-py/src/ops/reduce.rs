//! Reduction operations (sum, max, min)

use crate::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Sum reduction: result = Σ buffer[i]
///
/// Returns a scalar value (f32) containing the sum of all elements.
/// Internally uses a 3-element output buffer and extracts the result.
#[pyfunction]
pub fn sum(a: &PyBuffer) -> PyResult<f32> {
    let buf_a = a
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let n = a.inner.len();
    let mut executor = a.executor.write();

    // Reduction requires 3-element output buffer (for temporaries)
    let mut buf_output = executor
        .allocate::<f32>(3)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    hologram_core::ops::reduce::sum(&mut executor, buf_a, &mut buf_output, n)
        .map_err(|e| PyRuntimeError::new_err(format!("sum failed: {}", e)))?;

    // Extract result from output[0]
    let result = buf_output
        .to_vec(&executor)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read result: {}", e)))?;

    Ok(result[0])
}

/// Maximum reduction: result = max(buffer[i])
///
/// Returns a scalar value (f32) containing the maximum element.
#[pyfunction]
pub fn max_reduce(a: &PyBuffer) -> PyResult<f32> {
    let buf_a = a
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let n = a.inner.len();
    let mut executor = a.executor.write();

    // Reduction requires 3-element output buffer
    let mut buf_output = executor
        .allocate::<f32>(3)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    hologram_core::ops::reduce::max(&mut executor, buf_a, &mut buf_output, n)
        .map_err(|e| PyRuntimeError::new_err(format!("max_reduce failed: {}", e)))?;

    // Extract result from output[0]
    let result = buf_output
        .to_vec(&executor)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read result: {}", e)))?;

    Ok(result[0])
}

/// Minimum reduction: result = min(buffer[i])
///
/// Returns a scalar value (f32) containing the minimum element.
#[pyfunction]
pub fn min_reduce(a: &PyBuffer) -> PyResult<f32> {
    let buf_a = a
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let n = a.inner.len();
    let mut executor = a.executor.write();

    // Reduction requires 3-element output buffer
    let mut buf_output = executor
        .allocate::<f32>(3)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    hologram_core::ops::reduce::min(&mut executor, buf_a, &mut buf_output, n)
        .map_err(|e| PyRuntimeError::new_err(format!("min_reduce failed: {}", e)))?;

    // Extract result from output[0]
    let result = buf_output
        .to_vec(&executor)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read result: {}", e)))?;

    Ok(result[0])
}
