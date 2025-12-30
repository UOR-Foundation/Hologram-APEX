//! Loss function operations

use crate::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Mean Squared Error (MSE) loss: loss = mean((predictions - targets)^2)
///
/// # Arguments
/// * `predictions` - Predicted values
/// * `targets` - Target (ground truth) values
///
/// # Returns
/// Scalar loss value (f32)
#[pyfunction]
pub fn mse_loss(predictions: &PyBuffer, targets: &PyBuffer) -> PyResult<f32> {
    if predictions.inner.len() != targets.inner.len() {
        return Err(PyValueError::new_err(format!(
            "Buffer size mismatch: {} != {}",
            predictions.inner.len(),
            targets.inner.len()
        )));
    }

    let buf_pred = predictions
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;
    let buf_target = targets
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let n = predictions.inner.len();
    let mut executor = predictions.executor.write();

    // Loss operations require 3-element output buffer
    let mut buf_output = executor
        .allocate::<f32>(3)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    hologram_core::ops::loss::mse(&mut executor, buf_pred, buf_target, &mut buf_output, n)
        .map_err(|e| PyRuntimeError::new_err(format!("mse_loss failed: {}", e)))?;

    // Extract result from output[0]
    let result = buf_output
        .to_vec(&executor)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read result: {}", e)))?;

    Ok(result[0])
}

/// Cross-Entropy loss: loss = -mean(Σ target * log(prediction))
///
/// # Arguments
/// * `predictions` - Predicted probabilities (should be in [0, 1])
/// * `targets` - Target labels (typically one-hot encoded)
///
/// # Returns
/// Scalar loss value (f32)
#[pyfunction]
pub fn cross_entropy_loss(predictions: &PyBuffer, targets: &PyBuffer) -> PyResult<f32> {
    if predictions.inner.len() != targets.inner.len() {
        return Err(PyValueError::new_err(format!(
            "Buffer size mismatch: {} != {}",
            predictions.inner.len(),
            targets.inner.len()
        )));
    }

    let buf_pred = predictions
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;
    let buf_target = targets
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let n = predictions.inner.len();
    let mut executor = predictions.executor.write();

    // Loss operations require 3-element output buffer
    let mut buf_output = executor
        .allocate::<f32>(3)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    hologram_core::ops::loss::cross_entropy(&mut executor, buf_pred, buf_target, &mut buf_output, n)
        .map_err(|e| PyRuntimeError::new_err(format!("cross_entropy_loss failed: {}", e)))?;

    // Extract result from output[0]
    let result = buf_output
        .to_vec(&executor)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read result: {}", e)))?;

    Ok(result[0])
}
