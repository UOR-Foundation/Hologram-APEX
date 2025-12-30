//! Activation function operations

use crate::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
#[pyfunction]
pub fn sigmoid(a: &PyBuffer) -> PyResult<PyBuffer> {
    let buf_a = a
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let n = a.inner.len();
    let mut executor = a.executor.write();
    let mut buf_b = executor
        .allocate::<f32>(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    hologram_core::ops::activation::sigmoid(&mut executor, buf_a, &mut buf_b, n)
        .map_err(|e| PyRuntimeError::new_err(format!("sigmoid failed: {}", e)))?;

    Ok(PyBuffer::from_f32(buf_b, a.executor.clone()))
}

/// Hyperbolic tangent activation: tanh(x)
#[pyfunction]
pub fn tanh(a: &PyBuffer) -> PyResult<PyBuffer> {
    let buf_a = a
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let n = a.inner.len();
    let mut executor = a.executor.write();
    let mut buf_b = executor
        .allocate::<f32>(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    hologram_core::ops::activation::tanh(&mut executor, buf_a, &mut buf_b, n)
        .map_err(|e| PyRuntimeError::new_err(format!("tanh failed: {}", e)))?;

    Ok(PyBuffer::from_f32(buf_b, a.executor.clone()))
}

/// ReLU activation: max(0, x)
/// Note: This is also available in the math module as ops::math::relu
#[pyfunction]
pub fn relu(a: &PyBuffer) -> PyResult<PyBuffer> {
    let buf_a = a
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let n = a.inner.len();
    let mut executor = a.executor.write();
    let mut buf_b = executor
        .allocate::<f32>(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    hologram_core::ops::math::relu(&mut executor, buf_a, &mut buf_b, n)
        .map_err(|e| PyRuntimeError::new_err(format!("relu failed: {}", e)))?;

    Ok(PyBuffer::from_f32(buf_b, a.executor.clone()))
}

/// Softmax activation (not yet implemented in hologram-stdlib)
/// Placeholder for future implementation
#[pyfunction]
pub fn softmax(_a: &PyBuffer) -> PyResult<PyBuffer> {
    Err(PyRuntimeError::new_err(
        "softmax not yet implemented in hologram-stdlib",
    ))
}
