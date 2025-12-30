//! Math operations

use crate::buffer::{BufferVariant, PyBuffer};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Element-wise addition: C = A + B
///
/// # Example
/// ```python
/// c = hg.ops.vector_add(a, b)
/// ```
#[pyfunction]
pub fn vector_add(a: &PyBuffer, b: &PyBuffer) -> PyResult<PyBuffer> {
    // Validate inputs
    if a.inner.len() != b.inner.len() {
        return Err(PyValueError::new_err(format!(
            "Buffer size mismatch: {} != {}",
            a.inner.len(),
            b.inner.len()
        )));
    }

    if a.inner.dtype() != b.inner.dtype() {
        return Err(PyValueError::new_err(format!(
            "Dtype mismatch: {} != {}",
            a.inner.dtype(),
            b.inner.dtype()
        )));
    }

    // Only f32 supported for now
    let buf_a = a
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;
    let buf_b = b
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    // Allocate output buffer
    let n = a.inner.len();
    let mut executor = a.executor.write();
    let mut buf_c = executor
        .allocate::<f32>(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    // Execute operation
    hologram_core::ops::math::vector_add(&mut executor, buf_a, buf_b, &mut buf_c, n)
        .map_err(|e| PyRuntimeError::new_err(format!("vector_add failed: {}", e)))?;

    Ok(PyBuffer::from_f32(buf_c, a.executor.clone()))
}

/// Element-wise multiplication: C = A * B
#[pyfunction]
pub fn vector_mul(a: &PyBuffer, b: &PyBuffer) -> PyResult<PyBuffer> {
    if a.inner.len() != b.inner.len() {
        return Err(PyValueError::new_err(format!(
            "Buffer size mismatch: {} != {}",
            a.inner.len(),
            b.inner.len()
        )));
    }

    let buf_a = a
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;
    let buf_b = b
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let n = a.inner.len();
    let mut executor = a.executor.write();
    let mut buf_c = executor
        .allocate::<f32>(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    hologram_core::ops::math::vector_mul(&mut executor, buf_a, buf_b, &mut buf_c, n)
        .map_err(|e| PyRuntimeError::new_err(format!("vector_mul failed: {}", e)))?;

    Ok(PyBuffer::from_f32(buf_c, a.executor.clone()))
}

/// Element-wise division: C = A / B
#[pyfunction]
pub fn vector_div(a: &PyBuffer, b: &PyBuffer) -> PyResult<PyBuffer> {
    if a.inner.len() != b.inner.len() {
        return Err(PyValueError::new_err(format!(
            "Buffer size mismatch: {} != {}",
            a.inner.len(),
            b.inner.len()
        )));
    }

    let buf_a = a
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;
    let buf_b = b
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let n = a.inner.len();
    let mut executor = a.executor.write();
    let mut buf_c = executor
        .allocate::<f32>(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    hologram_core::ops::math::vector_div(&mut executor, buf_a, buf_b, &mut buf_c, n)
        .map_err(|e| PyRuntimeError::new_err(format!("vector_div failed: {}", e)))?;

    Ok(PyBuffer::from_f32(buf_c, a.executor.clone()))
}

/// Negation: C = -A
#[pyfunction]
pub fn neg(a: &PyBuffer) -> PyResult<PyBuffer> {
    let buf_a = a
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let n = a.inner.len();
    let mut executor = a.executor.write();
    let mut buf_b = executor
        .allocate::<f32>(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    hologram_core::ops::math::neg(&mut executor, buf_a, &mut buf_b, n)
        .map_err(|e| PyRuntimeError::new_err(format!("neg failed: {}", e)))?;

    Ok(PyBuffer::from_f32(buf_b, a.executor.clone()))
}

/// Absolute value: C = |A|
#[pyfunction]
pub fn abs(a: &PyBuffer) -> PyResult<PyBuffer> {
    let buf_a = a
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let n = a.inner.len();
    let mut executor = a.executor.write();
    let mut buf_b = executor
        .allocate::<f32>(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    hologram_core::ops::math::abs(&mut executor, buf_a, &mut buf_b, n)
        .map_err(|e| PyRuntimeError::new_err(format!("abs failed: {}", e)))?;

    Ok(PyBuffer::from_f32(buf_b, a.executor.clone()))
}

/// Element-wise minimum: C = min(A, B)
#[pyfunction]
pub fn min(a: &PyBuffer, b: &PyBuffer) -> PyResult<PyBuffer> {
    if a.inner.len() != b.inner.len() {
        return Err(PyValueError::new_err(format!(
            "Buffer size mismatch: {} != {}",
            a.inner.len(),
            b.inner.len()
        )));
    }

    let buf_a = a
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;
    let buf_b = b
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let n = a.inner.len();
    let mut executor = a.executor.write();
    let mut buf_c = executor
        .allocate::<f32>(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    hologram_core::ops::math::min(&mut executor, buf_a, buf_b, &mut buf_c, n)
        .map_err(|e| PyRuntimeError::new_err(format!("min failed: {}", e)))?;

    Ok(PyBuffer::from_f32(buf_c, a.executor.clone()))
}

/// Element-wise maximum: C = max(A, B)
#[pyfunction]
pub fn max(a: &PyBuffer, b: &PyBuffer) -> PyResult<PyBuffer> {
    if a.inner.len() != b.inner.len() {
        return Err(PyValueError::new_err(format!(
            "Buffer size mismatch: {} != {}",
            a.inner.len(),
            b.inner.len()
        )));
    }

    let buf_a = a
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;
    let buf_b = b
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let n = a.inner.len();
    let mut executor = a.executor.write();
    let mut buf_c = executor
        .allocate::<f32>(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    hologram_core::ops::math::max(&mut executor, buf_a, buf_b, &mut buf_c, n)
        .map_err(|e| PyRuntimeError::new_err(format!("max failed: {}", e)))?;

    Ok(PyBuffer::from_f32(buf_c, a.executor.clone()))
}
