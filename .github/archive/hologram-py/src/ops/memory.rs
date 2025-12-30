//! Memory operations

use crate::buffer::PyBuffer;
use hologram_core::ops;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Copy src buffer into dst buffer (in-place)
///
/// # Arguments
/// * `src` - Source buffer
/// * `dst` - Destination buffer (modified in-place)
///
/// # Example
/// ```python
/// hg.ops.copy(src, dst)  # dst now contains src's data
/// ```
#[pyfunction]
pub fn copy(src: &PyBuffer, dst: &mut PyBuffer) -> PyResult<()> {
    if src.inner.len() != dst.inner.len() {
        return Err(PyValueError::new_err(format!(
            "Buffer size mismatch: src={}, dst={}",
            src.inner.len(),
            dst.inner.len()
        )));
    }

    let buf_src = src
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;
    let buf_dst = dst
        .inner
        .as_f32_mut()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let mut executor = src.executor.write();
    hologram_core::ops::memory::copy(&mut executor, buf_src, buf_dst)
        .map_err(|e| PyRuntimeError::new_err(format!("copy failed: {}", e)))?;

    Ok(())
}

/// Fill buffer with a constant value (in-place)
///
/// # Arguments
/// * `buffer` - Buffer to fill (modified in-place)
/// * `value` - Value to fill with
///
/// # Example
/// ```python
/// hg.ops.fill(buf, 42.0)  # Fill buffer with 42.0
/// ```
#[pyfunction]
pub fn fill(buffer: &mut PyBuffer, value: f32) -> PyResult<()> {
    let buf = buffer
        .inner
        .as_f32_mut()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    let mut executor = buffer.executor.write();
    hologram_core::ops::memory::fill(&mut executor, buf, value)
        .map_err(|e| PyRuntimeError::new_err(format!("fill failed: {}", e)))?;

    Ok(())
}
