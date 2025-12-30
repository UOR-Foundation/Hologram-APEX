//! Linear algebra operations (GEMM)

use crate::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// General Matrix Multiply (GEMM): C = A @ B
///
/// # Arguments
/// * `a` - Input matrix A (M×K)
/// * `b` - Input matrix B (K×N)
/// * `m` - Number of rows in A and C
/// * `n` - Number of columns in B and C
/// * `k` - Number of columns in A and rows in B
///
/// # Returns
/// Output matrix C (M×N)
///
/// # Example
/// ```python
/// # 4×6 @ 6×8 = 4×8
/// c = hg.ops.gemm(a, b, m=4, n=8, k=6)
/// ```
#[pyfunction]
#[pyo3(signature = (a, b, m, n, k))]
pub fn gemm(a: &PyBuffer, b: &PyBuffer, m: usize, n: usize, k: usize) -> PyResult<PyBuffer> {
    // Validate inputs
    let buf_a = a
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;
    let buf_b = b
        .inner
        .as_f32()
        .ok_or_else(|| PyValueError::new_err("Only float32 buffers supported for now"))?;

    // Validate dimensions
    if a.inner.len() < m * k {
        return Err(PyValueError::new_err(format!(
            "Buffer A too small: need {}, have {}",
            m * k,
            a.inner.len()
        )));
    }
    if b.inner.len() < k * n {
        return Err(PyValueError::new_err(format!(
            "Buffer B too small: need {}, have {}",
            k * n,
            b.inner.len()
        )));
    }

    // Allocate output C (M×N)
    let mut executor = a.executor.write();
    let mut buf_c = executor
        .allocate::<f32>(m * n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to allocate output: {}", e)))?;

    // Execute GEMM (note: parameter order is m, k, n in hologram-stdlib)
    hologram_core::ops::linalg::gemm(&mut executor, buf_a, buf_b, &mut buf_c, m, k, n)
        .map_err(|e| PyRuntimeError::new_err(format!("gemm failed: {}", e)))?;

    Ok(PyBuffer::from_f32(buf_c, a.executor.clone()))
}
