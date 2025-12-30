//! Transcendental function operations (exp, log, sqrt, pow)
//!
//! Note: These are not yet implemented in hologram-stdlib as separate operations.
//! They are used internally by activation functions but not exposed as public ops.

use crate::buffer::PyBuffer;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Exponential: e^x
/// Note: Not yet exposed as a public operation in hologram-stdlib
#[pyfunction]
pub fn exp(_a: &PyBuffer) -> PyResult<PyBuffer> {
    Err(PyRuntimeError::new_err(
        "exp not yet exposed as public op in hologram-stdlib (used internally by activations)",
    ))
}

/// Natural logarithm: ln(x)
/// Note: Not yet exposed as a public operation in hologram-stdlib
#[pyfunction]
pub fn log(_a: &PyBuffer) -> PyResult<PyBuffer> {
    Err(PyRuntimeError::new_err(
        "log not yet exposed as public op in hologram-stdlib (used internally by activations)",
    ))
}

/// Square root: √x
/// Note: Not yet exposed as a public operation in hologram-stdlib
#[pyfunction]
pub fn sqrt(_a: &PyBuffer) -> PyResult<PyBuffer> {
    Err(PyRuntimeError::new_err(
        "sqrt not yet exposed as public op in hologram-stdlib",
    ))
}

/// Power: x^y
/// Note: Not yet exposed as a public operation in hologram-stdlib
#[pyfunction]
pub fn pow(_a: &PyBuffer, _exponent: f32) -> PyResult<PyBuffer> {
    Err(PyRuntimeError::new_err(
        "pow not yet exposed as public op in hologram-stdlib",
    ))
}
