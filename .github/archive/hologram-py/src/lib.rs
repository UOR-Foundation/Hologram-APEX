//! Hologram Python Bindings
//!
//! PyO3-based Python extension module for Hologram Atlas runtime.
//! Provides a PyTorch-like API for executing operations on Atlas.

use pyo3::prelude::*;

mod buffer;
mod executor;
mod ops;

use buffer::PyBuffer;
use executor::PyExecutor;

/// Hologram Python extension module
#[pymodule]
fn _hologram(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Register core types
    m.add_class::<PyExecutor>()?;
    m.add_class::<PyBuffer>()?;

    // Register math operations
    m.add_function(wrap_pyfunction!(ops::math::vector_add, m)?)?;
    m.add_function(wrap_pyfunction!(ops::math::vector_mul, m)?)?;
    m.add_function(wrap_pyfunction!(ops::math::vector_div, m)?)?;
    m.add_function(wrap_pyfunction!(ops::math::neg, m)?)?;
    m.add_function(wrap_pyfunction!(ops::math::abs, m)?)?;
    m.add_function(wrap_pyfunction!(ops::math::min, m)?)?;
    m.add_function(wrap_pyfunction!(ops::math::max, m)?)?;

    // Register activation functions
    m.add_function(wrap_pyfunction!(ops::activation::relu, m)?)?;
    m.add_function(wrap_pyfunction!(ops::activation::sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(ops::activation::tanh, m)?)?;
    m.add_function(wrap_pyfunction!(ops::activation::softmax, m)?)?;

    // Register transcendental functions
    m.add_function(wrap_pyfunction!(ops::transcendental::exp, m)?)?;
    m.add_function(wrap_pyfunction!(ops::transcendental::log, m)?)?;
    m.add_function(wrap_pyfunction!(ops::transcendental::sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(ops::transcendental::pow, m)?)?;

    // Register reduction operations
    m.add_function(wrap_pyfunction!(ops::reduce::sum, m)?)?;
    m.add_function(wrap_pyfunction!(ops::reduce::max_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(ops::reduce::min_reduce, m)?)?;

    // Register linear algebra
    m.add_function(wrap_pyfunction!(ops::linalg::gemm, m)?)?;

    // Register loss functions
    m.add_function(wrap_pyfunction!(ops::loss::mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(ops::loss::cross_entropy_loss, m)?)?;

    // Register memory operations
    m.add_function(wrap_pyfunction!(ops::memory::copy, m)?)?;
    m.add_function(wrap_pyfunction!(ops::memory::fill, m)?)?;

    Ok(())
}
