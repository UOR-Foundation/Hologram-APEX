//! PyExecutor - Python wrapper around hologram_core::Executor

use numpy::{PyArray1, PyReadonlyArray1};
use parking_lot::RwLock;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;

use crate::buffer::{BufferVariant, PyBuffer};

/// Atlas executor for running kernels
///
/// # Example
/// ```python
/// import hologram as hg
/// exec = hg.Executor()
/// buf = exec.allocate(1024)  # Allocate 1024 f32 elements
/// ```
#[pyclass(name = "Executor")]
pub struct PyExecutor {
    pub(crate) executor: Arc<RwLock<hologram_core::Executor>>,
}

#[pymethods]
impl PyExecutor {
    /// Create a new Atlas executor
    #[new]
    fn new() -> PyResult<Self> {
        let executor = hologram_core::Executor::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create executor: {}", e)))?;

        Ok(Self {
            executor: Arc::new(RwLock::new(executor)),
        })
    }

    /// Allocate a buffer in Atlas linear pool
    ///
    /// # Arguments
    /// * `size` - Number of elements
    /// * `dtype` - Data type ("float32", "float64", "int32", "uint32")
    ///
    /// # Returns
    /// Buffer object
    ///
    /// # Example
    /// ```python
    /// buf = exec.allocate(1024, dtype="float32")
    /// ```
    #[pyo3(signature = (size, dtype="float32"))]
    fn allocate(&self, size: usize, dtype: &str) -> PyResult<PyBuffer> {
        let variant = match dtype {
            "float32" | "f32" => {
                let mut executor = self.executor.write();
                let buffer = executor
                    .allocate(size)
                    .map_err(|e| PyRuntimeError::new_err(format!("Allocation failed: {}", e)))?;
                BufferVariant::F32(buffer)
            }
            "float64" | "f64" => {
                let mut executor = self.executor.write();
                let buffer = executor
                    .allocate(size)
                    .map_err(|e| PyRuntimeError::new_err(format!("Allocation failed: {}", e)))?;
                BufferVariant::F64(buffer)
            }
            "int32" | "i32" => {
                let mut executor = self.executor.write();
                let buffer = executor
                    .allocate(size)
                    .map_err(|e| PyRuntimeError::new_err(format!("Allocation failed: {}", e)))?;
                BufferVariant::I32(buffer)
            }
            "uint32" | "u32" => {
                let mut executor = self.executor.write();
                let buffer = executor
                    .allocate(size)
                    .map_err(|e| PyRuntimeError::new_err(format!("Allocation failed: {}", e)))?;
                BufferVariant::U32(buffer)
            }
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unsupported dtype: {}. Supported: float32, float64, int32, uint32",
                    dtype
                )));
            }
        };

        Ok(PyBuffer {
            inner: variant,
            executor: Arc::clone(&self.executor),
        })
    }

    /// Create Atlas buffer from NumPy array (copy)
    ///
    /// # Arguments
    /// * `array` - NumPy array (contiguous, supported dtype)
    ///
    /// # Returns
    /// Buffer with copied data
    ///
    /// # Example
    /// ```python
    /// import numpy as np
    /// a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    /// buf = exec.from_numpy(a)
    /// ```
    fn from_numpy(&self, py: Python, array: PyObject) -> PyResult<PyBuffer> {
        // Try to convert to different numpy array types
        // Try f32 first (most common)
        if let Ok(arr) = array.extract::<PyReadonlyArray1<f32>>(py) {
            let data = arr.as_slice()?;
            let mut executor = self.executor.write();
            let mut buffer = executor
                .allocate(data.len())
                .map_err(|e| PyRuntimeError::new_err(format!("Allocation failed: {}", e)))?;

            buffer
                .copy_from_slice(&mut executor, data)
                .map_err(|e| PyRuntimeError::new_err(format!("Copy failed: {}", e)))?;

            return Ok(PyBuffer {
                inner: BufferVariant::F32(buffer),
                executor: Arc::clone(&self.executor),
            });
        }

        // Try f64
        if let Ok(arr) = array.extract::<PyReadonlyArray1<f64>>(py) {
            let data = arr.as_slice()?;
            let mut executor = self.executor.write();
            let mut buffer = executor
                .allocate(data.len())
                .map_err(|e| PyRuntimeError::new_err(format!("Allocation failed: {}", e)))?;

            buffer
                .copy_from_slice(&mut executor, data)
                .map_err(|e| PyRuntimeError::new_err(format!("Copy failed: {}", e)))?;

            return Ok(PyBuffer {
                inner: BufferVariant::F64(buffer),
                executor: Arc::clone(&self.executor),
            });
        }

        // Try i32
        if let Ok(arr) = array.extract::<PyReadonlyArray1<i32>>(py) {
            let data = arr.as_slice()?;
            let mut executor = self.executor.write();
            let mut buffer = executor
                .allocate(data.len())
                .map_err(|e| PyRuntimeError::new_err(format!("Allocation failed: {}", e)))?;

            buffer
                .copy_from_slice(&mut executor, data)
                .map_err(|e| PyRuntimeError::new_err(format!("Copy failed: {}", e)))?;

            return Ok(PyBuffer {
                inner: BufferVariant::I32(buffer),
                executor: Arc::clone(&self.executor),
            });
        }

        // Try u32
        if let Ok(arr) = array.extract::<PyReadonlyArray1<u32>>(py) {
            let data = arr.as_slice()?;
            let mut executor = self.executor.write();
            let mut buffer = executor
                .allocate(data.len())
                .map_err(|e| PyRuntimeError::new_err(format!("Allocation failed: {}", e)))?;

            buffer
                .copy_from_slice(&mut executor, data)
                .map_err(|e| PyRuntimeError::new_err(format!("Copy failed: {}", e)))?;

            return Ok(PyBuffer {
                inner: BufferVariant::U32(buffer),
                executor: Arc::clone(&self.executor),
            });
        }

        Err(PyRuntimeError::new_err(
            "Array must be 1D with dtype float32, float64, int32, or uint32",
        ))
    }

    fn __repr__(&self) -> String {
        "Executor()".to_string()
    }
}
