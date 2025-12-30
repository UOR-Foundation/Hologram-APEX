//! PyBuffer - Python wrapper around hologram_core::Buffer with NumPy integration

use numpy::{PyArray1, ToPyArray};
use parking_lot::RwLock;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::sync::Arc;

/// Type-erased buffer variant to handle multiple dtypes
pub enum BufferVariant {
    F32(hologram_core::Buffer<f32>),
    F64(hologram_core::Buffer<f64>),
    I32(hologram_core::Buffer<i32>),
    U32(hologram_core::Buffer<u32>),
}

impl BufferVariant {
    /// Get the size (number of elements) of the buffer
    pub fn len(&self) -> usize {
        match self {
            BufferVariant::F32(b) => b.len(),
            BufferVariant::F64(b) => b.len(),
            BufferVariant::I32(b) => b.len(),
            BufferVariant::U32(b) => b.len(),
        }
    }

    /// Get dtype as string
    pub fn dtype(&self) -> &str {
        match self {
            BufferVariant::F32(_) => "float32",
            BufferVariant::F64(_) => "float64",
            BufferVariant::I32(_) => "int32",
            BufferVariant::U32(_) => "uint32",
        }
    }

    /// Get a reference to the f32 buffer (if it is one)
    pub fn as_f32(&self) -> Option<&hologram_core::Buffer<f32>> {
        match self {
            BufferVariant::F32(b) => Some(b),
            _ => None,
        }
    }

    /// Get a mutable reference to the f32 buffer (if it is one)
    pub fn as_f32_mut(&mut self) -> Option<&mut hologram_core::Buffer<f32>> {
        match self {
            BufferVariant::F32(b) => Some(b),
            _ => None,
        }
    }

    /// Take ownership of f32 buffer
    pub fn into_f32(self) -> Option<hologram_core::Buffer<f32>> {
        match self {
            BufferVariant::F32(b) => Some(b),
            _ => None,
        }
    }
}

/// Atlas memory buffer
///
/// # Example
/// ```python
/// import numpy as np
/// import hologram as hg
///
/// exec = hg.Executor()
/// a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
/// buf = exec.from_numpy(a)
///
/// # Get data back
/// result = buf.to_numpy()
/// ```
#[pyclass(name = "Buffer")]
pub struct PyBuffer {
    pub(crate) inner: BufferVariant,
    pub(crate) executor: Arc<RwLock<hologram_core::Executor>>,
}

#[pymethods]
impl PyBuffer {
    /// Buffer shape (1D for now)
    #[getter]
    fn shape(&self, py: Python) -> PyResult<Py<PyTuple>> {
        let shape = PyTuple::new(py, &[self.inner.len()]);
        Ok(shape.into())
    }

    /// NumPy dtype
    #[getter]
    fn dtype(&self) -> String {
        self.inner.dtype().to_string()
    }

    /// Number of elements
    #[getter]
    fn size(&self) -> usize {
        self.inner.len()
    }

    /// Copy buffer to NumPy array
    ///
    /// # Returns
    /// NumPy array (copy of Atlas buffer data)
    ///
    /// # Example
    /// ```python
    /// buf = exec.allocate(100)
    /// array = buf.to_numpy()  # Returns numpy array
    /// ```
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        match &self.inner {
            BufferVariant::F32(buf) => {
                let executor = self.executor.read();
                let data = buf
                    .to_vec(&*executor)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to read buffer: {}", e)))?;
                Ok(data.to_pyarray(py).to_object(py))
            }
            BufferVariant::F64(buf) => {
                let executor = self.executor.read();
                let data = buf
                    .to_vec(&*executor)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to read buffer: {}", e)))?;
                Ok(data.to_pyarray(py).to_object(py))
            }
            BufferVariant::I32(buf) => {
                let executor = self.executor.read();
                let data = buf
                    .to_vec(&*executor)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to read buffer: {}", e)))?;
                Ok(data.to_pyarray(py).to_object(py))
            }
            BufferVariant::U32(buf) => {
                let executor = self.executor.read();
                let data = buf
                    .to_vec(&*executor)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to read buffer: {}", e)))?;
                Ok(data.to_pyarray(py).to_object(py))
            }
        }
    }

    /// NumPy array protocol - enables np.array(buffer)
    fn __array__(&self, py: Python) -> PyResult<PyObject> {
        self.to_numpy(py)
    }

    fn __repr__(&self) -> String {
        format!("Buffer(size={}, dtype={})", self.inner.len(), self.inner.dtype())
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

impl PyBuffer {
    /// Create a new PyBuffer from a typed buffer
    pub fn from_f32(buffer: hologram_core::Buffer<f32>, executor: Arc<RwLock<hologram_core::Executor>>) -> Self {
        Self {
            inner: BufferVariant::F32(buffer),
            executor,
        }
    }
}
