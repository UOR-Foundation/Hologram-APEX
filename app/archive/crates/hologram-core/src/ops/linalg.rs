//! Linear algebra operations for matrix computations
//!
//! **CURRENTLY UNIMPLEMENTED**
//!
//! Matrix operations are temporarily unavailable in the operations API due to
//! architectural changes. Please use the Tensor API instead:
//!
//! ```ignore
//! use hologram_core::{Tensor, Executor};
//!
//! let mut exec = Executor::new()?;
//! let a = Tensor::<f32>::from_data(&mut exec, &a_data, vec![m, k])?;
//! let b = Tensor::<f32>::from_data(&mut exec, &b_data, vec![k, n])?;
//!
//! // Matrix multiplication using Tensor API
//! let c = a.matmul(&exec, &b)?;
//! ```
//!
//! This module will be re-implemented once the ISA instruction set
//! supports the necessary primitives for matrix operations.

use crate::buffer::Buffer;
use crate::error::{Error, Result};
use crate::executor::Executor;

pub fn gemm<T: bytemuck::Pod + 'static>(
    _exec: &mut Executor,
    _a: &Buffer<T>,
    _b: &Buffer<T>,
    _c: &mut Buffer<T>,
    _m: usize,
    _k: usize,
    _n: usize,
) -> Result<()> {
    Err(Error::InvalidOperation(
        "GEMM is currently unimplemented. Please use the Tensor API (tensor.matmul()) instead.".to_string(),
    ))
}

pub fn matvec<T: bytemuck::Pod + 'static>(
    _exec: &mut Executor,
    _a: &Buffer<T>,
    _x: &Buffer<T>,
    _y: &mut Buffer<T>,
    _m: usize,
    _n: usize,
) -> Result<()> {
    Err(Error::InvalidOperation(
        "Matvec is currently unimplemented. Please use the Tensor API (tensor.matmul()) instead.".to_string(),
    ))
}
