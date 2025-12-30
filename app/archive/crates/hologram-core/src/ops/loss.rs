//! Loss functions for neural network training
//!
//! **CURRENTLY UNIMPLEMENTED**
//!
//! Loss functions are temporarily unavailable in the operations API due to
//! architectural changes. Please use the Tensor API instead:
//!
//! ```ignore
//! use hologram_core::{Tensor, Executor};
//!
//! let mut exec = Executor::new()?;
//! let predictions = Tensor::<f32>::from_data(&mut exec, &pred_data, vec![batch, classes])?;
//! let targets = Tensor::<f32>::from_data(&mut exec, &target_data, vec![batch, classes])?;
//!
//! // Compute loss using tensor operations
//! // MSE: ((predictions - targets)^2).mean()
//! // Cross-entropy: -(targets * predictions.log()).sum()
//! ```
//!
//! This module will be re-implemented once the ISA instruction set
//! supports the necessary primitives for loss computation.

use crate::buffer::Buffer;
use crate::error::{Error, Result};
use crate::executor::Executor;

pub fn mse<T: bytemuck::Pod + 'static>(
    _exec: &mut Executor,
    _predictions: &Buffer<T>,
    _targets: &Buffer<T>,
    _output: &mut Buffer<T>,
    _n: usize,
) -> Result<()> {
    Err(Error::InvalidOperation(
        "MSE loss is currently unimplemented. Please use the Tensor API instead.".to_string(),
    ))
}

pub fn cross_entropy<T: bytemuck::Pod + 'static>(
    _exec: &mut Executor,
    _predictions: &Buffer<T>,
    _targets: &Buffer<T>,
    _output: &mut Buffer<T>,
    _n: usize,
) -> Result<()> {
    Err(Error::InvalidOperation(
        "Cross-entropy loss is currently unimplemented. Please use the Tensor API instead.".to_string(),
    ))
}

pub fn binary_cross_entropy<T: bytemuck::Pod + 'static>(
    _exec: &mut Executor,
    _predictions: &Buffer<T>,
    _targets: &Buffer<T>,
    _output: &mut Buffer<T>,
    _n: usize,
) -> Result<()> {
    Err(Error::InvalidOperation(
        "Binary cross-entropy loss is currently unimplemented. Please use the Tensor API instead.".to_string(),
    ))
}
