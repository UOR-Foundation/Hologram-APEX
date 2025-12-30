//! Reduction operations that reduce arrays to single values
//!
//! This module provides standard reduction operations used in machine learning.
//! All operations reduce an input array to a single scalar value.
//!
//! ## Available Reductions
//!
//! - [`sum`] - Sum of all elements
//! - [`mean`] - Arithmetic mean of all elements
//! - [`min`] - Minimum value
//! - [`max`] - Maximum value
//!
//! ## Output Buffer Requirements
//!
//! **IMPORTANT**: All reduction operations require the output buffer to have
//! at least 3 elements for internal temporaries. The result is stored in the
//! first element (index 0).
//!
//! ## Example
//!
//! ```ignore
//! use hologram_core::{Executor, ops};
//!
//! let mut exec = Executor::new()?;
//! let mut input = exec.allocate::<f32>(1024)?;
//! let mut output = exec.allocate::<f32>(3)?;  // Must have 3 elements!
//!
//! // Compute sum
//! ops::reduce::sum(&mut exec, &input, &mut output, 1024)?;
//! let sum_value = output.to_vec(&exec)?[0];  // Result in first element
//! ```

use crate::buffer::Buffer;
use crate::error::{Error, Result};
use crate::executor::Executor;
use crate::instrumentation::{ExecutionMetrics, Instant};
use crate::sync::write_lock;

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate buffer sizes for reduction operations
fn validate_reduction_buffers<T: bytemuck::Pod>(
    input: &Buffer<T>,
    output: &Buffer<T>,
    n: usize,
    op_name: &str,
) -> Result<()> {
    if input.len() < n {
        return Err(Error::InvalidOperation(format!(
            "{}: input buffer has length {} but operation requires {}",
            op_name,
            input.len(),
            n
        )));
    }
    if output.len() < 3 {
        return Err(Error::InvalidOperation(format!(
            "{}: output buffer must have at least 3 elements (has {})",
            op_name,
            output.len()
        )));
    }
    Ok(())
}

// ============================================================================
// Sum Reduction
// ============================================================================

/// Sum reduction: output[0] = sum(input[0..n])
///
/// Computes the sum of all elements in the input array.
///
/// # Output Buffer
///
/// The output buffer must have at least 3 elements. The result is stored
/// in `output[0]`.
///
/// # Example
///
/// ```ignore
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
/// let input = exec.allocate::<f32>(1024)?;
/// let mut output = exec.allocate::<f32>(3)?;
///
/// ops::reduce::sum(&mut exec, &input, &mut output, 1024)?;
/// let sum_value = output.to_vec(&exec)?[0];
/// ```
#[tracing::instrument(skip(exec, input, output), fields(n = n))]
pub fn sum<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    input: &Buffer<T>,
    output: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    let start = Instant::now();
    validate_reduction_buffers(input, output, n, "sum")?;

    let handle_input = exec.handle_from_buffer(input)?.id();
    let handle_output = exec.handle_from_buffer(output)?.id();

    // Build reduction ISA program using ADD instruction
    let ty = crate::isa_builder::type_from_rust_type::<T>();
    let program = crate::isa_builder::build_reduction_op(handle_input, handle_output, n, ty, |dst, src1, src2| {
        hologram_backends::Instruction::ADD { ty, dst, src1, src2 }
    })?;

    // ISA builder embeds buffer handles in instructions, no need for params
    let config = hologram_backends::LaunchConfig::linear(1, 1);

    write_lock(&exec.backend)
        .execute_program(&program, &config)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("reduce_sum", n, start);
    metrics.log();

    Ok(())
}

// ============================================================================
// Mean Reduction
// ============================================================================

/// Mean reduction: output[0] = mean(input[0..n])
///
/// Computes the arithmetic mean of all elements in the input array.
/// This is implemented as sum followed by division.
///
/// # Output Buffer
///
/// The output buffer must have at least 3 elements. The result is stored
/// in `output[0]`.
///
/// # Example
///
/// ```ignore
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
/// let input = exec.allocate::<f32>(1024)?;
/// let mut output = exec.allocate::<f32>(3)?;
///
/// ops::reduce::mean(&mut exec, &input, &mut output, 1024)?;
/// let mean_value = output.to_vec(&exec)?[0];
/// ```
#[tracing::instrument(skip(exec, input, output), fields(n = n))]
pub fn mean<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    input: &Buffer<T>,
    output: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    let start = Instant::now();
    validate_reduction_buffers(input, output, n, "mean")?;

    // Step 1: Compute sum
    sum(exec, input, output, n)?;

    // Step 2: Divide sum by n to get mean
    // For now, use the CPU fallback for this scalar operation
    let mut result = output.to_vec(exec)?;
    result[0] = divide_by_n(result[0], n);
    output.copy_from_slice(exec, &result)?;

    let metrics = ExecutionMetrics::new("reduce_mean", n, start);
    metrics.log();

    Ok(())
}

/// Helper to divide a value by n (generic over numeric types)
fn divide_by_n<T: bytemuck::Pod>(value: T, n: usize) -> T {
    // For f32/f64, we can divide directly
    // For integers, we'll just return the value (mean makes less sense for ints)
    let bytes = bytemuck::bytes_of(&value);

    // Check if it's f32
    if std::mem::size_of::<T>() == 4 {
        let val = f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let mean = val / n as f32;
        return bytemuck::cast(mean);
    }

    // Check if it's f64
    if std::mem::size_of::<T>() == 8 {
        let val = f64::from_ne_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let mean = val / n as f64;
        return bytemuck::cast(mean);
    }

    // For other types, just return the sum
    value
}

// ============================================================================
// Min Reduction
// ============================================================================

/// Min reduction: output[0] = min(input[0..n])
///
/// Finds the minimum value in the input array.
///
/// # Output Buffer
///
/// The output buffer must have at least 3 elements. The result is stored
/// in `output[0]`.
///
/// # Example
///
/// ```ignore
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
/// let input = exec.allocate::<f32>(1024)?;
/// let mut output = exec.allocate::<f32>(3)?;
///
/// ops::reduce::min(&mut exec, &input, &mut output, 1024)?;
/// let min_value = output.to_vec(&exec)?[0];
/// ```
#[tracing::instrument(skip(exec, input, output), fields(n = n))]
pub fn min<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    input: &Buffer<T>,
    output: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    let start = Instant::now();
    validate_reduction_buffers(input, output, n, "min")?;

    let handle_input = exec.handle_from_buffer(input)?.id();
    let handle_output = exec.handle_from_buffer(output)?.id();

    // Build reduction ISA program using MIN instruction
    let ty = crate::isa_builder::type_from_rust_type::<T>();
    let program = crate::isa_builder::build_reduction_op(handle_input, handle_output, n, ty, |dst, src1, src2| {
        hologram_backends::Instruction::MIN { ty, dst, src1, src2 }
    })?;

    // ISA builder embeds buffer handles in instructions, no need for params
    let config = hologram_backends::LaunchConfig::linear(1, 1);

    write_lock(&exec.backend)
        .execute_program(&program, &config)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("reduce_min", n, start);
    metrics.log();

    Ok(())
}

// ============================================================================
// Max Reduction
// ============================================================================

/// Max reduction: output[0] = max(input[0..n])
///
/// Finds the maximum value in the input array.
///
/// # Output Buffer
///
/// The output buffer must have at least 3 elements. The result is stored
/// in `output[0]`.
///
/// # Example
///
/// ```ignore
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
/// let input = exec.allocate::<f32>(1024)?;
/// let mut output = exec.allocate::<f32>(3)?;
///
/// ops::reduce::max(&mut exec, &input, &mut output, 1024)?;
/// let max_value = output.to_vec(&exec)?[0];
/// ```
#[tracing::instrument(skip(exec, input, output), fields(n = n))]
pub fn max<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    input: &Buffer<T>,
    output: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    let start = Instant::now();
    validate_reduction_buffers(input, output, n, "max")?;

    let handle_input = exec.handle_from_buffer(input)?.id();
    let handle_output = exec.handle_from_buffer(output)?.id();

    // Build reduction ISA program using MAX instruction
    let ty = crate::isa_builder::type_from_rust_type::<T>();
    let program = crate::isa_builder::build_reduction_op(handle_input, handle_output, n, ty, |dst, src1, src2| {
        hologram_backends::Instruction::MAX { ty, dst, src1, src2 }
    })?;

    // ISA builder embeds buffer handles in instructions, no need for params
    let config = hologram_backends::LaunchConfig::linear(1, 1);

    write_lock(&exec.backend)
        .execute_program(&program, &config)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("reduce_max", n, start);
    metrics.log();

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() -> Result<()> {
        let mut exec = Executor::new()?;
        let mut input = exec.allocate::<f32>(8)?;
        let mut output = exec.allocate::<f32>(3)?;

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        input.copy_from_slice(&mut exec, &data)?;

        sum(&mut exec, &input, &mut output, 8)?;

        let result = output.to_vec(&exec)?;
        let sum_value = result[0];
        assert!((sum_value - 36.0).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_mean() -> Result<()> {
        let mut exec = Executor::new()?;
        let mut input = exec.allocate::<f32>(4)?;
        let mut output = exec.allocate::<f32>(3)?;

        let data: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];
        input.copy_from_slice(&mut exec, &data)?;

        mean(&mut exec, &input, &mut output, 4)?;

        let result = output.to_vec(&exec)?;
        let mean_value = result[0];
        assert!((mean_value - 5.0).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_min() -> Result<()> {
        let mut exec = Executor::new()?;
        let mut input = exec.allocate::<f32>(6)?;
        let mut output = exec.allocate::<f32>(3)?;

        let data: Vec<f32> = vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0];
        input.copy_from_slice(&mut exec, &data)?;

        min(&mut exec, &input, &mut output, 6)?;

        let result = output.to_vec(&exec)?;
        let min_value = result[0];
        assert!((min_value - 1.0).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_max() -> Result<()> {
        let mut exec = Executor::new()?;
        let mut input = exec.allocate::<f32>(6)?;
        let mut output = exec.allocate::<f32>(3)?;

        let data: Vec<f32> = vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0];
        input.copy_from_slice(&mut exec, &data)?;

        max(&mut exec, &input, &mut output, 6)?;

        let result = output.to_vec(&exec)?;
        let max_value = result[0];
        assert!((max_value - 9.0).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_output_buffer_too_small() {
        let mut exec = Executor::new().unwrap();
        let input = exec.allocate::<f32>(4).unwrap();
        let mut output = exec.allocate::<f32>(1).unwrap(); // Too small!

        let result = sum(&mut exec, &input, &mut output, 4);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must have at least 3 elements"));
    }
}
