//! Activation functions for neural networks
//!
//! This module provides standard activation functions used in neural networks.
//! All operations support SIMD acceleration when available (f32, n ≤ 262K elements).
//!
//! ## Available Activations
//!
//! - [`relu`] - Rectified Linear Unit: max(0, x)
//! - [`sigmoid`] - Sigmoid: 1 / (1 + exp(-x))
//! - [`tanh`] - Hyperbolic tangent: tanh(x)
//! - [`gelu`] - Gaussian Error Linear Unit (approximation)
//! - [`softmax`] - Softmax normalization (async)
//!
//! ## Example
//!
//! ```ignore
//! use hologram_core::{Executor, ops};
//!
//! let mut exec = Executor::new()?;
//! let mut input = exec.allocate::<f32>(1024)?;
//! let mut output = exec.allocate::<f32>(1024)?;
//!
//! // Apply ReLU activation
//! ops::activation::relu(&mut exec, &input, &mut output, 1024)?;
//! ```

use crate::buffer::Buffer;
use crate::error::{Error, Result};
use crate::executor::Executor;
use crate::instrumentation::{ExecutionMetrics, Instant};
use crate::ops::parallel::parallel_unary_op;
use crate::sync::write_lock;

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate buffer sizes
fn validate_buffers<T: bytemuck::Pod>(buffers: &[&Buffer<T>], n: usize, op_name: &str) -> Result<()> {
    for (i, buf) in buffers.iter().enumerate() {
        if buf.len() < n {
            return Err(Error::InvalidOperation(format!(
                "{}: buffer {} has length {} but operation requires {}",
                op_name,
                i,
                buf.len(),
                n
            )));
        }
    }
    Ok(())
}

// ============================================================================
// ReLU (Rectified Linear Unit)
// ============================================================================

/// ReLU activation: c[i] = max(0, a[i])
///
/// Rectified Linear Unit activation function.
/// Sets all negative values to zero, passes positive values unchanged.
///
/// # Performance
///
/// For CPU backend, f32 type, n ≤ 262,144: inline SIMD kernel
///
/// # Example
///
/// ```ignore
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
/// let a = exec.allocate::<f32>(1024)?;
/// let mut c = exec.allocate::<f32>(1024)?;
///
/// ops::activation::relu(&mut exec, &a, &mut c, 1024)?;
/// ```
#[tracing::instrument(skip(exec, a, c), fields(n = n))]
pub fn relu<T: bytemuck::Pod + 'static>(exec: &mut Executor, a: &Buffer<T>, c: &mut Buffer<T>, n: usize) -> Result<()> {
    // SIMD fast path - checked FIRST to avoid instrumentation overhead
    if crate::ops::math::try_inline_simd_unary(exec, a, c, n, crate::kernel::inline::relu)? {
        return Ok(());
    }

    // Slow path with instrumentation
    let start = Instant::now();

    validate_buffers(&[a, c], n, "relu")?;

    // Fall back to ISA execution
    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    // Use precompiled RELU program
    let program = &crate::precompiled_programs::RELU;
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    // R4 = element count for bounds checking
    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_a)
        .with_register(hologram_backends::Register::new(3), handle_c)
        .with_register(hologram_backends::Register::new(4), n as u64);

    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("relu", n, start);
    metrics.log();

    Ok(())
}

/// Parallel ReLU: output[i] = max(0, input[i])
///
/// Uses operation-level chunking for large vectors.
pub fn relu_par<T: bytemuck::Pod + Send + Sync + 'static>(
    exec: &mut Executor,
    input: &Buffer<T>,
    output: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    parallel_unary_op(exec, input, output, n, |exec, input, output, chunk_n| {
        relu(exec, input, output, chunk_n)
    })
}

// ============================================================================
// Sigmoid
// ============================================================================

/// Sigmoid activation: c[i] = 1 / (1 + exp(-a[i]))
///
/// Smooth S-shaped activation function that maps values to (0, 1).
///
/// # Example
///
/// ```ignore
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
/// let a = exec.allocate::<f32>(1024)?;
/// let mut c = exec.allocate::<f32>(1024)?;
///
/// ops::activation::sigmoid(&mut exec, &a, &mut c, 1024)?;
/// ```
#[tracing::instrument(skip(exec, a, c), fields(n = n))]
pub fn sigmoid<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    a: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    // SIMD fast path
    if crate::ops::math::try_inline_simd_unary(exec, a, c, n, crate::kernel::inline::sigmoid)? {
        return Ok(());
    }

    let start = Instant::now();
    validate_buffers(&[a, c], n, "sigmoid")?;

    // Fall back to ISA execution
    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    let program = &crate::precompiled_programs::SIGMOID;
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_a)
        .with_register(hologram_backends::Register::new(3), handle_c)
        .with_register(hologram_backends::Register::new(4), n as u64);

    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("sigmoid", n, start);
    metrics.log();

    Ok(())
}

// ============================================================================
// Tanh
// ============================================================================

/// Hyperbolic tangent activation: c[i] = tanh(a[i])
///
/// Maps values to (-1, 1) range.
///
/// # Example
///
/// ```ignore
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
/// let a = exec.allocate::<f32>(1024)?;
/// let mut c = exec.allocate::<f32>(1024)?;
///
/// ops::activation::tanh(&mut exec, &a, &mut c, 1024)?;
/// ```
#[tracing::instrument(skip(exec, a, c), fields(n = n))]
pub fn tanh<T: bytemuck::Pod + 'static>(exec: &mut Executor, a: &Buffer<T>, c: &mut Buffer<T>, n: usize) -> Result<()> {
    // SIMD fast path
    if crate::ops::math::try_inline_simd_unary(exec, a, c, n, crate::kernel::inline::tanh)? {
        return Ok(());
    }

    let start = Instant::now();
    validate_buffers(&[a, c], n, "tanh")?;

    // Fall back to ISA execution
    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    let program = &crate::precompiled_programs::TANH;
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_a)
        .with_register(hologram_backends::Register::new(3), handle_c)
        .with_register(hologram_backends::Register::new(4), n as u64);

    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("tanh", n, start);
    metrics.log();

    Ok(())
}

// ============================================================================
// GELU
// ============================================================================

/// Gaussian Error Linear Unit activation: c[i] = gelu(a[i])
///
/// Smooth approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// # Example
///
/// ```ignore
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
/// let a = exec.allocate::<f32>(1024)?;
/// let mut c = exec.allocate::<f32>(1024)?;
///
/// ops::activation::gelu(&mut exec, &a, &mut c, 1024)?;
/// ```
#[tracing::instrument(skip(exec, a, c), fields(n = n))]
pub fn gelu<T: bytemuck::Pod + 'static>(exec: &mut Executor, a: &Buffer<T>, c: &mut Buffer<T>, n: usize) -> Result<()> {
    // SIMD fast path
    if crate::ops::math::try_inline_simd_unary(exec, a, c, n, crate::kernel::inline::gelu)? {
        return Ok(());
    }

    let start = Instant::now();
    validate_buffers(&[a, c], n, "gelu")?;

    // Fall back to ISA execution
    let handle_a = exec.handle_from_buffer(a)?.id();
    let handle_c = exec.handle_from_buffer(c)?.id();

    let program = &crate::precompiled_programs::GELU;
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_a)
        .with_register(hologram_backends::Register::new(3), handle_c)
        .with_register(hologram_backends::Register::new(4), n as u64);

    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("gelu", n, start);
    metrics.log();

    Ok(())
}

// ============================================================================
// Softmax
// ============================================================================

/// Softmax activation: output[i] = exp(input[i]) / sum(exp(input))
///
/// Normalizes values to probability distribution (sums to 1.0).
///
/// # Example
///
/// ```ignore
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
/// let input = exec.allocate::<f32>(10)?;
/// let mut output = exec.allocate::<f32>(10)?;
///
/// futures::executor::block_on(async {
///     ops::activation::softmax(&mut exec, &input, &mut output, 10).await
/// })?;
/// ```
#[tracing::instrument(skip(exec, input, output), fields(n = n))]
pub async fn softmax<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    input: &Buffer<T>,
    output: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    let start = Instant::now();
    validate_buffers(&[input, output], n, "softmax")?;

    // Softmax requires multiple operations:
    // 1. Find max value (for numerical stability)
    // 2. Subtract max and compute exp
    // 3. Sum exp values
    // 4. Divide each exp by sum

    // For now, use precompiled SOFTMAX program if available
    let handle_input = exec.handle_from_buffer(input)?.id();
    let handle_output = exec.handle_from_buffer(output)?.id();

    let program = &crate::precompiled_programs::SOFTMAX;
    let config = hologram_backends::LaunchConfig::linear(n as u32, 256);

    let params = hologram_backends::ExecutionParams::new(config)
        .with_register(hologram_backends::Register::new(1), handle_input)
        .with_register(hologram_backends::Register::new(3), handle_output)
        .with_register(hologram_backends::Register::new(4), n as u64);

    write_lock(&exec.backend)
        .execute_program_with_params(program, &params)
        .map_err(|e| Error::InvalidOperation(format!("Backend execution failed: {}", e)))?;

    let metrics = ExecutionMetrics::new("softmax", n, start);
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
    fn test_relu() -> Result<()> {
        let mut exec = Executor::new()?;
        let mut input = exec.allocate::<f32>(8)?;
        let mut output = exec.allocate::<f32>(8)?;

        let data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 4.0];
        input.copy_from_slice(&mut exec, &data)?;

        relu(&mut exec, &input, &mut output, 8)?;

        let result = output.to_vec(&exec)?;
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result[4], 2.0);
        assert_eq!(result[5], 3.0);
        assert_eq!(result[6], 0.0);
        assert_eq!(result[7], 4.0);

        Ok(())
    }

    #[test]
    fn test_sigmoid() -> Result<()> {
        let mut exec = Executor::new()?;
        let mut input = exec.allocate::<f32>(3)?;
        let mut output = exec.allocate::<f32>(3)?;

        let data: Vec<f32> = vec![0.0, 1.0, -1.0];
        input.copy_from_slice(&mut exec, &data)?;

        sigmoid(&mut exec, &input, &mut output, 3)?;

        let result = output.to_vec(&exec)?;
        assert!((result[0] - 0.5).abs() < 0.01); // sigmoid(0) = 0.5
        assert!(result[1] > 0.7); // sigmoid(1) ≈ 0.73
        assert!(result[2] < 0.3); // sigmoid(-1) ≈ 0.27

        Ok(())
    }
}
