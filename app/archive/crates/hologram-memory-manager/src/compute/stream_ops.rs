//! Stream Operations
//!
//! Defines the `StreamOp` trait for composable scalar operations on chunks.
//!
//! ## Design
//!
//! Stream operations are stateless transformations that process chunks using
//! scalar operations on the chunk data. They can be chained in a pipeline
//! to create complex data processing flows.
//!
//! ## Zero-Copy Architecture
//!
//! Operations work directly on CPU-resident chunk data without copying to/from
//! device buffers. This provides:
//! - **Zero intermediate copies**: Direct access to Arc slice data
//! - **Simple scalar operations**: No device allocation overhead
//! - **Cache-friendly**: Data stays in CPU cache
//!
//! For device-accelerated compute, use hologram-core operations directly
//! instead of StreamOps.
//!
//! ## Example
//!
//! ```
//! use hologram_memory_manager::compute::{StreamOp, StreamOpContext};
//! use hologram_memory_manager::{ChunkWithGauge, Result};
//! use std::sync::Arc;
//!
//! struct ReLUOp;
//!
//! impl StreamOp for ReLUOp {
//!     fn process(&self, _ctx: &mut StreamOpContext, chunk: &ChunkWithGauge) -> Result<ChunkWithGauge> {
//!         // Access chunk data directly (zero-copy)
//!         let data = chunk.data().expect("CPU-resident chunk");
//!         let input: &[f32] = bytemuck::cast_slice(data);
//!
//!         // Apply scalar operation
//!         let output: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();
//!
//!         // Create result chunk
//!         let bytes: Vec<u8> = bytemuck::cast_slice(&output).to_vec();
//!         let source: Arc<[u8]> = bytes.into();
//!         Ok(ChunkWithGauge::from_arc(source, chunk.gauge, chunk.primorial, chunk.index))
//!     }
//! }
//! ```

use crate::chunking::ChunkWithGauge;
use crate::Result;
use hologram_core::Executor;

/// Context for stream operations
///
/// Provides access to the hologram-core executor for SIMD operations.
pub struct StreamOpContext {
    executor: Executor,
}

impl StreamOpContext {
    /// Create new operation context
    pub fn new() -> Result<Self> {
        let executor = Executor::new().map_err(|e| crate::ProcessorError::ExecutionError(e.to_string()))?;
        Ok(Self { executor })
    }

    /// Get executor reference
    pub fn executor(&mut self) -> &mut Executor {
        &mut self.executor
    }
}

impl Default for StreamOpContext {
    fn default() -> Self {
        Self::new().expect("Failed to create default StreamOpContext")
    }
}

/// Stream operation trait
///
/// Implementors define scalar transformations on CPU-resident chunks.
///
/// ## Design Principles
///
/// - **Stateless**: No internal state, pure transformations
/// - **Composable**: Can be chained in pipelines
/// - **Zero-Copy**: Direct access to chunk data (no device copies)
/// - **Type-safe**: Generic over data types via bytemuck::Pod
///
/// ## Performance Note
///
/// These operations use scalar processing on CPU data. For device-accelerated
/// compute (SIMD, GPU), use hologram-core operations directly with device-resident
/// buffers instead of StreamOps.
pub trait StreamOp: Send + Sync {
    /// Process a chunk with scalar operations
    ///
    /// # Arguments
    ///
    /// - `ctx`: Execution context (may be unused for scalar ops)
    /// - `chunk`: Input chunk to process (must be CPU-resident)
    ///
    /// # Returns
    ///
    /// Processed chunk with same metadata (gauge, primorial, index)
    fn process(&self, ctx: &mut StreamOpContext, chunk: &ChunkWithGauge) -> Result<ChunkWithGauge>;

    /// Optional operation name for debugging
    fn name(&self) -> &str {
        "UnnamedOp"
    }
}

/// ReLU activation operation (max(0, x))
///
/// Applies scalar ReLU to f32 chunks.
#[derive(Debug, Clone)]
pub struct ReLUOp;

impl StreamOp for ReLUOp {
    fn process(&self, _ctx: &mut StreamOpContext, chunk: &ChunkWithGauge) -> Result<ChunkWithGauge> {
        // Process directly on chunk data (zero-copy for CPU-resident)
        let data = chunk.data().ok_or_else(|| {
            crate::ProcessorError::InvalidOperation("ReLU operation requires CPU-resident chunk".to_string())
        })?;

        // Cast to f32 slice
        let input: &[f32] = bytemuck::cast_slice(data);

        // Apply ReLU: max(0, x) - scalar operation
        let output: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();

        // Convert back to bytes
        let output_bytes: Vec<u8> = bytemuck::cast_slice(&output).to_vec();

        // Create new chunk with processed data
        use std::sync::Arc;
        let source: Arc<[u8]> = output_bytes.into();
        Ok(ChunkWithGauge::from_arc(
            source,
            chunk.gauge,
            chunk.primorial,
            chunk.index,
        ))
    }

    fn name(&self) -> &str {
        "ReLU"
    }
}

/// Scalar multiplication operation (x * scale)
///
/// Applies scalar multiplication to f32 chunks.
#[derive(Debug, Clone)]
pub struct ScalarMulOp {
    /// Scalar multiplier
    pub scale: f32,
}

impl StreamOp for ScalarMulOp {
    fn process(&self, _ctx: &mut StreamOpContext, chunk: &ChunkWithGauge) -> Result<ChunkWithGauge> {
        // Process directly on chunk data (zero-copy for CPU-resident)
        let data = chunk.data().ok_or_else(|| {
            crate::ProcessorError::InvalidOperation("ScalarMul operation requires CPU-resident chunk".to_string())
        })?;

        // Cast to f32 slice
        let input: &[f32] = bytemuck::cast_slice(data);

        // Apply scalar multiplication: x * scale
        let output: Vec<f32> = input.iter().map(|&x| x * self.scale).collect();

        // Convert back to bytes
        let output_bytes: Vec<u8> = bytemuck::cast_slice(&output).to_vec();

        // Create new chunk with processed data
        use std::sync::Arc;
        let source: Arc<[u8]> = output_bytes.into();
        Ok(ChunkWithGauge::from_arc(
            source,
            chunk.gauge,
            chunk.primorial,
            chunk.index,
        ))
    }

    fn name(&self) -> &str {
        "ScalarMul"
    }
}

/// Scalar addition operation (x + offset)
///
/// Applies scalar addition to f32 chunks.
#[derive(Debug, Clone)]
pub struct ScalarAddOp {
    /// Scalar offset
    pub offset: f32,
}

impl StreamOp for ScalarAddOp {
    fn process(&self, _ctx: &mut StreamOpContext, chunk: &ChunkWithGauge) -> Result<ChunkWithGauge> {
        // Process directly on chunk data (zero-copy for CPU-resident)
        let data = chunk.data().ok_or_else(|| {
            crate::ProcessorError::InvalidOperation("ScalarAdd operation requires CPU-resident chunk".to_string())
        })?;

        // Cast to f32 slice
        let input: &[f32] = bytemuck::cast_slice(data);

        // Apply scalar addition: x + offset
        let output: Vec<f32> = input.iter().map(|&x| x + self.offset).collect();

        // Convert back to bytes
        let output_bytes: Vec<u8> = bytemuck::cast_slice(&output).to_vec();

        // Create new chunk with processed data
        use std::sync::Arc;
        let source: Arc<[u8]> = output_bytes.into();
        Ok(ChunkWithGauge::from_arc(
            source,
            chunk.gauge,
            chunk.primorial,
            chunk.index,
        ))
    }

    fn name(&self) -> &str {
        "ScalarAdd"
    }
}

/// Clip operation (clamp values to [min, max])
///
/// Applies scalar clipping to f32 chunks.
#[derive(Debug, Clone)]
pub struct ClipOp {
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
}

impl StreamOp for ClipOp {
    fn process(&self, _ctx: &mut StreamOpContext, chunk: &ChunkWithGauge) -> Result<ChunkWithGauge> {
        // Process directly on chunk data (zero-copy for CPU-resident)
        let data = chunk.data().ok_or_else(|| {
            crate::ProcessorError::InvalidOperation("Clip operation requires CPU-resident chunk".to_string())
        })?;

        // Cast to f32 slice
        let input: &[f32] = bytemuck::cast_slice(data);

        // Apply clip: clamp(x, min, max)
        let output: Vec<f32> = input.iter().map(|&x| x.clamp(self.min, self.max)).collect();

        // Convert back to bytes
        let output_bytes: Vec<u8> = bytemuck::cast_slice(&output).to_vec();

        // Create new chunk with processed data
        use std::sync::Arc;
        let source: Arc<[u8]> = output_bytes.into();
        Ok(ChunkWithGauge::from_arc(
            source,
            chunk.gauge,
            chunk.primorial,
            chunk.index,
        ))
    }

    fn name(&self) -> &str {
        "Clip"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::PeriodDrivenChunker;

    #[test]
    fn test_relu_op() {
        let mut ctx = StreamOpContext::new().unwrap();

        // Create test data with negative values
        let float_data: Vec<f32> = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
        let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

        let chunker = PeriodDrivenChunker::new_fast(5);
        let chunks = chunker.chunk_fast(byte_data).unwrap();
        let chunk = &chunks[0];

        // Apply ReLU
        let relu = ReLUOp;
        let result = relu.process(&mut ctx, chunk).unwrap();

        // Verify ReLU applied correctly
        let result_floats: &[f32] = bytemuck::cast_slice(result.data().expect("CPU-resident chunk"));
        assert_eq!(result_floats[0], 0.0); // -1.0 -> 0.0
        assert_eq!(result_floats[1], 2.0); // 2.0 -> 2.0
        assert_eq!(result_floats[2], 0.0); // -3.0 -> 0.0
        assert_eq!(result_floats[3], 4.0); // 4.0 -> 4.0
    }

    #[test]
    fn test_scalar_mul_op() {
        let mut ctx = StreamOpContext::new().unwrap();

        let float_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

        let chunker = PeriodDrivenChunker::new_fast(5);
        let chunks = chunker.chunk_fast(byte_data).unwrap();
        let chunk = &chunks[0];

        // Apply 2.0 * x
        let mul = ScalarMulOp { scale: 2.0 };
        let result = mul.process(&mut ctx, chunk).unwrap();

        // Verify multiplication
        let result_floats: &[f32] = bytemuck::cast_slice(result.data().expect("CPU-resident chunk"));
        assert_eq!(result_floats[0], 2.0);
        assert_eq!(result_floats[1], 4.0);
        assert_eq!(result_floats[2], 6.0);
        assert_eq!(result_floats[3], 8.0);
    }

    #[test]
    fn test_scalar_add_op() {
        let mut ctx = StreamOpContext::new().unwrap();

        let float_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

        let chunker = PeriodDrivenChunker::new_fast(5);
        let chunks = chunker.chunk_fast(byte_data).unwrap();
        let chunk = &chunks[0];

        // Apply x + 10.0
        let add = ScalarAddOp { offset: 10.0 };
        let result = add.process(&mut ctx, chunk).unwrap();

        // Verify addition
        let result_floats: &[f32] = bytemuck::cast_slice(result.data().expect("CPU-resident chunk"));
        assert_eq!(result_floats[0], 11.0);
        assert_eq!(result_floats[1], 12.0);
        assert_eq!(result_floats[2], 13.0);
        assert_eq!(result_floats[3], 14.0);
    }

    #[test]
    fn test_clip_op() {
        let mut ctx = StreamOpContext::new().unwrap();

        let float_data: Vec<f32> = vec![-5.0, 0.0, 5.0, 10.0, 15.0];
        let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

        let chunker = PeriodDrivenChunker::new_fast(5);
        let chunks = chunker.chunk_fast(byte_data).unwrap();
        let chunk = &chunks[0];

        // Apply clip to [0.0, 10.0]
        let clip = ClipOp { min: 0.0, max: 10.0 };
        let result = clip.process(&mut ctx, chunk).unwrap();

        // Verify clipping
        let result_floats: &[f32] = bytemuck::cast_slice(result.data().expect("CPU-resident chunk"));
        assert_eq!(result_floats[0], 0.0); // -5.0 clipped to 0.0
        assert_eq!(result_floats[1], 0.0); // 0.0 stays 0.0
        assert_eq!(result_floats[2], 5.0); // 5.0 stays 5.0
        assert_eq!(result_floats[3], 10.0); // 10.0 stays 10.0
        assert_eq!(result_floats[4], 10.0); // 15.0 clipped to 10.0
    }
}
