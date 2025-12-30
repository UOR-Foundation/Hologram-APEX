//! Compute Pipeline
//!
//! Provides a high-level API for chaining SIMD operations on streaming data.
//!
//! ## Architecture
//!
//! ```text
//! Input Stream
//!     ↓
//! Chunk (primorial-driven)
//!     ↓
//! Operation 1 (SIMD)
//!     ↓
//! Operation 2 (SIMD)
//!     ↓
//! Operation N (SIMD)
//!     ↓
//! Embed to Memory Pool
//! ```
//!
//! ## Example
//!
//! ```
//! use hologram_memory_manager::{ComputePipeline, ScalarMulOp, ClipOp};
//!
//! // Create f32 data (converting from u8 would happen in real usage)
//! let float_data: Vec<f32> = (0..1024).map(|i| (i % 256) as f32).collect();
//! let data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();
//!
//! // Create pipeline: normalize [0, 255] -> [0, 1], then clip
//! let pipeline = ComputePipeline::new()
//!     .add_op(ScalarMulOp { scale: 1.0 / 255.0 })  // Scale to [0, 1]
//!     .add_op(ClipOp { min: 0.0, max: 1.0 });      // Ensure bounds
//!
//! let op_count = pipeline.operation_count();
//!
//! // Process data (consumes pipeline)
//! let context = pipeline.execute(data, 10).unwrap();
//!
//! println!("Processed {} bytes through {} operations",
//!     context.total_bytes,
//!     op_count
//! );
//! ```

use crate::chunking::{ChunkWithGauge, PeriodDrivenChunker};
use crate::compute::{StreamOp, StreamOpContext};
use crate::memory::{EmbeddedBlock, UniversalMemoryPool};
use crate::stream::StreamContext;
use crate::Result;
use rayon::prelude::*;

/// Compute pipeline with chainable SIMD operations
///
/// Allows building complex data transformations by composing stream operations.
pub struct ComputePipeline {
    /// Operations to apply in sequence
    operations: Vec<Box<dyn StreamOp>>,
}

impl ComputePipeline {
    /// Create new empty pipeline
    ///
    /// # Example
    ///
    /// ```
    /// use hologram_memory_manager::ComputePipeline;
    ///
    /// let pipeline = ComputePipeline::new();
    /// assert_eq!(pipeline.operation_count(), 0);
    /// ```
    pub fn new() -> Self {
        Self { operations: Vec::new() }
    }

    /// Add operation to pipeline
    ///
    /// Operations are applied in the order they are added.
    ///
    /// # Example
    ///
    /// ```
    /// use hologram_memory_manager::{ComputePipeline, ReLUOp, ScalarMulOp};
    ///
    /// let pipeline = ComputePipeline::new()
    ///     .add_op(ReLUOp)
    ///     .add_op(ScalarMulOp { scale: 2.0 });
    ///
    /// assert_eq!(pipeline.operation_count(), 2);
    /// ```
    pub fn add_op<O: StreamOp + 'static>(mut self, op: O) -> Self {
        self.operations.push(Box::new(op));
        self
    }

    /// Get number of operations in pipeline
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    /// Check if pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Execute pipeline on input data
    ///
    /// Chunks data using primorial sequence, applies all operations in parallel,
    /// then embeds into memory pool.
    ///
    /// # Arguments
    ///
    /// - `data`: Input data to process
    /// - `levels`: Maximum chunking levels (primorial depth)
    ///
    /// # Returns
    ///
    /// `StreamContext` with processed data in memory pool
    ///
    /// # Example
    ///
    /// ```
    /// use hologram_memory_manager::{ComputePipeline, ScalarMulOp};
    ///
    /// let pipeline = ComputePipeline::new()
    ///     .add_op(ScalarMulOp { scale: 0.5 });
    ///
    /// let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
    /// let context = pipeline.execute(data, 7).unwrap();
    ///
    /// assert_eq!(context.total_bytes, 100);
    /// assert!(context.pool.blocks().len() > 0);
    /// ```
    pub fn execute(self, data: Vec<u8>, levels: usize) -> Result<StreamContext> {
        // Chunk data using primorial sequence
        let chunker = PeriodDrivenChunker::new_fast(levels);
        let chunks = chunker.chunk_fast(data)?;

        // Process chunks in parallel
        let processed: Vec<ChunkWithGauge> = if self.operations.is_empty() {
            // No operations: pass through chunks unchanged
            chunks
        } else {
            // Apply operations to each chunk in parallel
            // Skip chunks that don't meet alignment requirements (e.g., not f32-aligned)
            chunks
                .into_par_iter()
                .filter_map(|chunk| {
                    // Skip chunks that aren't f32-aligned (most operations work with f32)
                    if chunk.len() % 4 != 0 {
                        // Return original chunk unchanged
                        return Some(Ok(chunk));
                    }

                    // Each thread gets its own context (executor)
                    let mut ctx = match StreamOpContext::new() {
                        Ok(ctx) => ctx,
                        Err(e) => return Some(Err(e)),
                    };

                    // Apply operations sequentially within this chunk
                    let mut current = chunk;
                    for op in &self.operations {
                        current = match op.process(&mut ctx, &current) {
                            Ok(c) => c,
                            Err(e) => return Some(Err(e)),
                        };
                    }

                    Some(Ok(current))
                })
                .collect::<Result<Vec<_>>>()?
        };

        // Embed processed chunks into memory pool
        let mut pool = UniversalMemoryPool::new();
        let mut total_bytes = 0;

        for chunk in processed {
            total_bytes += chunk.len();

            let block = EmbeddedBlock::new(chunk.storage().clone(), chunk.gauge, chunk.primorial, chunk.index);

            pool.insert_block(block);
        }

        Ok(StreamContext {
            gauges_count: pool.gauges_constructed(),
            total_bytes,
            pool,
        })
    }

    /// Execute pipeline on pre-chunked data
    ///
    /// Use this when you already have chunked data and want to apply operations.
    ///
    /// # Example
    ///
    /// ```
    /// use hologram_memory_manager::{Stream, ComputePipeline, ReLUOp, DEFAULT_MAX_CHUNK_LEVELS};
    ///
    /// let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
    /// let stream = Stream::new(data);
    /// let chunked = stream.chunk(DEFAULT_MAX_CHUNK_LEVELS).unwrap();
    ///
    /// let pipeline = ComputePipeline::new().add_op(ReLUOp);
    /// let context = pipeline.execute_on_chunks(chunked.chunks().to_vec()).unwrap();
    /// ```
    pub fn execute_on_chunks(self, chunks: Vec<ChunkWithGauge>) -> Result<StreamContext> {
        // Process chunks in parallel
        let processed: Vec<ChunkWithGauge> = if self.operations.is_empty() {
            chunks
        } else {
            chunks
                .into_par_iter()
                .filter_map(|chunk| {
                    // Skip chunks that aren't f32-aligned
                    if chunk.len() % 4 != 0 {
                        return Some(Ok(chunk));
                    }

                    let mut ctx = match StreamOpContext::new() {
                        Ok(ctx) => ctx,
                        Err(e) => return Some(Err(e)),
                    };

                    let mut current = chunk;
                    for op in &self.operations {
                        current = match op.process(&mut ctx, &current) {
                            Ok(c) => c,
                            Err(e) => return Some(Err(e)),
                        };
                    }
                    Some(Ok(current))
                })
                .collect::<Result<Vec<_>>>()?
        };

        // Embed into memory pool
        let mut pool = UniversalMemoryPool::new();
        let mut total_bytes = 0;

        for chunk in processed {
            total_bytes += chunk.len();

            let block = EmbeddedBlock::new(chunk.storage().clone(), chunk.gauge, chunk.primorial, chunk.index);

            pool.insert_block(block);
        }

        Ok(StreamContext {
            gauges_count: pool.gauges_constructed(),
            total_bytes,
            pool,
        })
    }
}

impl Default for ComputePipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{ClipOp, ReLUOp, ScalarMulOp};

    #[test]
    fn test_pipeline_creation() {
        let pipeline = ComputePipeline::new();
        assert_eq!(pipeline.operation_count(), 0);
        assert!(pipeline.is_empty());
    }

    #[test]
    fn test_pipeline_add_ops() {
        let pipeline = ComputePipeline::new().add_op(ReLUOp).add_op(ScalarMulOp { scale: 2.0 });

        assert_eq!(pipeline.operation_count(), 2);
        assert!(!pipeline.is_empty());
    }

    #[test]
    fn test_pipeline_execute_no_ops() {
        let pipeline = ComputePipeline::new();

        let data: Vec<u8> = (0..100).collect();
        let context = pipeline.execute(data, 7).unwrap();

        assert_eq!(context.total_bytes, 100);
        assert!(!context.pool.blocks().is_empty());
    }

    #[test]
    fn test_pipeline_execute_with_ops() {
        // Create large test data to ensure some f32-aligned chunks
        // 1024 f32 values = 4096 bytes
        let float_data: Vec<f32> = (0..1024).map(|i| i as f32 - 512.0).collect();
        let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

        // Pipeline: ReLU (max with 0) then scale by 2.0
        let pipeline = ComputePipeline::new().add_op(ReLUOp).add_op(ScalarMulOp { scale: 2.0 });

        let context = pipeline.execute(byte_data.clone(), 10).unwrap();

        assert_eq!(context.total_bytes, byte_data.len());
        assert!(!context.pool.blocks().is_empty());
    }

    #[test]
    fn test_pipeline_normalization() {
        // Normalize [0, 255] byte data to [0, 1] float
        // Use larger data for better chunk alignment
        let byte_data: Vec<u8> = (0..=255).cycle().take(4096).collect();

        // Convert to f32 first (would need a conversion op in real usage)
        let float_data: Vec<f32> = byte_data.iter().map(|&b| b as f32).collect();
        let float_bytes: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

        let pipeline = ComputePipeline::new()
            .add_op(ScalarMulOp { scale: 1.0 / 255.0 })
            .add_op(ClipOp { min: 0.0, max: 1.0 });

        let context = pipeline.execute(float_bytes.clone(), 10).unwrap();

        assert_eq!(context.total_bytes, float_bytes.len());
    }

    #[test]
    fn test_pipeline_on_chunks() {
        use crate::Stream;
        use crate::DEFAULT_MAX_CHUNK_LEVELS;

        // Use larger data for better chunk alignment
        let float_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

        // Create chunked stream
        let stream = Stream::new(byte_data.clone());
        let chunked = stream.chunk(DEFAULT_MAX_CHUNK_LEVELS).unwrap();

        // Apply pipeline to chunks
        let pipeline = ComputePipeline::new().add_op(ScalarMulOp { scale: 0.5 });

        let context = pipeline.execute_on_chunks(chunked.chunks().to_vec()).unwrap();

        assert_eq!(context.total_bytes, byte_data.len());
    }
}
