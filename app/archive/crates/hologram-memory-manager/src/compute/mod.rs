//! Compute Module
//!
//! Provides SIMD-accelerated compute operations for stream processing.
//!
//! This module integrates hologram-core's SIMD-backed operations into the processor,
//! enabling compute-bound streaming instead of memory-bound data movement.
//!
//! ## Architecture
//!
//! ```text
//! Input Stream
//!     ↓
//! Chunking (primorial-driven)
//!     ↓
//! SIMD Operations (via hologram-core)
//!     ↓
//! Domain Heads (extract modalities)
//!     ↓
//! Output Modality
//! ```
//!
//! ## Performance
//!
//! - **SIMD Ladder**: AVX-512 → AVX2 → SSE4.1 → scalar fallback
//! - **Zero-Copy**: Buffer handles, no intermediate allocations
//! - **Compile-Time**: Precompiled programs, no runtime compilation
//! - **Parallel**: Rayon-based chunk parallelization
//!
//! ## Example
//!
//! ```
//! use hologram_memory_manager::{ComputePipeline, ScalarMulOp};
//!
//! // Create f32 data for SIMD operations
//! let float_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
//! let data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();
//!
//! // Apply SIMD operations during embedding
//! let pipeline = ComputePipeline::new()
//!     .add_op(ScalarMulOp { scale: 0.5 });
//!
//! let context = pipeline.execute(data, 10).unwrap();
//! assert_eq!(context.total_bytes, 1024 * 4); // 1024 f32 = 4096 bytes
//! ```

pub mod pipeline;
pub mod stream_ops;

pub use pipeline::ComputePipeline;
pub use stream_ops::{ClipOp, ReLUOp, ScalarAddOp, ScalarMulOp, StreamOp, StreamOpContext};
