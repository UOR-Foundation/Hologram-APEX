//! # Processor - Stream Processing with Automatic Gauge Construction
//!
//! The processor crate provides production-grade stream processing where **chunking IS
//! the gauge generator**. It enables arbitrary-sized inputs to be processed with:
//!
//! - **Automatic gauge construction** via period-driven chunking
//! - **Universal memory pools** for content-addressed storage
//! - **Domain heads as mediatypes** for different interpretations
//! - **Integration with circuit** for canonical circuit compilation
//!
//! ## Core Concept
//!
//! ```text
//! Input → Period-Driven Chunking → Gauge Construction → Memory Pool
//!     Detect periodicities
//!            ↓
//!     Match to primorials [30, 210, 2310, ...]
//!            ↓
//!     Construct gauges [{2,3,5}, {2,3,5,7}, ...]
//!            ↓
//!     Embed blocks with gauges + temporal info
//!            ↓
//!     Domain heads extract modalities
//! ```
//!
//! ## Quick Start
//!
//! ```
//! use hologram_memory_manager::{Stream, DEFAULT_MAX_CHUNK_LEVELS};
//!
//! // Create stream from data
//! let data: Vec<u8> = (0..1000).map(|i| i as u8).collect();
//! let stream = Stream::new(data);
//!
//! // Chunk with automatic gauge construction (using default chunk levels)
//! let chunked = stream.chunk(DEFAULT_MAX_CHUNK_LEVELS).unwrap();
//! let context = chunked.embed().unwrap();
//!
//! println!("Embedded {} bytes into {} blocks",
//!     context.total_bytes,
//!     context.pool.blocks().len()
//! );
//! ```
//!
//! ## Architecture
//!
//! - **chunking** - Period-driven chunking with automatic gauge construction
//! - **memory** - Universal content-addressed memory pools
//! - **domain** - Domain heads for modality interpretation
//! - **stream** - Stream processing API
//! - **compiler** - Circuit-to-stream compilation
//! - **executor** - Stream execution engine
//!
//! ## Features
//!
//! - `tracing` - Enable tracing instrumentation (optional)

// Error handling
mod error;
pub use error::{ProcessorError, Result};

// Gauge system (vendored from quantum-768)
pub mod gauge;
pub use gauge::Gauge;

// Display implementations
mod display;

// Constants and defaults
pub mod constants;
pub use constants::DEFAULT_MAX_CHUNK_LEVELS;

// Core modules
pub mod chunking;
pub mod compute;
pub mod domain;
pub mod memory;

// Stream processing modules
pub mod compiler;
pub mod executor;
pub mod stream;

// Re-exports for convenience
pub use chunking::{ChunkWithGauge, PeriodDrivenChunker};
pub use compiler::{CircuitStreamCompiler, CompiledCircuitStream};
pub use compute::{ClipOp, ComputePipeline, ReLUOp, ScalarAddOp, ScalarMulOp, StreamOp, StreamOpContext};
pub use domain::{
    AggregateDomainHead, DomainHead, DomainHeadRegistry, FilterDomainHead, FilterType, Modality, NormalizeDomainHead,
    RawDomainHead,
};
pub use executor::StreamProcessor;
pub use memory::{EmbeddedBlock, MemoryStorage, UniversalMemoryPool};
pub use stream::{ChunkedStream, Stream, StreamContext};
