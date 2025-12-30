//! Universal Memory Pool Module
//!
//! This module provides content-addressed memory pools that store embedded blocks
//! with their automatically constructed gauges. The pool is domain-neutral - it
//! doesn't interpret data, just stores it uniformly. Interpretation happens at
//! the domain head layer.
//!
//! # Core Concept
//!
//! ```text
//! Input → Primordial Chunking → Gauge Construction → Embedded Blocks → Memory Pool
//!                                                              ↓
//!                                      Domain heads extract modalities
//! ```

pub mod block;
pub mod pool;
pub mod storage;

pub use block::EmbeddedBlock;
pub use pool::{EmbeddingResult, UniversalMemoryPool};
pub use storage::MemoryStorage;
