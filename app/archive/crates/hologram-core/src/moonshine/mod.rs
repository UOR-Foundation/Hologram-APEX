//! MoonshineHRM Compiled Operations
//!
//! This module implements the MoonshineHRM compiled system, replacing ISA-based
//! runtime execution with pre-computed lookup tables.
//!
//! ## Architecture
//!
//! ```text
//! [COMPILE TIME]
//! Input Patterns → Atlas Embedding → Pre-computation → .mshr Binary
//!    (values)        (h₂, d, ℓ)      (96×96 tables)    (mmap-ready)
//!
//! [RUNTIME]
//! Input → Hash (10ns) → Lookup (5ns) → SIMD Load (20ns) → Output
//! Total: ~35ns per operation (O(1) guaranteed)
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use hologram_core::moonshine::CompiledOperation;
//!
//! // Load compiled operation
//! let add_op = CompiledOperation::load("ops/vector_add.mshr")?;
//!
//! // Execute with O(1) lookup
//! let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let result = add_op.execute_f32(&input)?;
//! ```
//!
//! ## Modules
//!
//! - `format` - .mshr binary format specification
//! - `operation` - CompiledOperation struct and execution
//! - `registry` - Operation loading and management
//! - `hash` - Input hashing for lookups

pub mod format;
pub mod hash;
pub mod operation;
pub mod registry;

// Re-exports
pub use format::{DataType, HashEntry, Manifest, MshrHeader};
pub use hash::hash_input_f32;
pub use operation::CompiledOperation;
pub use registry::OperationRegistry;
