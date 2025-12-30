//! # Atlas Runtime - Computational Memory Model Implementation
//!
//! This crate implements the Atlas Runtime specification (v1.0) - a portable execution
//! environment for the Atlas ISA that treats computation as **structured memory access**
//! rather than traditional parallel execution.
//!
//! ## Core Concept
//!
//! Atlas defines a **computational memory space** with inherent mathematical structure:
//! - **96 Resonance Classes** - The fundamental partitioning of the space
//! - **48×256 Boundary Lens (Φ)** - Spatial addressing with locality guarantees
//! - **1-Skeleton** - Legal neighbor traversal graph
//! - **Mirror Involution** - Paired class symmetry
//! - **Phase Counter** - Temporal ordering (mod 768)
//! - **Resonance Accumulator R\[96\]** - Per-class state tracking
//!
//! ## Architecture
//!
//! The processor doesn't "compute in parallel" - it **resolves addresses** into the
//! Atlas space. The mathematical structure (neighbors, mirrors, phase) dictates where
//! to read/write next. Speed comes from:
//! - Small, regular tiles (12 KiB per class → L1 resident)
//! - Predictable strides (256-byte pages → cache-line friendly)
//! - Tiny LUTs (neighbors, mirrors → fits in L1I)
//! - SIMD-friendly widths (48, 256 are power-of-2 aligned)
//! - Zero coordination overhead (pure address resolution)
//!
//! ## Memory Layout
//!
//! Physical layout is a contiguous slab with logical tiling:
//! ```text
//! Total Space: 96 classes × 48 pages × 256 bytes = 1,179,648 bytes (~1.125 MiB)
//!
//! Class 0:  [48 pages × 256 bytes] = 12,288 bytes
//! Class 1:  [48 pages × 256 bytes] = 12,288 bytes
//! ...
//! Class 95: [48 pages × 256 bytes] = 12,288 bytes
//! ```
//!
//! Address calculation (BOUND.MAP):
//! ```text
//! offset = class * 12_288 + page * 256 + byte
//! ```
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use atlas_runtime::{AtlasSpace, bound_map};
//!
//! // Create the computational memory space
//! let mut space = AtlasSpace::new();
//!
//! // Resolve an address in the Atlas structure
//! let class = 42u8;
//! let page = 10u8;
//! let byte = 128u8;
//! let offset = bound_map(class, page, byte);
//!
//! // Access via boundary lens (Φ)
//! let data = space.read_boundary(class, page, byte);
//! ```
//!
//! ## Host Runtime
//!
//! The [`runtime`] module provides a CPU-backed host API (devices, contexts, queues, bursts) that ties
//! kernel metadata to [`Validator`] enforcement while orchestrating phase-ordered execution and
//! resonance accounting. It is intended as a reference implementation for integration tests and
//! prototype tooling that targets Atlas semantics.

pub mod addressing;
pub mod buffer;
pub mod error;
pub mod kernel;
pub mod phase;
pub mod resonance;
pub mod runtime;
pub mod space;
pub mod topology;
pub mod validation;

pub use addressing::{
    bound_map, bound_unmap, PhiDesc, BOUND_MAP, CLASS_STRIDE, PAGES_PER_CLASS, PAGE_SIZE, TOTAL_SPACE,
};
pub use buffer::{BufferAllocator, BufferHandle, MemoryPool};
pub use error::{AtlasError, Result};
pub use kernel::{KernelParam, LaunchDesc};
pub use phase::{decompose_phase, phase_in_window, PhaseCounter, PHASE_MODULUS};
pub use resonance::{ResonanceAccumulator, ResonanceStats};
pub use runtime::{
    enumerate_devices, Burst, Context, ContextDesc, Device, Kernel, KernelInvocation, PhaseAdvance, Queue, QueueDesc,
    QueueKind,
};
pub use space::AtlasSpace;
pub use topology::{MirrorTable, NeighborTable};
pub use validation::Validator;

// Re-export core types for convenience
pub use atlas_core::AtlasRatio;
pub use atlas_isa::{
    BlockDim, BoundaryFootprint, ClassMask, GridDim, KernelMetadata, LaneIdx, LaunchConfig, PhaseWindow,
};
