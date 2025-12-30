//! # Atlas ISA - Target-Agnostic Instruction Set Architecture
//!
//! This crate implements the Atlas ISA specification, a constraint-first compute ISA
//! designed for portability across CPU, GPU, TPU, DSP, and FPGA targets.
//!
//! The Atlas ISA is **target-agnostic**—it defines execution semantics independent
//! of specific hardware APIs. Backends map Atlas constructs to their native
//! parallelism models as described in the crate specification (`SPEC.md`).
//!
//! ## Core Concepts (§2 of spec)
//!
//! - **Lane**: The minimal execution agent (the "thread" of Atlas ISA)
//! - **Block**: Group of lanes that can synchronize via BAR.SYNC
//! - **Grid**: 3D iteration space of blocks
//! - **Kernel**: Parametric entry point executing over grid × block
//!
//! ## Atlas Invariants (§8 of spec)
//!
//! - **C96 Class System**: 96 resonance classes with R96 classifier
//! - **Mirror Pairing**: Involutive symmetry across classes
//! - **Unity Neutrality**: Zero net resonance delta
//! - **Boundary Lens**: 48 × 256 torus addressing (Φ encoding)
//! - **Phase Window**: 768-cycle modular counter for temporal scheduling
//!
//! ## Memory Model (§5 of spec)
//!
//! Four address spaces:
//! - **Global**: Flat 64-bit, accessible by all lanes
//! - **Shared**: Per-block scratch memory
//! - **Const**: Read-only, initialized at module load
//! - **Local**: Per-lane spill space
//!
//! ## Example
//!
//! ```
//! use atlas_isa::{PhiCoordinate, r96_classify, GridDim, BlockDim, LaunchConfig};
//!
//! // Classify a byte to its resonance class
//! let byte_value = 42u8;
//! let class = r96_classify(byte_value);
//! assert!(class.as_u8() < 96);
//!
//! // Encode boundary coordinates (§2.2: Φ lens)
//! let coord = PhiCoordinate::new(10, 128).unwrap();
//! let encoded = coord.encode();
//! assert_eq!(PhiCoordinate::decode(encoded), coord);
//!
//! // Define launch configuration for a kernel
//! let config = LaunchConfig::new(
//!     GridDim::new(96, 1, 1),      // Grid: 96 blocks (one per class)
//!     BlockDim::new(256, 1, 1),     // Block: 256 lanes
//!     0                              // No shared memory
//! );
//! ```

pub mod canonicalization;
pub mod constants;
pub mod instructions;
pub mod memory;
pub mod metadata;
pub mod types;
pub mod uor;

pub use atlas_core::AtlasRatio;
pub use canonicalization::{canonicalize, class_index, equivalent, is_canonical};
pub use constants::*;
pub use instructions::{
    Address, Condition, Instruction, InstructionCategory, Label, MemoryScope, Predicate, Program, ProgramError,
    ProgramResult, Register, Type,
};
pub use metadata::*;
pub use types::*;
pub use uor::{phi_decode, phi_encode, r96_classify, truth_add, truth_zero, PhiCoordinate, ResonanceClass};

use num_rational::Ratio;

/// Result type for Atlas ISA operations
pub type Result<T> = std::result::Result<T, AtlasError>;

/// Errors that can occur in Atlas ISA operations
#[derive(Debug, thiserror::Error)]
pub enum AtlasError {
    #[error("Invalid class ID: {0} (must be in [0, 96))")]
    InvalidClassId(u32),

    #[error("Invalid page: {0} (must be in [0, 48))")]
    InvalidPage(u32),

    #[error("Invalid byte: {0} (must be in [0, 256))")]
    InvalidByte(u32),

    #[error("Invalid phase: {0} (must be in [0, 768))")]
    InvalidPhase(u32),

    #[error("Unity neutrality violated: delta_sum = {0}")]
    UnityNeutralityViolated(Ratio<i64>),

    #[error("Invalid ratio denominator: {0}")]
    InvalidDenominator(i64),

    #[error("Class mask conflict: kernel requires classes that are unavailable")]
    ClassMaskConflict,

    #[error("Phase window incompatible: phase {current} not in [{begin}, {end})")]
    PhaseWindowIncompatible { current: u32, begin: u32, end: u32 },

    #[error("Boundary out of range: ({page}, {byte})")]
    BoundaryOutOfRange { page: u32, byte: u32 },

    #[error("Parameter packing error: {0}")]
    ParameterPackingError(String),

    #[error("Invalid kernel metadata: {0}")]
    InvalidMetadata(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("String error: {0}")]
    String(String),
}

impl From<atlas_core::AtlasError> for AtlasError {
    fn from(err: atlas_core::AtlasError) -> Self {
        use atlas_core::AtlasError as CoreError;

        match err {
            CoreError::InvalidClassId(id) => AtlasError::InvalidClassId(id),
            CoreError::InvalidPage(page) => AtlasError::InvalidPage(page),
            CoreError::InvalidByte(byte) => AtlasError::InvalidByte(byte),
            CoreError::InvalidPhase(phase) => AtlasError::InvalidPhase(phase),
            CoreError::UnityNeutralityViolated(delta) => AtlasError::UnityNeutralityViolated(delta),
            CoreError::InvalidDenominator(denom) => AtlasError::InvalidDenominator(denom),
            CoreError::ClassMaskConflict => AtlasError::ClassMaskConflict,
            CoreError::PhaseWindowIncompatible { current, begin, end } => {
                AtlasError::PhaseWindowIncompatible { current, begin, end }
            }
            CoreError::BoundaryOutOfRange { page, byte } => AtlasError::BoundaryOutOfRange { page, byte },
        }
    }
}

impl From<String> for AtlasError {
    fn from(err: String) -> Self {
        AtlasError::String(err)
    }
}
