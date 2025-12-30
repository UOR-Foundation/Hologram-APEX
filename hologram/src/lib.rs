//! # Hologram - High-Performance Compute Acceleration
//!
//! Hologram provides general compute acceleration via canonical form compilation.
//!
//! ## Overview
//!
//! Hologram compiles operations to their canonical geometric representation, enabling:
//! - **Canonical Compilation**: Operations reduced to optimal form
//! - **Lowest Latency**: Canonical forms enable fastest execution
//! - **Universal Compute**: General-purpose acceleration
//! - **Multiple Backends**: CPU (SIMD), CUDA, Metal, WebGPU
//!
//! ## Architecture
//!
//! - **Two-torus lattice**: 48 × 256 cells
//! - **Monster group**: 196,884-dimensional representation
//! - **96-class system**: Canonical geometric classes
//! - **MoonshineHRM**: Algebraic framework (⊕, ⊗, ⊙)
//! - **Pattern-based canonicalization**: Minimal operation count
//!
//! ## Quick Start
//!
//! ```rust
//! use hologram::{Executor, ops};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create executor
//!     let mut exec = Executor::new()?;
//!
//!     // Allocate buffers
//!     let mut a = exec.allocate::<f32>(1024)?;
//!     let mut b = exec.allocate::<f32>(1024)?;
//!     let mut c = exec.allocate::<f32>(1024)?;
//!
//!     // Initialize data
//!     a.copy_from_slice(&mut exec, &vec![1.0; 1024])?;
//!     b.copy_from_slice(&mut exec, &vec![2.0; 1024])?;
//!
//!     // Execute operation (compiles to canonical kernel)
//!     ops::math::vector_add(&mut exec, &a, &b, &mut c, 1024)?;
//!
//!     // Get results
//!     let result = c.to_vec(&exec)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Features
//!
//! - **`default`**: Threading support
//! - **`ffi`**: Multi-language FFI bindings (Python, Swift, Kotlin, TypeScript, C++)
//! - **`cuda`**: NVIDIA GPU acceleration
//! - **`metal`**: Apple GPU acceleration (macOS/iOS)
//! - **`webgpu`**: Browser-based GPU acceleration
//!
//! ## Documentation
//!
//! - [Architecture](https://docs.rs/hologram/latest/hologram/#architecture)
//! - [API Reference](https://docs.rs/hologram)
//! - [Examples](https://github.com/UOR-Foundation/hologramapp/tree/main/hologram/examples)

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

// Re-export core types and modules
pub use hologram_core::{
    // Address mapping
    fits_in_class,
    max_elements_per_class,
    offset_to_phi_coordinate,
    // Operations modules
    ops,

    phi_coordinate_to_offset,
    // Instrumentation
    AggregateStatistics,
    // HRM types
    Atlas,
    BackendType,
    // Primary types
    Buffer,
    CompilationMetrics,
    // DLPack interoperability
    DLDataType,
    DLDataTypeCode,
    DLDevice,
    DLDeviceType,
    DLManagedTensor,
    DLPackType,
    DLTensor,
    // Error handling
    Error,
    ExecutionMetrics,
    Executor,
    GeneratorMetrics,
    GriessVector,
    KernelLoader,
    OptimizationMetrics,
    Result,
    ScaledAtlas,
    Tensor,
    BYTES_PER_CLASS,
    BYTES_PER_PAGE,
    PAGES_PER_CLASS,
};

// Re-export compiler types
pub use hologram_compiler::{
    Canonicalizer,
    // Primary compiler API
    CircuitCompiler,
    // Class system
    ClassRange,
    ClassTarget,
    CompiledCircuit,

    // Generator types
    Generator,
    GeneratorCall,
    MergeVariant,
    MultiQubitConstraint,
    NQubitState,
    QuantumGate,
    // Quantum computing (768-cycle)
    QuantumState,
    SplitVariant,

    Transform,
};

// Re-export backends types
pub use hologram_backends::{
    // Circuit to ISA translation
    circuit_to_isa,
    Address,
    // Primary backend API
    Backend,
    // Backend utilities
    BufferHandle,
    Condition,
    CpuBackend,
    CudaBackend,
    ExecutionParams,
    Instruction,
    Label,
    LaunchConfig,
    MetalBackend,
    PoolHandle,
    Predicate,

    // ISA types
    Program,
    ProgramCache,

    Register,
    Type as IsaType,
    WasmBackend,
};

// Re-export configuration types
pub use hologram_common::config::{
    BackendConfig, CompilerConfig, ConfigBuilder, HologramConfig as Config, LoggingConfig,
    MergedCompilerConfig, PerformanceConfig,
};

// Module re-exports for advanced use
pub use hologram_backends as backends;
pub use hologram_common as common;
pub use hologram_compiler as compiler;

// Conditional FFI exports
#[cfg(feature = "ffi")]
pub use hologram_ffi as ffi;

/// Version of the Hologram crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
