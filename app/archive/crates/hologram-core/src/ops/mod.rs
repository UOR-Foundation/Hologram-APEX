//! High-performance operations using compiled inline kernels
//!
//! **ARCHITECTURE**: Operations use compiled SIMD kernels for maximum performance.
//! Inline kernels are generated at build-time from JSON schemas and compiled directly
//! into the binary with AVX512/AVX2/SSE4.1 SIMD support.
//!
//! ## Performance Path
//!
//! 1. **Inline SIMD kernels** (for n ≤ 262K elements, f32, CPU backend)
//!    - Zero FFI overhead
//!    - AVX512/AVX2/SSE4.1 acceleration
//!    - 42ns execution time
//!    - 1.9-7.3× faster than native loops
//!
//! 2. **ISA fallback** (for other cases)
//!    - Uses precompiled ISA programs
//!    - Backend-agnostic execution
//!
//! ## Modules
//!
//! - `math` - Mathematical operations (add, sub, mul, div, min, max, abs, neg, etc.)
//! - `activation` - Activation functions (relu, sigmoid, tanh, gelu, softmax)
//! - `reduce` - Reduction operations (sum, mean, min, max)
//! - `loss` - Loss functions (mse, cross_entropy, binary_cross_entropy)
//! - `linalg` - Linear algebra (gemm, matvec)
//! - `memory` - Memory operations (copy, fill)
//! - `parallel` - Parallel execution utilities
//! - `traits` - Operation trait interfaces

pub mod activation;
pub mod linalg;
pub mod loss;
pub mod math;
pub mod memory;
pub mod parallel;
pub mod reduce;
pub mod traits;
