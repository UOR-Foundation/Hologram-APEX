# hologram-core Specification

**Status:** Approved
**Version:** 0.1.0
**Last Updated:** 2025-01-18

## Overview

`hologram-core` provides the mathematical foundation and runtime for Hologram compute acceleration. It implements the two-torus lattice, Monster group representation, MoonshineHRM algebraic framework, and high-level operations.

## Purpose

Core responsibilities:
- Mathematical primitives (atlas, torus, monster, algebra)
- Runtime execution (Executor, Buffer, Tensor)
- High-level operations (math, activation, reduce, loss, linalg)
- DLPack interoperability for zero-copy tensor exchange

## Architecture

```
hologram-core
├── Mathematical Foundation
│   ├── Atlas (96-class system)
│   ├── Torus (48×256 lattice)
│   ├── Monster (196,884-dim representation)
│   └── Algebra (⊕, ⊗, ⊙ generators)
├── Runtime
│   ├── Executor (backend-agnostic)
│   ├── Buffer<T> (type-safe memory)
│   ├── Tensor<T> (multi-dimensional arrays)
│   └── Address mapping (96-class)
├── Operations
│   ├── Math (element-wise)
│   ├── Activation (neural network)
│   ├── Reduce (aggregations)
│   ├── Loss (loss functions)
│   ├── LinAlg (linear algebra)
│   └── Memory (copy, fill)
└── Interop
    └── DLPack (zero-copy tensor exchange)
```

## Public API

### Core Types

#### Executor

```rust
/// Backend-agnostic operation execution engine
pub struct Executor {
    backend: Box<dyn Backend>,
    config: Config,
}

impl Executor {
    /// Create executor with specified backend
    pub fn new(backend_type: BackendType) -> Result<Self>;

    /// Create executor with custom configuration
    pub fn with_config(backend_type: BackendType, config: Config) -> Result<Self>;

    /// Allocate type-safe buffer
    pub fn allocate<T: bytemuck::Pod>(&mut self, size: usize) -> Result<Buffer<T>>;

    /// Execute ISA program
    pub fn execute_program(&mut self, program: &Program) -> Result<()>;

    /// Get backend type
    pub fn backend_type(&self) -> BackendType;
}
```

#### Buffer<T>

```rust
/// Type-safe memory buffer
pub struct Buffer<T> {
    handle: BufferHandle,
    size: usize,
    backend: Arc<Mutex<dyn Backend>>,
    _phantom: PhantomData<T>,
}

impl<T: bytemuck::Pod> Buffer<T> {
    /// Get buffer size in elements
    pub fn size(&self) -> usize;

    /// Copy data from slice to buffer
    pub fn copy_from_slice(&mut self, data: &[T]) -> Result<()>;

    /// Copy buffer data to vector
    pub fn copy_to_vec(&self) -> Result<Vec<T>>;

    /// Get element count
    pub fn len(&self) -> usize;

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool;
}
```

#### Tensor<T>

```rust
/// Multi-dimensional array with PyTorch-like API
pub struct Tensor<T> {
    buffer: Arc<Buffer<T>>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

impl<T: bytemuck::Pod> Tensor<T> {
    /// Create tensor from buffer and shape
    pub fn from_buffer(buffer: Buffer<T>, shape: Vec<usize>) -> Result<Self>;

    /// Get tensor shape
    pub fn shape(&self) -> &[usize];

    /// Get tensor strides
    pub fn strides(&self) -> &[isize];

    /// Get number of dimensions
    pub fn ndim(&self) -> usize;

    /// Get total number of elements
    pub fn numel(&self) -> usize;

    /// Check if tensor is contiguous
    pub fn is_contiguous(&self) -> bool;

    // Zero-copy operations (modify offset/shape/strides only)

    /// Select index along dimension (reduces dimensionality by 1)
    pub fn select(&self, dim: usize, index: usize) -> Result<Tensor<T>>;

    /// Narrow tensor along dimension
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Tensor<T>>;

    /// Transpose 2D tensor
    pub fn transpose(&self) -> Result<Tensor<T>>;

    /// Permute dimensions
    pub fn permute(&self, dims: &[usize]) -> Result<Tensor<T>>;

    /// Reshape tensor (must preserve element count)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor<T>>;

    /// Check if broadcastable with other tensor
    pub fn is_broadcast_compatible_with(&self, other: &Tensor<T>) -> bool;

    // Linear algebra operations

    /// Matrix multiplication (2D tensors)
    pub fn matmul(&self, exec: &mut Executor, other: &Tensor<T>) -> Result<Tensor<T>>;

    /// Batch matrix multiplication
    pub fn bmm(&self, exec: &mut Executor, other: &Tensor<T>) -> Result<Tensor<T>>;

    // Data access

    /// Copy tensor data to vector
    pub fn to_vec(&self) -> Result<Vec<T>>;

    /// Create contiguous copy
    pub fn contiguous(&self) -> Result<Tensor<T>>;
}
```

### Operations API

#### Math Operations

```rust
pub mod ops {
    pub mod math {
        use super::*;

        /// Element-wise addition: c = a + b
        pub fn vector_add<T: bytemuck::Pod>(
            exec: &mut Executor,
            a: &Buffer<T>,
            b: &Buffer<T>,
            c: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Element-wise subtraction: c = a - b
        pub fn vector_sub<T: bytemuck::Pod>(
            exec: &mut Executor,
            a: &Buffer<T>,
            b: &Buffer<T>,
            c: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Element-wise multiplication: c = a * b
        pub fn vector_mul<T: bytemuck::Pod>(
            exec: &mut Executor,
            a: &Buffer<T>,
            b: &Buffer<T>,
            c: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Element-wise division: c = a / b
        pub fn vector_div<T: bytemuck::Pod>(
            exec: &mut Executor,
            a: &Buffer<T>,
            b: &Buffer<T>,
            c: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Element-wise minimum: c = min(a, b)
        pub fn min<T: bytemuck::Pod>(
            exec: &mut Executor,
            a: &Buffer<T>,
            b: &Buffer<T>,
            c: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Element-wise maximum: c = max(a, b)
        pub fn max<T: bytemuck::Pod>(
            exec: &mut Executor,
            a: &Buffer<T>,
            b: &Buffer<T>,
            c: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Absolute value: output = abs(input)
        pub fn abs<T: bytemuck::Pod>(
            exec: &mut Executor,
            input: &Buffer<T>,
            output: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Negation: output = -input
        pub fn neg<T: bytemuck::Pod>(
            exec: &mut Executor,
            input: &Buffer<T>,
            output: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// ReLU: output = max(0, input)
        pub fn relu<T: bytemuck::Pod>(
            exec: &mut Executor,
            input: &Buffer<T>,
            output: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;
    }

    pub mod activation {
        use super::*;

        /// Sigmoid: σ(x) = 1 / (1 + e^(-x))
        pub fn sigmoid<T: bytemuck::Pod>(
            exec: &mut Executor,
            input: &Buffer<T>,
            output: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Hyperbolic tangent: tanh(x)
        pub fn tanh<T: bytemuck::Pod>(
            exec: &mut Executor,
            input: &Buffer<T>,
            output: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// GELU: x * Φ(x) where Φ is CDF of standard normal
        pub fn gelu<T: bytemuck::Pod>(
            exec: &mut Executor,
            input: &Buffer<T>,
            output: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Softmax: softmax(x_i) = exp(x_i) / sum(exp(x_j))
        pub fn softmax<T: bytemuck::Pod>(
            exec: &mut Executor,
            input: &Buffer<T>,
            output: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;
    }

    pub mod reduce {
        use super::*;

        /// Sum reduction
        /// NOTE: output buffer must have size >= 3 for temporaries
        pub fn sum<T: bytemuck::Pod>(
            exec: &mut Executor,
            input: &Buffer<T>,
            output: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Mean reduction
        /// NOTE: output buffer must have size >= 3 for temporaries
        pub fn mean<T: bytemuck::Pod>(
            exec: &mut Executor,
            input: &Buffer<T>,
            output: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Min reduction
        /// NOTE: output buffer must have size >= 3 for temporaries
        pub fn min<T: bytemuck::Pod>(
            exec: &mut Executor,
            input: &Buffer<T>,
            output: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Max reduction
        /// NOTE: output buffer must have size >= 3 for temporaries
        pub fn max<T: bytemuck::Pod>(
            exec: &mut Executor,
            input: &Buffer<T>,
            output: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;
    }

    pub mod loss {
        use super::*;

        /// Mean squared error: MSE = mean((pred - target)^2)
        /// NOTE: output buffer must have size >= 3 for temporaries
        pub fn mse<T: bytemuck::Pod>(
            exec: &mut Executor,
            pred: &Buffer<T>,
            target: &Buffer<T>,
            output: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Cross entropy loss
        /// NOTE: output buffer must have size >= 3 for temporaries
        pub fn cross_entropy<T: bytemuck::Pod>(
            exec: &mut Executor,
            pred: &Buffer<T>,
            target: &Buffer<T>,
            output: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Binary cross entropy
        /// NOTE: output buffer must have size >= 3 for temporaries
        pub fn binary_cross_entropy<T: bytemuck::Pod>(
            exec: &mut Executor,
            pred: &Buffer<T>,
            target: &Buffer<T>,
            output: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;
    }

    pub mod linalg {
        use super::*;

        /// General matrix multiplication: C = A * B
        pub fn gemm<T: bytemuck::Pod>(
            exec: &mut Executor,
            a: &Buffer<T>,
            b: &Buffer<T>,
            c: &mut Buffer<T>,
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<()>;

        /// Matrix-vector multiplication: y = A * x
        pub fn matvec<T: bytemuck::Pod>(
            exec: &mut Executor,
            a: &Buffer<T>,
            x: &Buffer<T>,
            y: &mut Buffer<T>,
            m: usize,
            n: usize,
        ) -> Result<()>;
    }

    pub mod memory {
        use super::*;

        /// Copy data: dst = src
        pub fn copy<T: bytemuck::Pod>(
            exec: &mut Executor,
            src: &Buffer<T>,
            dst: &mut Buffer<T>,
            n: usize,
        ) -> Result<()>;

        /// Fill buffer with value
        pub fn fill<T: bytemuck::Pod>(
            exec: &mut Executor,
            buffer: &mut Buffer<T>,
            value: T,
            n: usize,
        ) -> Result<()>;
    }
}
```

### Mathematical Foundation

#### Atlas Module

```rust
pub mod atlas {
    /// Number of resonance classes
    pub const CLASS_COUNT: usize = 96;

    /// Pages per class
    pub const PAGES_PER_CLASS: usize = 48;

    /// Bytes per page
    pub const BYTES_PER_PAGE: usize = 256;

    /// Total bytes per class
    pub const BYTES_PER_CLASS: usize = PAGES_PER_CLASS * BYTES_PER_PAGE;  // 12,288

    /// Maximum f32 elements per class
    pub const F32_PER_CLASS: usize = BYTES_PER_CLASS / 4;  // 3,072

    /// Resonance class index (0-95)
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ResonanceClass(u8);

    impl ResonanceClass {
        pub fn new(class: u8) -> Result<Self>;
        pub const unsafe fn new_unchecked(class: u8) -> Self;
        pub fn get(&self) -> u8;
    }

    /// Classify data into resonance class
    pub fn classify(data: &[u8]) -> ResonanceClass;

    /// Get class label
    pub fn label(class: ResonanceClass) -> &'static str;

    /// Get mirror class
    pub fn mirror(class: ResonanceClass) -> ResonanceClass;
}
```

#### Torus Module

```rust
pub mod torus {
    /// Two-torus coordinate (page, resonance)
    pub type TorusCoord = (u8, u8);

    /// Projection: π: ℤ → ℤ₄₈ × ℤ₉₆
    pub fn project(n: i64) -> TorusCoord;

    /// Lifting (right inverse): λ(π(n), n) ≈ n
    pub fn lift(coord: TorusCoord, original: i64) -> i64;

    /// Compute offset from coordinates
    pub fn coord_to_offset(page: usize, byte: usize) -> usize;

    /// Φ encoding: (page, byte) → u32
    pub fn phi_encode(page: u32, byte: u32) -> u32;

    /// Φ decoding: u32 → (page, byte)
    pub fn phi_decode(phi: u32) -> (u32, u32);
}
```

#### Monster Module

```rust
pub mod monster {
    /// Monster representation dimension
    pub const REPRESENTATION_DIM: usize = 196_884;

    /// Number of conjugacy classes
    pub const CONJUGACY_CLASSES: usize = 194;

    /// Monster element representation
    pub struct MonsterElement {
        // Internal representation
    }

    /// O(1) addition routing via Monster representation
    pub fn route_add(a: TorusCoord, b: TorusCoord) -> TorusCoord;

    /// O(1) multiplication routing via Monster representation
    pub fn route_mul(a: TorusCoord, b: TorusCoord) -> TorusCoord;

    /// Lift torus coordinate to Monster representation space
    pub fn lift_to_monster(coord: TorusCoord) -> MonsterElement;

    /// Project Monster element back to torus
    pub fn project_to_torus(elem: MonsterElement) -> TorusCoord;
}
```

#### Algebra Module

```rust
pub mod algebra {
    /// Algebraic addition: a ⊕ b
    pub fn add(a: u64, b: u64) -> u64;

    /// Algebraic multiplication: a ⊗ b
    pub fn mul(a: u64, b: u64) -> u64;

    /// Scalar multiplication: k ⊙ a
    pub fn scalar(k: u64, a: u64) -> u64;

    /// Coherence check: π(a + b) = π(a) ⊕ π(b)
    pub fn check_addition_coherence(a: i64, b: i64) -> bool;

    /// Coherence check: π(a * b) = π(a) ⊗ π(b)
    pub fn check_multiplication_coherence(a: i64, b: i64) -> bool;
}
```

### Interoperability

#### DLPack Module

```rust
pub mod interop {
    pub mod dlpack {
        /// DLPack tensor for zero-copy exchange
        pub struct DLPackTensor {
            // DLPack C struct representation
        }

        /// Convert Buffer to DLPack tensor
        pub fn buffer_to_dlpack<T: bytemuck::Pod>(buffer: &Buffer<T>) -> DLPackTensor;

        /// Create Buffer from DLPack tensor (zero-copy if possible)
        pub fn dlpack_to_buffer<T: bytemuck::Pod>(dlpack: &DLPackTensor) -> Result<Buffer<T>>;

        /// Convert Tensor to DLPack tensor
        pub fn tensor_to_dlpack<T: bytemuck::Pod>(tensor: &Tensor<T>) -> DLPackTensor;

        /// Create Tensor from DLPack tensor
        pub fn dlpack_to_tensor<T: bytemuck::Pod>(dlpack: &DLPackTensor) -> Result<Tensor<T>>;
    }
}
```

## Internal Structure

```
crates/core/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # Public API + re-exports
│   ├── atlas/
│   │   ├── mod.rs
│   │   ├── graph.rs            # 96-vertex Atlas graph
│   │   ├── class.rs            # Resonance classes
│   │   ├── invariants.rs       # Mathematical invariants
│   │   └── constants.rs        # Core constants
│   ├── torus/
│   │   ├── mod.rs
│   │   ├── lattice.rs          # 48×256 lattice structure
│   │   ├── projection.rs       # Projection π
│   │   └── lifting.rs          # Lifting λ
│   ├── monster/
│   │   ├── mod.rs
│   │   ├── representation.rs   # 196,884-dim representation
│   │   ├── conjugacy.rs        # 194 conjugacy classes
│   │   └── routing.rs          # O(1) routing algorithms
│   ├── algebra/
│   │   ├── mod.rs
│   │   ├── generators.rs       # ⊕, ⊗, ⊙ operations
│   │   ├── coherence.rs        # Coherence proofs
│   │   └── derived.rs          # MatMul, Conv, etc.
│   ├── runtime/
│   │   ├── mod.rs
│   │   ├── executor.rs         # Executor implementation
│   │   ├── buffer.rs           # Buffer<T> implementation
│   │   ├── tensor.rs           # Tensor<T> implementation
│   │   └── address.rs          # 96-class addressing
│   ├── ops/
│   │   ├── mod.rs
│   │   ├── math.rs             # Element-wise operations (< 1K lines)
│   │   ├── activation.rs       # Activations (< 1K lines)
│   │   ├── reduce.rs           # Reductions (< 1K lines)
│   │   ├── loss.rs             # Loss functions (< 1K lines)
│   │   ├── linalg.rs           # Linear algebra (< 1K lines)
│   │   └── memory.rs           # Memory operations (< 1K lines)
│   ├── interop/
│   │   ├── mod.rs
│   │   └── dlpack.rs           # DLPack tensor exchange
│   ├── error.rs                # Error types
│   └── prelude.rs              # Common imports
└── tests/
    ├── atlas_tests.rs          # Atlas module tests
    ├── torus_tests.rs          # Torus module tests
    ├── monster_tests.rs        # Monster module tests
    ├── algebra_tests.rs        # Algebra module tests
    ├── executor_tests.rs       # Executor tests
    ├── buffer_tests.rs         # Buffer tests
    ├── tensor_tests.rs         # Tensor tests
    └── ops_tests.rs            # Operations tests
```

## Dependencies

### External Dependencies

```toml
[dependencies]
# Type safety
bytemuck = "1.14"

# Error handling
thiserror = "1.0"

# Arbitrary-precision arithmetic
num-bigint = "0.4"

# Concurrent data structures
dashmap = "5.5"
parking_lot = "0.12"

# Internal dependencies
hologram-backends = { path = "../backends", version = "0.1.0" }
hologram-config = { path = "../config", version = "0.1.0" }

[dev-dependencies]
# Property-based testing
proptest = "1.4"

# Benchmarking
criterion = "0.5"
```

### Internal Dependencies

- **hologram-backends**: Backend trait and implementations
- **hologram-config**: Configuration management

## Testing Requirements

### Unit Tests

**Coverage target:** ≥80% line coverage, ≥95% for runtime/*

All public functions must have unit tests covering:
- ✅ Normal operation (happy path)
- ✅ Edge cases (empty buffers, zero-size, boundary conditions)
- ✅ Error conditions (invalid inputs, allocation failures)

### Property-Based Tests

Use `proptest` for mathematical invariants:

```rust
proptest! {
    #[test]
    fn test_torus_projection_lifting(n: i64) {
        let coord = torus::project(n);
        let lifted = torus::lift(coord, n);
        // Lifting should approximate original (within modular bounds)
        prop_assert!((lifted - n).abs() < 48 * 96);
    }

    #[test]
    fn test_addition_coherence(a: i64, b: i64) {
        // π(a + b) = π(a) ⊕ π(b)
        prop_assert!(algebra::check_addition_coherence(a, b));
    }

    #[test]
    fn test_buffer_roundtrip(data in prop::collection::vec(any::<f32>(), 0..1000)) {
        let mut exec = Executor::new(BackendType::Cpu)?;
        let mut buffer = exec.allocate::<f32>(data.len())?;
        buffer.copy_from_slice(&data)?;
        let result = buffer.copy_to_vec()?;
        prop_assert_eq!(data, result);
    }
}
```

### Integration Tests

```rust
// tests/end_to_end.rs
#[test]
fn test_vector_addition_pipeline() -> Result<()> {
    let mut exec = Executor::new(BackendType::Cpu)?;

    let mut a = exec.allocate::<f32>(1024)?;
    let mut b = exec.allocate::<f32>(1024)?;
    let mut c = exec.allocate::<f32>(1024)?;

    let data_a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..1024).map(|i| (i * 2) as f32).collect();

    a.copy_from_slice(&data_a)?;
    b.copy_from_slice(&data_b)?;

    ops::math::vector_add(&mut exec, &a, &b, &mut c, 1024)?;

    let result = c.copy_to_vec()?;
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 3.0);
    assert_eq!(result[100], 300.0);

    Ok(())
}
```

## Performance Requirements

### Latency Targets

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| Buffer allocation (1K elements) | < 100ns | CPU backend |
| Element-wise operation (1K f32) | < 1µs | CPU SIMD, single-threaded |
| Tensor view creation | < 50ns | Zero-copy operations |
| Matrix multiply (128×128 f32) | < 100µs | CPU SIMD |
| DLPack conversion | < 100ns | Zero-copy when possible |

### Throughput Targets

| Metric | Target | Backend |
|--------|--------|---------|
| CPU SIMD GFLOPS | > 10 | f32, single-threaded |
| Memory bandwidth utilization | > 80% | Of theoretical peak |

### Optimization Guidelines

1. **Prefer O(1) complexity**
   - Use array indexing instead of iteration
   - Hash maps for lookups instead of linear search
   - Cache computed values

2. **Zero-copy where possible**
   - Tensor views share buffer
   - DLPack conversion avoids copies
   - In-place operations preferred

3. **SIMD optimization**
   - CPU backend uses SIMD for f32 operations ≤262K elements
   - Direct SIMD dispatch bypasses ISA interpretation
   - ~42ns execution vs ~1000ns ISA interpretation (23× faster)

4. **Compile-time computation**
   - Use `const` for all constants
   - Generic const parameters for fixed-size structures
   - Type-level guarantees

## Examples

### Basic Operations

```rust
use hologram_core::{Executor, BackendType, ops};

fn main() -> Result<()> {
    // Create executor
    let mut exec = Executor::new(BackendType::Cpu)?;

    // Allocate buffers
    let mut a = exec.allocate::<f32>(1024)?;
    let mut b = exec.allocate::<f32>(1024)?;
    let mut c = exec.allocate::<f32>(1024)?;

    // Fill buffers
    let data_a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..1024).map(|i| (i * 2) as f32).collect();
    a.copy_from_slice(&data_a)?;
    b.copy_from_slice(&data_b)?;

    // Execute operation
    ops::math::vector_add(&mut exec, &a, &b, &mut c, 1024)?;

    // Get results
    let result = c.copy_to_vec()?;
    println!("result[0] = {}", result[0]);  // 0.0
    println!("result[1] = {}", result[1]);  // 3.0

    Ok(())
}
```

### Tensor Operations

```rust
use hologram_core::{Executor, BackendType, Tensor};

fn main() -> Result<()> {
    let mut exec = Executor::new(BackendType::Cpu)?;

    // Create tensors
    let buf_a = exec.allocate::<f32>(32 * 64)?;
    let buf_b = exec.allocate::<f32>(64 * 16)?;

    let tensor_a = Tensor::from_buffer(buf_a, vec![32, 64])?;
    let tensor_b = Tensor::from_buffer(buf_b, vec![64, 16])?;

    // Matrix multiplication
    let result = tensor_a.matmul(&mut exec, &tensor_b)?;
    assert_eq!(result.shape(), &[32, 16]);

    // Zero-copy view operations
    let row = tensor_a.select(0, 5)?;  // Select row 5
    assert_eq!(row.shape(), &[64]);

    let slice = tensor_a.narrow(1, 0, 32)?;  // First 32 columns
    assert_eq!(slice.shape(), &[32, 32]);

    let transposed = tensor_a.transpose()?;
    assert_eq!(transposed.shape(), &[64, 32]);

    Ok(())
}
```

### Reduction Operations

```rust
use hologram_core::{Executor, BackendType, ops};

fn main() -> Result<()> {
    let mut exec = Executor::new(BackendType::Cpu)?;

    let mut input = exec.allocate::<f32>(1000)?;
    let data: Vec<f32> = (1..=1000).map(|i| i as f32).collect();
    input.copy_from_slice(&data)?;

    // NOTE: Reduction output must have size >= 3 for temporaries
    let mut sum_out = exec.allocate::<f32>(3)?;
    ops::reduce::sum(&mut exec, &input, &mut sum_out, 1000)?;

    let result = sum_out.copy_to_vec()?;
    println!("sum = {}", result[0]);  // 500500.0

    Ok(())
}
```

## Migration from Current Codebase

### Port Mapping

| Current Location | New Location |
|------------------|--------------|
| `atlas-core/src/lib.rs` | `atlas/graph.rs` |
| `atlas-core/src/invariants.rs` | `atlas/invariants.rs` |
| `atlas-core/src/constants.rs` | `atlas/constants.rs` |
| `hrm-spec/src/torus/*` | `torus/*` |
| `hrm-spec/src/monster/*` | `monster/*` |
| `hrm-spec/src/algebra/*` | `algebra/*` |
| `hologram-core/src/executor.rs` | `runtime/executor.rs` |
| `hologram-core/src/buffer.rs` | `runtime/buffer.rs` |
| `hologram-core/src/tensor.rs` | `runtime/tensor.rs` |
| `hologram-core/src/ops/*` | `ops/*` |
| `hologram-core/src/interop/dlpack.rs` | `interop/dlpack.rs` |

### What to Delete

- Legacy compiler code (superseded by hologram-compiler)
- Deprecated test files
- MoonshineHRM compiled operations (move to separate crate if needed)
- Runtime ISA interpretation (replaced by precompiled operations)

### Simplifications During Port

1. **Remove obsolete APIs** - Delete deprecated functions
2. **Inline utilities** - Inline functions used only once
3. **Break down large files** - Ensure all files < 1K lines
4. **Consolidate duplicates** - Merge duplicate implementations
5. **Update error handling** - Use thiserror consistently

## Error Handling

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("Buffer allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Invalid buffer size: expected {expected}, got {actual}")]
    InvalidBufferSize { expected: usize, actual: usize },

    #[error("Tensor shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("Invalid dimension: {0}")]
    InvalidDimension(usize),

    #[error("Backend error: {0}")]
    BackendError(#[from] hologram_backends::BackendError),

    #[error("Configuration error: {0}")]
    ConfigError(#[from] hologram_config::ConfigError),
}

pub type Result<T> = std::result::Result<T, CoreError>;
```

## Automatic Differentiation (Autograd)

### Gradient Tracking

```rust
/// Automatic differentiation engine
pub struct Autograd {
    tape: GradientTape,
    retain_graph: bool,
}

impl Autograd {
    pub fn new() -> Self;

    /// Enable gradient tracking
    pub fn enable_grad(&mut self);

    /// Disable gradient tracking
    pub fn disable_grad(&mut self);

    /// Record operation on tape
    pub fn record(&mut self, op: Operation, inputs: &[TensorId], output: TensorId);

    /// Compute gradients via backpropagation
    pub fn backward(&mut self, loss: TensorId) -> Result<HashMap<TensorId, Tensor<f32>>>;

    /// Clear gradient tape
    pub fn clear(&mut self);
}

/// Gradient tape for recording operations
pub struct GradientTape {
    operations: Vec<RecordedOp>,
    gradients: HashMap<TensorId, Tensor<f32>>,
}

#[derive(Debug, Clone)]
pub struct RecordedOp {
    pub operation: Operation,
    pub inputs: Vec<TensorId>,
    pub output: TensorId,
    pub backward_fn: BackwardFn,
}

pub type BackwardFn = fn(&[Tensor<f32>], &Tensor<f32>) -> Vec<Tensor<f32>>;
```

### Tensor with Gradients

```rust
/// Tensor with gradient tracking
pub struct GradTensor<T> {
    pub data: Tensor<T>,
    pub grad: Option<Tensor<T>>,
    pub requires_grad: bool,
    pub grad_fn: Option<BackwardFn>,
}

impl<T: NumericType> GradTensor<T> {
    /// Create tensor requiring gradients
    pub fn with_grad(data: Tensor<T>) -> Self;

    /// Compute gradients
    pub fn backward(&mut self, grad: Option<Tensor<T>>) -> Result<()>;

    /// Zero gradients
    pub fn zero_grad(&mut self);

    /// Detach from computation graph
    pub fn detach(&self) -> Tensor<T>;
}
```

### Gradient Functions

```rust
pub mod grad_ops {
    /// Gradient for element-wise addition
    pub fn add_backward(grad_output: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>);

    /// Gradient for element-wise multiplication
    pub fn mul_backward(
        grad_output: &Tensor<f32>,
        lhs: &Tensor<f32>,
        rhs: &Tensor<f32>,
    ) -> (Tensor<f32>, Tensor<f32>);

    /// Gradient for matrix multiplication
    pub fn matmul_backward(
        grad_output: &Tensor<f32>,
        lhs: &Tensor<f32>,
        rhs: &Tensor<f32>,
    ) -> (Tensor<f32>, Tensor<f32>);

    /// Gradient for ReLU
    pub fn relu_backward(grad_output: &Tensor<f32>, input: &Tensor<f32>) -> Tensor<f32>;

    /// Gradient for sigmoid
    pub fn sigmoid_backward(grad_output: &Tensor<f32>, output: &Tensor<f32>) -> Tensor<f32>;

    /// Gradient for softmax
    pub fn softmax_backward(grad_output: &Tensor<f32>, output: &Tensor<f32>) -> Tensor<f32>;
}
```

## Graph-Based Computation

### Computation Graph

```rust
/// Computation graph for operations
pub struct ComputationGraph {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    toposort: Vec<NodeId>,
}

impl ComputationGraph {
    pub fn new() -> Self;

    /// Add tensor node
    pub fn add_tensor(&mut self, tensor: Tensor<f32>) -> NodeId;

    /// Add operation node
    pub fn add_operation(
        &mut self,
        op: Operation,
        inputs: &[NodeId],
    ) -> NodeId;

    /// Topological sort for execution order
    pub fn toposort(&mut self) -> Result<Vec<NodeId>>;

    /// Execute graph
    pub fn execute(&mut self, exec: &mut Executor) -> Result<HashMap<NodeId, Tensor<f32>>>;

    /// Optimize graph (fusion, elimination)
    pub fn optimize(&mut self) -> Result<()>;

    /// Visualize graph
    pub fn to_dot(&self) -> String;
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: NodeId,
    pub kind: NodeKind,
}

#[derive(Debug, Clone)]
pub enum NodeKind {
    Tensor(Tensor<f32>),
    Operation(Operation),
    Constant(f32),
}

#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from: NodeId,
    pub to: NodeId,
}
```

### Graph Optimization

```rust
/// Graph optimizer
pub struct GraphOptimizer {
    passes: Vec<OptimizationPass>,
}

impl GraphOptimizer {
    pub fn new() -> Self;

    /// Add optimization pass
    pub fn add_pass(&mut self, pass: OptimizationPass);

    /// Optimize computation graph
    pub fn optimize(&self, graph: &mut ComputationGraph) -> Result<()>;
}

pub trait OptimizationPass {
    fn name(&self) -> &str;
    fn run(&self, graph: &mut ComputationGraph) -> Result<bool>;
}

/// Common optimization passes
pub struct ConstantFolding;
pub struct DeadCodeElimination;
pub struct OperatorFusion;
pub struct CommonSubexpressionElimination;
```

## Lazy Evaluation and Fusion

### Lazy Tensor

```rust
/// Lazy-evaluated tensor
pub enum LazyTensor<T> {
    /// Materialized tensor
    Materialized(Tensor<T>),

    /// Lazy operation
    Lazy {
        operation: LazyOp,
        inputs: Vec<LazyTensor<T>>,
    },
}

impl<T: NumericType> LazyTensor<T> {
    /// Create lazy tensor
    pub fn lazy(op: LazyOp, inputs: Vec<LazyTensor<T>>) -> Self;

    /// Force evaluation
    pub fn eval(&self, exec: &mut Executor) -> Result<Tensor<T>>;

    /// Check if materialized
    pub fn is_materialized(&self) -> bool;

    /// Optimize lazy computation tree
    pub fn optimize(&mut self) -> Result<()>;
}

#[derive(Debug, Clone)]
pub enum LazyOp {
    Add,
    Mul,
    MatMul,
    Reduce(ReduceOp),
    Reshape(Vec<usize>),
    Transpose,
}
```

### Operation Fusion

```rust
/// Kernel fusion engine
pub struct FusionEngine {
    fusion_rules: Vec<FusionRule>,
}

impl FusionEngine {
    pub fn new() -> Self;

    /// Fuse operations in lazy tensor tree
    pub fn fuse(&self, tensor: &mut LazyTensor<f32>) -> Result<()>;

    /// Check if operations can be fused
    pub fn can_fuse(&self, ops: &[LazyOp]) -> bool;

    /// Create fused kernel
    pub fn create_fused_kernel(&self, ops: &[LazyOp]) -> Result<FusedKernel>;
}

#[derive(Debug, Clone)]
pub struct FusionRule {
    pub pattern: Vec<LazyOp>,
    pub replacement: FusedKernel,
}

#[derive(Debug, Clone)]
pub struct FusedKernel {
    pub name: String,
    pub program: Program,
}
```

## Distributed Execution

### Distributed Executor

```rust
/// Distributed executor across multiple nodes
pub struct DistributedExecutor {
    local_executor: Executor,
    cluster: ClusterManager,
    sharding_strategy: ShardingStrategy,
}

impl DistributedExecutor {
    /// Create distributed executor
    pub fn new(node_addresses: &[String]) -> Result<Self>;

    /// Execute tensor operation distributedly
    pub fn execute_distributed<T: NumericType>(
        &mut self,
        op: Operation,
        tensors: &[DistributedTensor<T>],
    ) -> Result<DistributedTensor<T>>;

    /// All-reduce operation
    pub fn all_reduce<T: NumericType>(
        &mut self,
        tensor: &DistributedTensor<T>,
        op: ReduceOp,
    ) -> Result<DistributedTensor<T>>;

    /// All-gather operation
    pub fn all_gather<T: NumericType>(
        &mut self,
        tensor: &DistributedTensor<T>,
    ) -> Result<DistributedTensor<T>>;

    /// Broadcast operation
    pub fn broadcast<T: NumericType>(
        &mut self,
        tensor: &DistributedTensor<T>,
        root: usize,
    ) -> Result<()>;
}

/// Distributed tensor across nodes
pub struct DistributedTensor<T> {
    pub local_shard: Tensor<T>,
    pub global_shape: Vec<usize>,
    pub shard_spec: ShardSpec,
    pub node_id: usize,
}

#[derive(Debug, Clone)]
pub struct ShardSpec {
    pub axis: usize,
    pub num_shards: usize,
    pub shard_id: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum ShardingStrategy {
    DataParallel,
    ModelParallel,
    PipelineParallel,
    Hybrid,
}
```

### Cluster Management

```rust
/// Cluster manager for distributed execution
pub struct ClusterManager {
    nodes: Vec<NodeInfo>,
    coordinator: NodeId,
}

impl ClusterManager {
    /// Initialize cluster
    pub fn init(node_addresses: &[String]) -> Result<Self>;

    /// Get node count
    pub fn node_count(&self) -> usize;

    /// Send tensor to node
    pub fn send_tensor<T: NumericType>(
        &mut self,
        tensor: &Tensor<T>,
        dest: NodeId,
    ) -> Result<()>;

    /// Receive tensor from node
    pub fn receive_tensor<T: NumericType>(
        &mut self,
        src: NodeId,
    ) -> Result<Tensor<T>>;

    /// Barrier synchronization
    pub fn barrier(&mut self) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub id: NodeId,
    pub address: String,
    pub capabilities: NodeCapabilities,
}

#[derive(Debug, Clone)]
pub struct NodeCapabilities {
    pub backend_type: BackendType,
    pub memory_gb: usize,
    pub compute_units: usize,
}
```

## GPU-Optimized Kernels

### GPU Kernel Compiler

```rust
/// GPU kernel compiler and optimizer
pub struct GpuKernelCompiler {
    backend: GpuBackend,
    optimizer: KernelOptimizer,
}

impl GpuKernelCompiler {
    pub fn new(backend: GpuBackend) -> Self;

    /// Compile operation to optimized GPU kernel
    pub fn compile_kernel(&self, op: Operation) -> Result<GpuKernel>;

    /// Compile with fusion
    pub fn compile_fused(&self, ops: &[Operation]) -> Result<GpuKernel>;

    /// Auto-tune kernel parameters
    pub fn auto_tune(&self, kernel: &mut GpuKernel) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct GpuKernel {
    pub name: String,
    pub source: String,
    pub launch_config: LaunchConfig,
    pub shared_memory: usize,
    pub registers_per_thread: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum GpuBackend {
    Cuda,
    Metal,
    WebGpu,
    Vulkan,
}
```

### Kernel Optimization

```rust
/// GPU kernel optimizer
pub struct KernelOptimizer {
    passes: Vec<KernelOptPass>,
}

impl KernelOptimizer {
    pub fn new() -> Self;

    /// Optimize kernel code
    pub fn optimize(&self, kernel: &mut GpuKernel) -> Result<()>;

    /// Optimize memory access patterns
    pub fn optimize_memory_access(&self, kernel: &mut GpuKernel) -> Result<()>;

    /// Optimize register usage
    pub fn optimize_registers(&self, kernel: &mut GpuKernel) -> Result<()>;

    /// Vectorize operations
    pub fn vectorize(&self, kernel: &mut GpuKernel) -> Result<()>;
}

pub trait KernelOptPass {
    fn name(&self) -> &str;
    fn run(&self, kernel: &mut GpuKernel) -> Result<()>;
}

/// Common kernel optimization passes
pub struct Coalescing;         // Memory coalescing
pub struct Tiling;              // Loop tiling
pub struct Unrolling;           // Loop unrolling
pub struct Vectorization;       // SIMD vectorization
pub struct SharedMemoryOpt;     // Shared memory optimization
```

## Dynamic Shapes

### Dynamic Tensor

```rust
/// Tensor with dynamic (runtime-determined) shapes
pub struct DynamicTensor<T> {
    buffer: Buffer<T>,
    shape: DynamicShape,
    strides: Vec<usize>,
}

impl<T: NumericType> DynamicTensor<T> {
    /// Create dynamic tensor
    pub fn new(exec: &mut Executor, initial_shape: Vec<usize>) -> Result<Self>;

    /// Reshape at runtime
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<()>;

    /// Get current shape
    pub fn shape(&self) -> &[usize];

    /// Maximum allocated capacity
    pub fn capacity(&self) -> usize;

    /// Resize (reallocate if needed)
    pub fn resize(&mut self, exec: &mut Executor, new_shape: Vec<usize>) -> Result<()>;
}

#[derive(Debug, Clone)]
pub enum DynamicShape {
    /// Fully known at compile time
    Static(Vec<usize>),

    /// Partially known (e.g., batch size unknown)
    Partial(Vec<Option<usize>>),

    /// Fully dynamic
    Dynamic,
}

impl DynamicShape {
    /// Check if shape is known
    pub fn is_known(&self) -> bool;

    /// Get known dimensions
    pub fn known_dims(&self) -> Vec<usize>;

    /// Validate against shape
    pub fn validate(&self, shape: &[usize]) -> Result<()>;
}
```

### Shape Inference

```rust
/// Shape inference engine
pub struct ShapeInference {
    rules: HashMap<Operation, ShapeInferenceRule>,
}

impl ShapeInference {
    pub fn new() -> Self;

    /// Infer output shape for operation
    pub fn infer_shape(
        &self,
        op: &Operation,
        input_shapes: &[DynamicShape],
    ) -> Result<DynamicShape>;

    /// Validate shapes for operation
    pub fn validate_shapes(
        &self,
        op: &Operation,
        shapes: &[DynamicShape],
    ) -> Result<()>;
}

pub type ShapeInferenceRule = fn(&[DynamicShape]) -> Result<DynamicShape>;

// Shape inference rules
pub mod shape_rules {
    pub fn add_shapes(shapes: &[DynamicShape]) -> Result<DynamicShape>;
    pub fn matmul_shapes(shapes: &[DynamicShape]) -> Result<DynamicShape>;
    pub fn reshape_shape(shapes: &[DynamicShape]) -> Result<DynamicShape>;
    pub fn broadcast_shapes(shapes: &[DynamicShape]) -> Result<DynamicShape>;
}
```

## Quantization Support

### Quantized Tensor

```rust
/// Quantized tensor (INT8/INT16)
pub struct QuantizedTensor {
    data: Buffer<i8>,
    scale: f32,
    zero_point: i8,
    shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Quantize from floating-point tensor
    pub fn quantize(tensor: &Tensor<f32>, dtype: QuantDType) -> Result<Self>;

    /// Dequantize to floating-point
    pub fn dequantize(&self) -> Result<Tensor<f32>>;

    /// Get quantization parameters
    pub fn qparams(&self) -> (f32, i8);

    /// Quantized matrix multiplication
    pub fn qmatmul(
        &self,
        exec: &mut Executor,
        other: &QuantizedTensor,
    ) -> Result<QuantizedTensor>;
}

#[derive(Debug, Clone, Copy)]
pub enum QuantDType {
    INT8,
    INT16,
    UINT8,
    UINT16,
}
```

### Quantization Schemes

```rust
/// Quantization calibration
pub struct QuantizationCalibrator {
    method: QuantMethod,
    calibration_data: Vec<Tensor<f32>>,
}

impl QuantizationCalibrator {
    pub fn new(method: QuantMethod) -> Self;

    /// Add calibration data
    pub fn add_sample(&mut self, tensor: Tensor<f32>);

    /// Compute quantization parameters
    pub fn calibrate(&self) -> Result<QuantParams>;
}

#[derive(Debug, Clone, Copy)]
pub enum QuantMethod {
    /// Symmetric quantization
    Symmetric,

    /// Asymmetric quantization
    Asymmetric,

    /// Per-channel quantization
    PerChannel,

    /// Dynamic quantization
    Dynamic,
}

#[derive(Debug, Clone)]
pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i8,
    pub dtype: QuantDType,
}
```

### Quantized Operations

```rust
pub mod quantized_ops {
    /// Quantized matrix multiplication
    pub fn qmatmul(
        exec: &mut Executor,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
        c: &mut QuantizedTensor,
    ) -> Result<()>;

    /// Quantized convolution
    pub fn qconv2d(
        exec: &mut Executor,
        input: &QuantizedTensor,
        kernel: &QuantizedTensor,
        output: &mut QuantizedTensor,
    ) -> Result<()>;

    /// Quantized element-wise operations
    pub fn qadd(
        exec: &mut Executor,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
        c: &mut QuantizedTensor,
    ) -> Result<()>;

    pub fn qrelu(
        exec: &mut Executor,
        input: &QuantizedTensor,
        output: &mut QuantizedTensor,
    ) -> Result<()>;
}
```

## Macro-Based Operation Definitions

### Operation Definition Macro

```rust
/// Define high-level operation with automatic ISA compilation
macro_rules! define_operation {
    (
        $(#[$meta:meta])*
        pub fn $name:ident<$T:ident: NumericType>(
            exec: &mut Executor,
            $($param:ident: $param_ty:ty),* $(,)?
        ) -> Result<()> {
            circuit: $circuit:expr,
            constraints: [$($constraint:expr),* $(,)?]
        }
    ) => {
        $(#[$meta])*
        pub fn $name<$T: NumericType>(
            exec: &mut Executor,
            $($param: $param_ty),*
        ) -> Result<()> {
            // Compile circuit to ISA
            let compiler = CircuitCompiler::new();
            let compiled = compiler.compile($circuit)?;

            // Validate constraints
            $(
                if !($constraint) {
                    return Err(CoreError::ConstraintViolation(
                        stringify!($constraint).to_string()
                    ));
                }
            )*

            // Execute compiled program
            exec.execute_program(&compiled.to_isa())?;

            Ok(())
        }
    };
}

// Usage example
define_operation! {
    /// Element-wise addition
    pub fn vector_add<T: NumericType>(
        exec: &mut Executor,
        a: &Buffer<T>,
        b: &Buffer<T>,
        c: &mut Buffer<T>,
        n: usize,
    ) -> Result<()> {
        circuit: "merge@c[0..N]",
        constraints: [
            a.len() == n,
            b.len() == n,
            c.len() == n,
        ]
    }
}

define_operation! {
    /// ReLU activation
    pub fn relu<T: NumericType>(
        exec: &mut Executor,
        input: &Buffer<T>,
        output: &mut Buffer<T>,
        n: usize,
    ) -> Result<()> {
        circuit: "max(x, 0)@c[0..N]",
        constraints: [
            input.len() == n,
            output.len() == n,
        ]
    }
}
```

### Test Helper Macros

```rust
/// Test operation with automatic setup/teardown
macro_rules! test_operation {
    (
        name: $name:ident,
        operation: $op:expr,
        input: $input:expr,
        expected: $expected:expr,
        tolerance: $tol:expr
    ) => {
        #[test]
        fn $name() -> Result<()> {
            let mut exec = Executor::new(BackendType::Cpu)?;

            let input_data: Vec<f32> = $input;
            let expected_data: Vec<f32> = $expected;

            let mut input_buf = exec.allocate::<f32>(input_data.len())?;
            let mut output_buf = exec.allocate::<f32>(expected_data.len())?;

            input_buf.copy_from_slice(&input_data)?;

            // Execute operation
            $op(&mut exec, &input_buf, &mut output_buf, input_data.len())?;

            // Verify results
            let output_data = output_buf.copy_to_vec()?;
            for (i, (actual, expected)) in output_data.iter().zip(expected_data.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() < $tol,
                    "Mismatch at index {}: expected {}, got {}",
                    i,
                    expected,
                    actual
                );
            }

            Ok(())
        }
    };
}

// Usage
test_operation! {
    name: test_vector_add,
    operation: ops::math::vector_add,
    input: vec![1.0, 2.0, 3.0, 4.0],
    expected: vec![2.0, 4.0, 6.0, 8.0],
    tolerance: 1e-6
}
```

## Generic Operation Patterns

### Generic Operation Trait

```rust
/// Generic operation interface
pub trait GenericOperation<T: NumericType> {
    /// Operation name
    fn name(&self) -> &str;

    /// Execute operation
    fn execute(
        &self,
        exec: &mut Executor,
        inputs: &[&Buffer<T>],
        outputs: &mut [&mut Buffer<T>],
    ) -> Result<()>;

    /// Get operation circuit
    fn circuit(&self) -> &str;

    /// Validate constraints
    fn validate(&self, inputs: &[&Buffer<T>], outputs: &[&Buffer<T>]) -> Result<()>;
}

/// Implement operation for all numeric types
macro_rules! impl_generic_operation {
    ($op_name:ident, $circuit:expr) => {
        pub struct $op_name;

        impl<T: NumericType> GenericOperation<T> for $op_name {
            fn name(&self) -> &str {
                stringify!($op_name)
            }

            fn execute(
                &self,
                exec: &mut Executor,
                inputs: &[&Buffer<T>],
                outputs: &mut [&mut Buffer<T>],
            ) -> Result<()> {
                self.validate(inputs, outputs)?;

                let compiler = CircuitCompiler::new();
                let compiled = compiler.compile($circuit)?;
                exec.execute_program(&compiled.to_isa())?;

                Ok(())
            }

            fn circuit(&self) -> &str {
                $circuit
            }

            fn validate(&self, inputs: &[&Buffer<T>], outputs: &[&Buffer<T>]) -> Result<()> {
                // Validate buffer sizes, etc.
                Ok(())
            }
        }
    };
}

// Usage
impl_generic_operation!(Add, "merge@c[0..N]");
impl_generic_operation!(Mul, "multiply@c[0..N]");
impl_generic_operation!(Relu, "max(x, 0)@c[0..N]");
```

## Future Enhancements

- [ ] Sparse tensor support
- [ ] Mixed-precision training helpers
- [ ] Checkpoint/restore for large models
- [ ] Memory-mapped tensor support
- [ ] Custom user-defined operations
- [ ] JIT compilation of tensor expressions

## References

- [Atlas Embeddings Theory](../../architecture/atlas-embeddings.md)
- [Monster Group Representation](../../architecture/monster-group.md)
- [MoonshineHRM Specification](../../architecture/moonshine-hrm.md)
- [DLPack Specification](https://github.com/dmlc/dlpack)
