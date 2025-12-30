# Getting Started with Hologram

This guide will help you get started with Hologram, a mathematically-driven tensor computation framework built on the Monster group and Griess algebra.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Basic Concepts](#basic-concepts)
- [Your First Program](#your-first-program)
- [Next Steps](#next-steps)

## Installation

### From crates.io

```bash
cargo add hologram
```

### From source

```bash
git clone https://github.com/your-org/hologram.git
cd hologram
cargo build --release
```

### Feature Flags

Hologram supports multiple backend implementations through feature flags:

```toml
[dependencies]
hologram = { version = "0.1", features = ["cuda", "metal"] }
```

Available features:
- `cuda` - NVIDIA GPU support via CUDA
- `metal` - Apple Silicon GPU support via Metal
- `wgpu` - WebGPU support for browsers and cross-platform GPU
- `ffi` - FFI bindings for Python, Kotlin, Swift, etc.

## Quick Start

```rust
use hologram::{Executor, Tensor, ops};

fn main() -> hologram::Result<()> {
    // Create a CPU executor
    let executor = Executor::new_auto()?;

    // Create tensors
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;

    // Perform operations
    let c = ops::add(&executor, &a, &b)?;

    // Get results
    let result = c.to_vec()?;
    println!("Result: {:?}", result);

    Ok(())
}
```

## Basic Concepts

### Executor

The `Executor` is the runtime that manages computation. It handles:
- Backend selection (CPU, CUDA, Metal, WebGPU)
- Memory management
- Kernel execution

Create an executor:

```rust
use hologram::Executor;

// Auto-detect best backend
let executor = Executor::new_auto()?;

// Or specify a backend
let cpu_executor = Executor::new_cpu()?;
```

### Tensors

`Tensor` represents multi-dimensional arrays with a specific shape:

```rust
use hologram::Tensor;

// From a vector with shape
let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

// Zeros tensor
let zeros = Tensor::zeros(&executor, &[3, 3])?;

// Ones tensor
let ones = Tensor::ones(&executor, &[2, 3])?;

// Random tensor (uniform distribution)
let random = Tensor::rand(&executor, &[4, 4])?;
```

### Operations

Hologram provides a rich set of tensor operations:

#### Element-wise Operations

```rust
use hologram::ops;

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3])?;

// Arithmetic
let sum = ops::add(&executor, &a, &b)?;
let diff = ops::sub(&executor, &a, &b)?;
let prod = ops::mul(&executor, &a, &b)?;
let quot = ops::div(&executor, &a, &b)?;

// Mathematical functions
let sqrt_a = ops::sqrt(&executor, &a)?;
let exp_a = ops::exp(&executor, &a)?;
let log_a = ops::log(&executor, &a)?;
```

#### Reduction Operations

```rust
// Reduce operations
let sum = ops::reduce_sum(&executor, &tensor)?;
let mean = ops::reduce_mean(&executor, &tensor)?;
let max = ops::reduce_max(&executor, &tensor)?;
let min = ops::reduce_min(&executor, &tensor)?;
```

#### Activation Functions

```rust
// Neural network activation functions
let relu_out = ops::relu(&executor, &tensor)?;
let sigmoid_out = ops::sigmoid(&executor, &tensor)?;
let tanh_out = ops::tanh(&executor, &tensor)?;
let gelu_out = ops::gelu(&executor, &tensor)?;
```

### Buffers

For lower-level operations, you can work directly with `Buffer`:

```rust
use hologram::Buffer;

// Allocate a buffer
let buffer = Buffer::allocate(&executor, 1024)?;

// Copy data to buffer
buffer.copy_from_host(&[1.0f32, 2.0, 3.0, 4.0])?;

// Copy data from buffer
let mut result = vec![0.0f32; 4];
buffer.copy_to_host(&mut result)?;
```

## Your First Program

Let's build a simple neural network layer:

```rust
use hologram::{Executor, Tensor, ops, Result};

fn dense_layer(
    executor: &Executor,
    input: &Tensor,
    weights: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    // Matrix multiplication: output = input @ weights
    let matmul = ops::matmul(executor, input, weights)?;

    // Add bias
    let with_bias = ops::add(executor, &matmul, bias)?;

    // Apply activation
    ops::relu(executor, &with_bias)
}

fn main() -> Result<()> {
    let executor = Executor::new_auto()?;

    // Input: batch_size=4, features=3
    let input = Tensor::rand(&executor, &[4, 3])?;

    // Weights: features_in=3, features_out=5
    let weights = Tensor::rand(&executor, &[3, 5])?;

    // Bias: features_out=5
    let bias = Tensor::zeros(&executor, &[5])?;

    // Forward pass
    let output = dense_layer(&executor, &input, &weights, &bias)?;

    println!("Output shape: {:?}", output.shape());
    println!("Output data: {:?}", output.to_vec()?);

    Ok(())
}
```

## Next Steps

Now that you understand the basics, explore these topics:

1. **[Multi-Backend Execution](multi-backend.md)** - Learn how to use different compute backends (CPU, CUDA, Metal, WebGPU)
2. **[Circuit Compilation](circuit-compilation.md)** - Understand how to compile quantum circuits to the Atlas ISA
3. **[Advanced Operations](advanced-operations.md)** - Learn about advanced tensor operations and optimizations
4. **[FFI Bindings](ffi-bindings.md)** - Use Hologram from Python, Kotlin, and other languages
5. **[Performance Tuning](performance-tuning.md)** - Optimize your Hologram applications

## Examples

Check out the [`examples/`](../../examples/) directory for more comprehensive examples:

- [`01_basic_executor.rs`](../../examples/01_basic_executor.rs) - Basic executor usage
- [`02_tensor_operations.rs`](../../examples/02_tensor_operations.rs) - Tensor operations
- [`03_circuit_compilation.rs`](../../examples/03_circuit_compilation.rs) - Circuit compilation
- [`04_multi_backend.rs`](../../examples/04_multi_backend.rs) - Multi-backend execution
- [`python/basic_usage.py`](../../examples/python/basic_usage.py) - Python FFI example
- [`python/dlpack_pytorch.py`](../../examples/python/dlpack_pytorch.py) - PyTorch integration

## Getting Help

- **Documentation**: Run `cargo doc --open` to browse API documentation
- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/your-org/hologram/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/your-org/hologram/discussions)

## Common Patterns

### Resource Management

Hologram uses Rust's RAII pattern for automatic resource cleanup:

```rust
{
    let executor = Executor::new_auto()?;
    let tensor = Tensor::zeros(&executor, &[100, 100])?;
    // Resources automatically freed when dropping out of scope
}
```

### Error Handling

All fallible operations return `Result<T, Error>`:

```rust
use hologram::Result;

fn my_computation() -> Result<Tensor> {
    let executor = Executor::new_auto()?;
    let tensor = Tensor::zeros(&executor, &[10, 10])?;
    let result = ops::relu(&executor, &tensor)?;
    Ok(result)
}
```

### Zero-Copy Operations

Hologram minimizes memory copies through:
- Reference counting (`Arc<T>`)
- View types (slices, strides)
- DLPack integration for external frameworks

```rust
// No copy - creates a view
let slice = tensor.slice(&[0..2, 0..3])?;

// No copy - reshapes metadata only
let reshaped = tensor.reshape(&[1, 4])?;
```

## Configuration

Configure Hologram via environment variables:

```bash
# Set backend
export HOLOGRAM_BACKEND=cuda

# Set device ID for multi-GPU systems
export HOLOGRAM_DEVICE_ID=0

# Enable profiling
export HOLOGRAM_ENABLE_PROFILING=true

# Set log level
export RUST_LOG=hologram=debug
```

Or use the configuration API:

```rust
use hologram::config::{HologramConfig, BackendConfig, BackendType};

let mut config = HologramConfig::default();
config.backend.backend_type = BackendType::Cuda;
config.backend.device_id = 0;
config.backend.enable_backend_profiling = true;

// Save configuration
config.backend.save("backend.toml")?;
```

Happy computing with Hologram! 🚀
