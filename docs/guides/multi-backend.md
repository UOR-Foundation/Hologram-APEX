# Multi-Backend Execution

Hologram supports multiple compute backends for maximum performance and portability across different hardware platforms.

## Table of Contents

- [Supported Backends](#supported-backends)
- [Backend Selection](#backend-selection)
- [CPU Backend](#cpu-backend)
- [CUDA Backend](#cuda-backend)
- [Metal Backend](#metal-backend)
- [WebGPU Backend](#webgpu-backend)
- [Performance Comparison](#performance-comparison)
- [Best Practices](#best-practices)

## Supported Backends

| Backend | Platform | Hardware | Feature Flag |
|---------|----------|----------|--------------|
| **CPU** | All platforms | x86_64, ARM64 with SIMD | Default (always available) |
| **CUDA** | Linux, Windows | NVIDIA GPUs (Compute 5.0+) | `cuda` |
| **Metal** | macOS, iOS | Apple Silicon, AMD/Intel GPUs | `metal` |
| **WebGPU** | Browser, Native | Modern GPUs with WebGPU support | `wgpu` |

## Backend Selection

### Automatic Selection

The simplest way to get started is to let Hologram automatically detect the best available backend:

```rust
use hologram::Executor;

// Automatically selects the best backend
let executor = Executor::new_auto()?;
```

Auto-detection priority:
1. **Metal** on macOS/iOS
2. **CUDA** if NVIDIA GPU is available
3. **WebGPU** on wasm32 targets
4. **CPU** as fallback

### Explicit Backend Selection

For fine-grained control, explicitly specify the backend:

```rust
use hologram::Executor;

// CPU backend (always available)
let cpu = Executor::new_cpu()?;

// CUDA backend (requires `cuda` feature)
#[cfg(feature = "cuda")]
let cuda = Executor::new_cuda()?;

// Metal backend (requires `metal` feature)
#[cfg(feature = "metal")]
let metal = Executor::new_metal()?;

// WebGPU backend (requires `wgpu` feature)
#[cfg(feature = "wgpu")]
let webgpu = Executor::new_webgpu()?;
```

### Configuration-Based Selection

Use the configuration system for runtime backend selection:

```rust
use hologram::config::{BackendConfig, BackendType};

// Load from environment or config file
let mut config = BackendConfig::from_env()?;

// Or set explicitly
config.backend_type = BackendType::Cuda;
config.device_id = 0; // For multi-GPU systems

let executor = Executor::from_config(&config)?;
```

### Environment Variables

Configure the backend via environment variables:

```bash
# Select backend
export HOLOGRAM_BACKEND=cuda

# Select GPU device (for multi-GPU systems)
export HOLOGRAM_DEVICE_ID=1

# Enable backend profiling
export HOLOGRAM_ENABLE_PROFILING=true
```

## CPU Backend

The CPU backend uses SIMD instructions (AVX2, NEON) for accelerated computation.

### Features

- **Universal compatibility**: Works on all platforms
- **SIMD optimization**: Automatic vectorization
- **Multi-threading**: Parallel operations via Rayon
- **No external dependencies**: Always available

### Example

```rust
use hologram::{Executor, Tensor, ops};

let executor = Executor::new_cpu()?;

// Create large tensors
let a = Tensor::rand(&executor, &[1000, 1000])?;
let b = Tensor::rand(&executor, &[1000, 1000])?;

// Parallel matrix multiplication
let c = ops::matmul(&executor, &a, &b)?;

println!("Result shape: {:?}", c.shape());
```

### Performance Tips

- Enable optimizations: `cargo build --release`
- Use AVX2 on x86_64: `RUSTFLAGS="-C target-cpu=native"`
- Batch operations to amortize overhead
- Use in-place operations when possible

## CUDA Backend

The CUDA backend provides GPU acceleration on NVIDIA hardware.

### Requirements

- NVIDIA GPU with Compute Capability 5.0+ (Maxwell or newer)
- CUDA Toolkit 11.0 or newer
- cuDNN 8.0+ (optional, for neural network operations)

### Installation

```bash
# Install CUDA toolkit (Ubuntu/Debian)
sudo apt-get install nvidia-cuda-toolkit

# Verify installation
nvcc --version
nvidia-smi
```

Add the `cuda` feature to your `Cargo.toml`:

```toml
[dependencies]
hologram = { version = "0.1", features = ["cuda"] }
```

### Example

```rust
#[cfg(feature = "cuda")]
use hologram::{Executor, Tensor, ops};

#[cfg(feature = "cuda")]
fn run_on_cuda() -> hologram::Result<()> {
    // Create CUDA executor
    let executor = Executor::new_cuda()?;

    // Large-scale computations benefit from GPU
    let a = Tensor::rand(&executor, &[4096, 4096])?;
    let b = Tensor::rand(&executor, &[4096, 4096])?;

    // GPU-accelerated matrix multiplication
    let c = ops::matmul(&executor, &a, &b)?;

    println!("Computed on GPU: {:?}", c.shape());
    Ok(())
}
```

### Multi-GPU Systems

For systems with multiple GPUs:

```rust
use hologram::config::BackendConfig;

// Select GPU 1
let mut config = BackendConfig::default();
config.backend_type = BackendType::Cuda;
config.device_id = 1;

let executor = Executor::from_config(&config)?;
```

### Performance Tips

- Use larger batch sizes (>1024 elements) to saturate GPU
- Minimize CPU-GPU transfers
- Pin CPU memory for faster transfers
- Use tensor cores on Volta+ GPUs for FP16 operations

## Metal Backend

The Metal backend provides GPU acceleration on Apple platforms.

### Requirements

- macOS 11.0+ or iOS 14.0+
- Apple Silicon (M1/M2/M3) or AMD/Intel GPU

### Installation

Add the `metal` feature:

```toml
[dependencies]
hologram = { version = "0.1", features = ["metal"] }
```

### Example

```rust
#[cfg(feature = "metal")]
use hologram::{Executor, Tensor, ops};

#[cfg(feature = "metal")]
fn run_on_metal() -> hologram::Result<()> {
    // Create Metal executor (auto-detects on macOS)
    let executor = Executor::new_metal()?;

    // Leverage Apple Silicon unified memory
    let tensor = Tensor::rand(&executor, &[2048, 2048])?;
    let result = ops::relu(&executor, &tensor)?;

    println!("Computed on Metal: {:?}", result.shape());
    Ok(())
}
```

### Unified Memory Advantage

Apple Silicon's unified memory architecture allows zero-copy data sharing between CPU and GPU:

```rust
// No explicit copy needed - data shared via unified memory
let tensor = Tensor::from_vec(data, &shape)?;
let gpu_result = ops::process(&metal_executor, &tensor)?;
let cpu_data = gpu_result.to_vec()?; // Fast memcpy within unified memory
```

### Performance Tips

- Leverage unified memory to avoid copies
- Use Neural Engine for ML operations (automatic)
- Batch operations for better GPU utilization
- Profile with Instruments.app for optimization

## WebGPU Backend

The WebGPU backend enables GPU acceleration in browsers and cross-platform applications.

### Requirements

- Modern browser with WebGPU support (Chrome 113+, Safari 16.4+, Firefox nightly)
- Or native WebGPU implementation via `wgpu`

### Installation

Add the `wgpu` feature:

```toml
[dependencies]
hologram = { version = "0.1", features = ["wgpu"] }
```

For WebAssembly:

```toml
[dependencies]
hologram = { version = "0.1", features = ["wgpu"] }
wasm-bindgen = "0.2"
```

### Example (Native)

```rust
#[cfg(feature = "wgpu")]
use hologram::{Executor, Tensor, ops};

#[cfg(feature = "wgpu")]
fn run_on_webgpu() -> hologram::Result<()> {
    let executor = Executor::new_webgpu()?;

    let tensor = Tensor::rand(&executor, &[512, 512])?;
    let result = ops::sigmoid(&executor, &tensor)?;

    println!("WebGPU result: {:?}", result.shape());
    Ok(())
}
```

### Example (WASM)

```rust
use hologram::{Executor, Tensor, ops};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub async fn run_in_browser() -> Result<JsValue, JsValue> {
    // WebGPU is async in browsers
    let executor = Executor::new_webgpu()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let tensor = Tensor::ones(&executor, &[100, 100])
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(JsValue::from_str("Computation complete"))
}
```

### Performance Tips

- Use compute shaders for custom operations
- Minimize buffer synchronization
- Batch small operations
- Cache pipeline state objects

## Performance Comparison

Benchmarks on common operations (relative to CPU baseline):

| Operation | CPU (x1) | CUDA (RTX 3090) | Metal (M2 Max) | WebGPU (Chrome) |
|-----------|----------|------------------|----------------|-----------------|
| Matrix Mul (1024x1024) | 1.0x | **45x** | 12x | 8x |
| Element-wise Add | 1.0x | **28x** | 15x | 10x |
| Reduce Sum | 1.0x | **32x** | 14x | 9x |
| ReLU Activation | 1.0x | **40x** | 16x | 11x |
| Conv2D | 1.0x | **52x** | 18x | 12x |

*Benchmarks run on: Intel i9-12900K (CPU), NVIDIA RTX 3090 (CUDA), Apple M2 Max (Metal), Chrome 120 (WebGPU)*

## Best Practices

### 1. Profile Before Optimizing

Use the built-in profiling to identify bottlenecks:

```rust
use hologram::config::BackendConfig;

let mut config = BackendConfig::default();
config.enable_backend_profiling = true;

let executor = Executor::from_config(&config)?;

// Operations are now profiled
let result = ops::matmul(&executor, &a, &b)?;
```

### 2. Minimize Data Transfers

Keep data on the device as long as possible:

```rust
// ❌ Bad: Multiple transfers
let a = Tensor::from_vec(data_a, &shape)?;
let a_cpu = a.to_vec()?; // Transfer to CPU
let b = Tensor::from_vec(data_b, &shape)?;
let b_cpu = b.to_vec()?; // Transfer to CPU

// ✅ Good: Compute on device
let a = Tensor::from_vec(data_a, &shape)?;
let b = Tensor::from_vec(data_b, &shape)?;
let c = ops::add(&executor, &a, &b)?; // Stays on GPU
let result = c.to_vec()?; // One transfer at the end
```

### 3. Use Appropriate Backend for Workload

- **CPU**: Small tensors (<1000 elements), low-latency operations
- **CUDA**: Large-scale training, batch processing
- **Metal**: macOS/iOS apps, unified memory workflows
- **WebGPU**: Browser-based ML, cross-platform visualization

### 4. Batch Operations

Combine multiple operations into a single kernel launch:

```rust
// ❌ Bad: Multiple kernel launches
let x1 = ops::relu(&executor, &input)?;
let x2 = ops::mul(&executor, &x1, &weights)?;
let x3 = ops::add(&executor, &x2, &bias)?;

// ✅ Better: Fused operation (when available)
let output = ops::fused_linear(&executor, &input, &weights, &bias)?;
```

### 5. Monitor Resource Usage

```rust
use hologram::config::BackendConfig;

let config = BackendConfig::default();

// Set memory limits
config.cache_size_mb = Some(512);

// Set operation timeout
config.timeout_ms = Some(5000);

let executor = Executor::from_config(&config)?;
```

## Troubleshooting

### CUDA Backend Not Available

```
Error: CUDA backend not available: No CUDA-capable device found
```

**Solutions:**
1. Verify GPU: `nvidia-smi`
2. Install CUDA toolkit
3. Check driver version: `nvidia-smi | grep "Driver Version"`
4. Rebuild with `cuda` feature: `cargo build --features cuda`

### Metal Backend Fails on macOS

```
Error: Metal backend initialization failed
```

**Solutions:**
1. Update to macOS 11.0+
2. Check Metal support: `system_profiler SPDisplaysDataType | grep Metal`
3. Rebuild with `metal` feature

### WebGPU Not Available in Browser

```
Error: WebGPU not supported in this browser
```

**Solutions:**
1. Use Chrome 113+ or Safari 16.4+
2. Enable WebGPU in `chrome://flags`
3. Verify GPU acceleration is enabled

## Next Steps

- [Performance Tuning](performance-tuning.md) - Optimize your Hologram applications
- [Advanced Operations](advanced-operations.md) - Learn about custom kernels and advanced features
- [Circuit Compilation](circuit-compilation.md) - Compile quantum circuits for different backends

---

For more examples, see [`examples/04_multi_backend.rs`](../../examples/04_multi_backend.rs).
