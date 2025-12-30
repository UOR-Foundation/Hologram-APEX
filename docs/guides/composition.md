# Operation Composition Guide

This guide explains how external runtimes (like hologram-onnx) can compose complex operations from Hologram's primitive operations using the composition APIs.

## Architecture: Primitives + Composition

Hologram follows a **Primitives + Composition** architecture:

1. **hologram-core** provides mathematical primitives (add, mul, exp, reduce_sum, etc.)
2. **External runtimes** compose these primitives into complex operations (GroupNorm, Attention, Conv2D)

This design ensures:
- No endless growth of operations in core
- Runtime-specific optimizations stay in runtime crates
- All primitives available to any runtime

## Available Primitives

### Math Operations (`hologram_core::ops::math`)

| Operation | Function | Description |
|-----------|----------|-------------|
| Addition | `vector_add`, `scalar_add`, `broadcast_add` | Element-wise addition |
| Subtraction | `vector_sub`, `scalar_sub`, `broadcast_sub` | Element-wise subtraction |
| Multiplication | `vector_mul`, `scalar_mul`, `broadcast_mul` | Element-wise multiplication |
| Division | `vector_div`, `scalar_div`, `broadcast_div` | Element-wise division |
| Min/Max | `min`, `max` | Element-wise min/max |
| Abs/Neg | `abs`, `neg` | Absolute value/negation |
| Clip | `clip` | Clamp values to range |
| Sqrt | `sqrt`, `rsqrt` | Square root, reciprocal sqrt |
| Exp/Log | `exp`, `log` | Exponential and natural log |
| Erf | `erf` | Error function |
| Power | `pow` | Element-wise power |
| Where | `where_select` | Conditional selection |
| Gather | `gather` | Index-based gather |

### Activation Functions (`hologram_core::ops::activation`)

| Operation | Function | Description |
|-----------|----------|-------------|
| ReLU | `relu` | max(0, x) |
| Sigmoid | `sigmoid` | 1 / (1 + exp(-x)) |
| Tanh | `tanh` | Hyperbolic tangent |
| GELU | `gelu` | Gaussian Error Linear Unit |
| Softmax | `softmax` | Softmax normalization |

### Reduction Operations (`hologram_core::ops::reduce`)

| Operation | Function | Description |
|-----------|----------|-------------|
| Sum | `sum` | Sum reduction |
| Mean | `mean` | Average |
| Min | `min` | Minimum value |
| Max | `max` | Maximum value |

### Scan Operations (`hologram_core::ops::scan`)

| Operation | Function | Description |
|-----------|----------|-------------|
| Cumulative Sum | `cumsum` | Prefix sum |
| Cumulative Product | `cumprod` | Prefix product |

### Linear Algebra (`hologram_core::ops::linalg`)

| Operation | Function | Description |
|-----------|----------|-------------|
| MatMul | `matmul` | Matrix multiplication |
| MatVec | `matvec` | Matrix-vector product |
| Outer Product | `outer_product` | Outer product |

## Composition API

The `hologram_compiler` crate provides the composition infrastructure:

```rust
use hologram_compiler::{BufferId, DecomposeContext, Decomposable};
```

### DecomposeContext

Builder API for composing operations:

```rust
let mut ctx = DecomposeContext::new();

// Register input buffers
let x = BufferId::new(0);
let scale = BufferId::new(1);
let bias = BufferId::new(2);

// Compose operations
let mean = ctx.reduce_mean(x, Some(vec![1]))?;
let centered = ctx.sub(x, mean)?;
let variance = ctx.reduce_mean(ctx.mul(centered, centered)?, Some(vec![1]))?;
let std_inv = ctx.recip(ctx.sqrt(ctx.add_scalar(variance, 1e-5)?)?)?;
let normalized = ctx.mul(centered, std_inv)?;
let scaled = ctx.mul(normalized, scale)?;
let output = ctx.add(scaled, bias)?;
```

### Decomposable Trait

Implement for custom operations:

```rust
use hologram_compiler::{Decomposable, DecomposeContext, BufferId, Result};

struct LayerNorm {
    epsilon: f32,
}

impl Decomposable for LayerNorm {
    fn decompose(
        &self,
        ctx: &mut DecomposeContext,
        inputs: Vec<BufferId>,
    ) -> Result<Vec<BufferId>> {
        let x = inputs[0];
        let scale = inputs[1];
        let bias = inputs[2];

        // Compute mean
        let mean = ctx.reduce_mean(x, Some(vec![1]))?;

        // Compute variance
        let centered = ctx.sub(x, mean)?;
        let sq = ctx.mul(centered, centered)?;
        let variance = ctx.reduce_mean(sq, Some(vec![1]))?;

        // Normalize
        let var_eps = ctx.add_scalar(variance, self.epsilon)?;
        let std = ctx.sqrt(var_eps)?;
        let std_inv = ctx.recip(std)?;
        let normalized = ctx.mul(centered, std_inv)?;

        // Scale and shift
        let scaled = ctx.mul(normalized, scale)?;
        let output = ctx.add(scaled, bias)?;

        Ok(vec![output])
    }
}
```

### DecomposerRegistry

Register decomposers for runtime lookup:

```rust
use hologram_compiler::DecomposerRegistry;

let mut registry = DecomposerRegistry::new();

registry.register_simple("LayerNorm", |epsilon: f32| {
    Box::new(LayerNorm { epsilon })
});

// Lookup and use
if let Some(decomposer) = registry.get_decomposer("LayerNorm") {
    let outputs = decomposer.decompose(&mut ctx, inputs)?;
}
```

## Example: Composing GroupNorm

GroupNorm can be composed entirely from primitives:

```rust
struct GroupNorm {
    num_groups: usize,
    epsilon: f32,
}

impl Decomposable for GroupNorm {
    fn decompose(
        &self,
        ctx: &mut DecomposeContext,
        inputs: Vec<BufferId>,
    ) -> Result<Vec<BufferId>> {
        let x = inputs[0];      // (N, C, H, W)
        let scale = inputs[1];  // (C,)
        let bias = inputs[2];   // (C,)

        // Reshape to (N, G, C/G, H, W) for group computation
        // ... reshape logic using buffer metadata ...

        // Per-group mean and variance
        let mean = ctx.reduce_mean(x, Some(vec![2, 3, 4]))?;
        let centered = ctx.broadcast_sub(x, mean)?;
        let sq = ctx.mul(centered, centered)?;
        let variance = ctx.reduce_mean(sq, Some(vec![2, 3, 4]))?;

        // Normalize
        let var_eps = ctx.add_scalar(variance, self.epsilon)?;
        let std_inv = ctx.recip(ctx.sqrt(var_eps)?)?;
        let normalized = ctx.broadcast_mul(centered, std_inv)?;

        // Reshape back and apply scale/bias
        let scaled = ctx.broadcast_mul(normalized, scale)?;
        let output = ctx.broadcast_add(scaled, bias)?;

        Ok(vec![output])
    }
}
```

## Example: Composing Attention

Single-head attention from primitives:

```rust
struct Attention {
    d_model: usize,
}

impl Decomposable for Attention {
    fn decompose(
        &self,
        ctx: &mut DecomposeContext,
        inputs: Vec<BufferId>,
    ) -> Result<Vec<BufferId>> {
        let q = inputs[0];  // (seq_len, d_model)
        let k = inputs[1];  // (seq_len, d_model)
        let v = inputs[2];  // (seq_len, d_model)

        // Compute attention scores: Q * K^T
        let scores = ctx.outer_product(q, k)?;  // (seq_len, seq_len)

        // Scale by 1/sqrt(d_model)
        let scale = 1.0 / (self.d_model as f32).sqrt();
        let scaled_scores = ctx.mul_scalar(scores, scale)?;

        // Softmax (simplified - actual impl needs proper axis handling)
        let exp_scores = ctx.exp(scaled_scores)?;
        let sum_exp = ctx.reduce_sum(exp_scores, Some(vec![1]))?;
        let attention = ctx.broadcast_div(exp_scores, sum_exp)?;

        // Apply attention to values
        let output = ctx.outer_product(attention, v)?;

        Ok(vec![output])
    }
}
```

## Direct Primitive Usage

For maximum performance, use primitives directly from hologram-core:

```rust
use hologram_core::{Executor, Buffer, ops};

fn layer_norm_direct(
    exec: &mut Executor,
    x: &Buffer<f32>,
    scale: &Buffer<f32>,
    bias: &Buffer<f32>,
    output: &mut Buffer<f32>,
    batch_size: usize,
    features: usize,
    epsilon: f32,
) -> hologram_core::Result<()> {
    // Allocate temporaries
    let mut mean = exec.allocate::<f32>(batch_size)?;
    let mut variance = exec.allocate::<f32>(batch_size)?;
    let mut centered = exec.allocate::<f32>(batch_size * features)?;

    // Compute mean (per batch)
    ops::reduce::mean(exec, x, &mut mean, batch_size)?;

    // Center the data
    ops::math::broadcast_sub(exec, x, &mean, &mut centered, batch_size, features)?;

    // Compute variance
    let mut sq = exec.allocate::<f32>(batch_size * features)?;
    ops::math::vector_mul(exec, &centered, &centered, &mut sq, batch_size * features)?;
    ops::reduce::mean(exec, &sq, &mut variance, batch_size)?;

    // Normalize: (x - mean) / sqrt(var + eps)
    let mut std_inv = exec.allocate::<f32>(batch_size)?;
    ops::math::scalar_add(exec, &variance, epsilon, &mut variance, batch_size)?;
    ops::math::sqrt(exec, &variance, &mut std_inv, batch_size)?;
    ops::math::rsqrt(exec, &variance, &mut std_inv, batch_size)?;

    // Apply normalization
    ops::math::broadcast_mul(exec, &centered, &std_inv, output, batch_size, features)?;

    // Scale and bias
    ops::math::broadcast_mul(exec, output, scale, output, batch_size, features)?;
    ops::math::broadcast_add(exec, output, bias, output, batch_size, features)?;

    Ok(())
}
```

## Performance Considerations

1. **SIMD Fast Path**: Operations automatically use SIMD (AVX-512/AVX2/SSE4.1) for f32 arrays under 262K elements
2. **Zero-Copy**: Buffer handles enable zero-copy data access
3. **Parallel Variants**: Use `*_par` variants for large arrays (e.g., `vector_add_par`)
4. **Composition Overhead**: DecomposeContext builds a computation graph; for hot paths, use direct primitives

## Migration from Core Operations

If you're migrating complex operations out of hologram-core to a runtime:

1. **Identify primitive decomposition**: Break operation into math primitives
2. **Implement Decomposable trait**: Create struct implementing decomposition logic
3. **Register with runtime**: Add to runtime's DecomposerRegistry
4. **Test equivalence**: Verify outputs match original implementation

This keeps hologram-core focused on primitives while allowing runtimes to provide optimized implementations for their specific use cases.
