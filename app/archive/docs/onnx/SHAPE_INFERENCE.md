# ONNX Shape Inference in hologram-onnx-compiler

## Overview

The hologram-onnx-compiler now includes **full ONNX shape propagation** to correctly infer tensor shapes throughout the computation graph. This enables compilation of complex models with multidimensional tensors.

## Architecture

### Shape Inference Flow

```
ONNX Model
    ↓
1. Load ONNX → HologramGraph
    ↓
2. Apply Input Shapes (user-provided or auto-detected)
    ↓
3. Run Shape Inference Pass
    │  ├─ Traverse graph in topological order
    │  ├─ For each operation:
    │  │   ├─ Gather input shapes
    │  │   ├─ Apply operator-specific shape rules
    │  │   └─ Store output shapes in graph.shapes
    │  └─ Propagate through entire graph
    ↓
4. Optimize Graph (with known shapes)
    ↓
5. Execute Operators (with correct multidimensional shapes)
    ↓
6. Serialize Binary
```

### Implementation

**File**: `/workspace/hologram-sdk/rust/hologram-onnx-compiler/src/hrm/shape_inference.rs`

The `ShapeInference` engine implements ONNX shape rules for 30+ operators:

- **Element-wise ops** (Add, Sub, Mul, Div): Broadcasting rules
- **Matrix ops** (MatMul, Gemm): Matrix multiplication semantics
- **Shape ops** (Reshape, Transpose, Concat, Squeeze, Unsqueeze): Dimension manipulation
- **Reductions** (ReduceSum, ReduceMean, etc.): Dimension reduction with keepdims
- **Normalization** (LayerNormalization, Softmax): Shape preservation

## Supported Operators

### Element-wise Operations (Broadcasting)

```rust
// Add, Sub, Mul, Div, Equal, Greater, Less
broadcast_shapes([2, 1, 4], [3, 4]) → [2, 3, 4]
```

### Matrix Operations

```rust
// MatMul: [M, K] × [K, N] → [M, N]
MatMul([4, 8], [8, 3]) → [4, 3]

// Batched MatMul: [B, M, K] × [B, K, N] → [B, M, N]
MatMul([2, 4, 8], [2, 8, 3]) → [2, 4, 3]

// Gemm: Y = alpha * A * B + beta * C
Gemm([4, 8], [8, 3]) → [4, 3]
```

### Shape Manipulation

```rust
// Transpose
Transpose([2, 3, 4], perm=[0, 2, 1]) → [2, 4, 3]

// Squeeze (remove dimensions of size 1)
Squeeze([2, 1, 4, 1], axes=[1, 3]) → [2, 4]

// Unsqueeze (add dimensions of size 1)
Unsqueeze([2, 4], axes=[0, 2]) → [1, 2, 1, 4]

// Concat
Concat([[2, 3], [2, 4]], axis=1) → [2, 7]

// Flatten
Flatten([2, 3, 4], axis=1) → [2, 12]
```

### Reductions

```rust
// ReduceSum, ReduceMean, ReduceMax, ReduceMin
ReduceSum([2, 3, 4], axes=[1], keepdims=true) → [2, 1, 4]
ReduceSum([2, 3, 4], axes=[1], keepdims=false) → [2, 4]
```

## Usage

### 1. Automatic Input Shape Detection

The compiler auto-detects shapes for common models:

```bash
# CLIP text encoder automatically uses [1, 77] for input_ids
cargo run --bin hologram-onnx-compiler -- \
  --input clip_text_encoder.onnx \
  --output clip.holo \
  --verbose
```

Output:
```
📂 Step 1: Loading ONNX model...
   ✓ Detected CLIP text encoder - using shape [1, 77] for input_ids
   🔍 Inferring tensor shapes...
      Input 'input_ids': [1, 77]
      "Gather" [1024, 768], [1, 77] → [1, 77, 768]
      "MatMul" [1, 77, 768], [768, 768] → [1, 77, 768]
      ...
   ✓ Inferred shapes for 247 tensors
```

### 2. User-Provided Input Shapes

Specify custom input shapes for your model:

```bash
cargo run --bin hologram-onnx-compiler -- \
  --input model.onnx \
  --output model.holo \
  --input-shapes '{"x": [1, 3, 224, 224], "mask": [1, 224, 224]}'
```

### 3. Programmatic API

```rust
use hologram_onnx_compiler::Compiler;
use std::collections::HashMap;

let mut input_shapes = HashMap::new();
input_shapes.insert("input".to_string(), vec![1, 3, 224, 224]);

let compiler = Compiler::new()
    .with_verbose(true)
    .with_input_shapes(input_shapes);

compiler.compile("model.onnx", "model.holo")?;
```

## Shape Inference Examples

### Example 1: Linear Layer

```
Input: x [1, 256]
Initializer: weight [256, 128]
Initializer: bias [128]

MatMul(x, weight) → [1, 128]
Add([1, 128], [128]) → [1, 128]  # Broadcasting bias
```

### Example 2: Multi-Head Attention

```
Input: hidden_states [1, 77, 768]

# Query projection
MatMul([1, 77, 768], [768, 768]) → [1, 77, 768]
Reshape([1, 77, 768], [1, 77, 12, 64]) → [1, 77, 12, 64]
Transpose([1, 77, 12, 64], [0, 2, 1, 3]) → [1, 12, 77, 64]

# Key projection (same)
# Value projection (same)

# Attention scores
MatMul([1, 12, 77, 64], [1, 12, 64, 77]) → [1, 12, 77, 77]
Softmax([1, 12, 77, 77]) → [1, 12, 77, 77]

# Apply attention
MatMul([1, 12, 77, 77], [1, 12, 77, 64]) → [1, 12, 77, 64]
Transpose([1, 12, 77, 64], [0, 2, 1, 3]) → [1, 77, 12, 64]
Reshape([1, 77, 12, 64], [1, 77, 768]) → [1, 77, 768]
```

### Example 3: Convolutional Network

```
Input: image [1, 3, 224, 224]

Conv2D([1, 3, 224, 224], kernel=[64, 3, 7, 7]) → [1, 64, 112, 112]
BatchNorm([1, 64, 112, 112]) → [1, 64, 112, 112]
Relu([1, 64, 112, 112]) → [1, 64, 112, 112]
MaxPool([1, 64, 112, 112]) → [1, 64, 56, 56]
```

## Broadcasting Rules

The shape inference engine implements NumPy-style broadcasting:

```rust
// Compatible broadcasts
[3, 4] + [4] → [3, 4]
[2, 1, 4] + [3, 4] → [2, 3, 4]
[2, 3, 4] + [1] → [2, 3, 4]

// Incompatible broadcasts (error)
[3, 4] + [3, 5] → Error: dimension mismatch
```

## Error Handling

Shape inference detects and reports errors:

### 1. Dimension Mismatch

```
Error: MatMul dimension mismatch: 256 != 128
```

**Fix**: Ensure matrix dimensions are compatible for multiplication.

### 2. Unknown Input Tensor

```
Error: Unknown input tensor 'x' for node NodeIndex(5)
```

**Fix**: Provide input shapes using `--input-shapes` or ensure the tensor is defined in the graph.

### 3. Shape Not Inferred

```
Error: Shape not yet inferred for tensor 'hidden' (producer NodeIndex(3))
```

**Fix**: This indicates a graph topology issue. Verify the ONNX model is valid.

## Testing

Run tests to verify shape inference:

```bash
# Test with add.onnx (simple element-wise)
cargo run --bin hologram-onnx-compiler -- \
  --input tests/test_models/add.onnx \
  --output /tmp/add.holo \
  --verbose

# Test with linear.onnx (matrix multiplication)
cargo run --bin hologram-onnx-compiler -- \
  --input tests/test_models/linear.onnx \
  --output /tmp/linear.holo \
  --input-shapes '{"x": [1, 3]}' \
  --verbose

# Test with matmul.onnx (batched matrix multiplication)
cargo run --bin hologram-onnx-compiler -- \
  --input tests/test_models/matmul.onnx \
  --output /tmp/matmul.holo \
  --verbose
```

## Implementation Details

### Operator Shape Rules

Each operator implements specific shape inference logic:

```rust
fn infer_op_output_shapes(
    &self,
    op_type: &str,
    input_shapes: &[Vec<i64>],
    attributes: &HashMap<String, AttributeProto>,
) -> Result<Vec<Vec<i64>>> {
    match op_type {
        "Add" | "Sub" | "Mul" | "Div" => {
            // Broadcasting
            Ok(vec![broadcast_shapes(&input_shapes[0], &input_shapes[1])?])
        }
        "MatMul" => {
            // Matrix multiplication
            self.infer_matmul_shape(&input_shapes[0], &input_shapes[1])
        }
        "Transpose" => {
            // Permutation
            let perm = get_ints_attr(attributes, "perm");
            // ...apply permutation...
        }
        // ... 30+ more operators
    }
}
```

### Graph Traversal

Shape inference uses topological sort to ensure shapes are inferred in dependency order:

```rust
// Traverse graph in topological order
let topo_order = graph.topological_sort()?;

for node_id in topo_order {
    // Gather input shapes (already inferred)
    let input_shapes = gather_input_shapes(node_id, graph)?;

    // Infer output shapes
    let output_shapes = infer_op_output_shapes(op_type, input_shapes)?;

    // Store for downstream nodes
    graph.shapes.insert((node_id, output_slot), output_shape);
}
```

## Benefits

1. **Correct Multidimensional Support**: Enables compilation of complex models with proper tensor shapes
2. **Early Error Detection**: Catches shape mismatches before execution
3. **Optimization Opportunities**: Enables shape-aware graph optimizations
4. **Type Safety**: Ensures operators receive correctly-shaped inputs

## Future Enhancements

- [ ] Dynamic dimension support (use symbolic shapes)
- [ ] Shape inference for control flow ops (If, Loop)
- [ ] Automatic input shape extraction from ONNX value_info
- [ ] Shape-aware memory planning
- [ ] Compile-time shape validation for all operators

## Related Files

- **Shape Inference**: `src/hrm/shape_inference.rs`
- **Graph IR**: `src/hrm/graph/ir.rs`
- **Compiler Integration**: `src/compiler/mod.rs`
- **Executor**: `src/compiler/executor.rs`

## Conclusion

Full ONNX shape propagation is now integrated into the hologram-onnx-compiler, enabling compilation of complex models with multidimensional tensors. The system correctly infers shapes using ONNX operator semantics and detects mismatches early in the compilation process.
