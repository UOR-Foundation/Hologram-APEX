# Hologram ONNX Compiler

**Status**: Design Phase
**Location**: `hologram-sdk/rust/hologram-onnx-compiler`
**Purpose**: Convert any model format (HuggingFace, PyTorch, TensorFlow, etc.) to Hologram's optimized IR

## Overview

The ONNX Compiler transforms arbitrary model formats into Hologram's internal representation, enabling:
- Direct model loading without ONNX export
- Compile-time optimization opportunities
- Robust handling of all model architectures
- Integration with `hologram-core`'s canonical compilation

## Architecture

```
Model (HF/PyTorch/TF/ONNX)
           ↓
    ONNX Compiler
           ↓
    Hologram IR (graph + operations)
           ↓
    hologram-core (canonical compilation)
           ↓
    Optimized Execution
```

## Key Differences from hologram-compiler

| Component | Purpose | Layer |
|-----------|---------|-------|
| **hologram-compiler** | Geometric algebra canonicalization | Core (`crates/`) |
| **hologram-onnx-compiler** | Model format conversion | SDK (`hologram-sdk/rust/`) |

- `hologram-compiler`: Compiles **operations** to canonical forms (H²=I, X²=I, etc.)
- `hologram-onnx-compiler`: Compiles **models** to Hologram IR (graph structure + ops)

## Design Goals

### 1. Universal Model Support
- Load any model format:
  - HuggingFace (SafeTensors, PyTorch pickle)
  - PyTorch (.pt, .pth files)
  - TensorFlow (.pb files)
  - ONNX (.onnx files)
- Support all architectures (transformers, CNNs, diffusion models, etc.)
- Handle custom model configurations

### 2. Compile-Time Optimization
- **Graph optimization**: Fold constants, eliminate dead code
- **Operation fusion**: Combine operations where beneficial
- **Memory planning**: Pre-compute buffer allocations
- **Quantization**: Apply quantization schemes at compile time

### 3. Integration with hologram-core
- Generate operations that compile to canonical forms
- Leverage geometric algebra optimizations
- Maintain compatibility with all backends (CPU, WebGPU, Metal, etc.)

## Proposed Crate Structure

```
hologram-sdk/rust/hologram-onnx-compiler/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # Public API
│   ├── loader/
│   │   ├── mod.rs              # Model loading interface
│   │   ├── safetensors.rs      # SafeTensors loader
│   │   ├── pytorch.rs          # PyTorch pickle loader
│   │   └── onnx.rs             # ONNX loader (reuse hologram-onnx)
│   ├── ir/
│   │   ├── mod.rs              # Hologram IR definition
│   │   ├── graph.rs            # Computation graph
│   │   ├── operation.rs        # Operation nodes
│   │   └── tensor.rs           # Tensor metadata
│   ├── transformers/
│   │   ├── mod.rs              # Transformer-specific logic
│   │   ├── bert.rs             # BERT architecture
│   │   ├── gpt.rs              # GPT architecture
│   │   ├── clip.rs             # CLIP architecture
│   │   └── config.rs           # Model config parsing
│   ├── optimizer/
│   │   ├── mod.rs              # Optimization passes
│   │   ├── constant_folding.rs # Fold constants
│   │   ├── dead_code.rs        # Eliminate dead nodes
│   │   └── fusion.rs           # Fuse operations
│   ├── compiler.rs             # Main compilation pipeline
│   └── error.rs                # Error types
└── tests/
    ├── integration_test.rs
    └── models/                 # Test model files
```

## Compilation Pipeline

```rust
// High-level API
pub struct OnnxCompiler {
    config: CompilerConfig,
}

impl ModelCompiler {
    pub fn compile(&self, model_path: &Path) -> Result<CompiledModel> {
        // 1. Load model weights and config
        let model = self.load_model(model_path)?;

        // 2. Parse architecture (BERT, GPT, etc.)
        let architecture = self.detect_architecture(&model)?;

        // 3. Convert to Hologram IR
        let ir = self.build_ir(model, architecture)?;

        // 4. Optimize IR
        let optimized_ir = self.optimize(ir)?;

        // 5. Compile to hologram-core operations
        let compiled = self.compile_to_core(optimized_ir)?;

        Ok(compiled)
    }
}

// Compiled output
pub struct CompiledModel {
    graph: Graph,
    weights: HashMap<String, Tensor>,
    metadata: ModelMetadata,
}
```

## Integration Points

### 1. With hologram-onnx
- Share common IR representation
- Reuse ONNX operators where applicable
- Provide ONNX as a fallback loader

### 2. With hologram-core
- Generate operations that leverage canonical compilation
- Use `hologram-core` buffers and executors
- Compile to ISA via `hologram-compiler`

### 3. With hologram-ai (future)
- Provide model architectures as building blocks
- Support custom models built from layers
- Enable fine-tuning and training

## Example Usage

```rust
use hologram_onnx_compiler::{OnnxCompiler, CompilerConfig};

// Compile a HuggingFace BERT model
let compiler = OnnxCompiler::new(CompilerConfig::default());
let model = compiler.compile("models/bert-base-uncased")?;

// Execute inference
let executor = OnnxExecutor::new(model.graph, model.weights)?;
let outputs = executor.run(inputs).await?;
```

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Create `hologram-onnx-compiler` crate
- [ ] Define Hologram IR types
- [ ] Implement SafeTensors loader
- [ ] Basic graph construction
- [ ] Integration with hologram-core

### Phase 2: Transformer Support (Week 3-4)
- [ ] BERT model compilation
- [ ] GPT model compilation
- [ ] CLIP model compilation
- [ ] Config file parsing
- [ ] Tokenizer integration

### Phase 3: Optimization (Week 5-6)
- [ ] Constant folding pass
- [ ] Dead code elimination
- [ ] Operation fusion
- [ ] Memory optimization
- [ ] Quantization support

### Phase 4: Production Ready (Week 7-8)
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Performance benchmarks
- [ ] Error handling
- [ ] WASM support

## Advantages Over ONNX Export

| Aspect | ONNX Export | ONNX Compiler |
|--------|-------------|----------------|
| **Reliability** | Can fail or produce broken graphs | Direct, controlled conversion |
| **Optimization** | Limited post-export | Full compile-time optimization |
| **Flexibility** | Fixed after export | Customizable compilation |
| **Debugging** | Opaque export process | Clear compilation stages |
| **Maintenance** | External tool dependency | Integrated solution |

## Dependencies

```toml
[dependencies]
hologram-core = { path = "../../../crates/hologram-core" }
hologram-onnx = { path = "../hologram-onnx" }  # Reuse IR and operators
safetensors = "0.4"  # SafeTensors loading
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
tracing = "0.1"

[dev-dependencies]
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

## Next Steps

1. **Review this design** - Ensure architecture aligns with project goals
2. **Create initial crate** - Set up `hologram-model-compiler` structure
3. **Prototype SafeTensors loader** - Prove concept with simple model
4. **Define IR** - Solidify Hologram IR representation
5. **Implement BERT** - First complete model compilation

## Questions to Resolve

1. Should IR be shared between `hologram-onnx` and `hologram-onnx-compiler`?
2. What level of type checking should occur at compile time?
3. How do we handle dynamic shapes vs static shapes?
4. Should quantization be part of compilation or a separate tool?
5. Do we need a custom tokenizer implementation, or integrate existing ones?

## References

- [SafeTensors Format](https://github.com/huggingface/safetensors)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [ONNX Operator Schemas](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- Project: `hologram-onnx` implementation
- Project: `hologram-compiler` for canonical compilation
