# hologram-onnx-compiler Complete Refactoring Plan

## Vision: Graph-Based Compilation with HRM Integration

Modernize the ONNX compiler to use:
1. **HologramGraph** - petgraph-based graph structure
2. **Macro-generated operators** - type-generic, zero-boilerplate ops
3. **hologram-hrm Atlas** - Griess algebra address resolution

## Current State (Before Refactor)

### Problems

```rust
// ❌ Pass 0-4 use InvariantStructure (flat vectors)
pub struct InvariantStructure {
    nodes: Vec<NodeInfo>,          // No graph structure
    topology: Vec<u32>,            // Manual ordering
    // ... ~500 lines of redundant code
}

// ❌ Operators execute without graph context
for (op_id, node) in invariant.nodes.iter().enumerate() {
    // Manual index arithmetic, no dependency tracking
}

// ❌ Atlas is passed but unused
fn execute(&self, _atlas: &Atlas, inputs: &[&[T]]) -> Result<Vec<T>> {
    // _atlas parameter ignored!
}
```

### Code Redundancy

- **2,150 lines** of obsolete code across 5 files
- **3 different** graph representations (InvariantStructure, HologramGraph, raw ONNX)
- **Manual implementations** of algorithms that petgraph provides
- **No integration** between macro ops, graph structure, and HRM

## Target State (After Refactor)

### Unified Architecture

```rust
// ✅ Single graph structure
pub struct HologramGraph {
    graph: StableGraph<GraphNode, Dependency>,
    embeddings: FxHashMap<(NodeId, u8), GriessVector>,  // HRM integration
    shapes: FxHashMap<(NodeId, u8), Vec<i64>>,
    // ... ~600 lines, all necessary
}

// ✅ Graph-aware execution
let atlas = Atlas::with_cache()?;
for node_id in graph.topological_sort()? {
    let node = graph.node(node_id).unwrap();
    let operator = OnnxOperator::from_node(node)?;

    // Dependencies automatically tracked by graph
    let inputs = graph.inputs(node_id)
        .map(|(src_id, _)| &results[src_id])
        .collect();

    // Execute with HRM Atlas for address resolution
    let outputs = operator.execute(&atlas, &inputs)?;

    // Store embedding in graph
    let embedding = embed_to_griess(&outputs, &atlas)?;
    graph.embeddings.insert((node_id, 0), embedding);
}

// ✅ Atlas used for address resolution
fn execute(&self, atlas: &Atlas, inputs: &[&[T]]) -> Result<Vec<T>> {
    // Use atlas.get_vector(class) for Griess embeddings
    // Enable compile-time address resolution
}
```

## Refactoring Steps

### Phase 1: Core Graph Integration (Priority 1)

**Goal**: Replace InvariantStructure with HologramGraph in all passes

#### Step 1.1: Update Pass 1 (Collection)

**File**: `src/hrm/pass1_collector.rs`

**BEFORE**:
```rust
// Lines 100-150
use crate::invariant_extractor::{InvariantExtractor, InvariantStructure};

pub fn collect_and_analyze(&self, onnx_bytes: &[u8]) -> Result<CollectionManifest> {
    let extractor = InvariantExtractor::new();
    let invariant = extractor.extract(onnx_bytes)?;  // ❌ Creates flat structure

    for (op_id, node) in invariant.nodes.iter().enumerate() {
        // Manual iteration
    }
}
```

**AFTER**:
```rust
use crate::hrm::graph::{HologramGraph, NodeId};
use hologram_hrm::Atlas;

pub fn collect_and_analyze(&self, onnx_bytes: &[u8]) -> Result<(HologramGraph, CollectionManifest)> {
    // Parse ONNX
    let model = ModelProto::decode(onnx_bytes)?;

    // Build graph using existing method
    let graph = HologramGraph::from_onnx(&model.graph.unwrap())?;  // ✅ Use graph structure

    // Topological traversal (automatic ordering)
    for node_id in graph.topological_sort()? {
        let node = graph.node(node_id).unwrap();
        // Graph-aware processing
    }

    Ok((graph, manifest))
}
```

**Changes**:
- Delete `invariant_extractor.rs` import
- Use `HologramGraph::from_onnx()`
- Iterate with `graph.topological_sort()`
- Return `HologramGraph` for later passes

**Lines Deleted**: ~200

#### Step 1.2: Update Pass 3 (Pre-Computation)

**File**: `src/hrm/pass3_precomputer.rs`

**BEFORE**:
```rust
// Lines 1400-1500
pub fn precompute(&self, invariant: &InvariantStructure, manifest: &CollectionManifest)
    -> Result<PrecomputeResult>
{
    for (op_id, node) in invariant.nodes.iter().enumerate() {
        // Manual node iteration
        let inputs = self.get_inputs_for_op(op_id, node, &invariant)?;
        let outputs = self.execute_operation_direct(
            &node.op_type,
            &inputs[0],  // Hardcoded input access
            // ...
        )?;
    }
}
```

**AFTER**:
```rust
pub fn precompute(&self, graph: &mut HologramGraph, manifest: &CollectionManifest)
    -> Result<PrecomputeResult>
{
    // Create Atlas for HRM integration
    let atlas = Atlas::with_cache()?;

    // Topological execution (dependencies guaranteed)
    let mut results: FxHashMap<NodeId, Vec<f32>> = FxHashMap::default();

    for node_id in graph.topological_sort()? {
        let node = graph.node(node_id).unwrap();

        // Get inputs from graph dependencies
        let input_data: Vec<&[f32]> = graph.inputs(node_id)
            .iter()
            .map(|(src_id, dep)| {
                if let Dependency::Data { output_slot, .. } = dep {
                    results[src_id].as_slice()
                } else {
                    &[]
                }
            })
            .collect();

        // Execute operator with Atlas
        let operator = OnnxOperator::from_node_metadata(
            &node.op_type,
            &self.get_input_shapes(node_id, graph),
            pattern_len,
        )?;
        let outputs = operator.execute(&atlas, &input_data)?;

        // Store result
        results.insert(node_id, outputs.clone());

        // Compute Griess embedding (HRM integration)
        let embedding = self.embed_to_griess(&outputs, &atlas)?;
        graph.embeddings.insert((node_id, 0), embedding);
    }

    Ok(PrecomputeResult { graph, results })
}
```

**Changes**:
- Accept `HologramGraph` instead of `InvariantStructure`
- Create `Atlas::with_cache()` for HRM integration
- Use `graph.topological_sort()` for execution order
- Use `graph.inputs()` for dependency tracking
- Store results in `HashMap<NodeId, Vec<f32>>`
- Populate `graph.embeddings` with Griess vectors

**Lines Deleted**: ~300
**Lines Added**: ~150
**Net**: -150

#### Step 1.3: Delete Obsolete Code

**Files to DELETE**:
- `src/invariant_extractor.rs` (~500 lines)

**Imports to UPDATE**:
- Remove all `use crate::invariant_extractor::*`
- Add `use crate::hrm::graph::{HologramGraph, NodeId}`
- Add `use hologram_hrm::Atlas`

**Total Deleted**: ~500 lines

### Phase 2: HRM Integration (Priority 2)

**Goal**: Use hologram-hrm Atlas for address resolution and embeddings

#### Step 2.1: Implement Griess Embedding

**File**: `src/hrm/pass3_precomputer.rs`

**NEW METHOD**:
```rust
impl Pass3Precomputer {
    /// Embed operation output into Griess algebra using Atlas
    ///
    /// Maps f32 outputs → Griess 196,884-dim vectors for address resolution
    fn embed_to_griess(&self, outputs: &[f32], atlas: &Atlas) -> Result<GriessVector> {
        use hologram_hrm::embed::embed_integer;
        use hologram_hrm::griess::GriessVector;

        // Convert outputs to symbolic representation
        let symbolic = self.outputs_to_symbolic(outputs)?;

        // Embed into Griess algebra
        let embedding = embed_integer(&symbolic, atlas)?;

        Ok(embedding)
    }

    /// Convert f32 outputs to symbolic integer for embedding
    fn outputs_to_symbolic(&self, outputs: &[f32]) -> Result<SymbolicInteger> {
        // Strategy 1: Hash-based (fast)
        let hash = ahash::AHasher::default();
        for &val in outputs {
            hash.write_u32(val.to_bits());
        }
        let hash_value = hash.finish();

        // Convert to base-96 symbolic integer
        SymbolicInteger::from_u64(hash_value)
    }
}
```

**Integration Points**:
- Call `embed_to_griess()` after each operation in Pass 3
- Store embeddings in `graph.embeddings`
- Use embeddings in Pass 4 for address resolution

#### Step 2.2: Use Atlas in Operators

**File**: `src/hrm/ops/macros.rs`

**UPDATE MACRO**:
```rust
#[macro_export]
#[doc(hidden)]
macro_rules! _impl_elementwise_op {
    ($name:ident($x:ident): $doc:literal => $expr:expr) => {
        impl<T: $crate::hrm::numeric::Numeric> $crate::hrm::ops::OnnxHRMNode<T> for $name {
            fn execute(&self, atlas: &hologram_hrm::Atlas, inputs: &[&[T]]) -> $crate::error::Result<Vec<T>> {
                self.validate_inputs(inputs)?;

                // Example: Use atlas for address resolution (if needed)
                // let class_vector = atlas.get_vector(compute_class(inputs))?;

                // Standard element-wise execution
                Ok(inputs[0].iter().map(|&$x| $expr).collect())
            }
            // ...
        }
    };
}
```

**Note**: Most element-wise ops won't use Atlas directly, but it's available for:
- Custom ops that need Griess embeddings
- Address-based optimizations
- Future HRM-aware operations

### Phase 3: Graph-Based Optimizations (Priority 3)

**Goal**: Use petgraph algorithms for optimization passes

#### Step 3.1: Update Pass 0 (Graph Optimization)

**File**: `src/hrm/pass0_graph_optimizer.rs`

**BEFORE**:
```rust
// Lines 400-600: Custom subgraph detection
fn find_subgraph_patterns(&self, onnx_graph: &GraphProto) -> Vec<SubgraphPattern> {
    // Manual graph traversal
    // Custom hashing
    // ~200 lines of redundant code
}
```

**AFTER**:
```rust
use petgraph::algo::isomorphism::is_isomorphic_matching;

fn find_subgraph_patterns(&self, graph: &HologramGraph) -> Vec<SubgraphPattern> {
    // Use petgraph subgraph isomorphism
    let pg = graph.petgraph();

    // Detect patterns using petgraph algorithms
    let patterns = detect_transformer_layers(pg);

    patterns
}

fn detect_transformer_layers(pg: &StableGraph<GraphNode, Dependency>) -> Vec<SubgraphPattern> {
    // Use petgraph pattern matching
    // 10x less code, more robust
}
```

**Benefits**:
- Delete ~200 lines of custom graph algorithms
- Use battle-tested petgraph implementations
- Better pattern detection

#### Step 3.2: Implement Dead Code Elimination

**File**: `src/hrm/pass0_graph_optimizer.rs`

**NEW METHOD**:
```rust
impl Pass0GraphOptimizer {
    /// Eliminate dead code using graph structure
    fn eliminate_dead_code(&self, graph: &mut HologramGraph) -> Result<usize> {
        let mut removed = 0;

        // Build consumer map
        graph.build_consumer_map();

        // Remove nodes with zero consumers that aren't graph outputs
        let to_remove: Vec<NodeId> = graph.petgraph()
            .node_indices()
            .filter(|&node_id| {
                // Has no consumers and isn't graph output
                graph.outputs(node_id).is_empty() && !graph.is_graph_output(node_id)
            })
            .collect();

        for node_id in to_remove {
            graph.remove_node(node_id)?;
            removed += 1;
        }

        Ok(removed)
    }
}
```

**Benefits**:
- Use `graph.is_graph_output()` (already implemented)
- Use `graph.outputs()` for consumer tracking
- Automatic dependency tracking via petgraph

### Phase 4: Unified Macro System (Priority 4)

**Goal**: Design flexible macros for all ONNX operator types

#### Step 4.1: Extend Macro System

**File**: `src/hrm/ops/macros.rs`

**NEW MACROS**:
```rust
/// Define indexing operations (Gather, Slice, etc.)
#[macro_export]
macro_rules! define_indexing_ops {
    (
        $(
            $name:ident {
                inputs: $num_inputs:expr,
                compute: $compute:expr,
            }
        ),* $(,)?
    ) => {
        $(
            #[derive(Debug, Clone)]
            pub struct $name {
                pub axis: i64,
            }

            impl<T: Numeric> OnnxHRMNode<T> for $name {
                fn execute(&self, atlas: &Atlas, inputs: &[&[T]]) -> Result<Vec<T>> {
                    if inputs.len() != $num_inputs {
                        return Err(CompilerError::InvalidModel(format!(
                            "{} requires {} inputs, got {}",
                            stringify!($name), $num_inputs, inputs.len()
                        )));
                    }

                    let compute: fn(&[&[T]], i64) -> Vec<T> = $compute;
                    Ok(compute(inputs, self.axis))
                }

                fn validate_inputs(&self, inputs: &[&[T]]) -> Result<()> {
                    if inputs.len() != $num_inputs {
                        return Err(CompilerError::InvalidModel(format!(
                            "{} requires {} inputs, got {}",
                            stringify!($name), $num_inputs, inputs.len()
                        )));
                    }
                    Ok(())
                }

                fn op_type(&self) -> &'static str {
                    stringify!($name)
                }
            }
        )*
    };
}

/// Define shape operations (Reshape, Flatten, etc.)
#[macro_export]
macro_rules! define_shape_ops {
    (
        $(
            $name:ident: $doc:literal => $transform:expr
        ),* $(,)?
    ) => {
        $(
            #[doc = $doc]
            #[derive(Debug, Clone, Copy)]
            pub struct $name;

            impl<T: Numeric> OnnxHRMNode<T> for $name {
                fn execute(&self, _atlas: &Atlas, inputs: &[&[T]]) -> Result<Vec<T>> {
                    self.validate_inputs(inputs)?;
                    // Shape ops don't modify data
                    Ok($transform(inputs[0]))
                }

                fn validate_inputs(&self, inputs: &[&[T]]) -> Result<()> {
                    if inputs.is_empty() {
                        return Err(CompilerError::InvalidModel(format!(
                            "{} requires at least 1 input", stringify!($name)
                        )));
                    }
                    Ok(())
                }

                fn op_type(&self) -> &'static str {
                    stringify!($name)
                }
            }
        )*
    };
}
```

#### Step 4.2: Apply New Macros

**File**: `src/hrm/ops/tensor.rs`

**BEFORE** (~200 lines of manual implementations):
```rust
pub struct GatherOp { pub axis: i64 }
impl<T: Numeric> OnnxHRMNode<T> for GatherOp {
    fn execute(&self, _atlas: &Atlas, inputs: &[&[T]]) -> Result<Vec<T>> {
        // ~30 lines of boilerplate
    }
    // ...
}

pub struct ReshapeOp;
impl<T: Numeric> OnnxHRMNode<T> for ReshapeOp {
    fn execute(&self, _atlas: &Atlas, inputs: &[&[T]]) -> Result<Vec<T>> {
        // ~20 lines of boilerplate
    }
    // ...
}
// Repeat for 8 operators
```

**AFTER** (~50 lines total):
```rust
// Indexing operations
define_indexing_ops! {
    GatherOp {
        inputs: 2,
        compute: |inputs, axis| {
            gather_impl(inputs[0], inputs[1], axis)
        },
    },

    SliceOp {
        inputs: 1,
        compute: |inputs, axis| {
            slice_impl(inputs[0], axis)
        },
    },
}

// Shape operations
define_shape_ops! {
    ReshapeOp: "Reshape tensor (data unchanged)" => |data| data.to_vec(),
    FlattenOp: "Flatten tensor (data unchanged)" => |data| data.to_vec(),
    TransposeOp: "Transpose tensor (data unchanged)" => |data| data.to_vec(),
    SqueezeOp: "Remove dimensions of size 1" => |data| data.to_vec(),
    UnsqueezeOp: "Add dimensions of size 1" => |data| data.to_vec(),
}
```

**Lines Deleted**: ~150
**Lines Added**: ~50
**Net**: -100

## Migration Timeline

### Week 1: Core Infrastructure

- ✅ Day 1: Fix Gather operation (DONE)
- ⏳ Day 2: Replace InvariantStructure in Pass 1
- ⏳ Day 3: Update Pass 3 for graph traversal
- ⏳ Day 4: Delete invariant_extractor.rs
- ⏳ Day 5: Integration testing

### Week 2: HRM Integration

- Day 1: Implement Griess embedding in Pass 3
- Day 2: Update operators to use Atlas
- Day 3: Store embeddings in HologramGraph
- Day 4: Update Pass 4 to use embeddings
- Day 5: End-to-end testing

### Week 3: Graph Optimizations

- Day 1: Migrate Pass 0 to HologramGraph
- Day 2: Implement dead code elimination
- Day 3: Use petgraph for pattern matching
- Day 4: Optimize fusion passes
- Day 5: Performance benchmarking

### Week 4: Macro System

- Day 1: Design unified macro architecture
- Day 2: Implement indexing_ops macro
- Day 3: Implement shape_ops macro
- Day 4: Migrate all tensor ops
- Day 5: Cleanup and documentation

## Success Metrics

### Code Quality

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total lines | 4,800 | 2,650 | **-45%** |
| Graph implementations | 3 | 1 | **-67%** |
| Manual algorithms | 15 | 0 | **-100%** |
| Macro-generated ops | 54 | 70+ | **+30%** |

### Performance

| Metric | Before | After | Expected |
|--------|--------|-------|----------|
| Pass 0 time | N/A | O(V+E) | Sublinear |
| Pass 3 execution | O(N²) | O(N) | **2-10x faster** |
| Dead code detection | Manual | O(V+E) | **100x faster** |
| Memory usage | 3× redundant | 1× | **-67%** |

### Maintainability

- **Single source of truth**: HologramGraph for all graph operations
- **Standard algorithms**: petgraph instead of custom implementations
- **Type safety**: Macro-generated operators guarantee correctness
- **HRM integration**: Direct Atlas usage for address resolution

## Next Steps

1. **Start Phase 1, Step 1.1**: Update Pass 1 to use HologramGraph
2. **Test incrementally**: Ensure CLIP model still compiles
3. **Delete obsolete code**: Remove invariant_extractor.rs
4. **Document changes**: Update architecture docs

This refactoring will transform hologram-onnx-compiler from a legacy flat-structure compiler to a modern graph-based system with full HRM integration.
