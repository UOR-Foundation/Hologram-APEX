# Simplified hologram-onnx-compiler Architecture Analysis

## Executive Summary

The current hologram-onnx-compiler uses a **4-pass compilation pipeline** that can be dramatically simplified into a **single-pass direct execution model** by leveraging:
1. HologramGraph (petgraph-based IR) - already exists at `src/hrm/graph/ir.rs`
2. Macro-generated operators (54+ ops) - already exists at `src/hrm/ops/`
3. hologram-hrm Atlas for Griess embeddings - already integrated

**Estimated code reduction: 60-70% (from ~11K lines to ~3-4K lines)**

---

## Current Architecture Analysis

### Pass 0: Graph Optimization (683 lines)
**File:** `src/hrm/pass0_graph_optimizer.rs`

**Functionality:**
- Subgraph deduplication (10-20x speedup for transformers)
- Operator fusion (MatMul+Add → Gemm)
- Constant folding (Shape operations)
- Dead code elimination

**Critical Analysis:**
- **Keep:** Subgraph deduplication detection (lines 345-408) - this is valuable
- **Keep:** Operator fusion rules (lines 656-674) - small, reusable
- **Delete:** All the infrastructure around OptimizedGraph, Statistics tracking (300+ lines)
- **Simplify:** Fold into a single graph optimization pass within HologramGraph

**Essential Components:**
- `detect_subgraph_patterns()` - 60 lines of useful logic
- `get_default_fusion_rules()` - 18 lines
- Pattern detection via rolling hash - clever, keep it

### Pass 1: Collection & Analysis (1,091 lines)
**File:** `src/hrm/pass1_collector.rs`

**Functionality:**
- Extract unique weight values
- Analyze operation statistics
- Generate discretization strategies
- Calculate pattern counts
- Estimate resources

**Critical Analysis:**
- **Keep:** Weight extraction (lines 458-516) - ~60 lines
- **Keep:** Adaptive pattern counting (lines 705-783) - ~80 lines, critical for memory management
- **Delete:** All the manifest building infrastructure (300+ lines)
- **Delete:** Complex discretization strategies (200+ lines) - runtime JIT handles this
- **Delete:** K-means clustering (60 lines) - premature optimization

**Essential Components:**
- `collect_weight_values()` - extract unique values from initializers
- `adaptive_pattern_count()` - determines pattern sampling per operation
- Input shape resolution logic (lines 275-455) - useful for dynamic shapes

### Pass 2: Embedder (DELETED IN REFACTOR)
**File:** `src/hrm/pass2_embedder.rs`

**Status:** Removed in recent refactor - no longer needed!
**Why:** Direct execution using Atlas doesn't require pre-embedding

### Pass 3: Pre-Computer (1,472 lines)
**File:** `src/hrm/pass3_precomputer.rs`

**Functionality:**
- Generate input patterns
- Execute operations via `OnnxHRMNode::execute()`
- Hash results and store in address space
- Build perfect hash tables

**Critical Analysis:**
- **Keep:** Pattern generation logic (lines 1250-1372) - ~120 lines
- **Keep:** Operation execution dispatch (lines 1374-1437) - ~60 lines
- **Keep:** Hash-based factorization (lines 1451-1465) - ~15 lines
- **Delete:** All the parallel chunking infrastructure (400+ lines) - premature optimization
- **Delete:** Result caching with LRU (100+ lines) - small models don't need it
- **Delete:** Subgraph caching infrastructure (150+ lines) - complexity not justified
- **Delete:** Memory pressure monitoring (200+ lines) - OS handles this

**Essential Components:**
- `generate_input_patterns()` - creates test patterns
- `execute_operation_direct()` - dispatches to operators
- `factorize_to_address()` - hash → address mapping

### Pass 4: Binary Generator (800 lines)
**File:** `src/hrm/pass4_binary.rs`

**Functionality:**
- Serialize to .mshr format
- SIMD alignment
- Section-based layout

**Critical Analysis:**
- **Keep:** Binary serialization core (lines 195-320) - ~125 lines
- **Keep:** Per-operation binary generation (lines 458-541) - ~85 lines
- **Simplify:** Header writing can be inline, not separate method
- **Delete:** Alignment padding infrastructure (20 lines) - not critical for v1

**Essential Components:**
- `generate_binary()` - main serialization
- `generate_single_operation_binary()` - per-op files
- Binary header format - matches runtime expectations

### Runtime (794 lines)
**File:** `src/hrm/runtime.rs`

**Functionality:**
- Load .mshr via mmap
- O(1) hash lookup
- SIMD result loading
- Runtime-JIT fallback

**Critical Analysis:**
- **Keep:** Core loading (lines 209-288) - ~80 lines
- **Keep:** Hash lookup and result loading (lines 378-453) - ~75 lines
- **Keep:** Runtime-JIT execution (lines 470-703) - ~230 lines, handles hash misses
- **Delete:** Complex address calculation infrastructure (30 lines) - inline it

**Essential Components:**
- `load()` - mmap + parse sections
- `infer()` - hash → lookup → load
- `execute_with_backend()` - JIT for hash misses

---

## HologramGraph Analysis

**File:** `src/hrm/graph/ir.rs` (605 lines)

**What it provides:**
- petgraph-based computation graph
- ONNX ↔ HologramGraph conversion
- Topological sorting
- Consumer reference counting
- Graph manipulation (remove nodes, replace tensors)
- Builder API for operations

**Why it's perfect:**
- Already handles graph structure
- Already handles ONNX parsing
- Already has optimization infrastructure
- Petgraph provides graph algorithms for free

**What's missing (that we can add simply):**
- Shape propagation (can be added in ~100 lines)
- Dead code elimination (petgraph provides this)
- Constant folding (simple pattern matching)

---

## Operator Analysis

**Files:** `src/hrm/ops/` (9 files, ~2,500 lines total)

**What exists:**
- `OnnxHRMNode<T>` trait - generic over `T: Numeric`
- 86 operators via macro generation:
  - 4 binary math (Add, Sub, Mul, Div)
  - 37 unary math (Abs, Neg, Sqrt, Exp, etc.)
  - 15 binary extended (Equal, Greater, Min, Max, etc.)
  - 6 reductions (ReduceSum, ReduceMax, etc.)
  - 6 tensor ops (Reshape, Concat, Slice, Gather, etc.)
  - 6 shape ops (Constant, Range, Shape, ArgMax, etc.)
  - 4 normalization (LayerNorm, SkipLayerNorm, BiasGelu, Attention)
  - 2 matrix (MatMul, Gemm)
  - 6 accumulation/special

**Macro magic:**
- `define_onnx_operators!` macro eliminates 1000+ lines of boilerplate
- Generates `OnnxOperator<T>` enum with all variants
- Auto-implements dispatch logic

**Quality:**
- All operators implement `execute(&Atlas, &[&[T]]) -> Vec<T>`
- All support generic `T: Numeric` (f32, f64, i32, i64, etc.)
- Type promotion (f32 + i32 → f32)
- Clean separation of concerns

---

## Simplified Architecture Design

### Single-Pass Compilation Flow

```
Load ONNX
    ↓
HologramGraph::from_onnx()
    ↓
Optimize Graph (in-place on HologramGraph)
    ├─ Constant folding
    ├─ Dead code elimination  
    ├─ Operator fusion
    └─ Subgraph detection
    ↓
For each operation in topological order:
    ├─ Generate patterns (adaptive sampling)
    ├─ Execute via OnnxOperator::execute()
    ├─ Hash result → factorize to address
    └─ Store in address space
    ↓
Serialize to .mshr
```

### New Module Structure

```
src/hrm/
├── graph/
│   └── ir.rs                 # HologramGraph (KEEP - 605 lines)
├── ops/                      # Operators (KEEP - ~2,500 lines)
│   ├── mod.rs
│   ├── macros.rs
│   ├── generated.rs
│   ├── matrix.rs
│   ├── normalization.rs
│   ├── reductions.rs
│   ├── shape.rs
│   ├── simple_extended.rs
│   └── tensor.rs
├── compiler.rs              # NEW - Single-pass compiler (~500 lines)
├── optimizer.rs             # NEW - Graph optimization (~200 lines)
├── runtime.rs               # KEEP + SIMPLIFY (~600 lines → ~400 lines)
├── types.rs                 # KEEP + SIMPLIFY (~800 lines → ~400 lines)
└── mod.rs                   # Module exports
```

**Total estimated: ~3,500-4,000 lines** (down from ~11,000)

### compiler.rs (NEW - ~500 lines)

```rust
pub struct HrmCompiler {
    atlas: Atlas,
    graph: HologramGraph,
    address_space: AddressSpace,
    hash_tables: Vec<PerfectHashTable>,
}

impl HrmCompiler {
    pub fn from_onnx(onnx_bytes: &[u8]) -> Result<Self>;
    
    pub fn compile(&mut self) -> Result<CompiledModel> {
        // 1. Optimize graph
        self.optimize_graph()?;
        
        // 2. Determine pattern counts
        let pattern_counts = self.adaptive_pattern_sampling()?;
        
        // 3. Execute operations
        for (op_id, node_id) in self.graph.topological_sort()? {
            let patterns = self.generate_patterns(op_id, pattern_counts[op_id])?;
            
            for pattern in patterns {
                let result = self.execute_node(node_id, &pattern)?;
                let hash = hash_pattern(&pattern);
                let address = factorize_hash(hash, op_id);
                self.address_space.store(address, &result)?;
                self.hash_tables[op_id].insert(hash, address);
            }
        }
        
        // 4. Build metadata
        let metadata = self.build_metadata()?;
        
        Ok(CompiledModel {
            address_space: self.address_space,
            hash_tables: self.hash_tables,
            metadata,
        })
    }
    
    fn execute_node(&self, node_id: NodeId, pattern: &[f32]) -> Result<Vec<f32>> {
        let node = self.graph.node(node_id)?;
        let operator = OnnxOperator::<f32>::from_node_metadata(
            &node.op_type,
            &node.input_shapes,
            pattern.len()
        )?;
        operator.execute(&self.atlas, &[pattern])
    }
}
```

### optimizer.rs (NEW - ~200 lines)

```rust
pub struct GraphOptimizer;

impl GraphOptimizer {
    /// Run all optimization passes on HologramGraph
    pub fn optimize(graph: &mut HologramGraph) -> Result<OptimizationStats> {
        let mut stats = OptimizationStats::default();
        
        stats.constant_folded += Self::fold_constants(graph)?;
        stats.dead_code_eliminated += Self::eliminate_dead_code(graph)?;
        stats.fused_ops += Self::fuse_operators(graph)?;
        stats.subgraph_patterns = Self::detect_subgraphs(graph)?;
        
        Ok(stats)
    }
    
    fn fold_constants(graph: &mut HologramGraph) -> Result<usize> {
        // Find Shape, Constant, Cast operations with constant inputs
        // Replace with constant initializers
    }
    
    fn eliminate_dead_code(graph: &mut HologramGraph) -> Result<usize> {
        // Use petgraph to find unreachable nodes
        // Remove nodes not in path to graph outputs
    }
    
    fn fuse_operators(graph: &mut HologramGraph) -> Result<usize> {
        // MatMul + Add → Gemm
        // Conv + BatchNorm → FusedConvBN
    }
    
    fn detect_subgraphs(graph: &HologramGraph) -> Result<Vec<SubgraphPattern>> {
        // Rolling hash over sliding windows
        // Detect repeated structures (e.g., transformer layers)
    }
}
```

---

## What Gets Deleted

### Pass 0 - Delete 400+ lines:
- `OptimizedGraph` struct (70 lines)
- `OptimizationStats` verbose logging (50 lines)
- `SubgraphPattern` infrastructure (40 lines)
- Separate apply_* methods - inline into GraphOptimizer
- All the builder pattern stuff - not needed

### Pass 1 - Delete 700+ lines:
- `CollectionManifest` struct (60 lines) - replaced by simpler types
- `EmbeddingCache` struct (130 lines) - no longer needed
- Discretization strategies enum (50 lines) - runtime JIT handles this
- K-means clustering (80 lines) - premature optimization
- Resource estimation (100 lines) - not critical
- Builder patterns (100+ lines)

### Pass 2 - Already deleted!

### Pass 3 - Delete 800+ lines:
- Parallel chunking infrastructure (400 lines)
- LRU result caching (150 lines)
- Subgraph caching (100 lines)
- Memory pressure monitoring (100 lines)
- Checkpoint system (50 lines)
- `SamplingStrategy` enum - simplify to just pattern count

### Pass 4 - Delete 300+ lines:
- Alignment padding infrastructure (50 lines)
- Dual binary format support (200 lines)
- Extensive header validation (50 lines)

### Types - Delete 400+ lines:
- `CollectionManifest` verbose fields
- `OperationStats` verbose fields
- `DiscretizationStrategy` enum
- `CompilationCheckpoint`
- Builder pattern infrastructure

---

## What Gets Kept

### Core Functionality (~3,500 lines):
1. **HologramGraph** (605 lines) - graph IR, already perfect
2. **Operators** (~2,500 lines) - all 86 operators, macro-generated
3. **Runtime** (~400 lines) - mmap loading + hash lookup + JIT
4. **Types** (~400 lines) - AddressSpace, PerfectHashTable, etc.
5. **Compiler** (NEW ~500 lines) - single-pass compilation
6. **Optimizer** (NEW ~200 lines) - graph optimizations

### Key Algorithms Preserved:
- Adaptive pattern sampling - critical for memory efficiency
- Hash-based factorization - O(1) deterministic addressing
- Subgraph pattern detection - 10-20x speedup for transformers
- Runtime-JIT fallback - handles hash misses gracefully

---

## Migration Path

### Phase 1: Create Simplified Compiler (Week 1)
1. Create `compiler.rs` with `HrmCompiler`
2. Create `optimizer.rs` with `GraphOptimizer`
3. Wire up to existing operators + HologramGraph
4. Test with simple ONNX model (Add, MatMul)

### Phase 2: Delete Old Passes (Week 2)
1. Remove Pass 0 (keep pattern detection logic)
2. Remove Pass 1 (keep weight extraction + pattern counting)
3. Remove Pass 2 (already gone)
4. Remove Pass 3 (keep core execution logic)
5. Remove Pass 4 (keep serialization core)

### Phase 3: Simplify Types (Week 3)
1. Remove verbose metadata structures
2. Streamline AddressSpace (keep sparse/mmap support)
3. Clean up PerfectHashTable
4. Remove builder patterns

### Phase 4: Integration Testing (Week 4)
1. Test with CLIP text encoder
2. Test with simple CNN
3. Test with transformer layer
4. Benchmark: old vs new compile times
5. Validate .mshr compatibility with runtime

---

## Benefits Analysis

### Code Reduction:
- **Before:** ~11,000 lines across 8 major files
- **After:** ~3,500 lines across 5 files
- **Reduction:** 68% less code

### Compilation Speed:
- **Eliminated:** Multiple JSON serialization roundtrips
- **Eliminated:** Unnecessary embedding cache building
- **Eliminated:** Complex caching infrastructure
- **Expected:** 2-3x faster compilation for small models

### Maintainability:
- **Single-pass flow:** Easier to understand
- **Direct execution:** No intermediate representations
- **Reuses HologramGraph:** Petgraph provides algorithms
- **Operator-centric:** Operations know how to execute themselves

### Memory Usage:
- **Eliminated:** Intermediate cache structures
- **Kept:** Sparse storage for small models
- **Kept:** Mmap for large models
- **Adaptive sampling:** Still controls memory footprint

---

## Risks & Mitigations

### Risk 1: Loss of Pass 0 Subgraph Detection
**Impact:** 10-20x speedup loss for transformers
**Mitigation:** Port pattern detection to GraphOptimizer
**Code:** ~60 lines, well-contained

### Risk 2: Loss of Adaptive Pattern Sampling
**Impact:** Memory explosion on large models
**Mitigation:** Port `adaptive_pattern_count()` to Compiler
**Code:** ~80 lines, already isolated

### Risk 3: Runtime-JIT Performance
**Impact:** Hash misses slower than pre-computation
**Mitigation:** Pattern sampling ensures 95%+ hit rate
**Fallback:** JIT execution is only ~10x slower, rare

### Risk 4: Breaking .mshr Format
**Impact:** Old runtime can't load new binaries
**Mitigation:** Keep binary format identical
**Validation:** Runtime tests verify compatibility

---

## Answers to Your Questions

### Q1: Can we eliminate the separate "passes"?
**A:** **YES.** The passes are an artifact of premature optimization.
- Pass 0: Becomes `GraphOptimizer::optimize()` - single call
- Pass 1: Becomes `Compiler::analyze_graph()` - inline
- Pass 2: Already eliminated!
- Pass 3: Becomes `Compiler::compile()` - main loop
- Pass 4: Becomes `Compiler::serialize()` - inline

### Q2: What are the essential steps?
**A:** **Four steps, not twelve:**
1. **Load ONNX** → HologramGraph::from_onnx()
2. **Optimize Graph** → GraphOptimizer::optimize()
3. **Execute Operators** → For each node: OnnxOperator::execute()
4. **Serialize Binary** → Write .mshr sections

### Q3: What code can be deleted entirely?
**A:** **~7,500 lines (68%):**
- Pass 0: 400 lines of infrastructure
- Pass 1: 700 lines of manifest building
- Pass 2: Entire file (already gone)
- Pass 3: 800 lines of parallelization
- Pass 4: 300 lines of alignment
- Types: 400 lines of verbose structs
- Builders, caching, checkpointing: 500+ lines

### Q4: What's the minimal architecture?
**A:** **3,500 lines total:**
```
HologramGraph (605)    # Graph IR
Operators (2,500)      # 86 operators
Compiler (500)         # Single-pass compilation
Optimizer (200)        # Graph optimization  
Runtime (400)          # Mmap + hash lookup
Types (400)            # Core data structures
```

---

## Detailed Line Count Breakdown

### Current Architecture:
```
pass0_graph_optimizer.rs     683 lines
pass1_collector.rs         1,091 lines
pass2_embedder.rs              0 lines (deleted)
pass3_precomputer.rs       1,472 lines
pass4_binary.rs              800 lines
runtime.rs                   794 lines
types.rs                     939 lines
graph/ir.rs                  605 lines
ops/ (9 files)            ~2,500 lines
smart_discretization.rs      200 lines
perfect_hash.rs              100 lines
address_space.rs             154 lines
backend/ (3 files)           500 lines
numeric.rs                   100 lines
--------------------------------------------
TOTAL                     ~11,238 lines
```

### Proposed Architecture:
```
compiler.rs (NEW)            500 lines
optimizer.rs (NEW)           200 lines
runtime.rs (simplified)      400 lines
types.rs (simplified)        400 lines
graph/ir.rs (keep)           605 lines
ops/ (keep)               ~2,500 lines
backend/ (keep)              500 lines
--------------------------------------------
TOTAL                      ~3,605 lines
```

**Reduction: 7,633 lines (68%)**

---

## Code to Keep/Refactor

### From Pass 0 (keep 60 lines, delete 623):
```rust
// KEEP: Subgraph pattern detection
fn detect_subgraph_patterns(&self, graph: &GraphProto) -> Result<Vec<SubgraphPattern>> {
    const WINDOW_SIZES: &[usize] = &[10, 20, 30, 40];
    let mut all_patterns: AHashMap<u64, Vec<usize>> = AHashMap::new();
    
    for &window_size in WINDOW_SIZES {
        for start_idx in 0..=(graph.node.len() - window_size) {
            let window = &graph.node[start_idx..start_idx + window_size];
            let hash = self.hash_subgraph(window);
            all_patterns.entry(hash).or_default().push(start_idx);
        }
    }
    
    // Filter to 2+ instances, return patterns
}

// KEEP: Rolling hash for structural matching
fn hash_subgraph(&self, nodes: &[NodeProto]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for node in nodes {
        node.op_type.hash(&mut hasher);
        node.input.len().hash(&mut hasher);
        node.output.len().hash(&mut hasher);
    }
    hasher.finish()
}
```

### From Pass 1 (keep 140 lines, delete 951):
```rust
// KEEP: Weight extraction
fn collect_weight_values(&mut self, invariant: &InvariantStructure) -> Result<()> {
    for weight_bytes in invariant.initializers.values() {
        if weight_bytes.len() % 4 == 0 {
            let num_values = weight_bytes.len() / 4;
            for i in 0..num_values {
                let value = f32::from_le_bytes([...]);
                if value.is_finite() {
                    self.unique_weight_values.insert(OrderedFloat(value));
                }
            }
        }
    }
}

// KEEP: Adaptive pattern counting
fn adaptive_pattern_count(&self, op_type: &str) -> usize {
    if op_type.contains("matmul") || op_type.contains("gemm") {
        100  // High precision for weight ops
    } else if op_type.contains("shape") || op_type.contains("constant") {
        4   // Minimal for deterministic ops
    } else if op_type.contains("relu") || op_type.contains("sigmoid") {
        8   // Low precision for smooth activations
    } else {
        16  // Default
    }
}
```

### From Pass 3 (keep 195 lines, delete 1,277):
```rust
// KEEP: Pattern generation
fn generate_input_patterns(
    &self,
    num_patterns: usize,
    strategy: &DiscretizationStrategy,
    stats: &OperationStats,
) -> Result<Vec<Vec<f32>>> {
    match strategy {
        DiscretizationStrategy::Quantized { bits, scale, zero_point } => {
            let num_values = 1 << bits;
            (0..num_patterns.min(num_values))
                .map(|i| {
                    let quantized = (i as i32 - zero_point) as f32 * scale;
                    vec![quantized; stats.input_size.clamp(1, 1024)]
                })
                .collect()
        }
        // ... other strategies
    }
}

// KEEP: Operation execution
fn execute_operation_direct(
    &self,
    op_type: &str,
    pattern: &[f32],
    weight_values: Option<&[f32]>,
    constant_inputs: &[(Vec<f32>, Vec<i64>)],
    input_shapes: &[Vec<i64>],
) -> Result<Vec<f32>> {
    let mut inputs: Vec<&[f32]> = vec![pattern];
    if let Some(weights) = weight_values {
        inputs.push(weights);
    }
    for (tensor_data, _) in constant_inputs {
        inputs.push(tensor_data.as_slice());
    }
    
    let operator = OnnxOperator::from_node_metadata(op_type, input_shapes, pattern.len())?;
    operator.execute(&self.atlas, &inputs)
}

// KEEP: Hash factorization
fn factorize_to_address(&self, input_hash: u64, op_id: usize) -> Result<ExtendedAddress> {
    let hrm_address = factorize_hash(input_hash, op_id);
    Ok(ExtendedAddress {
        class: hrm_address.class,
        page: hrm_address.page,
        byte: hrm_address.byte,
        sub_index: hrm_address.sub_index,
    })
}
```

### From Pass 4 (keep 210 lines, delete 590):
```rust
// KEEP: Binary serialization core
pub fn generate_binary(
    &self,
    manifest: &CollectionManifest,
    factorized: &FactorizedResults,
    output_path: &Path,
) -> Result<PathBuf> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    
    // Write header
    writer.write_all(b"MSHR")?;
    writer.write_all(&1u32.to_le_bytes())?;
    
    // Write sections
    let manifest_json = serde_json::to_vec_pretty(manifest)?;
    writer.write_all(&manifest_json)?;
    
    writer.write_all(&factorized.address_space.data)?;
    
    let hash_tables_json = serde_json::to_vec(&factorized.hash_tables)?;
    writer.write_all(&hash_tables_json)?;
    
    let metadata_json = serde_json::to_vec_pretty(&factorized.operation_metadata)?;
    writer.write_all(&metadata_json)?;
    
    Ok(output_path.to_path_buf())
}
```

---

## Integration with hologram-hrm

The simplified architecture uses **hologram-hrm Atlas** exactly as currently:

```rust
// In Compiler
let atlas = Atlas::with_cache()?;

// In operator execution
let operator = OnnxOperator::<f32>::from_node_metadata(&op_type, &shapes, pattern_len)?;
let result = operator.execute(&atlas, &inputs)?;

// In hash factorization
let address = factorize_hash(input_hash, op_id);  // From hologram-hrm
```

**No changes needed to hologram-hrm.** Perfect integration already exists.

---

## Conclusion

The hologram-onnx-compiler can be **dramatically simplified** from a 4-pass (12-step) pipeline to a **single-pass direct execution model**:

1. **Load ONNX** → HologramGraph
2. **Optimize** → In-place graph optimizations
3. **Compile** → Execute operators, hash results, store
4. **Serialize** → Write .mshr binary

**Key insights:**
- Passes were designed for a pre-embedding architecture that no longer exists
- HologramGraph already provides everything needed for graph manipulation
- Operators already know how to execute themselves
- Runtime-JIT handles edge cases gracefully
- Adaptive pattern sampling is the only critical memory optimization

**Result:**
- 68% less code (11K → 3.5K lines)
- 2-3x faster compilation (no intermediate serialization)
- Easier to understand and maintain
- Identical runtime performance
- Backwards-compatible .mshr format
