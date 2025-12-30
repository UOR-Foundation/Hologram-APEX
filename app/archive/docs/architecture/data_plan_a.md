# Data Plan A: Discretization-Based Pre-Compilation

**Date**: November 18, 2025
**Status**: Proposed Architecture
**Approach**: Traditional discretization with pattern sampling and hash table lookup

---

## Executive Summary

Plan A proposes a **discretization-based pre-compilation architecture** that compiles ONNX models by sampling finite pattern spaces for each operation and building hash tables for O(1) lookup. This approach balances implementation complexity with performance by pre-computing 95% of common cases and providing runtime fallback for edge cases.

### Key Characteristics

- **Finite Pattern Space**: 100-100K patterns per operation
- **Discretization**: Quantization, clustering, hashed buckets, or vocabulary-based
- **Hybrid Execution**: 95% hash table lookup, 5% runtime computation
- **Apache Arrow**: Zero-copy I/O for pattern data
- **Time to Market**: 2-3 months for production-ready implementation

---

## Architecture Overview

### Compilation Pipeline (4 Passes)

```
ONNX Model
    ↓
Pass 1: Value Collection
    → Extract all unique values from weights/constants
    → Identify value ranges and distributions per operation
    ↓
Pass 2: Pattern Embedding
    → Apply discretization strategy to input space
    → Generate finite representative patterns
    ↓
Pass 3: Pre-Computation
    → Execute operations on all patterns
    → Build hash tables: pattern → result
    ↓
Pass 4: Serialization
    → Write .holo binary (hash tables + metadata)
    → Apache Arrow format for efficient loading
```

### Execution Model

```rust
// Compile-time: Build hash tables
for pattern in discretized_patterns {
    let result = execute_operation(pattern);
    hash_table.insert(hash(pattern), result);
}

// Runtime: O(1) lookup with fallback
fn execute(input: &[f32]) -> Vec<f32> {
    let key = hash(discretize(input));
    match hash_table.get(key) {
        Some(result) => result.clone(),  // 95% case
        None => compute_runtime(input),   // 5% fallback
    }
}
```

---

## Mathematical Foundation

### Discretization Strategies

#### 1. Quantized Discretization

**Concept**: Bucket continuous values into discrete bins

```
Value Range: [min, max]
Bins: N = 256 (8-bit quantization)
Quantize: q(x) = ⌊(x - min) / (max - min) × (N-1)⌋
```

**Pattern Space Size**:
- Vector of dimension D with N bins: N^D patterns
- Example: D=77, N=256 → 256^77 ≈ 10^185 (too large!)
- **Solution**: Sample representative patterns (100-10K per operation)

**Advantages**:
- Simple to implement
- Fast quantization (single multiply + floor)
- Uniform coverage of value space

**Disadvantages**:
- Exponential pattern growth with dimension
- Information loss from quantization
- Requires careful bin selection

#### 2. Clustered Discretization

**Concept**: K-means clustering of actual weight values

```
1. Extract all weight values from ONNX model
2. Run k-means clustering: centroids = {c₁, c₂, ..., cₖ}
3. Quantize: q(x) = argmin_i |x - cᵢ|
```

**Pattern Space Size**:
- K clusters → K^D theoretical patterns
- Example: K=16, D=77 → 16^77 ≈ 10^93 (still too large)
- **Solution**: Sample only patterns that appear in real weights

**Advantages**:
- Adapts to actual data distribution
- Better precision for common values
- Reduces quantization error

**Disadvantages**:
- Requires pass over all weights
- Clustering overhead (O(K×N×iterations))
- Still exponential in dimension

#### 3. Hashed Buckets

**Concept**: Hash high-dimensional patterns to fixed-size buckets

```
Hash: h(x) = murmur3(x) % B  where B = bucket count
Pattern: p(x) = [h(x₀), h(x₁), ..., h(xₙ)]
```

**Pattern Space Size**:
- B buckets per element → B^D theoretical patterns
- Example: B=64, D=77 → manageable with sampling
- **Key insight**: Hash collisions group similar patterns

**Advantages**:
- Constant-time hashing
- Implicit similarity via collisions
- No training required

**Disadvantages**:
- Collision-dependent accuracy
- Hard to tune bucket count
- No control over which patterns collide

#### 4. Vocabulary-Based

**Concept**: Build vocabulary of most frequent weight values

```
1. Extract all weights: W = {w₁, w₂, ..., wₙ}
2. Sort by frequency: vocab = top_k(W, k=1000)
3. Quantize: q(x) = nearest(x, vocab) ∪ {UNK}
```

**Pattern Space Size**:
- V vocabulary items → V^D theoretical patterns
- Example: V=1000, D=77 → 1000^77 (too large)
- **Solution**: Only generate patterns from actual weights

**Advantages**:
- Exact representation of common values
- Small vocabulary (1K-10K items)
- Natural handling of sparse weights

**Disadvantages**:
- Unknown values map to UNK (requires fallback)
- Vocabulary extraction overhead
- Not suitable for activations (continuous values)

---

## Implementation Details

### Pass 1: Value Collection

**Goal**: Extract all constants/weights and identify value distributions

```rust
struct ValueCollector {
    weight_values: HashMap<String, Vec<f32>>,
    value_ranges: HashMap<String, (f32, f32)>,
    value_histograms: HashMap<String, Histogram>,
}

impl ValueCollector {
    fn collect(&mut self, graph: &GraphProto) -> Result<()> {
        // Extract initializers (weights)
        for init in &graph.initializer {
            let values = parse_tensor_data(init)?;
            self.weight_values.insert(init.name.clone(), values.clone());

            // Compute statistics
            let (min, max) = values.iter()
                .fold((f32::MAX, f32::MIN), |(min, max), &v| {
                    (min.min(v), max.max(v))
                });
            self.value_ranges.insert(init.name.clone(), (min, max));

            // Build histogram
            let hist = Histogram::from_values(&values, 256);
            self.value_histograms.insert(init.name.clone(), hist);
        }
        Ok(())
    }
}
```

**Output**:
- Weight value distributions per tensor
- Min/max ranges for quantization
- Histograms for adaptive binning

### Pass 2: Pattern Embedding

**Goal**: Generate finite representative patterns using selected strategy

```rust
enum DiscretizationStrategy {
    Quantized { bins: usize },
    Clustered { clusters: usize },
    HashedBuckets { buckets: usize },
    Vocabulary { size: usize },
}

struct PatternGenerator {
    strategy: DiscretizationStrategy,
    patterns_per_op: usize, // Target: 100-10K
}

impl PatternGenerator {
    fn generate_patterns(&self,
                         op_type: &str,
                         input_shapes: &[Vec<i64>],
                         value_stats: &ValueStats) -> Vec<Pattern> {
        match self.strategy {
            DiscretizationStrategy::Quantized { bins } => {
                self.generate_quantized_patterns(input_shapes, bins, value_stats)
            }
            DiscretizationStrategy::Clustered { clusters } => {
                self.generate_clustered_patterns(input_shapes, clusters, value_stats)
            }
            DiscretizationStrategy::HashedBuckets { buckets } => {
                self.generate_hashed_patterns(input_shapes, buckets)
            }
            DiscretizationStrategy::Vocabulary { size } => {
                self.generate_vocab_patterns(input_shapes, size, value_stats)
            }
        }
    }

    fn generate_quantized_patterns(&self,
                                   shapes: &[Vec<i64>],
                                   bins: usize,
                                   stats: &ValueStats) -> Vec<Pattern> {
        let mut patterns = Vec::new();

        // Strategy: Sample representative patterns
        // 1. Corner cases: min, max, zero
        // 2. Uniform sampling across bins
        // 3. High-density regions from histogram

        let total_elements: usize = shapes[0].iter().product::<i64>() as usize;
        let (min, max) = stats.range;
        let bin_width = (max - min) / bins as f32;

        // Corner cases
        patterns.push(vec![min; total_elements]);
        patterns.push(vec![max; total_elements]);
        patterns.push(vec![0.0; total_elements]);

        // Uniform sampling (10% of target)
        let uniform_count = self.patterns_per_op / 10;
        for _ in 0..uniform_count {
            let bin = rand::random::<usize>() % bins;
            let value = min + bin as f32 * bin_width;
            patterns.push(vec![value; total_elements]);
        }

        // High-density sampling from histogram (90% of target)
        let density_count = self.patterns_per_op - patterns.len();
        let high_density_bins = stats.histogram.top_bins(bins / 10);
        for _ in 0..density_count {
            let bin = high_density_bins[rand::random::<usize>() % high_density_bins.len()];
            let value = min + bin as f32 * bin_width;
            patterns.push(vec![value; total_elements]);
        }

        patterns
    }
}
```

**Output**:
- 100-10K representative patterns per operation
- Patterns cover: corner cases, uniform distribution, high-density regions
- Stored as Apache Arrow arrays for efficient I/O

### Pass 3: Pre-Computation

**Goal**: Execute operations on all patterns and build hash tables

```rust
struct PreComputationEngine {
    executor: GraphExecutor,
    hash_tables: HashMap<String, PatternHashTable>,
}

struct PatternHashTable {
    // Map: pattern_hash → result
    entries: AHashMap<u64, Vec<f32>>,
    metadata: OperationMetadata,
}

impl PreComputationEngine {
    fn precompute_operation(&mut self,
                           node: &NodeProto,
                           patterns: &[Pattern]) -> Result<()> {
        let mut hash_table = PatternHashTable::new();

        // Execute operation on each pattern
        for pattern in patterns {
            // Hash the input pattern
            let hash = compute_pattern_hash(pattern);

            // Execute operation
            let inputs = pattern.to_tensors()?;
            let result = self.executor.execute_node(node, &inputs)?;

            // Store in hash table
            hash_table.entries.insert(hash, result.to_vec()?);
        }

        // Store hash table for this operation
        self.hash_tables.insert(node.name.clone(), hash_table);

        Ok(())
    }

    fn compute_pattern_hash(pattern: &[f32]) -> u64 {
        // Fast hash function (murmur3 or xxhash)
        let mut hasher = XxHash64::default();

        // Quantize before hashing to handle floating point precision
        for &value in pattern {
            let quantized = (value * 1000.0).round() as i32;
            hasher.write_i32(quantized);
        }

        hasher.finish()
    }
}
```

**Output**:
- Hash tables: `HashMap<u64, Vec<f32>>` per operation
- Metadata: input shapes, output shapes, operation type

**Performance Characteristics**:
- Pre-computation time: 10-30 minutes for large models
- Hash table size: 10-500 MB per operation
- Lookup time: O(1) with ~5ns per hash

### Pass 4: Serialization

**Goal**: Write .holo binary with hash tables in Apache Arrow format

```rust
use arrow::array::{UInt64Array, Float32Array};
use arrow::record_batch::RecordBatch;
use arrow::ipc::writer::FileWriter;

struct HoloSerializer {
    arrow_schema: Schema,
}

impl HoloSerializer {
    fn serialize(&self,
                 graph: &HologramGraph,
                 hash_tables: &HashMap<String, PatternHashTable>,
                 output_path: &Path) -> Result<()> {
        let mut file = File::create(output_path)?;

        // Write header
        self.write_header(&mut file)?;

        // Write each operation's hash table in Arrow format
        for (op_name, hash_table) in hash_tables {
            self.write_hash_table(&mut file, op_name, hash_table)?;
        }

        Ok(())
    }

    fn write_hash_table(&self,
                        file: &mut File,
                        op_name: &str,
                        hash_table: &PatternHashTable) -> Result<()> {
        // Convert hash table to Arrow format
        let keys: Vec<u64> = hash_table.entries.keys().copied().collect();
        let values: Vec<f32> = hash_table.entries.values()
            .flat_map(|v| v.iter().copied())
            .collect();

        let key_array = UInt64Array::from(keys);
        let value_array = Float32Array::from(values);

        // Create Arrow RecordBatch
        let batch = RecordBatch::try_from_iter(vec![
            ("pattern_hash", Arc::new(key_array) as ArrayRef),
            ("result_data", Arc::new(value_array) as ArrayRef),
        ])?;

        // Write to file using Arrow IPC format
        let mut writer = FileWriter::try_new(file, &batch.schema())?;
        writer.write(&batch)?;
        writer.finish()?;

        Ok(())
    }
}
```

**Binary Format**:
```
.holo file structure:
┌─────────────────────────────────┐
│ Header (magic, version)         │
├─────────────────────────────────┤
│ Operation 1 Hash Table (Arrow)  │
│   - Keys: u64[] (pattern hashes)│
│   - Values: f32[] (results)     │
├─────────────────────────────────┤
│ Operation 2 Hash Table (Arrow)  │
├─────────────────────────────────┤
│ ...                             │
├─────────────────────────────────┤
│ Metadata (JSON)                 │
│   - Operation types             │
│   - Input/output shapes         │
│   - Discretization strategy     │
└─────────────────────────────────┘
```

---

## Apache Arrow Integration

### Why Apache Arrow?

1. **Zero-Copy Loading**: Memory-map .holo files directly
2. **Efficient Storage**: Columnar format with compression
3. **Interoperability**: Standard format for data exchange
4. **Performance**: 10-50x faster than JSON/protobuf

### Arrow Schema

```rust
use arrow::datatypes::{Schema, Field, DataType};

fn create_hash_table_schema() -> Schema {
    Schema::new(vec![
        Field::new("pattern_hash", DataType::UInt64, false),
        Field::new("result_data", DataType::Float32, false),
        Field::new("result_shape", DataType::List(
            Box::new(Field::new("item", DataType::Int64, false))
        ), false),
    ])
}
```

### Loading at Runtime

```rust
use arrow::ipc::reader::FileReader;

struct HoloRuntime {
    hash_tables: HashMap<String, ArrowHashTable>,
}

struct ArrowHashTable {
    keys: UInt64Array,    // Pattern hashes
    values: Float32Array, // Flattened results
    shapes: ListArray,    // Result shapes
}

impl HoloRuntime {
    fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = FileReader::try_new(file, None)?;

        let mut hash_tables = HashMap::new();

        // Read each operation's hash table
        for batch in reader {
            let batch = batch?;
            let op_name = batch.schema().metadata.get("op_name").unwrap();

            let keys = batch.column(0)
                .as_any().downcast_ref::<UInt64Array>().unwrap();
            let values = batch.column(1)
                .as_any().downcast_ref::<Float32Array>().unwrap();
            let shapes = batch.column(2)
                .as_any().downcast_ref::<ListArray>().unwrap();

            hash_tables.insert(
                op_name.clone(),
                ArrowHashTable {
                    keys: keys.clone(),
                    values: values.clone(),
                    shapes: shapes.clone(),
                }
            );
        }

        Ok(Self { hash_tables })
    }

    fn execute(&self, op_name: &str, input: &[f32]) -> Option<Vec<f32>> {
        let hash_table = self.hash_tables.get(op_name)?;

        // Compute input hash
        let hash = compute_pattern_hash(input);

        // Binary search in sorted keys (Arrow arrays are sorted)
        let idx = hash_table.keys.binary_search(&hash).ok()?;

        // Extract result (zero-copy slice)
        let shape = hash_table.shapes.value(idx);
        let total_elements: usize = shape.as_any()
            .downcast_ref::<Int64Array>().unwrap()
            .values().iter().product::<i64>() as usize;

        let start_idx = idx * total_elements;
        let end_idx = start_idx + total_elements;

        Some(hash_table.values.values()[start_idx..end_idx].to_vec())
    }
}
```

---

## Performance Analysis

### Memory Requirements

**Per Operation**:
- Patterns: `P × D × 4 bytes` (P=pattern count, D=dimension)
- Hash table: `P × (8 + R×4) bytes` (R=result size)
- Example: MatMul [77, 1024]
  - Patterns: 1000 × 77 × 4 = 308 KB
  - Results: 1000 × (8 + 1024×4) = 4.1 MB
  - **Total per operation: ~4.5 MB**

**Full Model** (100 operations):
- Total: 100 × 4.5 MB = **450 MB** (manageable)

**With higher pattern counts** (10K patterns):
- Total: 100 × 45 MB = **4.5 GB** (acceptable for desktop/server)

### Compilation Time

**Benchmarks** (estimated):
- Pass 1 (Value Collection): 1-5 seconds
- Pass 2 (Pattern Generation): 5-30 seconds
- Pass 3 (Pre-Computation): 10-30 minutes (depends on pattern count)
- Pass 4 (Serialization): 5-10 seconds

**Total**: 15-30 minutes for large models (SD, CLIP, etc.)

**Parallelization**:
- Pre-computation is embarrassingly parallel
- Use rayon to parallelize pattern execution
- Expected speedup: 8-16x on multi-core systems

### Runtime Performance

**Hash Table Lookup**:
- Hash computation: ~5-10ns
- Binary search: O(log P) ≈ 10-20ns for P=1000
- Memory copy: ~1-5µs for 1K elements
- **Total per operation: 1-5µs**

**Fallback Computation**:
- When pattern not found (5% case)
- Execute operation at runtime
- Typical cost: 10-100µs per operation
- **Amortized cost: 0.05 × 100µs = 5µs**

**Combined Performance**:
- 95% case: 5µs (lookup)
- 5% case: 100µs (fallback)
- **Average: 0.95×5 + 0.05×100 = 9.75µs per operation**

**Full Model Inference** (100 operations):
- Total latency: 100 × 10µs = **1ms** (excellent!)

---

## Pros and Cons

### Advantages ✅

1. **Simpler Implementation**
   - Well-understood discretization techniques
   - Standard hash table data structures
   - No complex mathematical machinery required

2. **Fast Time to Market**
   - 2-3 months to production-ready system
   - Can iterate on discretization strategy
   - Proven pattern: similar to neural network quantization

3. **Manageable Memory**
   - 450 MB - 4.5 GB for typical models
   - Fits in RAM on consumer hardware
   - Can be memory-mapped for larger models

4. **Parallel Compilation**
   - Pre-computation is embarrassingly parallel
   - Linear speedup with CPU cores
   - Can distribute across machines

5. **Flexible Discretization**
   - Can tune strategy per operation type
   - Adaptive based on value distributions
   - Easy to add new strategies

6. **Good Coverage**
   - 95% lookup rate achievable with proper sampling
   - Fallback ensures correctness for all inputs
   - Gradual improvement as patterns added

### Disadvantages ❌

1. **Incomplete Pre-Computation**
   - 5% fallback required (adds latency variance)
   - Cannot guarantee 100% O(1) lookup
   - Edge cases require runtime computation

2. **Information Loss**
   - Quantization introduces approximation error
   - Some patterns map to nearest neighbor
   - Accuracy depends on discretization granularity

3. **Pattern Selection Challenge**
   - How to choose representative patterns?
   - May miss important edge cases
   - Requires tuning per model

4. **Exponential Pattern Space**
   - N^D theoretical patterns
   - Must carefully sample to keep finite
   - High-dimensional inputs are challenging

5. **Model-Specific Optimization**
   - Each model needs custom tuning
   - Discretization strategy may vary
   - Harder to generalize across domains

6. **Storage Overhead**
   - 450 MB - 4.5 GB per model
   - Larger than original ONNX (10-100 MB)
   - Requires efficient compression

---

## Production Readiness Assessment

### Maturity: Medium-High

**Existing Infrastructure**:
- ✅ ONNX graph parsing (implemented)
- ✅ Operation execution (implemented)
- ✅ Apache Arrow integration (library available)
- ⚠️ Hash table construction (needs implementation)
- ⚠️ Discretization strategies (needs implementation)
- ⚠️ Pattern sampling (needs implementation)

**Implementation Effort**:
- Core pipeline: 4-6 weeks
- Discretization strategies: 2-4 weeks
- Arrow integration: 1-2 weeks
- Testing & optimization: 2-3 weeks
- **Total: 9-15 weeks (2-3 months)**

### Risk Assessment

**Technical Risks**:
- ⚠️ Pattern coverage may be insufficient (mitigated by fallback)
- ⚠️ Hash collisions could impact accuracy (use 64-bit hashes)
- ⚠️ Memory requirements could exceed budget (tune pattern count)

**Performance Risks**:
- ⚠️ Fallback rate may be higher than 5% (requires validation)
- ⚠️ Pre-computation time may be too long (parallelize)
- ✅ Lookup performance is guaranteed O(1)

**Operational Risks**:
- ⚠️ Model-specific tuning required (document best practices)
- ✅ Deployment is straightforward (single .holo file)
- ✅ Runtime dependencies are minimal (Arrow library)

### Recommendation

**Plan A is SUITABLE for production if**:
1. 5% fallback rate is acceptable
2. ~5-10µs average latency per operation is sufficient
3. 2-3 month development timeline is acceptable
4. Model-specific tuning is feasible

**Plan A is NOT SUITABLE if**:
1. 100% O(1) lookup is required (no fallback)
2. Sub-microsecond latency is needed
3. Generalization across all models without tuning is required
4. Storage overhead (5-10x ONNX size) is prohibitive

---

## Comparison with Plan B

See `/docs/architecture/data_plan_b.md` for the holographic factorization approach.

**Key Differences**:
- **Pattern Space**: Plan A = finite sampled patterns, Plan B = infinite compositional space
- **Lookup Rate**: Plan A = 95% (with fallback), Plan B = 100% (pure lookup)
- **Implementation**: Plan A = simpler (2-3 months), Plan B = complex (4-6 months)
- **Memory**: Plan A = 450 MB - 4.5 GB, Plan B = 18.8 GB (boundary basis)
- **Accuracy**: Plan A = approximation via discretization, Plan B = exact via base-96 encoding

**Use Cases**:
- **Choose Plan A if**: Faster time to market, simpler implementation, acceptable fallback rate
- **Choose Plan B if**: 100% pre-computation required, mathematical rigor needed, higher memory acceptable

---

## References

1. **Quantization Techniques**: Neural Network Quantization (Gholami et al., 2021)
2. **Hash Tables**: Perfect Hash Functions (Fredman et al., 1984)
3. **Apache Arrow**: Arrow Columnar Format (arrow.apache.org)
4. **K-means Clustering**: Lloyd's Algorithm (Lloyd, 1982)
5. **Murmur3 Hash**: MurmurHash (Appleby, 2016)

---

**Status**: Proposed for evaluation against Plan B
