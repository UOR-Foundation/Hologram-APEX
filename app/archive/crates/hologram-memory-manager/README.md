# Processor - Stream Processing with Automatic Gauge Construction

Production-grade stream processor where **chunking IS the gauge generator**.

## Overview

The processor crate provides a complete stream processing system built on the principle that period-driven chunking automatically generates the gauge structure. By detecting natural periodicities in input data and matching them to primorial boundaries, chunking adaptively generates optimal gauge structures. This enables universal content-addressed memory with domain heads acting as "mediatypes" to extract different interpretations from the same embedded data.

## Core Concept

```
Input → Period Detection → Primorial Matching → Gauge Construction → Memory Pool
    Autocorrelation analysis
            ↓
    Detected periods: [30, 210, ...]
            ↓
    Match to primordials: [30, 210, 2310, ...]
            ↓
    Primes: [{2,3,5}, {2,3,5,7}, ...]
            ↓
    Gauges: [G{2,3,5}, G{2,3,5,7}, ...]
            ↓
    Domain heads extract different modalities
```

**Key Insight**: Period-driven chunking automatically detects data periodicities and matches them to primorial boundaries, generating optimal gauge structures. No manual gauge selection, no hardcoded enums, infinite scalability through adaptive mathematical decomposition.

## Features

- **Period-Driven Chunking** - Automatic period detection via autocorrelation analysis
- **Adaptive Gauge Construction** - Gauges constructed from detected periods and primorial boundaries
- **Hybrid Optimization** - Entropy checks, sampling, and early termination for 6× speedup
- **Universal Memory Pools** - Content-agnostic storage with embedded gauges
- **Domain Heads as Mediatypes** - Multiple interpretations of same data (like MIME types for compute)
- **Stream Processing API** - Chainable operations (map, filter, chunk, embed)
- **Circuit Integration** - Integration points with circuit compilation
- **Zero Manual Maintenance** - Gauges scale infinitely without code changes

## Quick Start

### Basic Streaming

```rust
use hologram_memory_manager::{Stream, StreamProcessor};

// Create and process stream
let data: Vec<u8> = (0..1000).collect();
let stream = Stream::new(data);

// Chunk with automatic gauge construction (7 primorial levels)
let chunked = stream.chunk(7)?;

// Embed into memory pool
let context = chunked.embed()?;

println!("Embedded {} bytes into {} blocks with {} gauges",
    context.total_bytes,
    context.pool.len(),
    context.gauges_count
);
```

### Domain Heads

```rust
use hologram_memory_manager::{StreamProcessor, StatisticsDomainHead, RawDomainHead, Modality};

// Setup processor with domain heads
let mut processor = StreamProcessor::new();
processor.register_domain_head(StatisticsDomainHead);
processor.register_domain_head(RawDomainHead);

// Process input
let input: Vec<u8> = (0..200).collect();
let context = processor.process(input.clone())?;

// Extract statistics modality
let stats = processor.extract_modality(&context, "application/statistics")?;
match stats {
    Modality::Statistics { mean, variance, min, max, .. } => {
        println!("Mean: {:.2}, Variance: {:.2}, Range: {} to {}",
            mean, variance, min, max);
    }
    _ => unreachable!(),
}

// Extract raw modality (data reconstruction)
let raw = processor.extract_modality(&context, "application/octet-stream")?;
match raw {
    Modality::Raw(data) => {
        assert_eq!(data, input); // Perfect reconstruction
    }
    _ => unreachable!(),
}
```

## Architecture

### Modules

#### `chunking`
Period-driven chunking with automatic gauge construction:
- `PeriodDrivenChunker` - Detects periodicities and chunks at primorial boundaries
- `generate_n_primorials()` - Generate N primorials in sequence
- `factor_primorial()` - Extract primes from primorial value
- **Hybrid optimization**: Entropy checks, 1MB sampling, early termination (correlation > 0.8)

#### `memory`
Universal content-addressed memory pools:
- `UniversalMemoryPool` - Pool for embedded blocks
- `EmbeddedBlock` - Block with data + constructed gauge
- `EmbeddingResult` - Metadata from embedding operation

#### `domain`
Domain heads as mediatypes:
- `DomainHead` trait - Interface for modality extraction
- `DomainHeadRegistry` - Runtime registration of domain heads
- `StatisticsDomainHead` - Extract statistical properties
- `RawDomainHead` - Reconstruct original data
- `Modality` enum - Output formats (Statistics, Raw, Factors, Spectrum, etc.)

#### `stream`
Stream processing API:
- `Stream<T>` - Lazy stream with map/filter operations
- `ChunkedStream<T>` - Chunked stream with gauges
- `StreamContext` - Embedded context with pool

#### `executor`
Stream execution engine:
- `StreamProcessor` - Main processor with domain head registry
- Process arbitrary inputs with automatic gauge construction
- Extract multiple modalities from single embedding

#### `compiler`
Circuit integration (stub for future):
- `CircuitStreamCompiler` - Compile circuits to streams
- `CompiledCircuitStream` - Compiled circuit representation

## Examples

### Run Examples

```bash
# Basic streaming
cargo run --example basic_streaming

# Domain heads
cargo run --example domain_heads
```

## How It Works

### 1. Period Detection

The chunker analyzes input data for natural periodicities using autocorrelation:

```rust
// Quick entropy check - skip detection for non-periodic data
if !looks_periodic(input) {
    return chunk_with_default_sequence(input);  // Fast path
}

// Sample-based detection (first 1MB only)
let sample = &input[..1MB.min(input.len())];

// Early termination on strong period (correlation > 0.8)
for &primordial in &primordials {
    let correlation = autocorrelation(sample, primordial);
    if correlation > 0.8 {
        return chunk_at_single_period(input, primordial);
    }
}
```

**Optimization strategy**:
- **Entropy heuristic**: Skips detection for high-entropy data (6× speedup)
- **Sampling**: Analyzes first 1MB only (100× reduction for 100MB input)
- **Early termination**: Stops on first strong period (saves scanning all primordials)

### 2. Primorial Matching

Detected periods are matched to primorial boundaries:
- 1# = 1
- 2# = 2
- 3# = 2 × 3 = 6
- 5# = 2 × 3 × 5 = 30 ← Common period for many data types
- 7# = 2 × 3 × 5 × 7 = 210
- 11# = 2 × 3 × 5 × 7 × 11 = 2,310
- 13# = 2 × 3 × 5 × 7 × 11 × 13 = 30,030

### 3. Automatic Gauge Construction

For each detected period/primordial:
1. Extract primes from primordial: `30 → {2, 3, 5}`
2. Construct gauge: `Gauge::for_primes(&[2, 3, 5])`
3. Create chunk with data + gauge
4. Embed block with gauge

**Result**: Gauge structure emerges automatically from detected periodicities.

### 4. Universal Memory Pool

- Content-agnostic storage
- Blocks carry their constructed gauges
- No circuit-defined structure
- Uniform embedding for all inputs

### 5. Domain Heads as Mediatypes

Different circuits interpret the same pool in different ways:
- **Raw** → `application/octet-stream` → reconstructed data
- **(Future) Shor's** → `application/semiprime-factors` → prime factors
- **(Future) FFT** → `application/frequency-spectrum` → frequency spectrum
- **(Future) Compression** → `application/x-compressed` → compressed data

## Testing

```bash
# Run all tests
cargo test --package hologram-memory-manager

# Run specific module tests
cargo test --package hologram-memory-manager chunking
cargo test --package hologram-memory-manager memory
cargo test --package hologram-memory-manager domain

# Run integration tests (these are ignored by default)
cargo test --package hologram-memory-manager --test simd_domain_heads_integration -- --ignored

# Run all tests including ignored integration tests
cargo test --package hologram-memory-manager -- --include-ignored
```

**Test Coverage**: 93 tests covering all modules and integration scenarios

**Note**: Integration tests in `simd_domain_heads_integration.rs` are marked with `#[ignore]` and do not run by default in the workspace test pass. Run them explicitly with `--ignored` when needed.

## Benchmarking

The processor crate includes comprehensive benchmarks using [Criterion.rs](https://github.com/bheisler/criterion.rs) for performance analysis and optimization.

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench --package hologram-memory-manager

# Run specific benchmark suite
cargo bench --package hologram-memory-manager --bench domain_head_performance
cargo bench --package hologram-memory-manager --bench embedding_performance
cargo bench --package hologram-memory-manager --bench statistics
cargo bench --package hologram-memory-manager --bench optimization_methods
cargo bench --package hologram-memory-manager --bench period_chunking
cargo bench --package hologram-memory-manager --bench simd_domain_heads
cargo bench --package hologram-memory-manager --bench streaming
```

### Available Benchmarks

#### 1. **domain_head_performance** - Raw extraction benchmarks
Measures raw domain head extraction performance across different data sizes and implementation methods:
- `raw_extraction` - Benchmarks extraction from 1KB to 10MB
- `extraction_methods` - Compares extend_from_slice, pre-allocated, and iterator chain methods
- `block_access_patterns` - Direct iteration vs iterator chains

**Key metrics**: Throughput (MB/s), latency across data sizes

#### 2. **embedding_performance** - Embedding pipeline benchmarks
Analyzes the complete embedding pipeline to identify bottlenecks:
- `embedding` - Overall embedding performance from 1KB to 10MB
- `chunking` - Compares chunk_fast (zero-copy) vs chunk_and_construct_gauges (detection)
- `fast_vs_detection` - Fast path vs detection path comparison
- `block_allocation` - Arc-based vs move ownership overhead

**Key metrics**: Throughput (MB/s), chunking strategy comparison

#### 3. **statistics** - Statistics computation benchmarks
Compares different statistics computation strategies:
- `data_collection` - extend_from_slice, with_capacity, flat_map approaches
- `statistics_methods` - Multi-pass vs single-pass (Welford's) vs partial optimization
- `individual_stats` - Sum, mean, variance, min, max operations

**Key metrics**: Single-pass vs multi-pass speedup, cache efficiency

#### 4. **optimization_methods** - Implementation comparison benchmarks
Systematically benchmarks different extraction implementations:
- `extraction_methods` - 5 different methods including unsafe memcpy
- `allocation_overhead` - Allocation vs copy overhead breakdown
- `block_count_impact` - Effect of primordial levels (5-15) on performance

**Key metrics**: Method rankings, optimization opportunities

#### 5. **period_chunking** - Chunking strategy benchmarks
Evaluates period detection and chunking approaches:
- `chunking_strategies` - Fast path vs detection path from 1KB to 10MB
- `primordial_levels` - Impact of different primordial levels (5-15)
- `data_patterns` - Random, periodic, and constant data patterns

**Key metrics**: Fast path speedup (typically 6-10×), pattern sensitivity

#### 6. **simd_domain_heads** - SIMD acceleration benchmarks
Measures SIMD-accelerated domain head operations:
- `normalize_simd` - Normalization across 256 to 64K elements
- `filter_simd` - ReLU and clip filtering performance
- `aggregate_simd` - SIMD reduction operations
- `raw_parallel` - Parallel block copying (sequential vs parallel paths)

**Key metrics**: SIMD scaling, parallel efficiency

#### 7. **streaming** - Large input streaming benchmarks
Tests processor scalability with large streaming inputs:
- `streaming_large_inputs` - 1MB to 100MB inputs
- `streaming_throughput` - Sustained throughput measurement

**Key metrics**: Scalability, throughput consistency

### Benchmark Results Format

Criterion generates comprehensive reports including:
- **HTML reports** in `target/criterion/`
- **Statistical analysis** (mean, std dev, outliers)
- **Performance comparisons** between runs
- **Throughput measurements** (MB/s, GB/s)

### Performance Targets

Based on current benchmarks:

| Operation | Size | Target Throughput |
|-----------|------|------------------|
| Embedding (fast path) | 1-10MB | > 1900 MB/s |
| Raw extraction | 1-10MB | > 800 MB/s |
| SIMD normalization | 64KB | > 500 MB/s |
| Statistics (single-pass) | 1MB | > 600 MB/s |
| Period detection | 1MB sample | < 50ms |

### Optimization Insights

From benchmark analysis:

1. **Fast path is 6-10× faster** than detection path for non-periodic data
2. **Pre-allocation** reduces allocation overhead by ~30%
3. **Single-pass statistics** (Welford's algorithm) is 1.5-2× faster than multi-pass
4. **SIMD operations** scale linearly with data size
5. **Parallel extraction** benefits kick in at ~64KB threshold

## Comparison with Experiments

The processor crate **graduates** the data-driven gauge construction from experiments:

| Aspect | experiments/data_driven_gauges | processor |
|--------|-------------------------------|-----------|
| Status | Experimental | Production |
| API | Basic | Complete Stream API |
| Domain Heads | 2 (Stats, Raw) | Registry + extensible |
| Integration | None | Executor + compiler stubs |
| Documentation | Experiment docs | Full README + API docs |
| Examples | Demo only | Multiple examples |

## Dependencies

- `quantum-768` - Gauge system
- `hologram-compiler` - Circuit compilation (for integration)
- `bytemuck` - Type-safe byte casting
- `thiserror` - Error handling

## Future Enhancements

### Additional Domain Heads
- **ShorsDomainHead** - Extract semiprime factors
- **FFTDomainHead** - Extract frequency spectrum
- **CompressionDomainHead** - Compress using gauge structure

### k-bonacci Integration
Integrate k-bonacci sequences for enhanced decomposition patterns.

### Circuit Execution
Complete circuit-to-stream compilation and execution.

### Performance Optimization

**Current Performance** (100MB input, 10 primordial levels):
- **Throughput**: 1934-1997 MB/s
- **Optimization**: 6.66× speedup from hybrid strategy
  - Entropy-based fast path (skips detection for non-periodic data)
  - Sample-based detection (1MB sampling vs full input)
  - Early termination (stops on correlation > 0.8)

**Tuning Parameters** (optimal configuration):
- Primordial levels: 10
- Correlation threshold: 0.8
- Proven through instrumentation tests in `tests/period_chunking_tuning.rs`

**Future optimizations**:
- Cache commonly-detected periods
- Parallel autocorrelation for large inputs
- SIMD optimization for correlation computation

## Contributing

The processor crate provides the foundation for universal canonical compute. Contributions should maintain:

1. **No Manual Gauge Management** - Gauges must be constructed automatically
2. **Domain-Neutral Storage** - Memory pools must be content-agnostic
3. **Mediatype Pattern** - Domain heads must follow the mediatype paradigm
4. **Infinite Scalability** - No artificial limits or hardcoded bounds

## License

MIT OR Apache-2.0
