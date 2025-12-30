# Benchmarking Specification

**Status:** Approved
**Version:** 0.1.0
**Last Updated:** 2025-01-18

## Overview

This specification defines the comprehensive benchmarking strategy for Hologram. It covers performance measurement, profiling, regression detection, and continuous performance monitoring.

## Benchmarking Philosophy

**🎯 CRITICAL: Performance is a feature, not an optimization.**

Core principles:

1. **Measure Everything** - Benchmark all critical operations
2. **Continuous Monitoring** - Track performance over time
3. **Regression Detection** - Catch performance degradation early
4. **Reproducible** - Consistent results across runs
5. **Representative** - Benchmark realistic workloads
6. **Actionable** - Results guide optimization decisions

## Performance Targets

### Core Operations

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| **Buffer Allocation** | < 100 ns | O(1) allocation from pool |
| **Vector Add (256 elements)** | < 50 ns | SIMD-optimized CPU path |
| **Vector Add (1M elements)** | < 100 μs | Parallelized execution |
| **Matrix Multiply (32×32)** | < 5 μs | Cache-friendly tiling |
| **Matrix Multiply (1024×1024)** | < 50 ms | Multi-threaded BLAS |
| **Canonicalization** | < 1 μs | Pattern matching + rewriting |
| **Generator Compilation** | < 10 μs | Circuit → ISA translation |
| **Tensor Reshape** | < 10 ns | Zero-copy view creation |
| **FFI Call Overhead** | < 10 ns | Minimal boundary crossing |

### Backend-Specific Targets

**CPU Backend:**
- Vector operations: 42 ns (measured with SIMD)
- Memory bandwidth: > 80% theoretical peak
- Cache hit rate: > 95% for hot paths

**GPU Backends (CUDA/Metal/WebGPU):**
- Kernel launch overhead: < 20 μs
- Memory transfer (H→D): > 10 GB/s
- Compute utilization: > 70% theoretical peak
- Occupancy: > 75% for parallel kernels

## Benchmarking Tools

### Primary: Criterion.rs

**Why Criterion:**
- Statistical analysis (outlier detection, confidence intervals)
- HTML reports with graphs
- Comparison against baseline
- Regression detection
- Platform-agnostic

**Installation:**

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "core_benchmarks"
harness = false
```

### Secondary: perf (Linux profiling)

**Usage:**
```bash
# Record performance counters
perf record --call-graph dwarf cargo bench

# Analyze results
perf report

# Generate flamegraph
perf script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg
```

### Profiling: cargo-flamegraph

```bash
# Install
cargo install flamegraph

# Generate flamegraph
cargo flamegraph --bench core_benchmarks

# Open flamegraph.svg in browser
```

## Benchmark Organization

### Directory Structure

```
hologram/
├── benches/
│   ├── core_benchmarks.rs       # Core operations
│   ├── compiler_benchmarks.rs   # Canonicalization & compilation
│   ├── backend_benchmarks.rs    # Backend-specific benchmarks
│   ├── tensor_benchmarks.rs     # Tensor operations
│   └── e2e_benchmarks.rs        # End-to-end workflows
├── crates/
│   ├── core/
│   │   └── benches/
│   │       ├── buffer_bench.rs
│   │       ├── ops_bench.rs
│   │       └── tensor_bench.rs
│   ├── compiler/
│   │   └── benches/
│   │       ├── canonicalization_bench.rs
│   │       └── compilation_bench.rs
│   └── backends/
│       └── benches/
│           ├── cpu_bench.rs
│           ├── cuda_bench.rs
│           └── webgpu_bench.rs
└── target/
    └── criterion/              # Benchmark results
        ├── reports/            # HTML reports
        └── baseline/           # Baseline for comparison
```

## Micro-Benchmarks

### Buffer Operations

**File:** `crates/core/benches/buffer_bench.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use hologram_core::Executor;

fn benchmark_buffer_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_allocation");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let mut exec = Executor::new().unwrap();
                b.iter(|| {
                    exec.allocate::<f32>(size).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn benchmark_buffer_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_copy");

    for size in [256, 1024, 4096, 16384, 65536].iter() {
        group.throughput(Throughput::Bytes((*size * 4) as u64)); // f32 = 4 bytes

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let mut exec = Executor::new().unwrap();
                let mut buffer = exec.allocate::<f32>(size).unwrap();
                let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

                b.iter(|| {
                    buffer.copy_from_slice(&data).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_buffer_allocation, benchmark_buffer_copy);
criterion_main!(benches);
```

### Vector Operations

**File:** `crates/core/benches/ops_bench.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use hologram_core::{ops, Executor};

fn benchmark_vector_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_add");

    for size in [256, 1024, 4096, 16384, 65536, 262144, 1048576].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let mut exec = Executor::new().unwrap();
                let mut a = exec.allocate::<f32>(size).unwrap();
                let mut b = exec.allocate::<f32>(size).unwrap();
                let mut c = exec.allocate::<f32>(size).unwrap();

                let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                a.copy_from_slice(&data).unwrap();
                b.copy_from_slice(&data).unwrap();

                b.iter(|| {
                    ops::math::vector_add(&exec, &a, &b, &mut c, size).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn benchmark_vector_operations_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_ops_comparison");
    let size = 16384;

    group.throughput(Throughput::Elements(size as u64));

    let mut exec = Executor::new().unwrap();
    let mut a = exec.allocate::<f32>(size).unwrap();
    let mut b = exec.allocate::<f32>(size).unwrap();
    let mut c = exec.allocate::<f32>(size).unwrap();

    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    a.copy_from_slice(&data).unwrap();
    b.copy_from_slice(&data).unwrap();

    // Compare different operations
    group.bench_function("add", |b| {
        b.iter(|| ops::math::vector_add(&exec, &a, &b, &mut c, size).unwrap());
    });

    group.bench_function("mul", |b| {
        b.iter(|| ops::math::vector_mul(&exec, &a, &b, &mut c, size).unwrap());
    });

    group.bench_function("relu", |b| {
        b.iter(|| ops::math::relu(&exec, &a, &mut c, size).unwrap());
    });

    group.bench_function("sigmoid", |b| {
        b.iter(|| ops::activation::sigmoid(&exec, &a, &mut c, size).unwrap());
    });

    group.finish();
}

criterion_group!(benches, benchmark_vector_add, benchmark_vector_operations_comparison);
criterion_main!(benches);
```

### Tensor Operations

**File:** `crates/core/benches/tensor_bench.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use hologram_core::{Executor, Tensor};

fn benchmark_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_matmul");

    for size in [16, 32, 64, 128, 256].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let mut exec = Executor::new().unwrap();

                let buf_a = exec.allocate::<f32>(size * size).unwrap();
                let buf_b = exec.allocate::<f32>(size * size).unwrap();

                let tensor_a = Tensor::from_buffer(buf_a, vec![size, size]).unwrap();
                let tensor_b = Tensor::from_buffer(buf_b, vec![size, size]).unwrap();

                b.iter(|| {
                    tensor_a.matmul(&exec, &tensor_b).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn benchmark_tensor_views(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_views");
    let size = 1024;

    let mut exec = Executor::new().unwrap();
    let buf = exec.allocate::<f32>(size * size).unwrap();
    let tensor = Tensor::from_buffer(buf, vec![size, size]).unwrap();

    // Zero-copy operations should be extremely fast
    group.bench_function("select", |b| {
        b.iter(|| tensor.select(0, 42).unwrap());
    });

    group.bench_function("narrow", |b| {
        b.iter(|| tensor.narrow(1, 0, 512).unwrap());
    });

    group.bench_function("transpose", |b| {
        b.iter(|| tensor.transpose().unwrap());
    });

    group.finish();
}

criterion_group!(benches, benchmark_matmul, benchmark_tensor_views);
criterion_main!(benches);
```

## Compiler Benchmarks

### Canonicalization

**File:** `crates/compiler/benches/canonicalization_bench.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use hologram_compiler::Canonicalizer;

fn benchmark_canonicalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("canonicalization");

    let test_circuits = vec![
        ("simple", "H·H"),
        ("medium", "H·X·X·H·Z·Z"),
        ("complex", "H·X·H·X·H·Z·H·Z·H·X·H"),
        ("very_complex", "H·X·H·X·H·Z·H·Z·H·X·H·X·H·Z·H·Z·H·X·H"),
    ];

    for (name, circuit) in test_circuits.iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            circuit,
            |b, &circuit| {
                b.iter(|| {
                    Canonicalizer::canonicalize(circuit).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn benchmark_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_matching");

    // Benchmark individual pattern rules
    group.bench_function("H²→I", |b| {
        b.iter(|| {
            Canonicalizer::canonicalize("H·H").unwrap();
        });
    });

    group.bench_function("X²→I", |b| {
        b.iter(|| {
            Canonicalizer::canonicalize("X·X").unwrap();
        });
    });

    group.bench_function("HXH→Z", |b| {
        b.iter(|| {
            Canonicalizer::canonicalize("H·X·H").unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_canonicalization, benchmark_pattern_matching);
criterion_main!(benches);
```

## Backend Benchmarks

### CPU Backend

**File:** `crates/backends/benches/cpu_bench.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use hologram_backends::{Backend, CpuBackend, Program, LaunchConfig};

fn benchmark_cpu_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_backend");

    let sizes = [256, 1024, 4096, 16384];

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(&format!("vector_add_{}", size), |b| {
            let mut backend = CpuBackend::new().unwrap();

            let program = create_vector_add_program(size);
            let config = LaunchConfig {
                grid_size: (size / 256, 1, 1),
                block_size: (256, 1, 1),
            };

            b.iter(|| {
                backend.execute_program(&program, &config).unwrap();
            });
        });
    }

    group.finish();
}

fn benchmark_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_comparison");
    let size = 16384;

    group.throughput(Throughput::Elements(size as u64));

    // Compare SIMD and scalar implementations
    group.bench_function("simd", |b| {
        let mut backend = CpuBackend::with_simd(true).unwrap();
        let program = create_vector_add_program(size);
        let config = default_launch_config(size);

        b.iter(|| {
            backend.execute_program(&program, &config).unwrap();
        });
    });

    group.bench_function("scalar", |b| {
        let mut backend = CpuBackend::with_simd(false).unwrap();
        let program = create_vector_add_program(size);
        let config = default_launch_config(size);

        b.iter(|| {
            backend.execute_program(&program, &config).unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_cpu_execution, benchmark_simd_vs_scalar);
criterion_main!(benches);
```

## End-to-End Benchmarks

### Full Workflow Benchmarks

**File:** `benches/e2e_benchmarks.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use hologram_core::{ops, Executor, Tensor};

fn benchmark_neural_network_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_network");

    // Simulate a typical neural network layer: matmul + bias + relu
    group.bench_function("forward_pass_128x128", |b| {
        let mut exec = Executor::new().unwrap();

        // Input: 128 features
        let input_buf = exec.allocate::<f32>(128).unwrap();
        let input = Tensor::from_buffer(input_buf, vec![128]).unwrap();

        // Weights: 128x128
        let weights_buf = exec.allocate::<f32>(128 * 128).unwrap();
        let weights = Tensor::from_buffer(weights_buf, vec![128, 128]).unwrap();

        // Bias: 128
        let bias_buf = exec.allocate::<f32>(128).unwrap();
        let mut output_buf = exec.allocate::<f32>(128).unwrap();

        b.iter(|| {
            // matmul
            let matmul_result = input.matmul(&exec, &weights).unwrap();

            // add bias
            let matmul_buf = matmul_result.buffer();
            ops::math::vector_add(&exec, matmul_buf, &bias_buf, &mut output_buf, 128).unwrap();

            // relu
            ops::math::relu(&exec, &output_buf, &mut output_buf, 128).unwrap();
        });
    });

    group.finish();
}

fn benchmark_image_processing_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_processing");

    // Simulate image processing: resize + normalize + convolution
    group.bench_function("process_256x256_image", |b| {
        let mut exec = Executor::new().unwrap();
        let size = 256 * 256 * 3; // 256x256 RGB image

        let mut input = exec.allocate::<f32>(size).unwrap();
        let mut normalized = exec.allocate::<f32>(size).unwrap();
        let mut output = exec.allocate::<f32>(size).unwrap();

        b.iter(|| {
            // Normalize: scale to [0, 1]
            ops::math::vector_mul_scalar(&exec, &input, 1.0/255.0, &mut normalized, size).unwrap();

            // Apply filter (simplified convolution)
            ops::math::vector_mul(&exec, &normalized, &normalized, &mut output, size).unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_neural_network_layer, benchmark_image_processing_pipeline);
criterion_main!(benches);
```

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks
cargo bench --workspace

# Run specific benchmark
cargo bench --bench core_benchmarks

# Run benchmarks matching pattern
cargo bench vector_add

# Save baseline for comparison
cargo bench --bench core_benchmarks -- --save-baseline baseline_v1

# Compare against baseline
cargo bench --bench core_benchmarks -- --baseline baseline_v1

# Generate verbose output
cargo bench -- --verbose
```

### Criterion Configuration

**File:** `.cargo/config.toml`

```toml
[bench]
# Criterion configuration
sample-size = 100         # Number of samples per benchmark
measurement-time = 5      # Measurement time in seconds
warm-up-time = 3         # Warm-up time in seconds
confidence-level = 0.95  # Statistical confidence level
```

### HTML Reports

After running benchmarks:

```bash
# Open HTML report
open target/criterion/report/index.html

# View specific benchmark
open target/criterion/vector_add/report/index.html
```

## Regression Detection

### Automatic Regression Detection

```bash
# Run benchmarks and compare
cargo bench --bench core_benchmarks -- --baseline main

# Fail CI if regression detected (> 10% slower)
cargo bench --bench core_benchmarks -- \
    --baseline main \
    --significance-level 0.05 \
    --noise-threshold 0.10
```

### CI Integration

**GitHub Actions** (see [ci.md](ci.md)):

```yaml
name: Benchmark

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run benchmarks
        run: cargo bench --workspace

      - name: Store benchmark results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'cargo'
          output-file-path: target/criterion/benchmarks.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          alert-threshold: '110%'  # Alert if 10% slower
          fail-on-alert: true      # Fail PR if regression detected
```

## Profiling

### CPU Profiling with perf

```bash
# Record performance data
perf record --call-graph dwarf cargo bench --bench core_benchmarks

# Analyze results
perf report

# Top functions
perf report --sort=symbol

# Generate flamegraph
perf script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg
```

### cargo-flamegraph

```bash
# Install
cargo install flamegraph

# Generate flamegraph for benchmark
cargo flamegraph --bench core_benchmarks -- --bench vector_add

# Open flamegraph.svg
```

### Valgrind Profiling

```bash
# Install valgrind
sudo apt install valgrind

# Cache profiling
valgrind --tool=cachegrind cargo bench --bench core_benchmarks

# Analyze cache misses
cg_annotate cachegrind.out.<pid>

# Heap profiling
valgrind --tool=massif cargo bench --bench core_benchmarks
ms_print massif.out.<pid>
```

## Performance Comparison

### Comparing Against Baselines

```bash
# Create baseline from main branch
git checkout main
cargo bench --workspace -- --save-baseline main

# Checkout feature branch
git checkout feature-branch

# Compare against baseline
cargo bench --workspace -- --baseline main

# View comparison
open target/criterion/report/index.html
```

### Comparing Against External Libraries

**Example:** Compare hologram vector_add vs ndarray

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use hologram_core::{ops, Executor};
use ndarray::Array1;

fn benchmark_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_ndarray");
    let size = 16384;

    // Hologram
    group.bench_function("hologram", |b| {
        let mut exec = Executor::new().unwrap();
        let mut a = exec.allocate::<f32>(size).unwrap();
        let mut b = exec.allocate::<f32>(size).unwrap();
        let mut c = exec.allocate::<f32>(size).unwrap();

        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        a.copy_from_slice(&data).unwrap();
        b.copy_from_slice(&data).unwrap();

        b.iter(|| {
            ops::math::vector_add(&exec, &a, &b, &mut c, size).unwrap();
        });
    });

    // ndarray
    group.bench_function("ndarray", |b| {
        let a = Array1::from_vec((0..size).map(|i| i as f32).collect());
        let b = Array1::from_vec((0..size).map(|i| i as f32).collect());

        b.iter(|| {
            let _c = &a + &b;
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_comparison);
criterion_main!(benches);
```

## Performance Metrics

### Tracked Metrics

| Metric | Tool | Target |
|--------|------|--------|
| **Latency** | Criterion | < target values (see Performance Targets) |
| **Throughput** | Criterion | > X ops/sec |
| **Memory Bandwidth** | perf | > 80% theoretical peak |
| **Cache Hit Rate** | cachegrind | > 95% L1, > 85% L2 |
| **CPU Utilization** | perf | > 70% for parallel ops |
| **GPU Utilization** | nvprof/Metal | > 70% compute utilization |
| **Compilation Time** | criterion | < 10 μs circuit→ISA |

### Reporting

Generate performance report:

```bash
# Run all benchmarks and generate report
cargo bench --workspace

# Extract key metrics
python scripts/extract_benchmark_metrics.py \
    target/criterion \
    > docs/performance/latest_results.md
```

## Best Practices

### 1. Benchmark Realistic Workloads

```rust
// ✅ Good: Benchmark realistic size
cargo bench -- vector_add_16384

// ❌ Bad: Benchmark tiny size that fits in cache
cargo bench -- vector_add_16
```

### 2. Use Proper Setup/Teardown

```rust
// ✅ Good: Setup outside measurement
group.bench_function("operation", |b| {
    let mut exec = Executor::new().unwrap();  // Setup
    let buffer = exec.allocate::<f32>(size).unwrap();

    b.iter(|| {
        // Only measure the operation
        operation(&buffer);
    });
});
```

### 3. Avoid Dead Code Elimination

```rust
// ✅ Good: Use black_box to prevent optimization
use criterion::black_box;

b.iter(|| {
    black_box(compute_value());
});

// ❌ Bad: Compiler might optimize away
b.iter(|| {
    compute_value();  // Result unused, might be eliminated
});
```

### 4. Control for External Factors

- Run on idle machine
- Disable CPU frequency scaling
- Close background applications
- Use consistent power settings
- Run multiple times and average

## Troubleshooting

### Noisy Benchmarks

**Problem:** High variance in results

**Solutions:**
```bash
# Increase sample size
cargo bench -- --sample-size 500

# Increase measurement time
cargo bench -- --measurement-time 10

# Check for background processes
top  # Kill unnecessary processes
```

### Unexpected Slowdowns

**Problem:** Benchmark slower than expected

**Solutions:**
```bash
# Profile to find bottleneck
cargo flamegraph --bench core_benchmarks

# Check for allocations
valgrind --tool=massif cargo bench

# Verify SIMD usage
objdump -d target/release/deps/<benchmark> | grep -i "vmov\|vadd"
```

## References

- [Criterion.rs User Guide](https://bheisler.github.io/criterion.rs/book/)
- [The Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Linux perf Tutorial](https://perf.wiki.kernel.org/index.php/Tutorial)
- [Flamegraph Documentation](https://www.brendangregg.com/flamegraphs.html)
- [Testing Specification](testing.md)
- [CI Specification](ci.md)
