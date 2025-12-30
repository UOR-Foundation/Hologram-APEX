//! CPU Baseline Benchmarks for WebGPU Performance Comparison
//!
//! Measures scalar WASM CPU performance to establish baseline for GPU speedup calculations.
//! These benchmarks run natively with Criterion for accurate timing.
//!
//! To benchmark GPU performance, use: `wasm-pack test --chrome` with webgpu_performance_test.rs

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Scalar CPU implementation of vector addition
fn cpu_vector_add(a: &[f32], b: &[f32], output: &mut [f32]) {
    for i in 0..a.len() {
        output[i] = a[i] + b[i];
    }
}

/// Scalar CPU implementation of vector multiplication
fn cpu_vector_mul(a: &[f32], b: &[f32], output: &mut [f32]) {
    for i in 0..a.len() {
        output[i] = a[i] * b[i];
    }
}

/// Scalar CPU implementation of vector min
fn cpu_vector_min(a: &[f32], b: &[f32], output: &mut [f32]) {
    for i in 0..a.len() {
        output[i] = a[i].min(b[i]);
    }
}

/// Scalar CPU implementation of vector max
fn cpu_vector_max(a: &[f32], b: &[f32], output: &mut [f32]) {
    for i in 0..a.len() {
        output[i] = a[i].max(b[i]);
    }
}

/// Scalar CPU implementation of vector subtraction
fn cpu_vector_sub(a: &[f32], b: &[f32], output: &mut [f32]) {
    for i in 0..a.len() {
        output[i] = a[i] - b[i];
    }
}

/// Scalar CPU implementation of vector division
fn cpu_vector_div(a: &[f32], b: &[f32], output: &mut [f32]) {
    for i in 0..a.len() {
        output[i] = a[i] / b[i];
    }
}

/// Scalar CPU implementation of vector abs
fn cpu_vector_abs(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].abs();
    }
}

/// Scalar CPU implementation of vector exp
fn cpu_vector_exp(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].exp();
    }
}

/// Scalar CPU implementation of vector log
fn cpu_vector_log(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].ln();
    }
}

/// Scalar CPU implementation of vector sqrt
fn cpu_vector_sqrt(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].sqrt();
    }
}

/// Scalar CPU implementation of vector sigmoid
fn cpu_vector_sigmoid(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = 1.0 / (1.0 + (-input[i]).exp());
    }
}

/// Scalar CPU implementation of vector tanh
fn cpu_vector_tanh(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].tanh();
    }
}

/// Scalar CPU implementation of reduction sum
fn cpu_reduce_sum(input: &[f32]) -> f32 {
    input.iter().sum()
}

/// Benchmark binary operations across various sizes
fn benchmark_binary_ops_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_binary_ops");

    // Test sizes: 100, 1K, 10K, 100K, 1M (as specified in Phase 4 requirements)
    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Vector Add
        group.bench_with_input(BenchmarkId::new("vector_add", size), size, |bencher, &size| {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                cpu_vector_add(black_box(&a), black_box(&b), black_box(&mut output));
            });
        });

        // Vector Mul
        group.bench_with_input(BenchmarkId::new("vector_mul", size), size, |bencher, &size| {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                cpu_vector_mul(black_box(&a), black_box(&b), black_box(&mut output));
            });
        });

        // Vector Min
        group.bench_with_input(BenchmarkId::new("vector_min", size), size, |bencher, &size| {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                cpu_vector_min(black_box(&a), black_box(&b), black_box(&mut output));
            });
        });

        // Vector Max
        group.bench_with_input(BenchmarkId::new("vector_max", size), size, |bencher, &size| {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                cpu_vector_max(black_box(&a), black_box(&b), black_box(&mut output));
            });
        });

        // Vector Sub
        group.bench_with_input(BenchmarkId::new("vector_sub", size), size, |bencher, &size| {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                cpu_vector_sub(black_box(&a), black_box(&b), black_box(&mut output));
            });
        });

        // Vector Div
        group.bench_with_input(BenchmarkId::new("vector_div", size), size, |bencher, &size| {
            let a: Vec<f32> = (0..size).map(|i| i as f32 + 1.0).collect();
            let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32 + 1.0).collect();
            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                cpu_vector_div(black_box(&a), black_box(&b), black_box(&mut output));
            });
        });
    }

    group.finish();
}

/// Benchmark unary operations across various sizes
fn benchmark_unary_ops_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_unary_ops");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Vector Abs
        group.bench_with_input(BenchmarkId::new("vector_abs", size), size, |bencher, &size| {
            let input: Vec<f32> = (0..size).map(|i| i as f32 - (size / 2) as f32).collect();
            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                cpu_vector_abs(black_box(&input), black_box(&mut output));
            });
        });

        // Vector Exp
        group.bench_with_input(BenchmarkId::new("vector_exp", size), size, |bencher, &size| {
            let input: Vec<f32> = (0..size).map(|i| (i % 10) as f32).collect();
            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                cpu_vector_exp(black_box(&input), black_box(&mut output));
            });
        });

        // Vector Log
        group.bench_with_input(BenchmarkId::new("vector_log", size), size, |bencher, &size| {
            let input: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                cpu_vector_log(black_box(&input), black_box(&mut output));
            });
        });

        // Vector Sqrt
        group.bench_with_input(BenchmarkId::new("vector_sqrt", size), size, |bencher, &size| {
            let input: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                cpu_vector_sqrt(black_box(&input), black_box(&mut output));
            });
        });

        // Vector Sigmoid
        group.bench_with_input(BenchmarkId::new("vector_sigmoid", size), size, |bencher, &size| {
            let input: Vec<f32> = (0..size).map(|i| (i % 10) as f32 - 5.0).collect();
            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                cpu_vector_sigmoid(black_box(&input), black_box(&mut output));
            });
        });

        // Vector Tanh
        group.bench_with_input(BenchmarkId::new("vector_tanh", size), size, |bencher, &size| {
            let input: Vec<f32> = (0..size).map(|i| (i % 10) as f32 - 5.0).collect();
            let mut output = vec![0.0f32; size];

            bencher.iter(|| {
                cpu_vector_tanh(black_box(&input), black_box(&mut output));
            });
        });
    }

    group.finish();
}

/// Benchmark reduction operations across various sizes
fn benchmark_reduce_ops_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_reduce_ops");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Reduce Sum
        group.bench_with_input(BenchmarkId::new("reduce_sum", size), size, |bencher, &size| {
            let input: Vec<f32> = (0..size).map(|i| i as f32).collect();

            bencher.iter(|| {
                black_box(cpu_reduce_sum(black_box(&input)));
            });
        });
    }

    group.finish();
}

/// Benchmark dispatch threshold crossover points
///
/// This tests operations specifically at the default dispatch thresholds
/// to validate the heuristics: binary=1024, unary=1024, reduction=512
fn benchmark_dispatch_thresholds(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_dispatch_thresholds");

    // Test sizes around dispatch thresholds
    let threshold_sizes = [
        ("below_binary", 512),     // Below binary threshold (1024)
        ("at_binary", 1024),       // At binary threshold
        ("above_binary", 2048),    // Above binary threshold
        ("below_reduction", 256),  // Below reduction threshold (512)
        ("at_reduction", 512),     // At reduction threshold
        ("above_reduction", 1024), // Above reduction threshold
    ];

    for (label, size) in threshold_sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Binary operation (vector_add)
        group.bench_with_input(
            BenchmarkId::new(format!("vector_add_{}", label), size),
            size,
            |bencher, &size| {
                let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
                let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
                let mut output = vec![0.0f32; size];

                bencher.iter(|| {
                    cpu_vector_add(black_box(&a), black_box(&b), black_box(&mut output));
                });
            },
        );

        // Unary operation (vector_abs)
        group.bench_with_input(
            BenchmarkId::new(format!("vector_abs_{}", label), size),
            size,
            |bencher, &size| {
                let input: Vec<f32> = (0..size).map(|i| i as f32 - (size / 2) as f32).collect();
                let mut output = vec![0.0f32; size];

                bencher.iter(|| {
                    cpu_vector_abs(black_box(&input), black_box(&mut output));
                });
            },
        );

        // Reduction operation (reduce_sum)
        group.bench_with_input(
            BenchmarkId::new(format!("reduce_sum_{}", label), size),
            size,
            |bencher, &size| {
                let input: Vec<f32> = (0..size).map(|i| i as f32).collect();

                bencher.iter(|| {
                    black_box(cpu_reduce_sum(black_box(&input)));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_binary_ops_baseline,
    benchmark_unary_ops_baseline,
    benchmark_reduce_ops_baseline,
    benchmark_dispatch_thresholds
);
criterion_main!(benches);
