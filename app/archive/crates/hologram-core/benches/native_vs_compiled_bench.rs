//! Native CPU vs Compiled Canonical Kernels Benchmark
//!
//! Tests the COMPILED canonical kernel path (inline SIMD kernels with AVX512/AVX2/SSE4.1).
//! These are the fused canonical operations generated from MoonshineHRM.
//!
//! ## Architecture
//!
//! - **Native**: Simple Rust loops (baseline)
//! - **Compiled Canonical**: ops::math functions → inline SIMD kernels (AVX512/AVX2/SSE4.1)
//!
//! ## Expected Results
//!
//! Compiled kernels should be 1.9-7.3× faster than native loops due to SIMD acceleration.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hologram_core::{ops, Executor};

// ============================================================================
// Native CPU Implementations (Baseline)
// ============================================================================

#[inline(never)]
fn native_vector_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] + b[i];
    }
}

#[inline(never)]
fn native_vector_sub(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] - b[i];
    }
}

#[inline(never)]
fn native_vector_mul(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] * b[i];
    }
}

#[inline(never)]
fn native_vector_div(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] / b[i];
    }
}

// ============================================================================
// Binary Operations: Native vs Compiled Canonical
// ============================================================================

fn benchmark_binary_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_ops");

    // Test sizes: 256 (tiny), 4K (small), 64K (medium), 262K (large - at SIMD threshold)
    for size in [256, 4_096, 65_536, 262_144].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // ========================================================================
        // Vector Add: a + b
        // ========================================================================

        group.bench_with_input(BenchmarkId::new("add/native", size), size, |bencher, &size| {
            let a = vec![1.0f32; size];
            let b = vec![2.0f32; size];
            let mut c = vec![0.0f32; size];

            bencher.iter(|| {
                native_vector_add(black_box(&a), black_box(&b), black_box(&mut c));
            });
        });

        group.bench_with_input(BenchmarkId::new("add/compiled", size), size, |bencher, &size| {
            let mut exec = Executor::new().unwrap();
            let a = exec.allocate::<f32>(size).unwrap();
            let b = exec.allocate::<f32>(size).unwrap();
            let mut c = exec.allocate::<f32>(size).unwrap();

            bencher.iter(|| {
                ops::math::vector_add(&mut exec, &a, &b, &mut c, size).unwrap();
            });
        });

        // ========================================================================
        // Vector Sub: a - b
        // ========================================================================

        group.bench_with_input(BenchmarkId::new("sub/native", size), size, |bencher, &size| {
            let a = vec![3.0f32; size];
            let b = vec![1.0f32; size];
            let mut c = vec![0.0f32; size];

            bencher.iter(|| {
                native_vector_sub(black_box(&a), black_box(&b), black_box(&mut c));
            });
        });

        group.bench_with_input(BenchmarkId::new("sub/compiled", size), size, |bencher, &size| {
            let mut exec = Executor::new().unwrap();
            let a = exec.allocate::<f32>(size).unwrap();
            let b = exec.allocate::<f32>(size).unwrap();
            let mut c = exec.allocate::<f32>(size).unwrap();

            bencher.iter(|| {
                ops::math::vector_sub(&mut exec, &a, &b, &mut c, size).unwrap();
            });
        });

        // ========================================================================
        // Vector Mul: a * b
        // ========================================================================

        group.bench_with_input(BenchmarkId::new("mul/native", size), size, |bencher, &size| {
            let a = vec![2.0f32; size];
            let b = vec![3.0f32; size];
            let mut c = vec![0.0f32; size];

            bencher.iter(|| {
                native_vector_mul(black_box(&a), black_box(&b), black_box(&mut c));
            });
        });

        group.bench_with_input(BenchmarkId::new("mul/compiled", size), size, |bencher, &size| {
            let mut exec = Executor::new().unwrap();
            let a = exec.allocate::<f32>(size).unwrap();
            let b = exec.allocate::<f32>(size).unwrap();
            let mut c = exec.allocate::<f32>(size).unwrap();

            bencher.iter(|| {
                ops::math::vector_mul(&mut exec, &a, &b, &mut c, size).unwrap();
            });
        });

        // ========================================================================
        // Vector Div: a / b
        // ========================================================================

        group.bench_with_input(BenchmarkId::new("div/native", size), size, |bencher, &size| {
            let a = vec![6.0f32; size];
            let b = vec![2.0f32; size];
            let mut c = vec![0.0f32; size];

            bencher.iter(|| {
                native_vector_div(black_box(&a), black_box(&b), black_box(&mut c));
            });
        });

        group.bench_with_input(BenchmarkId::new("div/compiled", size), size, |bencher, &size| {
            let mut exec = Executor::new().unwrap();
            let a = exec.allocate::<f32>(size).unwrap();
            let b = exec.allocate::<f32>(size).unwrap();
            let mut c = exec.allocate::<f32>(size).unwrap();

            bencher.iter(|| {
                ops::math::vector_div(&mut exec, &a, &b, &mut c, size).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_binary_ops);
criterion_main!(benches);
