//! SIMD Kernel-Only Benchmark
//!
//! This benchmark isolates the SIMD kernel performance by comparing:
//! 1. Native scalar loop
//! 2. Direct SIMD kernel call (no executor overhead)
//! 3. Full ops::math path (with executor overhead)
//!
//! This helps identify where overhead comes from.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hologram_core::{ops, Executor};

// Native scalar implementation (baseline)
fn native_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] + b[i];
    }
}

// Direct SIMD kernel call (no executor overhead)
unsafe fn direct_simd_add(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    hologram_core::kernel::inline::vector_add(a, b, c, n);
}

fn benchmark_add_isolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_isolation");

    for size in [256, 4_096, 65_536, 262_144].iter() {
        // 1. Native scalar loop (baseline)
        group.bench_with_input(BenchmarkId::new("native", size), size, |bencher, &size| {
            let a = vec![1.0f32; size];
            let b = vec![2.0f32; size];
            let mut c = vec![0.0f32; size];
            bencher.iter(|| {
                native_add(black_box(&a), black_box(&b), black_box(&mut c));
            });
        });

        // 2. Direct SIMD kernel call (no executor overhead)
        group.bench_with_input(BenchmarkId::new("simd_only", size), size, |bencher, &size| {
            let a = vec![1.0f32; size];
            let b = vec![2.0f32; size];
            let mut c = vec![0.0f32; size];
            bencher.iter(|| unsafe {
                direct_simd_add(
                    black_box(a.as_ptr()),
                    black_box(b.as_ptr()),
                    black_box(c.as_mut_ptr()),
                    black_box(size),
                );
            });
        });

        // 3. Full ops::math path (with executor overhead)
        group.bench_with_input(BenchmarkId::new("ops_math", size), size, |bencher, &size| {
            let mut exec = Executor::new().unwrap();
            let a = exec.allocate::<f32>(size).unwrap();
            let b = exec.allocate::<f32>(size).unwrap();
            let mut c = exec.allocate::<f32>(size).unwrap();
            bencher.iter(|| {
                ops::math::vector_add(&mut exec, &a, &b, &mut c, size).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_add_isolation);
criterion_main!(benches);
