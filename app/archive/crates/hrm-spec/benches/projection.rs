//! Projection Performance Benchmark
//!
//! Measures performance of ℤ → T² projection.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hrm_spec::prelude::*;

fn benchmark_projection(c: &mut Criterion) {
    c.bench_function("projection_small", |b| {
        let n = BigInt::from(12345);
        b.iter(|| {
            let coord = StandardProjection.project(black_box(&n));
            black_box(coord);
        });
    });
    
    c.bench_function("projection_large", |b| {
        let n = BigInt::parse_bytes(b"123456789012345678901234567890", 10).unwrap();
        b.iter(|| {
            let coord = StandardProjection.project(black_box(&n));
            black_box(coord);
        });
    });
    
    c.bench_function("projection_batch_1000", |b| {
        let numbers: Vec<BigInt> = (0..1000).map(BigInt::from).collect();
        b.iter(|| {
            for n in &numbers {
                let coord = StandardProjection.project(black_box(n));
                black_box(coord);
            }
        });
    });
}

criterion_group!(benches, benchmark_projection);
criterion_main!(benches);
