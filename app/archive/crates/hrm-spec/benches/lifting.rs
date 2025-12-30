//! Lifting Performance Benchmark
//!
//! Measures performance of T² → ℤ lifting.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hrm_spec::prelude::*;

fn benchmark_lifting(c: &mut Criterion) {
    let lifting = O1Lifting;
    
    c.bench_function("lifting_small", |b| {
        let coord = TorusCoordinate { page: 5, resonance: 10 };
        let hint = BigInt::from(123);
        b.iter(|| {
            let result = lifting.lift(black_box(&coord), black_box(&hint));
            black_box(result);
        });
    });
    
    c.bench_function("lifting_large_hint", |b| {
        let coord = TorusCoordinate { page: 23, resonance: 47 };
        let hint = BigInt::parse_bytes(b"123456789012345678901234567890", 10).unwrap();
        b.iter(|| {
            let result = lifting.lift(black_box(&coord), black_box(&hint));
            black_box(result);
        });
    });
    
    c.bench_function("lifting_batch_1000", |b| {
        let coords: Vec<TorusCoordinate> = (0..1000)
            .map(|i| TorusCoordinate {
                page: (i % 48) as u8,
                resonance: (i % 96) as u8,
            })
            .collect();
        let hint = BigInt::from(1000000);
        
        b.iter(|| {
            for coord in &coords {
                let result = lifting.lift(black_box(coord), black_box(&hint));
                black_box(result);
            }
        });
    });
}

criterion_group!(benches, benchmark_lifting);
criterion_main!(benches);
