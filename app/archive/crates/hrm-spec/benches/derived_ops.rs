//! Derived Operations Performance Benchmark
//!
//! Measures performance of MatMul, Conv, Attention.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hrm_spec::prelude::*;
use hrm_spec::derived::matmul::Matrix;

fn benchmark_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    
    for size in [2, 4, 8, 16].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let matrix: Matrix = vec![
                vec![TorusCoordinate { page: 1, resonance: 2 }; size];
                size
            ];
            b.iter(|| {
                let result = matmul(black_box(&matrix), black_box(&matrix)).unwrap();
                black_box(result);
            });
        });
    }
    
    group.finish();
}

fn benchmark_convolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolution");
    
    for signal_len in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("linear", signal_len),
            signal_len,
            |b, &len| {
                let signal = vec![TorusCoordinate { page: 1, resonance: 2 }; len];
                let kernel = vec![TorusCoordinate { page: 3, resonance: 5 }; 5];
                b.iter(|| {
                    let result = convolve(black_box(&signal), black_box(&kernel));
                    black_box(result);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("circular", signal_len),
            signal_len,
            |b, &len| {
                let signal = vec![TorusCoordinate { page: 1, resonance: 2 }; len];
                let kernel = vec![TorusCoordinate { page: 3, resonance: 5 }; 5];
                b.iter(|| {
                    let result = circular_convolve(black_box(&signal), black_box(&kernel));
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction");
    
    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("sum", size), size, |b, &size| {
            let coords = vec![TorusCoordinate { page: 1, resonance: 2 }; size];
            b.iter(|| {
                let result = reduce_sum(black_box(&coords));
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("product", size), size, |b, &size| {
            let coords = vec![TorusCoordinate { page: 2, resonance: 3 }; size];
            b.iter(|| {
                let result = reduce_product(black_box(&coords));
                black_box(result);
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_matmul, benchmark_convolution, benchmark_reduction);
criterion_main!(benches);
