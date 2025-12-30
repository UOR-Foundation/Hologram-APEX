//! Routing Protocol Performance Benchmark
//!
//! Measures O(1) routing operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hrm_spec::prelude::*;

fn benchmark_routing(c: &mut Criterion) {
    let routing = StandardRouting;
    
    c.bench_function("routing_addition", |bencher| {
        let a = TorusCoordinate { page: 3, resonance: 5 };
        let b = TorusCoordinate { page: 7, resonance: 11 };
        bencher.iter(|| {
            let result = routing.route_addition(black_box(&a), black_box(&b));
            black_box(result);
        });
    });
    
    c.bench_function("routing_multiplication", |bencher| {
        let a = TorusCoordinate { page: 3, resonance: 5 };
        let b = TorusCoordinate { page: 7, resonance: 11 };
        bencher.iter(|| {
            let result = routing.route_multiplication(black_box(&a), black_box(&b));
            black_box(result);
        });
    });
    
    c.bench_function("coherence_verification", |bencher| {
        let verifier = CoherenceVerifier::new();
        let a = BigInt::from(123);
        let b = BigInt::from(456);
        bencher.iter(|| {
            let add_coherent = verifier.verify_addition_coherence(black_box(&a), black_box(&b));
            let mul_coherent = verifier.verify_multiplication_coherence(black_box(&a), black_box(&b));
            black_box((add_coherent, mul_coherent));
        });
    });
    
    c.bench_function("routing_batch_operations", |bencher| {
        let coords: Vec<TorusCoordinate> = (0..100)
            .map(|i| TorusCoordinate {
                page: (i % 48) as u8,
                resonance: (i % 96) as u8,
            })
            .collect();
        
        bencher.iter(|| {
            let mut result = TorusCoordinate::zero();
            for coord in &coords {
                result = routing.route_addition(black_box(&result), black_box(coord));
            }
            black_box(result);
        });
    });
}

criterion_group!(benches, benchmark_routing);
criterion_main!(benches);
