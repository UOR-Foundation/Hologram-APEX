//! Streaming Performance Benchmarks
//!
//! Tests the processor's ability to handle large streaming inputs efficiently.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hologram_memory_manager::UniversalMemoryPool;

fn benchmark_streaming_large_inputs(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_large_inputs");

    // Test various large input sizes
    let sizes = vec![
        (1024 * 1024, "1MB"),
        (10 * 1024 * 1024, "10MB"),
        (50 * 1024 * 1024, "50MB"),
        (100 * 1024 * 1024, "100MB"),
    ];

    for (size, label) in sizes {
        group.throughput(Throughput::Bytes(size as u64));
        group.sample_size(10); // Reduce samples for large inputs

        group.bench_with_input(BenchmarkId::from_parameter(label), &size, |b, &size| {
            let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

            b.iter(|| {
                let mut pool = UniversalMemoryPool::new();
                let result = pool.embed(data.clone(), 10).unwrap();
                black_box(&result);
            });
        });
    }

    group.finish();
}

fn benchmark_streaming_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_throughput");

    // Measure throughput for sustained streaming (1MB input)
    let size = 1024 * 1024;
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    group.throughput(Throughput::Bytes(size as u64));

    group.bench_function("embed_1MB", |b| {
        b.iter(|| {
            let mut pool = UniversalMemoryPool::new();
            let result = pool.embed(data.clone(), 10).unwrap();
            black_box(&result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_streaming_large_inputs,
    benchmark_streaming_throughput
);
criterion_main!(benches);
