//! Period Chunking Benchmarks
//!
//! Benchmarks period detection and chunking strategies.
//!
//! Run with: cargo bench --package processor --bench period_chunking

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hologram_memory_manager::chunking::PeriodDrivenChunker;

/// Benchmark period detection vs fast path chunking
fn benchmark_chunking_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking_strategies");

    let sizes = vec![
        (1024, "1KB"),
        (10 * 1024, "10KB"),
        (100 * 1024, "100KB"),
        (1024 * 1024, "1MB"),
        (10 * 1024 * 1024, "10MB"),
    ];

    for (size, label) in sizes {
        group.throughput(Throughput::Bytes(size as u64));

        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        // Fast path (no detection)
        group.bench_with_input(BenchmarkId::new("fast_path", label), &data, |b, data| {
            let chunker = PeriodDrivenChunker::new_fast(10);
            b.iter(|| {
                let data_clone = data.clone();
                let result = chunker.chunk_fast(data_clone).unwrap();
                black_box(result);
            });
        });

        // Detection path (with period detection)
        group.bench_with_input(BenchmarkId::new("detection_path", label), &data, |b, data| {
            let chunker = PeriodDrivenChunker::new(10);
            b.iter(|| {
                let result = chunker.chunk_and_construct_gauges(data).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark different primorial levels
fn benchmark_primorial_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("primorial_levels");

    let size = 1024 * 1024; // 1MB
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    group.throughput(Throughput::Bytes(size as u64));

    for levels in [5, 7, 10, 12, 15] {
        group.bench_with_input(BenchmarkId::new("level", levels), &data, |b, data| {
            let chunker = PeriodDrivenChunker::new_fast(levels);
            b.iter(|| {
                let data_clone = data.clone();
                let result = chunker.chunk_fast(data_clone).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark different data patterns
fn benchmark_data_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_patterns");

    let size = 1024 * 1024; // 1MB
    group.throughput(Throughput::Bytes(size as u64));

    // Random-like pattern
    let random_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    // Periodic pattern (period = 16)
    let periodic_data: Vec<u8> = (0..size).map(|i| (i % 16) as u8).collect();

    // Constant pattern
    let constant_data: Vec<u8> = vec![42u8; size];

    let chunker = PeriodDrivenChunker::new(10);

    group.bench_function("random_pattern", |b| {
        b.iter(|| {
            let result = chunker.chunk_and_construct_gauges(&random_data).unwrap();
            black_box(result);
        });
    });

    group.bench_function("periodic_pattern", |b| {
        b.iter(|| {
            let result = chunker.chunk_and_construct_gauges(&periodic_data).unwrap();
            black_box(result);
        });
    });

    group.bench_function("constant_pattern", |b| {
        b.iter(|| {
            let result = chunker.chunk_and_construct_gauges(&constant_data).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_chunking_strategies,
    benchmark_primorial_levels,
    benchmark_data_patterns
);
criterion_main!(benches);
