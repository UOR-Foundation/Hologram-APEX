//! Domain Head Performance Benchmarks
//!
//! Benchmarks domain head extraction operations to identify bottlenecks.
//!
//! Run with: cargo bench --package processor --bench domain_head_performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hologram_memory_manager::{DomainHead, RawDomainHead, UniversalMemoryPool};

/// Benchmark raw domain head extraction across different data sizes
fn benchmark_raw_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_extraction");

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

        group.bench_with_input(BenchmarkId::from_parameter(label), &data, |b, data| {
            let mut pool = UniversalMemoryPool::new();
            pool.embed(data.clone(), 10).unwrap();
            let head = RawDomainHead;

            b.iter(|| {
                let result = head.extract(&pool).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark different raw extraction methods
fn benchmark_extraction_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("extraction_methods");

    let size = 1024 * 1024; // 1MB
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    let mut pool = UniversalMemoryPool::new();
    pool.embed(data.clone(), 10).unwrap();

    group.throughput(Throughput::Bytes(size as u64));

    // Method 1: extend_from_slice
    group.bench_function("extend_from_slice", |b| {
        b.iter(|| {
            let mut result = Vec::new();
            for block in pool.blocks() {
                if let Some(data) = block.data() {
                    result.extend_from_slice(data);
                }
            }
            black_box(result);
        });
    });

    // Method 2: Pre-allocated Vec
    group.bench_function("pre_allocated", |b| {
        b.iter(|| {
            let total_size: usize = pool.blocks().iter().map(|b| b.len()).sum();
            let mut result = Vec::with_capacity(total_size);
            for block in pool.blocks() {
                if let Some(data) = block.data() {
                    result.extend_from_slice(data);
                }
            }
            black_box(result);
        });
    });

    // Method 3: Iterator chain
    group.bench_function("iterator_chain", |b| {
        b.iter(|| {
            let result: Vec<u8> = pool
                .blocks()
                .iter()
                .filter_map(|b| b.data())
                .flat_map(|data| data.iter().copied())
                .collect();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark different block access patterns
fn benchmark_block_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_access_patterns");

    let size = 1024 * 1024; // 1MB
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    let mut pool = UniversalMemoryPool::new();
    pool.embed(data.clone(), 10).unwrap();

    group.throughput(Throughput::Bytes(size as u64));

    // Pattern 1: Direct block iteration
    group.bench_function("direct_iteration", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for block in pool.blocks() {
                if let Some(data) = block.data() {
                    for &byte in data {
                        sum += byte as u64;
                    }
                }
            }
            black_box(sum);
        });
    });

    // Pattern 2: Iterator chain
    group.bench_function("iterator_chain", |b| {
        b.iter(|| {
            let sum: u64 = pool
                .blocks()
                .iter()
                .filter_map(|b| b.data())
                .flat_map(|data| data.iter())
                .map(|&b| b as u64)
                .sum();
            black_box(sum);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_raw_extraction,
    benchmark_extraction_methods,
    benchmark_block_access_patterns
);
criterion_main!(benches);
