//! Embedding Performance Benchmarks
//!
//! Benchmarks the embedding pipeline to identify bottlenecks:
//! - Primordial sequence generation
//! - Chunking operations
//! - Gauge construction
//! - Block creation
//!
//! Run with: cargo bench --package processor --bench embedding_performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hologram_memory_manager::chunking::PeriodDrivenChunker;
use hologram_memory_manager::UniversalMemoryPool;

/// Benchmark overall embedding performance
fn benchmark_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding");

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
            b.iter(|| {
                let mut pool = UniversalMemoryPool::new();
                let result = pool.embed(data.clone(), 10).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark chunking operations
fn benchmark_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking");

    let size = 1024 * 1024; // 1MB
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    group.throughput(Throughput::Bytes(size as u64));

    // Benchmark chunk_fast (zero-copy Arc)
    group.bench_function("chunk_fast", |b| {
        let chunker = PeriodDrivenChunker::new_fast(10);
        b.iter(|| {
            let data_clone = data.clone();
            let result = chunker.chunk_fast(data_clone).unwrap();
            black_box(result);
        });
    });

    // Benchmark chunk_and_construct_gauges (with detection)
    group.bench_function("chunk_and_construct_gauges", |b| {
        let chunker = PeriodDrivenChunker::new(10);
        b.iter(|| {
            let result = chunker.chunk_and_construct_gauges(&data).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark fast path vs detection path
fn benchmark_fast_vs_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("fast_vs_detection");

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

        // Benchmark fast path (default embed with const gauges)
        group.bench_with_input(BenchmarkId::new("fast_path", label), &data, |b, data| {
            b.iter(|| {
                let mut pool = UniversalMemoryPool::new();
                let result = pool.embed(data.clone(), 10).unwrap();
                black_box(result);
            });
        });

        // Benchmark detection path (opt-in period detection)
        group.bench_with_input(BenchmarkId::new("detection_path", label), &data, |b, data| {
            b.iter(|| {
                let mut pool = UniversalMemoryPool::new();
                let result = pool.embed_with_detection(data, 10).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark block allocation overhead
fn benchmark_block_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_allocation");

    let size = 1024 * 1024; // 1MB
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    group.throughput(Throughput::Bytes(size as u64));

    let chunker = PeriodDrivenChunker::new(10);
    let chunks = chunker.chunk_and_construct_gauges(&data).unwrap();

    // Method 1: Arc-based blocks (zero-copy)
    group.bench_function("arc_clone", |b| {
        b.iter(|| {
            let mut blocks = Vec::new();
            for (i, chunk) in chunks.iter().enumerate() {
                use hologram_memory_manager::memory::EmbeddedBlock;
                let block = EmbeddedBlock::new(chunk.storage().clone(), chunk.gauge, chunk.primorial, i);
                blocks.push(block);
            }
            black_box(blocks);
        });
    });

    // Method 2: Move ownership
    group.bench_function("move_ownership", |b| {
        b.iter(|| {
            let mut chunks_owned = chunks.clone();
            let mut blocks = Vec::with_capacity(chunks_owned.len());
            for (i, chunk) in chunks_owned.drain(..).enumerate() {
                use hologram_memory_manager::memory::EmbeddedBlock;
                let block = EmbeddedBlock::new(chunk.storage().clone(), chunk.gauge, chunk.primorial, i);
                blocks.push(block);
            }
            black_box(blocks);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_embedding,
    benchmark_chunking,
    benchmark_fast_vs_detection,
    benchmark_block_allocation
);
criterion_main!(benches);
