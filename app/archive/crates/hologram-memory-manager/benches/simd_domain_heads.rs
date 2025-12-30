//! SIMD Domain Heads Performance Benchmarks
//!
//! Measures the performance of SIMD-accelerated domain heads:
//! - NormalizeDomainHead (SIMD normalization)
//! - FilterDomainHead (SIMD filtering)
//! - AggregateDomainHead (SIMD reductions)
//! - RawDomainHead (parallel block copying)
//!
//! Run with: cargo bench --package processor --bench simd_domain_heads

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hologram_memory_manager::{
    AggregateDomainHead, DomainHead, EmbeddedBlock, FilterDomainHead, Gauge, MemoryStorage, NormalizeDomainHead,
    RawDomainHead, UniversalMemoryPool,
};
use std::sync::Arc;

/// Benchmark normalization across different data sizes
fn benchmark_normalize_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_simd");

    // Test various sizes to measure SIMD scaling
    let sizes = vec![
        (256, "256_elements"),
        (1024, "1KB_elements"),
        (4096, "4KB_elements"),
        (16384, "16KB_elements"),
        (65536, "64KB_elements"),
    ];

    for (num_elements, label) in sizes {
        let byte_size = num_elements * 4; // f32 = 4 bytes
        group.throughput(Throughput::Bytes(byte_size as u64));

        // Create f32-aligned test data
        let float_data: Vec<f32> = (0..num_elements).map(|i| i as f32).collect();
        let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

        group.bench_with_input(BenchmarkId::from_parameter(label), &byte_data, |b, data| {
            let mut pool = UniversalMemoryPool::new();
            let source: Arc<[u8]> = data.clone().into();
            let len = source.len();
            let storage = MemoryStorage::from_arc_slice(source, 0, len);
            let block = EmbeddedBlock::new(storage, Gauge::GAUGE_23, 6, 0);
            pool.insert_block(block);

            let head = NormalizeDomainHead::new(0.0, (num_elements - 1) as f32);

            b.iter(|| {
                let result = head.extract(&pool).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark filtering operations (ReLU and Clip)
fn benchmark_filter_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_simd");

    let num_elements = 16384; // 64 KB of f32 data
    let byte_size = num_elements * 4;
    group.throughput(Throughput::Bytes(byte_size as u64));

    // Create f32-aligned test data: [-8192, -8191, ..., 8191]
    let float_data: Vec<f32> = (-8192..8192).map(|i| i as f32).collect();
    let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

    // Benchmark ReLU (positive only)
    {
        let mut pool = UniversalMemoryPool::new();
        let source: Arc<[u8]> = byte_data.clone().into();
        let len = source.len();
        let storage = MemoryStorage::from_arc_slice(source, 0, len);
        let block = EmbeddedBlock::new(storage, Gauge::GAUGE_23, 6, 0);
        pool.insert_block(block);

        let head = FilterDomainHead::positive_only();

        group.bench_function("relu", |b| {
            b.iter(|| {
                let result = head.extract(&pool).unwrap();
                black_box(result);
            });
        });
    }

    // Benchmark Clip
    {
        let mut pool = UniversalMemoryPool::new();
        let source: Arc<[u8]> = byte_data.into();
        let len = source.len();
        let storage = MemoryStorage::from_arc_slice(source, 0, len);
        let block = EmbeddedBlock::new(storage, Gauge::GAUGE_23, 6, 0);
        pool.insert_block(block);

        let head = FilterDomainHead::clip(-1000.0, 1000.0);

        group.bench_function("clip", |b| {
            b.iter(|| {
                let result = head.extract(&pool).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark SIMD aggregation (min, max, sum)
fn benchmark_aggregate_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregate_simd");

    // Test various sizes to measure SIMD reduction scaling
    let sizes = vec![
        (1024, "1KB_elements"),
        (4096, "4KB_elements"),
        (16384, "16KB_elements"),
        (65536, "64KB_elements"),
        (262144, "256KB_elements"),
    ];

    for (num_elements, label) in sizes {
        let byte_size = num_elements * 4; // f32 = 4 bytes
        group.throughput(Throughput::Bytes(byte_size as u64));

        // Create f32-aligned test data
        let float_data: Vec<f32> = (0..num_elements).map(|i| (i % 10000) as f32).collect();
        let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

        group.bench_with_input(BenchmarkId::from_parameter(label), &byte_data, |b, data| {
            let mut pool = UniversalMemoryPool::new();
            let source: Arc<[u8]> = data.clone().into();
            let len = source.len();
            let storage = MemoryStorage::from_arc_slice(source, 0, len);
            let block = EmbeddedBlock::new(storage, Gauge::GAUGE_23, 6, 0);
            pool.insert_block(block);

            let head = AggregateDomainHead;

            b.iter(|| {
                let result = head.extract(&pool).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark parallel block copying in RawDomainHead
fn benchmark_raw_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_parallel_copy");

    // Test sizes around the 64 KB threshold
    let sizes = vec![
        (32 * 1024, "32KB_sequential"),
        (64 * 1024, "64KB_threshold"),
        (128 * 1024, "128KB_parallel"),
        (256 * 1024, "256KB_parallel"),
        (1024 * 1024, "1MB_parallel"),
    ];

    for (size, label) in sizes {
        group.throughput(Throughput::Bytes(size as u64));

        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        group.bench_with_input(BenchmarkId::from_parameter(label), &data, |b, data| {
            // Create manual block to test parallel copying directly
            let mut pool = UniversalMemoryPool::new();
            let source: Arc<[u8]> = data.clone().into();
            let len = data.len();
            let storage = MemoryStorage::from_arc_slice(source, 0, len);
            let block = EmbeddedBlock::new(storage, Gauge::GAUGE_23, 6, 0);
            pool.insert_block(block);

            let head = RawDomainHead;

            b.iter(|| {
                let result = head.extract(&pool).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark multi-block parallel processing
fn benchmark_raw_multi_block_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_multi_block_parallel");

    // Test different block counts with fixed total size (1 MB)
    let total_size = 1024 * 1024;
    let block_counts = vec![
        (1, "1_block"),
        (4, "4_blocks"),
        (16, "16_blocks"),
        (64, "64_blocks"),
        (256, "256_blocks"),
    ];

    for (num_blocks, label) in block_counts {
        group.throughput(Throughput::Bytes(total_size as u64));

        let block_size = total_size / num_blocks;

        group.bench_with_input(BenchmarkId::from_parameter(label), &num_blocks, |b, &num_blocks| {
            let mut pool = UniversalMemoryPool::new();

            // Create multiple blocks
            for i in 0..num_blocks {
                let data: Vec<u8> = (0..block_size).map(|j| ((i + j) % 256) as u8).collect();
                let source: Arc<[u8]> = data.into();
                let len = source.len();
                let storage = MemoryStorage::from_arc_slice(source, 0, len);
                let block = EmbeddedBlock::new(storage, Gauge::GAUGE_23, 6, i);
                pool.insert_block(block);
            }

            let head = RawDomainHead;

            b.iter(|| {
                let result = head.extract(&pool).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark full pipeline with all SIMD operations
fn benchmark_full_simd_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_simd_pipeline");

    let num_elements = 65536; // 256 KB of f32 data
    let byte_size = num_elements * 4;
    group.throughput(Throughput::Bytes(byte_size as u64));

    // Create f32-aligned test data
    let float_data: Vec<f32> = (0..num_elements).map(|i| (i as f32 - 32768.0) / 100.0).collect();
    let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

    let mut pool = UniversalMemoryPool::new();
    let source: Arc<[u8]> = byte_data.into();
    let len = source.len();
    let storage = MemoryStorage::from_arc_slice(source, 0, len);
    let block = EmbeddedBlock::new(storage, Gauge::GAUGE_23, 6, 0);
    pool.insert_block(block);

    group.bench_function("normalize_filter_aggregate", |b| {
        let normalize_head = NormalizeDomainHead::new(-400.0, 400.0);
        let filter_head = FilterDomainHead::positive_only();
        let aggregate_head = AggregateDomainHead;

        b.iter(|| {
            // Step 1: Normalize
            let normalized = normalize_head.extract(&pool).unwrap();
            black_box(&normalized);

            // Step 2: Filter
            let filtered = filter_head.extract(&pool).unwrap();
            black_box(&filtered);

            // Step 3: Aggregate
            let stats = aggregate_head.extract(&pool).unwrap();
            black_box(stats);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_normalize_simd,
    benchmark_filter_simd,
    benchmark_aggregate_simd,
    benchmark_raw_parallel,
    benchmark_raw_multi_block_parallel,
    benchmark_full_simd_pipeline,
);
criterion_main!(benches);
