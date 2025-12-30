//! Optimization Methods Benchmarks
//!
//! Systematically benchmarks different implementation strategies
//! to identify optimal approaches for raw extraction.
//!
//! Run with: cargo bench --package processor --bench optimization_methods

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hologram_memory_manager::UniversalMemoryPool;

/// Benchmark different extraction implementations
fn benchmark_extraction_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("extraction_methods");

    let size = 10 * 1024 * 1024; // 10MB
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    let mut pool = UniversalMemoryPool::new();
    pool.embed(data.clone(), 10).unwrap();

    group.throughput(Throughput::Bytes(size as u64));

    // Method 1: extend_from_slice per block
    group.bench_function("extend_from_slice", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(pool.total_bytes());
            for block in pool.blocks() {
                if let Some(data) = block.data() {
                    result.extend_from_slice(data);
                }
            }
            black_box(result);
        });
    });

    // Method 2: Reserve then extend
    group.bench_function("reserve_exact", |b| {
        b.iter(|| {
            let mut result = Vec::new();
            result.reserve_exact(pool.total_bytes());
            for block in pool.blocks() {
                if let Some(data) = block.data() {
                    result.extend_from_slice(data);
                }
            }
            black_box(result);
        });
    });

    // Method 3: flat_map iterator
    group.bench_function("flat_map", |b| {
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

    // Method 4: Unsafe bulk memcpy
    group.bench_function("unsafe_memcpy", |b| {
        b.iter(|| {
            let mut result: Vec<u8> = Vec::with_capacity(pool.total_bytes());
            unsafe {
                let mut offset = 0;
                for block in pool.blocks() {
                    if let Some(data) = block.data() {
                        std::ptr::copy_nonoverlapping(data.as_ptr(), result.as_mut_ptr().add(offset), data.len());
                        offset += data.len();
                    }
                }
                result.set_len(pool.total_bytes());
            }
            black_box(result);
        });
    });

    // Method 5: Pre-sized Vec + manual copy
    group.bench_function("pre_sized_copy", |b| {
        b.iter(|| {
            let mut result = vec![0u8; pool.total_bytes()];
            let mut offset = 0;
            for block in pool.blocks() {
                if let Some(data) = block.data() {
                    result[offset..offset + data.len()].copy_from_slice(data);
                    offset += data.len();
                }
            }
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark allocation vs copy overhead
fn benchmark_allocation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_overhead");

    let size = 10 * 1024 * 1024; // 10MB
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    let mut pool = UniversalMemoryPool::new();
    pool.embed(data.clone(), 10).unwrap();

    group.throughput(Throughput::Bytes(size as u64));

    // Just allocation
    group.bench_function("allocation_only", |b| {
        b.iter(|| {
            let result = Vec::<u8>::with_capacity(pool.total_bytes());
            black_box(result);
        });
    });

    // Allocation + copy
    group.bench_function("allocation_plus_copy", |b| {
        b.iter(|| {
            let mut result = Vec::with_capacity(pool.total_bytes());
            for block in pool.blocks() {
                if let Some(data) = block.data() {
                    result.extend_from_slice(data);
                }
            }
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark impact of block count on extraction
fn benchmark_block_count_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_count_impact");

    let total_size = 10 * 1024 * 1024; // 10MB
    group.throughput(Throughput::Bytes(total_size as u64));

    // Test with different primorial levels (different block counts)
    for levels in [5, 7, 10, 12, 15] {
        let data: Vec<u8> = (0..total_size).map(|i| (i % 256) as u8).collect();

        let mut pool = UniversalMemoryPool::new();
        pool.embed(data.clone(), levels).unwrap();

        group.bench_with_input(BenchmarkId::new("levels", levels), &pool, |b, pool| {
            b.iter(|| {
                let mut result = Vec::with_capacity(pool.total_bytes());
                for block in pool.blocks() {
                    if let Some(data) = block.data() {
                        result.extend_from_slice(data);
                    }
                }
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_extraction_methods,
    benchmark_allocation_overhead,
    benchmark_block_count_impact
);
criterion_main!(benches);
