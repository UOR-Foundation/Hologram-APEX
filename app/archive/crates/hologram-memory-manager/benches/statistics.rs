//! Statistics Performance Benchmarks
//!
//! Benchmarks different statistics computation methods to identify optimal approaches.
//!
//! Run with: cargo bench --package processor --bench statistics

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use hologram_memory_manager::UniversalMemoryPool;

/// Benchmark data collection methods
fn benchmark_data_collection(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_collection");

    let size = 1024 * 1024; // 1MB
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    let mut pool = UniversalMemoryPool::new();
    pool.embed(data.clone(), 5).unwrap();

    group.throughput(Throughput::Bytes(size as u64));

    // Method 1: Vec::new() + extend_from_slice
    group.bench_function("extend_from_slice", |b| {
        b.iter(|| {
            let mut all_data = Vec::new();
            for block in pool.blocks() {
                if let Some(data) = block.data() {
                    all_data.extend_from_slice(data);
                }
            }
            black_box(all_data);
        });
    });

    // Method 2: Vec::with_capacity() + extend_from_slice
    group.bench_function("with_capacity", |b| {
        b.iter(|| {
            let total_size: usize = pool.blocks().iter().map(|b| b.len()).sum();
            let mut all_data = Vec::with_capacity(total_size);
            for block in pool.blocks() {
                if let Some(data) = block.data() {
                    all_data.extend_from_slice(data);
                }
            }
            black_box(all_data);
        });
    });

    // Method 3: flat_map + collect
    group.bench_function("flat_map_collect", |b| {
        b.iter(|| {
            let all_data: Vec<u8> = pool
                .blocks()
                .iter()
                .filter_map(|b| b.data())
                .flat_map(|data| data.iter().copied())
                .collect();
            black_box(all_data);
        });
    });

    group.finish();
}

/// Benchmark single-pass vs multi-pass statistics
fn benchmark_statistics_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistics_methods");

    let size = 1024 * 1024; // 1MB
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    let mut pool = UniversalMemoryPool::new();
    pool.embed(data.clone(), 5).unwrap();

    let all_data: Vec<u8> = pool
        .blocks()
        .iter()
        .filter_map(|b| b.data())
        .flat_map(|data| data.iter().copied())
        .collect();

    group.throughput(Throughput::Bytes(size as u64));

    // Multi-pass (current implementation)
    group.bench_function("multi_pass", |b| {
        b.iter(|| {
            let sum: u64 = all_data.iter().map(|&b| b as u64).sum();
            let mean = sum as f64 / all_data.len() as f64;

            let variance = all_data
                .iter()
                .map(|&b| {
                    let diff = b as f64 - mean;
                    diff * diff
                })
                .sum::<f64>()
                / all_data.len() as f64;

            let min = *all_data.iter().min().unwrap();
            let max = *all_data.iter().max().unwrap();

            black_box((mean, variance, min, max));
        });
    });

    // Single-pass (Welford's algorithm)
    group.bench_function("single_pass_welford", |b| {
        b.iter(|| {
            let mut count = 0u64;
            let mut mean = 0.0f64;
            let mut m2 = 0.0f64;
            let mut min = u8::MAX;
            let mut max = u8::MIN;

            for &value in &all_data {
                if value < min {
                    min = value;
                }
                if value > max {
                    max = value;
                }

                count += 1;
                let delta = value as f64 - mean;
                mean += delta / count as f64;
                let delta2 = value as f64 - mean;
                m2 += delta * delta2;
            }

            let variance = m2 / count as f64;

            black_box((mean, variance, min, max));
        });
    });

    // Partial optimization (combined min/max)
    group.bench_function("partial_optimization", |b| {
        b.iter(|| {
            let sum: u64 = all_data.iter().map(|&b| b as u64).sum();
            let mean = sum as f64 / all_data.len() as f64;

            let variance = all_data
                .iter()
                .map(|&b| {
                    let diff = b as f64 - mean;
                    diff * diff
                })
                .sum::<f64>()
                / all_data.len() as f64;

            let (min, max) = all_data
                .iter()
                .fold((u8::MAX, u8::MIN), |(min, max), &val| (min.min(val), max.max(val)));

            black_box((mean, variance, min, max));
        });
    });

    group.finish();
}

/// Benchmark individual statistics operations
fn benchmark_individual_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("individual_stats");

    let size = 1024 * 1024; // 1MB
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    group.throughput(Throughput::Bytes(size as u64));

    // Sum computation
    group.bench_function("sum", |b| {
        b.iter(|| {
            let sum: u64 = data.iter().map(|&b| b as u64).sum();
            black_box(sum);
        });
    });

    // Mean computation
    group.bench_function("mean", |b| {
        b.iter(|| {
            let sum: u64 = data.iter().map(|&b| b as u64).sum();
            let mean = sum as f64 / data.len() as f64;
            black_box(mean);
        });
    });

    // Variance computation
    group.bench_function("variance", |b| {
        let sum: u64 = data.iter().map(|&b| b as u64).sum();
        let mean = sum as f64 / data.len() as f64;

        b.iter(|| {
            let variance = data
                .iter()
                .map(|&b| {
                    let diff = b as f64 - mean;
                    diff * diff
                })
                .sum::<f64>()
                / data.len() as f64;
            black_box(variance);
        });
    });

    // Min computation
    group.bench_function("min", |b| {
        b.iter(|| {
            let min = *data.iter().min().unwrap();
            black_box(min);
        });
    });

    // Max computation
    group.bench_function("max", |b| {
        b.iter(|| {
            let max = *data.iter().max().unwrap();
            black_box(max);
        });
    });

    // Combined min/max
    group.bench_function("min_max_combined", |b| {
        b.iter(|| {
            let (min, max) = data
                .iter()
                .fold((u8::MAX, u8::MIN), |(min, max), &val| (min.min(val), max.max(val)));
            black_box((min, max));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_data_collection,
    benchmark_statistics_methods,
    benchmark_individual_stats
);
criterion_main!(benches);
