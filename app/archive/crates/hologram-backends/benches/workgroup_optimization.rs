//! Workgroup Size Optimization Benchmarks
//!
//! Tests different workgroup sizes (64, 128, 256, 512) to identify optimal
//! configuration for various operation types and data sizes.
//!
//! Run with: `cargo bench --bench workgroup_optimization`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Simulated compute operation with different workgroup sizes
///
/// In actual WebGPU, workgroup size affects:
/// - Occupancy (how many workgroups fit in GPU simultaneously)
/// - Memory coalescing (access patterns)
/// - Shared memory usage
/// - Register pressure
fn simulate_workgroup_dispatch(data_size: usize, workgroup_size: usize) -> usize {
    let num_workgroups = data_size.div_ceil(workgroup_size);

    // Simulate work proportional to dispatched threads
    // In real GPU, this represents kernel launch overhead + execution
    num_workgroups * workgroup_size
}

/// Benchmark binary operations with different workgroup sizes
fn benchmark_binary_workgroup_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("workgroup_binary_ops");

    let data_sizes = [1_000, 10_000, 100_000, 1_000_000];
    let workgroup_sizes = [64, 128, 256, 512];

    for data_size in data_sizes.iter() {
        group.throughput(Throughput::Elements(*data_size as u64));

        for workgroup_size in workgroup_sizes.iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("vector_add_wg{}", workgroup_size), data_size),
                &(data_size, workgroup_size),
                |bencher, &(size, wg_size)| {
                    let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
                    let b: Vec<f32> = (0..*size).map(|i| (i * 2) as f32).collect();
                    let mut output = vec![0.0f32; *size];

                    bencher.iter(|| {
                        // Simulate workgroup dispatch
                        let threads = simulate_workgroup_dispatch(*size, *wg_size);

                        // Perform actual computation (CPU baseline)
                        for i in 0..threads.min(*size) {
                            output[i] = a[i] + b[i];
                        }

                        black_box(&output);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark unary operations with different workgroup sizes
fn benchmark_unary_workgroup_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("workgroup_unary_ops");

    let data_sizes = [1_000, 10_000, 100_000, 1_000_000];
    let workgroup_sizes = [64, 128, 256, 512];

    for data_size in data_sizes.iter() {
        group.throughput(Throughput::Elements(*data_size as u64));

        for workgroup_size in workgroup_sizes.iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("vector_abs_wg{}", workgroup_size), data_size),
                &(data_size, workgroup_size),
                |bencher, &(size, wg_size)| {
                    let input: Vec<f32> = (0..*size).map(|i| i as f32 - (*size / 2) as f32).collect();
                    let mut output = vec![0.0f32; *size];

                    bencher.iter(|| {
                        let threads = simulate_workgroup_dispatch(*size, *wg_size);

                        for i in 0..threads.min(*size) {
                            output[i] = input[i].abs();
                        }

                        black_box(&output);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory operations with different workgroup sizes
fn benchmark_memory_workgroup_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("workgroup_memory_ops");

    let data_sizes = [1_000, 10_000, 100_000, 1_000_000];
    let workgroup_sizes = [64, 128, 256, 512];

    for data_size in data_sizes.iter() {
        group.throughput(Throughput::Elements(*data_size as u64));

        for workgroup_size in workgroup_sizes.iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("copy_wg{}", workgroup_size), data_size),
                &(data_size, workgroup_size),
                |bencher, &(size, wg_size)| {
                    let input: Vec<f32> = (0..*size).map(|i| i as f32).collect();
                    let mut output = vec![0.0f32; *size];

                    bencher.iter(|| {
                        let threads = simulate_workgroup_dispatch(*size, *wg_size);

                        // Memory copy operation
                        output[..threads.min(*size)].copy_from_slice(&input[..threads.min(*size)]);

                        black_box(&output);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark occupancy vs workgroup size tradeoffs
///
/// Tests the relationship between:
/// - Small workgroups: More workgroups, better scheduling flexibility
/// - Large workgroups: Fewer workgroups, better memory coalescing
fn benchmark_occupancy_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("workgroup_occupancy");

    let data_sizes: [usize; 3] = [10_000, 100_000, 1_000_000];
    let workgroup_sizes: [usize; 4] = [64, 128, 256, 512];

    for data_size in data_sizes.iter() {
        for workgroup_size in workgroup_sizes.iter() {
            let num_workgroups = (*data_size).div_ceil(*workgroup_size);
            let occupancy_factor = num_workgroups.min(256); // Assume 256 max concurrent workgroups

            group.bench_with_input(
                BenchmarkId::new(
                    format!("size{}_wg{}_occupancy{}", data_size, workgroup_size, occupancy_factor),
                    data_size,
                ),
                &(data_size, workgroup_size, occupancy_factor),
                |bencher, &(size, wg_size, occ): &(&usize, &usize, usize)| {
                    let input: Vec<f32> = (0..*size).map(|i| i as f32).collect();
                    let mut output = vec![0.0f32; *size];

                    bencher.iter(|| {
                        // Simulate occupancy-limited execution
                        let batches = (*size).div_ceil(*wg_size * occ);

                        for batch in 0..batches {
                            let start: usize = batch * (*wg_size) * occ;
                            let end: usize = (start + (*wg_size) * occ).min(*size);

                            for i in start..end {
                                output[i] = input[i] * 2.0;
                            }
                        }

                        black_box(&output);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark dispatch overhead for different workgroup configurations
///
/// Measures the fixed cost of dispatching workgroups
fn benchmark_dispatch_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("workgroup_dispatch_overhead");

    let data_sizes: [usize; 3] = [1_000, 10_000, 100_000];
    let workgroup_sizes: [usize; 4] = [64, 128, 256, 512];

    for data_size in data_sizes.iter() {
        for workgroup_size in workgroup_sizes.iter() {
            let num_workgroups = (*data_size).div_ceil(*workgroup_size);

            group.bench_with_input(
                BenchmarkId::new(
                    format!("size{}_wg{}_dispatches{}", data_size, workgroup_size, num_workgroups),
                    data_size,
                ),
                &num_workgroups,
                |bencher, &dispatches| {
                    bencher.iter(|| {
                        // Simulate dispatch overhead (proportional to number of workgroups)
                        let overhead: usize = (0..dispatches).map(|i| i % 100).sum();
                        black_box(overhead);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_binary_workgroup_sizes,
    benchmark_unary_workgroup_sizes,
    benchmark_memory_workgroup_sizes,
    benchmark_occupancy_analysis,
    benchmark_dispatch_overhead
);
criterion_main!(benches);
