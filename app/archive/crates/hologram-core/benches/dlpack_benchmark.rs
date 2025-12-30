//! DLPack performance benchmarks
//!
//! Measures overhead of DLPack export/import and verifies zero-copy semantics

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hologram_core::{Executor, Tensor};

/// Benchmark DLPack export overhead
fn bench_dlpack_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("dlpack_export");

    // Test various tensor sizes
    for size in [100, 1000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut exec = Executor::new().unwrap();
            let buffer = exec.allocate::<f32>(size).unwrap();
            let tensor = Tensor::from_buffer(buffer, vec![size]).unwrap();

            b.iter(|| {
                // Export to DLPack (should be ~100-200ns, independent of size)
                let dlpack = exec.tensor_to_dlpack(black_box(&tensor)).unwrap();
                black_box(dlpack);
            });
        });
    }

    group.finish();
}

/// Benchmark DLPack import overhead
fn bench_dlpack_import(c: &mut Criterion) {
    let mut group = c.benchmark_group("dlpack_import");

    // Test various tensor sizes
    for size in [100, 1000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // Create source data
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

            b.iter(|| {
                let mut exec = Executor::new().unwrap();

                // Create simulated DLPack tensor
                let shape_vec: Vec<i64> = vec![size as i64];
                let strides_vec: Vec<i64> = vec![1];

                let shape_ptr = Box::into_raw(shape_vec.into_boxed_slice()) as *mut i64;
                let strides_ptr = Box::into_raw(strides_vec.into_boxed_slice()) as *mut i64;
                let data_ptr = data.as_ptr() as *mut std::ffi::c_void;

                use hologram_core::interop::dlpack::{DLDataType, DLDevice, DLManagedTensor, DLTensor};

                let dl_tensor = DLTensor {
                    data: data_ptr,
                    device: DLDevice::cpu(),
                    ndim: 1,
                    dtype: DLDataType::float32(),
                    shape: shape_ptr,
                    strides: strides_ptr,
                    byte_offset: 0,
                };

                extern "C" fn test_deleter(_tensor: *mut DLManagedTensor) {}

                let managed = unsafe { DLManagedTensor::new(dl_tensor, std::ptr::null_mut(), test_deleter) };
                let managed_ptr = Box::into_raw(managed) as u64;

                // Import from DLPack (copy-based, should scale with size)
                let tensor: Tensor<f32> = exec.tensor_from_dlpack(black_box(managed_ptr)).unwrap();
                black_box(tensor);

                // Clean up
                unsafe {
                    let _ = Box::from_raw(managed_ptr as *mut DLManagedTensor);
                    let _ = Box::from_raw(shape_ptr as *mut [i64; 1]);
                    let _ = Box::from_raw(strides_ptr as *mut [i64; 1]);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark DLPack round-trip
fn bench_dlpack_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("dlpack_round_trip");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut exec = Executor::new().unwrap();

                // Create original tensor
                let buffer = exec.allocate::<f32>(size).unwrap();
                let tensor = Tensor::from_buffer(buffer, vec![size]).unwrap();

                // Export to DLPack
                let dlpack = exec.tensor_to_dlpack(black_box(&tensor)).unwrap();
                let dlpack_ptr = Box::into_raw(dlpack) as u64;

                // Import back
                let imported: Tensor<f32> = exec.tensor_from_dlpack(black_box(dlpack_ptr)).unwrap();
                black_box(imported);

                // Clean up
                unsafe {
                    use hologram_core::interop::dlpack::DLManagedTensor;
                    let dlpack = Box::from_raw(dlpack_ptr as *mut DLManagedTensor);
                    if let Some(deleter) = dlpack.deleter {
                        deleter(Box::into_raw(dlpack));
                    }
                }
            });
        });
    }

    group.finish();
}

/// Benchmark DLPack export with 2D tensors
fn bench_dlpack_export_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("dlpack_export_2d");

    // Test square matrices of various sizes
    for dim in [10, 32, 64, 128, 256, 512].iter() {
        let size = dim * dim;
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            let mut exec = Executor::new().unwrap();
            let buffer = exec.allocate::<f32>(dim * dim).unwrap();
            let tensor = Tensor::from_buffer(buffer, vec![dim, dim]).unwrap();

            b.iter(|| {
                let dlpack = exec.tensor_to_dlpack(black_box(&tensor)).unwrap();
                black_box(dlpack);
            });
        });
    }

    group.finish();
}

/// Benchmark contiguous conversion before export
fn bench_dlpack_export_after_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("dlpack_export_transpose");

    for dim in [32, 64, 128, 256].iter() {
        let size = dim * dim;
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, &dim| {
            b.iter(|| {
                let mut exec = Executor::new().unwrap();
                let buffer = exec.allocate::<f32>(dim * dim).unwrap();
                let tensor = Tensor::from_buffer(buffer, vec![dim, dim]).unwrap();

                // Transpose (non-contiguous)
                let transposed = tensor.transpose().unwrap();

                // Make contiguous
                let contiguous = futures::executor::block_on(transposed.contiguous(&mut exec)).unwrap();

                // Export
                let dlpack = exec.tensor_to_dlpack(black_box(&contiguous)).unwrap();
                black_box(dlpack);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dlpack_export,
    bench_dlpack_import,
    bench_dlpack_round_trip,
    bench_dlpack_export_2d,
    bench_dlpack_export_after_transpose
);
criterion_main!(benches);
