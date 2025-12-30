//! WebGPU Performance Measurements
//!
//! Measures actual GPU execution times for comparison with CPU baseline.
//! Run with: `wasm-pack test --chrome crates/hologram-backends --features webgpu`
//!
//! These tests output performance metrics to the console for analysis.
//! Compare with CPU baseline benchmarks (webgpu_baseline.rs) to calculate speedups.

#![cfg(all(feature = "webgpu", target_arch = "wasm32"))]

use hologram_backends::backends::wasm::webgpu::{WebGpuDevice, WebGpuExecutor};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// Helper to create executor
async fn create_executor() -> Option<WebGpuExecutor> {
    if !WebGpuDevice::is_available() {
        return None;
    }

    match WebGpuDevice::new().await {
        Ok(device) => WebGpuExecutor::new(device).ok(),
        Err(_) => None,
    }
}

/// Measure operation time in microseconds
async fn measure_time<F, Fut>(op: F) -> f64
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let window = web_sys::window().expect("no window");
    let performance = window.performance().expect("no performance");

    let start = performance.now();
    op().await;
    let end = performance.now();

    (end - start) * 1000.0 // Convert milliseconds to microseconds
}

/// Performance test for binary operations across sizes
#[wasm_bindgen_test]
async fn perf_binary_operations() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => {
            web_sys::console::log_1(&"Skipping: WebGPU not available".into());
            return;
        }
    };

    web_sys::console::log_1(&"\n=== Binary Operations Performance ===".into());
    web_sys::console::log_1(&"Format: operation, size, time_us, throughput_Melem/s".into());

    // Test sizes: 100, 1K, 10K, 100K, 1M
    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i * 2) as f32).collect();

        // Vector Add
        let time_us = measure_time(|| {
            let a = a.clone();
            let b = b.clone();
            async move {
                let _ = executor.vector_add(&a, &b).await;
            }
        })
        .await;
        let throughput = (*size as f64) / time_us; // Melem/s
        web_sys::console::log_1(&format!("vector_add, {}, {:.2}, {:.2}", size, time_us, throughput).into());

        // Vector Mul
        let time_us = measure_time(|| {
            let a = a.clone();
            let b = b.clone();
            async move {
                let _ = executor.vector_mul(&a, &b).await;
            }
        })
        .await;
        let throughput = (*size as f64) / time_us;
        web_sys::console::log_1(&format!("vector_mul, {}, {:.2}, {:.2}", size, time_us, throughput).into());

        // Vector Min
        let time_us = measure_time(|| {
            let a = a.clone();
            let b = b.clone();
            async move {
                let _ = executor.vector_min(&a, &b).await;
            }
        })
        .await;
        let throughput = (*size as f64) / time_us;
        web_sys::console::log_1(&format!("vector_min, {}, {:.2}, {:.2}", size, time_us, throughput).into());

        // Vector Max
        let time_us = measure_time(|| {
            let a = a.clone();
            let b = b.clone();
            async move {
                let _ = executor.vector_max(&a, &b).await;
            }
        })
        .await;
        let throughput = (*size as f64) / time_us;
        web_sys::console::log_1(&format!("vector_max, {}, {:.2}, {:.2}", size, time_us, throughput).into());

        // Vector Sub
        let time_us = measure_time(|| {
            let a = a.clone();
            let b = b.clone();
            async move {
                let _ = executor.vector_sub(&a, &b).await;
            }
        })
        .await;
        let throughput = (*size as f64) / time_us;
        web_sys::console::log_1(&format!("vector_sub, {}, {:.2}, {:.2}", size, time_us, throughput).into());

        // Vector Div
        let time_us = measure_time(|| {
            let a = a.clone();
            let b = b.clone();
            async move {
                let _ = executor.vector_div(&a, &b).await;
            }
        })
        .await;
        let throughput = (*size as f64) / time_us;
        web_sys::console::log_1(&format!("vector_div, {}, {:.2}, {:.2}", size, time_us, throughput).into());
    }
}

/// Performance test for unary operations across sizes
#[wasm_bindgen_test]
async fn perf_unary_operations() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    web_sys::console::log_1(&"\n=== Unary Operations Performance ===".into());
    web_sys::console::log_1(&"Format: operation, size, time_us, throughput_Melem/s".into());

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        let input: Vec<f32> = (0..*size).map(|i| i as f32 - (*size / 2) as f32).collect();

        // Vector Abs
        let time_us = measure_time(|| {
            let input = input.clone();
            async move {
                let _ = executor.vector_abs(&input).await;
            }
        })
        .await;
        let throughput = (*size as f64) / time_us;
        web_sys::console::log_1(&format!("vector_abs, {}, {:.2}, {:.2}", size, time_us, throughput).into());

        // Vector Exp
        let exp_input: Vec<f32> = (0..*size).map(|i| (i % 10) as f32).collect();
        let time_us = measure_time(|| {
            let input = exp_input.clone();
            async move {
                let _ = executor.vector_exp(&input).await;
            }
        })
        .await;
        let throughput = (*size as f64) / time_us;
        web_sys::console::log_1(&format!("vector_exp, {}, {:.2}, {:.2}", size, time_us, throughput).into());

        // Vector Log
        let log_input: Vec<f32> = (0..*size).map(|i| (i + 1) as f32).collect();
        let time_us = measure_time(|| {
            let input = log_input.clone();
            async move {
                let _ = executor.vector_log(&input).await;
            }
        })
        .await;
        let throughput = (*size as f64) / time_us;
        web_sys::console::log_1(&format!("vector_log, {}, {:.2}, {:.2}", size, time_us, throughput).into());

        // Vector Sqrt
        let sqrt_input: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let time_us = measure_time(|| {
            let input = sqrt_input.clone();
            async move {
                let _ = executor.vector_sqrt(&input).await;
            }
        })
        .await;
        let throughput = (*size as f64) / time_us;
        web_sys::console::log_1(&format!("vector_sqrt, {}, {:.2}, {:.2}", size, time_us, throughput).into());

        // Vector Sigmoid
        let sigmoid_input: Vec<f32> = (0..*size).map(|i| (i % 10) as f32 - 5.0).collect();
        let time_us = measure_time(|| {
            let input = sigmoid_input.clone();
            async move {
                let _ = executor.vector_sigmoid(&input).await;
            }
        })
        .await;
        let throughput = (*size as f64) / time_us;
        web_sys::console::log_1(&format!("vector_sigmoid, {}, {:.2}, {:.2}", size, time_us, throughput).into());

        // Vector Tanh
        let tanh_input: Vec<f32> = (0..*size).map(|i| (i % 10) as f32 - 5.0).collect();
        let time_us = measure_time(|| {
            let input = tanh_input.clone();
            async move {
                let _ = executor.vector_tanh(&input).await;
            }
        })
        .await;
        let throughput = (*size as f64) / time_us;
        web_sys::console::log_1(&format!("vector_tanh, {}, {:.2}, {:.2}", size, time_us, throughput).into());
    }
}

/// Performance test for dispatch threshold crossover points
///
/// Tests operations at and around the default dispatch thresholds
/// to validate the heuristics: binary=1024, unary=1024, reduction=512
#[wasm_bindgen_test]
async fn perf_dispatch_thresholds() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    web_sys::console::log_1(&"\n=== Dispatch Threshold Performance ===".into());
    web_sys::console::log_1(&"Format: operation, size, threshold, time_us".into());

    // Test sizes around dispatch thresholds
    let threshold_tests = [
        ("binary", 512, 1024),    // Below, at threshold
        ("binary", 1024, 1024),   // At threshold
        ("binary", 2048, 1024),   // Above threshold
        ("unary", 512, 1024),     // Below threshold
        ("unary", 1024, 1024),    // At threshold
        ("unary", 2048, 1024),    // Above threshold
        ("reduction", 256, 512),  // Below threshold
        ("reduction", 512, 512),  // At threshold
        ("reduction", 1024, 512), // Above threshold
    ];

    for (op_type, size, threshold) in threshold_tests.iter() {
        match *op_type {
            "binary" => {
                let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
                let b: Vec<f32> = (0..*size).map(|i| (i * 2) as f32).collect();

                let time_us = measure_time(|| {
                    let a = a.clone();
                    let b = b.clone();
                    async move {
                        let _ = executor.vector_add(&a, &b).await;
                    }
                })
                .await;

                web_sys::console::log_1(
                    &format!("vector_add, {}, {} (threshold), {:.2}", size, threshold, time_us).into(),
                );
            }
            "unary" => {
                let input: Vec<f32> = (0..*size).map(|i| i as f32).collect();

                let time_us = measure_time(|| {
                    let input = input.clone();
                    async move {
                        let _ = executor.vector_abs(&input).await;
                    }
                })
                .await;

                web_sys::console::log_1(
                    &format!("vector_abs, {}, {} (threshold), {:.2}", size, threshold, time_us).into(),
                );
            }
            "reduction" => {
                // Reduction tests would go here when implemented
                web_sys::console::log_1(&format!("reduce_sum, {}, {} (threshold), TBD", size, threshold).into());
            }
            _ => {}
        }
    }
}

/// Performance comparison test that outputs a summary table
#[wasm_bindgen_test]
async fn perf_summary_comparison() {
    let mut executor = match create_executor().await {
        Some(e) => e,
        None => return,
    };

    web_sys::console::log_1(&"\n=== Performance Summary ===".into());
    web_sys::console::log_1(
        &"Compare these GPU times with CPU baseline (run: cargo bench --bench webgpu_baseline)".into(),
    );
    web_sys::console::log_1(&"\nSize | vector_add (us) | vector_mul (us) | vector_abs (us) | vector_exp (us)".into());
    web_sys::console::log_1(&"-----|-----------------|-----------------|-----------------|----------------".into());

    for size in [100, 1_000, 10_000, 100_000].iter() {
        let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i * 2) as f32).collect();

        let add_time = measure_time(|| {
            let a = a.clone();
            let b = b.clone();
            async move {
                let _ = executor.vector_add(&a, &b).await;
            }
        })
        .await;

        let mul_time = measure_time(|| {
            let a = a.clone();
            let b = b.clone();
            async move {
                let _ = executor.vector_mul(&a, &b).await;
            }
        })
        .await;

        let abs_time = measure_time(|| {
            let input = a.clone();
            async move {
                let _ = executor.vector_abs(&input).await;
            }
        })
        .await;

        let exp_input: Vec<f32> = (0..*size).map(|i| (i % 10) as f32).collect();
        let exp_time = measure_time(|| {
            let input = exp_input.clone();
            async move {
                let _ = executor.vector_exp(&input).await;
            }
        })
        .await;

        web_sys::console::log_1(
            &format!(
                "{:>5} | {:>15.2} | {:>15.2} | {:>15.2} | {:>15.2}",
                size, add_time, mul_time, abs_time, exp_time
            )
            .into(),
        );
    }
}
