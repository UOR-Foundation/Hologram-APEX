//! Benchmark: ISA Program-based Operations Performance
//!
//! Measures the performance of ISA program-based math operations
//! using precompiled programs from MoonshineHRM.

use hologram_core::{precompiled_programs, Executor};
use std::time::Instant;

fn benchmark_operation<F>(name: &str, iterations: usize, mut op: F) -> f64
where
    F: FnMut() -> hologram_core::Result<()>,
{
    // Warmup
    for _ in 0..10 {
        op().expect("Warmup failed");
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        op().expect("Operation failed");
    }
    let duration = start.elapsed();

    let avg_us = duration.as_micros() as f64 / iterations as f64;
    println!("{:30} {:12.2} µs/op  ({:8.2} Mops/s)", name, avg_us, 1.0 / avg_us);
    avg_us
}

fn main() -> hologram_core::Result<()> {
    println!("\n=== ISA Program-based Math Operations Performance ===\n");
    println!("Testing with 1024 f32 elements (4096 bytes)");
    println!("Iterations: 1000 per operation");
    println!("Memory: Standard allocations with ISA program execution\n");

    let iterations = 1000;
    let n = 1024;

    // ============================================================================
    // Binary Operations (ISA programs)
    // ============================================================================
    println!("--- Binary Operations ---\n");

    {
        let mut exec = Executor::new()?;
        let mut a = exec.allocate::<f32>(n)?;
        let mut b = exec.allocate::<f32>(n)?;
        let c = exec.allocate::<f32>(n)?;

        let data: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
        a.copy_from_slice(&mut exec, &data)?;
        b.copy_from_slice(&mut exec, &data)?;

        benchmark_operation("vector_add", iterations, || {
            exec.execute_isa_program(&precompiled_programs::VECTOR_ADD, &[&a, &b], &c, n)
        });
        benchmark_operation("vector_sub", iterations, || {
            exec.execute_isa_program(&precompiled_programs::VECTOR_SUB, &[&a, &b], &c, n)
        });
        benchmark_operation("vector_mul", iterations, || {
            exec.execute_isa_program(&precompiled_programs::MUL, &[&a, &b], &c, n)
        });
        benchmark_operation("vector_div", iterations, || {
            exec.execute_isa_program(&precompiled_programs::VECTOR_DIV, &[&a, &b], &c, n)
        });
        benchmark_operation("min", iterations, || {
            exec.execute_isa_program(&precompiled_programs::VECTOR_MIN, &[&a, &b], &c, n)
        });
        benchmark_operation("max", iterations, || {
            exec.execute_isa_program(&precompiled_programs::VECTOR_MAX, &[&a, &b], &c, n)
        });
    }
    println!();

    // ============================================================================
    // Unary Operations (ISA programs)
    // ============================================================================
    println!("--- Unary Operations ---\n");

    {
        let mut exec = Executor::new()?;
        let mut a = exec.allocate::<f32>(n)?;
        let b = exec.allocate::<f32>(n)?;

        let data: Vec<f32> = (0..n).map(|i| (i as f32) - 512.0).collect();
        a.copy_from_slice(&mut exec, &data)?;

        benchmark_operation("abs", iterations, || {
            exec.execute_isa_program(&precompiled_programs::VECTOR_ABS, &[&a], &b, n)
        });
        benchmark_operation("neg", iterations, || {
            exec.execute_isa_program(&precompiled_programs::VECTOR_NEG, &[&a], &b, n)
        });
        benchmark_operation("relu", iterations, || {
            exec.execute_isa_program(&precompiled_programs::RELU, &[&a], &b, n)
        });
    }
    println!();

    // ============================================================================
    // Summary
    // ============================================================================
    println!("=== Summary ===\n");
    println!("ISA Program-based operations characteristics:");
    println!("  ✓ Precompiled programs - build-time ISA generation from kernel schemas");
    println!("  ✓ Backend execution - all compute via ISA programs on backend");
    println!("  ✓ Zero runtime compilation - programs loaded as static constants");
    println!("  ✓ Type-generic operations - single ISA program handles all numeric types");
    println!("  ✓ Typical operation latency: varies by backend and operation complexity");
    println!("\nMoonshineHRM Architecture:");
    println!("  - Python kernel schemas → JSON → ISA programs (build-time)");
    println!("  - ISA programs embedded as Rust const declarations");
    println!("  - Executor.execute_isa_program() dispatches to backend");
    println!("  - Backend executes precompiled ISA instructions");
    println!("  - 57 total precompiled programs available\n");

    Ok(())
}
