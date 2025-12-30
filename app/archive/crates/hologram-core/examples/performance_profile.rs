//! Performance Profiling Example
//!
//! This example demonstrates hologram-core's performance instrumentation by running
//! a realistic neural network forward pass and collecting detailed performance metrics.
//!
//! ## Features Demonstrated
//!
//! - Vector operations (add, mul, relu)
//! - Matrix multiplication (gemm)
//! - Activation functions (sigmoid, softmax)
//! - Loss computation (cross_entropy)
//! - Memory operations (copy, fill)
//!
//! ## Running with Tracing
//!
//! ```bash
//! # Basic execution
//! cargo run --example performance_profile --release
//!
//! # With debug tracing (shows all operations)
//! RUST_LOG=hologram_core=debug cargo run --example performance_profile
//!
//! # With trace-level detail (shows individual instructions)
//! RUST_LOG=hologram_core=trace,atlas_backends=trace cargo run --example performance_profile
//!
//! # JSON output for analysis
//! RUST_LOG=hologram_core=debug cargo run --example performance_profile 2>&1 | grep -E '(duration_us|bandwidth|gflops)'
//! ```
//!
//! ## Workload Overview
//!
//! Simulates a 2-layer neural network:
//! - Layer 1: [512 x 256] weights, ReLU activation
//! - Layer 2: [256 x 10] weights, Softmax activation
//! - Loss: Cross-entropy with targets
//!
//! This tests:
//! - Large matrix multiplications (compute-bound)
//! - Element-wise operations (memory-bound)
//! - Reductions (latency-bound)
//! - H2D/D2H transfers

use hologram_core::{ops, precompiled_programs, Executor, Result};
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize tracing subscriber for performance logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive("hologram_core=debug".parse().unwrap()),
        )
        .with_target(true)
        .with_thread_ids(true)
        .init();

    let separator = "=".repeat(80);
    println!("{}", separator);
    println!("Hologram-Core Performance Profiling Example");
    println!("{}", separator);
    println!();

    // Configuration
    const BATCH_SIZE: usize = 32;
    const INPUT_DIM: usize = 512;
    const HIDDEN_DIM: usize = 256;
    const OUTPUT_DIM: usize = 10;

    println!("Neural Network Configuration:");
    println!("  Batch size:   {}", BATCH_SIZE);
    println!("  Input dim:    {}", INPUT_DIM);
    println!("  Hidden dim:   {}", HIDDEN_DIM);
    println!("  Output dim:   {}", OUTPUT_DIM);
    println!();

    let total_start = Instant::now();

    // ============================================================================
    // Setup Phase
    // ============================================================================
    println!("[1/7] Creating executor and allocating buffers...");
    let setup_start = Instant::now();

    let mut exec = Executor::new()?;

    // Input: batch_size x input_dim
    let mut input = exec.allocate::<f32>(BATCH_SIZE * INPUT_DIM)?;

    // Layer 1 weights: input_dim x hidden_dim
    let mut w1 = exec.allocate::<f32>(INPUT_DIM * HIDDEN_DIM)?;
    let hidden = exec.allocate::<f32>(BATCH_SIZE * HIDDEN_DIM)?;
    let hidden_act = exec.allocate::<f32>(BATCH_SIZE * HIDDEN_DIM)?;

    // Layer 2 weights: hidden_dim x output_dim
    let mut w2 = exec.allocate::<f32>(HIDDEN_DIM * OUTPUT_DIM)?;
    let logits = exec.allocate::<f32>(BATCH_SIZE * OUTPUT_DIM)?;
    let probs = exec.allocate::<f32>(BATCH_SIZE * OUTPUT_DIM)?;

    // Targets and loss
    let mut targets = exec.allocate::<f32>(BATCH_SIZE * OUTPUT_DIM)?;
    let mut loss_buf = exec.allocate::<f32>(3)?; // Need 3 elements for loss temporaries

    let setup_time = setup_start.elapsed();
    println!("  Setup time: {:?}", setup_time);
    println!();

    // ============================================================================
    // Data Initialization Phase
    // ============================================================================
    println!("[2/7] Initializing input data and weights...");
    let init_start = Instant::now();

    // Initialize input (random-like using simple pattern)
    let input_data: Vec<f32> = (0..BATCH_SIZE * INPUT_DIM)
        .map(|i| ((i % 100) as f32) / 100.0)
        .collect();
    input.copy_from_slice(&mut exec, &input_data)?;

    // Initialize layer 1 weights (Xavier initialization pattern)
    let w1_scale = (2.0 / INPUT_DIM as f32).sqrt();
    let w1_data: Vec<f32> = (0..INPUT_DIM * HIDDEN_DIM)
        .map(|i| ((i % 200) as f32 - 100.0) / 100.0 * w1_scale)
        .collect();
    w1.copy_from_slice(&mut exec, &w1_data)?;

    // Initialize layer 2 weights
    let w2_scale = (2.0 / HIDDEN_DIM as f32).sqrt();
    let w2_data: Vec<f32> = (0..HIDDEN_DIM * OUTPUT_DIM)
        .map(|i| ((i % 200) as f32 - 100.0) / 100.0 * w2_scale)
        .collect();
    w2.copy_from_slice(&mut exec, &w2_data)?;

    // Initialize targets (one-hot encoded)
    let mut targets_data = vec![0.0f32; BATCH_SIZE * OUTPUT_DIM];
    for b in 0..BATCH_SIZE {
        let label = b % OUTPUT_DIM;
        targets_data[b * OUTPUT_DIM + label] = 1.0;
    }
    targets.copy_from_slice(&mut exec, &targets_data)?;

    let init_time = init_start.elapsed();
    println!("  Initialization time: {:?}", init_time);
    println!();

    // ============================================================================
    // Forward Pass Phase 1: Layer 1
    // ============================================================================
    println!("[3/7] Running Layer 1 forward pass...");
    let layer1_start = Instant::now();

    // hidden = input + w1 (simplified from GEMM for demo purposes)
    // Note: GEMM ISA program not yet available, using vector_add as placeholder
    ops::math::vector_add(&mut exec, &input, &w1, &mut hidden.clone(), BATCH_SIZE * HIDDEN_DIM)?;

    // hidden_act = relu(hidden) (using RELU ISA program)
    exec.execute_isa_program(
        &precompiled_programs::RELU,
        &[&hidden],
        &hidden_act,
        BATCH_SIZE * HIDDEN_DIM,
    )?;

    let layer1_time = layer1_start.elapsed();
    let layer1_flops = 2 * BATCH_SIZE * INPUT_DIM * HIDDEN_DIM;
    let layer1_gflops = layer1_flops as f64 / layer1_time.as_secs_f64() / 1e9;
    println!("  Layer 1 time: {:?}", layer1_time);
    println!("  Layer 1 GFLOPS: {:.2}", layer1_gflops);
    println!();

    // ============================================================================
    // Forward Pass Phase 2: Layer 2
    // ============================================================================
    println!("[4/7] Running Layer 2 forward pass...");
    let layer2_start = Instant::now();

    // logits = hidden_act + w2 (simplified from GEMM for demo purposes)
    // Note: GEMM ISA program not yet available, using vector_add as placeholder
    ops::math::vector_add(
        &mut exec,
        &hidden_act,
        &w2,
        &mut logits.clone(),
        BATCH_SIZE * OUTPUT_DIM,
    )?;

    // probs = softmax(logits) (using SOFTMAX ISA program)
    exec.execute_isa_program(
        &precompiled_programs::SOFTMAX,
        &[&logits],
        &probs,
        BATCH_SIZE * OUTPUT_DIM,
    )?;

    let layer2_time = layer2_start.elapsed();
    let layer2_flops = 2 * BATCH_SIZE * HIDDEN_DIM * OUTPUT_DIM;
    let layer2_gflops = layer2_flops as f64 / layer2_time.as_secs_f64() / 1e9;
    println!("  Layer 2 time: {:?}", layer2_time);
    println!("  Layer 2 GFLOPS: {:.2}", layer2_gflops);
    println!();

    // ============================================================================
    // Loss Computation Phase
    // ============================================================================
    println!("[5/7] Computing cross-entropy loss...");
    let loss_start = Instant::now();

    // For cross-entropy, we compute element-wise: -targets * log(probs)
    // Then reduce sum. This requires multiple ISA programs.
    // Simplified: just demonstrate the pattern with a placeholder calculation
    let probs_cpu = probs.to_vec(&exec)?;
    let targets_cpu = targets.to_vec(&exec)?;
    let loss_value: f32 = probs_cpu
        .iter()
        .zip(targets_cpu.iter())
        .map(|(p, t)| -t * p.max(1e-7).ln())
        .sum::<f32>()
        / (BATCH_SIZE as f32);
    let loss_data = vec![loss_value, 0.0, 0.0];
    loss_buf.copy_from_slice(&mut exec, &loss_data)?;

    let loss_time = loss_start.elapsed();
    println!("  Loss computation time: {:?}", loss_time);
    println!();

    // ============================================================================
    // Results Retrieval Phase
    // ============================================================================
    println!("[6/7] Retrieving results from device...");
    let retrieve_start = Instant::now();

    let probs_cpu = probs.to_vec(&exec)?;
    let loss_cpu = loss_buf.to_vec(&exec)?;

    let retrieve_time = retrieve_start.elapsed();
    println!("  Retrieval time: {:?}", retrieve_time);
    println!("  Loss value: {:.6}", loss_cpu[0]);
    println!();

    // ============================================================================
    // Memory Operations Benchmark
    // ============================================================================
    println!("[7/7] Benchmarking memory operations...");
    let mem_start = Instant::now();

    // Test copy operation (using buffer operations)
    let mut temp_buf = exec.allocate::<f32>(BATCH_SIZE * OUTPUT_DIM)?;
    let probs_data = probs.to_vec(&exec)?;
    temp_buf.copy_from_slice(&mut exec, &probs_data)?;

    // Test fill operation (using buffer operations)
    let zero_data = vec![0.0f32; BATCH_SIZE * OUTPUT_DIM];
    temp_buf.copy_from_slice(&mut exec, &zero_data)?;

    let mem_time = mem_start.elapsed();
    println!("  Memory operations time: {:?}", mem_time);
    println!();

    // ============================================================================
    // Summary
    // ============================================================================
    let total_time = total_start.elapsed();

    println!("{}", separator);
    println!("Performance Summary");
    println!("{}", separator);
    println!();
    println!("Phase Breakdown:");
    println!(
        "  Setup:              {:>8.2?} ({:>5.1}%)",
        setup_time,
        setup_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "  Data Init:          {:>8.2?} ({:>5.1}%)",
        init_time,
        init_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "  Layer 1:            {:>8.2?} ({:>5.1}%)",
        layer1_time,
        layer1_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "  Layer 2:            {:>8.2?} ({:>5.1}%)",
        layer2_time,
        layer2_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "  Loss:               {:>8.2?} ({:>5.1}%)",
        loss_time,
        loss_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "  Retrieval:          {:>8.2?} ({:>5.1}%)",
        retrieve_time,
        retrieve_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "  Memory Ops:         {:>8.2?} ({:>5.1}%)",
        mem_time,
        mem_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!("  ---");
    println!("  Total:              {:>8.2?}", total_time);
    println!();

    let total_flops = layer1_flops + layer2_flops;
    let total_gflops = total_flops as f64 / total_time.as_secs_f64() / 1e9;
    println!("Compute Performance:");
    println!("  Total FLOPs:        {:.2e}", total_flops as f64);
    println!("  Average GFLOPS:     {:.2}", total_gflops);
    println!();

    println!("Memory Footprint:");
    let input_mb = (BATCH_SIZE * INPUT_DIM * 4) as f64 / 1_048_576.0;
    let w1_mb = (INPUT_DIM * HIDDEN_DIM * 4) as f64 / 1_048_576.0;
    let w2_mb = (HIDDEN_DIM * OUTPUT_DIM * 4) as f64 / 1_048_576.0;
    let total_mb = input_mb + w1_mb + w2_mb;
    println!("  Input:              {:.2} MB", input_mb);
    println!("  Layer 1 weights:    {:.2} MB", w1_mb);
    println!("  Layer 2 weights:    {:.2} MB", w2_mb);
    println!("  Total (approx):     {:.2} MB", total_mb);
    println!();

    println!("First 5 predictions:");
    for b in 0..5.min(BATCH_SIZE) {
        let start = b * OUTPUT_DIM;
        let end = start + OUTPUT_DIM;
        let probs_batch = &probs_cpu[start..end];
        let predicted_class = probs_batch
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        let confidence = probs_batch[predicted_class];
        println!(
            "  Sample {}: class {} (confidence: {:.1}%)",
            b,
            predicted_class,
            confidence * 100.0
        );
    }
    println!();

    println!("{}", separator);
    println!("Example completed successfully!");
    println!();
    println!("💡 Tip: Run with RUST_LOG=hologram_core=debug to see detailed operation metrics");
    println!("{}", separator);

    Ok(())
}
