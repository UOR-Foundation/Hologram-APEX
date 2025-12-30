use hologram_core::{precompiled_programs, Executor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(true)
        .with_thread_ids(true)
        .with_line_number(true)
        .init();

    println!("=== ISA Program Performance Profile ===\n");

    // Create executor
    println!("Creating executor...");
    let start = std::time::Instant::now();
    let mut exec = Executor::new()?;
    println!("Executor created in {:?}\n", start.elapsed());

    // Allocate buffers
    println!("Allocating buffers...");
    let start = std::time::Instant::now();
    let size = 1024;

    let mut a = exec.allocate::<f32>(size)?;
    let mut b = exec.allocate::<f32>(size)?;
    let c = exec.allocate::<f32>(size)?;

    // Initialize with data
    let data_a = vec![1.0f32; size];
    let data_b = vec![2.0f32; size];
    a.copy_from_slice(&mut exec, &data_a)?;
    b.copy_from_slice(&mut exec, &data_b)?;

    println!("Buffers allocated in {:?}\n", start.elapsed());

    // First operation (may trigger backend initialization)
    println!("=== First operation (may trigger initialization) ===");
    let start = std::time::Instant::now();
    exec.execute_isa_program(&precompiled_programs::VECTOR_ADD, &[&a, &b], &c, size)?;
    let first_op_time = start.elapsed();
    println!("First vector_add completed in {:?}\n", first_op_time);

    // Subsequent operations (warm)
    println!("=== Subsequent operations (warm) ===");
    let iterations = 10000;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        exec.execute_isa_program(&precompiled_programs::VECTOR_ADD, &[&a, &b], &c, size)?;
    }
    let total_time = start.elapsed();
    let avg_time = total_time / iterations;

    println!("Completed {} iterations in {:?}", iterations, total_time);
    println!("Average time per operation: {:?}", avg_time);
    println!("Throughput: {:.2} Mops/s", 1_000_000.0 / avg_time.as_nanos() as f64);
    println!(
        "Throughput: {:.2} Gelem/s",
        1_000.0 * size as f64 / avg_time.as_nanos() as f64
    );

    Ok(())
}
