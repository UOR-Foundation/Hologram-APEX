use hologram_core::{precompiled_programs, Executor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Quick ISA Program Performance Profile ===\n");

    let mut exec = Executor::new()?;
    let size = 1024;

    let mut a = exec.allocate::<f32>(size)?;
    let mut b = exec.allocate::<f32>(size)?;
    let c = exec.allocate::<f32>(size)?;

    let data_a = vec![1.0f32; size];
    let data_b = vec![2.0f32; size];
    a.copy_from_slice(&mut exec, &data_a)?;
    b.copy_from_slice(&mut exec, &data_b)?;

    // First operation
    let start = std::time::Instant::now();
    exec.execute_isa_program(&precompiled_programs::VECTOR_ADD, &[&a, &b], &c, size)?;
    println!("First vector_add: {:?}", start.elapsed());

    // Warm iterations (reduced to 100)
    let iterations = 100;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        exec.execute_isa_program(&precompiled_programs::VECTOR_ADD, &[&a, &b], &c, size)?;
    }
    let total_time = start.elapsed();
    let avg_time = total_time / iterations;

    println!("\nCompleted {} iterations in {:?}", iterations, total_time);
    println!("Average time per operation: {:?}", avg_time);
    println!("Throughput: {:.2} Mops/s", 1_000_000.0 / avg_time.as_micros() as f64);

    Ok(())
}
