use hologram_hrm::griess::resonance::*;
use hologram_hrm::griess::{add, product, scalar_mul};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MoonshineHRM Resonance Performance Profile ===\n");
    println!("Vector dimension: 196,884");
    println!("Classes: 96\n");

    // ===================================================================
    // PHASE 1: Baseline - Lift Operation
    // ===================================================================
    println!("--- Phase 1: Lift Operation (ℤ₉₆ → GriessVector) ---");

    let warmup_iterations = 10u32;
    let measure_iterations = 100u32;

    // Warmup
    for k in 0..warmup_iterations {
        let _ = lift((k % 96) as u8)?;
    }

    // Measure lift performance
    let start = Instant::now();
    for k in 0..measure_iterations {
        let _ = lift((k % 96) as u8)?;
    }
    let lift_total = start.elapsed();
    let lift_avg = lift_total / measure_iterations;

    println!("Total time (100 lifts): {:?}", lift_total);
    println!("Average lift time: {:?}", lift_avg);
    println!(
        "Lift throughput: {:.2} ops/sec\n",
        1_000_000_000.0 / lift_avg.as_nanos() as f64
    );

    // ===================================================================
    // PHASE 2: Resonate Operation (Nearest Class Projection)
    // ===================================================================
    println!("--- Phase 2: Resonate Operation (GriessVector → ℤ₉₆) ---");

    // Pre-generate test vectors
    let test_vectors: Vec<_> = (0..10).map(|k| lift(k * 9 + 5).unwrap()).collect();

    // Warmup
    for v in &test_vectors {
        let _ = resonate(v)?;
    }

    // Measure resonate performance
    let start = Instant::now();
    for _ in 0..measure_iterations {
        for v in &test_vectors {
            let _ = resonate(v)?;
        }
    }
    let resonate_total = start.elapsed();
    let resonate_avg = resonate_total / (measure_iterations * 10);

    println!("Total time (1000 resonates): {:?}", resonate_total);
    println!("Average resonate time: {:?}", resonate_avg);
    println!(
        "Resonate throughput: {:.2} ops/sec",
        1_000_000_000.0 / resonate_avg.as_nanos() as f64
    );
    println!("Note: Resonate performs 96 distance computations (196,884-dim each)\n");

    // ===================================================================
    // PHASE 3: Griess Operations
    // ===================================================================
    println!("--- Phase 3: Griess Algebra Operations ---");

    let v1 = lift(10)?;
    let v2 = lift(20)?;

    // Test product
    let start = Instant::now();
    for _ in 0..measure_iterations {
        let _ = product(&v1, &v2)?;
    }
    let product_total = start.elapsed();
    let product_avg = product_total / measure_iterations;

    println!("Product (196,884-dim Hadamard):");
    println!("  Average time: {:?}", product_avg);
    println!(
        "  Throughput: {:.2} ops/sec",
        1_000_000_000.0 / product_avg.as_nanos() as f64
    );

    // Test add
    let start = Instant::now();
    for _ in 0..measure_iterations {
        let _ = add(&v1, &v2)?;
    }
    let add_total = start.elapsed();
    let add_avg = add_total / measure_iterations;

    println!("Addition (196,884-dim):");
    println!("  Average time: {:?}", add_avg);
    println!(
        "  Throughput: {:.2} ops/sec",
        1_000_000_000.0 / add_avg.as_nanos() as f64
    );

    // Test scalar_mul
    let start = Instant::now();
    for _ in 0..measure_iterations {
        let _ = scalar_mul(&v1, 2.0)?;
    }
    let scalar_total = start.elapsed();
    let scalar_avg = scalar_total / measure_iterations;

    println!("Scalar multiplication (196,884-dim):");
    println!("  Average time: {:?}", scalar_avg);
    println!(
        "  Throughput: {:.2} ops/sec\n",
        1_000_000_000.0 / scalar_avg.as_nanos() as f64
    );

    // ===================================================================
    // PHASE 4: Tracked Operations (Full Pipeline)
    // ===================================================================
    println!("--- Phase 4: Tracked Operations (Compute + Route) ---");

    let mut tracks = ParallelResonanceTracks::new();
    let budget = 7u8;

    // Warmup
    for _ in 0..warmup_iterations {
        let _ = tracks.tracked_product(&v1, &v2, budget)?;
        tracks.reset();
    }

    // Measure tracked_product (includes product + resonate + budget accumulation)
    let start = Instant::now();
    for _ in 0..measure_iterations {
        let _ = tracks.tracked_product(&v1, &v2, budget)?;
        tracks.reset();
    }
    let tracked_total = start.elapsed();
    let tracked_avg = tracked_total / measure_iterations;

    println!("Tracked product (product + resonate + route):");
    println!("  Average time: {:?}", tracked_avg);
    println!(
        "  Throughput: {:.2} ops/sec",
        1_000_000_000.0 / tracked_avg.as_nanos() as f64
    );

    // Calculate overhead
    let expected_time = product_avg + resonate_avg;
    let overhead = tracked_avg.saturating_sub(expected_time);
    let overhead_pct = (overhead.as_nanos() as f64 / expected_time.as_nanos() as f64) * 100.0;

    println!("\nOverhead Analysis:");
    println!("  Product time: {:?}", product_avg);
    println!("  Resonate time: {:?}", resonate_avg);
    println!("  Expected total: {:?}", expected_time);
    println!("  Actual tracked time: {:?}", tracked_avg);
    println!("  Overhead: {:?} ({:.1}%)\n", overhead, overhead_pct);

    // ===================================================================
    // PHASE 5: Semiring Operations (Baseline)
    // ===================================================================
    println!("--- Phase 5: Semiring Operations (ℤ₉₆ arithmetic) ---");

    let iterations_semiring = 1_000_000;

    // Test resonance_mul
    let start = Instant::now();
    let mut result = 1u8;
    for i in 0..iterations_semiring {
        result = resonance_mul(result, (i % 96) as u8);
    }
    let mul_total = start.elapsed();
    let mul_avg = mul_total / iterations_semiring;

    println!("Resonance multiplication (⊗):");
    println!("  Total time (1M ops): {:?}", mul_total);
    println!("  Average time: {:?}", mul_avg);
    println!("  Throughput: {:.2} Mops/sec", 1_000.0 / mul_total.as_micros() as f64);

    // Test resonance_add
    let start = Instant::now();
    let mut result = 0u8;
    for i in 0..iterations_semiring {
        result = resonance_add(result, (i % 96) as u8);
    }
    let add_semiring_total = start.elapsed();
    let add_semiring_avg = add_semiring_total / iterations_semiring;

    println!("Resonance addition (⊕):");
    println!("  Total time (1M ops): {:?}", add_semiring_total);
    println!("  Average time: {:?}", add_semiring_avg);
    println!(
        "  Throughput: {:.2} Mops/sec\n",
        1_000.0 / add_semiring_total.as_micros() as f64
    );

    // ===================================================================
    // Summary
    // ===================================================================
    println!("=== Performance Summary ===\n");
    println!("Operation Breakdown:");
    println!("  lift (ℤ₉₆ → 196,884-dim): {:?}/op", lift_avg);
    println!("  resonate (196,884-dim → ℤ₉₆): {:?}/op", resonate_avg);
    println!("  griess_product (196,884-dim): {:?}/op", product_avg);
    println!("  griess_add (196,884-dim): {:?}/op", add_avg);
    println!("  tracked_product (full pipeline): {:?}/op", tracked_avg);
    println!("  semiring ⊗: {:?}/op", mul_avg);
    println!("  semiring ⊕: {:?}/op", add_semiring_avg);

    println!("\nBottleneck Analysis:");
    let slowest = resonate_avg.max(product_avg).max(add_avg);
    if slowest == resonate_avg {
        println!("  BOTTLENECK: resonate() - 96× distance computations");
        println!("  Optimization potential: cache canonical vectors, SIMD distance");
    } else if slowest == product_avg {
        println!("  BOTTLENECK: product() - 196,884 multiplications");
        println!("  Optimization potential: SIMD Hadamard product");
    } else {
        println!("  BOTTLENECK: add() - 196,884 additions");
        println!("  Optimization potential: SIMD vector addition");
    }

    println!("\nProptest Estimates (256 cases/property, 26 properties):");
    let proptest_ops_per_case = 3; // Average ops per property test case
    let total_tracked_ops = 256 * 26 * proptest_ops_per_case;
    let estimated_time = tracked_avg * total_tracked_ops;
    println!("  Total tracked operations: {}", total_tracked_ops);
    println!("  Estimated total time: {:?}", estimated_time);
    println!("  Estimated time (minutes): {:.1}", estimated_time.as_secs_f64() / 60.0);

    Ok(())
}
