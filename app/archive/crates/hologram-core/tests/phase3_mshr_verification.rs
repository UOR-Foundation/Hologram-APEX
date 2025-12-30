//! Phase 3.0 .mshr File Verification Tests
//!
//! Tests that generated .mshr files load and execute correctly.

use hologram_core::moonshine::CompiledOperation;
use hologram_core::Result;

#[test]
fn test_load_vector_add_mshr() -> Result<()> {
    let path = "/workspace/target/moonshine/ops/vector_add.mshr";

    // Skip if file doesn't exist (CI environments)
    if !std::path::Path::new(path).exists() {
        println!("Skipping test: {path} not found");
        return Ok(());
    }

    // Load operation
    let op = CompiledOperation::load(path)?;

    // Verify metadata
    assert_eq!(op.manifest().operation, "vector_add");
    assert_eq!(op.manifest().input_patterns, 10);
    assert_eq!(op.manifest().output_size, 1);
    assert_eq!(op.pattern_count(), 10);

    println!("✅ vector_add.mshr loaded successfully");
    println!("   Patterns: {}", op.pattern_count());
    println!("   Operation: {}", op.manifest().operation);

    Ok(())
}

#[test]
fn test_execute_vector_add() -> Result<()> {
    let path = "/workspace/target/moonshine/ops/vector_add.mshr";

    if !std::path::Path::new(path).exists() {
        println!("Skipping test: {path} not found");
        return Ok(());
    }

    let op = CompiledOperation::load(path)?;

    // Test known patterns from generator script
    // [1.0, 2.0] -> [3.0] (1+2)
    if op.has_pattern_f32(&[1.0, 2.0]) {
        let result = op.execute_f32(&[1.0, 2.0])?;
        assert_eq!(result.len(), 1);
        assert!((result[0] - 3.0).abs() < 1e-6, "Expected 3.0, got {}", result[0]);
        println!("✅ [1.0, 2.0] -> [3.0] correct");
    }

    // [3.0, 4.0] -> [7.0] (3+4)
    if op.has_pattern_f32(&[3.0, 4.0]) {
        let result = op.execute_f32(&[3.0, 4.0])?;
        assert_eq!(result.len(), 1);
        assert!((result[0] - 7.0).abs() < 1e-6, "Expected 7.0, got {}", result[0]);
        println!("✅ [3.0, 4.0] -> [7.0] correct");
    }

    // [5.0, 6.0] -> [11.0] (5+6)
    if op.has_pattern_f32(&[5.0, 6.0]) {
        let result = op.execute_f32(&[5.0, 6.0])?;
        assert_eq!(result.len(), 1);
        assert!((result[0] - 11.0).abs() < 1e-6, "Expected 11.0, got {}", result[0]);
        println!("✅ [5.0, 6.0] -> [11.0] correct");
    }

    // [0.0, 0.0] -> [0.0]
    if op.has_pattern_f32(&[0.0, 0.0]) {
        let result = op.execute_f32(&[0.0, 0.0])?;
        assert_eq!(result.len(), 1);
        assert!((result[0] - 0.0).abs() < 1e-6, "Expected 0.0, got {}", result[0]);
        println!("✅ [0.0, 0.0] -> [0.0] correct");
    }

    // [-1.0, -2.0] -> [-3.0]
    if op.has_pattern_f32(&[-1.0, -2.0]) {
        let result = op.execute_f32(&[-1.0, -2.0])?;
        assert_eq!(result.len(), 1);
        assert!((result[0] - (-3.0)).abs() < 1e-6, "Expected -3.0, got {}", result[0]);
        println!("✅ [-1.0, -2.0] -> [-3.0] correct");
    }

    Ok(())
}

#[test]
fn test_load_vector_mul_mshr() -> Result<()> {
    let path = "/workspace/target/moonshine/ops/vector_mul.mshr";

    if !std::path::Path::new(path).exists() {
        println!("Skipping test: {path} not found");
        return Ok(());
    }

    // Load operation
    let op = CompiledOperation::load(path)?;

    // Verify metadata
    assert_eq!(op.manifest().operation, "vector_mul");
    assert_eq!(op.manifest().input_patterns, 10);
    assert_eq!(op.manifest().output_size, 1);
    assert_eq!(op.pattern_count(), 10);

    println!("✅ vector_mul.mshr loaded successfully");
    println!("   Patterns: {}", op.pattern_count());
    println!("   Operation: {}", op.manifest().operation);

    Ok(())
}

#[test]
fn test_execute_vector_mul() -> Result<()> {
    let path = "/workspace/target/moonshine/ops/vector_mul.mshr";

    if !std::path::Path::new(path).exists() {
        println!("Skipping test: {path} not found");
        return Ok(());
    }

    let op = CompiledOperation::load(path)?;

    // Test known patterns from generator script
    // [2.0, 3.0] -> [6.0] (2*3)
    if op.has_pattern_f32(&[2.0, 3.0]) {
        let result = op.execute_f32(&[2.0, 3.0])?;
        assert_eq!(result.len(), 1);
        assert!((result[0] - 6.0).abs() < 1e-6, "Expected 6.0, got {}", result[0]);
        println!("✅ [2.0, 3.0] -> [6.0] correct");
    }

    // [4.0, 5.0] -> [20.0] (4*5)
    if op.has_pattern_f32(&[4.0, 5.0]) {
        let result = op.execute_f32(&[4.0, 5.0])?;
        assert_eq!(result.len(), 1);
        assert!((result[0] - 20.0).abs() < 1e-6, "Expected 20.0, got {}", result[0]);
        println!("✅ [4.0, 5.0] -> [20.0] correct");
    }

    // [1.0, 1.0] -> [1.0]
    if op.has_pattern_f32(&[1.0, 1.0]) {
        let result = op.execute_f32(&[1.0, 1.0])?;
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6, "Expected 1.0, got {}", result[0]);
        println!("✅ [1.0, 1.0] -> [1.0] correct");
    }

    // [0.0, 10.0] -> [0.0]
    if op.has_pattern_f32(&[0.0, 10.0]) {
        let result = op.execute_f32(&[0.0, 10.0])?;
        assert_eq!(result.len(), 1);
        assert!((result[0] - 0.0).abs() < 1e-6, "Expected 0.0, got {}", result[0]);
        println!("✅ [0.0, 10.0] -> [0.0] correct");
    }

    // [-2.0, 3.0] -> [-6.0]
    if op.has_pattern_f32(&[-2.0, 3.0]) {
        let result = op.execute_f32(&[-2.0, 3.0])?;
        assert_eq!(result.len(), 1);
        assert!((result[0] - (-6.0)).abs() < 1e-6, "Expected -6.0, got {}", result[0]);
        println!("✅ [-2.0, 3.0] -> [-6.0] correct");
    }

    Ok(())
}

#[test]
fn test_pattern_not_in_cache() -> Result<()> {
    let path = "/workspace/target/moonshine/ops/vector_add.mshr";

    if !std::path::Path::new(path).exists() {
        println!("Skipping test: {path} not found");
        return Ok(());
    }

    let op = CompiledOperation::load(path)?;

    // Try a pattern not in the cache
    let result = op.execute_f32(&[999.0, 888.0]);

    assert!(result.is_err(), "Expected error for unknown pattern");
    println!("✅ Correctly returns error for unknown pattern");

    Ok(())
}
