//! Domain Heads Integration Tests
//!
//! Comprehensive integration tests for domain heads:
//! - NormalizeDomainHead
//! - FilterDomainHead
//! - AggregateDomainHead
//! - RawDomainHead (with parallel block copying)
//!
//! These tests validate:
//! 1. Correctness of scalar operations on CPU-resident data
//! 2. Full streaming pipeline integration
//! 3. Performance characteristics
//! 4. Edge cases and error handling
//!
//! Run with: cargo test --package hologram-memory-manager --test simd_domain_heads_integration -- --ignored --nocapture

use hologram_memory_manager::{
    AggregateDomainHead, DomainHead, EmbeddedBlock, FilterDomainHead, Gauge, Modality, NormalizeDomainHead,
    ProcessorError, RawDomainHead, UniversalMemoryPool,
};
use std::sync::Arc;
use std::time::Instant;

#[test]
#[ignore = "Integration test - run explicitly with --ignored"]
fn test_normalize_domain_head_integration() {
    println!("\n=== NormalizeDomainHead Integration Test ===\n");

    // Create f32-aligned test data: [0, 1, 2, ..., 1023]
    let float_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

    // Create pool with aligned block
    let mut pool = UniversalMemoryPool::new();
    let source: Arc<[u8]> = byte_data.into();
    let block = EmbeddedBlock::from_arc(source, Gauge::GAUGE_23, 6, 0);
    pool.insert_block(block);

    println!("Input: {} f32 values (range 0-1023)", float_data.len());

    // Apply normalization
    let head = NormalizeDomainHead::new(0.0, 1023.0);
    let start = Instant::now();
    let modality = head.extract(&pool).unwrap();
    let duration = start.elapsed();

    println!("Normalization time: {:?}", duration);

    // Verify results
    match modality {
        Modality::Normalized(data) => {
            assert_eq!(data.len(), 1024, "Output size mismatch");

            // Check min value (should be ~0.0)
            assert!(
                (data[0] - 0.0).abs() < 0.001,
                "Min value incorrect: expected 0.0, got {}",
                data[0]
            );

            // Check max value (should be ~1.0)
            assert!(
                (data[1023] - 1.0).abs() < 0.001,
                "Max value incorrect: expected 1.0, got {}",
                data[1023]
            );

            // Check mid value (should be ~0.5)
            let mid_expected = 511.5 / 1023.0;
            assert!(
                (data[511] - mid_expected).abs() < 0.001,
                "Mid value incorrect: expected {}, got {}",
                mid_expected,
                data[511]
            );

            // Verify all values in [0, 1]
            for (i, &val) in data.iter().enumerate() {
                assert!((0.0..=1.0).contains(&val), "Value at index {} out of range: {}", i, val);
            }

            println!("✓ All {} normalized values validated", data.len());
        }
        _ => panic!("Expected Normalized modality"),
    }
}

#[test]
#[ignore = "Integration test - run explicitly with --ignored"]
fn test_filter_domain_head_integration() {
    println!("\n=== FilterDomainHead Integration Test ===\n");

    // Create f32-aligned test data: [-512, -511, ..., -1, 0, 1, ..., 511]
    let float_data: Vec<f32> = (-512..512).map(|i| i as f32).collect();
    let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

    println!("Input: {} f32 values (range -512 to 511)", float_data.len());

    // Test 1: Positive-only filter (ReLU)
    {
        let mut pool = UniversalMemoryPool::new();
        let source: Arc<[u8]> = byte_data.clone().into();
        let block = EmbeddedBlock::from_arc(source, Gauge::GAUGE_23, 6, 0);
        pool.insert_block(block);

        let head = FilterDomainHead::positive_only();
        let start = Instant::now();
        let modality = head.extract(&pool).unwrap();
        let duration = start.elapsed();

        println!("ReLU filter time: {:?}", duration);

        match modality {
            Modality::Filtered(data) => {
                assert_eq!(data.len(), 1024, "Output size mismatch");

                // Verify negative values are zeroed
                for (i, &val) in data.iter().enumerate().take(512) {
                    assert_eq!(val, 0.0, "Negative value not zeroed at index {}: {}", i, val);
                }

                // Verify positive values unchanged
                for (i, &val) in data.iter().enumerate().skip(512).take(512) {
                    let expected = (i - 512) as f32;
                    assert_eq!(
                        val, expected,
                        "Positive value changed at index {}: expected {}, got {}",
                        i, expected, val
                    );
                }

                println!(
                    "✓ ReLU filter validated: {} negative → 0, {} positive unchanged",
                    512, 512
                );
            }
            _ => panic!("Expected Filtered modality"),
        }
    }

    // Test 2: Clip filter
    {
        let mut pool = UniversalMemoryPool::new();
        let source: Arc<[u8]> = byte_data.clone().into();
        let block = EmbeddedBlock::from_arc(source, Gauge::GAUGE_23, 6, 0);
        pool.insert_block(block);

        let head = FilterDomainHead::clip(-100.0, 100.0);
        let start = Instant::now();
        let modality = head.extract(&pool).unwrap();
        let duration = start.elapsed();

        println!("Clip filter time: {:?}", duration);

        match modality {
            Modality::Filtered(data) => {
                assert_eq!(data.len(), 1024, "Output size mismatch");

                // Verify all values in [-100, 100]
                for (i, &val) in data.iter().enumerate() {
                    assert!(
                        (-100.0..=100.0).contains(&val),
                        "Value at index {} out of range: {}",
                        i,
                        val
                    );
                }

                // Check clipping bounds
                assert_eq!(data[0], -100.0, "Min clip incorrect");
                assert_eq!(data[1023], 100.0, "Max clip incorrect");

                println!("✓ Clip filter validated: all values in [-100, 100]");
            }
            _ => panic!("Expected Filtered modality"),
        }
    }
}

#[test]
#[ignore = "Integration test - run explicitly with --ignored"]
fn test_aggregate_domain_head_integration() {
    println!("\n=== AggregateDomainHead Integration Test ===\n");

    // Create f32-aligned test data: [0, 1, 2, ..., 999]
    let float_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

    let mut pool = UniversalMemoryPool::new();
    let source: Arc<[u8]> = byte_data.into();
    let block = EmbeddedBlock::from_arc(source, Gauge::GAUGE_23, 6, 0);
    pool.insert_block(block);

    println!("Input: {} f32 values (range 0-999)", float_data.len());

    let head = AggregateDomainHead;
    let start = Instant::now();
    let modality = head.extract(&pool).unwrap();
    let duration = start.elapsed();

    println!("Aggregation time: {:?}", duration);

    // Verify statistics
    match modality {
        Modality::Aggregated { min, max, sum, count } => {
            println!("Statistics computed:");
            println!("  Min:   {}", min);
            println!("  Max:   {}", max);
            println!("  Sum:   {}", sum);
            println!("  Count: {}", count);
            println!("  Mean:  {}", sum / count as f32);

            // Validate min
            assert_eq!(min, 0.0, "Min value incorrect");

            // Validate max
            assert_eq!(max, 999.0, "Max value incorrect");

            // Validate count
            assert_eq!(count, 1000, "Count incorrect");

            // Validate sum (0 + 1 + 2 + ... + 999 = 999 * 1000 / 2 = 499500)
            let expected_sum = 499500.0;
            assert!(
                (sum - expected_sum).abs() < 1.0,
                "Sum incorrect: expected {}, got {}",
                expected_sum,
                sum
            );

            // Validate mean
            let mean = sum / count as f32;
            let expected_mean = 499.5;
            assert!(
                (mean - expected_mean).abs() < 0.1,
                "Mean incorrect: expected {}, got {}",
                expected_mean,
                mean
            );

            println!("✓ All statistics validated");
        }
        _ => panic!("Expected Aggregated modality"),
    }
}

#[test]
#[ignore = "Integration test - run explicitly with --ignored"]
fn test_raw_domain_head_parallel_integration() {
    println!("\n=== RawDomainHead Parallel Integration Test ===\n");

    // Test 1: Small data (sequential path) - Use manual blocks instead of embed
    {
        println!("Test 1: Small data (< 64 KB) - Sequential path");

        let small_data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let mut pool = UniversalMemoryPool::new();

        // Create manual block instead of using embed (which may chunk/duplicate data)
        let source: Arc<[u8]> = small_data.clone().into();
        let block = EmbeddedBlock::from_arc(source, Gauge::GAUGE_23, 6, 0);
        pool.insert_block(block);

        let head = RawDomainHead;
        let start = Instant::now();
        let modality = head.extract(&pool).unwrap();
        let duration = start.elapsed();

        match modality {
            Modality::Raw(data) => {
                assert_eq!(data, small_data, "Small data reconstruction failed");
                println!("  ✓ Sequential path: {:?} for {} bytes", duration, data.len());
            }
            _ => panic!("Expected Raw modality"),
        }
    }

    // Test 2: Large data (parallel path) - Use manual blocks
    {
        println!("Test 2: Large data (> 64 KB) - Parallel path");

        let large_data: Vec<u8> = (0..256 * 1024).map(|i| (i % 256) as u8).collect();
        let mut pool = UniversalMemoryPool::new();

        // Create manual block to ensure exact data reconstruction
        let source: Arc<[u8]> = large_data.clone().into();
        let block = EmbeddedBlock::from_arc(source, Gauge::GAUGE_23, 6, 0);
        pool.insert_block(block);

        println!("  Input: {} bytes, {} blocks", large_data.len(), pool.blocks().len());

        let head = RawDomainHead;
        let start = Instant::now();
        let modality = head.extract(&pool).unwrap();
        let duration = start.elapsed();

        match modality {
            Modality::Raw(data) => {
                assert_eq!(data, large_data, "Large data reconstruction failed");
                let throughput = data.len() as f64 / duration.as_secs_f64() / 1_048_576.0;
                println!(
                    "  ✓ Parallel path: {:?} for {} KB ({:.2} MB/s)",
                    duration,
                    data.len() / 1024,
                    throughput
                );
            }
            _ => panic!("Expected Raw modality"),
        }
    }
}

#[test]
#[ignore = "Integration test - run explicitly with --ignored"]
fn test_full_streaming_pipeline_integration() {
    println!("\n=== Full Streaming Pipeline Integration Test ===\n");

    // Create realistic data: 100KB of f32 values
    let float_data: Vec<f32> = (0..25600).map(|i| (i as f32 - 12800.0) / 100.0).collect();
    let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

    println!("Pipeline input: {} f32 values", float_data.len());

    // Step 1: Create pool with aligned block
    let mut pool = UniversalMemoryPool::new();
    let source: Arc<[u8]> = byte_data.into();
    let block = EmbeddedBlock::from_arc(source, Gauge::GAUGE_23, 6, 0);
    pool.insert_block(block);

    // Step 2: Apply normalization
    println!("\nStep 1: Normalize data to [0, 1]");
    let normalize_head = NormalizeDomainHead::new(-128.0, 128.0);
    let start = Instant::now();
    let normalized = normalize_head.extract(&pool).unwrap();
    println!("  Time: {:?}", start.elapsed());

    match normalized {
        Modality::Normalized(data) => {
            println!("  Output: {} normalized f32 values", data.len());
            assert_eq!(data.len(), 25600);

            // Verify range
            for &val in &data {
                assert!((0.0..=1.0).contains(&val), "Normalized value out of range: {}", val);
            }
        }
        _ => panic!("Expected Normalized modality"),
    }

    // Step 3: Apply filtering
    println!("\nStep 2: Filter positive values (ReLU)");
    let filter_head = FilterDomainHead::positive_only();
    let start = Instant::now();
    let filtered = filter_head.extract(&pool).unwrap();
    println!("  Time: {:?}", start.elapsed());

    match filtered {
        Modality::Filtered(data) => {
            println!("  Output: {} filtered f32 values", data.len());
            assert_eq!(data.len(), 25600);

            // Verify no negative values
            for &val in &data {
                assert!(val >= 0.0, "Negative value after ReLU: {}", val);
            }
        }
        _ => panic!("Expected Filtered modality"),
    }

    // Step 4: Compute statistics
    println!("\nStep 3: Compute aggregated statistics");
    let aggregate_head = AggregateDomainHead;
    let start = Instant::now();
    let stats = aggregate_head.extract(&pool).unwrap();
    println!("  Time: {:?}", start.elapsed());

    match stats {
        Modality::Aggregated { min, max, sum, count } => {
            let mean = sum / count as f32;
            println!("  Min:   {:.4}", min);
            println!("  Max:   {:.4}", max);
            println!("  Mean:  {:.4}", mean);
            println!("  Count: {}", count);

            assert_eq!(count, 25600);
            assert!((-128.0..=128.0).contains(&min));
            assert!((-128.0..=128.0).contains(&max));
        }
        _ => panic!("Expected Aggregated modality"),
    }

    println!("\n✓ Full pipeline integration validated");
}

#[test]
#[ignore = "Integration test - run explicitly with --ignored"]
fn test_multiple_blocks_integration() {
    println!("\n=== Multiple Blocks Integration Test ===\n");

    // Create multiple f32-aligned blocks manually
    let mut pool = UniversalMemoryPool::new();

    // Block 1: [0, 1, 2, ..., 255]
    let float_data1: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let byte_data1: Vec<u8> = bytemuck::cast_slice(&float_data1).to_vec();
    let source1: Arc<[u8]> = byte_data1.into();
    let block1 = EmbeddedBlock::from_arc(source1.clone(), Gauge::GAUGE_23, 6, 0);
    pool.insert_block(block1);

    // Block 2: [256, 257, ..., 511]
    let float_data2: Vec<f32> = (256..512).map(|i| i as f32).collect();
    let byte_data2: Vec<u8> = bytemuck::cast_slice(&float_data2).to_vec();
    let source2: Arc<[u8]> = byte_data2.into();
    let block2 = EmbeddedBlock::from_arc(source2.clone(), Gauge::GAUGE_23, 6, 1);
    pool.insert_block(block2);

    println!("Created pool with {} blocks", pool.blocks().len());

    // Test normalization across blocks
    let head = NormalizeDomainHead::new(0.0, 511.0);
    let modality = head.extract(&pool).unwrap();

    match modality {
        Modality::Normalized(data) => {
            assert_eq!(data.len(), 512, "Combined output size incorrect");

            // Verify first block
            assert!((data[0] - 0.0).abs() < 0.001, "First value incorrect");
            assert!(
                (data[255] - (255.0 / 511.0)).abs() < 0.001,
                "Block 1 last value incorrect"
            );

            // Verify second block
            assert!(
                (data[256] - (256.0 / 511.0)).abs() < 0.001,
                "Block 2 first value incorrect"
            );
            assert!((data[511] - 1.0).abs() < 0.001, "Last value incorrect");

            println!("✓ Multi-block normalization validated");
        }
        _ => panic!("Expected Normalized modality"),
    }
}

#[test]
#[ignore = "Integration test - run explicitly with --ignored"]
fn test_error_handling_integration() {
    println!("\n=== Error Handling Integration Test ===\n");

    // Test 1: Empty pool
    {
        println!("Test 1: Empty pool");
        let pool = UniversalMemoryPool::new();
        let head = NormalizeDomainHead::new(0.0, 1.0);

        match head.extract(&pool) {
            Err(ProcessorError::DomainHeadNotFound(_)) => {
                println!("  ✓ Correctly rejected empty pool");
            }
            _ => panic!("Expected DomainHeadNotFound error for empty pool"),
        }
    }

    // Test 2: Non-f32-aligned block
    {
        println!("Test 2: Non-f32-aligned block");
        let mut pool = UniversalMemoryPool::new();

        // Create 3-byte block (not divisible by 4)
        let byte_data: Vec<u8> = vec![1, 2, 3];
        let source: Arc<[u8]> = byte_data.into();
        let block = EmbeddedBlock::from_arc(source.clone(), Gauge::GAUGE_23, 6, 0);
        pool.insert_block(block);

        let head = NormalizeDomainHead::new(0.0, 1.0);

        match head.extract(&pool) {
            Err(ProcessorError::DomainHeadNotFound(_)) => {
                println!("  ✓ Correctly rejected non-aligned block");
            }
            _ => panic!("Expected DomainHeadNotFound error for non-aligned block"),
        }
    }

    println!("\n✓ All error cases handled correctly");
}
