//! Integration tests for MoonshineHRM compiled operations
//!
//! Tests the full .mshr format implementation including:
//! - Creating .mshr files programmatically
//! - Loading via CompiledOperation
//! - Executing O(1) lookups
//! - OperationRegistry directory loading

use hologram_core::moonshine::{
    hash_input_f32, CompiledOperation, DataType, HashEntry, Manifest, MshrHeader, OperationRegistry,
};
use hologram_core::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use tempfile::TempDir;

// Use std::io::Result for file operations
type IoResult<T> = std::io::Result<T>;
type Result<T> = hologram_core::Result<T>;

/// Helper to create a test .mshr file with given patterns
fn create_test_mshr(path: &PathBuf, operation_name: &str, patterns: &[Vec<f32>], results: &[Vec<f32>]) -> IoResult<()> {
    assert_eq!(patterns.len(), results.len());

    // Create manifest
    let manifest = Manifest {
        operation: operation_name.to_string(),
        version: "1.0.0".to_string(),
        input_patterns: patterns.len(),
        output_size: results[0].len(),
        data_type: DataType::F32,
        hash_function: "fnv1a_64".to_string(),
        compilation_date: "2025-11-14T12:00:00Z".to_string(),
        atlas_version: "1.0.0".to_string(),
    };

    let manifest_json = serde_json::to_vec(&manifest).unwrap();

    // Build hash table
    let mut hash_entries: Vec<HashEntry> = patterns
        .iter()
        .enumerate()
        .map(|(idx, pattern)| {
            let hash = hash_input_f32(pattern);
            HashEntry::new(hash, idx as u32)
        })
        .collect();

    // Sort by hash for binary search
    hash_entries.sort_by_key(|e| e.key_hash);

    // Serialize hash table
    let hash_table_bytes: Vec<u8> = hash_entries
        .iter()
        .flat_map(|entry| {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&entry.key_hash.to_le_bytes());
            bytes.extend_from_slice(&entry.result_index.to_le_bytes());
            bytes.extend_from_slice(&entry._padding.to_le_bytes());
            bytes
        })
        .collect();

    // Serialize result data (all results concatenated)
    let result_data_bytes: Vec<u8> = results
        .iter()
        .flat_map(|result| result.iter().flat_map(|&f| f.to_le_bytes()).collect::<Vec<u8>>())
        .collect();

    // Calculate offsets
    let header_offset = 0u64;
    let manifest_offset = header_offset + MshrHeader::SIZE as u64;
    let manifest_size = manifest_json.len() as u64;
    let hash_table_offset = manifest_offset + manifest_size;
    let hash_table_size = hash_table_bytes.len() as u64;
    let result_data_offset = hash_table_offset + hash_table_size;
    let result_data_size = result_data_bytes.len() as u64;

    // Create header
    let header = MshrHeader::new(
        manifest_offset,
        manifest_size,
        hash_table_offset,
        hash_table_size,
        result_data_offset,
        result_data_size,
    );

    // Write file
    let mut file = File::create(path)?;
    file.write_all(&header.to_bytes())?;
    file.write_all(&manifest_json)?;
    file.write_all(&hash_table_bytes)?;
    file.write_all(&result_data_bytes)?;

    Ok(())
}

#[test]
fn test_create_and_load_mshr() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let mshr_path = temp_dir.path().join("test_add.mshr");

    // Create test patterns and results
    let patterns = vec![vec![1.0f32, 2.0, 3.0], vec![4.0f32, 5.0, 6.0], vec![7.0f32, 8.0, 9.0]];

    let results = vec![
        vec![2.0f32, 4.0, 6.0],    // Pattern 0 result
        vec![8.0f32, 10.0, 12.0],  // Pattern 1 result
        vec![14.0f32, 16.0, 18.0], // Pattern 2 result
    ];

    // Create .mshr file
    create_test_mshr(&mshr_path, "test_add", &patterns, &results).unwrap();

    // Load operation
    let op = CompiledOperation::load(&mshr_path)?;

    // Verify manifest
    assert_eq!(op.manifest().operation, "test_add");
    assert_eq!(op.manifest().input_patterns, 3);
    assert_eq!(op.manifest().output_size, 3);
    assert_eq!(op.pattern_count(), 3);

    Ok(())
}

#[test]
fn test_execute_lookup() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let mshr_path = temp_dir.path().join("test_op.mshr");

    // Create simple test patterns (double each value)
    let patterns = vec![vec![1.0f32, 2.0, 3.0], vec![10.0f32, 20.0, 30.0]];

    let results = vec![
        vec![2.0f32, 4.0, 6.0],    // Pattern 0 result (doubled)
        vec![20.0f32, 40.0, 60.0], // Pattern 1 result (doubled)
    ];

    create_test_mshr(&mshr_path, "test_double", &patterns, &results).unwrap();

    // Load operation
    let op = CompiledOperation::load(&mshr_path)?;

    // Test pattern 0 lookup
    let result0 = op.execute_f32(&patterns[0])?;
    assert_eq!(result0, vec![2.0, 4.0, 6.0]);

    // Test pattern 1 lookup
    let result1 = op.execute_f32(&patterns[1])?;
    assert_eq!(result1, vec![20.0, 40.0, 60.0]);

    Ok(())
}

#[test]
fn test_has_pattern() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let mshr_path = temp_dir.path().join("test_patterns.mshr");

    let patterns = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]];

    let results = vec![vec![10.0f32], vec![20.0f32]];

    create_test_mshr(&mshr_path, "test_has", &patterns, &results).unwrap();

    let op = CompiledOperation::load(&mshr_path)?;

    // Check existing patterns
    assert!(op.has_pattern_f32(&[1.0, 2.0]));
    assert!(op.has_pattern_f32(&[3.0, 4.0]));

    // Check non-existing pattern
    assert!(!op.has_pattern_f32(&[5.0, 6.0]));

    Ok(())
}

#[test]
fn test_pattern_not_found_error() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let mshr_path = temp_dir.path().join("test_error.mshr");

    let patterns = vec![vec![1.0f32, 2.0]];
    let results = vec![vec![10.0f32]];

    create_test_mshr(&mshr_path, "test_error", &patterns, &results).unwrap();

    let op = CompiledOperation::load(&mshr_path)?;

    // Try to execute with unknown pattern
    let result = op.execute_f32(&[99.0, 99.0]);

    assert!(result.is_err());
    match result {
        Err(Error::InvalidOperation(msg)) => {
            assert!(msg.contains("not found"));
        }
        _ => panic!("Expected InvalidOperation error"),
    }

    Ok(())
}

#[test]
fn test_operation_registry_from_directory() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();

    // Create multiple .mshr files
    let patterns1 = vec![vec![1.0f32]];
    let results1 = vec![vec![2.0f32]];
    create_test_mshr(
        &temp_dir.path().join("op1.mshr"),
        "operation_one",
        &patterns1,
        &results1,
    )
    .unwrap();

    let patterns2 = vec![vec![3.0f32]];
    let results2 = vec![vec![6.0f32]];
    create_test_mshr(
        &temp_dir.path().join("op2.mshr"),
        "operation_two",
        &patterns2,
        &results2,
    )
    .unwrap();

    // Create non-.mshr file (should be ignored)
    fs::write(temp_dir.path().join("readme.txt"), "test").unwrap();

    // Load registry
    let registry = OperationRegistry::from_directory(temp_dir.path())?;

    // Verify loaded operations
    assert_eq!(registry.len(), 2);
    assert!(registry.contains("operation_one"));
    assert!(registry.contains("operation_two"));

    // Get and execute operations
    let op1 = registry.get("operation_one")?;
    assert_eq!(op1.manifest().operation, "operation_one");

    let op2 = registry.get("operation_two")?;
    assert_eq!(op2.manifest().operation, "operation_two");

    // Verify names list
    let names = registry.operation_names();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"operation_one".to_string()));
    assert!(names.contains(&"operation_two".to_string()));

    Ok(())
}

#[test]
fn test_registry_get_nonexistent() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let registry = OperationRegistry::from_directory(temp_dir.path())?;

    let result = registry.get("nonexistent");
    assert!(result.is_err());

    match result {
        Err(Error::InvalidOperation(msg)) => {
            assert!(msg.contains("not found"));
        }
        _ => panic!("Expected InvalidOperation error"),
    }

    Ok(())
}

#[test]
fn test_multiple_pattern_lookups() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let mshr_path = temp_dir.path().join("multi_pattern.mshr");

    // Create 10 patterns
    let patterns: Vec<Vec<f32>> = (0..10)
        .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32])
        .collect();

    let results: Vec<Vec<f32>> = (0..10).map(|i| vec![(i * 10) as f32, (i * 20) as f32]).collect();

    create_test_mshr(&mshr_path, "multi_test", &patterns, &results).unwrap();

    let op = CompiledOperation::load(&mshr_path)?;

    // Test all patterns
    for (i, pattern) in patterns.iter().enumerate() {
        let result = op.execute_f32(pattern)?;
        assert_eq!(result, results[i]);
    }

    Ok(())
}

#[test]
fn test_bitwise_exact_matching() -> Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let mshr_path = temp_dir.path().join("bitwise.mshr");

    // Test that 0.0 and -0.0 are treated as different patterns
    let patterns = vec![vec![0.0f32], vec![-0.0f32]];

    let results = vec![
        vec![1.0f32], // Result for 0.0
        vec![2.0f32], // Result for -0.0
    ];

    create_test_mshr(&mshr_path, "bitwise_test", &patterns, &results).unwrap();

    let op = CompiledOperation::load(&mshr_path)?;

    // 0.0 and -0.0 should match different patterns
    assert!(op.has_pattern_f32(&[0.0]));
    assert!(op.has_pattern_f32(&[-0.0]));

    let result_pos = op.execute_f32(&[0.0])?;
    let result_neg = op.execute_f32(&[-0.0])?;

    // Results should be different
    assert_eq!(result_pos, vec![1.0]);
    assert_eq!(result_neg, vec![2.0]);

    Ok(())
}
