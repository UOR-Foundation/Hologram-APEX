//! Integration tests for ClassOperations trait implementation on CPUBackend
//!
//! These tests verify the full workflow of:
//! 1. Backend initialization
//! 2. Class operation execution
//! 3. Result verification

use atlas_backends::{class_ops::ClassOperations, cpu::CPUBackend, AtlasBackend};
use atlas_runtime::AtlasSpace;

/// Helper to initialize backend with space
fn setup_backend() -> (CPUBackend, AtlasSpace) {
    let mut space = AtlasSpace::new();
    let mut backend = CPUBackend::new().expect("Failed to create CPUBackend");
    backend.initialize(&space).expect("Failed to initialize backend");
    (backend, space)
}

#[test]
fn test_mark_operation_integration() {
    let (mut backend, _space) = setup_backend();

    // Mark should work on any initialized class
    // Note: Classes are initialized during boundary pool allocation
    // For now, this tests the error path (class not initialized)
    let result = backend.mark(0);

    // Expected: error since class 0 is not yet initialized (no boundary pool allocated)
    assert!(result.is_err(), "Mark should fail on uninitialized class");
}

#[test]
fn test_copy_operation_topology_validation() {
    let (mut backend, _space) = setup_backend();

    // Copy between non-neighbor classes should fail topology validation
    // Even if classes were initialized, this should fail because we're testing
    // topology validation
    let result = backend.copy(0, 95);

    // Expected: topology error (either class not initialized OR not neighbors)
    assert!(result.is_err(), "Copy should validate topology");
}

#[test]
fn test_swap_operation_topology_validation() {
    let (mut backend, _space) = setup_backend();

    // Swap between non-neighbor classes should fail topology validation
    let result = backend.swap(0, 95);

    // Expected: topology error
    assert!(result.is_err(), "Swap should validate topology");
}

#[test]
fn test_merge_operation_topology_validation() {
    let (mut backend, _space) = setup_backend();

    // Merge between non-neighbor classes should fail topology validation
    let result = backend.merge(0, 95, 5);

    // Expected: topology error
    assert!(result.is_err(), "Merge should validate topology");
}

#[test]
fn test_split_operation_topology_validation() {
    let (mut backend, _space) = setup_backend();

    // Split between non-neighbor classes should fail topology validation
    let result = backend.split(0, 95, 5);

    // Expected: topology error
    assert!(result.is_err(), "Split should validate topology");
}

#[test]
fn test_quote_operation_integration() {
    let (mut backend, _space) = setup_backend();

    // Quote should fail on uninitialized class
    let result = backend.quote(0);

    // Expected: error since class 0 is not yet initialized
    assert!(result.is_err(), "Quote should fail on uninitialized class");
}

#[test]
fn test_evaluate_operation_integration() {
    let (mut backend, _space) = setup_backend();

    // Evaluate should fail on uninitialized class
    let result = backend.evaluate(0);

    // Expected: error since class 0 is not yet initialized
    assert!(result.is_err(), "Evaluate should fail on uninitialized class");
}

#[test]
fn test_class_range_validation() {
    let (mut backend, _space) = setup_backend();

    // Test that class index 96 (out of range) is rejected
    let result = backend.mark(96);
    assert!(result.is_err(), "Operations should reject class index >= 96");

    // Test copy with out-of-range src
    let result = backend.copy(96, 0);
    assert!(result.is_err(), "Copy should reject out-of-range src");

    // Test copy with out-of-range dst
    let result = backend.copy(0, 96);
    assert!(result.is_err(), "Copy should reject out-of-range dst");

    // Test swap with out-of-range class_a
    let result = backend.swap(96, 0);
    assert!(result.is_err(), "Swap should reject out-of-range class_a");

    // Test swap with out-of-range class_b
    let result = backend.swap(0, 96);
    assert!(result.is_err(), "Swap should reject out-of-range class_b");

    // Test merge with out-of-range src
    let result = backend.merge(96, 0, 5);
    assert!(result.is_err(), "Merge should reject out-of-range src");

    // Test merge with out-of-range dst
    let result = backend.merge(0, 96, 5);
    assert!(result.is_err(), "Merge should reject out-of-range dst");

    // Test merge with out-of-range context
    let result = backend.merge(0, 5, 96);
    assert!(result.is_err(), "Merge should reject out-of-range context");
}

#[test]
fn test_backend_has_class_arithmetic() {
    let (backend, _space) = setup_backend();

    // Verify that ClassArithmetic was initialized
    // We can't directly access it, but we can verify the backend was created successfully
    assert_eq!(backend.name(), "CPU");
}

#[test]
fn test_class_operations_trait_available() {
    let (mut backend, _space) = setup_backend();

    // Verify that CPUBackend implements ClassOperations trait
    // by calling a method through the trait
    fn accepts_class_operations<T: ClassOperations>(_: &mut T) {}
    accepts_class_operations(&mut backend);
}
