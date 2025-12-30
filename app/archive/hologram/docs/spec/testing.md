# Testing Specification

**Status:** Approved
**Version:** 0.1.0
**Last Updated:** 2025-01-18

## Overview

This specification defines the comprehensive testing strategy for Hologram. It covers unit tests, integration tests, property-based tests, and continuous testing practices.

## Testing Philosophy

**🚨 CRITICAL: No feature is complete without comprehensive tests.**

Core principles:

1. **Test-Driven Development (TDD)** - Write tests before implementation
2. **High Coverage** - Minimum 80% code coverage, 100% for critical paths
3. **Fast Feedback** - Test suite completes in < 3 minutes
4. **Deterministic** - No flaky tests, reproducible results
5. **Comprehensive** - Unit, integration, property-based, and doc tests
6. **Maintainable** - Clear test names, well-organized, easy to understand

## Testing Levels

### 1. Unit Tests

**Scope:** Individual functions and methods

**Location:** Within each module in `#[cfg(test)]` sections

**Example:**

```rust
// crates/core/src/buffer.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        // Arrange
        let mut exec = Executor::new().unwrap();

        // Act
        let buffer = exec.allocate::<f32>(1024).unwrap();

        // Assert
        assert_eq!(buffer.size(), 1024);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_buffer_copy_from_slice() {
        let mut exec = Executor::new().unwrap();
        let mut buffer = exec.allocate::<f32>(4).unwrap();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        buffer.copy_from_slice(&data).unwrap();

        let result = buffer.to_vec().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    #[should_panic(expected = "allocation size exceeds limit")]
    fn test_buffer_allocation_fails_when_too_large() {
        let mut exec = Executor::new().unwrap();
        let _ = exec.allocate::<f32>(usize::MAX).unwrap();
    }

    #[test]
    fn test_buffer_error_handling() {
        let mut exec = Executor::new().unwrap();
        let buffer = exec.allocate::<f32>(4).unwrap();

        // Wrong size should error
        let result = buffer.copy_from_slice(&[1.0, 2.0]);
        assert!(result.is_err());
    }
}
```

**Requirements:**

- Test all public functions
- Test error paths
- Test edge cases (empty, single element, maximum size)
- Use descriptive test names: `test_<what>_<when>_<expected>`

### 2. Integration Tests

**Scope:** Component interactions, full workflows

**Location:** `crates/*/tests/` directories

**Example:**

```rust
// crates/core/tests/integration_test.rs

use hologram_core::{ops, Executor, Result};

#[test]
fn test_full_vector_addition_workflow() -> Result<()> {
    // Create executor
    let mut exec = Executor::new()?;

    // Allocate buffers
    let mut a = exec.allocate::<f32>(256)?;
    let mut b = exec.allocate::<f32>(256)?;
    let mut c = exec.allocate::<f32>(256)?;

    // Setup input data
    let data_a: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..256).map(|i| (i * 2) as f32).collect();

    a.copy_from_slice(&data_a)?;
    b.copy_from_slice(&data_b)?;

    // Execute operation
    ops::math::vector_add(&exec, &a, &b, &mut c, 256)?;

    // Verify results
    let result = c.to_vec()?;
    for i in 0..256 {
        let expected = data_a[i] + data_b[i];
        assert!((result[i] - expected).abs() < 1e-6);
    }

    Ok(())
}

#[test]
fn test_tensor_operations_compose() -> Result<()> {
    let mut exec = Executor::new()?;

    // Create tensors
    let buf_a = exec.allocate::<f32>(12)?;
    let buf_b = exec.allocate::<f32>(12)?;

    let tensor_a = Tensor::from_buffer(buf_a, vec![3, 4])?;
    let tensor_b = Tensor::from_buffer(buf_b, vec![4, 3])?;

    // Chain operations: select → transpose → matmul
    let slice = tensor_a.select(0, 1)?;
    let transposed = tensor_b.transpose()?;
    let result = slice.matmul(&exec, &transposed)?;

    // Verify shape
    assert_eq!(result.shape(), &[4, 4]);

    Ok(())
}
```

**Requirements:**

- Test complete workflows
- Test cross-crate interactions
- Test realistic usage scenarios
- Verify end-to-end correctness

### 3. Property-Based Tests

**Scope:** Mathematical invariants, algebraic properties

**Tool:** `proptest`

**Example:**

```rust
// crates/compiler/tests/canonicalization_properties.rs

use proptest::prelude::*;
use hologram_compiler::Canonicalizer;

proptest! {
    #[test]
    fn test_canonicalization_is_idempotent(circuit in any::<String>()) {
        // Canonicalizing twice gives same result as canonicalizing once
        if let Ok(once) = Canonicalizer::canonicalize(&circuit) {
            let twice = Canonicalizer::canonicalize(&once).unwrap();
            prop_assert_eq!(&once, &twice);
        }
    }

    #[test]
    fn test_hadamard_squared_is_identity(class in 0..96u8) {
        // H² = I for all classes
        let after_h = apply_hadamard(class);
        let after_hh = apply_hadamard(after_h);
        prop_assert_eq!(class, after_hh);
    }

    #[test]
    fn test_pauli_x_squared_is_identity(class in 0..96u8) {
        // X² = I for all classes
        let after_x = apply_pauli_x(class);
        let after_xx = apply_pauli_x(after_x);
        prop_assert_eq!(class, after_xx);
    }

    #[test]
    fn test_class_operations_stay_in_range(class in 0..96u8) {
        // All operations keep class in valid range [0, 96)
        let ops = vec![
            apply_hadamard(class),
            apply_pauli_x(class),
            apply_pauli_z(class),
            apply_rotation(class),
        ];

        for result in ops {
            prop_assert!(result < 96);
        }
    }

    #[test]
    fn test_vector_add_is_commutative(
        a in prop::collection::vec(any::<f32>(), 256),
        b in prop::collection::vec(any::<f32>(), 256)
    ) {
        let mut exec = Executor::new().unwrap();

        let mut buf_a = exec.allocate::<f32>(256).unwrap();
        let mut buf_b = exec.allocate::<f32>(256).unwrap();
        let mut result_ab = exec.allocate::<f32>(256).unwrap();
        let mut result_ba = exec.allocate::<f32>(256).unwrap();

        buf_a.copy_from_slice(&a).unwrap();
        buf_b.copy_from_slice(&b).unwrap();

        // a + b
        ops::math::vector_add(&exec, &buf_a, &buf_b, &mut result_ab, 256).unwrap();

        // b + a
        ops::math::vector_add(&exec, &buf_b, &buf_a, &mut result_ba, 256).unwrap();

        let ab = result_ab.to_vec().unwrap();
        let ba = result_ba.to_vec().unwrap();

        // Should be equal (within floating point tolerance)
        for i in 0..256 {
            prop_assert!((ab[i] - ba[i]).abs() < 1e-5);
        }
    }
}
```

**Requirements:**

- Test algebraic properties (commutativity, associativity, identity, inverse)
- Test invariants (bounds, monotonicity, conservation laws)
- Use randomized inputs
- Run 1000+ test cases per property

### 4. Documentation Tests

**Scope:** Examples in doc comments

**Example:**

````rust
/// Compute element-wise sum of two vectors
///
/// # Example
///
/// ```
/// use hologram_core::{ops, Executor};
///
/// let mut exec = Executor::new().unwrap();
/// let mut a = exec.allocate::<f32>(4).unwrap();
/// let mut b = exec.allocate::<f32>(4).unwrap();
/// let mut c = exec.allocate::<f32>(4).unwrap();
///
/// a.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
/// b.copy_from_slice(&[5.0, 6.0, 7.0, 8.0]).unwrap();
///
/// ops::math::vector_add(&exec, &a, &b, &mut c, 4).unwrap();
///
/// let result = c.to_vec().unwrap();
/// assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
/// ```
pub fn vector_add<T: bytemuck::Pod>(
    exec: &Executor,
    a: &Buffer<T>,
    b: &Buffer<T>,
    c: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    // Implementation
}
````

**Requirements:**

- Every public function has example in doc comment
- Examples compile and run
- Examples demonstrate typical usage
- Examples are concise and clear

## Test Organization

### Directory Structure

```
hologram/
├── crates/
│   ├── core/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── buffer.rs
│   │   │   │   └── #[cfg(test)] mod tests {...}
│   │   │   └── ops/
│   │   │       ├── math.rs
│   │   │       │   └── #[cfg(test)] mod tests {...}
│   │   └── tests/
│   │       ├── integration_test.rs
│   │       ├── tensor_tests.rs
│   │       └── fixtures/
│   │           └── test_data.json
│   ├── compiler/
│   │   ├── src/
│   │   │   └── ... (unit tests)
│   │   └── tests/
│   │       ├── canonicalization_properties.rs
│   │       └── pattern_matching_tests.rs
│   └── backends/
│       ├── src/
│       │   └── ... (unit tests)
│       └── tests/
│           ├── cpu_backend_tests.rs
│           ├── gpu_backend_tests.rs
│           └── backend_compliance.rs
└── tests/
    └── workspace_integration_tests.rs  # Cross-crate tests
```

### Test Naming Conventions

**Pattern:** `test_<component>_<action>_<expected_outcome>`

**Examples:**

- ✅ `test_buffer_allocate_succeeds_with_valid_size`
- ✅ `test_tensor_matmul_computes_correct_result`
- ✅ `test_canonicalizer_reduces_hadamard_squared_to_identity`
- ✅ `test_executor_returns_error_when_backend_unavailable`

**Avoid:**
- ❌ `test1`, `test2` (meaningless names)
- ❌ `it_works` (too generic)
- ❌ `test_stuff` (unclear what is being tested)

## Test Fixtures and Utilities

### Shared Test Utilities

**File:** `crates/core/src/test_utils.rs`

```rust
#[cfg(test)]
pub mod test_utils {
    use super::*;

    /// Create executor for testing (with small memory limits)
    pub fn create_test_executor() -> Result<Executor> {
        let config = Config {
            backend: BackendConfig {
                backend_type: BackendType::Cpu,
                device_id: 0,
            },
            memory: MemoryConfig {
                pool_size: 1024 * 1024, // 1 MB for tests
                max_buffers: 128,
            },
            ..Default::default()
        };

        Executor::with_config(config)
    }

    /// Generate test data with specific pattern
    pub fn generate_test_data(size: usize, pattern: TestPattern) -> Vec<f32> {
        match pattern {
            TestPattern::Zeros => vec![0.0; size],
            TestPattern::Ones => vec![1.0; size],
            TestPattern::Increasing => (0..size).map(|i| i as f32).collect(),
            TestPattern::Random(seed) => {
                let mut rng = StdRng::seed_from_u64(seed);
                (0..size).map(|_| rng.gen()).collect()
            }
        }
    }

    pub enum TestPattern {
        Zeros,
        Ones,
        Increasing,
        Random(u64),
    }

    /// Assert floating-point equality within tolerance
    pub fn assert_f32_eq(a: f32, b: f32, tolerance: f32) {
        assert!(
            (a - b).abs() < tolerance,
            "Values not equal: {} != {} (tolerance: {})",
            a, b, tolerance
        );
    }

    /// Assert vector equality within tolerance
    pub fn assert_vec_f32_eq(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len(), "Vector lengths differ");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tolerance,
                "Vectors differ at index {}: {} != {} (tolerance: {})",
                i, x, y, tolerance
            );
        }
    }
}
```

### Test Fixtures

**File:** `crates/core/tests/fixtures/tensor_data.json`

```json
{
  "matmul_2x3_3x2": {
    "a": {
      "shape": [2, 3],
      "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    },
    "b": {
      "shape": [3, 2],
      "data": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    },
    "expected": {
      "shape": [2, 2],
      "data": [58.0, 64.0, 139.0, 154.0]
    }
  }
}
```

**Usage:**

```rust
#[test]
fn test_matmul_with_fixture() {
    let fixture = load_fixture("tensor_data.json");
    let test_case = &fixture["matmul_2x3_3x2"];

    let mut exec = Executor::new().unwrap();

    let a = Tensor::from_data(&exec, &test_case.a.data, test_case.a.shape.clone()).unwrap();
    let b = Tensor::from_data(&exec, &test_case.b.data, test_case.b.shape.clone()).unwrap();

    let result = a.matmul(&exec, &b).unwrap();
    let result_data = result.to_vec().unwrap();

    assert_vec_f32_eq(&result_data, &test_case.expected.data, 1e-5);
}
```

## Coverage Requirements

### Minimum Coverage Targets

| Component | Unit Test Coverage | Integration Test Coverage |
|-----------|-------------------|---------------------------|
| hologram-core | 80% minimum | 60% minimum |
| hologram-compiler | 100% (critical path) | 80% minimum |
| hologram-backends | 85% minimum | 70% minimum |
| hologram-config | 75% minimum | 50% minimum |
| hologram-ffi | 80% minimum | 60% minimum |

### Critical Paths Requiring 100% Coverage

- Canonicalization rules (hologram-compiler)
- Generator compilation (hologram-compiler)
- Memory allocation/deallocation (hologram-core)
- Error handling paths (all crates)
- FFI boundary crossings (hologram-ffi)

### Measuring Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --workspace --all-features --out Html

# Open report
open tarpaulin-report.html

# CI: Generate XML for Codecov
cargo tarpaulin --workspace --all-features --out Xml
```

## Testing Tools

### Required Dependencies

```toml
[dev-dependencies]
# Property-based testing
proptest = "1.4"

# Assertions
assert_matches = "1.5"
pretty_assertions = "1.4"

# Test fixtures
serde_json = "1.0"
tempfile = "3.8"

# Benchmarking (also used for performance tests)
criterion = "0.5"

# Mocking
mockall = "0.12"
```

### Useful Cargo Commands

```bash
# Run all tests
cargo test --workspace --all-features

# Run specific test
cargo test test_buffer_allocate

# Run tests in specific crate
cargo test --package hologram-core

# Run tests with output
cargo test -- --nocapture

# Run tests with specific thread count
cargo test -- --test-threads=1

# Run only unit tests
cargo test --lib

# Run only integration tests
cargo test --test integration_test

# Run only doc tests
cargo test --doc

# Run ignored tests
cargo test -- --ignored

# Run tests matching pattern
cargo test tensor
```

## Continuous Testing

### Watch Mode

```bash
# Install cargo-watch
cargo install cargo-watch

# Auto-run tests on file changes
cargo watch -x test

# Auto-run specific tests
cargo watch -x "test buffer"

# Run clippy + tests
cargo watch -x clippy -x test
```

### Pre-Commit Hooks

**File:** `.githooks/pre-commit`

```bash
#!/bin/bash
set -e

echo "Running pre-commit checks..."

# Format check
echo "Checking formatting..."
cargo fmt --all --check

# Clippy
echo "Running clippy..."
cargo clippy --workspace --all-targets --all-features -- -D warnings

# Tests
echo "Running tests..."
cargo test --workspace --all-features

echo "✅ All pre-commit checks passed!"
```

### CI Integration

**GitHub Actions** (see [ci.md](ci.md)):

- Run full test suite on every PR
- Run tests on multiple platforms (Linux, macOS, Windows)
- Generate coverage reports
- Fail build if coverage drops below threshold

## Performance Testing

### Benchmark Tests

**File:** `crates/core/benches/buffer_bench.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use hologram_core::{Executor, ops};

fn benchmark_vector_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_add");

    for size in [256, 1024, 4096, 16384].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let mut exec = Executor::new().unwrap();
                let mut a = exec.allocate::<f32>(size).unwrap();
                let mut b = exec.allocate::<f32>(size).unwrap();
                let mut c = exec.allocate::<f32>(size).unwrap();

                let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                a.copy_from_slice(&data).unwrap();
                b.copy_from_slice(&data).unwrap();

                b.iter(|| {
                    ops::math::vector_add(&exec, &a, &b, &mut c, size).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_vector_add);
criterion_main!(benches);
```

### Performance Regression Tests

```rust
#[test]
fn test_vector_add_performance() {
    use std::time::Instant;

    let mut exec = Executor::new().unwrap();
    let mut a = exec.allocate::<f32>(1_000_000).unwrap();
    let mut b = exec.allocate::<f32>(1_000_000).unwrap();
    let mut c = exec.allocate::<f32>(1_000_000).unwrap();

    let data: Vec<f32> = (0..1_000_000).map(|i| i as f32).collect();
    a.copy_from_slice(&data).unwrap();
    b.copy_from_slice(&data).unwrap();

    let start = Instant::now();
    ops::math::vector_add(&exec, &a, &b, &mut c, 1_000_000).unwrap();
    let duration = start.elapsed();

    // Should complete in < 1ms (adjust based on your performance requirements)
    assert!(duration.as_millis() < 1, "Vector addition too slow: {:?}", duration);
}
```

## Best Practices

### 1. Arrange-Act-Assert Pattern

```rust
#[test]
fn test_example() {
    // Arrange: Setup test data
    let mut exec = Executor::new().unwrap();
    let input = vec![1.0, 2.0, 3.0];

    // Act: Execute the operation
    let result = process_data(&exec, &input).unwrap();

    // Assert: Verify the outcome
    assert_eq!(result, vec![2.0, 4.0, 6.0]);
}
```

### 2. Test One Thing Per Test

```rust
// ✅ Good: Single, clear purpose
#[test]
fn test_buffer_returns_correct_size() {
    let mut exec = Executor::new().unwrap();
    let buffer = exec.allocate::<f32>(256).unwrap();
    assert_eq!(buffer.size(), 256);
}

// ❌ Bad: Testing multiple things
#[test]
fn test_buffer() {
    let mut exec = Executor::new().unwrap();
    let buffer = exec.allocate::<f32>(256).unwrap();
    assert_eq!(buffer.size(), 256);  // Testing size
    assert!(!buffer.is_empty());     // Testing emptiness
    buffer.copy_from_slice(&[1.0]).unwrap();  // Testing copy
    // Too much in one test!
}
```

### 3. Use Descriptive Failure Messages

```rust
#[test]
fn test_with_good_error_messages() {
    let result = compute_value();
    assert_eq!(
        result, 42,
        "compute_value() returned {}, expected 42",
        result
    );
}
```

### 4. Avoid Test Interdependencies

```rust
// ✅ Good: Each test is independent
#[test]
fn test_a() {
    let exec = Executor::new().unwrap();
    // Test A
}

#[test]
fn test_b() {
    let exec = Executor::new().unwrap();  // Fresh executor
    // Test B
}

// ❌ Bad: Tests share state
static mut SHARED_EXEC: Option<Executor> = None;

#[test]
fn test_a() {
    unsafe { SHARED_EXEC = Some(Executor::new().unwrap()); }
    // Test A modifies shared state
}

#[test]
fn test_b() {
    // Test B depends on test_a running first
    unsafe { SHARED_EXEC.as_mut().unwrap(); }
}
```

### 5. Use Test Helpers for Common Setup

```rust
fn setup_test_environment() -> (Executor, Buffer<f32>, Buffer<f32>) {
    let mut exec = Executor::new().unwrap();
    let a = exec.allocate::<f32>(256).unwrap();
    let b = exec.allocate::<f32>(256).unwrap();
    (exec, a, b)
}

#[test]
fn test_with_helper() {
    let (exec, mut a, mut b) = setup_test_environment();
    // Test logic
}
```

## Troubleshooting Test Failures

### Flaky Tests

**Problem:** Tests sometimes pass, sometimes fail

**Solutions:**
- Remove timing dependencies
- Use deterministic random seeds
- Avoid relying on external state
- Increase timeouts for slow operations
- Run tests with `--test-threads=1` to isolate concurrency issues

### Memory Leaks in Tests

**Problem:** Tests consume excessive memory

**Solutions:**
```rust
#[test]
fn test_with_cleanup() {
    let mut exec = Executor::new().unwrap();
    let buffer = exec.allocate::<f32>(1_000_000).unwrap();

    // ... test logic ...

    // Explicit cleanup
    drop(buffer);
    drop(exec);
}
```

### Slow Tests

**Problem:** Test suite takes too long

**Solutions:**
- Use smaller test data where possible
- Mark slow tests with `#[ignore]` and run separately
- Parallelize tests (default in cargo test)
- Cache expensive setup in `lazy_static`

## References

- [Rust Testing Documentation](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Proptest Book](https://altsysrq.github.io/proptest-book/intro.html)
- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [cargo-tarpaulin Documentation](https://github.com/xd009642/tarpaulin)
- [Benchmarking Specification](benchmarking.md)
