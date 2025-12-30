# Claude Development Guide for Hologramapp

This document provides guidelines for AI-assisted development (specifically Claude) when working on the hologramapp project. It ensures consistency, quality, and adherence to project standards.

## Core Policy: No Backwards Compatibility

**🚨 CRITICAL: We do NOT support backwards compatibility. If code is obsolete, delete it.**

When migrating to new architectures or patterns:

- **Delete old code** - Don't mark it as `#[deprecated]` or `#[ignore]`
- **Don't add feature gates** - No `#[cfg(feature = "old-api")]`
- **Don't write migration helpers** - Just delete the old API
- **Clean removal over preservation** - Dead code creates technical debt

**Examples**:

- ❌ `#[deprecated(since = "2.0.0", note = "Use new_api instead")]`
- ❌ `#[ignore = "Legacy architecture"]`
- ❌ `#[cfg(feature = "backwards-compat")]`
- ✅ **Just delete it**

This keeps the codebase clean, maintainable, and forward-focused.

## Core Principle: Ruthless Simplicity

**🎯 CRITICAL: Keep files and methods as simple as necessary. Be ruthless in implementing with simplicity.**

When writing code:

- **Favor simplicity over cleverness** - Simple code is maintainable code
- **Delete unnecessary abstractions** - Don't add layers "for future flexibility"
- **Inline small functions** - Don't abstract until you have 3+ use cases
- **Remove unused parameters** - If it's not used, delete it
- **Eliminate dead code paths** - No "just in case" branches

**Examples**:

- ❌ Creating a trait with one implementation "for future backends"
- ❌ Adding `Option<T>` parameters that are always `Some` or always `None`
- ❌ Writing 5-line helper functions used once
- ❌ Keeping commented-out code "for reference"
- ✅ **Direct implementation** - Solve the actual problem at hand
- ✅ **Refactor when needed** - Add abstraction when you have real use cases

**YAGNI (You Aren't Gonna Need It)**: Don't build for imaginary future requirements.

## Core Principle: No Placeholders or TODOs - EVER

**🚨 CRITICAL: NEVER LEAVE PLACEHOLDERS OR TODOs. IMPLEMENT EVERY FUNCTION COMPLETELY.**

This is the most important rule in this entire document:

- ❌ **NO** `todo!()` macros
- ❌ **NO** `unimplemented!()` macros
- ❌ **NO** `// TODO:` comments
- ❌ **NO** stub functions
- ❌ **NO** placeholder implementations
- ❌ **NO** "for simplicity" or "this will be implemented later" comments

✅ **ONLY** complete, production-ready, fully-tested implementations

**If you cannot implement a function completely RIGHT NOW, do NOT create it.**

See the full section on "No TODOs, No Stubs, No Unimplemented Functions" below for complete details.

## Core Principle: Performance-First Design

**🚨 CRITICAL: Optimize for O(1) complexity, zero-copy operations, parallel execution, and compile-time computation.**

Performance is a core architectural principle in hologramapp. Every implementation must prioritize:

### 1. Prefer O(1) Constant-Time Complexity

Always prefer constant-time operations over linear or higher complexity:

**❌ BAD - O(n) iteration:**
```rust
pub fn get_element(&self, index: usize) -> Option<&T> {
    self.items.iter().find(|item| item.id == index)  // O(n) search
}
```

**✅ GOOD - O(1) direct access:**
```rust
pub fn get_element(&self, index: usize) -> Option<&T> {
    self.items.get(index)  // O(1) array indexing
}
```

**Design patterns:**
- Use array indexing instead of iteration where possible
- Use hash maps for lookups instead of linear search
- Cache computed values instead of recalculating
- Prefer direct memory access over traversal

### 2. Prefer Zero-Copy Operations

Avoid copying data whenever possible. Use references, slices, and views:

**❌ BAD - Unnecessary copying:**
```rust
pub fn process_data(&self) -> Vec<f32> {
    let mut result = self.data.clone();  // Copies entire vector
    result.iter_mut().for_each(|x| *x *= 2.0);
    result
}
```

**✅ GOOD - Zero-copy with in-place mutation:**
```rust
pub fn process_data(&mut self) {
    self.data.iter_mut().for_each(|x| *x *= 2.0);  // No allocation
}
```

**✅ GOOD - Zero-copy views (Tensor operations):**
```rust
pub fn select(&self, dim: usize, index: usize) -> Result<Tensor<T>> {
    // Only modifies offset/shape/strides - no data copy
    let mut new_shape = self.shape.clone();
    new_shape.remove(dim);
    let new_offset = self.offset + index * self.strides[dim];
    Ok(Tensor { buffer: self.buffer.clone(), shape: new_shape, offset: new_offset, .. })
}
```

**Zero-copy patterns:**
- Use slices (`&[T]`) instead of `Vec<T>` for read-only access
- Modify in-place with `&mut` instead of returning new allocations
- Use view types (like Tensor select/narrow) that share underlying buffer
- Pass references instead of cloning large structures

### 3. Prefer Parallel Operations

Design for parallelism from the start:

**❌ BAD - Sequential processing:**
```rust
pub fn process_batch(&self, items: &[Item]) -> Vec<Result> {
    items.iter().map(|item| self.process(item)).collect()  // Sequential
}
```

**✅ GOOD - Parallel processing:**
```rust
pub fn process_batch(&self, items: &[Item]) -> Vec<Result> {
    items.par_iter().map(|item| self.process(item)).collect()  // Parallel via rayon
}
```

**✅ GOOD - GPU-parallel operations:**
```rust
// Operations compile to parallel generator calls executed on GPU
ops::math::vector_add(&exec, &a, &b, &mut c, n)?;  // Parallel across n elements
```

**Parallelism patterns:**
- Use `par_iter()` from rayon for CPU parallelism
- Design operations to be data-parallel (no dependencies between elements)
- Use GPU operations via hologram-core for massive parallelism
- Avoid sequential bottlenecks and data dependencies

### 4. Prefer Compile-Time Over Runtime Complexity

Move complexity to compile time whenever possible:

**❌ BAD - Runtime computation:**
```rust
pub fn get_class_count(&self) -> usize {
    96  // Hardcoded but computed at runtime
}

pub fn allocate_buffer(&self) -> Vec<u8> {
    vec![0; 48 * 256]  // Multiplication at runtime
}
```

**✅ GOOD - Compile-time constants:**
```rust
pub const CLASS_COUNT: usize = 96;
pub const PAGES: usize = 48;
pub const BYTES_PER_PAGE: usize = 256;
pub const TOTAL_ELEMENTS: usize = PAGES * BYTES_PER_PAGE;  // Computed at compile time

pub fn allocate_buffer(&self) -> Vec<u8> {
    vec![0; TOTAL_ELEMENTS]  // Constant known at compile time
}
```

**✅ GOOD - Generic const parameters:**
```rust
pub struct FixedBuffer<const N: usize> {
    data: [u8; N],  // Size known at compile time
}

impl<const N: usize> FixedBuffer<N> {
    pub const fn new() -> Self {
        Self { data: [0; N] }  // Compile-time initialization
    }
}
```

**✅ GOOD - Type-level computation:**
```rust
// Use type system to encode invariants checked at compile time
pub struct Validated<T>(T);  // Type guarantees validation happened

impl<T> Validated<T> {
    pub fn new(value: T) -> Result<Self> {
        validate(&value)?;
        Ok(Validated(value))  // Type system ensures validation
    }

    pub fn get(&self) -> &T {
        &self.0  // No runtime check needed - type guarantees validity
    }
}
```

**Compile-time patterns:**
- Use `const` for all constants instead of runtime variables
- Use `const fn` for compile-time computation
- Use generic const parameters for fixed-size structures
- Use the type system to encode invariants
- Compute lookup tables at compile time with `const` or `lazy_static`

### Performance Decision Framework

When implementing any feature, ask:

1. **Can this be O(1)?** If not, why not? Document the reason.
2. **Can this be zero-copy?** Use views/slices instead of clones.
3. **Can this be parallel?** Design for data parallelism.
4. **Can this be compile-time?** Use `const`, generic parameters, or types.

### Examples of Performance-First Design

**Tensor operations:**
```rust
// ✅ Zero-copy views
let sliced = tensor.select(0, 2)?;       // No data copy, just offset change
let narrowed = tensor.narrow(1, 0, 4)?;  // No data copy, just shape change

// ✅ O(1) shape operations
let transposed = tensor.transpose()?;     // Only swaps strides, no data movement
```

**Parallel execution:**
```rust
// ✅ GPU-parallel operations
ops::math::vector_add(&exec, &a, &b, &mut c, 1_000_000)?;  // Parallel across 1M elements
```

**Compile-time safety:**
```rust
// ✅ Type-level guarantees
pub fn process(tensor: &Tensor<f32>) -> Result<()> {
    // Type system guarantees f32, no runtime type checking needed
}
```

## Naming Conventions

### Mathematical Terms

**🚨 CRITICAL: Use correct mathematical terminology.**

- **primorial** (NOT "primordial") - The product of the first n primes (e.g., 2×3×5 = 30)
  - Field names: `primorial`, `chunk.primorial`, `block.primorial`
  - This is a mathematical term from number theory
  - "primordial" is incorrect - it means "existing from the beginning"

**Examples**:
- ✅ `pub primorial: u64` - Correct mathematical term
- ❌ `pub primordial: u64` - Wrong word (means ancient/original)

When naming fields and variables related to primes and their products, always use the correct mathematical terminology.

## Project Vision: General Compute Acceleration

**🎯 PRIMARY GOAL: High-performance compute through canonical form compilation**

Hologramapp provides general compute acceleration by compiling operations to their canonical forms. This enables:

- **Canonical Compilation** - Operations compiled to optimal geometric representation
- **Hologram Compiler-Powered** - Pattern-based canonicalization reduces operations to minimal form
- **Lowest Latency** - Canonical forms enable fastest possible execution
- **Universal Compute** - General-purpose acceleration, not domain-specific

### Key Architectural Principles

1. **Canonical Form Compilation**: All operations reduced to canonical geometric representation
2. **Hologram Compiler Canonicalization**: Pattern-based rewriting (H²=I, X²=I, etc.) minimizes operation count
3. **Generator-Based Execution**: 7 fundamental generators (mark, copy, swap, merge, split, quote, evaluate)
4. **Performance through Simplification**: Fewer operations = lower latency
5. **Performance-First Design**: O(1) complexity, zero-copy operations, parallel execution, compile-time computation (see "Performance-First Design" section)
6. **🚨 ABSOLUTELY NO CPU FALLBACKS**: All operations MUST be implemented using hologram-compiler generators. If primitives are missing, extend hologram-compiler itself - never fall back to CPU implementations

## Core Principle: Task Completion Discipline

**🚨 CRITICAL: Complete every task fully. No shortcuts, no excuses.**

When working on a task:

1. **Never stop before completion** - If you start a task, finish it completely
2. **No excuses about constraints** - Do not cite "time constraints", "token limits", or "efficiency" as reasons to shortcut work
3. **Do the work sequentially** - If there are 50 files to update, update all 50 files one by one
4. **No "script-based approaches" to avoid work** - Actually do the edits, don't suggest automation as an excuse
5. **Finish what you start** - If you create a todo list with 26 items, complete all 26 items

### Anti-Patterns to Avoid

❌ "Given the time constraints, let me create a more efficient approach..."
❌ "Due to token limits, I'll write a script instead..."
❌ "To save time, let me batch these changes..."
❌ "I'll stub this out for now..."

✅ **Just do the work.** Edit every file. Complete every function. Finish every test.

### Why This Matters

Incomplete work compounds:

- Leaves the codebase in a broken state
- Creates technical debt that must be fixed later
- Wastes the user's time when they have to ask again
- Undermines trust in the development process

**If you're asked to complete 10 operations rewrites, rewrite all 10 operations. Period.**

### Completion Criteria

**A feature is not complete if the workspace tests don't pass or if there are compiler warnings.**

Before marking any task as complete:

- Run `cargo test --workspace` and ensure all tests pass
- Fix any test failures before moving to the next task
- Run `cargo clippy --workspace -- -D warnings` and fix all warnings
- Ensure the code compiles without any warnings (`cargo build --workspace`)
- Tests are the ultimate source of truth for correctness

## Core Principle: No TODOs, No Stubs, No Unimplemented Functions

**🚨 CRITICAL: NEVER LEAVE PLACEHOLDERS OR TODOs. IMPLEMENT THE FUNCTIONS COMPLETELY.**

**This is non-negotiable. Every function you write must be fully implemented, tested, and working.**

When writing code:

1. **No TODO comments** - If something needs to be done, do it now. Don't leave markers for "future work"
2. **No stub functions** - Every function must have a complete, working implementation
3. **No unimplemented!()** - Never use `unimplemented!()`, `todo!()`, or `panic!("not yet implemented")`
4. **No placeholder implementations** - Every function must do what its documentation says it does
5. **No placeholder comments** - No "this will be implemented later" or "for simplicity" comments
6. **No partial implementations** - If you start a function, finish it completely before moving on
7. **Minimize function count** - Don't create functions until you need them. Implement only what's required now

**If you cannot implement a function completely right now, DO NOT CREATE THE FUNCTION AT ALL.**

### Anti-Patterns to Avoid

❌ `// TODO: implement this later`
❌ `// FIXME: this is a temporary hack`
❌ `unimplemented!("will add this next")`
❌ `todo!("implement when we have time")`
❌ `panic!("not yet implemented")`
❌ Creating 20 function signatures with empty bodies "for future use"

✅ **Implement the function completely now**
✅ **If you can't implement it now, don't create it**
✅ **Only create functions you can fully implement immediately**

### Why This Matters

TODOs and stubs create confusion and technical debt:

- **Breaks the build** - Code that panics at runtime is broken code
- **False sense of progress** - Stub functions look complete but do nothing
- **Wastes time** - Future developers must figure out what was intended
- **Hides real work** - A codebase full of stubs obscures what's actually done
- **Violates contracts** - Functions that panic violate their type signatures

### Examples

**Bad - Stub function:**
```rust
pub fn process_data(input: &[f32]) -> Vec<f32> {
    // TODO: implement data processing
    unimplemented!()
}
```

**Good - Complete implementation:**
```rust
pub fn process_data(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x * 2.0).collect()
}
```

**Bad - Partial implementation with TODO:**
```rust
pub fn validate_input(data: &Data) -> Result<()> {
    // Check basic fields
    if data.name.is_empty() {
        return Err(Error::InvalidInput("name required"));
    }
    // TODO: add more validation
    Ok(())
}
```

**Good - Complete validation:**
```rust
pub fn validate_input(data: &Data) -> Result<()> {
    if data.name.is_empty() {
        return Err(Error::InvalidInput("name required"));
    }
    if data.value < 0.0 {
        return Err(Error::InvalidInput("value must be non-negative"));
    }
    if data.id == 0 {
        return Err(Error::InvalidInput("id must be non-zero"));
    }
    Ok(())
}
```

### Allowed Exceptions

The ONLY acceptable use of these patterns is:

1. **During active development** - While you're actively writing code in the same session
2. **Must be completed before commit** - All TODOs removed before committing
3. **Never committed to main** - No TODOs, stubs, or unimplemented!() in version control

**If you can't implement a function completely right now, don't create the function at all.**

## Core Principle: Test-Driven Development

**🚨 CRITICAL: No feature is complete without comprehensive tests.**

Every piece of code written must include:

1. **Unit tests** for individual functions and methods
2. **Integration tests** for component interactions
3. **Property-based tests** (using `proptest`) for mathematical invariants
4. **Documentation tests** in doc comments

### Test Coverage Requirements

- **Minimum 80% code coverage** for all crates
- **100% coverage** for hologram-compiler canonicalization rules and generator compilation
- **All public APIs** must have examples in doc comments that serve as tests
- **All error paths** must be tested
- **Canonicalization correctness** tests for all rewrite rules

## Development Workflow

### 1. Before Writing Code

1. **Understand the requirement** fully
2. **Review existing architecture** and patterns
3. **Write test cases first** (TDD approach)
4. **Design the API** (function signatures, types, traits)

### 2. Writing Code

1. **Implement incrementally** - small, testable chunks
2. **Run tests frequently** - `cargo test` after each change
3. **Document as you go** - doc comments with examples
4. **Follow Rust idioms** and best practices

### 3. After Writing Code

1. **Run full test suite**: `cargo test --workspace` - Fix all test failures
2. **Run clippy**: `cargo clippy --workspace -- -D warnings` - Fix all warnings (zero warnings required)
3. **Verify clean build**: `cargo build --workspace` - Ensure no compiler warnings
4. **Check formatting**: `cargo fmt --check` - Fix any formatting issues
5. **Verify documentation**: `cargo doc --no-deps --workspace`
6. **Update integration tests** if APIs changed

**CRITICAL**: All tests must pass and all warnings must be fixed before proceeding to the next task.

## Testing Standards

### Unit Tests

Every module must have a `#[cfg(test)]` section:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Arrange
        let input = create_test_input();

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result, expected_value);
    }

    #[test]
    fn test_error_handling() {
        let invalid_input = create_invalid_input();
        assert!(function_under_test(invalid_input).is_err());
    }
}
```

### Property-Based Tests

For canonicalization and class operations, use `proptest`:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_class_index_in_bounds(class in 0..96u8) {
        // All class operations stay in valid range
        let result = perform_class_operation(class);
        prop_assert!(result < 96);
    }

    #[test]
    fn test_canonicalization_idempotent(expr: String) {
        // Canonicalizing twice gives same result
        let once = Canonicalizer::canonicalize(&expr)?;
        let twice = Canonicalizer::canonicalize(&once)?;
        prop_assert_eq!(once, twice);
    }

    #[test]
    fn test_rewrite_preserves_semantics(circuit: String) {
        // Canonical form evaluates to same result
        let original = evaluate_circuit(&circuit)?;
        let canonical = Canonicalizer::canonicalize(&circuit)?;
        let canonical_result = evaluate_circuit(&canonical)?;
        prop_assert_eq!(original, canonical_result);
    }
}
```

### Integration Tests

Create `tests/` directories for integration tests:

```rust
// tests/integration_test.rs
use hologram_core::{ops, Executor, Result};

#[test]
fn test_full_operation_flow() -> Result<()> {
    let exec = Executor::new()?;

    // Allocate buffers
    let mut input = exec.allocate::<f32>(256)?;
    let mut output = exec.allocate::<f32>(256)?;

    // Setup data
    let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
    input.copy_from_slice(&data)?;

    // Execute operation
    ops::math::vector_add(&exec, &input, &input, &mut output, 256)?;

    // Verify results
    let result = output.to_vec()?;
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 2.0);

    Ok(())
}
```

### Documentation Tests

All public APIs must have examples in doc comments:

````rust
/// Compute element-wise sum of two vectors
///
/// # Example
///
/// ```
/// use hologram_core::{ops, Executor};
///
/// let exec = Executor::new().unwrap();
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

## Code Organization Standards

### Documentation Organization

**All `.md` documentation files should be stored in the `docs/` directory**, with two exceptions:

- `README.md` - Project overview at repository root
- `CLAUDE.md` - This development guide at repository root

**🚨 CRITICAL: All new documentation MUST be placed in a subdirectory under `/docs`. NEVER create documentation files directly in the `/docs` root.**

**Documentation Subdirectories:**

All documentation must be organized into feature-specific subdirectories:

- `docs/architecture/` - Core architecture, specifications, formal models
- `docs/performance/` - Benchmarking, optimization, profiling, tracing
- `docs/ffi/` - FFI API reference and development guides
- `docs/integrations/` - External library integrations (Candle, DLPack, PyTorch)
- `docs/model-server/` - Model server implementation
- `docs/migration/` - Migration guides and status
- `docs/kernels/` - Kernel development guides, quantum algorithms
- `docs/memory/` - Memory architecture
- `docs/sdk/` - SDK documentation and guides
- `docs/wasm/` - WASM deployment
- `docs/pytorch/` - PyTorch integration details
- `docs/webgpu/` - WebGPU implementation
- `docs/simd_gpu/` - SIMD/GPU optimization
- `docs/notes/` - Development notes and miscellaneous documentation

**When creating new documentation:**

1. Determine which feature area the documentation relates to
2. Place the file in the appropriate subdirectory (e.g., `docs/performance/NEW_DOC.md`)
3. If the feature doesn't have a subdirectory, create one (e.g., `docs/new-feature/`)
4. **NEVER** place documentation directly in `/docs/` root

**Examples:**

- ✅ `docs/performance/NEW_OPTIMIZATION.md` - Correct
- ✅ `docs/ffi/NEW_BINDING.md` - Correct
- ✅ `docs/new-feature/IMPLEMENTATION.md` - Correct (new subdirectory)
- ❌ `docs/NEW_DOC.md` - WRONG (not in subdirectory)

**Only files allowed in `/docs` root:**

- `FUTURE_PROMPTS.md` - Special file for future task tracking
- `README.md` - Documentation index/overview

### Common Utility Functions

**🚨 CRITICAL: Write common utility functions in the `hologram-common` crate when they can be used across multiple crates.**

When you identify functionality that is or could be used by multiple crates in the platform:

- **Place it in `hologram-common`** - Centralize shared utilities
- **Avoid duplication** - Don't copy-paste utility functions across crates
- **Make it reusable** - Design utilities to be generic and composable
- **Document thoroughly** - Common utilities need clear documentation with examples

**Examples of common utilities:**

- ✅ Error types used across multiple crates
- ✅ Type conversions and validation helpers
- ✅ Mathematical utilities (shape calculations, broadcasting logic)
- ✅ Common data structures (buffers, memory layouts)
- ✅ Shared traits and interfaces
- ✅ Logging and tracing utilities

**Anti-patterns to avoid:**

- ❌ Copying the same utility function into multiple crates
- ❌ Implementing similar functionality differently in each crate
- ❌ Creating crate-specific utilities when they could be generalized
- ❌ Leaving utilities in one crate when other crates need them

**When to use `hologram-common`:**

1. **The function is already used in 2+ crates** - Move it to common immediately
2. **The function could logically be used elsewhere** - Consider placing it in common from the start
3. **The function is a general utility** - Prefer common over crate-specific placement
4. **You're about to copy-paste code** - Stop and move it to common instead

**Example:**

```rust
// ❌ Bad: Same utility in multiple crates
// In hologram-core/src/utils.rs
pub fn validate_shape(shape: &[usize]) -> Result<()> {
    // validation logic
}

// In hologram-onnx/src/utils.rs
pub fn validate_shape(shape: &[usize]) -> Result<()> {
    // same validation logic (duplicated!)
}

// ✅ Good: Shared utility in hologram-common
// In hologram-common/src/shape.rs
pub fn validate_shape(shape: &[usize]) -> Result<()> {
    // validation logic (single source of truth)
}

// In hologram-core (uses common)
use hologram_common::shape::validate_shape;

// In hologram-onnx (uses common)
use hologram_common::shape::validate_shape;
```

**Benefits:**

- **DRY (Don't Repeat Yourself)** - Single source of truth for common functionality
- **Consistency** - All crates use the same validated implementation
- **Maintainability** - Fix bugs once, benefit everywhere
- **Testability** - Test common utilities once, comprehensively
- **Discoverability** - Developers know where to find shared utilities

### Module Structure

```
crate_name/
└── src/
    ├── lib.rs           # Public API and re-exports
    ├── types.rs         # Core type definitions
    ├── error.rs         # Error types (using thiserror)
    ├── module1.rs       # Focused functionality
    └── module2.rs
```

### Naming Conventions

- **Types**: `PascalCase` (e.g., `ClassIndex`, `GeneratorCall`)
- **Functions**: `snake_case` (e.g., `canonicalize`, `compile_circuit`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `CLASS_COUNT`, `MAX_GENERATORS`)
- **Traits**: `PascalCase`, descriptive (e.g., `Canonicalizer`, `Executor`)

### Error Handling

Use `thiserror` for error types:

```rust
#[derive(Debug, thiserror::Error)]
pub enum MyError {
    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("Operation failed: {0}")]
    OperationFailed(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, MyError>;
```


## Common Patterns

### Safe Construction with Validation

```rust
pub struct MyType {
    value: u8,
}

impl MyType {
    /// Create with validation
    pub fn new(value: u8) -> Result<Self> {
        if value >= LIMIT {
            return Err(MyError::InvalidValue);
        }
        Ok(Self { value })
    }

    /// Create without validation (unsafe)
    ///
    /// # Safety
    ///
    /// Caller must ensure value < LIMIT
    pub const unsafe fn new_unchecked(value: u8) -> Self {
        Self { value }
    }
}
```

### Thread-Safe State

Use parking_lot for locks:

```rust
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;

#[derive(Clone)]
pub struct SharedState {
    data: Arc<RwLock<Data>>,
}

impl SharedState {
    pub fn read(&self) -> Data {
        *self.data.read()
    }

    pub fn write(&self, value: Data) {
        *self.data.write() = value;
    }
}
```

### Builder Pattern for Complex Structs

**Use the Builder pattern where appropriate for ergonomic APIs, but don't overuse it.**

The Builder pattern is useful for structs with multiple optional fields or complex configuration. However, follow the YAGNI principle - don't add builders for simple structs.

**When to use the Builder pattern:**

- ✅ Structs with 4+ fields, especially if some are optional
- ✅ Public API configuration objects where ergonomics matter
- ✅ Complex initialization requiring validation across multiple fields
- ✅ When you need incremental construction with method chaining

**When NOT to use the Builder pattern:**

- ❌ Simple structs with 1-3 required fields - use direct construction
- ❌ Internal implementation details - prefer simplicity
- ❌ When `Default` trait or simple constructor is clearer
- ❌ Don't add complexity for imaginary future needs (YAGNI)

**Good example - Builder is appropriate:**

```rust
// Complex configuration with many optional fields
pub struct ExecutorConfig {
    device_id: u32,
    max_buffers: usize,
    enable_profiling: bool,
    cache_size: Option<usize>,
    timeout_ms: Option<u64>,
    backend: Backend,
}

#[derive(Default)]
pub struct ExecutorConfigBuilder {
    device_id: u32,
    max_buffers: usize,
    enable_profiling: bool,
    cache_size: Option<usize>,
    timeout_ms: Option<u64>,
    backend: Option<Backend>,
}

impl ExecutorConfigBuilder {
    pub fn new() -> Self {
        Self {
            device_id: 0,
            max_buffers: 1024,
            enable_profiling: false,
            cache_size: None,
            timeout_ms: None,
            backend: None,
        }
    }

    pub fn device_id(mut self, device_id: u32) -> Self {
        self.device_id = device_id;
        self
    }

    pub fn max_buffers(mut self, max_buffers: usize) -> Self {
        self.max_buffers = max_buffers;
        self
    }

    pub fn enable_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }

    pub fn cache_size(mut self, size: usize) -> Self {
        self.cache_size = Some(size);
        self
    }

    pub fn timeout_ms(mut self, timeout: u64) -> Self {
        self.timeout_ms = Some(timeout);
        self
    }

    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn build(self) -> Result<ExecutorConfig> {
        // Validate configuration
        if self.max_buffers == 0 {
            return Err(Error::InvalidConfig("max_buffers must be > 0"));
        }

        Ok(ExecutorConfig {
            device_id: self.device_id,
            max_buffers: self.max_buffers,
            enable_profiling: self.enable_profiling,
            cache_size: self.cache_size,
            timeout_ms: self.timeout_ms,
            backend: self.backend.unwrap_or(Backend::default()),
        })
    }
}

// Usage:
let config = ExecutorConfigBuilder::new()
    .device_id(1)
    .max_buffers(2048)
    .enable_profiling(true)
    .cache_size(512)
    .build()?;
```

**Bad example - Builder is unnecessary:**

```rust
// ❌ Don't use Builder for simple structs
pub struct Point {
    x: f32,
    y: f32,
}

// ❌ Overkill - just use a simple constructor
pub struct PointBuilder { /* ... */ }

// ✅ Simple constructor is clearer
impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

// Usage:
let point = Point::new(1.0, 2.0);  // Clear and simple
```

**Using derive_builder for simple cases:**

For straightforward builders without complex validation, consider using the `derive_builder` crate:

```rust
use derive_builder::Builder;

#[derive(Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct CompilerOptions {
    #[builder(default = "3")]
    optimization_level: u8,

    #[builder(default)]
    inline_threshold: usize,

    target_arch: String,

    #[builder(default)]
    debug_info: bool,
}

impl CompilerOptionsBuilder {
    fn validate(&self) -> Result<(), String> {
        if let Some(level) = self.optimization_level {
            if level > 3 {
                return Err("optimization_level must be 0-3".to_string());
            }
        }
        Ok(())
    }
}

// Usage:
let options = CompilerOptionsBuilder::default()
    .target_arch("x86_64".to_string())
    .optimization_level(2)
    .build()?;
```

**Key principles:**

1. **Validate in build()** - Always validate the configuration in the `build()` method
2. **Provide sensible defaults** - Make optional fields truly optional with good defaults
3. **Keep it simple** - Don't add builder complexity unless it genuinely improves ergonomics
4. **Document required fields** - Make it clear which fields must be set
5. **Return Result from build()** - Allow validation errors to be reported properly

## Performance Considerations

**See the "Performance-First Design" section above for comprehensive performance guidelines.**

Key performance principles (detailed above):
1. **Prefer O(1) complexity** - Use direct access instead of iteration
2. **Prefer zero-copy operations** - Use views/slices instead of clones
3. **Prefer parallel operations** - Design for data parallelism
4. **Prefer compile-time complexity** - Use `const`, generic parameters, type system

### Avoid Unnecessary Allocations

```rust
// ❌ Bad: allocates for every call
fn process(items: Vec<Item>) -> Vec<Result> {
    items.into_iter().map(|i| transform(i)).collect()
}

// ✅ Good: use iterators (zero allocation)
fn process(items: &[Item]) -> impl Iterator<Item = Result> + '_ {
    items.iter().map(|i| transform(i))
}

// ✅ Better: in-place mutation (zero copy)
fn process_inplace(items: &mut [Item]) {
    items.iter_mut().for_each(|item| transform_inplace(item))
}
```

### Use const where possible

```rust
// ✅ Good: compile-time constants
pub const PAGES: u32 = 48;
pub const BYTES_PER_PAGE: u32 = 256;
pub const TOTAL_ELEMENTS: usize = (PAGES * BYTES_PER_PAGE) as usize;

// ✅ Good: const fn for compile-time computation
pub const fn compute_offset(page: usize, byte: usize) -> usize {
    page * BYTES_PER_PAGE as usize + byte
}
```

### Benchmark Critical Paths

Create `benches/` directory for criterion benchmarks:

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_operation(c: &mut Criterion) {
    c.bench_function("operation_name", |b| {
        b.iter(|| {
            // Benchmark critical operation
        });
    });
}

criterion_group!(benches, benchmark_operation);
criterion_main!(benches);
```

## Documentation Standards

### Module-Level Documentation

Every module should have header documentation:

````rust
//! # Module Name - Brief Description
//!
//! Longer description of what this module provides.
//!
//! ## Example
//!
//! ```
//! use crate_name::module_name::Type;
//!
//! let value = Type::new();
//! ```
````

### Type Documentation

```rust
/// A class index in the 96-class geometric system
///
/// Represents one of 96 canonical classes in the geometric class system.
/// Each class provides a canonical geometric representation.
#[derive(Debug, Clone, Copy)]
pub struct ClassIndex(u8);
```

### Function Documentation

Use standard sections:

````rust
/// Brief description of what the function does
///
/// # Arguments
///
/// * `arg1` - Description of arg1
/// * `arg2` - Description of arg2
///
/// # Returns
///
/// Description of return value
///
/// # Errors
///
/// Returns `Err` if...
///
/// # Examples
///
/// ```
/// use crate_name::function_name;
///
/// let result = function_name(arg1, arg2)?;
/// assert_eq!(result, expected);
/// ```
pub fn function_name(arg1: Type1, arg2: Type2) -> Result<ReturnType> {
    // Implementation
}
````

## Codebase Architecture

### Crate Structure

The project is organized into primary crates:

#### **hologram-core** - Operations Library

- **High-Level Operations**: Pre-built operations that compile to canonical kernels
  - `ops::math`: Arithmetic operations (add, sub, mul, div, min, max, abs, neg, relu, etc.)
  - `ops::reduce`: Reductions (sum, min, max)
  - `ops::activation`: Neural network activations (sigmoid, tanh, gelu, softmax)
  - `ops::loss`: Loss functions (mse, cross_entropy, binary_cross_entropy)
  - `ops::linalg`: Linear algebra (gemm, matvec)
  - `ops::memory`: Memory operations (copy, fill)
- **Executor**: High-level interface for operation execution
- **Buffer<T>**: Type-safe memory abstraction
- **Tensor<T>**: Multi-dimensional arrays with PyTorch-like operations
- **Kernel Compilation**: Operations compile to canonical generator sequences

**Key responsibility**: Provide composable operations that leverage canonical compilation

See `crates/hologram-core/src/` for operation implementations

### Key Concepts

#### Execution Model

The compilation and execution flow:

```
Application Code
    ↓ calls
Hologram Core Operations (ops::math, ops::reduce, etc.)
    ↓ compiles via
Canonical Compilation (pattern rewriting)
    ↓ produces
Canonical Generator Sequence (minimal operation count)
    ↓ executes as
Optimized Kernel (lowest latency)
```

**Key principle**: Operations → Canonical Form → Fast Execution

Example:

```rust
// Application calls high-level operation
ops::math::vector_add(&exec, &a, &b, &mut c, n)?;

// ↓ Compiles to canonical circuit
"merge@c[0..N]"  // Addition as merge generator

// ↓ Canonicalization reduces if possible
// Pattern rewriting applies (e.g., merge·merge → merge)

// ↓ Executes minimal canonical form
GeneratorCall::MergeRange { start: 0, end: N, variant: Add }
```

#### Buffer Allocation and Operations

```rust
use hologram_core::{Executor, ops, Result};

let exec = Executor::new()?; // Creates executor

// Allocate buffers
let mut input = exec.allocate::<f32>(1024)?;
let mut output = exec.allocate::<f32>(1024)?;

// Copy data to buffers
input.copy_from_slice(&data)?;

// Execute operations (compiled to canonical kernels)
ops::math::vector_add(&exec, &input, &input, &mut output, 1024)?;

// Copy results back
let results = output.to_vec()?;
```

#### Tensor Operations

```rust
use hologram_core::{Tensor, ops};

// Create tensor from buffer
let tensor_a = Tensor::<f32>::from_buffer(buf_a, vec![4, 8])?;
let tensor_b = Tensor::<f32>::from_buffer(buf_b, vec![8, 3])?;

// Matrix multiplication
let result = tensor_a.matmul(&exec, &tensor_b)?;

// Zero-copy operations
let sliced = tensor.select(0, 2)?;       // Select index along dimension
let narrowed = tensor.narrow(1, 0, 4)?;  // Narrow range
let transposed = tensor.transpose()?;     // 2D transpose

// Check broadcasting compatibility
assert!(tensor_a.is_broadcast_compatible_with(&tensor_b));
```

#### Core Operations Pattern

All operations follow the pattern: `ops::module::function(&exec, inputs, &mut outputs, sizes)`

```rust
// Element-wise operations
ops::math::vector_add(&exec, &a, &b, &mut c, n)?;
ops::math::relu(&exec, &input, &mut output, n)?;

// Activations
ops::activation::sigmoid(&exec, &x, &mut y, n)?;
ops::activation::softmax(&exec, &input, &mut output, n)?;

// Reductions (output needs 3 elements for temporaries)
let mut sum_out = exec.allocate::<f32>(3)?;
ops::reduce::sum(&exec, &input, &mut sum_out, n)?;

// Loss functions (output needs 3 elements)
let mut loss = exec.allocate::<f32>(3)?;
ops::loss::mse(&exec, &pred, &target, &mut loss, n)?;
```

### Development Patterns

#### Writing New Operations

Operations in hologram-core compile to canonical generator sequences:

```rust
pub fn my_op<T: bytemuck::Pod>(
    exec: &Executor,
    input: &Buffer<T>,
    output: &mut Buffer<T>,
    n: usize,
) -> Result<()> {
    // Build circuit expression for this operation
    let circuit = format!("merge@c[0..{}]", calculate_class_range(n));

    // Compile to canonical form
    let compiled = compile_circuit(&circuit)?;

    // compiled.calls contains minimal generator sequence:
    // - Original: 50 ops
    // - Canonical: 12 ops (76% reduction)
    // - Reduction enables lowest latency execution

    // Execute canonical generator sequence
    exec.execute_compiled(&compiled)?;

    Ok(())
}
```

**Key steps**:

1. Express operation as circuit
2. Compile circuit → canonical generator sequence
3. Canonicalization automatically reduces operation count
4. Execute minimal canonical form
5. Add comprehensive tests verifying canonicalization

**Performance benefit**: Canonicalization reduces 4-8x operations for typical circuits, directly improving latency.

See existing operations in `hologram-core/src/ops/` for complete examples.

#### ONNX Operation Schemas

ONNX operations are implemented as Atlas kernel schemas that compile to JSON and then to ISA:

**Schema Location**: `/workspace/schemas/onnx/`

**Categories**:
- `core/`: Basic math (add, sub, mul, div, gemm, matmul) - 6 files
- `activation/`: Activation functions (relu, sigmoid, tanh, gelu, softmax, etc.) - 9 files
- `normalization/`: Normalization layers (batch_norm, layer_norm, instance_norm) - 3 files
- `reduction/`: Reduction ops (reduce_sum, reduce_mean, reduce_max, reduce_min) - 4 files
- `pooling/`: Pooling operations (averagepool, global_average_pool) - 2 files
- `shape/`: Shape manipulation (transpose, concat, reshape, flatten, squeeze, unsqueeze, slice, gather) - 8 files
- `conv/`: Convolution operations (conv.py) - 1 file
- `other/`: Utility operations (constant, argmax, shape, range, cast) - 5 files

**Total**: 38 ONNX operation schemas

**Schema Pattern**:
```python
"""
ONNX {Operation} Operation

ONNX Spec: https://onnx.ai/onnx/operators/onnx__{Op}.html

Brief description of the operation.

Shapes:
  - Input: shape description
  - Output: shape description
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def operation_name(
    Input: DeviceArray[f32],
    Output: DeviceArray[f32],
    n: u32
):
    """
    ONNX {Operation}: Output = func(Input)

    Parameters:
    - Input: Input tensor
    - Output: Output tensor
    - n: Total number of elements
    """
    idx = get_global_id()

    if idx < n:
        Output[idx] = transform(Input[idx])
```

**Compilation**:
```bash
cd /workspace/schemas/stdlib
python3 atlas_compile.py ../onnx/core/add.py
# Creates: ../onnx/core/add.json
```

**Important Constraints**:
- Atlas compiler supports: variables, constants, binary ops, comparisons, subscripts, calls, for loops
- Does NOT support: Python lists, dicts, complex data structures
- Keep operations simple and parallel (element-wise or simple matrix ops)
- Multi-pass operations (like softmax) split into separate kernels

**Runtime Integration**:
ONNX schemas → JSON → ISA Programs → hologram-core operations → canonical circuits

#### ONNX Operator Implementation Guidelines

**🚨 CRITICAL: All ONNX operators MUST implement the Numeric trait for type-generic support.**

When implementing ONNX operators in `/workspace/hologram-sdk/rust/hologram-onnx/src/ops/`:

1. **Use Generic Implementation Pattern**: Create a generic method using the `Numeric` trait bound
   ```rust
   async fn execute_generic<T>(&self, exec: &mut Executor, inputs: &[&OnnxTensor]) -> Result<Vec<OnnxTensor>>
   where
       T: crate::types::Numeric,
   {
       let input_data = inputs[0].read_data::<T>(exec).await?;
       // ... operate on data ...
       let output_data: Vec<T> = input_data.iter().map(|&x| /* operation */).collect();
       let output = OnnxTensor::from_data_typed(exec, &output_data, shape).await?;
       Ok(vec![output])
   }
   ```

2. **Dispatch on DataType**: Use pattern matching to dispatch to the generic implementation
   ```rust
   match inputs[0].dtype() {
       DataType::Float32 => self.execute_generic::<f32>(exec, inputs).await,
       DataType::Float64 => self.execute_generic::<f64>(exec, inputs).await,
       DataType::Int32 => self.execute_generic::<i32>(exec, inputs).await,
       DataType::Int64 => self.execute_generic::<i64>(exec, inputs).await,
       // ... all numeric types ...
       dtype => Err(OnnxError::UnsupportedType(dtype as i32)),
   }
   ```

3. **Support Type Promotion for Math Operators**: Binary operators should support mixed Float32/Int32 types
   ```rust
   if dtype_a != dtype_b {
       if dtype_a == DataType::Float32 && dtype_b == DataType::Int32 {
           // Cast Int32 to Float32 and execute
           let b_f32 = cast_to_f32(exec, inputs[1]).await?;
           return self.execute_f32(exec, &[inputs[0], &b_f32]).await;
       }
       // Handle reverse case...
   }
   ```

4. **Supported Numeric Types**: All operators should support:
   - Floating-point: `f32`, `f64`, `f16`, `bf16`
   - Signed integers: `i8`, `i16`, `i32`, `i64`
   - Unsigned integers: `u8`, `u16`, `u32`, `u64`

5. **Boolean Output**: For comparison operators (Equal, Greater, Less, etc.):
   - Output as `u8` tensor with values 0 (false) or 1 (true)
   - ONNX boolean tensors are represented as `u8` because `bool` is not a `Numeric` type

**Examples**:
- ✅ `AddOp`, `SubOp`, `MulOp`, `DivOp` - All support Numeric trait with type promotion
- ✅ `EqualOp` - Supports Numeric trait, outputs `u8` for boolean results
- ✅ `ReluOp`, `SigmoidOp` - Support Numeric trait for all numeric types

**Benefits**:
- **Type flexibility**: One implementation works for all numeric types
- **Consistency**: All operators follow the same pattern
- **Type safety**: Rust's type system ensures correctness
- **Performance**: Optimized GPU paths for `f32`, generic CPU fallback for other types

#### Testing Strategy

- **Unit tests**: In each module's `#[cfg(test)]` section
- **Integration tests**: In `tests/integration_test.rs`
- **Property tests**: Use `proptest` for mathematical invariants
- **Run frequently**: `cargo test --workspace`

### Important Implementation Notes

#### Reduction and Loss Function Output Buffers

Reduction operations (`sum`, `min`, `max`) and loss functions (`mse`, `cross_entropy`) require **at least 3 elements** in the output buffer for internal temporaries:

```rust
// ✅ Correct: output buffer has 3 elements
let mut sum_out = exec.allocate::<f32>(3)?;
ops::reduce::sum(&exec, &input, &mut sum_out, n)?;
let result = sum_out.to_vec()?[0];  // Result is in first element

// ❌ Wrong: output buffer too small
let mut sum_out = exec.allocate::<f32>(1)?;  // Will error!
ops::reduce::sum(&exec, &input, &mut sum_out, n)?;
```

#### Buffer Mutability

When passing buffers to operations:

- Input buffers: `&Buffer<T>` (immutable reference)
- Output buffers: `&mut Buffer<T>` (mutable reference)

```rust
ops::math::vector_add(&exec, &input_a, &input_b, &mut output, n)?;
                            //  ^         ^          ^^^^ mutable
                            //  |         immutable
                            //  immutable
```

#### Tensor Memory Layout

Tensors use row-major layout (C-style). For a 2D tensor `[M, N]`:

- Element at `[i, j]` is at linear index `i * N + j`
- Strides are computed automatically: `[N, 1]`

Zero-copy operations (select, narrow, slice, transpose) modify only shape/strides/offset, not data.

## WASM Demo Development

When making changes to the ONNX runtime or hologram-core that affect the browser demo:

**🚨 CRITICAL: Always use the official build script to rebuild the WASM demo.**

```bash
# Rebuild the WASM demo (from workspace root)
./scripts/build-wasm-demo.sh
```

**DO NOT** use `wasm-pack` directly, as it may fail due to package.json configuration issues. The official build script handles all necessary steps:
- Compiles hologram-onnx to WASM with WebGPU features
- Optimizes the WASM bundle
- Adds cache-busting to prevent stale JavaScript
- Places output in `/workspace/public/lib/onnx`

After rebuilding:
1. Restart the dev server if running (`pnpm dev` in `/workspace/public`)
2. Hard refresh the browser (Ctrl+Shift+R or Cmd+Shift+R)
3. Test the Stable Diffusion demo at `http://localhost:3000/demos/stable-diffusion`

**🚨 CRITICAL: NEVER use `killall node` or `killall -9 node`**

**NEVER EVER kill all node processes.** This can disrupt critical development processes and other important services.

**Forbidden commands:**
- ❌ `killall node`
- ❌ `killall -9 node`
- ❌ Any command that kills all node processes indiscriminately

**Instead:** If you need to stop a specific dev server, use targeted process management:
- Find the specific process: `lsof -ti:3000` (for port 3000)
- Kill only that process: `kill <pid>`
- Or let the user manage dev server processes manually

## Continuous Integration

Before committing code, run:

```bash
# Build everything (must produce zero warnings)
cargo build --workspace --all-targets

# Run all tests (must all pass)
cargo test --workspace

# Check formatting (must pass)
cargo fmt --check

# Run clippy (must produce zero warnings)
cargo clippy --workspace -- -D warnings

# Run integration tests (must all pass)
cargo test --workspace --test '*'
```

**🚨 CRITICAL: All commands must complete successfully with zero warnings before committing code.**

**IMPORTANT: Do NOT run `cargo doc` - it crashes the IDE. Skip documentation building in CI.**

## Checklist for New Features

- [ ] Feature implemented with clear, idiomatic Rust
- [ ] Unit tests written for all functions
- [ ] Property-based tests for mathematical properties
- [ ] Integration tests for cross-component functionality
- [ ] Documentation written with examples
- [ ] Error cases handled and tested
- [ ] Performance considered (benchmarks if critical)
- [ ] Code formatted (`cargo fmt`)
- [ ] **Zero compiler warnings** (`cargo build --workspace` produces no warnings)
- [ ] **Zero clippy warnings** (`cargo clippy --workspace -- -D warnings` passes)
- [ ] **All tests pass** (`cargo test --workspace`)
- [ ] Documentation builds (`cargo doc`)

## Checklist for Bug Fixes

- [ ] Root cause identified
- [ ] Test case added that reproduces the bug
- [ ] Fix implemented
- [ ] Test case now passes
- [ ] Related tests still pass
- [ ] Regression tests added to prevent recurrence
- [ ] **Zero compiler warnings** (`cargo build --workspace` produces no warnings)
- [ ] **Zero clippy warnings** (`cargo clippy --workspace -- -D warnings` passes)
- [ ] **All workspace tests pass** (`cargo test --workspace`)

## Review Criteria

When reviewing code (or having Claude review):

1. **Correctness**: Does it work? Are edge cases handled?
2. **Test Coverage**: Are there sufficient tests?
3. **Documentation**: Is the API documented with examples?
4. **Performance**: Are there obvious inefficiencies?
5. **Maintainability**: Is the code clear and well-organized?
6. **Idioms**: Does it follow Rust best practices?

## Common Pitfalls to Avoid

### Arbitrary Planning

Plan tasks based on architecture. Avoid time estimations for tasks.

### Type Conversions

```rust
// ❌ Bad: can overflow or wrap
let value = (large_number as u8) % LIMIT;

// ✅ Good: explicit range checking
let value = if large_number >= LIMIT as u64 {
    return Err(Error::OutOfRange);
} else {
    large_number as u8
};
```

### Iterator Ranges

```rust
// ❌ Bad: byte overflow (256 wraps to 0)
for byte in 0..BYTES_PER_PAGE as u8 {
    // This never executes!
}

// ✅ Good: use inclusive range
for byte in 0..=255u8 {
    // Correctly iterates over all 256 values
}
```

### Unsafe Code

```rust
// ✅ Always document safety invariants
/// # Safety
///
/// Caller must ensure class_index < 96
pub const unsafe fn new_unchecked(class_index: u8) -> Self {
    Self(class_index)
}

// ✅ Use safe constructor in tests
#[cfg(test)]
let class = unsafe { ClassIndex::new_unchecked(42) };
```

### Non-Production Code

**🚨 CRITICAL: Never output code that is non-production ready.**

All code must be complete, production-ready implementations. Never use phrases like:

- ❌ "In a real implementation..."
- ❌ "For demonstration purposes..."
- ❌ "This is a simplified version..."
- ❌ "This is a stub..."
- ❌ "Placeholder"
- ❌ "For simplicity..."
- ❌ "This will be implemented later..."
- ❌ "TODO: complete this"
- ❌ "We can add this functionality in the future..."

**Every line of code you write must be production-ready, fully functional, and completely implemented.**

## Resources

### External Resources

- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Effective Rust](https://www.lurklurk.org/effective-rust/)

### Project Documentation

- [Backend Architecture](docs/BACKEND_ARCHITECTURE.md)
- [Backend Trait Architecture](docs/BACKEND_TRAIT_ARCHITECTURE.md)
- [CPU Backend Tracing](docs/CPU_BACKEND_TRACING.md)

## Contact

For questions or clarifications on development practices, refer to:

- Project documentation in `docs/`
- Existing code patterns in the codebase

---

**Remember: The goal is correct, well-tested, maintainable code. Take the time to do it right.**
