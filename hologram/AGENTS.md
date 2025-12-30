# Agent Development Guidelines for Hologramapp

This document provides critical guidelines for all AI agents (including Task agents, Explore agents, and Plan agents) working on the hologramapp project.

## =� CRITICAL RULE: NO PLACEHOLDERS, NO TODOs, NO INCOMPLETE IMPLEMENTATIONS

**NEVER LEAVE PLACEHOLDERS OR TODOs. IMPLEMENT THE FUNCTIONS COMPLETELY.**

This is the most important rule for all agents working on this codebase:

### Absolute Prohibitions

1. **NO TODO comments** - Never write `// TODO:`, `// FIXME:`, or any variation
2. **NO unimplemented!() macros** - Never use `unimplemented!()`, `todo!()`, or `panic!("not yet implemented")`
3. **NO stub functions** - Every function must have a complete, working implementation
4. **NO placeholder code** - No "this will be implemented later" comments
5. **NO partial implementations** - If you start implementing a function, finish it completely
6. **NO function skeletons** - Don't create function signatures without full implementations

### What This Means In Practice

**L FORBIDDEN:**
```rust
pub fn process_data(input: &[f32]) -> Vec<f32> {
    // TODO: implement this
    unimplemented!()
}
```

**L FORBIDDEN:**
```rust
pub fn validate_input(data: &Data) -> Result<()> {
    if data.name.is_empty() {
        return Err(Error::InvalidInput("name required"));
    }
    // TODO: add more validation
    Ok(())
}
```

**L FORBIDDEN:**
```rust
// Stub for future implementation
pub fn complex_operation() {
    todo!("implement when we have time")
}
```

** REQUIRED:**
```rust
pub fn process_data(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x * 2.0).collect()
}
```

** REQUIRED:**
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

### Why This Is Non-Negotiable

Incomplete implementations:
- **Break the build** - Code that panics at runtime is broken
- **Violate contracts** - Functions that panic violate their type signatures
- **Create technical debt** - Future work must figure out what was intended
- **Waste time** - Users must ask again to get work completed
- **Undermine trust** - Incomplete work makes the codebase unreliable

### The Only Exception

The ONLY acceptable use of TODO/stub patterns is:
1. **During active development** - While actively writing code in the same session
2. **Must be completed before finishing** - All TODOs removed before marking task complete
3. **Never committed** - No TODOs, stubs, or unimplemented!() in committed code

**If you cannot implement a function completely right now, DO NOT CREATE IT.**

## 🚨 CRITICAL RULE: NO RUNTIME-JIT IN ONNX COMPILER

**ABSOLUTELY FORBIDDEN: Runtime-JIT compilation in `hologram-onnx` and `hologram-onnx-compiler`**

The ONNX compiler must perform 100% compile-time precomputation. Runtime-JIT compilation is strictly banned in these crates.

### Absolute Prohibitions

1. **NO runtime-JIT fallbacks** - All operations must be fully precomputed during compilation
2. **NO "runtime compilation" comments** - Never suggest operations will be "compiled at runtime"
3. **NO deferred execution** - All execution happens during Pass 3 precomputation
4. **NO lazy compilation** - Everything is compiled eagerly, nothing deferred to runtime

### Why This Is Critical

The entire architecture of MoonshineHRM depends on compile-time precomputation:
- **O(1) runtime lookups** - Runtime just looks up precomputed results in hash tables
- **No runtime overhead** - Zero compilation cost at inference time
- **Deterministic performance** - No JIT warmup, no compilation spikes
- **Memory-mapped execution** - Precomputed results stored in .mshr files

### What To Do Instead

If an operation cannot be executed during precomputation:

**❌ FORBIDDEN:**
```rust
// Operations that require constant inputs not available during precomputation
// These will use runtime-JIT compilation instead
"Gather" | "Slice" => {
    // Return pattern unchanged - actual execution happens at runtime
    Ok(pattern.to_vec())
}
```

**✅ REQUIRED:**
```rust
// Extract constant inputs from ONNX graph metadata and execute during precomputation
"Gather" => {
    let indices = self.get_constant_input(node, 1)?;
    let operator = GatherOp::new_with_indices(0, &indices);
    operator.execute(&self.atlas, &inputs)
}
```

If you cannot precompute an operation because:
- **Missing metadata** - Fix the metadata extraction to include required constants
- **Architecture limitation** - Extend the architecture to support the operation
- **Complex operation** - Break it down into precomputable steps

**NEVER fall back to runtime-JIT. Fix the precomputation pipeline instead.**

### Enforcement

Any code that mentions "runtime-JIT", "runtime compilation", "JIT fallback", or similar concepts in `hologram-onnx` or `hologram-onnx-compiler` will be rejected.

The only acceptable runtime behavior is:
1. **Lookup precomputed result** - Hash input → address → load result from memory
2. **Return result** - No computation, just memory access

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

## Core Principle: Complete Every Task Fully

**=� CRITICAL: Complete every task fully. No shortcuts, no excuses.**

When assigned a task:

1. **Never stop before completion** - Finish every task completely
2. **No excuses about constraints** - Don't cite "time", "tokens", or "efficiency" to shortcut work
3. **Do the work sequentially** - If there are 50 files to update, update all 50
4. **No script-based shortcuts** - Actually do the edits, don't suggest automation as an excuse
5. **Finish what you start** - If you create a todo list with 26 items, complete all 26

### Anti-Patterns to Avoid

L "Given time constraints, let me create a more efficient approach..."
L "Due to token limits, I'll write a script instead..."
L "To save time, let me batch these changes..."
L "I'll stub this out for now..."
L "This is a placeholder implementation..."
L "For simplicity, I'll implement a basic version..."

 **Just do the work.** Edit every file. Complete every function. Finish every test.

## Completion Criteria

**A task is NOT complete until:**

- All functions are fully implemented (no TODOs, no stubs, no unimplemented!())
- All tests pass (`cargo test --workspace`)
- Zero compiler warnings (`cargo build --workspace`)
- Zero clippy warnings (`cargo clippy --workspace -- -D warnings`)
- All code is properly formatted (`cargo fmt`)

## Guidelines for Different Agent Types

### Explore Agents

When exploring codebases:
- Report findings completely and accurately
- Never say "I couldn't find X" without thorough search
- Use all available tools (Grep, Glob, Read) comprehensively
- Provide complete file paths and line numbers

### Plan Agents

When planning implementations:
- Only plan what can be fully implemented
- Don't plan stub functions or TODO sections
- Every planned function must have a clear, complete implementation strategy
- If you can't plan a complete implementation, don't plan it at all

### Task Agents

When implementing tasks:
- Implement every function completely
- Write comprehensive tests for all code
- Fix all compiler warnings and errors
- Never leave partial implementations
- Mark tasks complete only when fully done

## Code Quality Requirements

All code written by agents must:

1. **Be production-ready** - No "demo", "stub", or "simplified" code
2. **Have complete implementations** - Every function does what its signature promises
3. **Include comprehensive tests** - Unit tests, integration tests, property tests
4. **Have zero warnings** - Both compiler and clippy warnings must be zero
5. **Be properly documented** - Clear doc comments with examples
6. **Follow Rust idioms** - Idiomatic, maintainable Rust code
7. **Use `hologram-common` for shared utilities** - Don't duplicate code across crates
8. **Use Builder pattern where appropriate** - For complex structs with many fields (see below)

## Builder Pattern for Complex Structs

**Use the Builder pattern where appropriate for ergonomic APIs, but don't overuse it.**

### When to Use Builder Pattern

✅ **USE Builder pattern for:**
- Structs with 4+ fields, especially if some are optional
- Public API configuration objects where ergonomics matter
- Complex initialization requiring validation across multiple fields
- When incremental construction with method chaining improves clarity

❌ **DON'T USE Builder pattern for:**
- Simple structs with 1-3 required fields - use direct construction instead
- Internal implementation details - prefer simplicity
- When `Default` trait or simple constructor is clearer
- Don't add complexity for imaginary future needs (YAGNI)

### Implementation Guidelines

1. **Validate in build()** - Always validate configuration in the `build()` method
2. **Provide sensible defaults** - Make optional fields truly optional with good defaults
3. **Keep it simple** - Don't add builder complexity unless it genuinely improves ergonomics
4. **Document required fields** - Make it clear which fields must be set before calling `build()`
5. **Return Result from build()** - Allow validation errors to be reported properly
6. **Use derive_builder when appropriate** - For simple cases without complex validation logic

### Example: When Builder is Appropriate

```rust
// ✅ GOOD: Complex configuration with many optional fields
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
```

### Example: When Builder is NOT Appropriate

```rust
// ❌ FORBIDDEN: Don't use Builder for simple structs
pub struct Point {
    x: f32,
    y: f32,
}

// ❌ Overkill - just use a simple constructor
pub struct PointBuilder { /* ... */ }

// ✅ REQUIRED: Simple constructor is clearer
impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}
```

### Using derive_builder

For straightforward builders without complex validation:

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
```

## Common Utility Functions

**🚨 CRITICAL: Write common utility functions in the `hologram-common` crate when they can be used across multiple crates.**

When you identify functionality that is or could be used by multiple crates:

- **Place it in `hologram-common`** - Centralize shared utilities
- **Avoid duplication** - Never copy-paste utility functions across crates
- **Make it reusable** - Design utilities to be generic and composable
- **Document thoroughly** - Common utilities need clear documentation with examples

**When to use `hologram-common`:**

1. **The function is already used in 2+ crates** - Move it to common immediately
2. **The function could logically be used elsewhere** - Consider placing it in common from the start
3. **The function is a general utility** - Prefer common over crate-specific placement
4. **You're about to copy-paste code** - Stop and move it to common instead

**Examples of what belongs in `hologram-common`:**

- ✅ Error types used across multiple crates
- ✅ Type conversions and validation helpers
- ✅ Mathematical utilities (shape calculations, broadcasting logic)
- ✅ Common data structures (buffers, memory layouts)
- ✅ Shared traits and interfaces
- ✅ Logging and tracing utilities

**Example:**

```rust
// ❌ FORBIDDEN: Duplicating utility in multiple crates
// In hologram-core/src/utils.rs
pub fn validate_shape(shape: &[usize]) -> Result<()> { /* ... */ }

// In hologram-onnx/src/utils.rs
pub fn validate_shape(shape: &[usize]) -> Result<()> { /* ... */ } // DUPLICATE!

// ✅ REQUIRED: Shared utility in hologram-common
// In hologram-common/src/shape.rs
pub fn validate_shape(shape: &[usize]) -> Result<()> {
    // validation logic (single source of truth)
}

// Both crates use the common implementation
use hologram_common::shape::validate_shape;
```

**Benefits:**
- **DRY principle** - Single source of truth
- **Consistency** - Same implementation everywhere
- **Maintainability** - Fix bugs once, benefit everywhere

## Never Output Non-Production Code

Never write code with phrases like:
- "In a real implementation..."
- "For demonstration purposes..."
- "This is a simplified version..."
- "This is a stub..."
- "Placeholder"
- "For simplicity..."
- "This will be implemented later..."
- "TODO: complete this"

**ALL CODE MUST BE PRODUCTION-READY, COMPLETE, AND FULLY FUNCTIONAL.**

## Summary

**The cardinal rule:** If you write a function signature, you MUST provide a complete, working implementation. No exceptions. No TODOs. No stubs. No placeholders.

**If you can't implement it completely now, don't create it at all.**

---

**Remember: Incomplete work is worse than no work. Do it right or don't do it.**