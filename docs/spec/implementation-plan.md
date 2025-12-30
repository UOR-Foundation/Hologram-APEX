# Hologram Core Completion Plan

**Version:** 1.0.0
**Created:** 2025-11-28

## Executive Summary

This plan outlines the implementation tasks required to complete hologram-core as an algebraic processor that fully utilizes the backend's mathematical capabilities. The backends have solved the hard mathematical problems; core needs the control logic to drive them.

## Current State Analysis

### What Exists

| Component | Location | Status |
|-----------|----------|--------|
| GriessVector (196,884-dim) | `core/src/griess/vector.rs` | ✅ Complete |
| lift() / resonate() / crush() | `core/src/griess/resonance.rs` | ✅ Complete (for GriessVector) |
| 96-class UOR primitives | `core/src/atlas/uor.rs` | ✅ Complete |
| PhiCoordinate addressing | `core/src/atlas/`, `core/src/isa_builder.rs` | ✅ Complete |
| HRM embed/decode | `core/src/hrm/embed/`, `core/src/hrm/decode/` | ✅ Complete |
| Character Table (194×194) | `backends/src/atlas/character_table.rs` | ✅ Complete |
| CharProduct instruction | `backends/src/isa/` | ✅ Complete |
| OrbitClassify instruction | `backends/src/isa/` | ✅ Complete |
| Standard SIMD GEMM | `core/src/ops/linalg.rs` | ✅ Complete |

### What's Missing

| Component | Gap | Impact |
|-----------|-----|--------|
| CharacterValue → ResonanceClass | No projection from backend CharProduct results | Cannot close computation loop |
| TensorStorage::Atlas | No compressed algebraic tensor storage | No space savings, no dispatch |
| Atlas GEMM path | linalg.rs only has SIMD path | Cannot use algebraic acceleration |
| Tensor.matmul() | Disabled during refactor | No high-level matrix API |
| ISA builder Atlas ops | No CharProduct/OrbitClassify builders | Cannot emit algebraic programs |

## Implementation Phases

### Phase 1: Close the Mathematical Loop

**Goal:** Enable round-trip computation through the algebraic backend

#### 1.1 Import CharacterValue Type

**File:** `crates/core/src/atlas/character_value.rs` (new)

```rust
/// Re-export or define CharacterValue from backends
/// This represents the 194-dimensional character space result
pub use hologram_backends::atlas::CharacterValue;

// OR define locally if backends doesn't export it:
pub enum CharacterValue {
    /// Integer character value
    Integer(i64),
    /// Cyclotomic value (root of unity)
    Cyclotomic { order: u32, coefficients: Vec<i64> },
}
```

**Tasks:**
- [ ] Determine if CharacterValue exists in backends or needs definition
- [ ] Create `crates/core/src/atlas/character_value.rs`
- [ ] Export from `crates/core/src/atlas/mod.rs`

#### 1.2 Implement nearest_resonance_class()

**File:** `crates/core/src/atlas/uor.rs`

```rust
/// Pre-computed canonical character signatures for each resonance class
static CANONICAL_SIGNATURES: LazyLock<[CharacterSignature; 96]> =
    LazyLock::new(compute_canonical_signatures);

/// Project CharacterValue to nearest resonance class
///
/// Given a character computation result from CharProduct instruction,
/// find the resonance class whose character signature minimizes distance.
pub fn nearest_resonance_class(value: &CharacterValue) -> ResonanceClass {
    let mut best_class = 0u8;
    let mut best_distance = f64::MAX;

    for (class_id, signature) in CANONICAL_SIGNATURES.iter().enumerate() {
        let distance = signature.distance_to(value);
        if distance < best_distance {
            best_distance = distance;
            best_class = class_id as u8;
        }
    }

    best_class
}
```

**Tasks:**
- [ ] Define `CharacterSignature` struct (194-dimensional)
- [ ] Implement `compute_canonical_signatures()` using CHARACTER_TABLE
- [ ] Implement `CharacterSignature::distance_to(&CharacterValue)`
- [ ] Implement `nearest_resonance_class()`
- [ ] Add unit tests for all 96 classes
- [ ] Add property tests for projection consistency

#### 1.3 Wire Griess Resonate to CharacterValue

**File:** `crates/core/src/griess/resonance.rs`

```rust
/// Extended resonate that handles both GriessVector and CharacterValue
pub fn resonate_character(value: &CharacterValue) -> Result<ResonanceClass> {
    Ok(nearest_resonance_class(value))
}
```

**Tasks:**
- [ ] Add `resonate_character()` function
- [ ] Update module exports
- [ ] Add integration tests

---

### Phase 2: Atlas Tensor Storage

**Goal:** Enable compressed algebraic storage with **bijective** reconstruction

#### 2.1 Define AtlasTensor with Bijective Storage

**File:** `crates/core/src/atlas_tensor.rs`

Atlas tensors are **separate from Tensor<T>** and store both resonance classes
(for algebraic operations) and torus offsets (for bijective reconstruction).

```rust
use hologram_backends::atlas::{TorusConfig, PhiCoordinate};

/// Atlas-compressed tensor with bijective torus embedding
///
/// Stores both resonance class (for algebraic operations) and torus offset
/// (for bijective reconstruction). Achieves ~40% compression vs dense while
/// maintaining full reconstruction within type-specific error bounds.
///
/// Storage: 5 bytes per element (1 class + 4 offset) vs 8 bytes for f64
pub struct AtlasTensor<T: TorusConfig> {
    /// Resonance class IDs [0, 96) for character product operations
    classes: Buffer<u8>,
    /// Linear torus offsets for bijective reconstruction
    offsets: Buffer<u32>,
    /// Tensor shape
    shape: Vec<usize>,
    /// Row-major strides
    strides: Vec<usize>,
    /// Buffer offset
    offset: usize,
    /// Type marker
    _phantom: PhantomData<T>,
}
```

**Bijection Property:** Values can be reconstructed within error bounds:
- f32: ≤ 1/192 (0.52% error)
- f64: ≤ 1/384 (0.26% error)

**Tasks:**
- [x] Define `AtlasTensor<T: TorusConfig>` struct
- [ ] Implement accessor methods (shape, strides, numel, etc.)
- [ ] Implement Debug and Clone traits

#### 2.2 Implement Bijective Embedding (from_tensor)

**File:** `crates/core/src/atlas_tensor.rs`

```rust
impl<T: TorusConfig + bytemuck::Pod> AtlasTensor<T> {
    /// Convert dense tensor to Atlas with bijective embedding
    pub fn from_tensor(tensor: &Tensor<T>, exec: &mut Executor) -> Result<Self> {
        let numel = tensor.numel();
        let mut classes = exec.allocate::<u8>(numel)?;
        let mut offsets = exec.allocate::<u32>(numel)?;

        let src_data = tensor.buffer().to_vec(exec)?;
        let mut class_data = Vec::with_capacity(numel);
        let mut offset_data = Vec::with_capacity(numel);

        for val in src_data.iter() {
            // Embed value into torus coordinate (π phase)
            let phi = embed_to_torus::<T>(*val);

            // Extract resonance class for algebraic operations
            let bytes = bytemuck::bytes_of(val);
            let class = bytes[0] % 96;

            class_data.push(class);
            offset_data.push(phi.to_linear_offset() as u32);
        }

        classes.copy_from_slice(exec, &class_data)?;
        offsets.copy_from_slice(exec, &offset_data)?;

        Ok(Self { classes, offsets, shape, strides, offset: 0, _phantom: PhantomData })
    }
}
```

**Tasks:**
- [ ] Implement `embed_to_torus<T>(value) -> PhiCoordinate<T>`
- [ ] Implement `from_tensor()` with proper embedding
- [ ] Add tests for embedding correctness

#### 2.3 Implement Bijective Lifting (to_tensor)

**File:** `crates/core/src/atlas_tensor.rs`

```rust
impl<T: TorusConfig + bytemuck::Pod> AtlasTensor<T> {
    /// Convert AtlasTensor back to dense Tensor<T>
    ///
    /// Reconstructs values from torus offsets with error ≤ 1/RES_MOD.
    pub fn to_tensor(&self, exec: &mut Executor) -> Result<Tensor<T>> {
        let numel = self.numel();
        let mut output = exec.allocate::<T>(numel)?;

        let offset_data = self.offsets.to_vec(exec)?;
        let output_data: Vec<T> = offset_data
            .iter()
            .map(|&offset| {
                let phi = PhiCoordinate::<T>::from_linear_offset(offset as usize)
                    .expect("stored offset should be valid");
                lift_from_torus::<T>(phi)
            })
            .collect();

        output.copy_from_slice(exec, &output_data)?;
        Tensor::from_buffer(output, self.shape.clone())
    }
}
```

**Tasks:**
- [ ] Implement `lift_from_torus<T>(phi) -> T`
- [ ] Implement `to_tensor()` with proper lifting
- [ ] Add bijective roundtrip tests
- [ ] Verify error bounds match spec (f32: 0.52%, f64: 0.26%)

---

### Phase 3: Atlas Operations Integration

**Goal:** Wire algebraic operations through the Atlas path

#### 3.1 Add Atlas GEMM

**File:** `crates/core/src/ops/linalg.rs`

```rust
/// Matrix multiplication with automatic Atlas dispatch
pub fn gemm<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    a: &Tensor<T>,
    b: &Tensor<T>,
    c: &mut Tensor<T>,
    m: usize, k: usize, n: usize,
) -> Result<()> {
    // Check for Atlas path
    if a.is_atlas() && b.is_atlas() && c.is_atlas() {
        return gemm_atlas(exec, a, b, c, m, k, n);
    }

    // Standard SIMD path
    gemm_dense(exec, a, b, c, m, k, n)
}

/// Atlas-accelerated GEMM using character products
fn gemm_atlas<T>(
    exec: &mut Executor,
    a: &Tensor<T>,
    b: &Tensor<T>,
    c: &mut Tensor<T>,
    m: usize, k: usize, n: usize,
) -> Result<()> {
    let a_data = a.atlas_data().unwrap();
    let b_data = b.atlas_data().unwrap();
    let c_data = c.atlas_data_mut().unwrap();

    // For each output element c[i,j]
    for i in 0..m {
        for j in 0..n {
            let mut accum = CharacterValue::zero();

            for kk in 0..k {
                let class_a = a_data.read_at(exec, i * k + kk)?;
                let class_b = b_data.read_at(exec, kk * n + j)?;

                // CharProduct via character table
                let product = CHARACTER_TABLE.product(class_a, class_b);
                accum = accum.add(&CharacterValue::Integer(product));
            }

            // Project back to resonance class
            let result_class = nearest_resonance_class(&accum);
            c_data.write_at(exec, i * n + j, result_class)?;
        }
    }

    Ok(())
}
```

**Tasks:**
- [ ] Add dispatch logic to `gemm()`
- [ ] Implement `gemm_atlas()` using CharProduct
- [ ] Add `gemm_dense()` wrapper for existing SIMD code
- [ ] Add benchmarks comparing Atlas vs Dense paths
- [ ] Add correctness tests

#### 3.2 Re-enable Tensor.matmul()

**File:** `crates/core/src/tensor.rs`

```rust
impl<T: bytemuck::Pod + 'static> Tensor<T> {
    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor<T>, exec: &mut Executor) -> Result<Tensor<T>> {
        // Validate shapes
        let (m, k1) = self.matrix_dims()?;
        let (k2, n) = other.matrix_dims()?;
        if k1 != k2 {
            return Err(Error::ShapeMismatch(...));
        }

        // Allocate output
        let mut result = Tensor::zeros(&[m, n], exec)?;

        // Dispatch to appropriate GEMM
        ops::linalg::gemm(exec, self, other, &mut result, m, k1, n)?;

        Ok(result)
    }
}
```

**Tasks:**
- [ ] Remove "temporarily disabled" error
- [ ] Implement proper matmul with Atlas dispatch
- [ ] Add shape validation
- [ ] Add comprehensive tests

#### 3.3 Add Atlas MatVec

**File:** `crates/core/src/ops/linalg.rs`

```rust
/// Matrix-vector multiply with Atlas dispatch
pub fn matvec<T: bytemuck::Pod + 'static>(
    exec: &mut Executor,
    a: &Tensor<T>,
    x: &Tensor<T>,
    y: &mut Tensor<T>,
    m: usize, n: usize,
) -> Result<()> {
    if a.is_atlas() && x.is_atlas() && y.is_atlas() {
        return matvec_atlas(exec, a, x, y, m, n);
    }
    matvec_dense(exec, a, x, y, m, n)
}
```

**Tasks:**
- [ ] Add dispatch logic to `matvec()`
- [ ] Implement `matvec_atlas()`
- [ ] Add tests

---

### Phase 4: ISA Builder Atlas Instructions

**Goal:** Enable emitting Atlas-specific ISA programs

#### 4.1 Add CharProduct Builder

**File:** `crates/core/src/isa_builder.rs`

```rust
/// Build CharProduct instruction
pub fn build_char_product(
    dst: Register,
    class_a: Register,
    class_b: Register,
) -> Instruction {
    Instruction::CharProduct {
        dst,
        char_i: class_a,
        char_j: class_b,
    }
}

/// Build CharProduct with immediate class values
pub fn build_char_product_imm(
    dst: Register,
    class_a: u8,
    class_b: u8,
) -> Instruction {
    Instruction::CharProductImm {
        dst,
        char_i: class_a,
        char_j: class_b,
    }
}
```

**Tasks:**
- [ ] Add `build_char_product()` function
- [ ] Add `build_char_product_imm()` for immediate values
- [ ] Add tests

#### 4.2 Add OrbitClassify Builder

**File:** `crates/core/src/isa_builder.rs`

```rust
/// Build OrbitClassify instruction
pub fn build_orbit_classify(
    dst: Register,
    coord: Register,
    torus_size: usize,
) -> Instruction {
    Instruction::OrbitClassify {
        dst,
        coord,
        torus_size,
    }
}
```

**Tasks:**
- [ ] Add `build_orbit_classify()` function
- [ ] Add tests

#### 4.3 Add Lift Instruction Builder

**File:** `crates/core/src/isa_builder.rs`

```rust
/// Build Lift instruction (CharacterValue → ResonanceClass)
pub fn build_lift(
    dst: Register,
    src: Register,
) -> Instruction {
    Instruction::Lift { dst, src }
}
```

**Tasks:**
- [ ] Define Lift instruction in backends ISA (if not exists)
- [ ] Add `build_lift()` function
- [ ] Add tests

---

### Phase 5: Invariant Verification

**Goal:** Ensure mathematical invariants are preserved

#### 5.1 Update Conservation Checks

**File:** `crates/core/src/atlas/invariants.rs` (new or existing)

```rust
/// Verify Unity Neutrality on computation results
///
/// After a Lift operation, check that unity classes remain neutral
pub fn verify_unity_neutrality(
    input_classes: &[ResonanceClass],
    output_class: ResonanceClass,
) -> bool {
    // Unity classes should not affect the result
    let non_unity: Vec<_> = input_classes.iter()
        .filter(|&&c| !is_unity(c))
        .collect();

    // Recompute without unity inputs and verify same result
    // ...
}

/// Verify Conservation of Information
///
/// The total "information content" should be preserved through operations
pub fn verify_information_conservation(
    inputs: &[ResonanceClass],
    output: ResonanceClass,
) -> bool {
    // Check that crush parity is preserved
    let input_parity = inputs.iter().fold(false, |acc, &c| acc ^ crush(c));
    let output_parity = crush(output);

    // For certain operations, parity should be preserved
    input_parity == output_parity
}
```

**Tasks:**
- [ ] Create/update `invariants.rs`
- [ ] Implement `verify_unity_neutrality()`
- [ ] Implement `verify_information_conservation()`
- [ ] Add invariant checks to Atlas operations
- [ ] Add tests for invariant verification

---

## Testing Requirements

### Unit Tests

Each implementation task must include:
- [ ] Basic functionality tests
- [ ] Edge case tests (empty input, single element, max size)
- [ ] Error condition tests

### Integration Tests

- [ ] Full pipeline test: Dense → Atlas → CharProduct → Lift → Dense
- [ ] Cross-crate integration: core ↔ backends
- [ ] Executor integration with Atlas tensors

### Property Tests (proptest)

- [ ] `nearest_resonance_class` is idempotent: `f(f(x)) = f(x)`
- [ ] Round-trip consistency: `from_atlas(to_atlas(t)) ≈ t` (within error bounds)
- [ ] CharProduct associativity verification
- [ ] Lift/Resonate adjunction properties

### Benchmarks

- [ ] Atlas GEMM vs Dense GEMM (varying sizes)
- [ ] `nearest_resonance_class` throughput
- [ ] Atlas tensor conversion overhead
- [ ] Memory usage: Atlas vs Dense storage

---

## Success Criteria

### Phase 1 Complete When:
- [ ] `nearest_resonance_class()` passes all 96 class tests
- [ ] CharacterValue type is usable in core
- [ ] Integration test passes: CharProduct → nearest_resonance_class

### Phase 2 Complete When:
- [ ] `AtlasTensor<T: TorusConfig>` struct exists with bijective storage
- [ ] `from_tensor()` embeds values using PhiCoordinate torus mapping
- [ ] `to_tensor()` lifts offsets back to values within error bounds
- [ ] Bijective roundtrip test passes: error ≤ 1/RES_MOD for each type
- [ ] Compression ratio: ~40% for f64 (5 bytes vs 8 bytes per element)

### Phase 3 Complete When:
- [ ] `gemm()` dispatches to Atlas path for Atlas tensors
- [ ] `Tensor::matmul()` is re-enabled and works
- [ ] Atlas GEMM produces correct results

### Phase 4 Complete When:
- [ ] ISA builders for CharProduct, OrbitClassify, Lift exist
- [ ] Generated programs execute correctly on backends

### Phase 5 Complete When:
- [ ] Invariant checks pass for all Atlas operations
- [ ] No regressions in existing tests

### Final Acceptance:
- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace -- -D warnings` produces zero warnings
- [ ] `cargo build --workspace` produces zero warnings
- [ ] All benchmarks show expected performance characteristics

---

## Estimated Complexity

| Phase | Files Modified | New Files | Estimated LOC |
|-------|----------------|-----------|---------------|
| Phase 1 | 2-3 | 1 | ~300 |
| Phase 2 | 1 | 0 | ~400 |
| Phase 3 | 2 | 0 | ~350 |
| Phase 4 | 1 | 0 | ~150 |
| Phase 5 | 0-1 | 1 | ~200 |
| **Total** | **6-8** | **2** | **~1,400** |

---

## Dependencies

```
Phase 1 ─┬─→ Phase 2 ─┬─→ Phase 3 ─→ Phase 5
         │            │
         └────────────┴─→ Phase 4
```

- Phase 2 depends on Phase 1 (needs nearest_resonance_class for from_atlas)
- Phase 3 depends on Phase 1 and Phase 2
- Phase 4 can proceed in parallel with Phase 2/3
- Phase 5 depends on Phase 3
