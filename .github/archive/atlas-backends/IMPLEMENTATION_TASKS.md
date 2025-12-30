# Atlas Backends Implementation Completion Tasks

**Status**: Phase 8 Complete → Phase 9+ Pending
**Document Version**: 1.1

Comprehensive task list for achieving 100% ISA compliance and full hologram-core integration.

---

## Task Organization

- **Phase**: Dependency-ordered implementation phases
- **Priority**: P0 (critical/blocking), P1 (important), P2 (enhancement)
- **Owner**: Component affected (atlas-isa, atlas-backends, hologram-core)
- **Spec Impact**: 🔴 Breaking | 🟡 Clarification | 🟢 None

---

## Phase 1: Upstream Atlas ISA Enhancements

**Critical**: All spec changes must be reviewed for backward compatibility.

### Task 1.1: Add Label Metadata to Program Type

**Priority**: P0 (blocks control flow)
**Owner**: atlas-isa
**Spec Impact**: 🔴 **BREAKING CHANGE**
**Dependencies**: None

**Change**: Convert `pub type Program = Vec<Instruction>` to:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub instructions: Vec<Instruction>,
    pub labels: HashMap<String, usize>,
}

impl Program {
    pub fn new() -> Self;
    pub fn from_instructions(instructions: Vec<Instruction>) -> Self;
    pub fn add_label(&mut self, name: impl Into<String>) -> Result<()>;
    pub fn resolve_label(&self, name: &str) -> Option<usize>;
}

// Compatibility helpers
impl From<Vec<Instruction>> for Program { /* ... */ }
impl From<Program> for Vec<Instruction> { /* ... */ }
```

**Acceptance Criteria**:
- [ ] Program struct with instructions + labels
- [ ] Constructor methods: `new()`, `from_instructions()`, `with_capacity()`
- [ ] Label management: `add_label()`, `resolve_label()`, `has_label()`
- [ ] Error handling for duplicate labels
- [ ] All existing tests updated
- [ ] Migration guide provided
- [ ] Atlas ISA SPEC.md §7.4 updated

**Files**:
- `crates/atlas-isa/src/instructions.rs`
- `crates/atlas-isa/SPEC.md`
- All test files using `Vec<Instruction>`

---

### Task 1.2: Document Address Mode Semantics

**Priority**: P1
**Owner**: atlas-isa
**Spec Impact**: 🟡 **SPEC CLARIFICATION**
**Dependencies**: None

**Goal**: Document semantics of BufferOffset, PhiCoordinate, RegisterIndirect in SPEC.md §5.4.

**Acceptance Criteria**:
- [ ] Address mode semantics documented
- [ ] Effective address formulas provided
- [ ] Bounds checking requirements specified
- [ ] Usage examples for each mode
- [ ] Selection guidelines

**Files**: `crates/atlas-isa/SPEC.md`

---

### Task 1.3: Add Error Type for Label Resolution

**Priority**: P0
**Owner**: atlas-isa
**Spec Impact**: 🟢
**Dependencies**: Task 1.1

**Implementation**:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ProgramError {
    #[error("duplicate label '{0}' at instruction {1}")]
    DuplicateLabel(String, usize),
    #[error("undefined label '{0}' referenced in instruction {1}")]
    UndefinedLabel(String, usize),
    #[error("label '{0}' points to invalid instruction index {1} (max: {2})")]
    InvalidLabelTarget(String, usize, usize),
}
```

**Acceptance Criteria**:
- [ ] ProgramError enum defined
- [ ] Used in Program::add_label()
- [ ] Unit tests for all error cases

**Files**: `crates/atlas-isa/src/instructions.rs`

---

## Phase 2: Atlas Backends Core Completeness

### Task 2.1: Complete CVT Type Conversion Matrix

**Priority**: P0 (40% of conversions missing)
**Owner**: atlas-backends
**Spec Impact**: 🟢
**Dependencies**: None

**Missing**: I16, I64, U16, U32, U64, F16, BF16 source types (84 conversions)

**Implementation**: Use macros for systematic conversion generation:

```rust
macro_rules! cvt_impl {
    ($src_t:ty, $dst_t:ty) => {{
        let value: $src_t = self.registers.read(*src)?;
        self.registers.write(*dst, value as $dst_t)?;
    }};
}

macro_rules! cvt_to_f16 {
    ($src_t:ty) => {{
        let value: $src_t = self.registers.read(*src)?;
        self.registers.write(*dst, f16::from_f32(value as f32))?;
    }};
}

// Similar for cvt_to_bf16, cvt_from_f16, cvt_from_bf16
```

**Acceptance Criteria**:
- [ ] All 144 type combinations (12×12 matrix)
- [ ] Remove "not yet implemented" fallback
- [ ] Unit test for each source type
- [ ] Property test: roundtrip conversions

**Files**: `crates/atlas-backends/src/cpu.rs` (execute_instruction CVT block)

---

### Task 2.2: Implement PhiCoordinate Addressing

**Priority**: P0 (blocks boundary lens)
**Owner**: atlas-backends
**Spec Impact**: 🟢
**Dependencies**: Task 1.2

**Implementation**:

```rust
atlas_isa::Address::PhiCoordinate { class, page, byte } => {
    // Validate: class < 96, page < 48
    let offset = (*class as usize) * CLASS_STRIDE + (*page as usize) * 256 + (*byte as usize);
    let boundary_base = self.boundary_pool.ok_or(BackendError::NotInitialized)?.as_ptr();
    let ptr = unsafe { boundary_base.add(offset) };
    // Load/store using ptr
}
```

**Refactor**: Extract `load_value()` and `store_value()` helpers to reduce duplication.

**Acceptance Criteria**:
- [ ] PhiCoordinate for LDG, STG, LDS, STS
- [ ] Bounds validation (class, page, byte)
- [ ] Linear offset calculation
- [ ] Unit tests for all 12 types
- [ ] Integration test: access all 96 classes

**Files**: `crates/atlas-backends/src/cpu.rs`

---

### Task 2.3: Implement RegisterIndirect Addressing

**Priority**: P1
**Owner**: atlas-backends
**Spec Impact**: 🟢
**Dependencies**: Task 2.2

**Implementation**:

```rust
atlas_isa::Address::RegisterIndirect { base, offset } => {
    let base_addr: u64 = self.registers.read(*base)?;
    let effective_addr = if *offset >= 0 {
        base_addr.checked_add(*offset as u64)
    } else {
        base_addr.checked_sub(offset.wrapping_neg() as u64)
    }.ok_or_else(|| BackendError::ExecutionFailed("address overflow"))?;
    let ptr = effective_addr as *const u8;
    // Load/store using ptr
}
```

**Safety Decision Required**: Choose validation strategy:
- **Option A**: Validate against known allocations (recommended for CPU)
- **Option B**: Unsafe mode (document as caller's responsibility)

**Acceptance Criteria**:
- [ ] RegisterIndirect for LDG, STG, LDS, STS
- [ ] Type checking: base must be u64
- [ ] Signed offset handling
- [ ] Overflow checking
- [ ] Safety strategy documented

**Files**:
- `crates/atlas-backends/src/cpu.rs`
- `crates/atlas-backends/SPEC.md`

---

### Task 2.4: Complete Reduction Type Support

**Priority**: P1
**Owner**: atlas-backends
**Spec Impact**: 🟢
**Dependencies**: None

**Missing**: I8, I16, I64, U8, U16, U32, U64 for REDUCE.ADD/MIN/MAX/MUL

**Implementation**: Use macro for systematic coverage:

```rust
macro_rules! reduce_add_impl {
    ($ty:ty, $zero:expr) => {{
        let mut sum: $ty = $zero;
        for i in 0..*count {
            let reg = Register::new(src_base.index() + i as u8);
            let value: $ty = self.registers.read(reg)?;
            sum = sum.wrapping_add(value);
        }
        self.registers.write(*dst, sum)?;
    }};
}

match ty {
    T::I8 => reduce_add_impl!(i8, 0),
    T::I16 => reduce_add_impl!(i16, 0),
    // ... all 10 types
}
```

**Acceptance Criteria**:
- [ ] All 10 types (I8-U64, F32/F64) for each REDUCE op
- [ ] Wrapping arithmetic for integers
- [ ] Remove "only implemented for f32, f64" errors
- [ ] Unit tests for each type/operation

**Files**: `crates/atlas-backends/src/cpu.rs`

---

### Task 2.5: Complete Transcendental Type Support

**Priority**: P2
**Owner**: atlas-backends
**Spec Impact**: 🟢
**Dependencies**: None

**Missing**: F16, BF16 for EXP, LOG, SQRT, SIN, COS, TANH, SIGMOID, etc.

**Strategy**: Convert to F32, compute, convert back:

```rust
macro_rules! transcendental_f16 {
    ($op:ident, $src:expr, $dst:expr) => {{
        let value: f16 = self.registers.read($src)?;
        let result = value.to_f32().$op();
        self.registers.write($dst, f16::from_f32(result))?;
    }};
}
```

**Acceptance Criteria**:
- [ ] F16/BF16 for all 10 transcendental ops
- [ ] Remove "only f32/f64" errors
- [ ] Unit tests for half-precision
- [ ] Document precision characteristics

**Files**: `crates/atlas-backends/src/cpu.rs`

---

### Task 2.6: Implement Control Flow Instructions

**Priority**: P0 (blocks structured programming)
**Owner**: atlas-backends
**Spec Impact**: 🟢
**Dependencies**: Tasks 1.1, 1.3

**Implementation**:

```rust
BRA { target, pred } => {
    let should_branch = pred.map_or(true, |p| self.registers.read_pred(p));
    if should_branch {
        self.program_counter = *self.labels.get(&target.0)
            .ok_or_else(|| BackendError::ExecutionFailed("undefined label"))?;
    } else {
        self.program_counter += 1;
    }
}

CALL { target } => {
    if self.call_stack.len() >= 256 {
        return Err(BackendError::ExecutionFailed("call stack overflow"));
    }
    self.call_stack.push(self.program_counter + 1);
    self.program_counter = *self.labels.get(&target.0)?;
}

RET => {
    self.program_counter = self.call_stack.pop()
        .ok_or_else(|| BackendError::ExecutionFailed("empty call stack"))?;
}

LOOP { counter, target } => {
    let count: u32 = self.registers.read(*counter)?;
    if count > 0 {
        self.registers.write(*counter, count - 1)?;
        self.program_counter = *self.labels.get(&target.0)?;
    } else {
        self.program_counter += 1;
    }
}
```

**Acceptance Criteria**:
- [ ] BRA (conditional and unconditional)
- [ ] CALL with stack push
- [ ] RET with stack pop
- [ ] LOOP with counter decrement
- [ ] Label resolution from HashMap
- [ ] Call stack depth limit (256)
- [ ] Unit tests: branch, call/ret, loop
- [ ] Integration tests: fibonacci, factorial
- [ ] Error tests: undefined label, stack overflow/underflow

**Files**: `crates/atlas-backends/src/cpu.rs`

---

## Phase 3: Hologram-Core Integration

### Task 3.1: Create ISA Program Builder

**Priority**: P0 (foundation for Phase 9)
**Owner**: hologram-core
**Spec Impact**: 🟢
**Dependencies**: Task 1.1

**Implementation**: Create `crates/hologram-core/src/program_builder.rs`:

```rust
pub struct ProgramBuilder {
    program: Program,
    next_reg: u8,
    next_label: u32,
}

impl ProgramBuilder {
    pub fn new() -> Self;
    pub fn alloc_reg(&mut self) -> Register;
    pub fn gen_label(&mut self, prefix: &str) -> Label;
    pub fn add_label(&mut self, label: Label) -> Result<()>;
    pub fn emit(&mut self, inst: Instruction);
    pub fn load_f32(&mut self, buffer: &Buffer<f32>, offset: usize) -> Result<Register>;
    pub fn store_f32(&mut self, reg: Register, buffer: &Buffer<f32>, offset: usize);
    pub fn exit(&mut self);
    pub fn build(self) -> Program;
}
```

**Acceptance Criteria**:
- [ ] Register allocation
- [ ] Label generation
- [ ] Typed helpers (load/store for all types)
- [ ] Control flow helpers
- [ ] Unit tests
- [ ] Documentation with examples

**Files**:
- Create: `crates/hologram-core/src/program_builder.rs`
- Modify: `crates/hologram-core/src/lib.rs`

---

### Task 3.2: Rewrite ops::math to ISA Programs

**Priority**: P0
**Owner**: hologram-core
**Spec Impact**: 🟢
**Dependencies**: Tasks 3.1, 2.6

**Operations** (12): vector_add, vector_sub, vector_mul, vector_div, scalar_add, scalar_mul, min, max, abs, neg, relu, clip

**Strategy**:
- For small n: unroll load/compute/store
- For large n: use rayon + chunked ISA programs
- Or: implement RegisterIndirect addressing for loops

**Acceptance Criteria**:
- [ ] All 12 operations rewritten
- [ ] No legacy Operation enum references
- [ ] Type-generic implementations
- [ ] All existing tests pass
- [ ] No performance regression

**Files**: `crates/hologram-core/src/ops/math.rs`

---

### Task 3.3: Rewrite ops::reduce to ISA Programs

**Priority**: P0
**Owner**: hologram-core
**Spec Impact**: 🟢
**Dependencies**: Tasks 2.4, 3.1

**Operations** (3): sum, min, max

**Strategy**: Use ISA REDUCE instructions:
- Single-pass for n ≤ 256
- Multi-pass tree reduction for n > 256

**Acceptance Criteria**:
- [ ] All 3 operations use ISA REDUCE
- [ ] Multi-pass reduction for large n
- [ ] All tests pass
- [ ] Performance benchmarks

**Files**: `crates/hologram-core/src/ops/reduce.rs`

---

### Task 3.4: Rewrite ops::activation and ops::loss

**Priority**: P1
**Owner**: hologram-core
**Spec Impact**: 🟢
**Dependencies**: Tasks 2.5, 3.1

**Operations**:
- **activation** (5): sigmoid, relu, tanh, softmax, gelu
- **loss** (3): mse, cross_entropy, binary_cross_entropy

**Strategy**: Use ISA transcendental instructions (SIGMOID, TANH, EXP, LOG)

**Acceptance Criteria**:
- [ ] All 8 operations rewritten
- [ ] Use ISA transcendentals
- [ ] SOFTMAX uses REDUCE.ADD
- [ ] All tests pass

**Files**:
- `crates/hologram-core/src/ops/activation.rs`
- `crates/hologram-core/src/ops/loss.rs`

---

### Task 3.5: Remove Legacy Operation Enum

**Priority**: P2
**Owner**: hologram-core
**Spec Impact**: 🟢
**Dependencies**: Tasks 3.2, 3.3, 3.4

**Goal**: Clean up after all operations migrated to ISA programs.

**Acceptance Criteria**:
- [ ] Operation enum removed
- [ ] No legacy operation API references
- [ ] Dead code removed
- [ ] All tests pass

**Files**:
- `crates/hologram-core/src/executor.rs`
- `crates/atlas-backends/src/cpu.rs`

---

## Phase 4: Testing and Validation

### Task 4.1: ISA Conformance Test Suite

**Priority**: P0
**Owner**: atlas-backends
**Spec Impact**: 🟢
**Dependencies**: All Phase 2 tasks

**Coverage Matrix**:

| Category | Instructions | Current |
|----------|--------------|---------|
| Data Movement | LDG, STG, LDS, STS, MOV, CVT | ⚠️ Partial |
| Arithmetic | ADD, SUB, MUL, DIV, MAD, FMA, MIN, MAX, ABS, NEG | ✅ Complete |
| Logic | AND, OR, XOR, NOT, SHL, SHR, SETcc, SEL | ✅ Complete |
| Control Flow | BRA, CALL, RET, LOOP, EXIT | ❌ Missing |
| Synchronization | BAR.SYNC, MEM.FENCE | ✅ Complete |
| Atlas-Specific | CLS.GET, MIRROR, UNITY.TEST, NBR.*, RES.ACCUM, PHASE.*, BOUND.MAP | ✅ Complete |
| Reductions | REDUCE.ADD/MIN/MAX/MUL | ⚠️ Partial |
| Transcendentals | EXP, LOG, SQRT, SIN, COS, TANH, SIGMOID, etc. | ⚠️ Partial |

**Acceptance Criteria**:
- [ ] All 55 ISA instructions tested
- [ ] All type combinations tested
- [ ] All address modes tested
- [ ] Error cases tested
- [ ] Property-based tests for invariants

**Files**: Create `crates/atlas-backends/tests/conformance.rs`

---

### Task 4.2: Property-Based Tests for Atlas Invariants

**Priority**: P1
**Owner**: atlas-backends
**Spec Impact**: 🟢
**Dependencies**: None

**Invariants**:
1. Unity neutrality (sum of resonance = 0)
2. Mirror involution (MIRROR(MIRROR(x)) = x)
3. Neighbor symmetry (if B in NBR(A), then A in NBR(B))
4. Phase modulus (phase < 768)
5. Boundary lens roundtrip

**Acceptance Criteria**:
- [ ] Property tests for all invariants
- [ ] 1000+ test cases per property

**Files**: `crates/atlas-backends/src/cpu.rs`

---

### Task 4.3: Phase 9 Integration Tests

**Priority**: P0
**Owner**: hologram-core
**Spec Impact**: 🟢
**Dependencies**: Tasks 3.2, 3.3, 3.4

**Scenarios**:
- Vector operations (large buffers)
- Reduction operations
- Neural network operations
- Multi-type tests

**Acceptance Criteria**:
- [ ] Tests for all ops::math (12)
- [ ] Tests for all ops::reduce (3)
- [ ] Tests for all ops::activation (5)
- [ ] Tests for all ops::loss (3)
- [ ] Large buffer tests (n > 10000)
- [ ] Performance benchmarks

**Files**: Create `crates/hologram-core/tests/phase9_integration.rs`

---

## Phase 5: Documentation and Cleanup

### Task 5.1: Update Atlas Backends SPEC.md

**Priority**: P1
**Owner**: atlas-backends
**Spec Impact**: 🟡
**Dependencies**: All Phase 2 tasks

**Updates**:
- Implementation status (Phase 8-9 complete)
- Address mode semantics
- Control flow requirements
- Benchmark results

**Acceptance Criteria**:
- [ ] Status reflects completion
- [ ] All new features documented
- [ ] Code examples provided
- [ ] Performance characteristics updated

**Files**: `crates/atlas-backends/SPEC.md`

---

### Task 5.2: Update Hologram-Core SPEC.md

**Priority**: P1
**Owner**: hologram-core
**Spec Impact**: 🟡
**Dependencies**: All Phase 3 tasks

**Updates**:
- ISA program generation approach
- ProgramBuilder API
- Architecture diagram
- Migration guide

**Acceptance Criteria**:
- [ ] Phase 9 implementation documented
- [ ] ProgramBuilder API documented
- [ ] Migration guide provided
- [ ] Examples for common patterns

**Files**: `crates/hologram-core/SPEC.md`

---

### Task 5.3: Remove Dead Code and Scaffolding

**Priority**: P2
**Owner**: atlas-backends, hologram-core
**Spec Impact**: 🟢
**Dependencies**: All previous tasks

**Cleanup**:
- Remove `#[allow(dead_code)]` scaffolding
- Remove "Phase X:" comments
- Remove `operation_api` feature flag
- Remove legacy test modules

**Acceptance Criteria**:
- [ ] All scaffolding removed
- [ ] cargo clippy passes
- [ ] All tests pass

**Files**:
- `crates/atlas-backends/src/cpu.rs`
- `crates/atlas-backends/Cargo.toml`
- `crates/hologram-core/src/executor.rs`

---

## Critical Path

**Must complete first** (P0):
1. Task 1.1: Program struct with labels
2. Task 1.3: ProgramError type
3. Task 2.1: CVT completion (40% gap)
4. Task 2.2: PhiCoordinate addressing
5. Task 2.6: Control flow instructions
6. Task 3.1: ProgramBuilder
7. Task 3.2: Rewrite ops::math

**Parallel tracks**:
- Task 1.2 (docs) + Tasks 2.3-2.5 (types/addressing)
- Tasks 3.3-3.4 after Task 3.1
- Tasks 4.1-4.2 as features complete

---

## Spec Change Summary

**🔴 BREAKING CHANGES**:
- Task 1.1: Program type structure

**🟡 SPEC CLARIFICATIONS**:
- Task 1.2: Address mode semantics
- Task 5.1: Atlas Backends SPEC.md
- Task 5.2: Hologram-Core SPEC.md

**🟢 NO SPEC IMPACT**: All other tasks

---

**Document Owner**: Atlas Backends Team
**Last Updated**: 2025-10-21
**Version**: 1.1 (Streamlined)
