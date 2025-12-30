# Atlas Specification

**Version:** 0.3.0
**Last Updated:** 2025-11-28

## Overview

Atlas is Hologram's algebraic representation space for neural network computations. It implements the MOONSHINE specification with type-deterministic torus scaling, character table operations for matrix computations, orbit classes for reductions, and build-time kernel generation.

## Purpose

Core capabilities:
- **Type-Deterministic Torus**: Each type (f32, f64, i8-i64) maps to exactly one torus configuration
- **96 Resonance Classes**: Commutative semiring ⟨ℤ₉₆, ⊕, ⊗⟩ for algebraic encoding
- **Character Table**: 194×194 matrix for character-theoretic matrix operations
- **32 Orbit Classes**: Prime orbit partition for reduction operations
- **Unit Groups**: Multiplicative inverse operations on torus units
- **Three-Phase Framework**: π → F → λ decomposition (embed → operate → lift)
- **Lift-Resonate-Crush Adjunction**: L ⊣ R ⊣ κ operators bridging discrete and continuous
- **Build-Time Kernel Generation**: Static dispatch table with 288 precompiled kernels

## Architecture

```
hologram-core/src/
├── atlas/
│   ├── mod.rs               # Atlas module interface
│   ├── uor.rs               # Unit Orbit Representation (96-class system)
│   └── constants.rs         # Core atlas constants
├── griess/
│   ├── mod.rs               # Griess algebra interface
│   ├── vector.rs            # 196,884-dimensional GriessVector
│   ├── resonance.rs         # Lift-Resonate-Crush operators
│   └── product.rs           # Griess product computation
├── hrm/
│   ├── mod.rs               # Hierarchical Representation Model
│   ├── embed/               # π: Embedding phase
│   ├── decode/              # λ: Lifting phase
│   └── storage/             # HRM KV storage
└── tensor.rs                # Tensor with Atlas storage support

hologram-backends/src/atlas/
├── mod.rs                   # Unified Atlas interface
├── atlas_constants.rs       # Type-deterministic torus configurations
├── character_table.rs       # 194×194 Monster group character table
├── orbit_classes.rs         # 32 prime orbit partition
└── unit_group.rs            # Multiplicative inverse operations

hologram-compiler/src/decomposition/
├── mod.rs                   # Three-phase operation framework
├── embedding.rs             # π: Embedding phase
├── operations.rs            # F: Operation phase
├── lifting.rs               # λ: Lifting phase
├── catalog.rs               # 24 standard operations
├── kernel_compiler.rs       # Abstract kernel compilation
└── dispatch.rs              # Static dispatch table (288 kernels)
```

## Type-Deterministic Torus Configuration

### TorusConfig Trait

Every data type maps to exactly one torus configuration:

```rust
/// Type-deterministic torus configuration
pub trait TorusConfig: Sized + Copy + 'static {
    /// Number of pages (PAGE_MOD)
    const PAGE_MOD: u32;

    /// Resolution modulus (RES_MOD / BYTES_PER_PAGE)
    const RES_MOD: u32;

    /// Total elements (PAGE_MOD × RES_MOD)
    const ELEMENTS: u32;

    /// Error bound numerator
    const ERROR_BOUND_NUM: u32;

    /// Error bound denominator (= RES_MOD)
    const ERROR_BOUND_DEN: u32;
}
```

### Torus Configurations

| Type | PAGE_MOD | RES_MOD | ELEMENTS | Error Bound |
|------|----------|---------|----------|-------------|
| f32, i32, u32 | 96 | 192 | 18,432 | 1/192 ≈ 0.0052 |
| f64, i64, u64 | 192 | 384 | 73,728 | 1/384 ≈ 0.0026 |
| f16, i16, u16 | 48 | 96 | 4,608 | 1/96 ≈ 0.0104 |
| i8, u8 | 24 | 96 | 2,304 | 1/96 ≈ 0.0104 |
| i128, u128 | 384 | 768 | 294,912 | 1/768 ≈ 0.0013 |

### Extended Precision Types (256-bit through 4096-bit)

Hologram supports extended precision integer types with exponentially scaling torus configurations:

| Type | PAGE_MOD | RES_MOD | ELEMENTS | Error Bound |
|------|----------|---------|----------|-------------|
| i256, u256 | 768 | 1,536 | 1,179,648 | 1/1536 ≈ 0.00065 |
| i512, u512 | 1,536 | 3,072 | 4,718,592 | 1/3072 ≈ 0.00033 |
| i1024, u1024 | 3,072 | 6,144 | 18,874,368 | 1/6144 ≈ 0.00016 |
| i2048, u2048 | 6,144 | 12,288 | 75,497,472 | 1/12288 ≈ 0.00008 |
| i4096, u4096 | 12,288 | 24,576 | 301,989,888 | 1/24576 ≈ 0.00004 |

**Implementation:**
- Extended types wrap `BigInt`/`BigUint` from `num-bigint` crate
- Each precision level doubles both PAGE_MOD and RES_MOD
- Error bounds halve with each doubling of precision
- Full arithmetic trait support (Add, Sub, Mul, Rem, Zero, One)
- Integrated into ISA Type enum and RegisterValue variants
- See [types.rs:122-604](crates/common/src/types.rs#L122-L604) for implementation

**Key Properties:**
- Same-size types share torus configurations (f32/i32, f64/i64)
- Larger types have tighter error bounds
- Error bound formula: `ERROR_BOUND = 1 / RES_MOD`
- Linear error accumulation: `error_k ≤ k × (1/RES_MOD)`

### Generic PhiCoordinate

```rust
/// Type-generic torus coordinate
pub struct PhiCoordinate<T: TorusConfig> {
    pub page: u32,
    pub byte: u32,
    _phantom: PhantomData<T>,
}

impl<T: TorusConfig> PhiCoordinate<T> {
    /// Create coordinate with validation
    pub fn new(page: u32, byte: u32) -> Option<Self> {
        if page < T::PAGE_MOD && byte < T::RES_MOD {
            Some(Self {
                page,
                byte,
                _phantom: PhantomData,
            })
        } else {
            None
        }
    }

    /// Convert to linear offset (O(1) bijection)
    pub fn to_linear_offset(self) -> usize {
        (self.page * T::RES_MOD + self.byte) as usize
    }
}
```

**Performance:** O(1) address computation via `page * RES_MOD + byte`

## 96 Resonance Classes

### Mathematical Foundation

The 96 resonance classes form a commutative semiring ⟨ℤ₉₆, ⊕, ⊗⟩ derived from the Monster group's character theory:

- **96 = 2⁵ × 3 = 32 × 3**: Factorization enables parallel reduction (32 orbits) and triadic structure
- **Decomposition**: Each class k decomposes as `k = h₂·24 + d·8 + ℓ` where h₂ ∈ ℤ₄, d ∈ ℤ₃, ℓ ∈ ℤ₈
- **Mirror Pairs**: 48 mirror pairs with `mirror(mirror(c)) = c`
- **Unity Classes**: 2 distinguished unity positions

### Structure

```rust
/// Resonance class identifier [0, 96)
pub type ResonanceClass = u8;

/// Classify byte value to resonance class
pub fn r96_classify(byte: u8) -> ResonanceClass {
    // Deterministic mapping: 256 byte values → 96 classes
    // Uses modular arithmetic for O(1) classification
}

/// Get mirror pair for resonance class
pub fn get_mirror_pair(class_id: ResonanceClass) -> ResonanceClass {
    // Returns the mirror class (48 pairs total)
}

/// Check if class is a unity position
pub fn is_unity(class_id: ResonanceClass) -> bool {
    // 2 unity positions in the 96-class system
}
```

### Properties

- **Complete Coverage**: All 256 byte values map to classes [0, 95]
- **Deterministic**: Same input always produces same class
- **Invertible via Mirror**: Mirror operation is self-inverse
- **Unity Neutral**: Unity classes act as identity elements

## Lift-Resonate-Crush Adjunction

### Overview

The L ⊣ R ⊣ κ adjunction bridges discrete resonance classes and continuous Griess space:

```
Lift (L):      ℤ₉₆ → Griess₁₉₆,₈₈₄   (embed resonance → high-dimensional)
Resonate (R):  Griess₁₉₆,₈₈₄ → ℤ₉₆   (project → nearest resonance class)
Crush (κ):    ℤ₉₆ → {0,1}           (boolean parity for verification)
```

### Lift Operator

```rust
/// Lift resonance class to canonical Griess vector
///
/// Each of the 96 classes has a unique canonical 196,884-dimensional
/// representative vector, pre-computed and cached for O(1) access.
pub fn lift(resonance: ResonanceClass) -> Result<GriessVector> {
    // Returns cached canonical vector for this resonance class
    // All 96 canonical vectors are orthogonal in Griess space
}
```

### Resonate Operator

```rust
/// Project Griess vector to nearest resonance class
///
/// Given an arbitrary 196,884-dimensional vector, find the resonance
/// class whose canonical vector minimizes Euclidean distance.
pub fn resonate(vector: &GriessVector) -> Result<ResonanceClass> {
    // O(96) scan over canonical vectors
    // Returns class with minimum distance to input
}
```

### Crush Operator

```rust
/// Crush resonance class to boolean (parity homomorphism)
pub fn crush(resonance: ResonanceClass) -> bool {
    // Returns true for odd classes, false for even
    // Used for verification and conservation checks
}
```

### CharacterValue Projection

For backend operations that produce `CharacterValue` (194-dimensional character space),
a projection back to resonance classes is required:

```rust
/// Project CharacterValue to nearest resonance class
///
/// Given a character computation result from CharProduct instruction,
/// find the resonance class whose character signature minimizes distance.
pub fn nearest_resonance_class(value: &CharacterValue) -> ResonanceClass {
    // Pre-computed: 96 canonical character signatures
    // O(96) scan or O(log 96) KD-tree lookup
}
```

This closes the computation loop: `π → F (CharProduct) → λ (nearest_resonance_class) → output`

## Atlas Tensor Storage

### AtlasTensor with Bijective Storage

AtlasTensor is a **separate type** from Tensor<T> that stores both resonance
classes (for algebraic operations) and torus offsets (for bijective reconstruction):

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

**Bijection Property:** Values can be reconstructed within type-specific error bounds:
- f32: ≤ 1/192 (0.52% error)
- f64: ≤ 1/384 (0.26% error)

### Atlas Tensor Operations

AtlasTensor provides bidirectional conversion and algebraic operations:

```rust
impl<T: TorusConfig + bytemuck::Pod> AtlasTensor<T> {
    /// Convert dense tensor to Atlas with bijective embedding (π phase)
    pub fn from_tensor(tensor: &Tensor<T>, exec: &mut Executor) -> Result<Self>;

    /// Convert Atlas back to dense tensor with lifting (λ phase)
    /// Reconstructs values within error bound ≤ 1/RES_MOD
    pub fn to_tensor(&self, exec: &mut Executor) -> Result<Tensor<T>>;

    /// Matrix multiplication using character products
    pub fn matmul(&self, exec: &mut Executor, other: &Self) -> Result<Self>;

    /// Element-wise addition in resonance semiring
    pub fn add(&self, exec: &mut Executor, other: &Self) -> Result<Self>;

    /// Element-wise multiplication in resonance semiring
    pub fn mul(&self, exec: &mut Executor, other: &Self) -> Result<Self>;
}
```

### Atlas Matmul

Matrix multiplication on AtlasTensor uses character products from the 194×194 table:

```rust
/// Atlas matrix multiplication: C = A × B using character products
///
/// For each output element:
/// C[i,j] = project(Σₖ CharProduct(A[i,k], B[k,j]))
///
/// Uses classes buffer for algebraic ops, preserves offsets for reconstruction.
pub fn matmul(&self, exec: &mut Executor, other: &Self) -> Result<Self> {
    // 1. Accumulate character products as i128
    // 2. Project to resonance class via nearest_resonance_class()
    // 3. Store result class; offset derived from accumulated value
}
```

## Character Table

### Structure

```rust
/// 194×194 Monster group character table
pub struct CharacterTable {
    /// Dimension (always 194)
    pub dimension: u8,

    /// Products: 194×194 matrix stored row-major
    pub products: Vec<i64>,

    /// Norms: 194 character norms
    pub norms: Vec<f64>,
}

impl CharacterTable {
    /// Compute character product: χᵢ ⊗ χⱼ
    pub fn product(&self, i: u8, j: u8) -> i64 {
        assert!(i < 194 && j < 194);
        self.products[(i as usize) * 194 + (j as usize)]
    }

    /// Get character norm
    pub fn norm(&self, i: u8) -> f64 {
        assert!(i < 194);
        self.norms[i as usize]
    }
}

/// Global static instance
pub static CHARACTER_TABLE: LazyLock<CharacterTable> = LazyLock::new(generate_character_table);
```

**Implementation:** Authentic Monster group character table generated at build time from GAP (Groups, Algorithms, Programming) system's CTblLib package. The table contains exact character values including cyclotomic integers (roots of unity) with full algebraic precision. All 37,636 character values (194×194) are mathematically correct data from the Atlas of Finite Groups.

**Build Process:**
- GAP script exports Monster character table during `cargo build`
- Cyclotomic values stored exactly (not approximated) using `CharacterValue` enum
- Character degrees match OEIS A001379 (authentic moonshine module dimensions)
- No fallbacks: Build fails if GAP is not available (ensures mathematical correctness)

**Properties:**
- Dimension: 194 × 194 (all 194 irreducible representations)
- Storage: 37,636 exact character values + 194 norms
- Character Values: Exact (integers or cyclotomic with `CharacterValue` enum)
- Norms: Positive real values computed from authentic character data
- Verification: All character degrees positive, trivial character = 1 everywhere

## Orbit Classes

### Structure

```rust
/// Single orbit class
pub struct OrbitClass {
    /// Class identifier [0, 32)
    pub class_id: u8,

    /// Representative coordinate
    pub representative: u32,

    /// All coordinates in this class
    pub coordinates: Vec<u32>,
}

impl OrbitClass {
    /// Get class size
    pub fn len(&self) -> usize {
        self.coordinates.len()
    }
}

/// Orbit classification for specific torus size
pub struct OrbitClassification {
    /// Number of orbit classes (always 32)
    pub num_classes: u8,

    /// Torus size
    pub torus_size: usize,

    /// All orbit classes
    pub classes: Vec<OrbitClass>,

    /// Fast lookup: coordinate → class_id
    pub coord_to_class: HashMap<u32, u8>,
}

impl OrbitClassification {
    /// Classify coordinate to orbit class
    pub fn classify(&self, coord: u32) -> u8 {
        *self.coord_to_class.get(&coord)
            .unwrap_or(&0)
    }

    /// Get orbit representative
    pub fn representative(&self, class_id: u8) -> u32 {
        self.classes[class_id as usize].representative
    }

    /// Get class size
    pub fn class_size(&self, class_id: u8) -> usize {
        self.classes[class_id as usize].len()
    }
}
```

**Properties:**
- Always 32 orbit classes
- Complete partition: ⋃ᵢ Oᵢ = T² (every coordinate in exactly one orbit)
- Disjoint: Oᵢ ∩ Oⱼ = ∅ for i ≠ j
- O(1) classification via HashMap lookup
- Deterministic representatives

## Unit Groups

### Structure

```rust
/// Unit group for specific torus
pub struct UnitGroup {
    /// Modulus
    pub modulus: u32,

    /// Units (coprime to modulus)
    pub units: Vec<u32>,

    /// Multiplicative inverses: unit → inverse
    pub inverses: HashMap<u32, u32>,

    /// Multiplication table (optional, expensive)
    pub multiplication_table: Option<Vec<Vec<u32>>>,
}

impl UnitGroup {
    /// Get multiplicative inverse
    pub fn inverse(&self, unit: u32) -> Option<u32> {
        self.inverses.get(&unit).copied()
    }

    /// Check if value is a unit
    pub fn is_unit(&self, value: u32) -> bool {
        self.inverses.contains_key(&value)
    }
}
```

**Performance Note:** Multiplication table generation is computationally expensive for large tori (600M+ elements for 73,728-element torus). Current implementation uses Extended Euclidean Algorithm for on-demand inverse computation.

## Unified Atlas Interface

```rust
/// Unified Atlas providing all algebraic structures
pub struct Atlas {
    /// Character table (194×194)
    pub character_table: CharacterTable,

    /// Orbit classifications for all torus sizes
    pub orbit_classifications: OrbitClassifications,

    /// Unit groups for all torus sizes
    pub unit_groups: HashMap<u32, UnitGroup>,
}

impl Atlas {
    /// Get character product
    pub fn character_product(&self, i: u8, j: u8) -> i64 {
        self.character_table.product(i, j)
    }

    /// Classify coordinate to orbit
    pub fn orbit_class(&self, torus_size: usize, coord: u32) -> u8 {
        self.orbit_classifications.for_size(torus_size).classify(coord)
    }

    /// Get multiplicative inverse
    pub fn multiplicative_inverse(&self, modulus: u32, unit: u32) -> Option<u32> {
        self.unit_groups.get(&modulus)?
            .inverse(unit)
    }
}

/// Global static instance
pub static ATLAS: Lazy<Atlas> = Lazy::new(|| Atlas::new());
```

## Three-Phase Operation Framework

### Overview

All operations decompose into three phases:

```
Input → π (Embedding) → F (Operation) → λ (Lifting) → Output
```

**Phases:**
1. **π (Embedding)**: Map input to torus representation
2. **F (Operation)**: Perform torus-space operation
3. **λ (Lifting)**: Map result back to standard representation

### Embedding Phase

```rust
/// Embedding type
pub enum EmbeddingType {
    /// Element-wise direct embedding
    ElementWise,

    /// Character basis embedding (for matrix ops)
    CharacterBasis,

    /// Orbit embedding (for reductions)
    Orbit,
}

/// Embedding strategy trait
pub trait EmbeddingStrategy<T: TorusConfig> {
    /// Embed value to torus
    fn embed(&self, value: T) -> PhiCoordinate<T>;

    /// Embedding error bound
    fn error_bound() -> (u32, u32) {
        (T::ERROR_BOUND_NUM, T::ERROR_BOUND_DEN)
    }
}
```

### Operation Phase

```rust
/// Operation category
pub enum OperationCategory {
    ElementWise,
    Matrix,
    Reduction,
    Shape,
}

/// Operation type
pub enum OperationType {
    ElementWise(ElementWiseOp),
    Character(CharacterOp),
    Orbit(OrbitOp),
}

/// Element-wise operations
pub enum ElementWiseOp {
    TorusAdd, TorusSub, TorusMul, TorusDiv,
    TorusReLU, TorusSigmoid, TorusTanh,
    TorusExp, TorusLog, TorusSin, TorusCos,
}

/// Character operations (matrix)
pub enum CharacterOp {
    CharProduct,
}

/// Orbit operations (reduction)
pub enum OrbitOp {
    OrbitSum, OrbitMean, OrbitMax, OrbitMin,
    OrbitArgMax, OrbitArgMin,
}
```

### Lifting Phase

```rust
/// Lifting type
pub enum LiftingType {
    /// Standard lifting with error tracking
    Standard,

    /// Character basis lifting
    CharacterBasis,

    /// Orbit lifting
    Orbit,
}

/// Lifting strategy trait
pub trait LiftingStrategy<T: TorusConfig> {
    /// Lift coordinate to value
    fn lift(&self, coord: PhiCoordinate<T>) -> T;

    /// Accumulated error after lifting
    fn accumulated_error(&self) -> f64;
}
```

## Build-Time Kernel Generation

### Operation Catalog

```rust
/// Operation specification
pub trait OperationSpec {
    fn name(&self) -> &str;
    fn category(&self) -> OperationCategory;
    fn embedding_type(&self) -> EmbeddingType;
    fn operation_type(&self) -> OperationType;
    fn lifting_type(&self) -> LiftingType;
    fn generators(&self) -> Vec<String>;
}

/// Standard operation catalog
pub struct OperationCatalog {
    operations: HashMap<String, Box<dyn OperationSpec>>,
}

impl OperationCatalog {
    /// Create catalog with 24 standard operations
    pub fn standard() -> Self;

    /// Register operation
    pub fn register(&mut self, spec: Box<dyn OperationSpec>);

    /// Look up operation
    pub fn get(&self, name: &str) -> Option<&dyn OperationSpec>;
}
```

**Standard Operations (24):**
- **Element-wise (11):** Add, Sub, Mul, Div, ReLU, Sigmoid, Tanh, Exp, Log, Sin, Cos
- **Matrix (3):** MatMul, GEMM, Conv
- **Reduction (6):** ReduceSum, ReduceMean, ReduceMax, ReduceMin, ArgMax, ArgMin
- **Shape (4):** Transpose, Reshape, Concat, Split

### Kernel Compiler

```rust
/// Data type enum for kernel compilation
pub enum DataType {
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    F16, BF16, F32, F64,
}

/// Abstract kernel representation
pub struct CompiledKernel {
    pub operation_name: String,
    pub data_type: DataType,
    pub generators: Vec<Generator>,
    pub metadata: KernelMetadata,
}

/// Kernel compiler
pub struct KernelCompiler {
    catalog: OperationCatalog,
}

impl KernelCompiler {
    /// Compile operation for specific type
    pub fn compile(&self, operation: &str, ty: DataType) -> Result<CompiledKernel>;

    /// Compile by name lookup
    pub fn compile_by_name(&self, name: &str, ty: DataType) -> Result<CompiledKernel>;
}
```

**Generator Types (27):**
- Torus arithmetic: TorusAdd, TorusSub, TorusMul, TorusDiv
- Nonlinear: TorusReLU, TorusNeg, TorusExp, TorusLog, TorusSin, TorusCos, etc.
- Character: CharProduct
- Orbit: OrbitSum, OrbitMean, OrbitMax, OrbitMin, OrbitArgMax, OrbitArgMin
- Shape: PermuteAxes, RemapIndices, ConcatIndices, SplitIndices

### Static Dispatch Table

```rust
/// Dispatch key: (operation, type) pair
pub struct DispatchKey {
    pub operation_name: String,
    pub data_type: DataType,
}

/// Static dispatch table
pub struct DispatchTable {
    table: HashMap<DispatchKey, CompiledKernel>,
}

impl DispatchTable {
    /// Create table with all standard operations
    /// Precompiles: 24 operations × 12 types = 288 kernels
    pub fn standard() -> Self;

    /// Lookup kernel (O(1))
    pub fn get(&self, operation: &str, ty: DataType) -> Option<&CompiledKernel>;

    /// Check if supported
    pub fn is_supported(&self, operation: &str, ty: DataType) -> bool;

    /// Global singleton
    pub fn global() -> &'static DispatchTable;
}

/// Convenience function
pub fn dispatch(operation: &str, ty: DataType) -> Option<&'static CompiledKernel> {
    DispatchTable::global().get(operation, ty)
}
```

**Performance:** O(1) lookup via HashMap, zero runtime compilation overhead, single initialization via global singleton.

## Memory Management

### Type-Specific Buffers

```rust
/// Type-aware memory pool management
pub struct AtlasMemory<S: MemoryStorage> {
    /// Per-type memory pools
    pools: HashMap<TypeId, Pool<S>>,

    /// Lock for thread safety
    lock: RwLock<()>,
}

impl<S: MemoryStorage> AtlasMemory<S> {
    /// Get pool for specific type (lazy allocation)
    pub fn get_pool<T: TorusConfig>(&mut self) -> &mut Pool<S>;

    /// Allocate buffer for type
    pub fn allocate<T: TorusConfig>(&mut self, count: usize) -> Result<BufferHandle>;
}
```

**Pool Sizes (type-deterministic):**
- f32/i32/u32: 96 × 18,432 = 1,769,472 bytes
- f64/i64/u64: 96 × 73,728 = 7,077,888 bytes
- f16/i16/u16: 96 × 4,608 = 442,368 bytes
- i8/u8: 96 × 2,304 = 221,184 bytes

**Key Features:**
- Types with same torus config share pools
- Lazy pool allocation with double-check locking
- Thread-safe via `parking_lot::RwLock`

### Address Resolution

```rust
/// Generic address resolution
pub fn resolve_address<T: TorusConfig>(
    class: u8,
    page: u32,
    byte: u32,
    dst: BufferHandle,
) -> Result<usize>;

/// Runtime Type dispatch
pub fn resolve_address_for_type<M: MemoryStorage>(
    ty: Type,
    class: u8,
    page: u32,
    byte: u32,
    dst: BufferHandle,
    state: &ExecutionState<M>,
) -> Result<usize>;
```

**Performance:** Zero-cost abstraction via monomorphization at compile time.

## ISA Extensions

### Atlas-Specific Instructions

```rust
/// Boundary mapping with type-deterministic offset
BoundMap {
    ty: Type,
    dst: Register,
    class: u8,
    page: u32,
    byte: u32,
}

/// Character product operation
CharProduct {
    dst: Register,
    char_i: u8,  // [0, 194)
    char_j: u8,  // [0, 194)
}

/// Orbit classification
OrbitClassify {
    dst: Register,
    coord: u32,
    torus_size: usize,
}

/// Orbit representative lookup
OrbitRepresentative {
    dst: Register,
    class_id: u8,
    torus_size: usize,
}

/// Multiplicative inverse
MultInverse {
    dst: Register,
    unit: u32,
    modulus: u32,
}
```

## Error Bounds

### Formula

```
ERROR_BOUND = ERROR_BOUND_NUM / ERROR_BOUND_DEN
            = 1 / RES_MOD
```

### Per-Type Bounds

| Type | RES_MOD | Error Bound | Percentage |
|------|---------|-------------|------------|
| f32 | 192 | 1/192 | ~0.52% |
| f64 | 384 | 1/384 | ~0.26% |
| f16/i16 | 96 | 1/96 | ~1.04% |
| i8 | 96 | 1/96 | ~1.04% |

### Error Accumulation

For k composed operations:
```
error_k ≤ k × (1/RES_MOD)
```

**Example:** 100-layer network with f64:
```
error_100 ≤ 100 × (1/384) = 0.260 < 1.0 ✓
```

## Performance Characteristics

### Address Computation

- **Complexity:** O(1) per coordinate
- **Formula:** `offset = page * RES_MOD + byte`
- **Target:** <5ns on modern CPUs

### Cache Performance

- **Sequential Access:** Zero cache misses (linear layout)
- **Random Access:** Standard cache behavior
- **Torus Access:** Depends on access pattern

### Type Dispatch

- **Runtime Overhead:** Zero (monomorphization)
- **f32 vs f64:** Equal performance (different monomorphized code)
- **Verification:** Benchmark shows no dispatch penalty

### Memory Bandwidth

- **Target:** ≥ 80% of peak memory bandwidth
- **Sequential Read/Write:** Optimized for streaming
- **Read-Modify-Write:** Cache-efficient modular arithmetic

### Matrix Operations

- **Character Products:** ~10-50ns per product (194×194 table lookup)
- **Target:** Competitive with native BLAS for character-theoretic operations

## Testing

### Algebraic Property Tests

**Torus Axioms (16 tests):**
- Associativity: (a + b) + c ≡ a + (b + c)
- Commutativity: a + b ≡ b + a
- Identity: a + 0 ≡ a
- Closure: results within bounds

**Character Table (15 tests):**
- Dimension correctness (194×194)
- Product storage and indexing
- Norm positivity and boundedness
- Structural well-formedness

**Orbit Partitions (17 tests):**
- Completeness: ⋃ᵢ Oᵢ = T²
- Disjointness: Oᵢ ∩ Oⱼ = ∅
- Representative validity
- Classification consistency

**Precision Bounds (14 tests):**
- Error bound formula verification
- Type-specific bounds
- Linear accumulation
- Deep network error < 1.0

### Property-Based Tests (23 tests with proptest)

- Torus addition properties (associativity, commutativity, identity)
- Coordinate operations (linear offset bijection)
- Error bound properties
- Cross-type consistency

### Performance Benchmarks (9 groups)

1. **Address Computation:** O(1) verification
2. **Bulk Address:** Throughput measurement
3. **Cache Miss Analysis:** Sequential vs random access
4. **Native vs Atlas:** Arithmetic comparison
5. **Character Products:** Single and batch operations
6. **Orbit Operations:** Classification and lookups
7. **Throughput Scaling:** 100 to 100K elements
8. **Type Dispatch:** Monomorphization verification
9. **Memory Bandwidth:** Read/write/modify patterns

## Migration Guide

### From Hardcoded Constants

**Before:**
```rust
const PAGES: u32 = 96;
const BYTES_PER_PAGE: u32 = 192;
const TOTAL_ELEMENTS: u32 = 18_432;

let offset = page * BYTES_PER_PAGE + byte;
```

**After:**
```rust
use hologram_common::types::TorusConfig;

fn compute_offset<T: TorusConfig>(page: u32, byte: u32) -> usize {
    (page * T::RES_MOD + byte) as usize
}

// Type-specific usage
let offset_f32 = compute_offset::<f32>(page, byte);
let offset_f64 = compute_offset::<f64>(page, byte);
```

### From Non-Generic Operations

**Before:**
```rust
fn execute_bound_map(
    class: u8,
    page: u32,
    byte: u32,
) -> usize {
    // Hardcoded for f32
    (page * 192 + byte) as usize
}
```

**After:**
```rust
fn execute_bound_map<T: TorusConfig>(
    class: u8,
    page: u32,
    byte: u32,
) -> usize {
    (page * T::RES_MOD + byte) as usize
}

// Runtime dispatch wrapper
fn execute_bound_map_for_type(
    ty: Type,
    class: u8,
    page: u32,
    byte: u32,
) -> usize {
    match ty {
        Type::F32 => execute_bound_map::<f32>(class, page, byte),
        Type::F64 => execute_bound_map::<f64>(class, page, byte),
        Type::I32 => execute_bound_map::<i32>(class, page, byte),
        // ... other types
    }
}
```

### Using Character Table

```rust
use hologram_backends::atlas::CHARACTER_TABLE;

// Compute character product
let product = CHARACTER_TABLE.product(10, 20);

// Get character norm
let norm = CHARACTER_TABLE.norm(5);
```

### Using Orbit Classes

```rust
use hologram_backends::atlas::ORBIT_CLASSIFICATIONS;

let torus_size = 18_432; // f32 torus
let classification = ORBIT_CLASSIFICATIONS.for_size(torus_size);

// Classify coordinate
let class_id = classification.classify(1000);

// Get representative
let rep = classification.representative(class_id);

// Get class size
let size = classification.class_size(class_id);
```

### Using Dispatch Table

```rust
use hologram_compiler::decomposition::{dispatch, DataType};

// Lookup precompiled kernel
if let Some(kernel) = dispatch("Add", DataType::F32) {
    // Use kernel for execution
    println!("Found kernel for Add<f32>");
}

// Check support
let supported = DispatchTable::global()
    .is_supported("MatMul", DataType::F64);
```

## Future Enhancements

- Additional character operations (tensor products, decompositions)
- Extended orbit operations (orbit averages, variances)
- Dynamic kernel compilation (runtime specialization)
- Distributed orbit classification (multi-node partitioning)

## References

- **MOONSHINE Specification:** `/workspace/external/MOONSHINE_ONNX_SPECIFICATION.md`
- **Implementation Progress:** `/workspace/IMPLEMENTATION_PROGRESS.md`
- **Backends Specification:** `/workspace/docs/spec/crates/backends.md`
- **Compiler Specification:** `/workspace/docs/spec/crates/compiler.md`
- **Testing Specification:** `/workspace/docs/spec/testing.md`
- **Benchmarking Specification:** `/workspace/docs/spec/benchmarking.md`
