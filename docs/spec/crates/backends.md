# hologram-backends Specification

**Status:** Approved
**Version:** 0.1.0
**Last Updated:** 2025-01-18

## Overview

`hologram-backends` provides the Atlas ISA (Instruction Set Architecture) and backend implementations for executing operations on different hardware (CPU, GPU, WASM, etc.).

## Purpose

Core responsibilities:
- Atlas ISA definition (50+ instructions)
- Atlas algebraic structures (character table, orbit classes, unit groups)
- Backend trait abstraction
- Backend implementations (CPU, CUDA, Metal, WASM, WebGPU)
- Circuit → ISA translation
- JSON kernel → ISA translation
- Pool storage (O(1) space streaming)
- Program caching (Blake3-based)

**Note:** For complete Atlas algebraic structures documentation (type-deterministic torus configuration, character table, orbit classes, unit groups, three-phase operation framework, and build-time kernel generation), see [Atlas Specification](../atlas.md).

## Architecture

```
hologram-backends
├── ISA (Atlas Instruction Set)
│   ├── Instructions (50+ opcodes)
│   ├── Types (F16, F32, F64, I8-I64, U8-U64)
│   ├── Address modes
│   └── Program representation
├── Backend Trait
│   ├── Program execution
│   ├── Buffer management
│   └── Pool management
├── Backend Implementations (in backends/ subdirectory)
│   ├── CPU (reference implementation + SIMD)
│   ├── CUDA (NVIDIA GPU)
│   ├── Metal (Apple GPU)
│   ├── WASM (WebAssembly CPU)
│   └── WebGPU (WASM GPU)
├── Translators
│   ├── Circuit → ISA
│   └── JSON → ISA
├── Pool Storage
│   └── O(1) space streaming
└── Program Utilities
    ├── Builder
    └── Cache (Blake3)
```

## Public API

### Atlas ISA

#### Instruction Types

```rust
/// Complete Atlas ISA
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    // === Data Movement ===
    /// Load from global memory
    LDG { dst: Register, src: Address, ty: Type },

    /// Store to global memory
    STG { dst: Address, src: Register, ty: Type },

    /// Load from shared memory
    LDS { dst: Register, src: Address, ty: Type },

    /// Store to shared memory
    STS { dst: Address, src: Register, ty: Type },

    /// Move between registers
    MOV { dst: Register, src: Register },

    /// Move immediate value
    MOV_IMM { dst: Register, value: Immediate },

    /// Type conversion
    CVT { dst: Register, src: Register, src_ty: Type, dst_ty: Type },

    // === Arithmetic ===
    /// Addition
    ADD { dst: Register, lhs: Register, rhs: Register },

    /// Subtraction
    SUB { dst: Register, lhs: Register, rhs: Register },

    /// Multiplication
    MUL { dst: Register, lhs: Register, rhs: Register },

    /// Division
    DIV { dst: Register, lhs: Register, rhs: Register },

    /// Fused multiply-add: dst = (a * b) + c
    FMA { dst: Register, a: Register, b: Register, c: Register },

    /// Minimum
    MIN { dst: Register, lhs: Register, rhs: Register },

    /// Maximum
    MAX { dst: Register, lhs: Register, rhs: Register },

    /// Absolute value
    ABS { dst: Register, src: Register },

    /// Negation
    NEG { dst: Register, src: Register },

    /// Remainder
    REM { dst: Register, lhs: Register, rhs: Register },

    // === Logic ===
    /// Bitwise AND
    AND { dst: Register, lhs: Register, rhs: Register },

    /// Bitwise OR
    OR { dst: Register, lhs: Register, rhs: Register },

    /// Bitwise XOR
    XOR { dst: Register, lhs: Register, rhs: Register },

    /// Bitwise NOT
    NOT { dst: Register, src: Register },

    /// Shift left
    SHL { dst: Register, src: Register, amount: Register },

    /// Shift right
    SHR { dst: Register, src: Register, amount: Register },

    /// Set predicate on comparison
    SETcc { pred: Predicate, lhs: Register, rhs: Register, cond: Condition },

    /// Select based on predicate
    SEL { dst: Register, pred: Predicate, true_val: Register, false_val: Register },

    // === Control Flow ===
    /// Conditional branch
    BRA { target: Label, pred: Option<Predicate> },

    /// Function call
    CALL { target: Label },

    /// Return from function
    RET,

    /// Loop start
    LOOP { counter: Register, iterations: usize },

    /// Exit/break
    EXIT,

    // === Synchronization ===
    /// Barrier synchronization
    BarSync { id: u8 },

    /// Memory fence
    MemFence { scope: MemoryScope },

    // === Atlas-Specific ===
    /// Get class index
    ClsGet { dst: Register, src: Register },

    /// Mirror pairing
    MIRROR { dst: Register, src: Register },

    /// Unity neutrality test
    UnityTest { pred: Predicate, src: Register },

    /// Neighbor operations
    NBRLoad { dst: Register, src: Register, offset: i8 },
    NBRStore { dst: Register, src: Register, offset: i8 },

    /// Resonance accumulation
    ResAccum { dst: Register, src: Register, class: u8 },

    /// Phase operations
    PhaseRead { dst: Register, class: u8 },
    PhaseWrite { src: Register, class: u8 },

    /// Boundary mapping
    BoundMap { dst: Register, src: Register },

    // === Reductions ===
    /// Reduce with addition
    ReduceAdd { dst: Register, src_array: Address, count: usize },

    /// Reduce with minimum
    ReduceMin { dst: Register, src_array: Address, count: usize },

    /// Reduce with maximum
    ReduceMax { dst: Register, src_array: Address, count: usize },

    /// Reduce with multiplication
    ReduceMul { dst: Register, src_array: Address, count: usize },

    // === Transcendentals ===
    /// Exponential (e^x)
    EXP { dst: Register, src: Register },

    /// Natural logarithm
    LOG { dst: Register, src: Register },

    /// Square root
    SQRT { dst: Register, src: Register },

    /// Sine
    SIN { dst: Register, src: Register },

    /// Cosine
    COS { dst: Register, src: Register },

    /// Tangent
    TAN { dst: Register, src: Register },

    /// Hyperbolic tangent
    TANH { dst: Register, src: Register },

    /// Sigmoid: 1 / (1 + e^(-x))
    SIGMOID { dst: Register, src: Register },

    // === Pool Storage ===
    /// Allocate pool
    PoolAlloc { handle: PoolHandle, size: usize },

    /// Free pool
    PoolFree { handle: PoolHandle },

    /// Load from pool
    PoolLoad { dst: Register, pool: PoolHandle, offset: usize },

    /// Store to pool
    PoolStore { pool: PoolHandle, offset: usize, src: Register },

    // === Higher-Order Operations ===
    /// Parallel map (unary operation across array)
    ParallelMap {
        dst_array: Address,
        src_array: Address,
        count: usize,
        operation: MapOperation,
    },

    /// Parallel map (binary operation across arrays)
    ParallelMapBinary {
        dst_array: Address,
        src_a: Address,
        src_b: Address,
        count: usize,
        operation: BinaryMapOperation,
    },

    /// Parallel reduce
    ParallelReduce {
        dst: Register,
        src_array: Address,
        count: usize,
        operation: ReduceOperation,
    },
}
```

#### Type System

```rust
/// Data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    // Floating point
    /// 16-bit float (IEEE 754 half precision)
    F16,
    /// 16-bit brain float
    BF16,
    /// 32-bit float (IEEE 754 single precision)
    F32,
    /// 64-bit float (IEEE 754 double precision)
    F64,

    // Signed integers
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 128-bit signed integer
    I128,
    /// 256-bit signed integer (extended precision)
    I256,
    /// 512-bit signed integer (extended precision)
    I512,
    /// 1024-bit signed integer (extended precision)
    I1024,
    /// 2048-bit signed integer (extended precision)
    I2048,
    /// 4096-bit signed integer (extended precision)
    I4096,

    // Unsigned integers
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// 128-bit unsigned integer
    U128,
    /// 256-bit unsigned integer (extended precision)
    U256,
    /// 512-bit unsigned integer (extended precision)
    U512,
    /// 1024-bit unsigned integer (extended precision)
    U1024,
    /// 2048-bit unsigned integer (extended precision)
    U2048,
    /// 4096-bit unsigned integer (extended precision)
    U4096,
}

impl Type {
    /// Get size in bytes
    pub fn size_bytes(&self) -> usize;

    /// Check if type is floating-point
    pub fn is_float(&self) -> bool;

    /// Check if type is integer
    pub fn is_integer(&self) -> bool;

    /// Check if type is signed
    pub fn is_signed(&self) -> bool;

    /// Check if type is unsigned
    pub fn is_unsigned(&self) -> bool;
}
```

**Extended Precision Types:**
- Types I256/U256 through I4096/U4096 provide arbitrary precision arithmetic
- Implemented using `BigInt`/`BigUint` wrappers from `num-bigint` crate
- Each precision level doubles the bit width (256, 512, 1024, 2048, 4096)
- Fully integrated with torus configurations (see [Atlas Specification](../atlas.md))
- See [types.rs](../../../crates/common/src/types.rs) for implementation

#### Address Modes

```rust
/// Memory address
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Address {
    /// Buffer + offset
    BufferOffset {
        handle: BufferHandle,
        offset: usize,
    },

    /// Pool + offset
    PoolOffset {
        handle: PoolHandle,
        offset: usize,
    },

    /// Shared memory offset
    Shared {
        offset: usize,
    },

    /// Class-based addressing
    Class {
        class_id: u8,
        offset: usize,
    },

    /// Register indirect
    Register {
        reg: Register,
    },
}
```

#### Map Operations

```rust
/// Unary map operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapOperation {
    Abs,
    Neg,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Relu,
    Sigmoid,
    Tanh,
}

/// Binary map operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryMapOperation {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
    Atan2,
}

/// Reduce operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOperation {
    Sum,
    Product,
    Min,
    Max,
}
```

#### Program Representation

```rust
/// ISA program
#[derive(Debug, Clone)]
pub struct Program {
    /// Instructions
    pub instructions: Vec<Instruction>,

    /// Entry point
    pub entry: Label,

    /// Metadata
    pub metadata: ProgramMetadata,
}

impl Program {
    /// Create empty program
    pub fn new() -> Self;

    /// Add instruction
    pub fn add(&mut self, instr: Instruction);

    /// Get instruction count
    pub fn len(&self) -> usize;

    /// Optimize program
    pub fn optimize(&mut self);
}
```

### Backend Trait

```rust
/// Backend execution interface
pub trait Backend: Send + Sync {
    /// Execute ISA program
    fn execute_program(
        &mut self,
        program: &Program,
        config: &LaunchConfig,
    ) -> Result<()>;

    /// Execute with parameters
    fn execute_program_with_params(
        &mut self,
        program: &Program,
        params: &ExecutionParams,
    ) -> Result<()>;

    // === Buffer Management ===

    /// Allocate buffer
    fn allocate_buffer(&mut self, size: usize) -> Result<BufferHandle>;

    /// Free buffer
    fn free_buffer(&mut self, handle: BufferHandle) -> Result<()>;

    /// Copy data to buffer
    fn copy_to_buffer(
        &mut self,
        handle: BufferHandle,
        data: &[u8],
    ) -> Result<()>;

    /// Copy data from buffer
    fn copy_from_buffer(
        &mut self,
        handle: BufferHandle,
        data: &mut [u8],
    ) -> Result<()>;

    /// Get buffer size
    fn buffer_size(&self, handle: BufferHandle) -> Result<usize>;

    // === Pool Management ===

    /// Allocate pool (O(1) space streaming)
    fn allocate_pool(&mut self, size: usize) -> Result<PoolHandle>;

    /// Free pool
    fn free_pool(&mut self, handle: PoolHandle) -> Result<()>;

    /// Copy to pool
    fn copy_to_pool(
        &mut self,
        handle: PoolHandle,
        offset: usize,
        data: &[u8],
    ) -> Result<()>;

    /// Copy from pool
    fn copy_from_pool(
        &mut self,
        handle: PoolHandle,
        offset: usize,
        data: &mut [u8],
    ) -> Result<()>;

    /// Get pool size
    fn pool_size(&self, handle: PoolHandle) -> Result<usize>;

    // === Type Casting ===

    /// Cast to any (for backend-specific features)
    fn as_any(&self) -> &dyn Any;

    /// Cast to mutable any
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
```

#### Launch Configuration

```rust
/// Execution configuration
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Grid dimensions (x, y, z)
    pub grid: (u32, u32, u32),

    /// Block dimensions (x, y, z)
    pub block: (u32, u32, u32),

    /// Shared memory size per block
    pub shared_memory: usize,
}

impl LaunchConfig {
    /// Create 1D configuration
    pub fn linear(total_threads: usize) -> Self;

    /// Create 2D configuration
    pub fn grid_2d(width: u32, height: u32) -> Self;

    /// Total threads
    pub fn total_threads(&self) -> usize;
}
```

### Backend Implementations

**CRITICAL: All backend implementations MUST be in `src/backends/` subdirectory**

#### CPU Backend

```rust
/// CPU backend (reference implementation + SIMD)
pub struct CpuBackend {
    buffers: DashMap<BufferHandle, Vec<u8>>,
    pools: DashMap<PoolHandle, Pool>,
    register_file: RegisterFile,
}

impl CpuBackend {
    /// Create CPU backend
    pub fn new() -> Self;

    /// Execute with SIMD optimization
    fn execute_simd(&mut self, program: &Program) -> Result<()>;

    /// Sequential execution (fallback)
    fn execute_sequential(&mut self, program: &Program) -> Result<()>;
}

impl Backend for CpuBackend {
    fn execute_program(&mut self, program: &Program, config: &LaunchConfig) -> Result<()> {
        // Try SIMD path for f32 operations ≤262K elements
        if self.can_use_simd(program) {
            self.execute_simd(program)
        } else {
            self.execute_sequential(program)
        }
    }

    // ... implement remaining trait methods
}
```

**SIMD Optimization:**
- Recognizes precompiled operations (ADD, SUB, MUL, DIV, ABS, NEG, RELU)
- Direct dispatch to SIMD kernels (AVX512/AVX2/SSE4.1)
- **Performance**: ~42ns execution vs ~1000ns ISA interpretation (23× faster)
- **Conditions**: f32 type, ≤262K elements, CPU backend

#### CUDA Backend

```rust
#[cfg(feature = "cuda")]
pub struct CudaBackend {
    device: CudaDevice,
    context: CudaContext,
    modules: HashMap<ProgramHash, CudaModule>,
}

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Create CUDA backend
    pub fn new(device_id: u32) -> Result<Self>;

    /// Compile program to PTX
    fn compile_to_ptx(&self, program: &Program) -> Result<String>;

    /// Launch kernel
    fn launch_kernel(&mut self, program: &Program, config: &LaunchConfig) -> Result<()>;
}
```

#### Metal Backend

```rust
#[cfg(feature = "metal")]
pub struct MetalBackend {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipelines: HashMap<ProgramHash, metal::ComputePipelineState>,
}

#[cfg(feature = "metal")]
impl MetalBackend {
    /// Create Metal backend
    pub fn new() -> Result<Self>;

    /// Compile program to MSL (Metal Shading Language)
    fn compile_to_msl(&self, program: &Program) -> Result<String>;
}
```

#### WASM Backend

```rust
#[cfg(target_arch = "wasm32")]
pub struct WasmBackend {
    buffers: HashMap<BufferHandle, Vec<u8>>,
    register_file: RegisterFile,
}

#[cfg(target_arch = "wasm32")]
impl WasmBackend {
    /// Create WASM backend
    pub fn new() -> Self;

    /// Sequential execution (no SIMD in WASM yet)
    fn execute_sequential(&mut self, program: &Program) -> Result<()>;
}
```

#### WebGPU Backend

```rust
#[cfg(feature = "webgpu")]
pub struct WebGpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: HashMap<ProgramHash, wgpu::ComputePipeline>,
}

#[cfg(feature = "webgpu")]
impl WebGpuBackend {
    /// Create WebGPU backend
    pub fn new() -> Result<Self>;

    /// Compile program to WGSL
    fn compile_to_wgsl(&self, program: &Program) -> Result<String>;

    /// Execute async
    pub async fn execute_async(&mut self, program: &Program) -> Result<()>;
}
```

### Translators

#### Circuit → ISA

```rust
/// Translate circuit to ISA program
pub struct CircuitToIsa {
    program_builder: ProgramBuilder,
}

impl CircuitToIsa {
    pub fn new() -> Self;

    /// Translate compiled circuit to ISA
    pub fn translate(&mut self, circuit: &CompiledCircuit) -> Result<Program>;

    /// Translate generator call to ISA instructions
    fn translate_generator(&mut self, gen: &GeneratorCall) -> Result<Vec<Instruction>>;
}
```

#### JSON → ISA

```rust
/// Translate JSON kernel to ISA program
pub struct JsonToIsa {
    program_builder: ProgramBuilder,
}

impl JsonToIsa {
    pub fn new() -> Self;

    /// Translate JSON kernel directly to ISA
    pub fn translate(&mut self, json: &serde_json::Value) -> Result<Program>;

    /// Used for simple operations (element-wise math)
    fn translate_simple_op(&mut self, op: &JsonOp) -> Result<Vec<Instruction>>;
}
```

### Pool Storage

```rust
/// O(1) space circular buffer for streaming computation
pub struct Pool {
    buffer: Vec<u8>,
    capacity: usize,
    head: usize,
    tail: usize,
}

impl Pool {
    /// Create pool with fixed capacity
    pub fn new(capacity: usize) -> Self;

    /// Write data (overwrites oldest if full)
    pub fn write(&mut self, data: &[u8]) -> Result<()>;

    /// Read data from offset
    pub fn read(&self, offset: usize, len: usize) -> Result<&[u8]>;

    /// Get used space
    pub fn len(&self) -> usize;

    /// Get capacity
    pub fn capacity(&self) -> usize;
}
```

**Use case:** Process unbounded data with constant memory

```rust
// Process 1GB dataset with 4KB pool
let pool = Pool::new(4096);
for chunk in dataset.chunks(1024) {
    pool.write(chunk)?;
    process(pool.read(0, 1024)?)?;
}
```

### Program Utilities

#### Program Builder

```rust
/// Build ISA programs programmatically
pub struct ProgramBuilder {
    program: Program,
    label_counter: usize,
}

impl ProgramBuilder {
    pub fn new() -> Self;

    /// Add instruction
    pub fn add(&mut self, instr: Instruction) -> &mut Self;

    /// Create label
    pub fn label(&mut self) -> Label;

    /// Build program
    pub fn build(self) -> Program;
}
```

#### Program Cache

```rust
/// Blake3-based program cache
pub struct ProgramCache {
    cache: LruCache<Blake3Hash, Program>,
}

impl ProgramCache {
    pub fn new(capacity: usize) -> Self;

    /// Get cached program
    pub fn get(&mut self, program: &Program) -> Option<&Program>;

    /// Insert program
    pub fn insert(&mut self, program: Program);

    /// Compute Blake3 hash
    fn hash(program: &Program) -> Blake3Hash;
}
```

## Internal Structure

**CRITICAL: All backends in `src/backends/` subdirectory**

```
crates/backends/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # Public API
│   ├── isa/                    # Atlas ISA
│   │   ├── mod.rs
│   │   ├── instruction.rs      # Instruction enum (< 1K lines)
│   │   ├── types.rs            # Type system (< 1K lines)
│   │   ├── address.rs          # Address modes (< 1K lines)
│   │   └── program.rs          # Program representation (< 1K lines)
│   ├── traits.rs               # Backend trait (< 1K lines)
│   ├── types.rs                # LaunchConfig, handles, etc. (< 1K lines)
│   ├── backends/               # ALL BACKENDS IN SUBDIRECTORY
│   │   ├── mod.rs
│   │   ├── cpu/
│   │   │   ├── mod.rs
│   │   │   ├── executor.rs     # Sequential execution (< 1K lines)
│   │   │   ├── simd.rs         # SIMD kernels (< 1K lines)
│   │   │   └── parallel.rs     # Rayon parallelism (< 1K lines)
│   │   ├── cuda/               # CUDA backend (optional)
│   │   │   ├── mod.rs
│   │   │   └── executor.rs     # (< 1K lines)
│   │   ├── metal/              # Metal backend (optional)
│   │   │   ├── mod.rs
│   │   │   └── executor.rs     # (< 1K lines)
│   │   ├── wasm/               # WASM backend
│   │   │   ├── mod.rs
│   │   │   └── executor.rs     # (< 1K lines)
│   │   └── webgpu/             # WebGPU backend (optional)
│   │       ├── mod.rs
│   │       └── executor.rs     # (< 1K lines)
│   ├── translators/            # High-level → ISA
│   │   ├── mod.rs
│   │   ├── circuit_to_isa.rs   # Circuit → ISA (< 1K lines)
│   │   └── json_to_isa.rs      # JSON → ISA (< 1K lines)
│   ├── pool.rs                 # Pool storage (< 1K lines)
│   ├── program_builder.rs      # ISA program builder (< 1K lines)
│   ├── program_cache.rs        # Blake3 caching (< 1K lines)
│   └── error.rs                # Error types
└── tests/
    ├── isa_tests.rs            # ISA instruction tests
    ├── backend_tests.rs        # Backend trait tests
    ├── cpu_tests.rs            # CPU backend tests
    ├── translator_tests.rs     # Translator tests
    └── pool_tests.rs           # Pool storage tests
```

## Dependencies

### External Dependencies

```toml
[dependencies]
# Error handling
thiserror = "1.0"

# Concurrent data structures (CPU backend)
dashmap = "5.5"
parking_lot = "0.12"

# Parallel execution (CPU backend)
rayon = "1.8"

# Hashing (program cache)
blake3 = "1.5"

# LRU cache
lru = "0.12"

# Type casting
bytemuck = "1.14"

# Optional features
cudarc = { version = "0.9", optional = true }
metal = { version = "0.27", optional = true }
wgpu = { version = "27.0", optional = true }

[features]
default = []
cuda = ["cudarc"]
metal = ["metal"]
webgpu = ["wgpu"]
```

### Internal Dependencies

- **hologram-compiler**: For CompiledCircuit type

## Testing Requirements

### Unit Tests

All ISA instructions must have unit tests:
- Correctness of each instruction
- Type conversions
- Address modes
- Edge cases

### Backend Tests

Each backend must have tests:
- Buffer allocation/deallocation
- Data transfer (host ↔ device)
- Program execution
- Pool storage operations

### Integration Tests

```rust
#[test]
fn test_cpu_backend_vector_add() -> Result<()> {
    let mut backend = CpuBackend::new();

    // Allocate buffers
    let a = backend.allocate_buffer(1024)?;
    let b = backend.allocate_buffer(1024)?;
    let c = backend.allocate_buffer(1024)?;

    // Fill data
    let data_a: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..256).map(|i| (i * 2) as f32).collect();
    backend.copy_to_buffer(a, bytemuck::cast_slice(&data_a))?;
    backend.copy_to_buffer(b, bytemuck::cast_slice(&data_b))?;

    // Build program
    let mut builder = ProgramBuilder::new();
    builder.add(Instruction::ParallelMapBinary {
        dst_array: Address::BufferOffset { handle: c, offset: 0 },
        src_a: Address::BufferOffset { handle: a, offset: 0 },
        src_b: Address::BufferOffset { handle: b, offset: 0 },
        count: 256,
        operation: BinaryMapOperation::Add,
    });
    let program = builder.build();

    // Execute
    backend.execute_program(&program, &LaunchConfig::linear(256))?;

    // Verify
    let mut result = vec![0f32; 256];
    backend.copy_from_buffer(c, bytemuck::cast_slice_mut(&mut result))?;
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 3.0);

    Ok(())
}
```

## Performance Requirements

### ISA Execution

| Backend | Instruction Throughput | Notes |
|---------|----------------------|-------|
| CPU (SIMD) | ~42ns/op | f32, ≤262K elements |
| CPU (interpreted) | ~1000ns/op | Fallback |
| CUDA | ~10ns/op | GPU kernel launch overhead |
| Metal | ~15ns/op | Apple GPU |
| WebGPU | ~50ns/op | Browser GPU |

### Memory Bandwidth

| Backend | Target Bandwidth | Notes |
|---------|-----------------|-------|
| CPU | > 80% peak | DDR4/DDR5 |
| CUDA | > 90% peak | HBM2/HBM3 |
| Metal | > 85% peak | Unified memory |

## Examples

### Creating a Backend

```rust
use hologram_backends::{CpuBackend, Backend};

// Create CPU backend
let mut backend = CpuBackend::new();

// Allocate buffer
let handle = backend.allocate_buffer(4096)?;

// Copy data
let data = vec![1.0f32; 1024];
backend.copy_to_buffer(handle, bytemuck::cast_slice(&data))?;
```

### Building ISA Program

```rust
use hologram_backends::{ProgramBuilder, Instruction, Register, BinaryMapOperation};

let mut builder = ProgramBuilder::new();

// Add instructions
builder
    .add(Instruction::ParallelMapBinary {
        dst_array: dst_addr,
        src_a: src_a_addr,
        src_b: src_b_addr,
        count: 1024,
        operation: BinaryMapOperation::Add,
    })
    .add(Instruction::ParallelMap {
        dst_array: dst_addr,
        src_array: dst_addr,
        count: 1024,
        operation: MapOperation::Relu,
    });

let program = builder.build();
```

### Pool Storage

```rust
use hologram_backends::{Backend, CpuBackend};

let mut backend = CpuBackend::new();

// Allocate pool (4KB)
let pool = backend.allocate_pool(4096)?;

// Stream process large dataset
for chunk in large_dataset.chunks(1024) {
    backend.copy_to_pool(pool, 0, chunk)?;
    // Process...
}
```

## Migration from Current Codebase

### Port Mapping

| Current Location | New Location |
|------------------|--------------|
| `hologram-backends/src/isa/*` | `isa/*` |
| `hologram-backends/src/backend/*` | `traits.rs` + `types.rs` |
| `hologram-backends/src/cpu.rs` | `backends/cpu/` |
| `hologram-backends/src/cuda.rs` | `backends/cuda/` |
| `hologram-backends/src/metal.rs` | `backends/metal/` |
| `hologram-backends/src/wasm.rs` | `backends/wasm/` |
| `hologram-backends/src/webgpu.rs` | `backends/webgpu/` |

### Critical Change

**🚨 ALL BACKENDS MUST BE IN `src/backends/` SUBDIRECTORY**

## Error Handling

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("Buffer allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Invalid buffer handle: {0:?}")]
    InvalidBuffer(BufferHandle),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Backend not available: {0}")]
    BackendUnavailable(String),

    #[error("ISA error: {0}")]
    IsaError(String),
}

pub type Result<T> = std::result::Result<T, BackendError>;
```

## ISA Extensions

### Simple Extensions

Basic instruction extensions for common operations:

```rust
/// Simple ISA extensions
#[derive(Debug, Clone, PartialEq)]
pub enum SimpleExtension {
    /// Fused multiply-subtract: dst = (a * b) - c
    FMS { dst: Register, a: Register, b: Register, c: Register },

    /// Reciprocal: dst = 1 / src
    RCP { dst: Register, src: Register },

    /// Reciprocal square root: dst = 1 / sqrt(src)
    RSQRT { dst: Register, src: Register },

    /// Clamp value to range: dst = clamp(src, min, max)
    CLAMP { dst: Register, src: Register, min: Register, max: Register },

    /// Linear interpolation: dst = a + (b - a) * t
    LERP { dst: Register, a: Register, b: Register, t: Register },

    /// Sign extraction: dst = sign(src) ∈ {-1, 0, 1}
    SIGN { dst: Register, src: Register },

    /// Floor operation
    FLOOR { dst: Register, src: Register },

    /// Ceiling operation
    CEIL { dst: Register, src: Register },

    /// Round to nearest
    ROUND { dst: Register, src: Register },

    /// Truncate (round toward zero)
    TRUNC { dst: Register, src: Register },
}
```

### Complex Extensions

Composite operations built from multiple instructions:

```rust
/// Complex ISA extensions
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexExtension {
    /// Matrix multiply: C = A × B
    GEMM {
        c: Address,
        a: Address,
        b: Address,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    },

    /// Batched matrix multiply
    BatchedGEMM {
        c: Address,
        a: Address,
        b: Address,
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
    },

    /// Convolution 2D
    Conv2D {
        output: Address,
        input: Address,
        kernel: Address,
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    },

    /// Layer normalization
    LayerNorm {
        output: Address,
        input: Address,
        gamma: Address,
        beta: Address,
        n: usize,
        eps: f32,
    },

    /// Batch normalization
    BatchNorm {
        output: Address,
        input: Address,
        gamma: Address,
        beta: Address,
        mean: Address,
        variance: Address,
        batch: usize,
        channels: usize,
        spatial: usize,
        eps: f32,
    },

    /// Softmax: output[i] = exp(input[i]) / sum(exp(input))
    Softmax {
        output: Address,
        input: Address,
        n: usize,
    },

    /// Attention mechanism: output = softmax(Q·K^T/√d)·V
    Attention {
        output: Address,
        query: Address,
        key: Address,
        value: Address,
        batch: usize,
        heads: usize,
        seq_len: usize,
        head_dim: usize,
    },

    /// FFT (Fast Fourier Transform)
    FFT {
        output: Address,
        input: Address,
        n: usize,
        inverse: bool,
    },
}
```

### Advanced Extensions

Future-facing capabilities for emerging workloads:

```rust
/// Advanced ISA extensions
#[derive(Debug, Clone, PartialEq)]
pub enum AdvancedExtension {
    /// Sparse matrix multiply (CSR format)
    SparseGEMM {
        c: Address,
        a_values: Address,
        a_indices: Address,
        a_ptr: Address,
        b: Address,
        m: usize,
        n: usize,
        k: usize,
    },

    /// Tensor contraction: C[i,j,k] = sum_l A[i,j,l] * B[l,k]
    TensorContract {
        output: Address,
        a: Address,
        b: Address,
        shape_a: Vec<usize>,
        shape_b: Vec<usize>,
        contract_dims: Vec<(usize, usize)>,
    },

    /// Quantized matrix multiply (INT8)
    QuantizedGEMM {
        c: Address,
        a: Address,
        b: Address,
        a_scale: f32,
        b_scale: f32,
        c_scale: f32,
        m: usize,
        n: usize,
        k: usize,
    },

    /// Mixed precision GEMM (FP16 inputs, FP32 accumulation)
    MixedPrecisionGEMM {
        c: Address,
        a: Address,
        b: Address,
        m: usize,
        n: usize,
        k: usize,
    },

    /// Flash Attention (memory-efficient attention)
    FlashAttention {
        output: Address,
        query: Address,
        key: Address,
        value: Address,
        batch: usize,
        heads: usize,
        seq_len: usize,
        head_dim: usize,
        block_size: usize,
    },

    /// Group normalization
    GroupNorm {
        output: Address,
        input: Address,
        gamma: Address,
        beta: Address,
        batch: usize,
        channels: usize,
        groups: usize,
        spatial: usize,
        eps: f32,
    },

    /// GELU activation (Gaussian Error Linear Unit)
    GELU { dst: Register, src: Register },

    /// Swish activation: x * sigmoid(x)
    SWISH { dst: Register, src: Register },

    /// LayerScale: element-wise scale with learnable parameter
    LayerScale {
        output: Address,
        input: Address,
        scale: Address,
        n: usize,
    },

    /// RoPE (Rotary Position Embedding)
    RoPE {
        output: Address,
        input: Address,
        freqs: Address,
        batch: usize,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
    },

    /// KV-Cache update for autoregressive generation
    KVCacheUpdate {
        cache: Address,
        new_kv: Address,
        position: usize,
        batch: usize,
        heads: usize,
        head_dim: usize,
    },

    /// Multi-GPU all-reduce
    AllReduce {
        buffer: Address,
        n: usize,
        operation: ReduceOperation,
        devices: Vec<u32>,
    },

    /// Multi-GPU all-gather
    AllGather {
        output: Address,
        input: Address,
        n: usize,
        devices: Vec<u32>,
    },
}
```

## Generic Backend Patterns

### Type-Generic Backend Trait

```rust
/// Generic numeric type support for backends
pub trait NumericType: bytemuck::Pod + Copy + Default + Send + Sync + 'static {
    /// Zero value
    fn zero() -> Self;

    /// One value
    fn one() -> Self;

    /// Convert from f64
    fn from_f64(value: f64) -> Self;

    /// Convert to f64
    fn to_f64(self) -> f64;

    /// Type identifier
    fn type_id() -> Type;
}

// Implementations for all numeric types
impl NumericType for f32 {
    #[inline]
    fn zero() -> Self { 0.0 }

    #[inline]
    fn one() -> Self { 1.0 }

    #[inline]
    fn from_f64(value: f64) -> Self { value as f32 }

    #[inline]
    fn to_f64(self) -> f64 { self as f64 }

    #[inline]
    fn type_id() -> Type { Type::F32 }
}

impl NumericType for f64 {
    #[inline]
    fn zero() -> Self { 0.0 }

    #[inline]
    fn one() -> Self { 1.0 }

    #[inline]
    fn from_f64(value: f64) -> Self { value }

    #[inline]
    fn to_f64(self) -> f64 { self }

    #[inline]
    fn type_id() -> Type { Type::F64 }
}

impl NumericType for i32 {
    #[inline]
    fn zero() -> Self { 0 }

    #[inline]
    fn one() -> Self { 1 }

    #[inline]
    fn from_f64(value: f64) -> Self { value as i32 }

    #[inline]
    fn to_f64(self) -> f64 { self as f64 }

    #[inline]
    fn type_id() -> Type { Type::I32 }
}

// Similar implementations for f16, i8, i16, i64, u8, u16, u32, u64
```

### Generic Buffer Operations

```rust
/// Type-safe generic buffer operations
pub trait TypedBuffer<T: NumericType> {
    /// Allocate typed buffer
    fn allocate_typed(backend: &mut dyn Backend, count: usize) -> Result<TypedBufferHandle<T>>;

    /// Copy typed data to buffer
    fn copy_typed_to(&mut self, backend: &mut dyn Backend, data: &[T]) -> Result<()>;

    /// Copy typed data from buffer
    fn copy_typed_from(&self, backend: &mut dyn Backend, data: &mut [T]) -> Result<()>;

    /// Get element count
    fn count(&self) -> usize;
}

/// Typed buffer handle with generic type parameter
pub struct TypedBufferHandle<T: NumericType> {
    handle: BufferHandle,
    count: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: NumericType> TypedBufferHandle<T> {
    /// Create from raw handle
    pub fn new(handle: BufferHandle, count: usize) -> Self {
        Self {
            handle,
            count,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get raw handle
    pub fn raw(&self) -> BufferHandle {
        self.handle
    }

    /// Get type
    pub fn ty() -> Type {
        T::type_id()
    }
}
```

### Generic Operation Execution

```rust
/// Generic operation executor
pub trait GenericExecutor {
    /// Execute generic unary operation
    fn execute_unary<T: NumericType>(
        &mut self,
        op: MapOperation,
        input: &TypedBufferHandle<T>,
        output: &mut TypedBufferHandle<T>,
    ) -> Result<()>;

    /// Execute generic binary operation
    fn execute_binary<T: NumericType>(
        &mut self,
        op: BinaryMapOperation,
        lhs: &TypedBufferHandle<T>,
        rhs: &TypedBufferHandle<T>,
        output: &mut TypedBufferHandle<T>,
    ) -> Result<()>;

    /// Execute generic reduction
    fn execute_reduce<T: NumericType>(
        &mut self,
        op: ReduceOperation,
        input: &TypedBufferHandle<T>,
        output: &mut TypedBufferHandle<T>,
    ) -> Result<()>;
}
```

## Macro-Based Extensibility

### ISA Instruction Definition Macro

```rust
/// Define ISA instruction with automatic builder methods
macro_rules! define_instruction {
    (
        $(#[$meta:meta])*
        $name:ident {
            $(
                $(#[$field_meta:meta])*
                $field:ident: $ty:ty
            ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name {
            $(
                $(#[$field_meta])*
                pub $field: $ty,
            )*
        }

        impl $name {
            /// Create instruction
            pub fn new($($field: $ty),*) -> Self {
                Self {
                    $($field),*
                }
            }

            /// Convert to generic instruction
            pub fn into_instruction(self) -> Instruction {
                Instruction::$name(self)
            }
        }

        /// Builder for instruction
        paste::paste! {
            impl ProgramBuilder {
                /// Add instruction to program
                pub fn [<add_ $name:snake>](&mut self, $($field: $ty),*) -> &mut Self {
                    self.add(Instruction::$name($name::new($($field),*)))
                }
            }
        }
    };
}

// Usage example
define_instruction! {
    /// Fused multiply-add instruction
    FMA {
        /// Destination register
        dst: Register,
        /// First operand
        a: Register,
        /// Second operand
        b: Register,
        /// Third operand
        c: Register,
    }
}

// Generates:
// - FMA struct
// - FMA::new() constructor
// - FMA::into_instruction() converter
// - ProgramBuilder::add_fma() builder method
```

### Backend Implementation Macro

```rust
/// Implement common backend operations
macro_rules! impl_backend_buffers {
    ($backend:ty) => {
        impl $backend {
            /// Allocate typed buffer
            pub fn allocate_typed<T: NumericType>(&mut self, count: usize) -> Result<TypedBufferHandle<T>> {
                let size = count * std::mem::size_of::<T>();
                let handle = self.allocate_buffer(size)?;
                Ok(TypedBufferHandle::new(handle, count))
            }

            /// Copy typed data to buffer
            pub fn copy_typed_to<T: NumericType>(
                &mut self,
                buffer: &TypedBufferHandle<T>,
                data: &[T],
            ) -> Result<()> {
                assert_eq!(data.len(), buffer.count);
                self.copy_to_buffer(buffer.raw(), bytemuck::cast_slice(data))
            }

            /// Copy typed data from buffer
            pub fn copy_typed_from<T: NumericType>(
                &mut self,
                buffer: &TypedBufferHandle<T>,
                data: &mut [T],
            ) -> Result<()> {
                assert_eq!(data.len(), buffer.count);
                self.copy_from_buffer(buffer.raw(), bytemuck::cast_slice_mut(data))
            }
        }
    };
}

// Usage: apply to all backends
impl_backend_buffers!(CpuBackend);
impl_backend_buffers!(CudaBackend);
impl_backend_buffers!(MetalBackend);
impl_backend_buffers!(WasmBackend);
impl_backend_buffers!(WebGpuBackend);
```

### Human-Readable Test Macros

```rust
/// Assert buffer contents match expected values
macro_rules! assert_buffer_eq {
    ($backend:expr, $buffer:expr, $expected:expr) => {{
        let mut actual = vec![0.0f32; $expected.len()];
        $backend.copy_from_buffer($buffer, bytemuck::cast_slice_mut(&mut actual))
            .expect("Failed to copy from buffer");

        for (i, (a, e)) in actual.iter().zip($expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "Buffer mismatch at index {}: expected {}, got {}",
                i, e, a
            );
        }
    }};
}

/// Assert typed buffer contents
macro_rules! assert_typed_buffer_eq {
    ($backend:expr, $buffer:expr, $expected:expr) => {{
        let mut actual = vec![<_ as NumericType>::zero(); $expected.len()];
        $backend.copy_typed_from(&$buffer, &mut actual)
            .expect("Failed to copy from typed buffer");

        for (i, (a, e)) in actual.iter().zip($expected.iter()).enumerate() {
            let a_f64 = a.to_f64();
            let e_f64 = e.to_f64();
            assert!(
                (a_f64 - e_f64).abs() < 1e-5,
                "Typed buffer mismatch at index {}: expected {}, got {}",
                i, e_f64, a_f64
            );
        }
    }};
}

/// Create and execute test program
macro_rules! test_program {
    ($backend:expr, $($instruction:expr),* $(,)?) => {{
        let mut builder = ProgramBuilder::new();
        $(
            builder.add($instruction);
        )*
        let program = builder.build();
        $backend.execute_program(&program, &LaunchConfig::linear(256))
            .expect("Program execution failed");
    }};
}

/// Test backend operation with setup and verification
macro_rules! backend_test {
    (
        name: $name:ident,
        backend: $backend_type:ty,
        setup: $setup:block,
        execute: $execute:block,
        verify: $verify:block $(,)?
    ) => {
        #[test]
        fn $name() -> Result<()> {
            let mut backend = <$backend_type>::new();

            // Setup phase
            $setup

            // Execute phase
            $execute

            // Verify phase
            $verify

            Ok(())
        }
    };
}

// Usage example
backend_test! {
    name: test_vector_add_f32,
    backend: CpuBackend,
    setup: {
        let a_handle = backend.allocate_typed::<f32>(256)?;
        let b_handle = backend.allocate_typed::<f32>(256)?;
        let c_handle = backend.allocate_typed::<f32>(256)?;

        let a_data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..256).map(|i| (i * 2) as f32).collect();

        backend.copy_typed_to(&a_handle, &a_data)?;
        backend.copy_typed_to(&b_handle, &b_data)?;
    },
    execute: {
        test_program!(
            backend,
            Instruction::ParallelMapBinary {
                dst_array: Address::BufferOffset { handle: c_handle.raw(), offset: 0 },
                src_a: Address::BufferOffset { handle: a_handle.raw(), offset: 0 },
                src_b: Address::BufferOffset { handle: b_handle.raw(), offset: 0 },
                count: 256,
                operation: BinaryMapOperation::Add,
            }
        );
    },
    verify: {
        let expected: Vec<f32> = (0..256).map(|i| (i + i * 2) as f32).collect();
        assert_typed_buffer_eq!(backend, c_handle, expected);
    },
}
```

## Additional Backend Capabilities

### TPU Backend

```rust
#[cfg(feature = "tpu")]
pub struct TpuBackend {
    device: TpuDevice,
    compiler: TpuCompiler,
    programs: HashMap<ProgramHash, TpuProgram>,
}

#[cfg(feature = "tpu")]
impl TpuBackend {
    /// Create TPU backend
    pub fn new(device_id: u32) -> Result<Self>;

    /// Compile to TPU XLA IR
    fn compile_to_xla(&self, program: &Program) -> Result<String>;
}
```

### Vulkan Backend

```rust
#[cfg(feature = "vulkan")]
pub struct VulkanBackend {
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pipelines: HashMap<ProgramHash, vk::Pipeline>,
}

#[cfg(feature = "vulkan")]
impl VulkanBackend {
    /// Create Vulkan backend
    pub fn new(device_id: u32) -> Result<Self>;

    /// Compile to SPIR-V
    fn compile_to_spirv(&self, program: &Program) -> Result<Vec<u32>>;
}
```

### OpenCL Backend

```rust
#[cfg(feature = "opencl")]
pub struct OpenClBackend {
    platform: opencl3::platform::Platform,
    device: opencl3::device::Device,
    context: opencl3::context::Context,
    kernels: HashMap<ProgramHash, opencl3::kernel::Kernel>,
}

#[cfg(feature = "opencl")]
impl OpenClBackend {
    /// Create OpenCL backend
    pub fn new(platform_id: usize, device_id: usize) -> Result<Self>;

    /// Compile to OpenCL C
    fn compile_to_opencl_c(&self, program: &Program) -> Result<String>;
}
```

### Dynamic Program Optimization

```rust
/// Runtime program optimizer
pub struct ProgramOptimizer {
    cache: HashMap<Program, Program>,
}

impl ProgramOptimizer {
    pub fn new() -> Self;

    /// Optimize program at runtime
    pub fn optimize(&mut self, program: &Program) -> Program;

    /// Apply peephole optimizations
    fn peephole_optimize(&self, program: &Program) -> Program;

    /// Dead code elimination
    fn eliminate_dead_code(&self, program: &Program) -> Program;

    /// Instruction fusion
    fn fuse_instructions(&self, program: &Program) -> Program;

    /// Constant propagation
    fn propagate_constants(&self, program: &Program) -> Program;
}
```

### Multi-GPU Execution

```rust
/// Multi-GPU execution coordinator
pub struct MultiGpuExecutor {
    backends: Vec<Box<dyn Backend>>,
    scheduler: WorkScheduler,
}

impl MultiGpuExecutor {
    /// Create multi-GPU executor
    pub fn new(device_ids: &[u32]) -> Result<Self>;

    /// Execute program across multiple GPUs
    pub fn execute_distributed(
        &mut self,
        program: &Program,
        strategy: DistributionStrategy,
    ) -> Result<()>;

    /// Data parallel execution
    fn execute_data_parallel(&mut self, program: &Program, batch_size: usize) -> Result<()>;

    /// Model parallel execution
    fn execute_model_parallel(&mut self, program: &Program, split_points: &[usize]) -> Result<()>;

    /// Pipeline parallel execution
    fn execute_pipeline_parallel(&mut self, program: &Program, stages: usize) -> Result<()>;
}

/// Distribution strategy for multi-GPU
#[derive(Debug, Clone, Copy)]
pub enum DistributionStrategy {
    DataParallel,
    ModelParallel,
    PipelineParallel,
    Hybrid,
}
```

### Distributed Execution

```rust
/// Distributed execution across multiple nodes
pub struct DistributedExecutor {
    nodes: Vec<NodeConnection>,
    coordinator: Coordinator,
}

impl DistributedExecutor {
    /// Create distributed executor
    pub fn new(node_addresses: &[String]) -> Result<Self>;

    /// Execute program across cluster
    pub fn execute_cluster(
        &mut self,
        program: &Program,
        strategy: ClusterStrategy,
    ) -> Result<()>;

    /// All-reduce operation
    pub fn all_reduce(&mut self, buffer: BufferHandle, op: ReduceOperation) -> Result<()>;

    /// All-gather operation
    pub fn all_gather(&mut self, local: BufferHandle, global: BufferHandle) -> Result<()>;

    /// Broadcast operation
    pub fn broadcast(&mut self, buffer: BufferHandle, root: usize) -> Result<()>;
}

/// Cluster distribution strategy
#[derive(Debug, Clone, Copy)]
pub enum ClusterStrategy {
    Replicated,
    Sharded,
    Hybrid,
}
```

## Future Enhancements

- [ ] ROCm/HIP backend (AMD GPUs)
- [ ] oneAPI backend (Intel GPUs)
- [ ] SYCL backend
- [ ] Automatic kernel generation from operations
- [ ] Runtime kernel caching and specialization
- [ ] Multi-device load balancing
- [ ] Heterogeneous execution (CPU+GPU)

## References

- [Atlas Specification](../atlas.md) - Complete Atlas algebraic structures (MOONSHINE implementation)
- [Atlas ISA Specification](../../architecture/atlas-isa.md)
- [Backend Architecture](../../architecture/backends.md)
- [SIMD Optimization](../../architecture/simd-optimization.md)
