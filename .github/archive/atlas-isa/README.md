# Atlas ISA — Instruction Set Architecture Specification

**A universal computational interface for analog, quantum, and classical backends**

## Scope

This crate defines the **Atlas Instruction Set Architecture (ISA)** — a universal computational interface that abstracts computation across diverse physical substrates. It is a **pure specification crate** with no execution, compilation, or runtime components.

Atlas is analogous to quantum computing interfaces (like Qiskit) but **backend-agnostic**: the same Atlas operations can execute on analog hardware, quantum computers, classical CPUs/GPUs, or any computational paradigm that implements the Atlas primitives.

The Atlas ISA provides:
- **Instruction opcodes** and their abstract semantics
- **Type system** for data and memory
- **Kernel metadata** structures for invariant enforcement
- **Atlas-12288 primitives** (R96 classification, Φ boundary lens, resonance classes)
- **Constants** defining the computational model

**What this crate does NOT contain:**
- ❌ Instruction execution (interpreters, JIT, codegen)
- ❌ Runtime systems
- ❌ Memory management implementations
- ❌ Compilation pipelines
- ❌ Backend-specific implementations

Higher-level abstractions (compilers, runtimes, execution engines) depend on this crate, never the reverse.

## Design Principles

### 1. Universal Computational Interface

Atlas abstracts computation across **any physical substrate** that can implement the primitive operations:

**Classical Backends:**
- CPU (scalar, SIMD, multi-core)
- GPU (CUDA, Metal, Vulkan, WebGPU)
- TPU/PJRT (JAX, XLA)
- FPGA/ASIC (custom accelerators)

**Analog/Neuromorphic:**
- Analog compute memory
- Memristor arrays
- Neuromorphic processors
- In-memory computing

**Quantum:**
- Superconducting qubits
- Trapped ions
- Photonic quantum computers
- Any quantum backend implementing Atlas primitives

**The domain (quantum, ML, graphics) is in the algorithm, not the backend.** A quantum algorithm written using Atlas can execute on classical hardware for testing, then deploy to quantum hardware without code changes.

Backend-specific mappings are defined by runtime implementations, not by this specification.

### 2. Speed Through Recursive Abstraction

Performance derives from the **mathematical structure** of Atlas, not from specific hardware:

- **96 Resonance Classes**: Hierarchical decomposition of problems across partitions
- **Φ Boundary Lens**: Spatial locality guarantees (48×256 structure per class)
- **Phase Counter**: Temporal ordering enabling deterministic composition
- **Topological Constraints**: Mirrors, neighbors, unity guide traversal

Operations compose recursively through these structures. Like how FFT achieves O(n log n) through its divide-and-conquer structure—regardless of whether executed on CPU, GPU, or quantum hardware—Atlas operations gain efficiency through compositional properties.

### 3. Constraint-First

Atlas enforces mathematical invariants as first-class primitives:
- **C96 Class System**: 96 resonance classes organizing computation
- **Mirror Pairing**: Symmetric operations across class pairs
- **Unity Neutrality**: Zero net resonance change
- **Boundary Lens**: 48×256 2D addressing torus (Φ-encoded)
- **Phase Modulus**: 768-cycle temporal counter

These invariants enable the recursive abstraction and are enforced by all backends.

### 4. Minimal Dependencies

This crate depends only on:
- `atlas-core` — Core Atlas-12288 mathematics
- Standard library utilities (serde, thiserror)

## Core Concepts

### Instruction Set (§7 of spec)

Instructions are organized into categories:

**Data Movement**
- `LDG`, `STG` — Global memory load/store
- `LDS`, `STS` — Shared memory load/store
- `MOV`, `CVT` — Register move, type conversion

**Arithmetic**
- `ADD`, `SUB`, `MUL`, `DIV` — Basic arithmetic
- `FMA` — Fused multiply-add
- `MIN`, `MAX`, `ABS`, `NEG` — Element-wise operations
- `SQRT` — Square root

**Logic & Comparison**
- `AND`, `OR`, `XOR`, `NOT` — Bitwise logic
- `SHL`, `SHR` — Bit shifts
- `SETCC` — Comparisons yielding predicates

**Control Flow**
- `BRA` — Branch (conditional/unconditional)
- `CALL`, `RET` — Function calls
- `LOOP` — Counted loop constructs

**Synchronization**
- `BAR.SYNC` — Block barrier (all lanes wait)
- `MEM.FENCE` — Memory fence (block/device/system scope)

**Atlas-Specific**
- `CLS.GET` — Read class ID
- `MIRROR` — Map class to its mirror
- `UNITY.TEST` — Check unity set membership
- `NBR.COUNT`, `NBR.GET` — Neighbor enumeration (1-skeleton)
- `RES.ACCUM` — Accumulate to resonance vector R[96]
- `PHASE.GET`, `PHASE.ADV` — Phase counter operations
- `BOUND.MAP` — Boundary lens address mapping
- `CHECK.UNITY` — Unity neutrality verification

### Type System (§6 of spec)

**Scalar Types**
- Integers: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
- Floats: `f16`, `bf16`, `f32`, `f64`
- Predicates: `bool` (1-bit)

**Atlas Types**
- `ResonanceClass` — C96 class identifier [0, 96)
- `PhiCoordinate` — Boundary lens coordinate (page, byte)
- `AtlasRatio` — Exact rational for resonance deltas

**Memory Spaces**
- Global — Flat 64-bit addressing
- Shared — Per-block scratch memory
- Const — Read-only constants
- Local — Per-lane private memory

### Computational Model (§2 of spec)

**Execution Hierarchy**
- **Lane** — Minimal execution unit (like a thread)
- **Block** — Group of lanes that can synchronize
- **Grid** — 3D array of blocks

**Launch Configuration**
```rust
use atlas_isa::{GridDim, BlockDim, LaunchConfig};

let config = LaunchConfig::new(
    GridDim::new(96, 1, 1),   // 96 blocks (one per class)
    BlockDim::new(256, 1, 1), // 256 lanes per block
    0                          // Shared memory size
);
```

**Mapping to Backends**

The execution model maps differently depending on the physical substrate:

- **CPU**: Grid→thread pool, Block→work chunk, Lane→scalar/SIMD lane
- **GPU**: Direct mapping to CUDA/Metal blocks and threads
- **TPU**: Grid→replicas, Block→tile program, Lane→vector element
- **Quantum**: Grid→circuit depth, Block→qubit groups, Lane→measurement basis
- **Analog**: Grid→compute units, Block→memory regions, Lane→analog channels

### Atlas Invariants (§8 of spec)

**Kernel Metadata** declares invariants:

```rust
use atlas_isa::{KernelMetadata, ClassMask, PhaseWindow, BoundaryFootprint};

let metadata = KernelMetadata {
    name: "vector_add".into(),
    classes_mask: ClassMask::all(),        // Touches all classes
    mirror_safe: true,                      // Invariant under mirroring
    unity_neutral: true,                    // Zero net resonance
    uses_boundary: false,                   // No boundary addressing
    boundary: BoundaryFootprint::full(),
    phase: PhaseWindow::full(),
};
```

**Invariant Enforcement** is implementation-defined:
- Compilers may verify statically
- Runtimes may check dynamically
- Hardware may enforce via traps

## Atlas-12288 Foundation

### R96 Classification

Every byte maps to one of 96 resonance classes:

```rust
use atlas_isa::r96_classify;

let byte_value = 42u8;
let class = r96_classify(byte_value);
assert!(class.as_u8() < 96);
```

Properties:
- **Surjective**: All 96 classes are represented
- **Deterministic**: Same byte always yields same class
- **Resonance-aware**: Classes group bytes by mathematical properties

### Φ Boundary Lens

48×256 torus for 2D structured addressing:

```rust
use atlas_isa::{PhiCoordinate, phi_encode, phi_decode};

// Encode coordinate to linear address
let coord = PhiCoordinate::new(10, 128).unwrap();
let linear = coord.encode();

// Decode back
let decoded = PhiCoordinate::decode(linear);
assert_eq!(decoded, coord);

// Or use functions directly
let encoded = phi_encode(10, 128);
let (page, byte) = phi_decode(encoded);
assert_eq!((page, byte), (10, 128));
```

The Φ lens enables:
- Efficient 2D tiling patterns
- Boundary-respecting memory layouts
- Natural mapping to matrix operations

### Mirror Pairing

Classes pair symmetrically:

```rust
use atlas_isa::{ResonanceClass, mirror_class};

let class = ResonanceClass::new(42).unwrap();
let mirrored = mirror_class(class);

// Mirror is involutive
assert_eq!(mirror_class(mirrored), class);
```

Mirror-safe operations enable:
- Symmetric computation patterns
- Conjugate operation fusion
- Scheduling optimizations

### Phase Counter

768-cycle modular counter for temporal scheduling:

```rust
use atlas_isa::{PhaseWindow, PHASE_MODULUS};

let window = PhaseWindow {
    begin: 0,
    span: 384,  // First half of cycle
};

assert!(window.contains(100));
assert!(!window.contains(500));
assert_eq!(PHASE_MODULUS, 768);
```

## Constants

```rust
use atlas_isa::{PAGES, BYTES_PER_PAGE, PHASE_MODULUS, TOTAL_CLASSES};

assert_eq!(PAGES, 48);
assert_eq!(BYTES_PER_PAGE, 256);
assert_eq!(PHASE_MODULUS, 768);
assert_eq!(TOTAL_CLASSES, 96);
```

## Conformance Profiles (§11 of spec)

Implementations declare capability profiles:

**Profile S (Scalar Core)** — Required baseline
- Data movement, arithmetic, control flow
- Barriers and fences
- Atlas-specific operations (CLS, MIRROR, etc.)

**Profile V (Vector/Matrix)** — Optional
- Vector packs (2/4-wide)
- Matrix multiply accelerators

**Profile T (Transcendentals)** — Optional
- `EXP`, `LOG`, `SIN`, `COS`, `TANH`
- Accuracy levels: approx, fast, precise

## Usage

```rust
use atlas_isa::{
    Opcode, InstructionCategory,
    ResonanceClass, PhiCoordinate,
    KernelMetadata, ClassMask,
    r96_classify, phi_encode,
    PAGES, BYTES_PER_PAGE,
};

// Classify bytes
let class = r96_classify(42);

// Work with boundary coordinates
let coord = PhiCoordinate::new(10, 128)?;
let linear = coord.encode();

// Define kernel metadata
let metadata = KernelMetadata::new("my_kernel");

// Reference opcodes
let opcode = Opcode::ADD;
assert_eq!(opcode.category(), InstructionCategory::Arithmetic);
```

## Specification

Full specification: [`SPEC.md`](SPEC.md)

The specification defines:
- Abstract semantics of all instructions
- Memory model and visibility rules
- Invariant enforcement requirements
- ABI and parameter passing
- Target mapping guidance
- Verification hooks

## Non-Goals

This crate does NOT provide:
- Execution engines (see runtime implementations)
- Code generation (see compiler implementations)
- Memory allocators (see runtime implementations)
- Language bindings (see language-specific crates)

## Dependencies

- `atlas-core` — Core Atlas-12288 mathematics
- `serde` — Serialization for metadata
- `thiserror` — Error types
- `bytemuck` — Zero-copy casting
- `num-traits`, `num-rational` — Numeric utilities

## See Also

- **Specification**: [`SPEC.md`](SPEC.md) — Complete ISA reference
- **Atlas-12288 Foundation**: `atlas-core` crate
- **Runtime Implementations**: `atlas-runtime` crate (or custom)
- **Compilers**: `hgir` and higher-level IRs

## License

See repository root for license information.
