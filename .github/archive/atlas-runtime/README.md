# Atlas Runtime — Memory and State Management

**Universal state management for Atlas computational backends**

## Scope

This crate provides **Atlas state management and memory abstractions** for implementing Atlas ISA backends on **any computational substrate** (classical, quantum, analog). It is **NOT an execution engine** — it provides the computational memory model that backends use to execute kernels.

The Atlas Runtime provides:
- **AtlasSpace**: 96-class resonance space with phase counter and accumulators
- **Memory management**: Linear and boundary-addressed buffer allocation
- **Φ boundary lens**: 48×256 2D addressing with coordinate mapping
- **Validation**: Invariant checking (unity neutrality, mirror safety, phase windows)
- **Host API**: Memory binding, Φ installation, neighbor tables, metadata validation

**What this crate does NOT contain:**
- ❌ Kernel execution engines (interpreters, JIT, codegen)
- ❌ Backend implementations (classical, quantum, analog)
- ❌ Compilation pipelines
- ❌ Domain-specific operations (ML, graphics, quantum algorithms)

Execution happens in **backend implementations** that use this runtime to manage Atlas state.

## Design Principles

### 1. Universal Backend Support

The runtime provides a **single state model** used by all backends:

**Classical Backends:**
- CPU: Direct memory access, rayon parallelism
- GPU: Device memory mapping, CUDA/Metal/Vulkan primitives
- FPGA/ASIC: Custom memory controllers

**Quantum Backends:**
- Qubit state tracking via resonance accumulators
- Circuit depth coordination via phase counter
- Measurement results stored in Atlas memory

**Analog/Neuromorphic:**
- Analog state mapped to Atlas classes
- Physical resonance tracked digitally
- Boundary lens maps to analog memory topology

### 2. State, Not Execution

The runtime manages **computational state** (resonance classes, phase, memory) but does not execute instructions. Backends implement execution strategies using this shared state model.

### 3. Backend Agnostic

A single runtime instance can coordinate across multiple backend types:
```
AtlasSpace
    ├─ Classical backend (CPU) for testing
    ├─ Quantum backend (IBM Q) for quantum ops
    └─ Analog backend (memristor array) for in-memory compute
```

Operations execute on the appropriate backend while sharing the same Atlas state.

### 4. Invariant Enforcement
The runtime validates Atlas invariants but leaves execution policy to backends:
- Unity neutrality (zero net resonance)
- Mirror safety (operation symmetry)
- Phase windows (temporal constraints)
- Boundary footprints (spatial constraints)
- Neighbor legality (1-skeleton respect)

## Core Components

### AtlasSpace

The computational memory model implementing the Atlas-12288 foundation:

```rust
use atlas_runtime::AtlasSpace;

// Create Atlas space with 96 resonance classes
let mut space = AtlasSpace::new();

// Query current state
let phase = space.phase();  // Returns current phase (mod 768)
let class = 42;
let resonance = space.resonance_at(class)?;  // Get R[96] at class

// Advance phase (for temporal scheduling)
space.advance_phase()?;
```

**AtlasSpace maintains:**
- **C96 Classes**: 96 resonance classes organizing computation
- **Phase Counter**: Modulo-768 temporal counter (§3.3 of spec)
- **Resonance Accumulator R[96]**: Exact rational deltas per class (§6.1)
- **Neighbor Table**: 1-skeleton adjacency (§3.4)
- **Mirror Mapping**: Involutive class pairing (§7)

### Memory Pools

Two memory allocation strategies per the dual architecture:

```rust
use atlas_runtime::{AtlasSpace, MemoryPool};

let mut space = AtlasSpace::new();

// Linear pool - unlimited, traditional flat addressing
let linear_handle = space.allocate_buffer(
    1024,             // size in bytes
    8,                // alignment
    MemoryPool::Linear
)?;

// Boundary pool - fixed 1.125 MB, Φ-addressed (48×256×96)
let boundary_handle = space.allocate_buffer(
    256 * 48,         // size in bytes (one class worth)
    1,                // alignment
    MemoryPool::Boundary
)?;
```

**Memory Pools:**
- **Linear**: Unlimited, flat 64-bit addressing, general-purpose
  - Pre-allocates capacity to avoid reallocation during execution
  - Dynamically grows as needed beyond initial capacity
- **Boundary**: Fixed 1,179,648 bytes (48×256×96), Φ-addressed, structured access
  - Uses huge page support (2MB/1GB pages via mmap) to reduce TLB misses
  - 64-byte aligned for cache-optimal access
  - Each class (12 KiB) fits entirely in L1 cache

### Buffer Abstraction

Typed memory views with Atlas addressing:

```rust
use atlas_runtime::Buffer;

// Linear buffer (flat addressing)
let linear_buf: Buffer<f32> = Buffer::new_linear(
    handle,
    offset,
    len,
    space_ref
);

// Boundary buffer (Φ-addressed)
let boundary_buf: Buffer<f32> = Buffer::new_boundary(
    handle,
    class_id,
    page_offset,
    byte_offset,
    width,
    height,
    phi_desc,
    space_ref
)?;

// Access data
let data: Vec<f32> = linear_buf.to_vec()?;
linear_buf.copy_from_slice(&new_data)?;
```

### Φ Boundary Lens

48×256 torus addressing for structured memory access:

```rust
use atlas_runtime::{PhiDesc, PhiCoordinate};

// Create Φ descriptor for 2D tiling
let phi = PhiDesc::identity(class_id);

// Map (page, byte) to linear address
let coord = PhiCoordinate::new(10, 128)?;
let linear_addr = coord.linear_index();

// Decode back
let recovered = PhiCoordinate::decode(linear_addr);
assert_eq!(coord, recovered);
```

**Φ addressing enables:**
- 2D matrix tiling patterns
- Boundary-respecting layouts
- Efficient transpose/reshape operations

### Validation

Check Atlas invariants before execution:

```rust
use atlas_runtime::validation::{validate_unity_neutral, validate_mirror_safe};
use atlas_isa::KernelMetadata;

let metadata = KernelMetadata {
    name: "my_kernel".into(),
    unity_neutral: true,
    mirror_safe: true,
    // ...
};

// Validate before execution
metadata.validate()?;

// Check unity neutrality with computed deltas
let deltas: [AtlasRatio; 96] = compute_deltas();
validate_unity_neutral(&deltas)?;

// Verify mirror safety property
validate_mirror_safe(&metadata, &operation_classes)?;
```

## Architecture

```
┌─────────────────────────────────────┐
│ Backend (atlas-backend-cpu)         │
│ - Native execution (rayon)          │
│ - Kernel compilation                │
│ - Performance optimization          │
└──────────────┬──────────────────────┘
               │ Uses
               ↓
┌─────────────────────────────────────┐
│ Atlas Runtime (this crate)          │
│ - AtlasSpace                        │
│ - Buffer management                 │
│ - Φ addressing                      │
│ - Validation                        │
└──────────────┬──────────────────────┘
               │ Implements
               ↓
┌─────────────────────────────────────┐
│ Atlas ISA (specification)           │
│ - Opcodes                           │
│ - Type system                       │
│ - Metadata                          │
│ - Constants                         │
└─────────────────────────────────────┘
```

## Usage Example

```rust
use atlas_runtime::{AtlasSpace, MemoryPool};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Atlas space
    let mut space = AtlasSpace::new();

    // Allocate buffers
    let size_bytes = 1024 * std::mem::size_of::<f32>();
    let input_handle = space.allocate_buffer(
        size_bytes,
        8,
        MemoryPool::Linear
    )?;

    let output_handle = space.allocate_buffer(
        size_bytes,
        8,
        MemoryPool::Linear
    )?;

    // Create typed buffer views
    let input: Buffer<f32> = Buffer::new_linear(
        input_handle,
        0,
        1024,
        Arc::new(RwLock::new(space))
    );

    // Backend would execute kernel here using these buffers
    // (Execution is NOT part of this crate)

    Ok(())
}
```

## Backend Implementation

To implement a backend using this runtime:

```rust
use atlas_runtime::{AtlasSpace, Buffer};
use atlas_isa::KernelMetadata;

pub struct MyBackend {
    space: Arc<RwLock<AtlasSpace>>,
}

impl MyBackend {
    pub fn execute_kernel(
        &self,
        metadata: &KernelMetadata,
        input: &Buffer<f32>,
        output: &mut Buffer<f32>,
    ) -> Result<()> {
        // 1. Validate metadata
        metadata.validate()?;

        // 2. Execute your backend-specific code
        // (native Rust, GPU launch, etc.)
        self.execute_native(input, output)?;

        // 3. Update Atlas state if needed
        let mut space = self.space.write();
        space.advance_phase()?;

        Ok(())
    }

    fn execute_native(&self, input: &Buffer<f32>, output: &mut Buffer<f32>) -> Result<()> {
        // Backend-specific execution
        // This crate doesn't implement this - you do!
        todo!("Implement your execution strategy")
    }
}
```

## Specification

Full specification: [`SPEC.md`](SPEC.md)

The specification defines:
- Atlas Space composition (§3-5)
- Memory model with dual pools (§4)
- Φ boundary lens addressing (§7)
- Resonance accumulator semantics (§6)
- Validation requirements (§8)
- Backend portability profiles (§9)

## Dependencies

- `atlas-core` — Core Atlas-12288 mathematics
- `atlas-isa` — ISA specification (types, constants, metadata)
- `hgir` — Kernel IR (for metadata only, not execution)
- `parking_lot` — Concurrent access to AtlasSpace
- `rayon` — Parallel iteration (for internal operations only)
- `thiserror` — Error types
- `serde`, `bincode` — Serialization

## Performance

**Speed comes from the recursive abstraction**, not from specific implementation details.

The current CPU-based implementation provides:
- **Cache-optimal layout**: Each 12 KiB class fits in L1 cache; 256-byte pages align with cache lines
- **Minimal overhead addressing**: `bound_map()` compiles to 2-3 instructions with no indirection
- **SIMD-friendly**: Contiguous byte dimension (stride-1) and 256-byte pages divisible by vector widths
- **Huge page support**: 2MB/1GB pages reduce TLB misses by ~4x on large operations
- **Pre-allocated capacity**: Linear pool avoids reallocation during execution
- **Lock-free boundary access**: Direct memory access without synchronization overhead

**Backend-specific optimizations:**
- Quantum backends track state differently (measurement statistics, expectation values)
- Analog backends map directly to physical memory topology
- GPU backends use device memory and texture addressing

**All backends must:**
- Maintain exact resonance tracking (canonical rationals per specification)
- Enforce Atlas invariants (unity neutrality, mirror safety, phase windows)
- Provide deterministic results (same inputs → same outputs)

## Testing

```bash
# Run all tests
cargo test -p atlas-runtime

# Run with invariant checking
RUST_LOG=atlas_runtime=debug cargo test

# Benchmarks (if any)
cargo bench -p atlas-runtime
```

## See Also

- **Specification**: [`SPEC.md`](SPEC.md) — Complete runtime reference
- **Atlas ISA**: `atlas-isa` crate — ISA specification
- **Atlas Core**: `atlas-core` crate — Foundation mathematics
- **Backend Example**: `hologram-stdlib` — Uses this runtime for CPU execution

## License

See repository root for license information.
