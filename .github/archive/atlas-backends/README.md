# Atlas Backends

**Execution layer implementing Atlas ISA on physical hardware substrates**

Atlas Backends provides the bridge between Atlas's mathematical abstraction and actual hardware execution (CPU, GPU, quantum, analog). Unlike traditional compute backends, Atlas backends are **topology-aware** — they leverage the mathematical structure of Atlas (96 resonance classes, Φ-addressing, phase coordination) to achieve performance.

## Key Concept: Data Shapes Computation

Traditional computing: Data is passive, computation is generic
```rust
// Same algorithm regardless of data
fn process(data: &[u8]) { /* generic loop */ }
```

Atlas computing: Data defines the computational topology
```rust
// Algorithm adapts to data's structure in Atlas space
fn process(data: &AtlasBuffer) {
    let topology = data.classify(); // Which classes, Φ-coordinates
    backend.execute_in_topology(operation, topology);
}
```

**Example:** Factorizing a 50-bit number creates different resonance class patterns than a 51-bit number. The backend allocates memory and structures execution based on this topology.

## Architecture

```
Hologram Core Operations
    ↓ calls
Atlas Backend Abstraction
    ↓ dispatches to
├─ CPU Backend (cache-resident topology)
├─ GPU Backend (thread blocks = classes)
├─ Quantum Backend (qubits = classes)
└─ Analog Backend (memristors = classes)
    ↓ operates on
Atlas Runtime (state management)
    ↓ implements
Atlas ISA (specification)
```

## CPU Backend: Cache-Resident Execution

The CPU backend leverages the cache hierarchy for Atlas's natural structure:

### Memory Layout
```
L1 Cache (32 KB per core)
  └─ Active class working set (2-3 classes being operated on)

L2 Cache (256 KB - 1 MB per core)
  └─ Boundary pool: 96 classes × 12 KB = 1.18 MB
     └─ Each class: 48 pages × 256 bytes

RAM
  └─ Linear pool (unlimited overflow storage)
```

### Why This Matters

The boundary pool (1.18 MB) is **designed to be cache-resident**:
- Fits comfortably in modern L2/L3 caches
- All 96 classes accessible at ~10-cycle latency
- Active classes loaded into L1 for ~1-cycle access

**Traditional approach:** Hope for cache hits (unpredictable)
**Atlas approach:** Guarantee cache residency (by design)

### Lookup-Resolve Pattern

Operations follow a **cache-optimized pattern**:

1. **Lookup** (L1): Read inputs from active classes in L1 cache
2. **Resolve** (L2): Use class topology (mirrors, neighbors) from L2
3. **Write-back** (L2): Store results to boundary pool in L2

This achieves near-L1 latency for the entire Atlas working set.

## Backend Abstraction

All backends implement a common interface:

```rust
pub trait AtlasBackend {
    /// Initialize cache-resident structures
    fn initialize(&mut self, space: &AtlasSpace) -> Result<()>;

    /// Allocate with topology awareness
    fn allocate(&mut self, topology: BufferTopology) -> Result<BackendHandle>;

    /// Execute operation within class topology
    fn execute(&mut self, op: Operation, context: &ExecutionContext) -> Result<()>;

    /// Synchronize state back to AtlasSpace
    fn synchronize(&mut self, space: &mut AtlasSpace) -> Result<()>;
}
```

### Key Difference from Traditional Backends

**Traditional GPU backend:**
```rust
backend.launch_kernel(grid, block, kernel_ptr, args);
```

**Atlas backend:**
```rust
backend.execute(
    operation,
    ExecutionContext {
        phase: 42,              // Temporal coordination
        active_classes: [0, 5], // Which classes are involved
        topology: &mirrors,     // Spatial relationships
        resonance: &state,      // Mathematical invariants
    }
);
```

The backend **uses the topology** to execute, not just memory addresses.

## Topology-Aware Allocation

Memory allocation is informed by the data's structure:

```rust
pub struct BufferTopology {
    /// Which resonance classes this data occupies
    pub active_classes: Vec<u8>,

    /// Φ-encoded (page, byte) coordinates
    pub phi_coordinates: Vec<(u8, u8)>,

    /// Preferred phase for temporal locality
    pub phase_affinity: Option<u16>,

    /// Memory pool (Boundary = cache-resident, Linear = RAM)
    pub pool: MemoryPool,
}
```

The backend allocates to optimize cache utilization based on which classes will be active.

## Example: Vector Addition

### Traditional Implementation
```rust
fn vector_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] + b[i];  // Hope data is in cache
    }
}
```

### Atlas Implementation
```rust
fn vector_add(backend: &mut impl AtlasBackend, a: &Buffer, b: &Buffer, c: &mut Buffer) {
    // 1. Determine which classes are active
    let topology = BufferTopology::from_buffers(&[a, b, c]);

    // 2. Activate classes into L1
    backend.activate_classes(&topology.active_classes)?;

    // 3. Execute with guaranteed cache residency
    backend.execute(
        Operation::VectorAdd { a, b, c },
        ExecutionContext {
            active_classes: topology.active_classes,
            phase: backend.current_phase(),
            topology: backend.topology_tables(),
        }
    )?;

    // Data was in L1 for entire operation - no cache misses
}
```

## Speed Through Recursive Abstraction

Traditional parallelism: Divide work across cores → communication overhead

Atlas parallelism: Each core operates on class topology subset → compose through mathematical relationships (mirrors, neighbors) that are cache-resident

**Speed doesn't come from brute-force parallelism** — it comes from the recursive structure being cache-resident and operations composing through topology rather than message passing.

## Backend Feature Matrix

| Backend | Status | Hardware | Use Case |
|---------|--------|----------|----------|
| **CPU** | ✅ Implemented | x86-64, ARM | Default, debugging, development |
| **GPU** | 🚧 Planned | CUDA, Metal, Vulkan | Large-scale numerical operations |
| **Quantum** | 🔬 Research | IBM Q, IonQ, Rigetti | Quantum algorithms on Atlas |
| **Analog** | 🔬 Research | Memristors, neuromorphic | In-memory computing |

## Universal Interface

Atlas backends are **substrate-agnostic**:

- Same operations run on CPU, GPU, quantum, or analog
- Backend selection is transparent to higher layers
- A quantum algorithm can develop on CPU, deploy to quantum hardware
- Domain (quantum, ML, graphics) is in the algorithm, not the backend

## Usage

```rust
use atlas_backends::{CPUBackend, AtlasBackend};
use atlas_runtime::AtlasSpace;

// Initialize backend
let space = AtlasSpace::new();
let mut backend = CPUBackend::new()?;
backend.initialize(&space)?;

// Allocate with topology awareness
let topology = BufferTopology {
    active_classes: vec![0, 1, 2],
    pool: MemoryPool::Boundary, // Cache-resident
    ..Default::default()
};
let buffer = backend.allocate(topology)?;

// Execute operations
let op = Operation::VectorAdd { /* ... */ };
let ctx = ExecutionContext { phase: 0, /* ... */ };
backend.execute(op, &ctx)?;

// Synchronize state
backend.synchronize(&mut space)?;
```

## Implementation Notes

### Cache Pinning (CPU Backend)

The CPU backend uses platform-specific mechanisms to pin the boundary pool in cache:

**Linux:**
```rust
mmap(size, MAP_LOCKED | MAP_HUGETLB | MAP_ANONYMOUS)
```

**macOS:**
```rust
mlock(addr, size) + madvise(MADV_WILLNEED)
```

**Windows:**
```rust
VirtualLock(addr, size) + large pages
```

### Class Activation

Before executing, the backend prefetches active classes into L1:

```rust
for &class in active_classes {
    let base = boundary_pool[class];
    for offset in (0..12_288).step_by(64) {  // 64-byte cache lines
        _mm_prefetch(base + offset, _MM_HINT_T0);
    }
}
```

This guarantees L1 residency during operation execution.

## See Also

- [SPEC.md](SPEC.md) - Full backend specification
- [Atlas ISA](../atlas-isa/) - Mathematical primitives backends implement
- [Atlas Runtime](../atlas-runtime/) - State management layer
- [Hologram Core](../hologram-core/) - High-level operations using backends

---

**Remember:** Atlas backends don't just execute on hardware — they **leverage topology for performance**. The mathematical structure is the accelerator.
