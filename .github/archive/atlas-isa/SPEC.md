# Atlas ISA (Universal) Specification — v1.1 Draft

**Status:** Draft v1.1 (universal mapping)\
**Audience:** Compiler/runtime authors, kernel developers, verification engineers\
**Scope:** Defines the abstract instruction set architecture (ISA) for **Atlas**, including computational model, state, memory, instructions, and invariants, **generalized for mapping to arbitrary target ISAs** (CPU, GPU, TPU/PJRT-class accelerators, DSP/SIMD, FPGA/ASIC). This ISA is the basis for Atlas vGPU (CUDA-Driver façade), Atlas vTPU (PJRT façade), and direct CPU/vector backends.

---

## 1. Design Intent

Atlas is a _constraint-first_, **target-agnostic** compute ISA. It exposes standard numerical operations while making Atlas invariants **first-class** and portable across diverse hardware:

- **C96 Class System:** 96 equivalence classes label operations/data domains.
- **Mirror Pairing (±):** A fixed involution pairs classes; operations MAY be mirror-safe for scheduling/transformations.
- **Unity Set:** Distinguished neutral classes with zero net resonance.
- **Boundary Lens:** A 2D torus address lens \((48,256)\) for structured kernels (optional on targets lacking 2D tilers).
- **Temporal Cycle:** A 768-phase modular counter for deterministic staging (virtualizable on targets without HW phase counters).
- **Atlas 1‑Skeleton:** A sparse, fixed neighbor relation among the 96 classes governing legal dataflow.

**Portability goals**

- Keep the **core feature set lean** and mapable: every instruction/invariant has a defined lowering to common targets.
- Use **profiles & capabilities** to negotiate optional features (vector packs, transcendentals, collectives).
- Prefer **legalization** (rewrite to core ops + fences) over target-specific intrinsics.

---

## 2. Computational Model

### 2.1 Entities

- **Program:** A collection of kernels plus metadata.
- **Kernel:** A parametric entry that runs over an _iteration space_ (grid × block) with access to memory and to Atlas control surfaces.
- **Lane:** The minimal execution agent (akin to a CUDA thread). Lanes are grouped into **Blocks**; blocks form a **Grid**.

### 2.2 Mapping (conventions)

Implementations may choose concrete mappings; the **reference mapping** is:

- `class_id = blockIdx.x mod 96` (C96 class per block)
- `boundary = ( threadIdx.x mod 48, threadIdx.y mod 256 )`
- `phase = (stream_phase + grid_linear_index) mod 768`

Other mappings are permitted if kernel metadata declares its expectations (§4).

### 2.3 Target parallelism abstraction

Atlas expresses parallelism as **grids/blocks/lanes**. Targets map these concepts to their native forms:

- **CPU:** grid→outer loops or thread pool; block→work chunk; lane→scalar or SIMD lane.
- **GPU:** direct mapping to blocks/threads.
- **TPU/PJRT:** grid→replicas/tiles; block→tile program; lane→vector element.
- **DSP/SIMD:** block→vector iteration; lane→SIMD lane.

### 2.4 Synchronization abstraction

- **BAR.SYNC** maps to: CPU `thread_barrier`/`std::barrier`, GPU `__syncthreads`, PJRT fence inside a tile.
- **MEM.FENCE {block, device, system}** maps to: CPU acquire/release fences at thread/agent/system scope; GPU memory scopes; PJRT device/host fences.

### 2.5 Execution model equivalence

A compliant backend MUST provide ordering equivalent to: within a block, program order + barrier semantics; across blocks, stream-ordered submission; across streams, ordering only via events/fences.

---

## 3. Machine State

### 3.1 Per‑Kernel (static)

- **Param block (PB):** Flat POD, ABI in §9.
- **Attributes:** flags declaring invariants (mirror_safe, unity_neutral), class mask, boundary footprint, phase window.

### 3.2 Per‑Block

- **Class ID:** `u7` in [0,95].
- **Shared Scratch:** optional; size requested at launch.

### 3.3 Per‑Lane

- **Registers:** typed virtual registers; compiler allocates.
- **Predicates:** boolean flags for control.
- **Boundary Coords:** `(bx: u6 in [0,47], by: u8 in [0,255])` (read-only if used).

### 3.4 Global Surfaces

- **Resonance Accumulator R[96]:** 96-dimensional rational vector (per stream/context) stored as `Ratio<i64>` and surfaced over the ABI as `AtlasRatio { numer, denom }` for neutrality checks. Denominators must be non-zero and deltas should remain in reduced form.
- **Phase Counter P:** `u10` modulo 768 per stream.
- **Neighbor Table N[96] -> List{class_id}:** immutable 1‑skeleton adjacency.

---

## 4. Kernel Metadata (Atlas Attributes)

Every kernel MAY declare the following attributes; runtimes MAY enforce:

- `classes_mask[96]` — bitset of classes the kernel may touch.
- `mirror_safe` — kernel is invariant under class-wise ± pairing.
- `unity_neutral` — net delta to R[96] is zero for the launch.
- `uses_boundary` — interprets device pointers in boundary lens ranges.
- `boundary_window` — `(x_min,x_max) × (y_min,y_max)` sub‑torus.
- `phase_begin, phase_span` — allowable phases in modulo‑768 schedule.

---

## 5. Memory Model

### 5.1 Address Spaces

- **Global:** flat 64‑bit, readable/writable by all lanes.
- **Shared:** per‑block scratch; lifetime = block.
- **Const:** read‑only, initialized at module load.
- **Local:** compiler-emulated per‑lane spill space.

### 5.2 Visibility & Ordering

- Operations in a block are ordered by program order unless noted.
- **Barriers** synchronize lanes in a block and commit shared/global writes with _release_ semantics.
- Runtimes provide **stream ordering**; cross‑stream visibility is guaranteed at fences/events.

### 5.3 Boundary Lens Access (optional)

If `uses_boundary=1`, an address pair `(x∈[0,48), y∈[0,256))` maps to global addresses via an implementation-defined tiling function `addr = Φ(class_id, (x,y), PB)`.

### 5.4 Address Modes

Atlas ISA defines three address modes for memory operations (LDG, STG, LDS, STS):

#### 5.4.1 BufferOffset

**Format**: `{handle: u32, offset: usize}`

**Semantics**: Direct offset into a linear buffer allocation.

**Effective Address**:
```
effective_addr = buffer_base_address(handle) + offset
```

**Bounds Checking**:
- Runtime MUST validate `handle` refers to a valid allocation
- Runtime MUST validate `offset + sizeof(ty) <= buffer_size(handle)`
- Out-of-bounds access results in execution error

**Usage**: Primary mode for array/tensor access. Most performant on all targets.

**Example**:
```rust
LDG { ty: F32, dst: r0, addr: BufferOffset { handle: 42, offset: 128 } }
// Load f32 from buffer 42 at byte offset 128 into register r0
```

**Target Mapping**:
- CPU: Direct pointer arithmetic `*(buffer_ptr + offset)`
- GPU: `global_mem[handle_base + offset]`
- TPU/PJRT: Device buffer access at offset

#### 5.4.2 PhiCoordinate

**Format**: `{class: u8, page: u8, byte: u8}`

**Semantics**: 2D boundary lens addressing into cache-resident memory.

**Constraints**:
- `class < 96` (C96 resonance class)
- `page < 48` (Φ boundary lens page dimension)
- `byte < 256` (Φ boundary lens byte dimension)

**Effective Address**:
```
linear_offset = class * (48 * 256) + page * 256 + byte
effective_addr = boundary_pool_base + linear_offset
```

**Bounds Checking**:
- Runtime MUST validate `class < 96`
- Runtime MUST validate `page < 48`
- Runtime MUST validate `byte < 256`
- Runtime MUST ensure boundary pool is allocated (size = 96 * 48 * 256 = 1,179,648 bytes)

**Usage**: Access to cache-resident Atlas boundary lens memory. Enables structured 2D tiling patterns.

**Example**:
```rust
STG { ty: U8, src: r1, addr: PhiCoordinate { class: 5, page: 12, byte: 200 } }
// Store u8 from r1 to boundary pool at (class=5, page=12, byte=200)
```

**Target Mapping**:
- CPU: L2-resident pool (1.18 MB) with linear offset calculation
- GPU: Shared memory or texture memory with 2D layout
- TPU/PJRT: On-chip memory with 2D tiling

#### 5.4.3 RegisterIndirect

**Format**: `{base: Register, offset: i64}`

**Semantics**: Base-displacement addressing using register value.

**Constraints**:
- `base` register MUST hold u64 type (64-bit pointer)
- Signed `offset` allows positive or negative displacement

**Effective Address**:
```
base_value = register_file[base].as_u64()
effective_addr = base_value + offset  // with overflow checking
```

**Bounds Checking**:
- Runtime MUST validate `base` register contains u64 type
- Runtime MUST validate address arithmetic does not overflow
- Runtime MAY validate effective address against known allocations (CPU backend)
- Runtime MAY allow unchecked access (GPU backend, caller responsibility)

**Usage**: Dynamic pointer-based access, indexed arrays, indirect lookups.

**Safety**: Implementations MUST document their validation strategy:
- **Option A (Validated)**: Check against tracked allocations (recommended for CPU)
- **Option B (Unchecked)**: No validation, caller ensures safety (GPU)

**Example**:
```rust
LDG { ty: F32, dst: r2, addr: RegisterIndirect { base: r10, offset: 64 } }
// Load f32 from address (r10 + 64) into r2
// Requires r10 to hold u64 pointer value
```

**Target Mapping**:
- CPU: Validated pointer dereference `*(base_ptr + offset)`
- GPU: Direct address calculation `mem[base + offset]`
- TPU/PJRT: Indirect buffer access (may require explicit bounds)

#### 5.4.4 Address Mode Selection Guidelines

| Use Case | Recommended Mode | Rationale |
|----------|------------------|-----------|
| Array/tensor element access | BufferOffset | Direct, bounds-checked, all targets |
| Atlas boundary lens operations | PhiCoordinate | 2D structured, cache-resident |
| Dynamic indexing, pointer chasing | RegisterIndirect | Flexible, pointer-based |
| Structured grid traversal | PhiCoordinate or BufferOffset | Depends on memory layout |
| Random access patterns | BufferOffset or RegisterIndirect | Depends on index computation |

### 5.5 Target mappings (normative guidance)

- **CPU:** Global maps to host DRAM; Shared maps to stack/heap scratch; `MEM.FENCE{device}` ⇒ thread-fence; `{system}` ⇒ atomic_thread_fence seq_cst.
- **GPU:** Global/Shared map to device memory and SM shared; fences use device/system scopes; default relaxed model with explicit fences.
- **TPU/PJRT:** Global buffers via PJRT device buffers; fences via PJRT events; Shared via per-program scratch.
- **DSP/SIMD:** Global in device memory; Shared in on-chip SRAM; fences via device memory barriers.

---

## 6. Types & Numerics

- **Scalars:** `i8/i16/i32/i64`, `u8/u16/u32/u64`, `f16/bf16/f32/f64`.
- **AtlasRatio:** ABI-safe rational (`{ numer: i64, denom: i64 }`) used for resonance deltas. Denominator must be non-zero.
- **Vectors:** packed 2/4‑wide variants (optional).
- **Predicates:** 1‑bit booleans.
- **Resonance units:** implementation chooses integer or fixed‑point; deltas accumulate into `R[96]`.

Rounding mode: round‑to‑nearest‑ties‑to‑even unless specified by instruction variant.

---

## 7. Instruction Set (Abstract Semantics)

Notation: `dst ← op(srcs)`; `@p{…}` predicated execution; `[]` optional. Mnemonics are abstract; concrete encodings are implementation-defined.

### 7.1 Data Movement

- **LDG/ STG** — Global load/store: `ldg.{ty} r, [addr]`; `stg.{ty} [addr], r`.
- **LDS/ STS** — Shared load/store.
- **MOV/ CVT** — Register move, type convert.

### 7.2 Arithmetic

- **ADD/ SUB/ MUL/ MAD** — integer/float variants.
- **FMA** — fused multiply-add.
- **DIV/ SQRT** — optional precise/approx variants.
- **MIN/ MAX/ ABS/ NEG** — per‑type.

### 7.3 Logic & Predication

- **AND/ OR/ XOR/ NOT** — integer/bitwise.
- **SETcc** — comparisons → predicate or integer (e.g., `set.lt.f32 p, a, b`).
- **SEL** — select by predicate.

### 7.4 Control Flow & Program Structure

#### 7.4.1 Program Type

A **Program** is a structured container for instruction sequences with label metadata:

```rust
pub struct Program {
    pub instructions: Vec<Instruction>,
    pub labels: HashMap<String, usize>,
}
```

**Constructors**:
- `Program::new()` — empty program
- `Program::from_instructions(Vec<Instruction>)` — create from instruction vector
- `Program::with_capacity(usize)` — pre-allocate capacity

**Label Management**:
- `add_label(name: impl Into<String>) -> ProgramResult<()>` — add label at current position
- `resolve_label(name: &str) -> Option<usize>` — resolve label to instruction index
- `has_label(name: &str) -> bool` — check if label exists

**Error Handling**: Label operations return `ProgramResult<T>`:

```rust
pub enum ProgramError {
    DuplicateLabel(String, usize),      // duplicate label definition
    UndefinedLabel(String, usize),      // label referenced but not defined
    InvalidLabelTarget(String, usize, usize),  // label points beyond program
}
```

#### 7.4.2 Control Flow Instructions

- **BRA {target: Label, pred: Option<Predicate>}** — branch to label (structured or predicated forms preferred)
  - Unconditional: `BRA {target: "loop_start", pred: None}`
  - Conditional: `BRA {target: "exit", pred: Some(p0)}`
  - Requires label to be defined in Program.labels
  - Backend resolves label name to instruction index

- **CALL {target: Label}** — in‑kernel subroutine call (optional; compilers may inline)
  - Pushes return address (PC+1) onto call stack
  - Branches to label target
  - Call stack depth limited to 256 levels

- **RET** — return from subroutine
  - Pops return address from call stack
  - Sets PC to return address
  - Error if call stack is empty

- **LOOP {counter: Register, target: Label}** — canonical counted loops (lowerable to BRA)
  - Decrements counter register
  - Branches to target if counter > 0
  - Falls through if counter == 0
  - Counter must hold u32 type

**Label Resolution**: Backends MUST resolve labels before execution:
1. Validate all referenced labels exist in Program.labels
2. Validate all labels point to valid instruction indices
3. Build runtime jump table mapping labels to instruction indices
4. Report UndefinedLabel or InvalidLabelTarget errors before execution begins

### 7.5 Synchronization

- **BAR.SYNC** — block barrier; all lanes wait; shared/global _release_ for writes prior to barrier, _acquire_ for reads after.
- **MEM.FENCE** — memory fence with scope `{block, device, system}`.

### 7.6 Atlas‑Specific

- **CLS.GET** — read class id into `r`: `cls.get r`.
- **MIRROR** — map class id `c` to its mirror `c'`: `mirror r_c' , r_c`.
- **UNITY.TEST** — `pred ← unity?(r_c)`.
- **NBR.COUNT / NBR.GET** — neighbor enumeration from 1‑skeleton: `k←nbr.count(c)`; `c_i←nbr.get(c, i)`.
- **RES.ACCUM** — accumulate signed delta `Δ[96]` (each entry a reduced rational) into `R[96]`; privileged or mediated (runtime-managed). Kernel form: `res.accum r_delta_id, r_value` where `r_value` holds `{numer, denom}` and the runtime validates neutrality via the Atlas core invariants when required.
- **PHASE.GET / PHASE.ADV** — read/advance modulo‑768 phase (runtime may virtualize advance).
- **BOUND.MAP** — `(x,y) → addr` via boundary lens: `addr ← bound.map c, x, y, PB`.
- **CHECK.UNITY_NEUTRAL** — hint or assert; if violated, raises fault per policy.

### 7.7 Reduction & Collective (intra‑block)

- **REDUCE.ADD/MIN/MAX** — tree reductions within a block to shared/global.
- **SCAN.EXCL/INCL** — prefix-sum forms.

### 7.8 Transcendentals (optional set)

- **EXP/ LOG/ SIN/ COS/ TANH** — accuracy levels `{approx, fast, precise}`.

### 7.9 Legalization & target mapping

Backends MUST provide a **legalization table** mapping each abstract instruction to:

- A **native op** (1:1), or
- A **macro-op** expansion (sequence of native ops with equivalent semantics), or
- A **library call** (e.g., transcendentals), or
- **Reject** (if unsupported; only allowed for optional instructions).

Legalization MUST preserve ordering, memory, and invariant semantics. Fences/barriers MUST be preserved or strengthened, never weakened.

---

## 8. Atlas Invariants — Semantics & Enforcement

### 8.1 Mirror Safety

If `mirror_safe=1`, the kernel’s effect is invariant under classwise mapping `c → mirror(c)`. The runtime MAY fuse or reorder conjugate operations.

### 8.2 Unity Neutrality

If `unity_neutral=1`, the net change to `R[96]` over the kernel launch must be zero: `Σ_launch ΔR = 0`. Deltas are expressed as exact rationals; denominators must never be zero and SHOULD be reduced. Implementations:

- **Proved:** compiler derives Δ statically and the runtime checks post‑launch.
- **Reported:** kernel writes its Δ to a side channel; runtime checks equals zero. Violations ⇒ **launch fault**.

### 8.3 Boundary Footprint

Accesses via `BOUND.MAP` must remain within `(x_min..x_max, y_min..y_max)` when declared; out‑of‑range ⇒ fault or clamped per policy.

### 8.4 Phase Window

A kernel with `(phase_begin, phase_span)` only executes when `P ∈ [begin, begin+span) mod 768`. Otherwise queued or rejected.

### 8.5 1‑Skeleton Respect

Legal neighbor traversals must use `NBR.*`; attempts to use non‑neighbors MAY be rejected in checked builds.

---

## 9. Kernel ABI (Parameters & Launch)

- Parameters are passed as a **flat POD block** `PB` laid out per a descriptor `(size, align)` list.
- Pointers in `PB` are 64‑bit device addresses; kernels must call the runtime memory interface to map them into host/lane addressable regions (in software implementations) or dereference directly (in hardware implementations).
- Launch provides: grid dims, block dims, shared scratch size, and stream/phase context.

---

## 10. Exception & Error Model

- **Illegal Memory:** out‑of‑range global/shared/local access.
- **Invariant Violation:** unity neutrality failure; disallowed neighbor; phase or boundary mismatch.
- **Arithmetic Faults:** NaNs allowed; divide-by-zero behavior defined per numeric mode.
- **Trap/Assert:** kernel can request abort with code.

Fault policy: **fail this launch**; context may remain valid or be poisoned, per runtime policy.

---

## 11. Conformance Profiles

Base instruction subsets:

- **Profile S (Scalar Core):** Loads/stores; int/float add/mul; barriers; CLS/MIRROR/UNITY/NBR/PHASE/BOUND; reductions.
- **Profile V (Vector/Matrix):** Adds vector packs and GEMM/CONV library calls (lowered to S or accelerator paths).
- **Profile T (Transcendentals):** Adds EXP/LOG/SIN/COS/TANH.

### 11.1 Target Profiles (capability matrix)

Backends declare one or more **target profiles**:

- **CPU:** S required; V optional (via SIMD); T optional (libm/vecmath). Strong memory model.
- **GPU:** S required; V encouraged (tensor cores map via library); T optional.
- **TPU/PJRT-class:** S required; V required (matrix units); T optional.
- **DSP/SIMD:** S required; V via fixed-width SIMD; T optional.
- **FPGA/ASIC:** S required; V/T via synthesized IP blocks.

Backends also expose feature bits: vector width, native f16/bf16, fast math, barrier scopes, boundary lens accelerator presence.

---

## 12. Performance Hints (Non‑binding)

- Prefer boundary-aligned tiles `(48×256)` for 2D kernels.
- Use mirror-safe flags to enable scheduler fusions.
- Batch small kernels to amortize phase checks and resonance accounting.

---

## 13. Verification Hooks

- **Resonance Snapshot:** read `R[96]` pre/post kernel.
- **Delta Channel:** write per‑kernel Δ (as `AtlasRatio[96]`) to a well-known global symbol for runtime audit.
- **Trace Markers:** lightweight begin/end stamps for profiling.

---

## 14. Example (Pseudocode)

**Vector Add (unity-neutral, mirror-safe):**

```
.kernel vec_add(PB: {ptr a, ptr b, ptr c, u32 n}) [mirror_safe, unity_neutral]
  r_class ← cls.get
  i ← lane_linear_id()
  if i ≥ n: return
  va ← ldg.f32 [a + 4*i]
  vb ← ldg.f32 [b + 4*i]
  vc ← add.f32 va, vb
  stg.f32 [c + 4*i], vc
  // implicit ΔR = 0
.end
```

**Host-side (Rust) metadata prep with rational deltas:**

```rust
use atlas_core::AtlasRatio;
use atlas_isa::{KernelMetadata, ClassMask, ResonanceClass};

let mut meta = KernelMetadata::new("vector_add");
meta.classes_mask = ClassMask::single(ResonanceClass::new(10)?);
meta.unity_neutral = true;
meta.mirror_safe = true;

// Runtime-provided resonance delta for validation
let mut delta = [AtlasRatio::new_raw(0, 1); 96];
delta[10] = AtlasRatio::new_raw(1, 4);
delta[42] = AtlasRatio::new_raw(-1, 4);

meta.validate()?;              // structural checks
atlas_core::verify_unity_neutrality(&delta)?; // exact rational neutrality check
```

**Boundary‑tiled blur (uses_boundary):**

```
.kernel blur3x3(PB:{ptr img_in, ptr img_out, u32 pitch}) [uses_boundary, boundary:(0,48)x(0,256)]
  c  ← cls.get
  (x,y) ← (thread.x % 48, thread.y % 256)
  acc ← 0
  for dx in -1..1:
    for dy in -1..1:
      addr ← bound.map c, (x+dx)mod48, (y+dy)mod256, PB
      acc  ← add.f32 acc, ldg.f32 [img_in + addr]
  acc ← mul.f32 acc, 1/9
  addr_o ← bound.map c, x,y, PB
  stg.f32 [img_out + addr_o], acc
.end
```

---

## 15. Versioning

- **ISA Version:** 1.1 (this document). Backward-compatible minor increments add instructions/attributes; major increments may alter semantics.
- **Discovery:** targets expose supported profiles, numeric features, invariant enforcement levels, and legalization summaries via a query interface.

---

## 16. Implementation Notes (for vGPU/vTPU)

- Software backends supply the memory mapping API (`map/unmap`) and maintain `R[96]` and `P` per stream.
- Hardware/accelerated targets can implement `RES.ACCUM`/`PHASE` as real instructions; software may virtualize via runtime calls.
- The Atlas Module ABI used by vGPU/vTPU (§6 of vGPU spec) is the canonical container for kernels and metadata.

---

---

## 17. Target Mapping Guides (normative, concise)

### 17.1 CPU (x86-64/ARM64/RISC-V)

- **Grid/Block/Lane:** outer parallel-for / thread pool; block = chunk; lane = scalar or SIMD lane.
- **BAR.SYNC:** thread barrier or `std::barrier`.
- **Fences:** C++ atomics: block→none/compile-time; device→`atomic_thread_fence(acq_rel)`; system→`seq_cst`.
- **Vectorization:** use auto-vectorize or explicit packs for V profile.

### 17.2 GPU (CUDA/Vulkan/Metal)

- **Grid/Block/Lane:** native blocks/threads.
- **Shared:** on-chip shared; **Global:** device memory.
- **Fences:** map scopes accordingly; honor barrier semantics.

### 17.3 TPU/PJRT

- **Executable:** StableHLO/XLA lower to Atlas kernels; tiles/replicas = grid.
- **Fences:** PJRT events; buffers = device buffers.

### 17.4 DSP/SIMD

- **Lane:** SIMD element; reductions via horizontal ops.
- **Shared:** local SRAM.

### 17.5 FPGA/ASIC

- **Legalization:** synthesize kernels to pipelines; BAR.SYNC via pipeline flush points; PHASE as schedule counter.

---

## 18. Implementer’s Checklists (per target)

Each checklist is **normative** for a conforming backend at Profile S. Items with (★) are required for Profile V, (✚) for T.

### 18.1 CPU Backend Checklist

1. **Parallel runtime**: thread pool with work-stealing; grid→nested loops; block→chunk.
2. **Barriers & fences**: implement BAR.SYNC via `std::barrier`; MEM.FENCE `{block,device,system}` via C++ atomics mapping.
3. **Memory**: Global = host DRAM; Shared = per-block heap/stack scratch; Local = spill.
4. **Legalization table**: map FMA/MAD to fused or `mul+add`; REDUCE to tree reductions; PHASE/RES.ACCUM via runtime calls.
5. **Atlas invariants**: R[96] accumulator per stream; enforce `unity_neutral` and phase window checks.
6. **Vectorization (★)**: auto-vectorize loops (pragma hints) or use intrinsics for pack ops.
7. **Transcendentals (✚)**: libm/vecmath calls; document accuracy.

### 18.2 GPU Backend Checklist

1. **Launch mapping**: grid/block/lane map 1:1 to native.
2. **Shared memory**: allocate per-block scratch; obey size caps; spill gracefully.
3. **Barriers**: BAR.SYNC → `__syncthreads` / subgroup barrier; fences with proper scope.
4. **Memory model**: DtoH/HtoD visibility at stream sync; global coherence per vendor model.
5. **Legalization**: REDUCE/SCAN via warp/block collectives; RES.ACCUM/PHASE via device-side runtime calls.
6. **Tensor paths (★)**: GEMM/CONV via vendor libraries or tensor cores; fall back to tiling.
7. **Math (✚)**: use fast/precise variants as advertised.

### 18.3 TPU/PJRT Backend Checklist

1. **PJRT surface**: buffers, compile(StableHLO→Atlas), execute, events, serialize.
2. **Lowering**: fuse elementwise; tile dots/convs; emit Atlas kernels + metadata.
3. **Scheduling**: assign phase; boundary lens optional; preserve data deps.
4. **Collectives**: stub or implement AllReduce via PJRT (Phase 2).
5. **Invariants**: enforce unity_neutral; maintain R[96]; debug-mode neighbor checks.
6. **AOT cache**: serialize executables; version pins to ISA 1.1.

### 18.4 DSP/SIMD Backend Checklist

1. **SIMD width**: expose vector width; pack loads/stores; horizontal reductions.
2. **SRAM**: map Shared to on-chip memory; banking conflicts documented.
3. **Fences**: device/system barriers per vendor.
4. **Legalization**: loops for ops lacking SIMD forms; library calls for transcendentals.
5. **Invariants**: same as CPU.

### 18.5 FPGA/ASIC Backend Checklist

1. **HLS flow**: lower kernels to pipelines; capture BAR.SYNC as pipeline fences.
2. **Memory**: on-chip BRAM for Shared; AXI/global for Global.
3. **Scheduler**: implement PHASE as programmable counter; RES.ACCUM in control plane.
4. **Verification**: formal properties for unity_neutral, boundary windows, and neighbor use.
5. **Legalization**: compile-time expansion; no dynamic stack.

---

## 19. Conformance Test Suite (CTS) Outline

**Tier 0 (Sanity)**: LD/ST roundtrip, ADD/MUL, barriers, fences, CLS/MIRROR/UNITY/NBR/PHASE sanity.\
**Tier 1 (Numerics)**: FMA accuracy, reductions, scans, mixed-precision conversions.\
**Tier 2 (Invariants)**: unity_neutral enforcement, phase gating, boundary window violations.\
**Tier 3 (Parallelism)**: multi-block ordering, cross-stream events (if supported).\
**Tier 4 (Profiles)**: vector packs (V), transcendentals (T).\
Each test declares required profiles and acceptable error bounds.

---

## 20. Legalization Table Template

| Instr      | Profile | Semantics              | CPU                | GPU                 | TPU/PJRT        | DSP            | FPGA            |
| ---------- | ------- | ---------------------- | ------------------ | ------------------- | --------------- | -------------- | --------------- |
| FMA.f32    | S       | `a*b + c` (round-even) | `fma` or `mul+add` | native or `mul+add` | library         | `mul+add`      | pipeline op     |
| REDUCE.ADD | S       | sum within block       | tree w/ barrier    | warp/block reduce   | library/fused   | horiz ops      | pipeline reduce |
| BAR.SYNC   | S       | block barrier          | `std::barrier`     | `__syncthreads`     | PJRT tile fence | device barrier | pipeline fence  |
| RES.ACCUM  | S       | R[96]+=Δ               | runtime call       | runtime call        | runtime         | runtime        | control plane   |
| PHASE.GET  | S       | read phase             | runtime            | runtime             | runtime         | runtime        | reg/counter     |

(Backends MUST publish their completed table in docs or via query interface.)

---

## 21. Sample Legalization — x86‑64 (AVX2 + FMA, Linux)

**Target profile:** CPU (S+V), optional T via libm/vecmath.\
**Assumptions:** LP64, little‑endian, SSE2 baseline, AVX2+FMA available, pthreads.

### 21.1 Scalar & Flow

| Instr    | Lowering                                                   |
| -------- | ---------------------------------------------------------- |
| MOV/CVT  | `mov`/`vcvtt*` intrinsics or scalar ops                    |
| BRA/LOOP | structured C++ control flow (no UB), branch hints optional |
| CALL/RET | inline or `noexcept` functions                             |

### 21.2 Memory & Sync

| Instr             | Lowering                                                   |
| ----------------- | ---------------------------------------------------------- |
| LDG/STG.{i32,f32} | scalar `*(T*)addr` or `_mm256_i32gather_epi32` for gathers |
| LDS/STS           | access to per-block scratch on heap/stack                  |
| BAR.SYNC          | `std::barrier` or `pthread_barrier_t`                      |
| MEM.FENCE.block   | compiler barrier (none)                                    |
| MEM.FENCE.device  | `atomic_thread_fence(memory_order_acq_rel)`                |
| MEM.FENCE.system  | `atomic_thread_fence(memory_order_seq_cst)`                |

### 21.3 Arithmetic & Vector (AVX2/FMA)

| Instr             | Lowering                                   |
| ----------------- | ------------------------------------------ |
| ADD/SUB/MUL.f32x8 | `_mm256_add_ps/sub_ps/mul_ps`              |
| FMA.f32x8         | `_mm256_fmadd_ps`                          |
| MIN/MAX.f32x8     | `_mm256_min_ps/_mm256_max_ps`              |
| AND/OR/XOR        | `_mm256_and_ps/_mm256_or_ps/_mm256_xor_ps` |
| SETcc             | `_mm256_cmp_ps` + movemask/select          |
| SEL               | `_mm256_blendv_ps`                         |

### 21.4 Reductions & Scans

| Instr                  | Lowering                                                             |
| ---------------------- | -------------------------------------------------------------------- |
| REDUCE.ADD.f32 (block) | tree reduce in shared scratch + barrier; lane‑local `_mm256_hadd_ps` |
| SCAN.INCL/EXCL         | work‑efficient scan using Blelloch algorithm over chunks             |

### 21.5 Atlas‑Specific

| Instr                     | Lowering                                                 |
| ------------------------- | -------------------------------------------------------- |
| CLS.GET/MIRROR/UNITY.TEST | inline functions over constants/tables                   |
| NBR.COUNT/GET             | table lookup (constexpr arrays)                          |
| RES.ACCUM                 | runtime call: `res_accum(delta_id, value)` (thread‑safe) |
| PHASE.GET/ADV             | runtime call reading TLS phase counter                   |
| BOUND.MAP                 | inline address calculation from `(class_id,(x,y),PB)`    |
| CHECK.UNITY_NEUTRAL       | debug assert; prod: elided or guarded                    |

### 21.6 Transcendentals (T profile)

| Instr                | Lowering                                             |
| -------------------- | ---------------------------------------------------- |
| EXP/LOG/SIN/COS/TANH | SVML/VecLib if available; else libm scalar in a loop |

### 21.7 Notes

- Prefer SoA layouts for vector kernels to maximize contiguous loads.
- Align hot arrays to 64 bytes; use `restrict`/`assume_aligned` where sound.
- Use `-ffast-math` only if kernel metadata allows (no strict IEEE needed).

---

## 22. Sample Legalization — CUDA GPU (SM80+, CUDA C++/PTX)

**Target profile:** GPU (S+V), optional T via device math.\
**Assumptions:** SM80 (Ampere) or newer; CUDA C++ with inline PTX allowed.

### 22.1 Launch & Memory

| Instr             | Lowering                                                 |
| ----------------- | -------------------------------------------------------- |
| Grid/Block/Lane   | native <<\<grid,block,shared,stream>>>                   |
| LDG/STG.{i32,f32} | `ld.global.*` / `st.global.*` (C++ loads/stores map 1:1) |
| LDS/STS           | `ld.shared.*` / `st.shared.*` (via `__shared__` arrays)  |
| MEM.FENCE.block   | `__threadfence_block()`                                  |
| MEM.FENCE.device  | `__threadfence()`                                        |
| MEM.FENCE.system  | `__threadfence_system()`                                 |
| BAR.SYNC          | `__syncthreads()`; warp-level via `__syncwarp()`         |

### 22.2 Arithmetic & Vector

| Instr           | Lowering                               |
| --------------- | -------------------------------------- |
| ADD/SUB/MUL.f32 | native SASS; CUDA `+,-,*`              |
| FMA.f32         | `fmaf` or inline PTX `fma.rn.f32`      |
| MIN/MAX         | `fminf/fmaxf`                          |
| Vector packs    | use `float2/4` or LDG.128 when aligned |

### 22.3 Reductions & Scans

| Instr              | Lowering                                                 |
| ------------------ | -------------------------------------------------------- |
| REDUCE.ADD (warp)  | `__shfl_down_sync` tree reduce                           |
| REDUCE.ADD (block) | warp reduce + shared-memory epilogue + `__syncthreads()` |
| SCAN               | CUB/Thrust block scan or custom shared-memory scan       |

### 22.4 Atlas‑Specific

| Instr                     | Lowering                                                                |
| ------------------------- | ----------------------------------------------------------------------- |
| CLS.GET/MIRROR/UNITY.TEST | constant memory tables; inline functions                                |
| NBR.COUNT/GET             | constant memory neighbor lists                                          |
| RES.ACCUM                 | device→host or device-global atomic add to R[96]; commit at stream sync |
| PHASE.GET/ADV             | device-global counter per stream; read via global load                  |
| BOUND.MAP                 | inline address calc; optionally texture/surface for 2D                  |
| CHECK.UNITY_NEUTRAL       | `assert` in debug; NOP in release with runtime audit                    |

### 22.5 Transcendentals (T)

| Instr                | Lowering                                                  |
| -------------------- | --------------------------------------------------------- |
| EXP/LOG/SIN/COS/TANH | `__expf`, `__logf`, `sinf/cosf/tanhf`; or fast-math flags |

### 22.6 Notes

- Use cooperative groups for finer barriers if needed.
- Prefer LDG.128/STS.128 for bandwidth; align shared memory to avoid bank conflicts.
- Use stream-ordered memory ops to align with RES/PHASE accounting.

---

## 23. Sample Legalization — PJRT Device (StableHLO/XLA → Atlas)

**Target profile:** TPU/PJRT-class (S+V), optional T via HLO conversions.\
**Assumptions:** PJRT C API, StableHLO input.

### 23.1 Program Flow

| Stage         | Lowering                                                     |
| ------------- | ------------------------------------------------------------ |
| HLO module    | parse/verify StableHLO; canonicalize                         |
| Fusion        | fuse elementwise ops around GEMM/CONV                        |
| Tiling        | choose tile sizes; map to boundary lens where beneficial     |
| Kernelization | emit Atlas kernels with metadata (class masks, unity, phase) |
| Scheduling    | assign phases; produce stream-ordered plan                   |

### 23.2 Instruction Mapping

| Atlas Instr     | Lowering to HLO/Runtime                     |
| --------------- | ------------------------------------------- |
| ADD/SUB/MUL/FMA | HLO fused to single kernel; or library GEMM |
| REDUCE/SCAN     | HLO reduces/scans → tiled Atlas reductions  |
| BAR.SYNC        | tile fence; ensure per-tile ordering        |
| RES.ACCUM/PHASE | runtime hooks around kernel launch          |
| BOUND.MAP       | tile index to buffer slices                 |

### 23.3 Notes

- Executables are cached (Serialize/Deserialize).
- Collectives: map to PJRT `AllReduce/AllGather` when enabled; else single‑device fallback.

---

## 24. Sample Legalization — Apple GPU (Metal)

**Target profile:** GPU (S+V), optional T via Metal fast/precise math.\
**Assumptions:** Apple Silicon (M‑series), Metal Shading Language (MSL), command buffers, threadgroup memory.

### 24.1 Launch & Memory

| Instr            | Lowering                                                                                                       |
| ---------------- | -------------------------------------------------------------------------------------------------------------- |
| Grid/Block/Lane  | `dispatchThreadgroups` (grid), `threadsPerThreadgroup` (block), thread = lane                                  |
| LDG/STG          | device buffer reads/writes via pointer/address space `device`                                                  |
| LDS/STS          | `threadgroup` memory arrays                                                                                    |
| MEM.FENCE.block  | `threadgroup_barrier(mem_flags::mem_threadgroup)`                                                              |
| MEM.FENCE.device | `threadgroup_barrier(mem_flags::mem_device)` (within TG); inter‑TG via `MTLFence` or command-buffer boundaries |
| MEM.FENCE.system | command buffer completion or `MTLEvent` (if available)                                                         |
| BAR.SYNC         | `threadgroup_barrier()`; SIMD‑group sync via `simdgroup_barrier()`                                             |

### 24.2 Arithmetic & Vector

| Instr               | Lowering                                                            |
| ------------------- | ------------------------------------------------------------------- |
| ADD/SUB/MUL/FMA.f32 | native MSL ops; `fma()` for fused                                   |
| Vector packs        | use `packed_float2/4` and aligned loads; or manual struct-of-arrays |

### 24.3 Reductions & Scans

| Instr                    | Lowering                                                            |
| ------------------------ | ------------------------------------------------------------------- |
| REDUCE.ADD (simdgroup)   | `simd_sum()` or explicit shuffles                                   |
| REDUCE.ADD (threadgroup) | simdgroup reduce → shared TG accumulation + `threadgroup_barrier()` |
| SCAN                     | TG shared‑memory scan; simdgroup intrinsics for lane ops            |

### 24.4 Atlas‑Specific

| Instr                     | Lowering                                                              |
| ------------------------- | --------------------------------------------------------------------- |
| CLS.GET/MIRROR/UNITY.TEST | constant buffers / function‑constant tables                           |
| NBR.COUNT/GET             | constant buffer lookup                                                |
| RES.ACCUM                 | write to device buffer; flush at command buffer boundary or via fence |
| PHASE.GET/ADV             | device buffer counter per stream/queue                                |
| BOUND.MAP                 | inline address calc; use 2D textures if sampling fits                 |
| CHECK.UNITY_NEUTRAL       | `assert` in debug; runtime audit otherwise                            |

### 24.5 Notes

- SIMD‑group (wave) size is implementation‑defined; always use simdgroup intrinsics rather than assuming 32/64.
- Prefer `fast::` math only if kernel metadata allows fast‑math.

---

## 25. Sample Legalization — RISC‑V Vector (RV64GCV)

**Target profile:** CPU/DSP hybrid (S+V), optional T via libm or vector libs.\
**Assumptions:** RV64 with V extension (RVV), VLEN implementation‑defined; intrinsics available.

### 25.1 Parallel & Memory

| Instr            | Lowering                                                                |
| ---------------- | ----------------------------------------------------------------------- |
| Grid/Block/Lane  | host threads (grid/block); lane = vector element controlled by `vsetvl` |
| LDG/STG          | `vlseX/vsseX` or `vleX/vseX` with proper alignment                      |
| LDS/STS          | per‑block scratch in host memory                                        |
| MEM.FENCE.block  | compiler barrier                                                        |
| MEM.FENCE.device | `fence rw, rw`                                                          |
| MEM.FENCE.system | `fence iorw, iorw` (strongest)                                          |
| BAR.SYNC         | host thread barrier (e.g., `std::barrier`)                              |

### 25.2 Arithmetic & Vector (RVV)

| Instr             | Lowering                                  |
| ----------------- | ----------------------------------------- |
| ADD/SUB/MUL.f32   | `vadd.vv/vsub.vv/vmul.vv`                 |
| FMA.f32           | `vfmacc.vf` / `vfmacc.vv`                 |
| MIN/MAX.f32       | `vfmin.vv/vfmax.vv`                       |
| Logic/Predication | masks via `vmseq/vmslt`; `vmerge` for SEL |
| CVT               | `vfcvt.*`/`vfncvt.*`                      |

### 25.3 Reductions & Scans

| Instr          | Lowering                                                     |
| -------------- | ------------------------------------------------------------ |
| REDUCE.ADD.f32 | `vfredsum.vs` with neutral init; hierarchical across threads |
| SCAN           | software scan using chunked vector ops + scalar carry        |

### 25.4 Atlas‑Specific

| Instr                     | Lowering                                                   |
| ------------------------- | ---------------------------------------------------------- |
| CLS.GET/MIRROR/UNITY.TEST | scalar tables; broadcast via `vmv.v.x`                     |
| NBR.COUNT/GET             | scalar lookup; broadcast IDs to vector registers           |
| RES.ACCUM                 | atomic adds to R[96] table (host) or per‑thread then merge |
| PHASE.GET/ADV             | TLS counter read/increment                                 |
| BOUND.MAP                 | scalar address calc then vectorized base+stride addressing |

### 25.5 Notes

- Always program `vl` via `vsetvl` to match remaining elements; handle tails with masks.
- Favor SoA memory layout to maximize contiguous `vle/vse` throughput.

---

**End of Atlas ISA Specification (v1.1 Universal Draft)**\*\*

---

**End of Atlas ISA Specification (v1.1 Universal Draft)**
