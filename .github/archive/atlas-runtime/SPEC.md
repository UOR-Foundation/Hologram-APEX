# Atlas Runtime Specification (Hologram Runtime)

**Version:** v1.0‑draft
**Date:** 2025‑10‑18
**Status:** Draft for implementation feedback
**Authority:** UOR Foundation (Atlas / Hologram Program)

> This document defines the **Atlas Runtime** (a.k.a. **Hologram Runtime**)—the portable execution environment and host API that implements the Atlas ISA’s computational memory model on heterogeneous targets (analog in‑memory engines, CPUs, GPUs, TPUs, FPGAs).

---

## 1. Scope

The runtime provides:

- An **object model** for constructing and managing an **Atlas Space**: resonance classes (C96), boundary lens (Φ), 1‑skeleton neighbor table, mirror involution, unity set, phase counter, and resonance accumulators.
- A **kernel execution model** with deterministic **phase‑ordered** scheduling and conformance checks on class traversal (neighbor legality, mirror safety, unity neutrality, boundary windows).
- A **host API/ABI** to allocate/bind memory, install Φ mappings, supply structural tables, submit kernels, and collect telemetry.
- **Portability profiles** mapping Atlas semantics to concrete backends (analog compute memory, CPU, GPU).

This spec is **normative** for the host/runtime boundary. ISA and kernel IR are referenced where needed but specified separately.

---

## 2. Conformance Language

Keywords **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are to be interpreted as in RFC 2119.

---

## 3. Concepts & Terms

- **Atlas Space**: The logical computational memory space governed by Atlas structure; parameterized by 96 **resonance classes**, each optionally exposing a **48×256 boundary** lens.
- **Resonance Classes (C96)**: Class identifiers `c ∈ {0,…,95}` with per‑class state and accumulators.
- **Boundary Lens (Φ)**: A mapping from `(class c, x∈[0,48), y∈[0,256)) → global_address`. Φ provides spatial locality guarantees but does **not** mandate physical tiling.
- **1‑Skeleton (Neighbors)**: The fixed adjacency relation over classes; indexes neighbor transitions permitted for data‑dependent steps.
- **Mirror (Involution)**: A pairing `mirror(c)` with `mirror(mirror(c)) = c`.
- **Unity Set**: Distinguished zero points / neutral elements per class that enable neutral operations.
- **Phase Counter**: Global modulo‑768 phase; used for temporal ordering and burst scheduling.
- **Resonance Accumulator R[96]**: Per‑class scalar/vector accumulator updated by kernels.

---

## 4. Architectural Model

### 4.1 Global Address Space

The runtime exposes a **flat 64‑bit global address space** to kernels. Memory objects (tensors, tables, scratch) are created by the host and bound into that space. The boundary lens (Φ) is a logical **address transform** layered over the same space.

### 4.2 Atlas Structures (per‑device)

- **Neighbor Table**: `neighbors: [C96][N_i]` (N_i MAY vary per class). Entries are `u8` class IDs.
- **Mirror Table**: `mirror: [C96] -> [0..95]` with involutive property.
- **Unity Mask/Set**: Implementation‑defined payload identifying neutral elements; exposed as a constant buffer.
- **Phase**: `phase: u16 mod 768`; readable by kernels; advanced by the runtime during phase‑ordered bursts.
- **Resonance Accumulator**: `R: array[96] of AtlasRatio` **with canonical exact rational semantics**. Each `AtlasRatio` represents a reduced fraction `n/d ∈ ℚ` where `n ∈ ℤ`, `d ∈ ℕ⁺`, `gcd(|n|,d)=1`, **denominator is always positive**, and **zero is encoded as `0/1`**. All runtime-visible results MUST be in this canonical form. Updates are mathematically associative and commutative at the semantic level.

### 4.3 Boundary Lens (Φ)

The runtime maintains **Φ descriptors** per class (or shared):

```
struct PhiDesc {
  base_ptr: u64;        // base of tile set
  stride_x: i64;        // address delta for x++
  stride_y: i64;        // address delta for y++
  swizzle_id: u16;      // optional layout code (e.g., morton, brick)
  window_x: u8;         // optional [x0,x1)
  window_y: u16;        // optional [y0,y1)
  class_offset: u64;    // per-class offset if shared base
}
```

Φ **MUST** be a pure function of its inputs during a kernel’s lifetime.

---

## 5. Execution Model

### 5.1 Kernels

A **kernel** is a statically compiled unit with the following attributes and entry ABI:

- **Attributes (subset)**:

  - `uses_boundary: bool` — indicates kernel will call `BOUND.MAP`/Φ.
  - `mirror_safe: bool` — kernel is invariant under `c ↔ mirror(c)`.
  - `unity_neutral: bool` — kernel’s update is neutral on unity set.
  - `phase_window: [p_min, p_max]` — legal phase(s) for launch.
  - `neighbor_stride: u8` — maximum neighbor hops per step.

- **Inputs**: bound buffers, Atlas tables, Φ descriptors, constants.
- **Side effects**: memory writes, `R[c]` updates.

### 5.2 Bursts and Phases

Hosts submit **bursts**: ordered packs of kernels tagged with phase predicates. The runtime executes kernels in **non‑preemptible phase epochs**:

1. Evaluate phase eligibility (current `phase` ∈ kernel’s `phase_window`).
2. Validate structural constraints (boundary windows, neighbor legality, mirror/unity contracts).
3. Issue kernel to the selected backend queue.
4. On completion, apply `R[c]` reductions and optionally advance `phase = (phase+Δ) mod 768`.

Backends **MUST** guarantee that within one epoch, each kernel observes **program order** for its own memory accesses. Cross‑kernel visibility is phase‑barriered.

### 5.3 Neighbor‑Constrained Traversal

Kernels that traverse classes **MUST** obtain next classes via the neighbor table (`NBR.GET(c,i)`/equivalent). Runtime validation **MUST** reject or fault if a kernel’s observed traversals violate the adjacency relation (when validation is enabled).

### 5.4 Determinism

Given identical inputs, Φ, and structural tables, execution is **deterministic** up to mathematically benign non‑determinism of associative reductions into `R[]`.

---

## 6. Memory Model

- **Coherence:** A backend **MUST** provide coherent reads/writes per buffer within a kernel. Coherence across kernels is defined by phase barriers.
- **Atomicity:** Updates to `R[c]` are logically atomic per class at epoch end.
- **Ordering:** Within a kernel, program order; across kernels, **happens‑before** is established only via phase advance or explicit host synchronization.
- **Aliasing:** Host MUST declare aliasing constraints for buffers bound through Φ to enable safe vectorization/fusion.

### 6.1 Resonance Arithmetic (Canonical — Single Semantics)

This section is **normative**.

- **Domain:** Resonance values live in exact rationals `ℚ` encoded as canonical fractions `n/d`.
- **Canonical Form:** `d > 0`, `gcd(|n|, d) = 1`, sign on numerator; `0` MUST be `0/1`.
- **Operation `RES_ACCUM(c, a/b)`:** Adds an input rational to accumulator `R[c]`:

  1. Let `R[c] = n/d`. Compute `n' = n·b + a·d`, `d' = d·b`.
  2. Compute `g = gcd(|n'|, |d'|)` (with `g ≥ 1`).
  3. Set `n'' = n'/g`, `d'' = d'/g`. If `d'' < 0`, set `n'' = -n''`, `d'' = -d''`.
  4. Write back `R[c] = n''/d''`.

- **Determinism:** Any association order inside an epoch MUST yield the same canonical result as sequential application of the above rule (i.e., exact rational addition). Tree reductions are permitted if they are algebraically exact.
- **Validity:** Inputs MUST satisfy `b ≠ 0`. Denominator `0` is a runtime error.
- **Observability:** Any read of `R[c]` by host or kernel observes the canonical form.

**Implementation note (non-normative):** How exactness is achieved is an implementation detail; conformance requires that the observed results are identical to the above rules.

**Overflow Handling:** When intermediate numerators/denominators exceed the 64-bit representable range, the reference runtime SHALL emit a structured warning (`tracing::warn`) identifying the affected class and operands, then return `AtlasError::ResonanceOverflow` without mutating `R[c]`. Backends MAY choose an equivalent telemetry mechanism provided the overflow is surfaced to the host.

---

## 7. Host API (language‑neutral sketch)

### 7.1 Devices, Contexts, Queues

```
Device dev = atlasEnumerateDevices()[k];
Context ctx = atlasCreateContext(dev, ContextDesc{ validation=true, profiling=true });
Queue q = atlasCreateQueue(ctx, QueueDesc{ kind=Compute });
```

### 7.2 Atlas Space Construction

```
AtlasSpace sp = atlasCreateSpace(ctx, SpaceDesc{ classes=96 });
atlasInstallNeighbors(sp, neighbors_table);
atlasInstallMirror(sp, mirror_table);
atlasInstallUnity(sp, unity_blob);
atlasSetPhase(sp, 0);
```

### 7.3 Memory & Φ

```
Buffer A = atlasAlloc(ctx, bytes, MemDesc{ usage=Tensor, alignment=64 });
Buffer B = atlasAlloc(ctx, bytes, MemDesc{ usage=Tensor, alignment=64 });
Buffer C = atlasAlloc(ctx, bytes, MemDesc{ usage=Tensor, alignment=64 });

PhiDesc phi = { base_ptr=A.base, stride_x=sx, stride_y=sy, swizzle_id=0 };
atlasBindPhi(sp, /*class=*/ALL, phi);
```

Per‑class overrides:

```
atlasBindPhiClass(sp, c, PhiDesc{ base_ptr=…, class_offset=… });
```

### 7.4 Kernels and Resonance

```
Kernel k = atlasCreateKernel(ctx, KernelDesc{
  entry="vec_add",
  attributes={ uses_boundary=false, mirror_safe=true, unity_neutral=true, phase_window=[0,767] },
  params={ A, B, C, N }
});

Burst burst = atlasBeginBurst(sp);
atlasEnqueue(burst, q, k, LaunchDesc{ grid=G, block=Bk });
atlasEndBurst(burst, AdvancePhase{ delta=1 });
```

**Resonance API:**

```
// Read canonical resonance value for class c
AtlasRatioBin r = atlasReadResonance(sp, /*class=*/c);

// Add a canonical rational to class c (host‑side enqueue of RES_ACCUM)
atlasResonanceAccumulate(sp, /*class=*/c, /*value=*/AtlasRatioBin{…});
```

### 7.5 Synchronization & Events

```
Event e = atlasSignal(q);
atlasWait(e);
```

Errors are returned as status codes and enriched with validation traces when enabled.

## 8. Validation & Safety Validation & Safety

- **Boundary Windows:** If a kernel declares `uses_boundary`, the runtime **MUST** enforce `(x,y)` within the active windows `(window_x, window_y)`.
- **Neighbor Legality:** If a kernel performs class traversal, the runtime **MUST** validate that each hop adheres to the 1‑skeleton.
- **Mirror Safety:** For `mirror_safe` kernels, transformations `c ↔ mirror(c)` **MUST NOT** change externally visible results.
- **Unity Neutrality:** For `unity_neutral` kernels, operations applied at unity points **MUST** be identity.
- **Phase Windows:** Kernels launched outside their `phase_window` **MUST** be rejected.

Validation **MAY** be disabled for performance; conformance suites MUST be run with validation enabled.

---

## 9. Backend Portability Profiles

### 9.1 Analog In‑Memory Engine (AIME)

- **Physical Layout:** Implementations **SHOULD** store class regions as physical 48×256 tiles per class to make Φ affine or identity.
- **Execution:** The backend serves loads/stores by address resolve followed by analog operation; phase epochs map to hardware cycles.
- **Consistency:** Tile‑local coherence; epoch commit for inter‑tile visibility.

### 9.2 CPU Backend

- **Φ Lowering:** Inline address arithmetic; prefer SoA, aligned 16/32‑byte vectors.
- **Parallelism:** SPMD over grid; reductions into `R[]` via tree reductions per class.
- **Validation:** Full software validation enabled in debug; sampling in release.

### 9.3 GPU Backend

- **Φ Lowering:** Row‑major or brick swizzle to maximize coalescing for `(x,y)` walks.
- **Synchronization:** Phase epochs align to kernel boundaries; `R[]` reduced via cooperative groups or atomics.
- **Tables:** Mirror/neighbor tables in constant memory.

---

## 10. Performance Guidelines

- **Exploit Φ:** Choose Φ so `(x,y)` traversals are contiguous (48 fast‑varying, 256 slow‑varying) unless backend favors alternate order.
- **Minimize Cross‑Class Hops:** Restrict neighbor traversal depth; prefer compute‑near‑data within a class.
- **Phase Locality:** Schedule bursts so dependent kernels share phases to reduce revalidation and table fetches.
- **R[] Usage:** Accumulate in registers/shared memory; emit one reduction per class.

---

## 11. Introspection, Telemetry, and Debugging

- **Counters:** Per‑epoch `loads/stores`, `Φ hits`, `neighbor hops`, `mirror traversals`, `unity elisions`, `L1/L2 hit ratios` (if available), `R[]` deltas.
- **Traces:** Optional structured traces of `(class, x, y, addr)` mappings and neighbor edges taken.
- **Validation Reports:** First failing hop, address, and class triplets; mirror/unity violations; out‑of‑window accesses.

---

## 12. Conformance Suite (Outline)

1. **Φ Correctness**: Identity, affine, Morton, brick swizzle mappings.
2. **Boundary Windows**: Out‑of‑range rejection and clamping errors.
3. **Neighbor Legality**: Random walks; adversarial traversals.
4. **Mirror/Unity**: Invariance and neutrality checks.
5. **Phase Control**: Window respect; epoch advance rules.
6. **R[] Reductions**: Associativity tolerance; determinism across backends.

---

## 13. Examples

### 13.1 Vector Add (no boundary lens)

```c
// attributes: uses_boundary=false, mirror_safe=true, unity_neutral=true
kernel vec_add(float* A, float* B, float* C, int N) {
  int i = global_id();
  if (i < N) C[i] = A[i] + B[i];
}
```

### 13.2 Boundary‑Tiled Stencil (Φ enabled)

```c
// attributes: uses_boundary=true, mirror_safe=false, unity_neutral=true
kernel stencil(Buffer in, Buffer out, PhiDesc phi, uint8 c) {
  int x = thread_x();
  int y = thread_y();
  if (!in_window(x,y, phi)) return; // runtime validates as well
  ptr center = BOUND_MAP(phi, c, x, y);
  ptr east   = BOUND_MAP(phi, c, x+1, y);
  ptr west   = BOUND_MAP(phi, c, x-1, y);
  *addr(out,c,x,y) = f(*center, *east, *west);
}
```

### 13.3 Neighbor Walk (legal hops only)

```c
// traverse k neighbors from class c0 within phase epoch
uint8 c = c0;
for (int s=0; s<k; ++s) {
  uint8 n = neighbors[c][select(s)]; // runtime supplies table
  // runtime validation ensures (c -> n) is legal
  c = n;
}
```

---

## 14. ABI Notes

### 14.1 Pointer & Endianness

- **Pointer Size:** 64‑bit; little‑endian.
- **Φ Descriptor Layout:** Packed, 8‑byte alignment.

### 14.2 Structural Tables

- **Table Encodings:** `neighbors` as CSR‑like packed list; `mirror` as 96‑byte array; `unity` as blob with schema ID.

### 14.3 `AtlasRatio` Canonical Binary Encoding

The host/runtime boundary uses a single canonical, variable‑precision encoding for exact rationals.

```
// Little‑endian container; big‑endian magnitudes for cross‑platform stability
struct AtlasRatioBin {
  u16 n_len;   // numerator magnitude length in bytes (may be 0 if n=0)
  u16 d_len;   // denominator magnitude length in bytes (>= 1)
  i8  n_sign;  // -1, 0, +1 (0 only when n=0); denominator is always positive
  u8  n_mag[n_len]; // big‑endian magnitude bytes of |n|
  u8  d_mag[d_len]; // big‑endian magnitude bytes of  d  (d ≥ 1)
}
```

**Canonicality requirements:**

- `(n_len == 0) ⇔ (n_sign == 0)` and in that case `d_mag` MUST encode `1`.
- No leading zero bytes in `n_mag` or `d_mag` unless the value is zero (which is only allowed for numerator).

### 14.4 Host Helpers

- `atlasEncodeRatio(int128 n, int128 d) -> AtlasRatioBin` (validates and reduces).
- `atlasDecodeRatio(AtlasRatioBin) -> (BigInt n, BigInt d)`.

---

## 15. Error Model

- `ATLAS_ERR_PHASE_WINDOW`: Launch outside allowed phase window.
- `ATLAS_ERR_BOUNDARY_VIOLATION`: (x,y) outside boundary window.
- `ATLAS_ERR_NEIGHBOR_ILLEGAL`: Traversal edge not in 1‑skeleton.
- `ATLAS_ERR_ALIASING`: Declared non‑aliasing violated.
- `ATLAS_ERR_RES_ZERO_DEN`: Attempt to form a ratio with denominator 0.
- `ATLAS_ERR_RES_NONCANONICAL`: Non‑canonical ratio provided at a host boundary (e.g., denominator negative, unreduced, or leading zeros).
- `ATLAS_ERR_BACKEND`: Backend‑specific failure.

All errors include an optional **trace id** and **first‑fault record**.

---

## 16. Security Considerations

- Structural tables are treated as **trusted inputs**; hosts MUST validate their provenance.
- Memory isolation **MUST** be enforced between contexts; Φ must not allow cross‑context address escape.
- Validation **SHOULD** be enabled in multi‑tenant deployments.

---

## 17. Glossary

- **AIME:** Analog In‑Memory Engine.
- **Φ (Phi):** Boundary lens address mapping.
- **Epoch:** A phase‑bounded execution slice.
- **Burst:** A host‑submitted group of kernels executed within one or more epochs.

---
