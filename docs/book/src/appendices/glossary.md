# Appendix A: Glossary of Terms

## Core Concepts

**12,288 Coordinate System**
: *Global hash ring / sharded keyspace with fixed 12,288 buckets*. A consistent-hashing style ring with known ring size (48×256 matrix). Coordinates determined by `addr = SHA3-256(content) mod 12,288`, then `page = addr ÷ 256`, `byte = addr mod 256`.

**96 Equivalence Classes**
: *Semantic buckets (C₉₆) used for checksums, sharding, and parallelism*. All 256 byte values map to exactly 96 classes via modular arithmetic. Functions as a checksum family over 96 categories for database integrity invariants.

**Atlas**
: The mathematical framework that reveals information's intrinsic structure, including the 96 classes, 12,288 coordinates, and conservation laws.

**Conservation Laws**
: Four system engineering guarantees:
- **R (Resonance)**: *Database integrity invariant* - checksum family over 96 categories
- **C (Cycle)**: *Orchestration scheduler* - 768-step round-robin with service window slots
- **Φ (Phi/Holographic)**: *Encode/decode correctness* - proof-of-correct serialization
- **ℛ (Reynolds)**: *Network flow conservation* - backpressure semantics

**Content-Determined Addressing**
: *Function `addr = mod_12288(SHA3-256(encode(content)))`*. Pure function computing address from content without external catalog. Encode = canonicalize + hash + modular projection; decode = recompute and verify address from content (no reverse mapping).

**Hologram**
: The computing platform that implements alignment with information's intrinsic structure, built on the discoveries of Atlas.

## Technical Terms

**Action Minimization**
: The computational paradigm where solutions emerge as minimum-energy configurations of carefully designed energy landscapes, borrowed from physics.

**BHIC (Block Header Information Certificate)**
: Proof-carrying receipts that demonstrate conservation law compliance and operation validity.

**Budget (β)**
: A measure of computational resource consumption that must decrease or remain constant through valid operations. Truth corresponds to β=0.

**CAR (Content Addressable Archive)**
: Package format for distributing UOR-BC modules via IPFS.

**CIM (Component & Interface Model)**
: Layer 1 of the Hologram stack, defining components, ports, contracts, and proofs.

**Proof / Receipt**
: *Append-only witness that a transformation preserved conservation: `(commitment hash, class-sum deltas mod 96, budget meter, time)`*. Verifiable in O(window + |receipt|). Proof-carrying transaction that demonstrates all conservation laws maintained.

**Gauge Invariance**
: Property where different representations of the same information are recognized as equivalent, similar to gauge symmetry in physics.

**Generator Architecture (GA)**
: Execution stack that realizes computation through sector-based action minimization, emitting proof-carrying receipts.

**Holographic Property**
: The principle that boundary information completely determines bulk properties and vice versa, enabling perfect reconstruction.

**Klein Probe**
: Boolean homomorphism test across four-element windows, part of the resonance verification system.

**Natural Organization**
: The inherent structure of information that emerges from mathematical properties rather than being imposed by system design.

**Poly-Ontological Object**
: Mathematical entities that possess simultaneous, irreducible existence across multiple mathematical categories.

**Proof-Carrying State**
: System where every piece of data carries mathematical proof of its validity and transformations.

**Resonance Alphabet (R96)**
: The 96 fundamental "information colors" that all data naturally falls into, determined by resonance evaluation.

**Reynolds Number (ℛ)**
: In Hologram context, a measure of information mixing and flow patterns, analogous to fluid dynamics.

**Round-Trip Property**
: The guarantee that projecting to boundary and reconstructing bulk returns identical information at β=0.

**SMM (Standard Model of Models)**
: Layer 0 of the Hologram stack, providing minimal vocabulary for models, interfaces, morphisms, and proofs.

**Structural Entropy**
: The inherent organization present in information before any external structure is imposed.

**SyncReceipt**
: Lightweight proof exchanged between nodes to maintain consistency without full state transfer.

**Takum Numbers**
: Arithmetic encodings that collapse geometric invariants into multiplicative scalars, making generation under law a number-theoretic object.

**Bytecode (UOR-BC)**
: *Conservation-checked op sequence over C₉₆ with explicit budgets; transport-safe; runtime-verifiable*. 96-class aware module format whose opcodes are stable over the C₉₆ semiring. Transport-safe sequence with budget annotations.

**UORID (Universal Object Reference Identifier)**
: Content-determined identifier derived from mathematical properties, ensuring global uniqueness without coordination.

## Conceptual Terms

**Alignment vs. Imposition**
: The fundamental paradigm shift from imposing structure on information (traditional) to aligning with information's inherent structure (Hologram).

**Computational Physics**
: The treatment of computation as a physical process governed by conservation laws and energy minimization.

**Coordinate-Free**
: Properties or operations that remain valid regardless of representation or reference frame choice.

**Deterministic Performance**
: Predictable, mathematically-bounded performance characteristics that emerge from structural alignment.

**Emergent Consistency**
: Consistency that arises naturally from conservation laws rather than being enforced by protocols.

**Information Physics**
: The study of information as a physical phenomenon with measurable properties and conservation laws.

**Intrinsic Security**
: Security properties that emerge from mathematical structure rather than being added through cryptographic layers.

**Natural Load Distribution**
: Automatic, optimal distribution of computational load based on information's inherent structure.

**Proof-Carrying Generation**
: Computation that generates validity proofs as a natural byproduct rather than requiring separate verification.

**Schema Compilation**
: The process of transforming high-level schemas into physics-aligned computational structures.

**Structural Synchronization**
: Synchronization achieved through structural alignment rather than message passing.

**Zero-Knowledge Consistency**
: The ability to verify consistency without accessing full state, using only conservation receipts.

**Storage**
: *12,288-shard K/V layout keyed by `addr`*. Replicas subscribe via receipts; dedupe at the class layer. Place record at shard `addr`, or publish proof that projection equals `addr`.

**Transport**
: *CTP-96 frames carrying `(content, addr, class, receipt)`*. Routers forward by `addr` math (no routing tables). Endpoints MUST reject on checksum/budget failure; version-negotiated profiles.

**Database**
: *Index-free store; query=route; transactions = proof-carrying transforms that preserve R96*. Partitioned key-value space with exactly 12,288 shards where shard ID = `addr`. Isolation = class-local windows.

**Orchestration**
: *C-cycle scheduling with budgets; no brokers; fairness by construction*. Fixed-length round-robin cycle (length 768) guaranteeing fairness and bounded latency through service window slots.

**Edge**
: *Node hosting a subset of coordinates; stateless beyond receipts*. Any node that hosts some subset of the 12,288 coordinate space.

**Service Provider**
: *Operator of a coordinate slice + proof verification endpoint*. Exposes subset of 12,288 space plus proof verification APIs. SLAs expressed in conservation and verification latencies.

**Embedding**
: *Mapping from a state (or dataset) to a 96-dimensional class histogram or moment vector*. Used for indexing, search, and equivalence testing.

**Encode/Decode**
: *Φ-consistent boundary serialization ↔ bulk reconstruction with acceptance test*. Encode = canonicalize + hash + modular projection; decode = recompute and verify (no reverse mapping).

**Checksum / Hash**
: *R96 checksum (class sums) vs. SHA3-256 digest (addressing)*. R96 provides semantic deduplication (3/8 compression); SHA3-256 provides uniform distribution for addressing.

## Mathematical Notation

**T**
: The boundary torus (ℤ/48ℤ) × (ℤ/256ℤ) with 12,288 points

**Ψ**
: Boundary trace or field configuration

**S[ψ]**
: Action functional to be minimized

**Φ**
: Holographic mapping operator between boundary and bulk

**β**
: Budget parameter, with β=0 indicating truth/validity

**R(s)**
: Resonance evaluation function mapping selectors to classes

**C₇₆₈**
: The 768-element conservation cycle

**ℛ**
: Reynolds operator for information flow

**⊗**
: Poly-ontological composition operator

**CNF**
: Canonical Normal Form under gauge and schedule equivalence