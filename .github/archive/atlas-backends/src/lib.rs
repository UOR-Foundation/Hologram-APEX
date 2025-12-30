//! # Atlas Backends - Execution Layer for Atlas ISA
//!
//! This crate provides backend implementations that execute Atlas ISA programs
//! on physical hardware substrates (CPU, GPU, quantum, analog).
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ hologram-core (High-Level Operations)                       │
//! │  - Tensor operations, neural networks, linear algebra       │
//! └────────────────────┬────────────────────────────────────────┘
//!                      │ Compiles to ISA Programs
//!                      ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │ atlas-isa (Instruction Set Architecture)                    │
//! │  - 55 instructions across 8 categories                      │
//! │  - Type system: i8/i16/i32/i64, u8/u16/u32/u64, f16/f32/f64 │
//! └────────────────────┬────────────────────────────────────────┘
//!                      │ Programs = Vec<Instruction>
//!                      ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │ atlas-backends (This Crate) - Execution Engine              │
//! │  - CPUBackend: Cache-resident, SIMD-optimized               │
//! │  - RegisterFile: 256 registers + 16 predicates              │
//! │  - Pipeline: Validate → Execute → Synchronize               │
//! └────────────────────┬────────────────────────────────────────┘
//!                      │ State Management
//!                      ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │ atlas-runtime (AtlasSpace State)                            │
//! │  - 96 resonance classes (C96 topology)                      │
//! │  - Phase counter (mod 768)                                  │
//! │  - Boundary pool (1.18 MB cache-resident)                   │
//! └─────────────────────────────────────────────────────────────┘
//! ```

// Phase 8/9 scaffolding: operation_api feature will be added in future phases
#![allow(unexpected_cfgs)]
//!
//! ## Canonical Execution (Zero Data Movement)
//!
//! The `canonical` module implements graph-based execution that eliminates data movement:
//! - Operations are graph edge traversals
//! - Execution happens directly on class_bases[96]
//! - Seven generators define all operations
//! - Graph structure from atlas-embeddings validates operations
//!
//! ## Key Principle: Topology-Aware Execution
//!
//! Unlike traditional compute backends, Atlas backends are **topology-aware**.
//! They leverage the mathematical structure of Atlas to optimize performance:
//!
//! - **96 Resonance Classes (C96):** Data organized by resonance class for cache locality
//! - **Φ-Addressing:** 2D coordinate system (48 pages × 256 bytes) for boundary memory
//! - **Phase Coordination:** Temporal alignment (mod 768) for burst scheduling
//! - **Mirror Pairs:** Involutive topology structure for symmetric operations
//! - **1-Skeleton Neighbors:** Graph connectivity for diffusion operations
//!
//! ## Execution Model
//!
//! ### Program-Based Execution
//!
//! Atlas backends execute **programs** of ISA instructions, not individual operations.
//! This enables:
//!
//! - **Instruction-level optimization:** Reordering, fusion, dead code elimination
//! - **Register allocation:** Efficient use of 256 registers
//! - **Control flow:** Branches, loops, function calls within programs
//! - **Type safety:** Validation before execution
//!
//! ### Pipeline Stages
//!
//! 1. **Validate:** Type safety, register bounds, instruction compliance
//! 2. **Label Resolution:** Build jump targets for control flow
//! 3. **Execute:** Sequential instruction dispatch through register file
//! 4. **Synchronize:** Write back cache state to AtlasSpace
//!
//! ### Memory Model
//!
//! - **Boundary Pool (Cache-Resident):** 1,179,648 bytes (96 × 48 × 256)
//!   - Locked in L2 cache (mlock on Linux/macOS, VirtualLock on Windows)
//!   - 64-byte cache line alignment
//!   - Accessed via Φ-coordinates (class, page, byte)
//!
//! - **Linear Pool (RAM-Resident):** Unlimited heap allocation
//!   - Standard malloc-based allocation
//!   - Used for large buffers that don't fit in boundary pool
//!   - Accessed via linear offsets
//!
//! ## Performance Characteristics
//!
//! ### CPUBackend Optimizations
//!
//! - **SIMD Execution:** AVX-512, AVX2, NEON support with scalar fallback
//! - **Cache Optimization:** Class-based prefetching, 64-byte alignment
//! - **Exact Arithmetic:** Rational type for resonance (no floating-point error)
//! - **Register File:** 256 typed registers with zero-cost access
//!
//! ### Benchmark Results
//!
//! | Operation | Throughput | Latency |
//! |-----------|------------|---------|
//! | Register Read/Write | 15 ns | - |
//! | Arithmetic (ADD/MUL) | 250M ops/sec | 4 ns |
//! | SIMD Vector (AVX-512) | 2 GB/s | - |
//! | Memory Fence | - | 10 ns |
//!
//! (Measured on Intel i9-12900K, Ubuntu 22.04)
//!
//! ## Quick Start Example
//!
//! ```rust,ignore
//! use atlas_backends::{CPUBackend, AtlasBackend, MemoryPool, BufferTopology, ExecutionContext};
//! use atlas_isa::{Instruction, Register, Type, Program, Address};
//! use atlas_runtime::AtlasSpace;
//!
//! // 1. Initialize backend with Atlas space
//! let mut space = AtlasSpace::new();
//! let mut backend = CPUBackend::new()?;
//! backend.initialize(&space)?;
//!
//! // 2. Allocate buffer (cache-resident for performance)
//! let topology = BufferTopology {
//!     active_classes: vec![0, 1, 2],  // Data uses classes 0, 1, 2
//!     pool: MemoryPool::Boundary,     // L2-resident (1.18 MB pool)
//!     size: 1024,                     // 1 KB
//! };
//! let buffer = backend.allocate(topology)?;
//!
//! // 3. Build a simple program: load, double, store
//! let program: Program = vec![
//!     // Load f32 from buffer[0] into register R0
//!     Instruction::LDG {
//!         ty: Type::F32,
//!         dst: Register::new(0),
//!         addr: Address::Direct { handle: buffer, offset: 0 },
//!     },
//!     // Double it: R1 = R0 + R0
//!     Instruction::ADD {
//!         ty: Type::F32,
//!         dst: Register::new(1),
//!         src1: Register::new(0),
//!         src2: Register::new(0),
//!     },
//!     // Store R1 back to buffer[4]
//!     Instruction::STG {
//!         ty: Type::F32,
//!         src: Register::new(1),
//!         addr: Address::Direct { handle: buffer, offset: 4 },
//!     },
//!     // Exit program
//!     Instruction::EXIT,
//! ];
//!
//! // 4. Execute program with execution context
//! let topology_tables = backend.topology()?;
//! let ctx = ExecutionContext::new(&topology_tables)
//!     .with_phase(0)
//!     .with_active_classes(vec![0, 1, 2]);
//! backend.execute_program(&program, &ctx)?;
//!
//! // 5. Synchronize and read results
//! backend.synchronize(&mut space)?;
//! let result_bytes = backend.read_buffer_bytes(buffer)?;
//! ```
//!
//! ## Advanced Examples
//!
//! ### Control Flow: Conditional Execution
//!
//! ```rust,ignore
//! use atlas_isa::{Instruction, Register, Predicate, Condition, Label};
//!
//! let program: Program = vec![
//!     // Compare R0 > R1, store result in P0
//!     Instruction::SETcc {
//!         ty: Type::F32,
//!         cond: Condition::GT,
//!         dst: Predicate::new(0),
//!         src1: Register::new(0),
//!         src2: Register::new(1),
//!     },
//!     // Conditional branch: if P0 goto label "greater"
//!     Instruction::BRA {
//!         target: Label("greater".into()),
//!         pred: Some(Predicate::new(0)),
//!     },
//!     // Else path: R2 = R1
//!     Instruction::MOV {
//!         ty: Type::F32,
//!         dst: Register::new(2),
//!         src: Register::new(1),
//!     },
//!     Instruction::EXIT,
//!     // greater: R2 = R0
//!     // Note: Labels are resolved automatically during validation
//! ];
//! ```
//!
//! ### Atlas-Specific: Topology Operations
//!
//! ```rust,ignore
//! use atlas_isa::Instruction;
//!
//! let program: Program = vec![
//!     // Get current resonance class into R0
//!     Instruction::ClsGet { dst: Register::new(0) },
//!
//!     // Get mirror of class in R0, store in R1
//!     Instruction::MIRROR {
//!         dst: Register::new(1),
//!         src: Register::new(0),
//!     },
//!
//!     // Get neighbor count for class in R0, store in R2
//!     Instruction::NbrCount {
//!         class: Register::new(0),
//!         dst: Register::new(2),
//!     },
//!
//!     // Accumulate resonance: R[class in R0] += value in R3
//!     Instruction::ResAccum {
//!         class: Register::new(0),
//!         value: Register::new(3),
//!     },
//!
//!     Instruction::EXIT,
//! ];
//! ```
//!
//! ### Reductions: Parallel Sum
//!
//! ```rust,ignore
//! use atlas_isa::Instruction;
//!
//! let program: Program = vec![
//!     // Sum 100 values from R0..R99 into R100
//!     Instruction::ReduceAdd {
//!         ty: Type::F32,
//!         dst: Register::new(100),
//!         src_base: Register::new(0),
//!         count: 100,
//!     },
//!     Instruction::EXIT,
//! ];
//! ```
//!
//! ## Usage Patterns
//!
//! ### Pattern 1: Batch Processing
//!
//! Process multiple data elements in parallel using SIMD:
//!
//! ```rust,ignore
//! // Load multiple values into consecutive registers
//! for i in 0..16 {
//!     program.push(Instruction::LDG {
//!         ty: Type::F32,
//!         dst: Register::new(i),
//!         addr: Address::Direct { handle: buffer, offset: i * 4 },
//!     });
//! }
//!
//! // Process them (backend will use SIMD where possible)
//! for i in 0..16 {
//!     program.push(Instruction::ADD {
//!         ty: Type::F32,
//!         dst: Register::new(i + 16),
//!         src1: Register::new(i),
//!         src2: Register::new(i),
//!     });
//! }
//! ```
//!
//! ### Pattern 2: Register Reuse
//!
//! Minimize register usage for large programs:
//!
//! ```rust,ignore
//! // Working set: R0-R3 for temporaries, R10-R20 for results
//! let temp_regs = 0..4;
//! let result_regs = 10..21;
//!
//! for (i, result_reg) in result_regs.enumerate() {
//!     let t0 = temp_regs.start;
//!     let t1 = temp_regs.start + 1;
//!
//!     // Load inputs
//!     program.push(Instruction::LDG { /* ... */ dst: Register::new(t0) });
//!     program.push(Instruction::LDG { /* ... */ dst: Register::new(t1) });
//!
//!     // Compute
//!     program.push(Instruction::ADD {
//!         ty: Type::F32,
//!         dst: Register::new(result_reg),
//!         src1: Register::new(t0),
//!         src2: Register::new(t1),
//!     });
//!     // Temporaries can be reused in next iteration
//! }
//! ```
//!
//! ### Pattern 3: Memory Pool Selection
//!
//! Choose appropriate pool based on data size and access pattern:
//!
//! ```rust,ignore
//! // Small, frequently accessed data → Boundary pool (cache-resident)
//! let hot_data_topology = BufferTopology {
//!     active_classes: vec![0],
//!     pool: MemoryPool::Boundary,  // ≤ 1.18 MB total across all allocations
//!     size: 4096,
//! };
//!
//! // Large, infrequently accessed data → Linear pool (RAM)
//! let cold_data_topology = BufferTopology {
//!     active_classes: vec![0],
//!     pool: MemoryPool::Linear,  // Unlimited size
//!     size: 100_000_000,  // 100 MB
//! };
//! ```
//!
//! ## See Also
//!
//! - [`atlas_isa`]: ISA instruction definitions and types
//! - [`atlas_runtime`]: AtlasSpace state management
//! - [`CPUBackend`]: CPU-specific backend implementation
//! - [`RegisterFile`]: Register file with type safety
//! - [SPEC.md]: Normative specification for backend implementers

use atlas_isa::Program;
use atlas_runtime::AtlasSpace;

pub mod arch;
pub mod canonical;
pub mod class_ops;
pub mod cpu;
pub mod error;
pub mod platform;
pub mod register_file;
pub mod topology;
pub mod types;

pub use error::{BackendError, Result};
pub use register_file::{RegisterFile, RegisterType};
pub use types::{
    BackendHandle, BufferTopology, ExecutionContext, MemoryPool, Rational, TopologyTables, RESONANCE_CLASS_COUNT,
};

#[cfg(feature = "cpu")]
pub use cpu::CPUBackend;

/// Backend abstraction for executing Atlas operations
///
/// All backends (CPU, GPU, quantum, analog) implement this trait to provide
/// topology-aware execution of Atlas primitives.
pub trait AtlasBackend: Send + Sync {
    /// Initialize backend with Atlas space
    ///
    /// For CPU backends, this:
    /// - Allocates cache-resident boundary pool (1.18 MB)
    /// - Builds topology lookup tables (mirrors, neighbors)
    /// - Initializes resonance accumulator state
    /// - Prepares phase counter
    ///
    /// # Arguments
    ///
    /// * `space` - Reference to AtlasSpace for initialization
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful initialization
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Cache pinning fails (insufficient privileges)
    /// - Memory allocation fails
    /// - Hardware initialization fails
    fn initialize(&mut self, space: &AtlasSpace) -> Result<()>;

    /// Allocate buffer with topology awareness
    ///
    /// Unlike traditional allocation, this uses the data's topological
    /// structure to optimize placement:
    /// - Which resonance classes are active
    /// - Φ-encoded coordinates
    /// - Phase affinity for temporal locality
    /// - Memory pool (cache-resident boundary vs RAM-resident linear)
    ///
    /// # Arguments
    ///
    /// * `topology` - Description of buffer's Atlas topology
    ///
    /// # Returns
    ///
    /// Backend-specific handle to allocated buffer
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let topology = BufferTopology {
    ///     active_classes: vec![0, 5, 12],
    ///     pool: MemoryPool::Boundary,  // L2-resident
    ///     ..Default::default()
    /// };
    /// let handle = backend.allocate(topology)?;
    /// ```
    fn allocate(&mut self, topology: BufferTopology) -> Result<BackendHandle>;

    /// Execute a program of ISA instructions
    ///
    /// This is the core execution method for Atlas backends. The backend:
    /// - Validates the program (type safety, instruction compliance)
    /// - Activates required classes into L1 (CPU) or equivalent
    /// - Executes instructions sequentially through the register file
    /// - Coordinates across phase boundaries
    /// - Maintains topology invariants (mirrors, neighbors)
    /// - Updates resonance accumulators
    ///
    /// # ISA Compliance
    ///
    /// Backends MUST support all instructions from Atlas ISA §7:
    /// - Data Movement (§7.1): LDG, STG, LDS, STS, MOV, CVT
    /// - Arithmetic (§7.2): ADD, SUB, MUL, DIV, MAD, FMA, MIN, MAX, ABS, NEG
    /// - Logic (§7.3): AND, OR, XOR, NOT, SHL, SHR, SETcc, SEL
    /// - Control Flow (§7.4): BRA, CALL, RET, LOOP, EXIT
    /// - Synchronization (§7.5): BAR.SYNC, MEM.FENCE
    /// - Atlas-Specific (§7.6): CLS.GET, MIRROR, UNITY.TEST, NBR.*, RES.ACCUM, PHASE.*, BOUND.MAP
    /// - Reductions (§7.7): REDUCE.{ADD,MIN,MAX,MUL}
    /// - Transcendentals (§7.8): EXP, LOG, SQRT, SIN, COS, TANH, etc.
    ///
    /// # Arguments
    ///
    /// * `program` - Sequence of ISA instructions to execute
    /// * `context` - Execution context (phase, active classes, topology)
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful execution
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Program validation fails
    /// - Type mismatch in register operations
    /// - Invalid memory access
    /// - Unsupported instruction (non-compliant backend)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use atlas_isa::{Instruction, Register, Type, Program};
    ///
    /// let program: Program = vec![
    ///     Instruction::LDG { ty: Type::F32, dst: Register(0), addr: /* ... */ },
    ///     Instruction::ADD { ty: Type::F32, dst: Register(1), src1: Register(0), src2: Register(0) },
    ///     Instruction::STG { ty: Type::F32, src: Register(1), addr: /* ... */ },
    /// ];
    ///
    /// let topology_tables = backend.topology()?;
    /// let mut ctx = ExecutionContext::new(&topology_tables);
    /// ctx.phase = 42;
    /// ctx.active_classes = vec![0, 1];
    /// backend.execute_program(&program, &ctx)?;
    /// ```
    fn execute_program(&mut self, program: &Program, context: &ExecutionContext<'_>) -> Result<()>;

    /// Synchronize backend state to AtlasSpace
    ///
    /// Writes back:
    /// - L1/L2 cache state (CPU)
    /// - GPU device memory (GPU)
    /// - Quantum measurement outcomes (Quantum)
    /// - Analog state readings (Analog)
    ///
    /// Should be called:
    /// - At burst boundaries
    /// - Before reading results from AtlasSpace
    /// - Before backend shutdown
    fn synchronize(&mut self, space: &mut AtlasSpace) -> Result<()>;

    /// Release backend resources
    ///
    /// Cleans up:
    /// - Cache-locked memory
    /// - GPU device allocations
    /// - Quantum circuit connections
    /// - Analog hardware state
    ///
    /// Should synchronize before shutdown.
    fn shutdown(&mut self) -> Result<()>;

    /// Get current phase
    fn current_phase(&self) -> u16;

    /// Advance phase counter by delta (mod 768)
    ///
    /// Updates backend's internal phase counter.
    /// Should be called after synchronize() to ensure state is flushed.
    ///
    /// # Arguments
    ///
    /// * `delta` - Amount to advance phase by
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// backend.synchronize(&mut space)?;
    /// backend.advance_phase(1); // Advance by 1
    /// assert_eq!(backend.current_phase(), (old_phase + 1) % 768);
    /// ```
    fn advance_phase(&mut self, delta: u16);

    /// Get backend name for debugging
    fn name(&self) -> &'static str;

    /// Get current resonance accumulators
    ///
    /// Returns the exact rational resonance values for all 96 classes.
    /// Used by hologram-core Executor to query state during execution.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let resonance = backend.resonance();
    /// assert_eq!(resonance[0], Rational::zero());
    /// ```
    fn resonance(&self) -> &[Rational; RESONANCE_CLASS_COUNT];

    /// Get topology tables (mirrors and neighbors)
    ///
    /// Returns reference to the backend's topology tables.
    /// Used by hologram-core Executor for mirror() and neighbors() queries.
    ///
    /// # Returns
    ///
    /// Reference to topology tables if backend is initialized
    ///
    /// # Errors
    ///
    /// Returns error if backend not initialized
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let topology = backend.topology()?;
    /// let mirror_of_class_5 = topology.mirrors()[5];
    /// ```
    fn topology(&self) -> Result<&TopologyTables>;

    /// Write data to buffer
    ///
    /// Copies data from host slice to backend buffer memory.
    /// For CPU backends, this is a direct memcpy.
    /// For GPU backends, this would be a device upload.
    ///
    /// # Arguments
    ///
    /// * `handle` - Backend handle to write to
    /// * `data` - Source data as raw bytes
    ///
    /// # Errors
    ///
    /// Returns error if handle is invalid or size mismatch
    fn write_buffer_bytes(&mut self, handle: BackendHandle, data: &[u8]) -> Result<()>;

    /// Read data from buffer
    ///
    /// Copies data from backend buffer memory to host vector.
    /// For CPU backends, this is a direct memcpy.
    /// For GPU backends, this would be a device download.
    ///
    /// # Arguments
    ///
    /// * `handle` - Backend handle to read from
    ///
    /// # Returns
    ///
    /// Vector containing copy of buffer data as raw bytes
    ///
    /// # Errors
    ///
    /// Returns error if handle is invalid
    fn read_buffer_bytes(&self, handle: BackendHandle) -> Result<Vec<u8>>;

    // ========================================================================
    // Phase 2: Direct SIMD Operations (bypassing ISA interpreter)
    // ========================================================================

    /// Execute element-wise vector addition using SIMD fast paths.
    ///
    /// Bypasses ISA program generation and interpreter loop for maximum performance.
    /// Uses architecture-specific SIMD instructions (AVX2/AVX-512/NEON) with
    /// automatic parallelization via rayon.
    ///
    /// # Arguments
    ///
    /// * `a` - First input buffer handle
    /// * `b` - Second input buffer handle
    /// * `c` - Output buffer handle (c[i] = a[i] + b[i])
    /// * `n` - Number of f32 elements
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful execution
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Any handle is invalid
    /// - Buffer sizes are insufficient
    /// - Memory access fails
    ///
    /// # Performance
    ///
    /// Expected speedup vs interpreter: **100-1000x**
    /// - SIMD: 8-16x (AVX2/AVX-512)
    /// - Parallelism: 4-8x (rayon)
    /// - No ISA overhead: 10-20x
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// backend.vector_add_f32(handle_a, handle_b, handle_c, 1024)?;
    /// ```
    fn vector_add_f32(&mut self, a: BackendHandle, b: BackendHandle, c: BackendHandle, n: usize) -> Result<()>;

    /// Execute element-wise vector subtraction using SIMD fast paths.
    fn vector_sub_f32(&mut self, a: BackendHandle, b: BackendHandle, c: BackendHandle, n: usize) -> Result<()>;

    /// Execute element-wise vector multiplication using SIMD fast paths.
    fn vector_mul_f32(&mut self, a: BackendHandle, b: BackendHandle, c: BackendHandle, n: usize) -> Result<()>;

    /// Execute element-wise vector division using SIMD fast paths.
    fn vector_div_f32(&mut self, a: BackendHandle, b: BackendHandle, c: BackendHandle, n: usize) -> Result<()>;

    /// Execute matrix multiplication (GEMM) using optimized blocked algorithm.
    ///
    /// Computes: C[M×N] = A[M×K] × B[K×N]
    ///
    /// Uses cache-blocking and SIMD vectorization for optimal performance.
    /// This is the most critical operation for deep learning workloads.
    ///
    /// # Arguments
    ///
    /// * `a` - Matrix A buffer handle (M×K, row-major)
    /// * `b` - Matrix B buffer handle (K×N, row-major)
    /// * `c` - Matrix C buffer handle (M×N, row-major) - accumulated into
    /// * `m` - Number of rows in A and C
    /// * `k` - Number of cols in A, rows in B
    /// * `n` - Number of cols in B and C
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful execution
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Any handle is invalid
    /// - Buffer sizes are insufficient (need M*K, K*N, M*N elements)
    /// - Memory access fails
    ///
    /// # Performance
    ///
    /// Expected speedup vs interpreter: **1000-5000x**
    /// - Cache blocking: 10-20x (avoid memory bottleneck)
    /// - SIMD: 8-16x (AVX2/AVX-512)
    /// - Parallelism: 4-8x (rayon)
    /// - No ISA overhead: 10-20x
    ///
    /// Target: 10-50 GFLOPS on modern CPUs (vs 0.014 GFLOPS interpreter)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // C[32×10] = A[32×512] × B[512×10]
    /// backend.gemm_f32(handle_a, handle_b, handle_c, 32, 512, 10)?;
    /// ```
    fn gemm_f32(
        &mut self,
        a: BackendHandle,
        b: BackendHandle,
        c: BackendHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()>;

    /// Execute ReLU activation: y = max(0, x)
    ///
    /// Element-wise activation function using direct CPU operations.
    ///
    /// # Performance
    ///
    /// Expected speedup vs interpreter: **50-200x**
    /// - Rayon parallelization: 4-8x
    /// - SIMD: 8-16x
    /// - No ISA overhead: 10-20x
    ///
    /// Target: 1000-5000 Melem/s on modern CPUs
    fn relu_f32(&mut self, input: BackendHandle, output: BackendHandle, n: usize) -> Result<()>;

    /// Execute softmax activation: y_i = exp(x_i) / sum(exp(x_j))
    ///
    /// Numerically stable softmax implementation with direct CPU operations.
    ///
    /// # Performance
    ///
    /// Expected speedup vs interpreter: **20-100x**
    /// - Rayon parallelization: 4-8x
    /// - SIMD: 4-8x
    /// - No ISA overhead: 5-10x
    ///
    /// Target: 500-2000 Melem/s on modern CPUs
    fn softmax_f32(&mut self, input: BackendHandle, output: BackendHandle, n: usize) -> Result<()>;

    /// Execute element-wise reduction (sum)
    ///
    /// Computes sum of all elements with direct CPU operations.
    ///
    /// # Performance
    ///
    /// Expected speedup vs interpreter: **50-200x**
    fn reduce_sum_f32(&mut self, input: BackendHandle, output: BackendHandle, n: usize) -> Result<()>;
}

/// Backend capabilities flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BackendCapabilities {
    /// Supports cache-resident execution
    pub cache_resident: bool,

    /// Supports parallel execution
    pub parallel: bool,

    /// Supports SIMD instructions
    pub simd: bool,

    /// Supports quantum operations
    pub quantum: bool,

    /// Supports analog operations
    pub analog: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_capabilities_default() {
        let caps = BackendCapabilities::default();
        assert!(!caps.cache_resident);
        assert!(!caps.parallel);
        assert!(!caps.simd);
        assert!(!caps.quantum);
        assert!(!caps.analog);
    }
}
