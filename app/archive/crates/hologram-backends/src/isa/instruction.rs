//! Atlas ISA Instruction Set
//!
//! Complete instruction set including:
//! - Data movement (LDG, STG, LDS, STS, MOV, CVT)
//! - Arithmetic (ADD, SUB, MUL, DIV, FMA, MIN, MAX, ABS, NEG)
//! - Logic (AND, OR, XOR, NOT, SHL, SHR, SETcc, SEL)
//! - Control flow (BRA, CALL, RET, LOOP, EXIT)
//! - Synchronization (BarSync, MemFence)
//! - Atlas-specific (ClsGet, MIRROR, UnityTest, NBR*, ResAccum, Phase*, BoundMap)
//! - Reductions (ReduceAdd, ReduceMin, ReduceMax, ReduceMul)
//! - Transcendentals (EXP, LOG, SQRT, SIN, COS, TANH, SIGMOID)
//! - Pool storage (PoolAlloc, PoolFree, PoolLoad, PoolStore)
//! - Higher-order operations (ParallelMap)

use super::types::{Address, Condition, Label, MemoryScope, Predicate, Register, Type};
use std::fmt;

/// Map operation for parallel_map_unary intrinsic (Phase 2B)
///
/// Specifies the unary operation to apply element-wise across an array.
/// Backends can optimize these with SIMD (CPU) or parallel kernels (GPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MapOperation {
    // Unary math
    Abs,
    Neg,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Ceil,
    Floor,
    Round,
    Reciprocal,
    Erf,

    // Activations
    Sigmoid,
    Tanh,
    Relu,
}

impl fmt::Display for MapOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MapOperation::Abs => write!(f, "abs"),
            MapOperation::Neg => write!(f, "neg"),
            MapOperation::Sqrt => write!(f, "sqrt"),
            MapOperation::Exp => write!(f, "exp"),
            MapOperation::Log => write!(f, "log"),
            MapOperation::Sin => write!(f, "sin"),
            MapOperation::Cos => write!(f, "cos"),
            MapOperation::Tan => write!(f, "tan"),
            MapOperation::Ceil => write!(f, "ceil"),
            MapOperation::Floor => write!(f, "floor"),
            MapOperation::Round => write!(f, "round"),
            MapOperation::Reciprocal => write!(f, "reciprocal"),
            MapOperation::Erf => write!(f, "erf"),
            MapOperation::Sigmoid => write!(f, "sigmoid"),
            MapOperation::Tanh => write!(f, "tanh"),
            MapOperation::Relu => write!(f, "relu"),
        }
    }
}

/// Binary map operations (element-wise operations on two arrays)
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BinaryMapOperation {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    // Comparison/Logic
    Min,
    Max,
    Atan2,
}

impl fmt::Display for BinaryMapOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryMapOperation::Add => write!(f, "add"),
            BinaryMapOperation::Sub => write!(f, "sub"),
            BinaryMapOperation::Mul => write!(f, "mul"),
            BinaryMapOperation::Div => write!(f, "div"),
            BinaryMapOperation::Pow => write!(f, "pow"),
            BinaryMapOperation::Min => write!(f, "min"),
            BinaryMapOperation::Max => write!(f, "max"),
            BinaryMapOperation::Atan2 => write!(f, "atan2"),
        }
    }
}

/// Reduce operations (reduce array to single value)
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReduceOperation {
    Sum,
    Product,
    Min,
    Max,
}

impl fmt::Display for ReduceOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReduceOperation::Sum => write!(f, "sum"),
            ReduceOperation::Product => write!(f, "product"),
            ReduceOperation::Min => write!(f, "min"),
            ReduceOperation::Max => write!(f, "max"),
        }
    }
}

/// Complete Atlas ISA instruction set
///
/// Every instruction type from the Atlas ISA specification plus pool storage extensions.
/// Backends MUST implement all instruction types for compliance.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Instruction {
    // ============================================================================================
    // Data Movement
    // ============================================================================================
    /// Load from global memory to register
    LDG { ty: Type, dst: Register, addr: Address },

    /// Store from register to global memory
    STG { ty: Type, src: Register, addr: Address },

    /// Load from shared memory to register
    LDS { ty: Type, dst: Register, addr: Address },

    /// Store from register to shared memory
    STS { ty: Type, src: Register, addr: Address },

    /// Move value from one register to another
    MOV { ty: Type, dst: Register, src: Register },

    /// Move immediate value to register
    ///
    /// Loads a constant value into a register. The value is interpreted according to the type.
    /// For integer types, the value is zero-extended or sign-extended as needed.
    /// For floating-point types, the value is reinterpreted as the bit pattern.
    #[allow(non_camel_case_types)]
    MOV_IMM { ty: Type, dst: Register, value: u64 },

    /// Convert between types
    CVT {
        src_ty: Type,
        dst_ty: Type,
        dst: Register,
        src: Register,
    },

    // ============================================================================================
    // Arithmetic
    // ============================================================================================
    /// Addition: dst = src1 + src2
    ADD {
        ty: Type,
        dst: Register,
        src1: Register,
        src2: Register,
    },

    /// Subtraction: dst = src1 - src2
    SUB {
        ty: Type,
        dst: Register,
        src1: Register,
        src2: Register,
    },

    /// Multiplication: dst = src1 * src2
    MUL {
        ty: Type,
        dst: Register,
        src1: Register,
        src2: Register,
    },

    /// Division: dst = src1 / src2
    DIV {
        ty: Type,
        dst: Register,
        src1: Register,
        src2: Register,
    },

    /// Multiply-add: dst = a * b + c
    MAD {
        ty: Type,
        dst: Register,
        a: Register,
        b: Register,
        c: Register,
    },

    /// Fused multiply-add: dst = a * b + c (single rounding)
    FMA {
        ty: Type,
        dst: Register,
        a: Register,
        b: Register,
        c: Register,
    },

    /// Minimum: dst = min(src1, src2)
    MIN {
        ty: Type,
        dst: Register,
        src1: Register,
        src2: Register,
    },

    /// Maximum: dst = max(src1, src2)
    MAX {
        ty: Type,
        dst: Register,
        src1: Register,
        src2: Register,
    },

    /// Absolute value: dst = |src|
    ABS { ty: Type, dst: Register, src: Register },

    /// Negation: dst = -src
    NEG { ty: Type, dst: Register, src: Register },

    // ============================================================================================
    // Logic
    // ============================================================================================
    /// Bitwise AND: dst = src1 & src2
    AND {
        ty: Type,
        dst: Register,
        src1: Register,
        src2: Register,
    },

    /// Bitwise OR: dst = src1 | src2
    OR {
        ty: Type,
        dst: Register,
        src1: Register,
        src2: Register,
    },

    /// Bitwise XOR: dst = src1 ^ src2
    XOR {
        ty: Type,
        dst: Register,
        src1: Register,
        src2: Register,
    },

    /// Bitwise NOT: dst = ~src
    NOT { ty: Type, dst: Register, src: Register },

    /// Shift left: dst = src << amount
    SHL {
        ty: Type,
        dst: Register,
        src: Register,
        amount: Register,
    },

    /// Shift right: dst = src >> amount
    SHR {
        ty: Type,
        dst: Register,
        src: Register,
        amount: Register,
    },

    /// Set condition code: dst = (src1 cond src2)
    SETcc {
        ty: Type,
        cond: Condition,
        dst: Predicate,
        src1: Register,
        src2: Register,
    },

    /// Select based on predicate: dst = pred ? src_true : src_false
    SEL {
        ty: Type,
        dst: Register,
        pred: Predicate,
        src_true: Register,
        src_false: Register,
    },

    // ============================================================================================
    // Control Flow
    // ============================================================================================
    /// Branch to label (conditional if pred is Some)
    BRA { target: Label, pred: Option<Predicate> },

    /// Call subroutine at label
    CALL { target: Label },

    /// Return from subroutine
    RET,

    /// Loop with register count
    LOOP { count: Register, body: Label },

    /// Exit program execution
    EXIT,

    // ============================================================================================
    // Synchronization
    // ============================================================================================
    /// Barrier synchronization
    BarSync { id: u8 },

    /// Memory fence
    MemFence { scope: MemoryScope },

    // ============================================================================================
    // Atlas-Specific
    // ============================================================================================
    /// Get current resonance class
    ClsGet { dst: Register },

    /// Get mirror class: dst = mirror(src)
    MIRROR { dst: Register, src: Register },

    /// Test unity neutrality: dst = (sum(R[96]) < epsilon)
    UnityTest { dst: Predicate, epsilon: f64 },

    /// Get neighbor count for class
    NbrCount { class: Register, dst: Register },

    /// Get neighbor by index
    NbrGet { class: Register, index: u8, dst: Register },

    /// Accumulate resonance: R[class] += value
    ResAccum { class: Register, value: Register },

    /// Get current phase counter
    PhaseGet { dst: Register },

    /// Advance phase counter
    PhaseAdv { delta: u16 },

    /// Map Φ-coordinates to linear address
    BoundMap {
        class: Register,
        page: Register,
        byte: Register,
        dst: Register,
    },

    // ============================================================================================
    // Reductions
    // ============================================================================================
    /// Parallel reduction: sum
    ReduceAdd {
        ty: Type,
        dst: Register,
        src_base: Register,
        count: u32,
    },

    /// Parallel reduction: minimum
    ReduceMin {
        ty: Type,
        dst: Register,
        src_base: Register,
        count: u32,
    },

    /// Parallel reduction: maximum
    ReduceMax {
        ty: Type,
        dst: Register,
        src_base: Register,
        count: u32,
    },

    /// Parallel reduction: product
    ReduceMul {
        ty: Type,
        dst: Register,
        src_base: Register,
        count: u32,
    },

    // ============================================================================================
    // Transcendentals
    // ============================================================================================
    /// Exponential: dst = e^src
    EXP { ty: Type, dst: Register, src: Register },

    /// Natural logarithm: dst = ln(src)
    LOG { ty: Type, dst: Register, src: Register },

    /// Base-2 logarithm: dst = log2(src)
    LOG2 { ty: Type, dst: Register, src: Register },

    /// Base-10 logarithm: dst = log10(src)
    LOG10 { ty: Type, dst: Register, src: Register },

    /// Square root: dst = sqrt(src)
    SQRT { ty: Type, dst: Register, src: Register },

    /// Reciprocal square root: dst = 1/sqrt(src)
    RSQRT { ty: Type, dst: Register, src: Register },

    /// Power: dst = src1^src2
    POW {
        ty: Type,
        dst: Register,
        src1: Register,
        src2: Register,
    },

    /// Sine: dst = sin(src)
    SIN { ty: Type, dst: Register, src: Register },

    /// Cosine: dst = cos(src)
    COS { ty: Type, dst: Register, src: Register },

    /// Tangent: dst = tan(src)
    TAN { ty: Type, dst: Register, src: Register },

    /// Hyperbolic tangent: dst = tanh(src)
    TANH { ty: Type, dst: Register, src: Register },

    /// Sigmoid: dst = 1 / (1 + e^(-src))
    SIGMOID { ty: Type, dst: Register, src: Register },

    // ============================================================================================
    // Normalization Operations
    // ============================================================================================
    /// Layer Normalization: output = gamma * (input - mean) / sqrt(variance + eps) + beta
    /// Normalizes across the last dimension (hidden_dim) for each position (batch, seq)
    LayerNorm {
        ty: Type,
        input: Register,    // Input buffer [batch_size * seq_len * hidden_dim]
        gamma: Register,    // Scale parameters [hidden_dim]
        beta: Register,     // Bias parameters [hidden_dim]
        output: Register,   // Output buffer [batch_size * seq_len * hidden_dim]
        num_positions: u32, // batch_size * seq_len
        hidden_dim: u32,    // Hidden dimension size
        eps: f32,           // Epsilon for numerical stability
    },

    /// Group Normalization: output = gamma * (input - mean) / sqrt(variance + eps) + beta
    /// Normalizes across groups of channels in NCHW format
    GroupNorm {
        ty: Type,
        input: Register,   // Input buffer [N, C, H, W]
        gamma: Register,   // Scale parameters [C]
        beta: Register,    // Bias parameters [C]
        output: Register,  // Output buffer [N, C, H, W]
        batch_size: u32,   // N
        num_channels: u32, // C
        height: u32,       // H
        width: u32,        // W
        num_groups: u32,   // Number of groups
        eps: f32,          // Epsilon for numerical stability
    },

    /// Batch Normalization: output = scale * (input - mean) / sqrt(variance + eps) + bias
    /// Normalizes per channel across batch and spatial dimensions in NCHW format (inference mode)
    BatchNorm {
        ty: Type,
        input: Register,    // Input buffer [N, C, H, W]
        scale: Register,    // Scale parameters [C]
        bias: Register,     // Bias parameters [C]
        mean: Register,     // Mean parameters [C]
        variance: Register, // Variance parameters [C]
        output: Register,   // Output buffer [N, C, H, W]
        batch_size: u32,    // N
        num_channels: u32,  // C
        height: u32,        // H
        width: u32,         // W
        eps: f32,           // Epsilon for numerical stability
    },

    // ============================================================================================
    // Convolution Operations
    // ============================================================================================
    /// 2D Convolution: output = conv2d(input, weights) + bias
    /// Performs 2D convolution in NCHW format with optional bias and grouped convolution support
    /// Supports full ONNX Conv specification with per-dimension parameters
    Conv2d {
        ty: Type,
        input: Register,   // Input buffer [N, C_in, H, W]
        weights: Register, // Weight buffer [C_out, C_in/group, K_h, K_w]
        bias: Register,    // Bias buffer [C_out] (0 = no bias)
        output: Register,  // Output buffer [N, C_out, H_out, W_out]
        batch_size: u32,   // N
        in_channels: u32,  // C_in
        in_height: u32,    // H
        in_width: u32,     // W
        out_channels: u32, // C_out
        kernel_h: u32,     // K_h
        kernel_w: u32,     // K_w
        stride_h: u32,     // Vertical stride
        stride_w: u32,     // Horizontal stride
        pad_top: u32,      // Top padding
        pad_left: u32,     // Left padding
        pad_bottom: u32,   // Bottom padding
        pad_right: u32,    // Right padding
        dilation_h: u32,   // Vertical dilation
        dilation_w: u32,   // Horizontal dilation
        group: u32,        // Number of groups (1 = no grouping)
        has_bias: u32,     // 0 = no bias, 1 = has bias
    },

    // ============================================================================================
    // Spatial Operations
    // ============================================================================================
    /// Nearest Neighbor Upsampling (2D)
    /// Increases spatial resolution by repeating each pixel scale_factor times
    NearestUpsample2d {
        ty: Type,
        input: Register,   // Input buffer [N, C, H, W]
        output: Register,  // Output buffer [N, C, H*scale, W*scale]
        batch_size: u32,   // N
        num_channels: u32, // C
        in_height: u32,    // H
        in_width: u32,     // W
        scale_factor: u32, // Upsampling scale (2x, 4x, etc.)
    },

    // ============================================================================================
    // Linear Algebra Operations
    // ============================================================================================
    /// General Matrix Multiplication (GEMM): C = A * B
    /// Performs matrix multiplication using GPU-accelerated tiled computation
    ///
    /// Matrix dimensions:
    /// - A: [M, K] (row-major)
    /// - B: [K, N] (row-major)
    /// - C: [M, N] (row-major)
    Gemm {
        ty: Type,
        matrix_a: Register, // Input matrix A [M, K]
        matrix_b: Register, // Input matrix B [K, N]
        matrix_c: Register, // Output matrix C [M, N]
        m: u32,             // Rows in A and C
        k: u32,             // Cols in A, rows in B
        n: u32,             // Cols in B and C
    },

    // ============================================================================================
    // Pool Storage (NEW)
    // ============================================================================================
    /// Allocate linear pool storage
    PoolAlloc { size: u64, dst: Register },

    /// Free linear pool storage
    PoolFree { handle: Register },

    /// Load from linear pool to register
    PoolLoad {
        ty: Type,
        pool: Register,
        offset: Register,
        dst: Register,
    },

    /// Store from register to linear pool
    PoolStore {
        ty: Type,
        pool: Register,
        offset: Register,
        src: Register,
    },

    // ============================================================================================
    // Higher-Order Operations (Phase 2B)
    // ============================================================================================
    /// Parallel map: Apply unary operation to all elements
    ///
    /// Applies the specified operation element-wise to the input array.
    /// Backends optimize this with SIMD (CPU) or parallel kernels (GPU).
    ///
    /// Equivalent to: for i in 0..count { output[i] = operation(input[i]) }
    ParallelMap {
        ty: Type,
        input: Register,
        output: Register,
        operation: MapOperation,
        count: u32,
    },

    /// Parallel map binary: Apply binary operation to pairs of elements
    ///
    /// Applies the specified operation element-wise to pairs from two input arrays.
    /// Backends optimize this with SIMD (CPU) or parallel kernels (GPU).
    ///
    /// Equivalent to: for i in 0..count { output[i] = operation(input_a[i], input_b[i]) }
    ParallelMapBinary {
        ty: Type,
        input_a: Register,
        input_b: Register,
        output: Register,
        operation: BinaryMapOperation,
        count: u32,
    },

    /// Parallel reduce: Reduce array to single value
    ///
    /// Reduces all elements in the input array to a single value using the specified operation.
    /// Backends optimize this with hierarchical reduction (tree reduction on GPU,
    /// SIMD horizontal reduction on CPU).
    ///
    /// Equivalent to: output[0] = reduce(operation, input[0..count])
    ParallelReduce {
        ty: Type,
        input: Register,
        output: Register,
        operation: ReduceOperation,
        count: u32,
    },
}

// ================================================================================================
// Display Implementation
// ================================================================================================

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Data Movement
            Instruction::LDG { ty, dst, addr } => write!(f, "ldg.{} {}, {}", ty, dst, addr),
            Instruction::STG { ty, src, addr } => write!(f, "stg.{} {}, {}", ty, addr, src),
            Instruction::LDS { ty, dst, addr } => write!(f, "lds.{} {}, {}", ty, dst, addr),
            Instruction::STS { ty, src, addr } => write!(f, "sts.{} {}, {}", ty, addr, src),
            Instruction::MOV { ty, dst, src } => write!(f, "mov.{} {}, {}", ty, dst, src),
            Instruction::MOV_IMM { ty, dst, value } => write!(f, "mov_imm.{} {}, {}", ty, dst, value),
            Instruction::CVT {
                src_ty,
                dst_ty,
                dst,
                src,
            } => write!(f, "cvt.{}.{} {}, {}", dst_ty, src_ty, dst, src),

            // Arithmetic
            Instruction::ADD { ty, dst, src1, src2 } => write!(f, "add.{} {}, {}, {}", ty, dst, src1, src2),
            Instruction::SUB { ty, dst, src1, src2 } => write!(f, "sub.{} {}, {}, {}", ty, dst, src1, src2),
            Instruction::MUL { ty, dst, src1, src2 } => write!(f, "mul.{} {}, {}, {}", ty, dst, src1, src2),
            Instruction::DIV { ty, dst, src1, src2 } => write!(f, "div.{} {}, {}, {}", ty, dst, src1, src2),
            Instruction::MAD { ty, dst, a, b, c } => {
                write!(f, "mad.{} {}, {}, {}, {}", ty, dst, a, b, c)
            }
            Instruction::FMA { ty, dst, a, b, c } => {
                write!(f, "fma.{} {}, {}, {}, {}", ty, dst, a, b, c)
            }
            Instruction::MIN { ty, dst, src1, src2 } => write!(f, "min.{} {}, {}, {}", ty, dst, src1, src2),
            Instruction::MAX { ty, dst, src1, src2 } => write!(f, "max.{} {}, {}, {}", ty, dst, src1, src2),
            Instruction::ABS { ty, dst, src } => write!(f, "abs.{} {}, {}", ty, dst, src),
            Instruction::NEG { ty, dst, src } => write!(f, "neg.{} {}, {}", ty, dst, src),

            // Logic
            Instruction::AND { ty, dst, src1, src2 } => write!(f, "and.{} {}, {}, {}", ty, dst, src1, src2),
            Instruction::OR { ty, dst, src1, src2 } => write!(f, "or.{} {}, {}, {}", ty, dst, src1, src2),
            Instruction::XOR { ty, dst, src1, src2 } => write!(f, "xor.{} {}, {}, {}", ty, dst, src1, src2),
            Instruction::NOT { ty, dst, src } => write!(f, "not.{} {}, {}", ty, dst, src),
            Instruction::SHL { ty, dst, src, amount } => write!(f, "shl.{} {}, {}, {}", ty, dst, src, amount),
            Instruction::SHR { ty, dst, src, amount } => write!(f, "shr.{} {}, {}, {}", ty, dst, src, amount),
            Instruction::SETcc {
                ty,
                cond,
                dst,
                src1,
                src2,
            } => write!(f, "set{}.{} {}, {}, {}", cond, ty, dst, src1, src2),
            Instruction::SEL {
                ty,
                dst,
                pred,
                src_true,
                src_false,
            } => write!(f, "sel.{} {}, {}, {}, {}", ty, dst, pred, src_true, src_false),

            // Control Flow
            Instruction::BRA { target, pred } => {
                if let Some(p) = pred {
                    write!(f, "bra.{} {}", p, target)
                } else {
                    write!(f, "bra {}", target)
                }
            }
            Instruction::CALL { target } => write!(f, "call {}", target),
            Instruction::RET => write!(f, "ret"),
            Instruction::LOOP { count, body } => write!(f, "loop {}, {}", count, body),
            Instruction::EXIT => write!(f, "exit"),

            // Synchronization
            Instruction::BarSync { id } => write!(f, "bar.sync {}", id),
            Instruction::MemFence { scope } => write!(f, "memfence.{}", scope),

            // Atlas-Specific
            Instruction::ClsGet { dst } => write!(f, "cls.get {}", dst),
            Instruction::MIRROR { dst, src } => write!(f, "mirror {}, {}", dst, src),
            Instruction::UnityTest { dst, epsilon } => write!(f, "unity.test {}, {}", dst, epsilon),
            Instruction::NbrCount { class, dst } => write!(f, "nbr.count {}, {}", dst, class),
            Instruction::NbrGet { class, index, dst } => {
                write!(f, "nbr.get {}, {}, {}", dst, class, index)
            }
            Instruction::ResAccum { class, value } => write!(f, "res.accum {}, {}", class, value),
            Instruction::PhaseGet { dst } => write!(f, "phase.get {}", dst),
            Instruction::PhaseAdv { delta } => write!(f, "phase.adv {}", delta),
            Instruction::BoundMap { class, page, byte, dst } => {
                write!(f, "bound.map {}, {}, {}, {}", dst, class, page, byte)
            }

            // Reductions
            Instruction::ReduceAdd {
                ty,
                dst,
                src_base,
                count,
            } => write!(f, "reduce.add.{} {}, {}, {}", ty, dst, src_base, count),
            Instruction::ReduceMin {
                ty,
                dst,
                src_base,
                count,
            } => write!(f, "reduce.min.{} {}, {}, {}", ty, dst, src_base, count),
            Instruction::ReduceMax {
                ty,
                dst,
                src_base,
                count,
            } => write!(f, "reduce.max.{} {}, {}, {}", ty, dst, src_base, count),
            Instruction::ReduceMul {
                ty,
                dst,
                src_base,
                count,
            } => write!(f, "reduce.mul.{} {}, {}, {}", ty, dst, src_base, count),

            // Transcendentals
            Instruction::EXP { ty, dst, src } => write!(f, "exp.{} {}, {}", ty, dst, src),
            Instruction::LOG { ty, dst, src } => write!(f, "log.{} {}, {}", ty, dst, src),
            Instruction::LOG2 { ty, dst, src } => write!(f, "log2.{} {}, {}", ty, dst, src),
            Instruction::LOG10 { ty, dst, src } => write!(f, "log10.{} {}, {}", ty, dst, src),
            Instruction::SQRT { ty, dst, src } => write!(f, "sqrt.{} {}, {}", ty, dst, src),
            Instruction::RSQRT { ty, dst, src } => write!(f, "rsqrt.{} {}, {}", ty, dst, src),
            Instruction::POW { ty, dst, src1, src2 } => write!(f, "pow.{} {}, {}, {}", ty, dst, src1, src2),
            Instruction::SIN { ty, dst, src } => write!(f, "sin.{} {}, {}", ty, dst, src),
            Instruction::COS { ty, dst, src } => write!(f, "cos.{} {}, {}", ty, dst, src),
            Instruction::TAN { ty, dst, src } => write!(f, "tan.{} {}, {}", ty, dst, src),
            Instruction::TANH { ty, dst, src } => write!(f, "tanh.{} {}, {}", ty, dst, src),
            Instruction::SIGMOID { ty, dst, src } => write!(f, "sigmoid.{} {}, {}", ty, dst, src),

            // Normalization
            Instruction::LayerNorm {
                ty,
                input,
                gamma,
                beta,
                output,
                num_positions,
                hidden_dim,
                eps,
            } => {
                write!(
                    f,
                    "layer_norm.{} {}, {}, {}, {}, {}, {}, {}",
                    ty, output, input, gamma, beta, num_positions, hidden_dim, eps
                )
            }
            Instruction::GroupNorm {
                ty,
                input,
                gamma,
                beta,
                output,
                batch_size,
                num_channels,
                height,
                width,
                num_groups,
                eps,
            } => {
                write!(
                    f,
                    "group_norm.{} {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                    ty, output, input, gamma, beta, batch_size, num_channels, height, width, num_groups, eps
                )
            }
            Instruction::BatchNorm {
                ty,
                input,
                scale,
                bias,
                mean,
                variance,
                output,
                batch_size,
                num_channels,
                height,
                width,
                eps,
            } => {
                write!(
                    f,
                    "batch_norm.{} {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                    ty, output, input, scale, bias, mean, variance, batch_size, num_channels, height, width, eps
                )
            }

            // Convolution
            Instruction::Conv2d {
                ty,
                input,
                weights,
                bias,
                output,
                batch_size,
                in_channels,
                in_height,
                in_width,
                out_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_top,
                pad_left,
                pad_bottom,
                pad_right,
                dilation_h,
                dilation_w,
                group,
                has_bias: _,
            } => {
                write!(
                    f,
                    "conv2d.{} {}, {}, {}, {}, [{}x{}x{}x{}], [{}x{}x{}x{}], stride=[{}x{}], pad=[{},{},{},{}], dilation=[{}x{}], group={}",
                    ty,
                    output,
                    input,
                    weights,
                    bias,
                    batch_size,
                    in_channels,
                    in_height,
                    in_width,
                    out_channels,
                    in_channels,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    pad_top,
                    pad_left,
                    pad_bottom,
                    pad_right,
                    dilation_h,
                    dilation_w,
                    group
                )
            }

            // Spatial Operations
            Instruction::NearestUpsample2d {
                ty,
                input,
                output,
                batch_size,
                num_channels,
                in_height,
                in_width,
                scale_factor,
            } => {
                write!(
                    f,
                    "nearest_upsample2d.{} {}, {}, {}, {}, {}, {}, {}",
                    ty, output, input, batch_size, num_channels, in_height, in_width, scale_factor
                )
            }

            // Linear Algebra
            Instruction::Gemm {
                ty,
                matrix_a,
                matrix_b,
                matrix_c,
                m,
                k,
                n,
            } => {
                write!(
                    f,
                    "gemm.{} {}, {}, {}, {}, {}, {}",
                    ty, matrix_c, matrix_a, matrix_b, m, k, n
                )
            }

            // Pool Storage
            Instruction::PoolAlloc { size, dst } => write!(f, "pool.alloc {}, {}", dst, size),
            Instruction::PoolFree { handle } => write!(f, "pool.free {}", handle),
            Instruction::PoolLoad { ty, pool, offset, dst } => {
                write!(f, "pool.load.{} {}, {}, {}", ty, dst, pool, offset)
            }
            Instruction::PoolStore { ty, pool, offset, src } => {
                write!(f, "pool.store.{} {}, {}, {}", ty, pool, offset, src)
            }

            // Higher-Order Operations
            Instruction::ParallelMap {
                ty,
                input,
                output,
                operation,
                count,
            } => {
                write!(f, "parallel_map.{} {}, {}, {}, {}", ty, output, input, operation, count)
            }

            Instruction::ParallelMapBinary {
                ty,
                input_a,
                input_b,
                output,
                operation,
                count,
            } => {
                write!(
                    f,
                    "parallel_map_binary.{} {}, {}, {}, {}, {}",
                    ty, output, input_a, input_b, operation, count
                )
            }

            Instruction::ParallelReduce {
                ty,
                input,
                output,
                operation,
                count,
            } => {
                write!(
                    f,
                    "parallel_reduce.{} {}, {}, {}, {}",
                    ty, output, input, operation, count
                )
            }
        }
    }
}

// ================================================================================================
// Helper Functions
// ================================================================================================

impl Instruction {
    /// Get the category of this instruction
    pub fn category(&self) -> InstructionCategory {
        match self {
            Instruction::LDG { .. }
            | Instruction::STG { .. }
            | Instruction::LDS { .. }
            | Instruction::STS { .. }
            | Instruction::MOV { .. }
            | Instruction::MOV_IMM { .. }
            | Instruction::CVT { .. } => InstructionCategory::DataMovement,

            Instruction::ADD { .. }
            | Instruction::SUB { .. }
            | Instruction::MUL { .. }
            | Instruction::DIV { .. }
            | Instruction::MAD { .. }
            | Instruction::FMA { .. }
            | Instruction::MIN { .. }
            | Instruction::MAX { .. }
            | Instruction::ABS { .. }
            | Instruction::NEG { .. } => InstructionCategory::Arithmetic,

            Instruction::AND { .. }
            | Instruction::OR { .. }
            | Instruction::XOR { .. }
            | Instruction::NOT { .. }
            | Instruction::SHL { .. }
            | Instruction::SHR { .. }
            | Instruction::SETcc { .. }
            | Instruction::SEL { .. } => InstructionCategory::Logic,

            Instruction::BRA { .. }
            | Instruction::CALL { .. }
            | Instruction::RET
            | Instruction::LOOP { .. }
            | Instruction::EXIT => InstructionCategory::ControlFlow,

            Instruction::BarSync { .. } | Instruction::MemFence { .. } => InstructionCategory::Synchronization,

            Instruction::ClsGet { .. }
            | Instruction::MIRROR { .. }
            | Instruction::UnityTest { .. }
            | Instruction::NbrCount { .. }
            | Instruction::NbrGet { .. }
            | Instruction::ResAccum { .. }
            | Instruction::PhaseGet { .. }
            | Instruction::PhaseAdv { .. }
            | Instruction::BoundMap { .. } => InstructionCategory::AtlasSpecific,

            Instruction::ReduceAdd { .. }
            | Instruction::ReduceMin { .. }
            | Instruction::ReduceMax { .. }
            | Instruction::ReduceMul { .. } => InstructionCategory::Reduction,

            Instruction::EXP { .. }
            | Instruction::LOG { .. }
            | Instruction::LOG2 { .. }
            | Instruction::LOG10 { .. }
            | Instruction::SQRT { .. }
            | Instruction::RSQRT { .. }
            | Instruction::POW { .. }
            | Instruction::SIN { .. }
            | Instruction::COS { .. }
            | Instruction::TAN { .. }
            | Instruction::LayerNorm { .. }
            | Instruction::GroupNorm { .. }
            | Instruction::BatchNorm { .. }
            | Instruction::TANH { .. }
            | Instruction::SIGMOID { .. } => InstructionCategory::Transcendental,

            Instruction::Conv2d { .. } | Instruction::NearestUpsample2d { .. } => InstructionCategory::Transcendental,

            Instruction::Gemm { .. } => InstructionCategory::Arithmetic,

            Instruction::PoolAlloc { .. }
            | Instruction::PoolFree { .. }
            | Instruction::PoolLoad { .. }
            | Instruction::PoolStore { .. } => InstructionCategory::PoolStorage,

            Instruction::ParallelMap { .. }
            | Instruction::ParallelMapBinary { .. }
            | Instruction::ParallelReduce { .. } => InstructionCategory::Arithmetic,
        }
    }

    /// Does this instruction modify control flow?
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self,
            Instruction::BRA { .. }
                | Instruction::CALL { .. }
                | Instruction::RET
                | Instruction::LOOP { .. }
                | Instruction::EXIT
        )
    }

    /// Does this instruction access memory?
    pub fn is_memory_access(&self) -> bool {
        matches!(
            self,
            Instruction::LDG { .. }
                | Instruction::STG { .. }
                | Instruction::LDS { .. }
                | Instruction::STS { .. }
                | Instruction::PoolLoad { .. }
                | Instruction::PoolStore { .. }
        )
    }

    /// Does this instruction access pool storage?
    pub fn is_pool_operation(&self) -> bool {
        matches!(
            self,
            Instruction::PoolAlloc { .. }
                | Instruction::PoolFree { .. }
                | Instruction::PoolLoad { .. }
                | Instruction::PoolStore { .. }
        )
    }
}

/// Instruction category
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InstructionCategory {
    DataMovement,
    Arithmetic,
    Logic,
    ControlFlow,
    Synchronization,
    AtlasSpecific,
    Reduction,
    Transcendental,
    PoolStorage,
}

impl fmt::Display for InstructionCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstructionCategory::DataMovement => write!(f, "Data Movement"),
            InstructionCategory::Arithmetic => write!(f, "Arithmetic"),
            InstructionCategory::Logic => write!(f, "Logic"),
            InstructionCategory::ControlFlow => write!(f, "Control Flow"),
            InstructionCategory::Synchronization => write!(f, "Synchronization"),
            InstructionCategory::AtlasSpecific => write!(f, "Atlas-Specific"),
            InstructionCategory::Reduction => write!(f, "Reduction"),
            InstructionCategory::Transcendental => write!(f, "Transcendental"),
            InstructionCategory::PoolStorage => write!(f, "Pool Storage"),
        }
    }
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_display() {
        let inst = Instruction::ADD {
            ty: Type::F32,
            dst: Register(0),
            src1: Register(1),
            src2: Register(2),
        };
        assert_eq!(inst.to_string(), "add.f32 r0, r1, r2");
    }

    #[test]
    fn test_instruction_category() {
        let add = Instruction::ADD {
            ty: Type::F32,
            dst: Register(0),
            src1: Register(1),
            src2: Register(2),
        };
        assert_eq!(add.category(), InstructionCategory::Arithmetic);

        let bra = Instruction::BRA {
            target: Label::new("loop"),
            pred: None,
        };
        assert_eq!(bra.category(), InstructionCategory::ControlFlow);
        assert!(bra.is_control_flow());

        let pool_alloc = Instruction::PoolAlloc {
            size: 4096,
            dst: Register(10),
        };
        assert_eq!(pool_alloc.category(), InstructionCategory::PoolStorage);
        assert!(pool_alloc.is_pool_operation());
    }

    #[test]
    fn test_pool_instructions_display() {
        let alloc = Instruction::PoolAlloc {
            size: 4096,
            dst: Register(0),
        };
        assert_eq!(alloc.to_string(), "pool.alloc r0, 4096");

        let free = Instruction::PoolFree { handle: Register(0) };
        assert_eq!(free.to_string(), "pool.free r0");

        let load = Instruction::PoolLoad {
            ty: Type::F32,
            pool: Register(0),
            offset: Register(1),
            dst: Register(2),
        };
        assert_eq!(load.to_string(), "pool.load.f32 r2, r0, r1");

        let store = Instruction::PoolStore {
            ty: Type::F32,
            pool: Register(0),
            offset: Register(1),
            src: Register(2),
        };
        assert_eq!(store.to_string(), "pool.store.f32 r0, r1, r2");
    }

    #[test]
    fn test_memory_access_detection() {
        let ldg = Instruction::LDG {
            ty: Type::F32,
            dst: Register(0),
            addr: Address::BufferOffset { handle: 1, offset: 0 },
        };
        assert!(ldg.is_memory_access());

        let pool_load = Instruction::PoolLoad {
            ty: Type::F32,
            pool: Register(0),
            offset: Register(1),
            dst: Register(2),
        };
        assert!(pool_load.is_memory_access());
        assert!(pool_load.is_pool_operation());

        let add = Instruction::ADD {
            ty: Type::F32,
            dst: Register(0),
            src1: Register(1),
            src2: Register(2),
        };
        assert!(!add.is_memory_access());
        assert!(!add.is_pool_operation());
    }

    #[test]
    fn test_instruction_serialization() {
        // Test arithmetic instruction
        let add = Instruction::ADD {
            ty: Type::F32,
            dst: Register(0),
            src1: Register(1),
            src2: Register(2),
        };
        let bytes = bincode::serialize(&add).unwrap();
        let loaded: Instruction = bincode::deserialize(&bytes).unwrap();
        assert_eq!(loaded, add);

        // Test control flow instruction
        let bra = Instruction::BRA {
            target: Label::new("loop"),
            pred: None,
        };
        let bytes = bincode::serialize(&bra).unwrap();
        let loaded: Instruction = bincode::deserialize(&bytes).unwrap();
        assert_eq!(loaded, bra);

        // Test pool instruction
        let pool_alloc = Instruction::PoolAlloc {
            size: 4096,
            dst: Register(10),
        };
        let bytes = bincode::serialize(&pool_alloc).unwrap();
        let loaded: Instruction = bincode::deserialize(&bytes).unwrap();
        assert_eq!(loaded, pool_alloc);

        // Test transcendental
        let sin = Instruction::SIN {
            ty: Type::F32,
            dst: Register(5),
            src: Register(6),
        };
        let bytes = bincode::serialize(&sin).unwrap();
        let loaded: Instruction = bincode::deserialize(&bytes).unwrap();
        assert_eq!(loaded, sin);
    }

    #[test]
    fn test_instruction_category_serialization() {
        let category = InstructionCategory::Arithmetic;
        let bytes = bincode::serialize(&category).unwrap();
        let loaded: InstructionCategory = bincode::deserialize(&bytes).unwrap();
        assert_eq!(loaded, category);
    }
}
