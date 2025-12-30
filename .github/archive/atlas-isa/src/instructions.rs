//! Atlas ISA Instruction Set
//!
//! This module defines the complete Atlas ISA instruction set as specified in §7 of the
//! Atlas ISA specification. All backends implementing `AtlasBackend` MUST support the
//! complete instruction set defined here.
//!
//! # Architecture
//!
//! - **Register-based**: 256 typed registers (0-255)
//! - **Predicate registers**: 16 boolean predicates (0-15)
//! - **Typed operations**: All operations specify their type explicitly
//! - **Type safety**: CVT instruction for explicit type conversions
//!
//! # Instruction Categories
//!
//! - **Data Movement** (§7.1): LDG, STG, LDS, STS, MOV, CVT
//! - **Arithmetic** (§7.2): ADD, SUB, MUL, DIV, MAD, FMA, MIN, MAX, ABS, NEG
//! - **Logic** (§7.3): AND, OR, XOR, NOT, SHL, SHR, SETcc, SEL
//! - **Control Flow** (§7.4): BRA, CALL, RET, LOOP, EXIT
//! - **Synchronization** (§7.5): BAR_SYNC, MEM_FENCE
//! - **Atlas-Specific** (§7.6): CLS_GET, MIRROR, UNITY_TEST, NBR_*, RES_ACCUM, PHASE_*, BOUND_MAP
//! - **Reductions** (§7.7): REDUCE_ADD, REDUCE_MIN, REDUCE_MAX, REDUCE_MUL
//! - **Transcendentals** (§7.8): EXP, LOG, SQRT, SIN, COS, TANH, etc.

use std::fmt;

// ================================================================================================
// Core Types
// ================================================================================================

/// Register identifier (0-255)
///
/// Represents one of 256 general-purpose registers in the Atlas execution model.
/// Registers are typed, with type tracking enforced at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Register(pub u8);

impl Register {
    /// Create a new register
    pub const fn new(index: u8) -> Self {
        Register(index)
    }

    /// Get register index
    pub const fn index(self) -> u8 {
        self.0
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}

/// Predicate register identifier (0-15)
///
/// Represents one of 16 predicate (boolean) registers used for conditional execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Predicate(pub u8);

impl Predicate {
    /// Create a new predicate register
    pub const fn new(index: u8) -> Self {
        assert!(index < 16, "Predicate index must be < 16");
        Predicate(index)
    }

    /// Get predicate index
    pub const fn index(self) -> u8 {
        self.0
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "p{}", self.0)
    }
}

/// Label for control flow targets
///
/// Labels mark positions in the instruction stream for branches, calls, and loops.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Label(pub String);

impl Label {
    /// Create a new label
    pub fn new(name: impl Into<String>) -> Self {
        Label(name.into())
    }

    /// Get label name
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{}", self.0)
    }
}

impl From<&str> for Label {
    fn from(s: &str) -> Self {
        Label(s.to_string())
    }
}

impl From<String> for Label {
    fn from(s: String) -> Self {
        Label(s)
    }
}

/// Type of data operated on by instructions
///
/// Matches the type system from Atlas ISA §3. All instructions that operate on
/// typed data must specify the type explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    // Signed integers
    I8,
    I16,
    I32,
    I64,

    // Unsigned integers
    U8,
    U16,
    U32,
    U64,

    // Floating point
    F16,  // IEEE 754 half precision
    BF16, // Brain floating point 16
    F32,  // IEEE 754 single precision
    F64,  // IEEE 754 double precision
}

impl Type {
    /// Size of this type in bytes
    pub const fn size_bytes(self) -> usize {
        match self {
            Type::I8 | Type::U8 => 1,
            Type::I16 | Type::U16 | Type::F16 | Type::BF16 => 2,
            Type::I32 | Type::U32 | Type::F32 => 4,
            Type::I64 | Type::U64 | Type::F64 => 8,
        }
    }

    /// Is this an integer type?
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::U8 | Type::U16 | Type::U32 | Type::U64
        )
    }

    /// Is this a floating-point type?
    pub const fn is_float(self) -> bool {
        matches!(self, Type::F16 | Type::BF16 | Type::F32 | Type::F64)
    }

    /// Is this a signed type?
    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::F16 | Type::BF16 | Type::F32 | Type::F64
        )
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::I8 => write!(f, "i8"),
            Type::I16 => write!(f, "i16"),
            Type::I32 => write!(f, "i32"),
            Type::I64 => write!(f, "i64"),
            Type::U8 => write!(f, "u8"),
            Type::U16 => write!(f, "u16"),
            Type::U32 => write!(f, "u32"),
            Type::U64 => write!(f, "u64"),
            Type::F16 => write!(f, "f16"),
            Type::BF16 => write!(f, "bf16"),
            Type::F32 => write!(f, "f32"),
            Type::F64 => write!(f, "f64"),
        }
    }
}

/// Condition code for comparisons
///
/// Used by SETcc instruction to set predicate registers based on comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Condition {
    EQ,  // Equal
    NE,  // Not equal
    LT,  // Less than
    LE,  // Less than or equal
    GT,  // Greater than
    GE,  // Greater than or equal
    LTU, // Less than (unsigned)
    LEU, // Less than or equal (unsigned)
    GTU, // Greater than (unsigned)
    GEU, // Greater than or equal (unsigned)
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Condition::EQ => write!(f, "eq"),
            Condition::NE => write!(f, "ne"),
            Condition::LT => write!(f, "lt"),
            Condition::LE => write!(f, "le"),
            Condition::GT => write!(f, "gt"),
            Condition::GE => write!(f, "ge"),
            Condition::LTU => write!(f, "ltu"),
            Condition::LEU => write!(f, "leu"),
            Condition::GTU => write!(f, "gtu"),
            Condition::GEU => write!(f, "geu"),
        }
    }
}

/// Memory fence scope
///
/// Defines the scope of memory ordering enforcement for MEM_FENCE instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryScope {
    Thread, // Single thread (no-op on most architectures)
    Block,  // Thread block/workgroup
    Device, // Entire device (GPU)
    System, // Across all devices (CPU + GPU)
}

impl fmt::Display for MemoryScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryScope::Thread => write!(f, "thread"),
            MemoryScope::Block => write!(f, "block"),
            MemoryScope::Device => write!(f, "device"),
            MemoryScope::System => write!(f, "system"),
        }
    }
}

/// Memory address
///
/// Represents a memory location that can be loaded from or stored to.
/// Supports multiple addressing modes:
/// - Buffer handle + offset (linear addressing)
/// - Φ-coordinates (boundary pool addressing: class, page, byte)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Address {
    /// Linear buffer addressing: handle + byte offset
    BufferOffset { handle: u64, offset: usize },

    /// Boundary pool Φ-coordinate addressing
    PhiCoordinate {
        class: u8, // Resonance class (0-95)
        page: u8,  // Page within class (0-47)
        byte: u8,  // Byte within page (0-255)
    },

    /// Register-indirect addressing (address in register)
    RegisterIndirect { base: Register, offset: i32 },
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Address::BufferOffset { handle, offset } => {
                write!(f, "[buf{} + {}]", handle, offset)
            }
            Address::PhiCoordinate { class, page, byte } => {
                write!(f, "[Φ({}, {}, {})]", class, page, byte)
            }
            Address::RegisterIndirect { base, offset } => {
                if *offset >= 0 {
                    write!(f, "[{} + {}]", base, offset)
                } else {
                    write!(f, "[{} - {}]", base, -offset)
                }
            }
        }
    }
}

// ================================================================================================
// Instruction Enum
// ================================================================================================

/// Complete Atlas ISA instruction set
///
/// Every instruction type from §7 of the Atlas ISA specification.
/// Backends MUST implement all instruction types for compliance.
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    // ============================================================================================
    // Data Movement (§7.1)
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

    /// Convert between types
    CVT {
        src_ty: Type,
        dst_ty: Type,
        dst: Register,
        src: Register,
    },

    // ============================================================================================
    // Arithmetic (§7.2)
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
    // Logic (§7.3)
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
    // Control Flow (§7.4)
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
    // Synchronization (§7.5)
    // ============================================================================================
    /// Barrier synchronization
    BarSync { id: u8 },

    /// Memory fence
    MemFence { scope: MemoryScope },

    // ============================================================================================
    // Atlas-Specific (§7.6)
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
    // Reductions (§7.7)
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
    // Transcendentals (§7.8)
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
            Instruction::SIN { ty, dst, src } => write!(f, "sin.{} {}, {}", ty, dst, src),
            Instruction::COS { ty, dst, src } => write!(f, "cos.{} {}, {}", ty, dst, src),
            Instruction::TAN { ty, dst, src } => write!(f, "tan.{} {}, {}", ty, dst, src),
            Instruction::TANH { ty, dst, src } => write!(f, "tanh.{} {}, {}", ty, dst, src),
            Instruction::SIGMOID { ty, dst, src } => write!(f, "sigmoid.{} {}, {}", ty, dst, src),
        }
    }
}

// ================================================================================================
// Program Error Type
// ================================================================================================

/// Errors that can occur when building or manipulating programs
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum ProgramError {
    /// Duplicate label definition
    #[error("duplicate label '{0}' at instruction {1}")]
    DuplicateLabel(String, usize),

    /// Undefined label referenced
    #[error("undefined label '{0}' referenced in instruction {1}")]
    UndefinedLabel(String, usize),

    /// Label points to invalid instruction index
    #[error("label '{0}' points to invalid instruction index {1} (max: {2})")]
    InvalidLabelTarget(String, usize, usize),
}

pub type ProgramResult<T> = std::result::Result<T, ProgramError>;

// ================================================================================================
// Program Type
// ================================================================================================

/// A program is a sequence of instructions with label metadata for control flow
///
/// Programs are executed by backends implementing `AtlasBackend::execute_program()`.
/// Labels mark jump targets for BRA, CALL, and LOOP instructions.
///
/// # Example
///
/// ```
/// use atlas_isa::{Program, Instruction, Register, Type, Label};
///
/// let mut program = Program::new();
///
/// // Add instructions
/// program.instructions.push(Instruction::MOV {
///     ty: Type::I32,
///     dst: Register::new(0),
///     src: Register::new(1),
/// });
///
/// // Add a label
/// program.add_label("loop_start").unwrap();
///
/// program.instructions.push(Instruction::BRA {
///     target: Label::new("loop_start"),
///     pred: None,
/// });
///
/// // Resolve label to instruction index
/// assert_eq!(program.resolve_label("loop_start"), Some(1));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// Instruction sequence
    pub instructions: Vec<Instruction>,

    /// Label definitions (name → instruction index)
    ///
    /// Labels mark jump targets for control flow instructions.
    /// Instruction indices are 0-based positions in the instructions vector.
    pub labels: std::collections::HashMap<String, usize>,
}

impl Program {
    /// Create a new empty program
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            labels: std::collections::HashMap::new(),
        }
    }

    /// Create a program from a sequence of instructions (no labels)
    pub fn from_instructions(instructions: Vec<Instruction>) -> Self {
        Self {
            instructions,
            labels: std::collections::HashMap::new(),
        }
    }

    /// Create a program with preallocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            instructions: Vec::with_capacity(capacity),
            labels: std::collections::HashMap::new(),
        }
    }

    /// Add a label at the current instruction position
    ///
    /// Labels mark the position for control flow instructions (BRA, CALL, LOOP).
    /// The label will point to the next instruction that will be added.
    ///
    /// # Errors
    ///
    /// Returns `ProgramError::DuplicateLabel` if a label with the same name already exists.
    ///
    /// # Example
    ///
    /// ```
    /// use atlas_isa::Program;
    ///
    /// let mut program = Program::new();
    /// program.add_label("start").unwrap();
    /// assert_eq!(program.resolve_label("start"), Some(0));
    /// ```
    pub fn add_label(&mut self, name: impl Into<String>) -> ProgramResult<()> {
        let name = name.into();
        let idx = self.instructions.len();

        if self.labels.contains_key(&name) {
            return Err(ProgramError::DuplicateLabel(name, idx));
        }

        self.labels.insert(name, idx);
        Ok(())
    }

    /// Resolve a label to its instruction index
    ///
    /// Returns `None` if the label is not defined.
    ///
    /// # Example
    ///
    /// ```
    /// use atlas_isa::Program;
    ///
    /// let mut program = Program::new();
    /// program.add_label("loop_start").unwrap();
    /// assert_eq!(program.resolve_label("loop_start"), Some(0));
    /// assert_eq!(program.resolve_label("undefined"), None);
    /// ```
    pub fn resolve_label(&self, name: &str) -> Option<usize> {
        self.labels.get(name).copied()
    }

    /// Check if a label exists
    pub fn has_label(&self, name: &str) -> bool {
        self.labels.contains_key(name)
    }

    /// Get the number of instructions in the program
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Check if the program is empty
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

// Compatibility helpers for migration from Vec<Instruction>
impl From<Vec<Instruction>> for Program {
    fn from(instructions: Vec<Instruction>) -> Self {
        Self::from_instructions(instructions)
    }
}

impl From<Program> for Vec<Instruction> {
    fn from(program: Program) -> Self {
        program.instructions
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
            | Instruction::SIN { .. }
            | Instruction::COS { .. }
            | Instruction::TAN { .. }
            | Instruction::TANH { .. }
            | Instruction::SIGMOID { .. } => InstructionCategory::Transcendental,
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
            Instruction::LDG { .. } | Instruction::STG { .. } | Instruction::LDS { .. } | Instruction::STS { .. }
        )
    }
}

/// Instruction category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionCategory {
    DataMovement,
    Arithmetic,
    Logic,
    ControlFlow,
    Synchronization,
    AtlasSpecific,
    Reduction,
    Transcendental,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_creation() {
        let r0 = Register::new(0);
        let r255 = Register::new(255);
        assert_eq!(r0.index(), 0);
        assert_eq!(r255.index(), 255);
    }

    #[test]
    fn test_type_properties() {
        assert_eq!(Type::F32.size_bytes(), 4);
        assert_eq!(Type::I64.size_bytes(), 8);
        assert!(Type::F32.is_float());
        assert!(Type::I32.is_integer());
        assert!(Type::I32.is_signed());
        assert!(!Type::U32.is_signed());
    }

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
    }

    // ============================================================================
    // Program Tests
    // ============================================================================

    #[test]
    fn test_program_new() {
        let program = Program::new();
        assert_eq!(program.len(), 0);
        assert!(program.is_empty());
        assert_eq!(program.labels.len(), 0);
    }

    #[test]
    fn test_program_from_instructions() {
        let instructions = vec![
            Instruction::MOV {
                ty: Type::I32,
                dst: Register::new(0),
                src: Register::new(1),
            },
            Instruction::EXIT,
        ];

        let program = Program::from_instructions(instructions);
        assert_eq!(program.len(), 2);
        assert_eq!(program.labels.len(), 0);
    }

    #[test]
    fn test_program_with_capacity() {
        let program = Program::with_capacity(10);
        assert_eq!(program.len(), 0);
        assert!(program.instructions.capacity() >= 10);
    }

    #[test]
    fn test_program_add_label() {
        let mut program = Program::new();

        // Add label at position 0
        program.add_label("start").unwrap();
        assert_eq!(program.resolve_label("start"), Some(0));
        assert!(program.has_label("start"));

        // Add instruction
        program.instructions.push(Instruction::EXIT);

        // Add label at position 1
        program.add_label("end").unwrap();
        assert_eq!(program.resolve_label("end"), Some(1));
    }

    #[test]
    fn test_program_duplicate_label() {
        let mut program = Program::new();

        program.add_label("loop").unwrap();

        // Try to add duplicate label
        let result = program.add_label("loop");
        assert!(result.is_err());

        match result {
            Err(ProgramError::DuplicateLabel(name, _idx)) => {
                assert_eq!(name, "loop");
            }
            _ => panic!("Expected DuplicateLabel error"),
        }
    }

    #[test]
    fn test_program_resolve_undefined_label() {
        let program = Program::new();
        assert_eq!(program.resolve_label("undefined"), None);
        assert!(!program.has_label("undefined"));
    }

    #[test]
    fn test_program_from_vec_instruction() {
        let instructions = vec![Instruction::MOV {
            ty: Type::I32,
            dst: Register::new(0),
            src: Register::new(1),
        }];

        let program: Program = instructions.into();
        assert_eq!(program.len(), 1);
    }

    #[test]
    fn test_program_to_vec_instruction() {
        let mut program = Program::new();
        program.instructions.push(Instruction::EXIT);

        let instructions: Vec<Instruction> = program.into();
        assert_eq!(instructions.len(), 1);
    }

    #[test]
    fn test_program_default() {
        let program: Program = Default::default();
        assert_eq!(program.len(), 0);
    }

    #[test]
    fn test_program_with_control_flow() {
        let mut program = Program::new();

        // loop_start:
        program.add_label("loop_start").unwrap();

        // MOV r0, r1
        program.instructions.push(Instruction::MOV {
            ty: Type::I32,
            dst: Register::new(0),
            src: Register::new(1),
        });

        // BRA loop_start
        program.instructions.push(Instruction::BRA {
            target: Label::new("loop_start"),
            pred: None,
        });

        assert_eq!(program.len(), 2);
        assert_eq!(program.resolve_label("loop_start"), Some(0));
    }

    #[test]
    fn test_program_multiple_labels() {
        let mut program = Program::new();

        program.add_label("label1").unwrap();
        program.instructions.push(Instruction::EXIT);

        program.add_label("label2").unwrap();
        program.instructions.push(Instruction::EXIT);

        program.add_label("label3").unwrap();
        program.instructions.push(Instruction::EXIT);

        assert_eq!(program.resolve_label("label1"), Some(0));
        assert_eq!(program.resolve_label("label2"), Some(1));
        assert_eq!(program.resolve_label("label3"), Some(2));
        assert_eq!(program.labels.len(), 3);
    }

    // ============================================================================
    // ProgramError Tests
    // ============================================================================

    #[test]
    fn test_program_error_display() {
        let err = ProgramError::DuplicateLabel("test".to_string(), 5);
        assert_eq!(err.to_string(), "duplicate label 'test' at instruction 5");

        let err = ProgramError::UndefinedLabel("missing".to_string(), 10);
        assert_eq!(
            err.to_string(),
            "undefined label 'missing' referenced in instruction 10"
        );

        let err = ProgramError::InvalidLabelTarget("bad".to_string(), 100, 50);
        assert_eq!(
            err.to_string(),
            "label 'bad' points to invalid instruction index 100 (max: 50)"
        );
    }

    #[test]
    fn test_program_error_equality() {
        let err1 = ProgramError::DuplicateLabel("test".to_string(), 5);
        let err2 = ProgramError::DuplicateLabel("test".to_string(), 5);
        let err3 = ProgramError::DuplicateLabel("other".to_string(), 5);

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }
}
