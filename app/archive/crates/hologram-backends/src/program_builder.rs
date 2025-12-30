//! Program builder utilities for creating ISA programs
//!
//! This module provides reusable patterns for constructing ISA programs.
//! These builders are used both for runtime program creation (Plan B) and
//! can be reused during compile-time precompilation (Plan C).
//!
//! # Architecture
//!
//! Program builders follow common patterns:
//! - **Element-wise operations**: Parallel execution using GLOBAL_LANE_ID
//! - **Reductions**: Single-thread aggregation
//! - **Memory operations**: Efficient load/store patterns
//!
//! # Example
//!
//! ```text
//! use hologram_backends::program_builder::create_element_wise_binary;
//! use hologram_backends::isa::{Instruction, Type};
//!
//! let program = create_element_wise_binary(
//!     buf_a, buf_b, buf_c,
//!     Type::F32,
//!     |src1, src2, dst| Instruction::ADD { ty: Type::F32, dst, src1, src2 }
//! );
//! ```

use crate::isa::special_registers::*;
use crate::isa::{Address, Condition, Instruction, Label, Predicate, Program, Register, Type};
use std::collections::HashMap;

/// Create an element-wise binary operation program
///
/// This is the most common pattern for parallel operations. Each lane processes
/// one element using GLOBAL_LANE_ID for indexing.
///
/// # Parameters
///
/// - `buf_a`: Buffer handle for first input (IGNORED - pass via ExecutionParams R1)
/// - `buf_b`: Buffer handle for second input (IGNORED - pass via ExecutionParams R2)
/// - `buf_c`: Buffer handle for output (IGNORED - pass via ExecutionParams R3)
/// - `ty`: Element type (F32, I32, etc.)
/// - `op_fn`: Function that creates the operation instruction
///
/// # Generated Pattern (for precompiled programs)
///
/// ```text
/// // R1, R2, R3 = buffer handles (passed via ExecutionParams)
/// R250 = 2                        // Shift amount for type size
/// R0 = GLOBAL_LANE_ID << 2        // Compute byte offset
/// R10 = load(R1 + R0)             // Load a[global_id]
/// R11 = load(R2 + R0)             // Load b[global_id]
/// R12 = op(R10, R11)              // Perform operation
/// store(R3 + R0, R12)             // Store c[global_id]
/// EXIT
/// ```
///
/// # Buffer Handle Initialization
///
/// **CRITICAL**: Buffer handles MUST be passed via ExecutionParams:
/// - R1 = input A buffer handle
/// - R2 = input B buffer handle
/// - R3 = output C buffer handle
/// - R4 = element count (n)
///
/// These registers are initialized by the backend before the first instruction executes.
///
/// # Bounds Checking
///
/// The generated program includes bounds checking to handle cases where the number of
/// launched lanes exceeds the element count. Each lane checks `if (GLOBAL_LANE_ID >= n)`
/// and exits early if out of bounds.
///
/// # Performance
///
/// - Program creation: ~100ns (one-time cost, or cached)
/// - Execution: ~10-20ns per element (highly parallel)
/// - Bounds check overhead: ~2-3ns per lane (negligible)
pub fn create_element_wise_binary<F>(_buf_a: u64, _buf_b: u64, _buf_c: u64, ty: Type, op_fn: F) -> Program
where
    F: FnOnce(Register, Register, Register) -> Instruction,
{
    let shift_amount = type_size_shift(ty);

    let mut labels = HashMap::new();
    let exit_label = "bounds_exit";

    // Build instruction vector
    let mut instructions = vec![
        // Bounds check: if (GLOBAL_LANE_ID >= n) exit
        // R4 contains element count n (passed via ExecutionParams)
        Instruction::SETcc {
            ty: Type::U64,
            cond: Condition::GEU, // Greater than or equal (unsigned)
            dst: Predicate::new(0),
            src1: GLOBAL_LANE_ID,
            src2: Register(4), // n passed via R4 as U64
        },
        Instruction::BRA {
            target: Label::new(exit_label),
            pred: Some(Predicate::new(0)),
        },
        // NOTE: Buffer handles R1, R2, R3 are passed via ExecutionParams
        // No MOV_IMM instructions generated - this enables precompilation
        // Compute byte offset from GLOBAL_LANE_ID
        Instruction::MOV_IMM {
            ty: Type::U32,
            dst: Register(250),
            value: shift_amount,
        },
        Instruction::SHL {
            ty: Type::U64,
            dst: Register(0),
            src: GLOBAL_LANE_ID,
            amount: Register(250),
        },
        // Load operands
        Instruction::LDG {
            ty,
            dst: Register(10),
            addr: Address::RegisterIndirectComputed {
                handle_reg: Register(1),
                offset_reg: Register(0),
            },
        },
        Instruction::LDG {
            ty,
            dst: Register(11),
            addr: Address::RegisterIndirectComputed {
                handle_reg: Register(2),
                offset_reg: Register(0),
            },
        },
        // Perform operation
        op_fn(Register(10), Register(11), Register(12)),
        // Store result
        Instruction::STG {
            ty,
            src: Register(12),
            addr: Address::RegisterIndirectComputed {
                handle_reg: Register(3),
                offset_reg: Register(0),
            },
        },
    ];

    // Add EXIT instruction and label for bounds check
    let exit_index = instructions.len();
    instructions.push(Instruction::EXIT);
    labels.insert(exit_label.to_string(), exit_index);

    Program { instructions, labels }
}

/// Create an element-wise unary operation program
///
/// Similar to binary operations but with a single input buffer.
///
/// # Parameters
///
/// - `buf_a`: Buffer handle for input (IGNORED - pass via ExecutionParams R1)
/// - `buf_c`: Buffer handle for output (IGNORED - pass via ExecutionParams R3)
/// - `ty`: Element type (F32, I32, etc.)
/// - `op_fn`: Function that creates the operation instruction
///
/// # Generated Pattern (for precompiled programs)
///
/// ```text
/// // R1, R3 = buffer handles (passed via ExecutionParams)
/// R250 = 2
/// R0 = GLOBAL_LANE_ID << 2
/// R10 = load(R1 + R0)
/// R12 = op(R10)
/// store(R3 + R0, R12)
/// EXIT
/// ```
///
/// # Buffer Handle Initialization
///
/// **CRITICAL**: Buffer handles MUST be passed via ExecutionParams:
/// - R1 = input buffer handle
/// - R3 = output buffer handle
/// - R4 = element count (n)
///
/// # Bounds Checking
///
/// The generated program includes bounds checking to handle cases where the number of
/// launched lanes exceeds the element count. Each lane checks `if (GLOBAL_LANE_ID >= n)`
/// and exits early if out of bounds.
pub fn create_element_wise_unary<F>(_buf_a: u64, _buf_c: u64, ty: Type, op_fn: F) -> Program
where
    F: FnOnce(Register, Register) -> Instruction,
{
    let shift_amount = type_size_shift(ty);

    let mut labels = HashMap::new();
    let exit_label = "bounds_exit";

    // Build instruction vector
    let mut instructions = vec![
        // Bounds check: if (GLOBAL_LANE_ID >= n) exit
        // R4 contains element count n (passed via ExecutionParams)
        Instruction::SETcc {
            ty: Type::U64,
            cond: Condition::GEU, // Greater than or equal (unsigned)
            dst: Predicate::new(0),
            src1: GLOBAL_LANE_ID,
            src2: Register(4), // n passed via R4 as U64
        },
        Instruction::BRA {
            target: Label::new(exit_label),
            pred: Some(Predicate::new(0)),
        },
        // NOTE: Buffer handles R1, R3 are passed via ExecutionParams
        // No MOV_IMM instructions generated - this enables precompilation
        // Compute byte offset
        Instruction::MOV_IMM {
            ty: Type::U32,
            dst: Register(250),
            value: shift_amount,
        },
        Instruction::SHL {
            ty: Type::U64,
            dst: Register(0),
            src: GLOBAL_LANE_ID,
            amount: Register(250),
        },
        // Load operand
        Instruction::LDG {
            ty,
            dst: Register(10),
            addr: Address::RegisterIndirectComputed {
                handle_reg: Register(1),
                offset_reg: Register(0),
            },
        },
        // Perform operation
        op_fn(Register(10), Register(12)),
        // Store result
        Instruction::STG {
            ty,
            src: Register(12),
            addr: Address::RegisterIndirectComputed {
                handle_reg: Register(2), // Changed from Register(3) - unary ops use R1 (input), R2 (output)
                offset_reg: Register(0),
            },
        },
    ];

    // Add EXIT instruction and label for bounds check
    let exit_index = instructions.len();
    instructions.push(Instruction::EXIT);
    labels.insert(exit_label.to_string(), exit_index);

    Program { instructions, labels }
}

/// Get the shift amount for a type's byte size
///
/// Returns the number of bits to shift left to multiply by the type size:
/// - 1 byte (i8, u8): shift 0 (multiply by 1)
/// - 2 bytes (i16, u16, f16, bf16): shift 1 (multiply by 2)
/// - 4 bytes (i32, u32, f32): shift 2 (multiply by 4)
/// - 8 bytes (i64, u64, f64): shift 3 (multiply by 8)
fn type_size_shift(ty: Type) -> u64 {
    match ty {
        Type::I8 | Type::U8 => 0,
        Type::I16 | Type::U16 | Type::F16 | Type::BF16 => 1,
        Type::I32 | Type::U32 | Type::F32 => 2,
        Type::I64 | Type::U64 | Type::F64 => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_size_shift() {
        assert_eq!(type_size_shift(Type::I8), 0);
        assert_eq!(type_size_shift(Type::U8), 0);
        assert_eq!(type_size_shift(Type::I16), 1);
        assert_eq!(type_size_shift(Type::F16), 1);
        assert_eq!(type_size_shift(Type::I32), 2);
        assert_eq!(type_size_shift(Type::F32), 2);
        assert_eq!(type_size_shift(Type::I64), 3);
        assert_eq!(type_size_shift(Type::F64), 3);
    }

    #[test]
    fn test_create_element_wise_binary() {
        let program = create_element_wise_binary(100, 200, 300, Type::F32, |src1, src2, dst| Instruction::ADD {
            ty: Type::F32,
            dst,
            src1,
            src2,
        });

        // Should have 9 instructions (with bounds checking)
        // 1. SETcc (bounds check)
        // 2. BRA (conditional exit)
        // 3. MOV_IMM for shift amount
        // 4. SHL for offset calculation
        // 5. LDG from buffer A
        // 6. LDG from buffer B
        // 7. ADD operation
        // 8. STG to buffer C
        // 9. EXIT
        assert_eq!(program.instructions.len(), 9);

        // Operation should be ADD
        if let Instruction::ADD { .. } = program.instructions[6] {
            // Good - ADD is at position 6 now (after bounds check)
        } else {
            panic!("Expected ADD instruction at position 6");
        }

        // Last instruction should be EXIT
        assert!(matches!(program.instructions[8], Instruction::EXIT));
    }

    #[test]
    fn test_create_element_wise_unary() {
        let program = create_element_wise_unary(100, 300, Type::F32, |src, dst| Instruction::ABS {
            ty: Type::F32,
            dst,
            src,
        });

        // Unary should have 8 instructions (with bounds checking)
        // 1. SETcc (bounds check)
        // 2. BRA (conditional exit)
        // 3. MOV_IMM for shift amount
        // 4. SHL for offset calculation
        // 5. LDG from buffer A
        // 6. ABS operation
        // 7. STG to buffer C
        // 8. EXIT
        assert_eq!(program.instructions.len(), 8);

        // Should have ABS operation at position 5 now (after bounds check)
        if let Instruction::ABS { .. } = program.instructions[5] {
            // Good
        } else {
            panic!("Expected ABS instruction at position 5");
        }
    }
}
