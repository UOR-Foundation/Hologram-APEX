//! Pattern detection for ISA instruction sequences.
//!
//! Analyzes sequences of Atlas ISA instructions to identify high-level
//! operations that can be executed via optimized SIMD fast paths.

use atlas_isa::{Instruction, Program, Register, Type as T};

/// High-level operation patterns detected in ISA instruction streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OperationPattern {
    /// Element-wise vector addition: c[i] = a[i] + b[i]
    VectorAdd {
        /// Output register containing result addresses
        dst_reg: Register,
        /// First input register containing addresses
        a_reg: Register,
        /// Second input register containing addresses
        b_reg: Register,
        /// Number of elements
        n: usize,
    },

    /// Element-wise vector subtraction: c[i] = a[i] - b[i]
    VectorSub {
        dst_reg: Register,
        a_reg: Register,
        b_reg: Register,
        n: usize,
    },

    /// Element-wise vector multiplication: c[i] = a[i] * b[i]
    VectorMul {
        dst_reg: Register,
        a_reg: Register,
        b_reg: Register,
        n: usize,
    },

    /// Element-wise vector division: c[i] = a[i] / b[i]
    VectorDiv {
        dst_reg: Register,
        a_reg: Register,
        b_reg: Register,
        n: usize,
    },

    /// Scalar addition: c[i] = a[i] + scalar
    ScalarAdd {
        dst_reg: Register,
        input_reg: Register,
        scalar_reg: Register,
        n: usize,
    },

    /// Scalar multiplication: c[i] = a[i] * scalar
    ScalarMul {
        dst_reg: Register,
        input_reg: Register,
        scalar_reg: Register,
        n: usize,
    },

    /// Matrix multiplication: C = A × B
    MatMul {
        c_reg: Register,
        a_reg: Register,
        b_reg: Register,
        m: usize,
        k: usize,
        n: usize,
    },

    /// Pattern not recognized - fallback to interpreter
    Unknown,
}

/// Detect operation pattern in an ISA program.
///
/// Analyzes the instruction sequence to identify vectorizable operations
/// that can bypass the interpreter and use SIMD fast paths.
pub(super) fn detect_pattern(program: &Program) -> OperationPattern {
    let instructions = &program.instructions;

    // Empty or trivial programs
    if instructions.len() < 3 {
        return OperationPattern::Unknown;
    }

    // Look for the characteristic pattern of vector operations:
    // 1. Loop of LDG (load) instructions
    // 2. Arithmetic operation (ADD/SUB/MUL/DIV)
    // 3. Loop of STG (store) instructions
    // 4. EXIT instruction

    // Count instruction types
    let mut load_count = 0;
    let mut store_count = 0;
    let mut add_count = 0;
    let mut sub_count = 0;
    let mut mul_count = 0;
    let mut div_count = 0;
    let mut has_exit = false;

    for inst in instructions {
        match inst {
            Instruction::LDG { .. } => load_count += 1,
            Instruction::STG { .. } => store_count += 1,
            Instruction::ADD { ty: T::F32, .. } => add_count += 1,
            Instruction::SUB { ty: T::F32, .. } => sub_count += 1,
            Instruction::MUL { ty: T::F32, .. } => mul_count += 1,
            Instruction::DIV { ty: T::F32, .. } => div_count += 1,
            Instruction::EXIT => has_exit = true,
            _ => {}
        }
    }

    // Must have exit instruction
    if !has_exit {
        return OperationPattern::Unknown;
    }

    // Detect vector binary operations
    // Pattern: 2n loads (n from A, n from B), n operations, n stores
    // Key distinction: load_count == 2 * store_count
    if load_count > 0 && store_count > 0 && load_count == 2 * store_count {
        let n = store_count;

        // Try to extract register information from first operations
        if let Some(pattern) = detect_binary_op(instructions, n, add_count, sub_count, mul_count, div_count) {
            return pattern;
        }
    }

    // Detect scalar operations
    // Pattern: n+1 loads (1 scalar, n vector), n operations, n stores
    // Key distinction: load_count == store_count + 1
    if load_count > store_count && load_count == store_count + 1 {
        if let Some(pattern) = detect_scalar_op(instructions, store_count, add_count, mul_count) {
            return pattern;
        }
    }

    // Detect matrix multiplication
    // Pattern: nested loops with many loads, muls, adds, stores
    if load_count > 10 && mul_count > 10 && add_count > 10 && store_count > 0 {
        if let Some(pattern) = detect_matmul(instructions) {
            return pattern;
        }
    }

    OperationPattern::Unknown
}

/// Detect binary operation pattern (add, sub, mul, div).
fn detect_binary_op(
    instructions: &[Instruction],
    n: usize,
    add_count: usize,
    sub_count: usize,
    mul_count: usize,
    div_count: usize,
) -> Option<OperationPattern> {
    // Determine which operation dominates
    let op_kind = if add_count >= n {
        "add"
    } else if sub_count >= n {
        "sub"
    } else if mul_count >= n {
        "mul"
    } else if div_count >= n {
        "div"
    } else {
        return None;
    };

    // Try to extract register information from the instruction stream
    // Look for first arithmetic operation to get register layout
    let mut dst_reg = None;
    let mut a_reg = None;
    let mut b_reg = None;

    for inst in instructions {
        match inst {
            Instruction::ADD {
                ty: T::F32,
                dst,
                src1,
                src2,
            } if op_kind == "add" => {
                dst_reg = Some(*dst);
                a_reg = Some(*src1);
                b_reg = Some(*src2);
                break;
            }
            Instruction::SUB {
                ty: T::F32,
                dst,
                src1,
                src2,
            } if op_kind == "sub" => {
                dst_reg = Some(*dst);
                a_reg = Some(*src1);
                b_reg = Some(*src2);
                break;
            }
            Instruction::MUL {
                ty: T::F32,
                dst,
                src1,
                src2,
            } if op_kind == "mul" => {
                dst_reg = Some(*dst);
                a_reg = Some(*src1);
                b_reg = Some(*src2);
                break;
            }
            Instruction::DIV {
                ty: T::F32,
                dst,
                src1,
                src2,
            } if op_kind == "div" => {
                dst_reg = Some(*dst);
                a_reg = Some(*src1);
                b_reg = Some(*src2);
                break;
            }
            _ => {}
        }
    }

    // Must have found register info
    let dst_reg = dst_reg?;
    let a_reg = a_reg?;
    let b_reg = b_reg?;

    match op_kind {
        "add" => Some(OperationPattern::VectorAdd {
            dst_reg,
            a_reg,
            b_reg,
            n,
        }),
        "sub" => Some(OperationPattern::VectorSub {
            dst_reg,
            a_reg,
            b_reg,
            n,
        }),
        "mul" => Some(OperationPattern::VectorMul {
            dst_reg,
            a_reg,
            b_reg,
            n,
        }),
        "div" => Some(OperationPattern::VectorDiv {
            dst_reg,
            a_reg,
            b_reg,
            n,
        }),
        _ => None,
    }
}

/// Detect scalar operation pattern (scalar add, scalar mul).
fn detect_scalar_op(
    instructions: &[Instruction],
    n: usize,
    add_count: usize,
    mul_count: usize,
) -> Option<OperationPattern> {
    let op_kind = if add_count >= n {
        "add"
    } else if mul_count >= n {
        "mul"
    } else {
        return None;
    };

    // Extract register information from first arithmetic operation
    let mut dst_reg = None;
    let mut input_reg = None;
    let mut scalar_reg = None;

    for inst in instructions {
        match inst {
            Instruction::ADD {
                ty: T::F32,
                dst,
                src1,
                src2,
            } if op_kind == "add" => {
                dst_reg = Some(*dst);
                input_reg = Some(*src1);
                scalar_reg = Some(*src2);
                break;
            }
            Instruction::MUL {
                ty: T::F32,
                dst,
                src1,
                src2,
            } if op_kind == "mul" => {
                dst_reg = Some(*dst);
                input_reg = Some(*src1);
                scalar_reg = Some(*src2);
                break;
            }
            _ => {}
        }
    }

    let dst_reg = dst_reg?;
    let input_reg = input_reg?;
    let scalar_reg = scalar_reg?;

    match op_kind {
        "add" => Some(OperationPattern::ScalarAdd {
            dst_reg,
            input_reg,
            scalar_reg,
            n,
        }),
        "mul" => Some(OperationPattern::ScalarMul {
            dst_reg,
            input_reg,
            scalar_reg,
            n,
        }),
        _ => None,
    }
}

/// Detect matrix multiplication pattern.
fn detect_matmul(_instructions: &[Instruction]) -> Option<OperationPattern> {
    // Matrix multiplication has a characteristic pattern:
    // - Many loads (2 * m * k * n for naive implementation)
    // - Many multiplications (m * n * k)
    // - Many additions (m * n * k for accumulation)
    // - Stores (m * n)
    //
    // For now, we'll use a simple heuristic based on instruction counts
    // In Phase 3, we'll implement a more sophisticated pattern matcher

    // TODO: Implement sophisticated GEMM pattern detection
    // For now, return None to use interpreter path
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_isa::{Address, Instruction, Register, Type as T};
    use std::collections::HashMap;

    #[test]
    fn test_detect_vector_add() {
        // Build a simple vector add program manually
        let n = 4;
        let mut instructions = Vec::new();

        for i in 0..n {
            let offset = i * 4;
            // Load a[i]
            instructions.push(Instruction::LDG {
                ty: T::F32,
                dst: Register::new(0),
                addr: Address::BufferOffset { handle: 0, offset },
            });
            // Load b[i]
            instructions.push(Instruction::LDG {
                ty: T::F32,
                dst: Register::new(1),
                addr: Address::BufferOffset { handle: 1, offset },
            });
            // Add
            instructions.push(Instruction::ADD {
                ty: T::F32,
                dst: Register::new(2),
                src1: Register::new(0),
                src2: Register::new(1),
            });
            // Store c[i]
            instructions.push(Instruction::STG {
                ty: T::F32,
                src: Register::new(2),
                addr: Address::BufferOffset { handle: 2, offset },
            });
        }
        instructions.push(Instruction::EXIT);

        let program = Program {
            instructions,
            labels: HashMap::new(),
        };
        let pattern = detect_pattern(&program);

        match pattern {
            OperationPattern::VectorAdd { n: detected_n, .. } => {
                assert_eq!(detected_n, n);
            }
            _ => panic!("Expected VectorAdd pattern, got {:?}", pattern),
        }
    }

    #[test]
    fn test_detect_unknown_pattern() {
        // Empty program
        let program = Program {
            instructions: vec![],
            labels: HashMap::new(),
        };
        let pattern = detect_pattern(&program);
        assert_eq!(pattern, OperationPattern::Unknown);

        // Program without exit
        let program = Program {
            instructions: vec![Instruction::LDG {
                ty: T::F32,
                dst: Register::new(0),
                addr: Address::BufferOffset { handle: 0, offset: 0 },
            }],
            labels: HashMap::new(),
        };
        let pattern = detect_pattern(&program);
        assert_eq!(pattern, OperationPattern::Unknown);
    }
}
