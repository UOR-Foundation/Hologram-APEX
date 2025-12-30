//! Direct JSON → ISA translation for kernel operations
//!
//! This module translates kernel JSON schemas directly to ISA Programs.
//! This is used for operations that are already in optimal form and don't
//! benefit from hologram-compiler canonicalization.
//!
//! For operations that can benefit from canonicalization (quantum circuits,
//! complex gate patterns), use the circuit_to_isa module instead.

use crate::isa::{Instruction, Program, Register, Type};
use crate::program_builder::{create_element_wise_binary, create_element_wise_unary};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// JSON schema structure (matching hologram-codegen build.rs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    pub version: String,
    pub kernel: KernelDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelDef {
    pub name: String,
    pub params: Vec<ParamDef>,
    pub body: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamDef {
    pub name: String,
    #[serde(rename = "type")]
    pub param_type: ParamType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum ParamType {
    #[serde(rename = "scalar")]
    Scalar {
        #[serde(rename = "type")]
        scalar_type: String,
    },
    #[serde(rename = "device_ptr")]
    DevicePtr,
    #[serde(rename = "device_array")]
    DeviceArray { element_type: Box<ParamType> },
}

/// Operation classification for translation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Binary element-wise: c[i] = a[i] op b[i]
    BinaryElementwise,
    /// Unary element-wise: c[i] = op(a[i])
    UnaryElementwise,
    /// Reduction: result = reduce(array, op)
    Reduction,
    /// Matrix operation (GEMM, GEMV)
    MatrixOp,
    /// Memory copy: y[i] = x[i] (reshape, flatten, transpose, etc.)
    MemoryCopy,
    /// Gather: output[i] = input[indices[i]]
    Gather,
    /// Complex / custom operation
    Complex,
}

/// Translate JSON schema to ISA Program
///
/// This bypasses hologram-compiler for simple operations that are already optimal.
/// For operations that can benefit from canonicalization, use hologram-compiler pipeline.
pub fn translate_json_to_isa(json: &JsonSchema) -> Result<Program, String> {
    let op_type = classify_operation(&json.kernel)?;

    match op_type {
        OperationType::BinaryElementwise => translate_binary_elementwise(json),
        OperationType::UnaryElementwise => translate_unary_elementwise(json),
        OperationType::Reduction => translate_reduction(json),
        OperationType::MatrixOp => translate_matrix_op(json),
        OperationType::MemoryCopy => translate_memory_copy(json),
        OperationType::Gather => translate_gather(json),
        OperationType::Complex => {
            // Check if this is an expected manual operation
            let name = json.kernel.name.to_lowercase();
            if name.contains("concat") || name.contains("attention") || name.contains("pool") {
                Err(format!(
                    "INFO: Operation '{}' requires manual ISA implementation (complex multi-pass algorithm). \
                     This is expected behavior - implement manually in hologram-core/src/precompiled_programs.rs",
                    json.kernel.name
                ))
            } else if name.contains("relu") {
                translate_relu(json)
            } else {
                Err(format!("ERROR: Unknown complex operation: {}", json.kernel.name))
            }
        }
    }
}

/// Classify operation type from JSON kernel definition
fn classify_operation(kernel: &KernelDef) -> Result<OperationType, String> {
    let name = kernel.name.to_lowercase();

    // Check parameter structure first (more reliable than name)
    // Count device_array parameters (inputs/outputs)
    let array_params = kernel
        .params
        .iter()
        .filter(|p| matches!(p.param_type, ParamType::DeviceArray { .. }))
        .count();

    // Gather operations: 3 device_array params (data input, indices input, output)
    if name.contains("gather") {
        return Ok(OperationType::Gather);
    }

    // Memory copy operations (shape manipulation)
    if name.contains("reshape")
        || name.contains("flatten")
        || name.contains("squeeze")
        || name.contains("unsqueeze")
        || name.contains("slice")
        || name.contains("transpose")
    {
        return Ok(OperationType::MemoryCopy);
    }

    // Binary operations: 3 device_array params (2 inputs + 1 output)
    if array_params == 3
        && (name.contains("add")
            || name.contains("sub")
            || name.contains("mul")
            || name.contains("div")
            || name.contains("min")
            || name.contains("max"))
    {
        return Ok(OperationType::BinaryElementwise);
    }

    // Special case: ReLU requires custom translation (max(0, x))
    if name.contains("relu") {
        return Ok(OperationType::Complex);
    }

    // Special case: Pooling operations are complex (require nested loops)
    if name.contains("pool") {
        return Ok(OperationType::Complex);
    }

    // Special case: Sum/dot are reductions, not unary ops (even with 2 array params)
    if name.contains("sum") || name.contains("dot") {
        return Ok(OperationType::Reduction);
    }

    // Special case: Scalar operations are binary (array op scalar), not unary
    // These have 2 device_array params (input, output) + scalar value params
    if array_params == 2 && name.contains("scalar") {
        return Ok(OperationType::BinaryElementwise);
    }

    // Unary operations: 2 device_array params (1 input + 1 output)
    if array_params == 2 {
        return Ok(OperationType::UnaryElementwise);
    }

    // Check operation name patterns as fallback (for tests with empty params, or when params are ambiguous)

    // Binary operations by name
    if name.contains("add")
        || name.contains("sub")
        || name.contains("mul")
        || name.contains("div")
        || name.contains("min")
        || name.contains("max")
    {
        return Ok(OperationType::BinaryElementwise);
    }

    // Unary operations by name
    if name.contains("sigmoid")
        || name.contains("tanh")
        || name.contains("exp")
        || name.contains("log")
        || name.contains("sin")
        || name.contains("cos")
        || name.contains("abs")
        || name.contains("neg")
    {
        return Ok(OperationType::UnaryElementwise);
    }

    if name.contains("sum") || name.contains("dot") {
        return Ok(OperationType::Reduction);
    }

    if name.contains("gemm") || name.contains("gemv") || name.contains("matmul") {
        return Ok(OperationType::MatrixOp);
    }

    Ok(OperationType::Complex)
}

/// Translate binary element-wise operation to ISA
fn translate_binary_elementwise(json: &JsonSchema) -> Result<Program, String> {
    let name = json.kernel.name.to_lowercase();

    // Determine operation instruction
    let op_fn: Box<dyn Fn(Register, Register, Register) -> Instruction> = if name.contains("add") {
        Box::new(|src1, src2, dst| Instruction::ADD {
            ty: Type::F32,
            dst,
            src1,
            src2,
        })
    } else if name.contains("sub") {
        Box::new(|src1, src2, dst| Instruction::SUB {
            ty: Type::F32,
            dst,
            src1,
            src2,
        })
    } else if name.contains("mul") {
        Box::new(|src1, src2, dst| Instruction::MUL {
            ty: Type::F32,
            dst,
            src1,
            src2,
        })
    } else if name.contains("div") {
        Box::new(|src1, src2, dst| Instruction::DIV {
            ty: Type::F32,
            dst,
            src1,
            src2,
        })
    } else if name.contains("min") {
        Box::new(|src1, src2, dst| Instruction::MIN {
            ty: Type::F32,
            dst,
            src1,
            src2,
        })
    } else if name.contains("max") {
        Box::new(|src1, src2, dst| Instruction::MAX {
            ty: Type::F32,
            dst,
            src1,
            src2,
        })
    } else {
        return Err(format!("Unknown binary operation: {}", name));
    };

    // Use program builder with placeholder handles (will be replaced at runtime)
    // For compile-time precompilation, we generate a program template
    Ok(create_element_wise_binary(0, 0, 0, Type::F32, op_fn))
}

/// Translate unary element-wise operation to ISA
fn translate_unary_elementwise(json: &JsonSchema) -> Result<Program, String> {
    let name = json.kernel.name.to_lowercase();

    // Determine operation instruction
    let op_fn: Box<dyn Fn(Register, Register) -> Instruction> = if name.contains("sigmoid") {
        Box::new(|src, dst| Instruction::SIGMOID {
            ty: Type::F32,
            dst,
            src,
        })
    } else if name.contains("tanh") {
        Box::new(|src, dst| Instruction::TANH {
            ty: Type::F32,
            dst,
            src,
        })
    } else if name.contains("exp") {
        Box::new(|src, dst| Instruction::EXP {
            ty: Type::F32,
            dst,
            src,
        })
    } else if name.contains("log") {
        Box::new(|src, dst| Instruction::LOG {
            ty: Type::F32,
            dst,
            src,
        })
    } else if name.contains("sin") {
        Box::new(|src, dst| Instruction::SIN {
            ty: Type::F32,
            dst,
            src,
        })
    } else if name.contains("cos") {
        Box::new(|src, dst| Instruction::COS {
            ty: Type::F32,
            dst,
            src,
        })
    } else if name.contains("abs") {
        Box::new(|src, dst| Instruction::ABS {
            ty: Type::F32,
            dst,
            src,
        })
    } else if name.contains("neg") {
        Box::new(|src, dst| Instruction::NEG {
            ty: Type::F32,
            dst,
            src,
        })
    } else {
        return Err(format!("Unknown unary operation: {}", name));
    };

    // Use program builder with placeholder handles
    Ok(create_element_wise_unary(0, 0, Type::F32, op_fn))
}

/// Translate ReLU operation to ISA
///
/// ReLU is max(0, x), which requires loading a zero constant and using MAX instruction
fn translate_relu(_json: &JsonSchema) -> Result<Program, String> {
    use crate::isa::special_registers::*;
    use crate::isa::{Address, Condition, Instruction, Label, Predicate, Program, Register, Type};
    use std::collections::HashMap;

    let shift_amount = 2; // F32 = 4 bytes = 2^2

    let mut labels = HashMap::new();
    let exit_label = "bounds_exit";

    // Build instruction vector for ReLU: max(0, x)
    let mut instructions = vec![
        // Bounds check: if (GLOBAL_LANE_ID >= n) exit
        Instruction::SETcc {
            ty: Type::U64,
            cond: Condition::GEU,
            dst: Predicate::new(0),
            src1: GLOBAL_LANE_ID,
            src2: Register(4), // n passed via R4
        },
        Instruction::BRA {
            target: Label::new(exit_label),
            pred: Some(Predicate::new(0)),
        },
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
        // Load input value
        Instruction::LDG {
            ty: Type::F32,
            dst: Register(10),
            addr: Address::RegisterIndirectComputed {
                handle_reg: Register(1),
                offset_reg: Register(0),
            },
        },
        // Load zero constant
        Instruction::MOV_IMM {
            ty: Type::F32,
            dst: Register(11),
            value: 0, // 0.0f32 as bits
        },
        // Compute max(value, 0.0)
        Instruction::MAX {
            ty: Type::F32,
            dst: Register(12),
            src1: Register(10),
            src2: Register(11),
        },
        // Store result
        Instruction::STG {
            ty: Type::F32,
            src: Register(12),
            addr: Address::RegisterIndirectComputed {
                handle_reg: Register(2), // Changed from Register(3) - unary ops use R1 (input), R2 (output)
                offset_reg: Register(0),
            },
        },
    ];

    // Add EXIT instruction and label
    let exit_index = instructions.len();
    instructions.push(Instruction::EXIT);
    labels.insert(exit_label.to_string(), exit_index);

    Ok(Program { instructions, labels })
}

/// Translate reduction operation to ISA
fn translate_reduction(json: &JsonSchema) -> Result<Program, String> {
    let name = json.kernel.name.to_lowercase();

    // Create reduction program
    // Note: count is a placeholder (0) - will be set at runtime based on array size
    let instruction = if name.contains("sum") || name.contains("add") || name.contains("dot") {
        Instruction::ReduceAdd {
            ty: Type::F32,
            dst: Register(0),
            src_base: Register(1),
            count: 0, // Placeholder - actual count determined at runtime
        }
    } else if name.contains("min") {
        Instruction::ReduceMin {
            ty: Type::F32,
            dst: Register(0),
            src_base: Register(1),
            count: 0, // Placeholder - actual count determined at runtime
        }
    } else if name.contains("max") {
        Instruction::ReduceMax {
            ty: Type::F32,
            dst: Register(0),
            src_base: Register(1),
            count: 0, // Placeholder - actual count determined at runtime
        }
    } else {
        return Err(format!("Unknown reduction operation: {}", name));
    };

    Ok(Program {
        instructions: vec![instruction, Instruction::EXIT],
        labels: HashMap::new(),
    })
}

/// Translate memory copy operation to ISA
///
/// Memory copy operations (reshape, flatten, squeeze, unsqueeze, slice) are
/// essentially copying data from input to output with different indexing.
/// For simplicity, we implement them as element-wise copies.
fn translate_memory_copy(_json: &JsonSchema) -> Result<Program, String> {
    // Use the same pattern as element-wise operations
    // The actual shape manipulation is handled by the caller setting up correct buffer views
    Ok(create_element_wise_unary(
        0,
        0,
        Type::F32,
        Box::new(|src, dst| Instruction::MOV {
            ty: Type::F32,
            dst,
            src,
        }),
    ))
}

/// Translate gather operation to ISA
///
/// Gather: output[i] = input[indices[i]]
/// Requires two LDG operations: first load the index, then load the value
fn translate_gather(_json: &JsonSchema) -> Result<Program, String> {
    use crate::isa::special_registers::*;
    use crate::isa::{Address, Condition, Label, Predicate};

    let shift_amount = 2; // F32 = 4 bytes = 2^2
    let mut labels = HashMap::new();
    let exit_label = "bounds_exit";

    let mut instructions = vec![
        // Bounds check: if (GLOBAL_LANE_ID >= n) exit
        Instruction::SETcc {
            ty: Type::U64,
            cond: Condition::GEU,
            dst: Predicate::new(0),
            src1: GLOBAL_LANE_ID,
            src2: Register(4), // n passed via R4
        },
        Instruction::BRA {
            target: Label::new(exit_label),
            pred: Some(Predicate::new(0)),
        },
        // Compute byte offset for indices array access: offset = idx * 4
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
        // Load index value: idx_val = indices[i]
        Instruction::LDG {
            ty: Type::U32,
            dst: Register(20),
            addr: Address::RegisterIndirectComputed {
                handle_reg: Register(2), // indices buffer
                offset_reg: Register(0),
            },
        },
        // Compute byte offset for data array access: data_offset = idx_val * 4
        Instruction::SHL {
            ty: Type::U64,
            dst: Register(21),
            src: Register(20),
            amount: Register(250),
        },
        // Load data value: val = input[idx_val]
        Instruction::LDG {
            ty: Type::F32,
            dst: Register(22),
            addr: Address::RegisterIndirectComputed {
                handle_reg: Register(1), // input data buffer
                offset_reg: Register(21),
            },
        },
        // Store to output: output[i] = val
        Instruction::STG {
            ty: Type::F32,
            src: Register(22),
            addr: Address::RegisterIndirectComputed {
                handle_reg: Register(3), // output buffer
                offset_reg: Register(0), // same offset as index access
            },
        },
    ];

    // Add EXIT instruction and label
    let exit_index = instructions.len();
    instructions.push(Instruction::EXIT);
    labels.insert(exit_label.to_string(), exit_index);

    Ok(Program { instructions, labels })
}

/// Translate matrix operation to ISA
fn translate_matrix_op(json: &JsonSchema) -> Result<Program, String> {
    let name = json.kernel.name.to_lowercase();

    if name.contains("gemm") {
        // GEMM: C = A * B
        // For now, create a simple program with GEMM instruction
        // Actual dimensions will be provided at runtime
        Ok(Program {
            instructions: vec![
                Instruction::Gemm {
                    ty: Type::F32,
                    matrix_a: Register(1), // Input matrix A
                    matrix_b: Register(2), // Input matrix B
                    matrix_c: Register(3), // Output matrix C
                    m: 0,                  // Placeholder - set at runtime
                    k: 0,                  // Placeholder - set at runtime
                    n: 0,                  // Placeholder - set at runtime
                },
                Instruction::EXIT,
            ],
            labels: HashMap::new(),
        })
    } else if name.contains("gemv") {
        // GEMV: y = A*x (decompose to GEMM with N=1)
        Ok(Program {
            instructions: vec![
                Instruction::Gemm {
                    ty: Type::F32,
                    matrix_a: Register(1), // M×N matrix
                    matrix_b: Register(2), // N×1 vector (treated as matrix)
                    matrix_c: Register(3), // M×1 result
                    m: 0,                  // Placeholder - set at runtime
                    k: 0,                  // Placeholder - set at runtime
                    n: 1,                  // Vector as N×1 matrix
                },
                Instruction::EXIT,
            ],
            labels: HashMap::new(),
        })
    } else {
        Err(format!("Unknown matrix operation: {}", json.kernel.name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_operations() {
        let mut kernel = KernelDef {
            name: "vector_add".to_string(),
            params: vec![],
            body: vec![],
        };

        assert_eq!(classify_operation(&kernel).unwrap(), OperationType::BinaryElementwise);

        kernel.name = "sigmoid".to_string();
        assert_eq!(classify_operation(&kernel).unwrap(), OperationType::UnaryElementwise);

        kernel.name = "sum".to_string();
        assert_eq!(classify_operation(&kernel).unwrap(), OperationType::Reduction);

        kernel.name = "gemm".to_string();
        assert_eq!(classify_operation(&kernel).unwrap(), OperationType::MatrixOp);
    }

    #[test]
    fn test_translate_vector_add() {
        let json = JsonSchema {
            version: "1.0".to_string(),
            kernel: KernelDef {
                name: "vector_add".to_string(),
                params: vec![],
                body: vec![],
            },
        };

        let program = translate_json_to_isa(&json).unwrap();
        assert!(!program.instructions.is_empty());
        assert!(matches!(program.instructions.last(), Some(Instruction::EXIT)));
    }
}
