//! IR → ISA Program Lowering
//!
//! This module translates normalized IR into ISA Programs for execution.
//! This is the final compilation stage before runtime.

use super::nodes::{AddressRange, Generator, IRNode};

/// Lower IR to ISA Program
///
/// Converts normalized IR into executable ISA instructions.
/// This is compile-time only - runtime never sees IR.
pub fn lower(ir: &IRNode) -> ISAProgram {
    match ir {
        IRNode::Atom(atom) => lower_atom(atom),
        IRNode::Seq(left, right) => {
            let left_program = lower(left);
            let right_program = lower(right);
            left_program.append(right_program)
        }
        IRNode::Par(left, right) => {
            let left_program = lower(left);
            let right_program = lower(right);
            left_program.parallel(right_program)
        }
        IRNode::Transform(transform, node) => {
            let mut program = lower(node);
            program.apply_transform(*transform);
            program
        }
    }
}

/// Lower atomic IR node to ISA instruction
fn lower_atom(atom: &super::nodes::AtomNode) -> ISAProgram {
    let instruction = match atom.generator {
        Generator::Mark => ISAInstruction::Mark {
            class: atom.class,
            range: atom.range,
        },
        Generator::Copy => ISAInstruction::Copy {
            class: atom.class,
            range: atom.range,
        },
        Generator::Swap => ISAInstruction::Swap {
            class: atom.class,
            range: atom.range,
        },
        Generator::Merge => ISAInstruction::Merge {
            class: atom.class,
            range: atom.range,
        },
        Generator::Split => ISAInstruction::Split {
            class: atom.class,
            range: atom.range,
        },
        Generator::Quote => ISAInstruction::Quote {
            class: atom.class,
            range: atom.range,
        },
        Generator::Evaluate => ISAInstruction::Evaluate {
            class: atom.class,
            range: atom.range,
        },
    };

    ISAProgram {
        instructions: vec![instruction],
    }
}

/// ISA Program - sequence of executable instructions
#[derive(Debug, Clone)]
pub struct ISAProgram {
    pub instructions: Vec<ISAInstruction>,
}

impl ISAProgram {
    /// Append another program sequentially
    pub fn append(mut self, other: ISAProgram) -> Self {
        self.instructions.extend(other.instructions);
        self
    }

    /// Parallel composition of programs
    pub fn parallel(mut self, other: ISAProgram) -> Self {
        // Simplified: just append for now
        // Real implementation would mark parallel execution
        self.instructions.extend(other.instructions);
        self
    }

    /// Apply transform to all instructions in program
    pub fn apply_transform(&mut self, _transform: crate::class::Transform) {
        // Transform application would modify class indices
        // Placeholder for now
    }

    /// Get total instruction count
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }
}

/// ISA Instruction types
#[derive(Debug, Clone)]
pub enum ISAInstruction {
    Mark {
        class: Option<u8>,
        range: Option<AddressRange>,
    },
    Copy {
        class: Option<u8>,
        range: Option<AddressRange>,
    },
    Swap {
        class: Option<u8>,
        range: Option<AddressRange>,
    },
    Merge {
        class: Option<u8>,
        range: Option<AddressRange>,
    },
    Split {
        class: Option<u8>,
        range: Option<AddressRange>,
    },
    Quote {
        class: Option<u8>,
        range: Option<AddressRange>,
    },
    Evaluate {
        class: Option<u8>,
        range: Option<AddressRange>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_atom() {
        let atom = IRNode::atom(Generator::Mark, None, None);
        let program = lower(&atom);
        assert_eq!(program.instruction_count(), 1);
    }

    #[test]
    fn test_lower_sequence() {
        let atom1 = IRNode::atom(Generator::Mark, None, None);
        let atom2 = IRNode::atom(Generator::Copy, None, None);
        let seq = IRNode::seq(atom1, atom2);

        let program = lower(&seq);
        assert_eq!(program.instruction_count(), 2);
    }
}
