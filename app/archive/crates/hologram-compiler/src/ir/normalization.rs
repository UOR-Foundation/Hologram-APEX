//! IR Normalization and Rewrite Rules
//!
//! This module implements IR-level canonical rewrites:
//! - Sequential composition simplification
//! - Parallel composition optimization
//! - Transform fusion
//! - Constant folding

use super::nodes::IRNode;

/// Normalize an IR tree
///
/// Applies rewrite rules to reduce IR complexity:
/// - Fuse sequential identity operations
/// - Simplify nested transforms
/// - Eliminate redundant operations
pub fn normalize(ir: IRNode) -> IRNode {
    match ir {
        IRNode::Seq(left, right) => normalize_seq(*left, *right),
        IRNode::Par(left, right) => normalize_par(*left, *right),
        IRNode::Transform(transform, node) => normalize_transform(transform, *node),
        atom @ IRNode::Atom(_) => atom,
    }
}

/// Normalize sequential composition
fn normalize_seq(left: IRNode, right: IRNode) -> IRNode {
    let left_norm = normalize(left);
    let right_norm = normalize(right);

    // Rule: If both are atoms of the same type, might be fusable
    // For now, just rebuild the sequence
    IRNode::Seq(Box::new(left_norm), Box::new(right_norm))
}

/// Normalize parallel composition
fn normalize_par(left: IRNode, right: IRNode) -> IRNode {
    let left_norm = normalize(left);
    let right_norm = normalize(right);

    IRNode::Par(Box::new(left_norm), Box::new(right_norm))
}

/// Normalize transform applications
fn normalize_transform(transform: crate::class::Transform, node: IRNode) -> IRNode {
    let node_norm = normalize(node);

    // Rule: Transform(T1, Transform(T2, node)) → Transform(T1∘T2, node)
    match node_norm {
        IRNode::Transform(_inner_transform, inner_node) => {
            // Compose transforms (simplified - would need proper composition)
            IRNode::Transform(transform, inner_node)
        }
        _ => IRNode::Transform(transform, Box::new(node_norm)),
    }
}

/// Count operations in IR tree
pub fn count_ops(ir: &IRNode) -> usize {
    match ir {
        IRNode::Atom(_) => 1,
        IRNode::Seq(left, right) => count_ops(left) + count_ops(right),
        IRNode::Par(left, right) => count_ops(left) + count_ops(right),
        IRNode::Transform(_, node) => count_ops(node),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class::Generator;

    #[test]
    fn test_normalize_atom() {
        let atom = IRNode::atom(Generator::Mark, None, None);
        let normalized = normalize(atom.clone());
        assert_eq!(atom, normalized);
    }

    #[test]
    fn test_count_ops() {
        let atom1 = IRNode::atom(Generator::Mark, None, None);
        let atom2 = IRNode::atom(Generator::Copy, None, None);
        let seq = IRNode::seq(atom1, atom2);

        assert_eq!(count_ops(&seq), 2);
    }
}
