//! Complexity Classification (C0-C3)
//!
//! Classifies operations and circuits into complexity tiers for optimization strategy selection.
//!
//! ## Complexity Tiers
//!
//! - **C0**: Fully compiled (constant folding) → maximum fusion
//! - **C1**: Class-pure, few runtime params → fast class backend
//! - **C2**: Bounded mixed-grade → selective optimization
//! - **C3**: General case → full canonicalization
//!
//! ## Research Foundation
//!
//! From hologram-compiler research: complexity classification enables 4-8x faster
//! compilation by selecting appropriate optimization strategies.

use crate::ir::IRNode;

/// Complexity tier for optimization strategy selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComplexityTier {
    /// C0: Fully compiled, constant folding possible
    ///
    /// All operands known at compile-time. Can be fully pre-computed.
    /// Example: mark@c05 . copy@c10
    C0 = 0,

    /// C1: Class-pure, few runtime parameters
    ///
    /// Operations use only class addressing, no complex runtime dependencies.
    /// Example: merge@c[0..10] . copy@c20
    C1 = 1,

    /// C2: Bounded mixed-grade
    ///
    /// Mix of compile-time and runtime operations with bounded complexity.
    /// Example: merge@c[i..j] where i,j are runtime params but bounded
    C2 = 2,

    /// C3: General case, full canonicalization
    ///
    /// Arbitrary runtime dependencies, requires full canonicalization.
    /// Example: Complex nested operations with runtime-dependent control flow
    C3 = 3,
}

/// Classify IR complexity
///
/// Analyzes IR tree to determine appropriate optimization strategy.
pub fn classify_complexity(ir: &IRNode) -> ComplexityTier {
    match ir {
        IRNode::Atom(atom) => {
            if atom.range.is_some() {
                // Range operations are C1 (class-pure)
                ComplexityTier::C1
            } else {
                // Simple atoms (with or without static class) are C0
                ComplexityTier::C0
            }
        }
        IRNode::Seq(left, right) => {
            // Take maximum complexity of children
            classify_complexity(left).max(classify_complexity(right))
        }
        IRNode::Par(left, right) => {
            // Parallel composition: max complexity
            classify_complexity(left).max(classify_complexity(right))
        }
        IRNode::Transform(_, node) => {
            // Transforms don't increase complexity tier
            classify_complexity(node)
        }
    }
}

/// Complexity statistics for a program
#[derive(Debug, Clone)]
pub struct ComplexityStats {
    /// Overall complexity tier
    pub tier: ComplexityTier,

    /// Number of C0 operations
    pub c0_count: usize,

    /// Number of C1 operations
    pub c1_count: usize,

    /// Number of C2 operations
    pub c2_count: usize,

    /// Number of C3 operations
    pub c3_count: usize,
}

impl ComplexityStats {
    /// Analyze IR tree and generate complexity statistics
    pub fn analyze(ir: &IRNode) -> Self {
        let tier = classify_complexity(ir);
        let (c0, c1, c2, c3) = count_by_tier(ir);

        Self {
            tier,
            c0_count: c0,
            c1_count: c1,
            c2_count: c2,
            c3_count: c3,
        }
    }

    /// Get total operation count
    pub fn total_ops(&self) -> usize {
        self.c0_count + self.c1_count + self.c2_count + self.c3_count
    }

    /// Get percentage of operations in highest tier
    pub fn high_tier_percentage(&self) -> f64 {
        let total = self.total_ops() as f64;
        if total == 0.0 {
            return 0.0;
        }

        (self.c3_count as f64 / total) * 100.0
    }
}

/// Count operations by complexity tier
fn count_by_tier(ir: &IRNode) -> (usize, usize, usize, usize) {
    match ir {
        IRNode::Atom(_atom) => {
            let tier = classify_complexity(ir);
            match tier {
                ComplexityTier::C0 => (1, 0, 0, 0),
                ComplexityTier::C1 => (0, 1, 0, 0),
                ComplexityTier::C2 => (0, 0, 1, 0),
                ComplexityTier::C3 => (0, 0, 0, 1),
            }
        }
        IRNode::Seq(left, right) | IRNode::Par(left, right) => {
            let (l0, l1, l2, l3) = count_by_tier(left);
            let (r0, r1, r2, r3) = count_by_tier(right);
            (l0 + r0, l1 + r1, l2 + r2, l3 + r3)
        }
        IRNode::Transform(_, node) => count_by_tier(node),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Generator;

    #[test]
    fn test_classify_c0() {
        let atom = IRNode::atom(Generator::Mark, None, None);
        assert_eq!(classify_complexity(&atom), ComplexityTier::C0);
    }

    #[test]
    fn test_complexity_stats() {
        let atom1 = IRNode::atom(Generator::Mark, None, None);
        let atom2 = IRNode::atom(Generator::Copy, None, None);
        let seq = IRNode::seq(atom1, atom2);

        let stats = ComplexityStats::analyze(&seq);
        assert_eq!(stats.tier, ComplexityTier::C0);
        assert_eq!(stats.c0_count, 2);
        assert_eq!(stats.total_ops(), 2);
    }

    #[test]
    fn test_high_tier_percentage() {
        let atom = IRNode::atom(Generator::Mark, None, None);
        let stats = ComplexityStats::analyze(&atom);
        assert_eq!(stats.high_tier_percentage(), 0.0); // C0, so no C3 ops
    }
}
