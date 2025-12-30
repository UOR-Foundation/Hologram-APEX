//! IR Node Definitions
//!
//! Defines the IRNode enum and related types for the intermediate representation.

use crate::class::Transform;

// Re-export Generator from class for use within IR module
pub use crate::class::Generator;

/// IR Node - Intermediate representation for compile-time canonicalization
///
/// Represents a circuit as a tree structure that can be canonicalized and lowered.
#[derive(Debug, Clone, PartialEq)]
pub enum IRNode {
    /// Atomic generator call
    Atom(AtomNode),

    /// Sequential composition (A; B)
    Seq(Box<IRNode>, Box<IRNode>),

    /// Parallel composition (A | B)
    Par(Box<IRNode>, Box<IRNode>),

    /// Automorphism transform
    Transform(Transform, Box<IRNode>),
}

/// Atomic generator call
#[derive(Debug, Clone, PartialEq)]
pub struct AtomNode {
    /// Generator type
    pub generator: Generator,

    /// Class index (if applicable, 0..95)
    pub class: Option<u8>,

    /// Address range (if applicable)
    pub range: Option<AddressRange>,
}

/// Address range for generator operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddressRange {
    pub start: u32,
    pub end: u32,
}

impl IRNode {
    /// Create a new atomic node
    pub fn atom(generator: Generator, class: Option<u8>, range: Option<AddressRange>) -> Self {
        IRNode::Atom(AtomNode {
            generator,
            class,
            range,
        })
    }

    /// Create a sequential composition
    pub fn seq(left: IRNode, right: IRNode) -> Self {
        IRNode::Seq(Box::new(left), Box::new(right))
    }

    /// Create a parallel composition
    pub fn par(left: IRNode, right: IRNode) -> Self {
        IRNode::Par(Box::new(left), Box::new(right))
    }

    /// Apply a transform to a node
    pub fn transform(transform: Transform, node: IRNode) -> Self {
        IRNode::Transform(transform, Box::new(node))
    }

    /// Check if this node is an atom
    pub fn is_atom(&self) -> bool {
        matches!(self, IRNode::Atom(_))
    }

    /// Check if this node is a sequence
    pub fn is_seq(&self) -> bool {
        matches!(self, IRNode::Seq(_, _))
    }

    /// Check if this node is parallel
    pub fn is_par(&self) -> bool {
        matches!(self, IRNode::Par(_, _))
    }

    /// Check if this node has a transform
    pub fn is_transform(&self) -> bool {
        matches!(self, IRNode::Transform(_, _))
    }
}

impl Generator {
    /// Get the generator name
    pub fn name(&self) -> &'static str {
        match self {
            Generator::Mark => "mark",
            Generator::Copy => "copy",
            Generator::Swap => "swap",
            Generator::Merge => "merge",
            Generator::Split => "split",
            Generator::Quote => "quote",
            Generator::Evaluate => "evaluate",
        }
    }
}

impl AddressRange {
    /// Create a new address range
    pub fn new(start: u32, end: u32) -> Self {
        assert!(start <= end, "Invalid address range: start > end");
        Self { start, end }
    }

    /// Get the size of the range
    pub fn size(&self) -> u32 {
        self.end - self.start
    }

    /// Check if this range overlaps with another
    pub fn overlaps(&self, other: &AddressRange) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_creation() {
        let node = IRNode::atom(Generator::Mark, None, None);
        assert!(node.is_atom());
    }

    #[test]
    fn test_seq_creation() {
        let left = IRNode::atom(Generator::Mark, None, None);
        let right = IRNode::atom(Generator::Copy, None, None);
        let seq = IRNode::seq(left, right);
        assert!(seq.is_seq());
    }

    #[test]
    fn test_par_creation() {
        let left = IRNode::atom(Generator::Mark, None, None);
        let right = IRNode::atom(Generator::Copy, None, None);
        let par = IRNode::par(left, right);
        assert!(par.is_par());
    }

    #[test]
    fn test_address_range() {
        let range = AddressRange::new(0, 10);
        assert_eq!(range.size(), 10);

        let other = AddressRange::new(5, 15);
        assert!(range.overlaps(&other));

        let disjoint = AddressRange::new(20, 30);
        assert!(!range.overlaps(&disjoint));
    }

    #[test]
    fn test_generator_names() {
        assert_eq!(Generator::Mark.name(), "mark");
        assert_eq!(Generator::Copy.name(), "copy");
        assert_eq!(Generator::Swap.name(), "swap");
        assert_eq!(Generator::Merge.name(), "merge");
        assert_eq!(Generator::Split.name(), "split");
        assert_eq!(Generator::Quote.name(), "quote");
        assert_eq!(Generator::Evaluate.name(), "evaluate");
    }
}
