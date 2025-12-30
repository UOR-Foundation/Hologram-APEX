//! Canonical Graph-Based Execution
//!
//! This module implements the canonical format for Atlas execution:
//! - Operations are graph edge traversals
//! - Execution happens directly on class_bases[96] (zero data movement)
//! - Seven generators define all operations
//! - Graph structure from atlas-embeddings defines valid operations

pub mod translator;
pub mod validation;

use crate::{BackendError, ExecutionContext, Result};
use atlas_core::atlas;
use atlas_isa::{Instruction, Program};

/// A graph operation: traversal from src class to dst class
#[derive(Debug, Clone)]
pub struct GraphOperation {
    /// Source class
    pub src: u8,
    /// Destination class
    pub dst: u8,
    /// Operation generator
    pub generator: Generator,
    /// Additional parameters
    pub params: OpParams,
}

/// The seven generators from sigil algebra
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Generator {
    /// Introduce/remove mark (d=0: neutral creation, guarded)
    Mark,
    /// Comultiplication/fan-out (d biases direction)
    Copy,
    /// Symmetry/braid on wires (permutation)
    Swap,
    /// Fold/meet operation (d=1: produce, d=2: consume)
    Merge,
    /// Case analysis/deconstruct (context ℓ selects case)
    Split,
    /// Suspend/delay (binds to context ℓ)
    Quote,
    /// Force/thunk discharge (consults scope h₂)
    Evaluate,
}

/// Operation parameters
#[derive(Debug, Clone, Default)]
pub struct OpParams {
    /// Transformation to apply (R, T_k, M)
    pub transform: Option<Transform>,
    /// Additional context (e.g., second input class)
    pub context: Option<u8>,
}

/// The three fundamental transformations from sigil algebra
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// Quarter-turn: R(h₂, d, ℓ) = ((h₂+1) mod 4, d, ℓ)
    QuarterTurn(i8),
    /// Inner-twist: T_k(h₂, d, ℓ) = (h₂, d, (ℓ+k) mod 8)
    InnerTwist(i8),
    /// Mirror: M flips d (1↔2) and reflects h₂, ℓ
    Mirror,
}

impl GraphOperation {
    /// Create a new graph operation
    pub fn new(src: u8, dst: u8, generator: Generator) -> Self {
        Self {
            src,
            dst,
            generator,
            params: OpParams::default(),
        }
    }

    /// Create with transformation
    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.params.transform = Some(transform);
        self
    }

    /// Create with context
    pub fn with_context(mut self, context: u8) -> Self {
        self.params.context = Some(context);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_operation_creation() {
        let op = GraphOperation::new(0, 1, Generator::Copy);
        assert_eq!(op.src, 0);
        assert_eq!(op.dst, 1);
        assert_eq!(op.generator, Generator::Copy);
    }

    #[test]
    fn test_graph_operation_with_transform() {
        let op = GraphOperation::new(0, 1, Generator::Copy).with_transform(Transform::Mirror);
        assert!(matches!(op.params.transform, Some(Transform::Mirror)));
    }

    #[test]
    fn test_graph_operation_with_context() {
        let op = GraphOperation::new(0, 5, Generator::Merge).with_context(2);
        assert_eq!(op.params.context, Some(2));
    }
}
