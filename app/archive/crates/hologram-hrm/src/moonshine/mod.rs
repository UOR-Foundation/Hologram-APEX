//! MoonshineHRM group operators
//!
//! The Moonshine module provides the group action framework for HRM embeddings.
//! Each of the 96 resonance classes corresponds to a group operator that can
//! act on Griess vectors.
//!
//! Key concepts:
//! - **MoonshineOperator**: Group element corresponding to a resonance class
//! - **Group composition**: Operator multiplication via resonance class arithmetic
//! - **Identity and inverse**: Group axioms
//! - **Action on vectors**: How operators transform Griess vectors

#![allow(missing_docs)]

pub mod action;

use crate::algebra::{LieAlgebra, MoonshineAlgebra};
use crate::{Error, GriessVector, Result};

/// Number of Moonshine operators (one per resonance class)
pub const NUM_OPERATORS: usize = 96;

/// MoonshineOperator: Group element acting on Griess space
///
/// Each operator corresponds to a resonance class k ∈ ℤ₉₆.
/// Operators form a group under composition with:
/// - Identity: k = 0
/// - Composition: k₁ ⊗ k₂ = (k₁ · k₂) mod 96
/// - Inverse: k⁻¹ such that k ⊗ k⁻¹ = 0
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoonshineOperator {
    /// Resonance class index k ∈ [0, 95]
    pub class: u8,
}

impl MoonshineOperator {
    /// Create a new Moonshine operator for given resonance class
    pub fn new(class: u8) -> Result<Self> {
        if class >= NUM_OPERATORS as u8 {
            return Err(Error::InvalidInput(format!(
                "Resonance class {} out of range [0, {})",
                class, NUM_OPERATORS
            )));
        }
        Ok(Self { class })
    }

    /// Identity operator (class 0)
    pub fn identity() -> Self {
        Self { class: 0 }
    }

    /// Check if this is the identity operator
    pub fn is_identity(&self) -> bool {
        self.class == 0
    }

    /// Compose two operators: self ∘ other
    ///
    /// Group composition uses addition in ℤ₉₆:
    /// (k₁ ⊗ k₂) = (k₁ + k₂) mod 96
    pub fn compose(&self, other: &Self) -> Self {
        let result_class = ((self.class as u16 + other.class as u16) % NUM_OPERATORS as u16) as u8;
        Self { class: result_class }
    }

    /// Compute the inverse operator
    ///
    /// Find k⁻¹ such that k ⊗ k⁻¹ = 0 (identity)
    /// For additive group ℤ₉₆: k⁻¹ = -k mod 96 = (96 - k) mod 96
    pub fn inverse(&self) -> Result<Self> {
        if self.is_identity() {
            return Ok(Self::identity());
        }

        // Additive inverse: -k = 96 - k
        let inv = (NUM_OPERATORS as u8 - self.class) % NUM_OPERATORS as u8;
        Ok(Self { class: inv })
    }

    /// Power operation: apply operator n times
    ///
    /// op^n = op ∘ op ∘ ... ∘ op (n times)
    /// For additive group: n*k mod 96
    pub fn pow(&self, n: u32) -> Self {
        let result_class = ((self.class as u64 * n as u64) % NUM_OPERATORS as u64) as u8;
        Self { class: result_class }
    }

    /// Get the Lie algebra generator corresponding to this operator
    ///
    /// Each operator corresponds to a generator of the Lie algebra.
    /// The operator is approximately exp(generator).
    pub fn to_generator(&self, algebra: &MoonshineAlgebra) -> Result<GriessVector> {
        algebra.generator(self.class as usize)
    }

    /// Create operator from Lie algebra exponential
    ///
    /// Given a generator index and scaling factor θ,
    /// creates operator ≈ exp(θ·g_i)
    pub fn from_exp(generator_idx: usize, theta: f64, _algebra: &MoonshineAlgebra) -> Result<Self> {
        // For small θ, exp(θ·g_i) ≈ operator for class i
        // For larger θ, need to scale the class index
        let scaled_class = ((generator_idx as f64 * (1.0 + theta)) as usize) % NUM_OPERATORS;
        Self::new(scaled_class as u8)
    }
}

/// Operator sequence for multi-digit encoding
///
/// Represents a sequence of operators to be composed:
/// G = g₀ ∘ g₁ ∘ g₂ ∘ ... ∘ gₙ
#[derive(Debug, Clone)]
pub struct OperatorSequence {
    pub operators: Vec<MoonshineOperator>,
}

impl OperatorSequence {
    /// Create an empty sequence
    pub fn new() -> Self {
        Self { operators: Vec::new() }
    }

    /// Create from a list of resonance classes
    pub fn from_classes(classes: &[u8]) -> Result<Self> {
        let operators: Result<Vec<_>> = classes.iter().map(|&c| MoonshineOperator::new(c)).collect();
        Ok(Self { operators: operators? })
    }

    /// Add an operator to the sequence
    pub fn push(&mut self, op: MoonshineOperator) {
        self.operators.push(op);
    }

    /// Compose all operators in the sequence
    ///
    /// Returns the single operator equivalent to applying all operators in order:
    /// result = gₙ ∘ ... ∘ g₁ ∘ g₀
    pub fn compose_all(&self) -> MoonshineOperator {
        let mut result = MoonshineOperator::identity();
        for op in &self.operators {
            result = result.compose(op);
        }
        result
    }

    /// Get the length of the sequence
    pub fn len(&self) -> usize {
        self.operators.len()
    }

    /// Check if sequence is empty
    pub fn is_empty(&self) -> bool {
        self.operators.is_empty()
    }
}

impl Default for OperatorSequence {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_creation() {
        let op = MoonshineOperator::new(5).unwrap();
        assert_eq!(op.class, 5);

        let identity = MoonshineOperator::identity();
        assert_eq!(identity.class, 0);
        assert!(identity.is_identity());
    }

    #[test]
    fn test_operator_composition() {
        let op1 = MoonshineOperator::new(3).unwrap();
        let op2 = MoonshineOperator::new(5).unwrap();

        // 3 + 5 = 8 mod 96 = 8
        let composed = op1.compose(&op2);
        assert_eq!(composed.class, 8);
    }

    #[test]
    fn test_identity_composition() {
        let op = MoonshineOperator::new(7).unwrap();
        let identity = MoonshineOperator::identity();

        let result1 = op.compose(&identity);
        assert_eq!(result1.class, 7); // 7 + 0 = 7

        let result2 = identity.compose(&op);
        assert_eq!(result2.class, 7); // 0 + 7 = 7
    }

    #[test]
    fn test_operator_power() {
        let op = MoonshineOperator::new(2).unwrap();

        // 3 * 2 = 6 (add 2 three times)
        let cubed = op.pow(3);
        assert_eq!(cubed.class, 6);

        // 6 * 2 = 12 (add 2 six times)
        let sixth = op.pow(6);
        assert_eq!(sixth.class, 12);
    }

    #[test]
    fn test_operator_inverse() {
        // Test that all operators have additive inverses
        for class in [1u8, 5, 7, 11, 13, 17, 19, 23, 25, 50, 95] {
            let op = MoonshineOperator::new(class).unwrap();
            let inv = op.inverse().unwrap();

            // Additive inverse: k + (-k) = 0
            let composed = op.compose(&inv);
            assert_eq!(
                composed.class,
                0,
                "Inverse of {} should be {}, got class {}",
                class,
                96 - class,
                composed.class
            );
        }
    }

    #[test]
    fn test_identity_inverse() {
        let identity = MoonshineOperator::identity();
        let inv = identity.inverse().unwrap();
        assert!(inv.is_identity());
    }

    #[test]
    fn test_operator_sequence() {
        let mut seq = OperatorSequence::new();
        assert!(seq.is_empty());

        seq.push(MoonshineOperator::new(2).unwrap());
        seq.push(MoonshineOperator::new(3).unwrap());
        seq.push(MoonshineOperator::new(5).unwrap());

        assert_eq!(seq.len(), 3);
        assert!(!seq.is_empty());

        // Compose: (2 + 3 + 5) mod 96 = 10
        let composed = seq.compose_all();
        assert_eq!(composed.class, 10);
    }

    #[test]
    fn test_sequence_from_classes() {
        let classes = vec![1, 2, 3, 4, 5];
        let seq = OperatorSequence::from_classes(&classes).unwrap();

        assert_eq!(seq.len(), 5);
        assert_eq!(seq.operators[0].class, 1);
        assert_eq!(seq.operators[4].class, 5);
    }

    #[test]
    fn test_invalid_class() {
        let result = MoonshineOperator::new(96);
        assert!(result.is_err());

        let result = MoonshineOperator::new(255);
        assert!(result.is_err());
    }
}
