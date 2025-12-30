//! Abstract Group Theory
//!
//! Defines group axioms from first principles:
//! 1. Closure: ∀ a,b ∈ G: a ○ b ∈ G
//! 2. Associativity: ∀ a,b,c ∈ G: (a ○ b) ○ c = a ○ (b ○ c)
//! 3. Identity: ∃ e ∈ G: ∀ a ∈ G: a ○ e = e ○ a = a
//! 4. Inverse: ∀ a ∈ G: ∃ a⁻¹ ∈ G: a ○ a⁻¹ = a⁻¹ ○ a = e

use std::fmt::Debug;

/// Abstract group trait
///
/// A group (G, ○) is a set G with a binary operation ○ satisfying
/// the four group axioms: closure, associativity, identity, and inverse.
pub trait Group: Clone + Eq + Debug {
    /// Group operation (abstract binary operation)
    fn op(&self, other: &Self) -> Self;
    
    /// Identity element
    fn identity() -> Self;
    
    /// Inverse element
    fn inverse(&self) -> Self;
    
    // Axiom verification (default implementations for testing)
    
    /// Verify closure: a ○ b ∈ G
    fn verify_closure(&self, other: &Self) -> bool {
        // Result type must be Self
        let _result = self.op(other);
        true // If it compiles, closure holds
    }
    
    /// Verify associativity: (a ○ b) ○ c = a ○ (b ○ c)
    fn verify_associativity(&self, b: &Self, c: &Self) -> bool {
        let lhs = self.op(b).op(c);
        let rhs = self.op(&b.op(c));
        lhs == rhs
    }
    
    /// Verify identity: a ○ e = e ○ a = a
    fn verify_identity(&self) -> bool {
        let e = Self::identity();
        self.op(&e) == *self && e.op(self) == *self
    }
    
    /// Verify inverse: a ○ a⁻¹ = a⁻¹ ○ a = e
    fn verify_inverse(&self) -> bool {
        let inv = self.inverse();
        let e = Self::identity();
        self.op(&inv) == e && inv.op(self) == e
    }
}

/// Abelian (commutative) group
pub trait AbelianGroup: Group {
    /// Verify commutativity: a ○ b = b ○ a
    fn verify_commutativity(&self, other: &Self) -> bool {
        self.op(other) == other.op(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Example: Integers under addition (unbounded)
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct AdditiveInt(i64);
    
    impl Group for AdditiveInt {
        fn op(&self, other: &Self) -> Self {
            AdditiveInt(self.0.wrapping_add(other.0))
        }
        
        fn identity() -> Self {
            AdditiveInt(0)
        }
        
        fn inverse(&self) -> Self {
            AdditiveInt(self.0.wrapping_neg())
        }
    }
    
    impl AbelianGroup for AdditiveInt {}
    
    #[test]
    fn test_group_axioms() {
        let a = AdditiveInt(5);
        let b = AdditiveInt(3);
        let c = AdditiveInt(7);
        
        // Closure
        assert!(a.verify_closure(&b));
        
        // Associativity
        assert!(a.verify_associativity(&b, &c));
        
        // Identity
        assert!(a.verify_identity());
        
        // Inverse
        assert!(a.verify_inverse());
        
        // Commutativity
        assert!(a.verify_commutativity(&b));
    }
}
