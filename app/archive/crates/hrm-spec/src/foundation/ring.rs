//! Ring Theory
//!
//! A ring (R, +, ×) is a set R with two binary operations satisfying:
//! 1. (R, +) is an abelian group
//! 2. (R, ×) is a monoid (associative with identity)
//! 3. Distributivity: a × (b + c) = (a × b) + (a × c)

use super::group::AbelianGroup;
use std::fmt::Debug;

/// Abstract ring trait
pub trait Ring: AbelianGroup + Clone + Eq + Debug {
    /// Multiplicative operation
    fn mul(&self, other: &Self) -> Self;
    
    /// Multiplicative identity (one)
    fn one() -> Self;
    
    /// Additive identity (zero) - inherited from Group
    fn zero() -> Self {
        Self::identity()
    }
    
    // Ring axiom verification
    
    /// Verify left distributivity: a × (b + c) = (a × b) + (a × c)
    fn verify_left_distributivity(&self, b: &Self, c: &Self) -> bool {
        let lhs = self.mul(&b.op(c));
        let rhs = self.mul(b).op(&self.mul(c));
        lhs == rhs
    }
    
    /// Verify right distributivity: (a + b) × c = (a × c) + (b × c)
    fn verify_right_distributivity(&self, b: &Self, c: &Self) -> bool {
        let lhs = self.op(b).mul(c);
        let rhs = self.mul(c).op(&b.mul(c));
        lhs == rhs
    }
    
    /// Verify multiplicative associativity: (a × b) × c = a × (b × c)
    fn verify_mul_associativity(&self, b: &Self, c: &Self) -> bool {
        let lhs = self.mul(b).mul(c);
        let rhs = self.mul(&b.mul(c));
        lhs == rhs
    }
    
    /// Verify multiplicative identity: a × 1 = 1 × a = a
    fn verify_mul_identity(&self) -> bool {
        let one = Self::one();
        self.mul(&one) == *self && one.mul(self) == *self
    }
    
    /// Verify all ring axioms
    fn verify_ring_axioms(&self, b: &Self, c: &Self) -> bool {
        self.verify_left_distributivity(b, c) &&
        self.verify_right_distributivity(b, c) &&
        self.verify_mul_associativity(b, c) &&
        self.verify_mul_identity()
    }
}

/// Commutative ring (multiplication is commutative)
pub trait CommutativeRing: Ring {
    fn verify_mul_commutativity(&self, other: &Self) -> bool {
        self.mul(other) == other.mul(self)
    }
}

/// Field (commutative ring where every non-zero element has multiplicative inverse)
pub trait Field: CommutativeRing {
    /// Multiplicative inverse
    fn mul_inverse(&self) -> Option<Self>;
    
    /// Verify multiplicative inverse: a × a⁻¹ = 1
    fn verify_mul_inverse(&self) -> bool {
        if let Some(inv) = self.mul_inverse() {
            let one = Self::one();
            self.mul(&inv) == one && inv.mul(self) == one
        } else {
            // Zero has no multiplicative inverse
            *self == Self::zero()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::group::{Group, AbelianGroup};
    
    // Example: Integers mod n
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ModularInt {
        value: i64,
        modulus: i64,
    }
    
    impl ModularInt {
        fn new(value: i64, modulus: i64) -> Self {
            Self {
                value: value.rem_euclid(modulus),
                modulus,
            }
        }
    }
    
    impl Group for ModularInt {
        fn op(&self, other: &Self) -> Self {
            assert_eq!(self.modulus, other.modulus);
            Self::new(self.value + other.value, self.modulus)
        }
        
        fn identity() -> Self {
            Self::new(0, 96) // Default modulus
        }
        
        fn inverse(&self) -> Self {
            Self::new(-self.value, self.modulus)
        }
    }
    
    impl AbelianGroup for ModularInt {}
    
    impl Ring for ModularInt {
        fn mul(&self, other: &Self) -> Self {
            assert_eq!(self.modulus, other.modulus);
            Self::new(self.value * other.value, self.modulus)
        }
        
        fn one() -> Self {
            Self::new(1, 96)
        }
    }
    
    impl CommutativeRing for ModularInt {}
    
    #[test]
    fn test_ring_axioms() {
        let a = ModularInt::new(5, 96);
        let b = ModularInt::new(7, 96);
        let c = ModularInt::new(11, 96);
        
        // Verify all ring axioms
        assert!(a.verify_ring_axioms(&b, &c));
        
        // Verify commutativity
        assert!(a.verify_mul_commutativity(&b));
    }
}
