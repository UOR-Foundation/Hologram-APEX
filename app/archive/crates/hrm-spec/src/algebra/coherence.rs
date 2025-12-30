//! Coherence Verification
//!
//! Verifies homomorphism properties:
//! - π(a + b) = π(a) ⊕ π(b)  (addition coherence)
//! - π(a × b) = π(a) ⊗ π(b)  (multiplication coherence)
//! - π(k × a) = k ⊙ π(a)     (scalar coherence)

use crate::torus::projection::{Projection, StandardProjection};
use crate::algebra::addition::add;
use crate::algebra::multiplication::mul;
use crate::algebra::scalar::scalar_mul_optimized;
use num_bigint::BigInt;

/// Coherence verifier
pub struct CoherenceVerifier {
    projection: StandardProjection,
}

impl CoherenceVerifier {
    pub fn new() -> Self {
        Self {
            projection: StandardProjection,
        }
    }

    /// Verify addition coherence: π(a + b) = π(a) ⊕ π(b)
    pub fn verify_addition_coherence(&self, a: &BigInt, b: &BigInt) -> bool {
        let sum = a + b;
        let projected_sum = self.projection.project(&sum);
        
        let projected_a = self.projection.project(a);
        let projected_b = self.projection.project(b);
        let added_projections = add(&projected_a, &projected_b);
        
        projected_sum.page == added_projections.page &&
        projected_sum.resonance == added_projections.resonance
    }

    /// Verify multiplication coherence: π(a × b) = π(a) ⊗ π(b)
    pub fn verify_multiplication_coherence(&self, a: &BigInt, b: &BigInt) -> bool {
        let product = a * b;
        let projected_product = self.projection.project(&product);
        
        let projected_a = self.projection.project(a);
        let projected_b = self.projection.project(b);
        let multiplied_projections = mul(&projected_a, &projected_b);
        
        projected_product.page == multiplied_projections.page &&
        projected_product.resonance == multiplied_projections.resonance
    }

    /// Verify scalar coherence: π(k × a) = k ⊙ π(a)
    pub fn verify_scalar_coherence(&self, k: &BigInt, a: &BigInt) -> bool {
        let product = k * a;
        let projected_product = self.projection.project(&product);
        
        let projected_a = self.projection.project(a);
        let scalar_multiplied = scalar_mul_optimized(k, &projected_a);
        
        projected_product.page == scalar_multiplied.page &&
        projected_product.resonance == scalar_multiplied.resonance
    }
}

impl Default for CoherenceVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition_coherence() {
        let verifier = CoherenceVerifier::new();
        
        let a = BigInt::from(37);
        let b = BigInt::from(53);
        
        assert!(verifier.verify_addition_coherence(&a, &b));
    }

    #[test]
    fn test_multiplication_coherence() {
        let verifier = CoherenceVerifier::new();
        
        // Test factorization routing constraint: 3 × 5 = 15
        let a = BigInt::from(3);
        let b = BigInt::from(5);
        
        assert!(verifier.verify_multiplication_coherence(&a, &b));
    }

    #[test]
    fn test_multiplication_coherence_large() {
        let verifier = CoherenceVerifier::new();
        
        // Test factorization routing constraint: 7 × 11 = 77
        let a = BigInt::from(7);
        let b = BigInt::from(11);
        
        assert!(verifier.verify_multiplication_coherence(&a, &b));
    }

    #[test]
    fn test_scalar_coherence() {
        let verifier = CoherenceVerifier::new();
        
        let k = BigInt::from(7);
        let a = BigInt::from(13);
        
        assert!(verifier.verify_scalar_coherence(&k, &a));
    }

    #[test]
    fn test_addition_coherence_large_numbers() {
        let verifier = CoherenceVerifier::new();
        
        let a = BigInt::from(123456789_u64);
        let b = BigInt::from(987654321_u64);
        
        assert!(verifier.verify_addition_coherence(&a, &b));
    }

    #[test]
    fn test_multiplication_coherence_large_numbers() {
        let verifier = CoherenceVerifier::new();
        
        let a = BigInt::from(12345_u64);
        let b = BigInt::from(67890_u64);
        
        assert!(verifier.verify_multiplication_coherence(&a, &b));
    }
}
