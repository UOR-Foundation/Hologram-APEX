//! Routing Protocol
//!
//! Defines the fundamental routing protocol through Monster representation space.
//! Key insight: Multiplication routing is MULTIPLICATIVE tensor product.

use crate::torus::coordinate::TorusCoordinate;
use crate::algebra::addition::add;
use crate::algebra::multiplication::mul;
use num_bigint::BigInt;

/// Routing protocol for operations
pub trait RoutingProtocol {
    /// Route addition through entanglement network
    fn route_addition(&self, a: &TorusCoordinate, b: &TorusCoordinate) -> TorusCoordinate;
    
    /// Route multiplication through entanglement network
    fn route_multiplication(&self, a: &TorusCoordinate, b: &TorusCoordinate) -> TorusCoordinate;
}

/// Standard routing via algebraic generators
pub struct StandardRouting;

impl RoutingProtocol for StandardRouting {
    fn route_addition(&self, a: &TorusCoordinate, b: &TorusCoordinate) -> TorusCoordinate {
        // Addition routes via ⊕
        add(a, b)
    }
    
    fn route_multiplication(&self, a: &TorusCoordinate, b: &TorusCoordinate) -> TorusCoordinate {
        // Multiplication routes via ⊗ (tensor product of channels)
        mul(a, b)
    }
}

/// Verify routing coherence
pub fn verify_routing_coherence(n_a: &BigInt, n_b: &BigInt) -> bool {
    use crate::algebra::coherence::CoherenceVerifier;
    
    let verifier = CoherenceVerifier::new();
    
    // Verify both addition and multiplication coherence
    verifier.verify_addition_coherence(n_a, n_b) &&
    verifier.verify_multiplication_coherence(n_a, n_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_routing_addition() {
        let routing = StandardRouting;
        let a = TorusCoordinate { page: 3, resonance: 5 };
        let b = TorusCoordinate { page: 7, resonance: 11 };
        
        let result = routing.route_addition(&a, &b);
        assert_eq!(result.page, 10);
        assert_eq!(result.resonance, 16);
    }

    #[test]
    fn test_standard_routing_multiplication() {
        let routing = StandardRouting;
        let a = TorusCoordinate { page: 3, resonance: 5 };
        let b = TorusCoordinate { page: 7, resonance: 11 };
        
        let result = routing.route_multiplication(&a, &b);
        assert_eq!(result.page, 21);
        assert_eq!(result.resonance, 55);
    }

    #[test]
    fn test_routing_coherence() {
        let a = BigInt::from(3);
        let b = BigInt::from(5);
        
        assert!(verify_routing_coherence(&a, &b));
    }

    #[test]
    fn test_factorization_routing() {
        // 7 × 11 = 77: page constraint is multiplicative
        let a = BigInt::from(7);
        let b = BigInt::from(11);
        
        assert!(verify_routing_coherence(&a, &b));
    }
}
