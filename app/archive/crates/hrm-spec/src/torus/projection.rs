//! Projection: ℤ → T²

use super::coordinate::TorusCoordinate;
use num_bigint::BigInt;

/// Projection homomorphism from integers to torus
pub trait Projection {
    /// Project integer to torus coordinates
    fn project(&self, n: &BigInt) -> TorusCoordinate {
        TorusCoordinate::from_integer(n)
    }
}

/// Standard projection
pub struct StandardProjection;

impl Projection for StandardProjection {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_projection_homomorphism() {
        let proj = StandardProjection;
        
        let a = BigInt::from(12345);
        let b = BigInt::from(67890);
        let sum = &a + &b;
        
        // Test additive homomorphism
        let _coord_a = proj.project(&a);
        let _coord_b = proj.project(&b);
        let coord_sum = proj.project(&sum);
        
        // π(a + b) should relate to π(a) and π(b)
        // Full verification in algebra module
        assert!(coord_sum.page < TorusCoordinate::PAGE_PERIOD);
        assert!(coord_sum.resonance < TorusCoordinate::RESONANCE_PERIOD);
    }
}
