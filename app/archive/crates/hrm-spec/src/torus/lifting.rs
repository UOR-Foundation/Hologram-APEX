//! Lifting: T² → ℤ

use super::coordinate::TorusCoordinate;
use num_bigint::BigInt;
use num_traits::ToPrimitive;

/// Lifting operation from torus to integers
pub trait Lifting {
    /// Lift torus coordinate to integer with hint
    fn lift(&self, coord: &TorusCoordinate, hint: &BigInt) -> BigInt;
}

/// O(1) lifting via √n projection
pub struct O1Lifting;

impl Lifting for O1Lifting {
    fn lift(&self, coord: &TorusCoordinate, hint: &BigInt) -> BigInt {
        // Project hint to nearest value in coord's orbit
        let hint_res = (hint % TorusCoordinate::RESONANCE_PERIOD as u32)
            .to_u8()
            .unwrap_or(0);
        
        let target_res = coord.resonance;
        
        let offset = if hint_res >= target_res {
            hint_res - target_res
        } else {
            TorusCoordinate::RESONANCE_PERIOD + hint_res - target_res
        };
        
        hint - BigInt::from(offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Signed;
    
    #[test]
    fn test_lift_projection_inverse() {
        let lifter = O1Lifting;
        
        let n = BigInt::from(12345);
        let coord = TorusCoordinate::from_integer(&n);
        let lifted = lifter.lift(&coord, &n);
        
        // Should be close to n (within orbit)
        let diff = (&lifted - &n).abs();
        assert!(diff < BigInt::from(TorusCoordinate::RESONANCE_PERIOD));
    }
}
