//! Scalar Multiplication (⊙)
//!
//! Defines scalar multiplication on torus coordinates:
//! k ⊙ (page, res) = k copies of addition

use crate::torus::coordinate::TorusCoordinate;
use crate::algebra::addition::add;
use crate::foundation::group::Group;
use num_bigint::BigInt;
use num_traits::{Zero, ToPrimitive, Signed};

/// Scalar multiplication via repeated addition
pub fn scalar_mul(k: &BigInt, coord: &TorusCoordinate) -> TorusCoordinate {
    if k.is_zero() {
        return TorusCoordinate::zero();
    }
    
    let mut result = TorusCoordinate::zero();
    let k_abs = k.abs();
    
    // Compute k_abs ⊙ coord via repeated addition
    // Convert to u64 for iteration (fallback to optimized if too large)
    if let Some(k_u64) = k_abs.to_u64() {
        for _ in 0..k_u64 {
            result = add(&result, coord);
        }
    } else {
        // For very large k, use optimized version
        return scalar_mul_optimized(k, coord);
    }
    
    // If k was negative, take inverse
    if k.is_negative() {
        result = result.inverse();
    }
    
    result
}

/// Optimized scalar multiplication using modular reduction
pub fn scalar_mul_optimized(k: &BigInt, coord: &TorusCoordinate) -> TorusCoordinate {
    // k ⊙ (page, res) = (k × page mod 48, k × res mod 96)
    let k_mod_page = (k % TorusCoordinate::PAGE_PERIOD as u32).to_u8().unwrap_or(0);
    let k_mod_res = (k % TorusCoordinate::RESONANCE_PERIOD as u32).to_u8().unwrap_or(0);
    
    TorusCoordinate {
        page: ((k_mod_page as u16 * coord.page as u16) % TorusCoordinate::PAGE_PERIOD as u16) as u8,
        resonance: ((k_mod_res as u16 * coord.resonance as u16) % TorusCoordinate::RESONANCE_PERIOD as u16) as u8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_zero() {
        let coord = TorusCoordinate { page: 5, resonance: 10 };
        let result = scalar_mul(&BigInt::from(0), &coord);
        assert_eq!(result.page, 0);
        assert_eq!(result.resonance, 0);
    }

    #[test]
    fn test_scalar_one() {
        let coord = TorusCoordinate { page: 5, resonance: 10 };
        let result = scalar_mul(&BigInt::from(1), &coord);
        assert_eq!(result.page, coord.page);
        assert_eq!(result.resonance, coord.resonance);
    }

    #[test]
    fn test_scalar_multiplication() {
        let coord = TorusCoordinate { page: 3, resonance: 5 };
        let result = scalar_mul(&BigInt::from(4), &coord);
        assert_eq!(result.page, 12);  // 4 × 3 = 12
        assert_eq!(result.resonance, 20);  // 4 × 5 = 20
    }

    #[test]
    fn test_scalar_multiplication_wraparound() {
        let coord = TorusCoordinate { page: 20, resonance: 40 };
        let result = scalar_mul(&BigInt::from(3), &coord);
        assert_eq!(result.page, 12);  // (3 × 20) % 48 = 60 % 48 = 12
        assert_eq!(result.resonance, 24);  // (3 × 40) % 96 = 120 % 96 = 24
    }

    #[test]
    fn test_scalar_optimized_matches_naive() {
        let coord = TorusCoordinate { page: 7, resonance: 13 };
        let k = BigInt::from(11);
        
        let naive = scalar_mul(&k, &coord);
        let optimized = scalar_mul_optimized(&k, &coord);
        
        assert_eq!(naive.page, optimized.page);
        assert_eq!(naive.resonance, optimized.resonance);
    }
}
