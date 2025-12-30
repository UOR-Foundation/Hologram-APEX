//! Harmonic Weights
//!
//! Defines weights for primes based on zeta-calibrated norms.
//! Implementation deferred to Phase 2.

use crate::foundation::exactmath::Exact;
use num_bigint::BigInt;

/// Harmonic weight for an integer
pub struct HarmonicWeight {
    /// The integer
    pub n: BigInt,
    /// Its harmonic weight
    pub weight: Exact,
}

/// Compute harmonic weight for integer
pub fn compute_weight(n: &BigInt) -> Exact {
    // Placeholder: weight = 1/n for now
    Exact::Rational(num_rational::BigRational::new(
        num_bigint::BigInt::from(1),
        n.clone()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_weight() {
        let n = BigInt::from(5);
        let weight = compute_weight(&n);
        // Verify weight is rational
        match weight {
            Exact::Rational(_) => {},
            _ => panic!("Expected rational weight"),
        }
    }
}
