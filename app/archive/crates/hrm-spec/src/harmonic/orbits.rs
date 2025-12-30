//! Prime Orbits
//!
//! Classification of primes by their torus orbits under Monster action.
//! Implementation deferred to Phase 2.

use num_bigint::BigInt;

/// Prime orbit classification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimeOrbit {
    /// Prime lands in resonant orbit
    Resonant { order: usize },
    /// Prime lands in wandering orbit
    Wandering,
}

/// Classify prime by its orbit structure
pub fn classify_prime_orbit(_p: &BigInt) -> PrimeOrbit {
    // Placeholder: all primes wandering for now
    PrimeOrbit::Wandering
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_prime() {
        let p = BigInt::from(7);
        let orbit = classify_prime_orbit(&p);
        assert_eq!(orbit, PrimeOrbit::Wandering);
    }
}
