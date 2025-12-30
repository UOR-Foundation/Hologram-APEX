//! SplitMix64 PRNG for deterministic vector generation
//!
//! This module implements the SplitMix64 pseudo-random number generator,
//! which is used to deterministically generate the 96 canonical Atlas vectors.
//!
//! # Algorithm
//!
//! SplitMix64 is a fast, high-quality PRNG with 64-bit state. For each base-96
//! digit `d` (from 0 to 95), we seed a SplitMix64 instance with `d` and use it
//! to generate 196,884 uniformly distributed f64 values in [0, 1).
//!
//! # Reference
//!
//! Steele, Guy L., Doug Lea, and Christine H. Flood. "Fast splittable
//! pseudorandom number generators." ACM SIGPLAN Notices 49.10 (2014): 453-472.

/// SplitMix64 PRNG state
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a new SplitMix64 PRNG seeded with the given value
    ///
    /// For Atlas generation, this is seeded with the base-96 digit (0-95).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut rng = SplitMix64::new(42);
    /// let value = rng.next_u64();
    /// ```
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate the next u64 value
    ///
    /// This implements the SplitMix64 state transition and output function.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Generate a uniformly distributed f64 in [0, 1)
    ///
    /// This converts the u64 output to a normalized f64 value.
    ///
    /// # Implementation
    ///
    /// We use the upper 53 bits of the u64 to create a uniform f64 in [0, 1).
    /// This provides full precision without bias.
    pub fn next_f64(&mut self) -> f64 {
        let value = self.next_u64();
        // Use upper 53 bits for uniform [0, 1) distribution
        let mantissa = value >> 11;
        (mantissa as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Generate a vector of n uniformly distributed f64 values in [0, 1)
    ///
    /// This is the core function used for generating Atlas vectors.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut rng = SplitMix64::new(0);
    /// let vector = rng.gen_f64_vec(196_884);
    /// assert_eq!(vector.len(), 196_884);
    /// ```
    pub fn gen_f64_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next_f64()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splitmix64_deterministic() {
        let mut rng1 = SplitMix64::new(42);
        let mut rng2 = SplitMix64::new(42);

        for _ in 0..1000 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_splitmix64_different_seeds() {
        let mut rng1 = SplitMix64::new(0);
        let mut rng2 = SplitMix64::new(1);

        // Different seeds should produce different sequences
        let v1 = rng1.next_u64();
        let v2 = rng2.next_u64();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_gen_f64_range() {
        let mut rng = SplitMix64::new(123);

        for _ in 0..1000 {
            let value = rng.next_f64();
            assert!((0.0..1.0).contains(&value), "Value {} out of range [0, 1)", value);
        }
    }

    #[test]
    fn test_gen_f64_vec() {
        let mut rng = SplitMix64::new(0);
        let vec = rng.gen_f64_vec(1000);

        assert_eq!(vec.len(), 1000);
        for &value in &vec {
            assert!((0.0..1.0).contains(&value));
        }
    }

    #[test]
    fn test_reproducible_sequences() {
        // Same seed should produce same sequence
        let mut rng1 = SplitMix64::new(42);
        let vec1 = rng1.gen_f64_vec(100);

        let mut rng2 = SplitMix64::new(42);
        let vec2 = rng2.gen_f64_vec(100);

        assert_eq!(vec1, vec2);
    }
}
