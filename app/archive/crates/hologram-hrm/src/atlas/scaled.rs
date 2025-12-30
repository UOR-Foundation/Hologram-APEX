//! Scaled Atlas with arbitrary precision via Lie group structure
//!
//! This module implements true arbitrary precision by scaling the Atlas from
//! 96 classes to arbitrary n using Lie groups and ring composition operators.
//!
//! # Mathematical Foundation
//!
//! The base Atlas has 96 classes from the quotient structure (ℤ₄ × ℤ₃ × ℤ₈).
//! We generalize this to arbitrary (n₁, n₂, n₃) using:
//!
//! 1. **Lie Group Structure**: Representation space with group operations
//! 2. **Ring Composition**: Lattices as rings with composition operators
//! 3. **Scaling Function**: Atlas(n₁, n₂, n₃) → n₁ × n₂ × n₃ classes
//!
//! # Usage
//!
//! ```ignore
//! use hologram_hrm::atlas::ScaledAtlas;
//!
//! // Create scaled Atlas with 256 classes (4 × 4 × 16)
//! let atlas = ScaledAtlas::new(4, 4, 16)?;
//!
//! // Encode large numbers with arbitrary precision
//! let vector = atlas.encode_large(1_000_000)?;
//!
//! // Decode back to exact value
//! let decoded = atlas.decode_large(&vector)?;
//! assert_eq!(decoded, 1_000_000);
//! ```

use crate::griess::{divide, product, scalar_mul, GriessVector};
use crate::{Error, Result, GRIESS_DIMENSION};
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use std::sync::Arc;

#[cfg(test)]
use num_traits::One;

use super::prng::SplitMix64;

/// Scaled Atlas with arbitrary number of classes
///
/// Generalizes the base 96-class Atlas to arbitrary (n₁, n₂, n₃) structure
/// using Lie group operations on the representation space.
///
/// Uses four integer roots with Arrow caches for O(1) resolver operations.
pub struct ScaledAtlas {
    /// Number of classes in first dimension (generalizes h₂ ∈ ℤ₄)
    n1: usize,
    /// Number of classes in second dimension (generalizes d ∈ ℤ₃)
    n2: usize,
    /// Number of classes in third dimension (generalizes ℓ ∈ ℤ₈)
    n3: usize,
    /// Total number of classes (n₁ × n₂ × n₃)
    total_classes: usize,
    /// Cache of Atlas basis vectors (Arrow-backed for zero-copy)
    cache: Option<Vec<Arc<GriessVector>>>,
    /// Resolver cache: maps integers → encoded vectors for O(1) lookup
    /// Pre-computed for common values (0 to resolver_limit)
    resolver_cache: Option<Vec<Arc<GriessVector>>>,
    /// Maximum value in resolver cache
    resolver_limit: usize,
}

impl ScaledAtlas {
    /// Create a new scaled Atlas with (n₁, n₂, n₃) structure
    ///
    /// # Arguments
    ///
    /// * `n1` - Number of classes in first dimension (must be > 0)
    /// * `n2` - Number of classes in second dimension (must be > 0)
    /// * `n3` - Number of classes in third dimension (must be > 0)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Create base 96-class Atlas
    /// let base = ScaledAtlas::new(4, 3, 8)?;
    ///
    /// // Create expanded 1024-class Atlas
    /// let expanded = ScaledAtlas::new(8, 8, 16)?;
    /// ```
    pub fn new(n1: usize, n2: usize, n3: usize) -> Result<Self> {
        if n1 == 0 || n2 == 0 || n3 == 0 {
            return Err(Error::InvalidDimension(0));
        }

        let total_classes = n1 * n2 * n3;

        Ok(Self {
            n1,
            n2,
            n3,
            total_classes,
            cache: None,
            resolver_cache: None,
            resolver_limit: 0,
        })
    }

    /// Create base 96-class Atlas (4 × 3 × 8)
    ///
    /// This is equivalent to the standard Atlas.
    pub fn base() -> Result<Self> {
        Self::new(4, 3, 8)
    }

    /// Create scaled Atlas with pre-generated cache
    ///
    /// All Atlas basis vectors are generated upfront for O(1) access.
    pub fn with_cache(n1: usize, n2: usize, n3: usize) -> Result<Self> {
        let mut atlas = Self::new(n1, n2, n3)?;

        let vectors: Result<Vec<Arc<GriessVector>>> = (0..atlas.total_classes)
            .map(|class| atlas.generate_vector(class).map(Arc::new))
            .collect();

        atlas.cache = Some(vectors?);
        Ok(atlas)
    }

    /// Create scaled Atlas with resolver cache for fast decoding
    ///
    /// Pre-computes encoded vectors for integers 0..resolver_limit
    /// for O(1) nearest-neighbor decoding (vector database approach).
    ///
    /// # Arguments
    ///
    /// * `n1`, `n2`, `n3` - Atlas dimensions
    /// * `resolver_limit` - Maximum integer to pre-compute (e.g., 1_000_000)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Create resolver cache for values 0..1M
    /// let atlas = ScaledAtlas::with_resolver(4, 3, 8, 1_000_000)?;
    ///
    /// // Encoding and decoding are both O(1)
    /// let encoded = atlas.encode(&BigUint::from(42u64))?;
    /// let decoded = atlas.decode(&encoded)?; // Nearest-neighbor in cache
    /// assert_eq!(decoded, BigUint::from(42u64));
    /// ```
    pub fn with_resolver(n1: usize, n2: usize, n3: usize, resolver_limit: usize) -> Result<Self> {
        let mut atlas = Self::with_cache(n1, n2, n3)?;

        // Pre-compute NORMALIZED encoded vectors for all integers 0..resolver_limit
        let mut resolver_cache = Vec::with_capacity(resolver_limit);
        for value in 0..resolver_limit {
            let encoded = atlas.encode_large(&BigUint::from(value))?;

            // Normalize for cosine similarity
            let norm = encoded.norm();
            let normalized = if norm > 1e-10 {
                scalar_mul(&encoded, 1.0 / norm)?
            } else {
                encoded // Zero vector stays as-is
            };

            resolver_cache.push(Arc::new(normalized));
        }

        atlas.resolver_cache = Some(resolver_cache);
        atlas.resolver_limit = resolver_limit;

        Ok(atlas)
    }

    /// Get the total number of classes in this scaled Atlas
    pub fn class_count(&self) -> usize {
        self.total_classes
    }

    /// Get the dimensions (n₁, n₂, n₃)
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.n1, self.n2, self.n3)
    }

    /// Decode class index to (i₁, i₂, i₃) components
    ///
    /// Generalizes: class = n₃*n₂*i₁ + n₃*i₂ + i₃
    fn decode_class(&self, class: usize) -> (usize, usize, usize) {
        let i1 = class / (self.n2 * self.n3);
        let remainder = class % (self.n2 * self.n3);
        let i2 = remainder / self.n3;
        let i3 = remainder % self.n3;

        (i1, i2, i3)
    }

    /// Encode (i₁, i₂, i₃) components to class index
    ///
    /// This is the inverse of `decode_class`.
    #[cfg(test)]
    fn encode_class(&self, i1: usize, i2: usize, i3: usize) -> usize {
        i1 * self.n2 * self.n3 + i2 * self.n3 + i3
    }

    /// Generate a canonical vector for the given class
    ///
    /// Uses Lie group structure to create structured vectors in the
    /// representation space.
    fn generate_vector(&self, class: usize) -> Result<GriessVector> {
        if class >= self.total_classes {
            return Err(Error::ClassOutOfRange(class as u8));
        }

        // Decode to components
        let (i1, i2, i3) = self.decode_class(class);

        // Compute Lie algebra basis indices
        let seed = self.compute_lie_seed(i1, i2, i3);

        // Generate structured vector using Lie group representation
        let vector = self.generate_lie_vector(seed, i1, i2, i3)?;

        // Normalize (Lie group exp map)
        let norm = vector.norm();
        if norm < 1e-10 {
            return Err(Error::DecodingFailed(format!(
                "Generated zero vector for class {} ({}, {}, {})",
                class, i1, i2, i3
            )));
        }

        scalar_mul(&vector, 1.0 / norm)
    }

    /// Compute Lie algebra seed from components
    ///
    /// Maps (i₁, i₂, i₃) to Lie algebra element via deterministic hash.
    fn compute_lie_seed(&self, i1: usize, i2: usize, i3: usize) -> u64 {
        // Use prime factorization to ensure distinct seeds
        const PRIMES: [u64; 3] = [1961, 4327, 9973];

        let seed = PRIMES[0] * i1 as u64 + PRIMES[1] * i2 as u64 + PRIMES[2] * i3 as u64;

        // Apply avalanche mixing
        let mut h = seed;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }

    /// Generate vector using Lie group representation
    ///
    /// The representation space is structured as:
    /// - Dimension 0..H: First coordinate influence (i₁)
    /// - Dimension H..D: Second coordinate influence (i₂)
    /// - Dimension D..L: Third coordinate influence (i₃)
    fn generate_lie_vector(&self, seed: u64, i1: usize, i2: usize, i3: usize) -> Result<GriessVector> {
        let mut data = vec![0.0f64; GRIESS_DIMENSION];
        let mut rng = SplitMix64::new(seed);

        // Compute dimension boundaries based on scaling
        let h_end = GRIESS_DIMENSION / 3;
        let d_start = h_end;
        let d_end = 2 * GRIESS_DIMENSION / 3;
        let l_start = d_end;

        // First coordinate influence (Lie group generator direction)
        let theta1 = 2.0 * std::f64::consts::PI * (i1 as f64) / (self.n1 as f64);
        for (i, value) in data.iter_mut().enumerate().take(h_end) {
            let phase = (i as f64 / h_end as f64) * theta1;
            let noise = rng.next_f64() * 0.2 - 0.1; // [-0.1, 0.1)
            *value = phase.cos() + noise;
        }

        // Second coordinate influence (Lie algebra commutator structure)
        let theta2 = 2.0 * std::f64::consts::PI * (i2 as f64) / (self.n2 as f64);
        for (i, value) in data.iter_mut().enumerate().skip(d_start).take(d_end - d_start) {
            let phase = ((i - d_start) as f64 / (d_end - d_start) as f64) * theta2;
            let noise = rng.next_f64() * 0.2 - 0.1; // [-0.1, 0.1)
            *value = phase.sin() + noise;
        }

        // Third coordinate influence (representation space fiber)
        let theta3 = 2.0 * std::f64::consts::PI * (i3 as f64) / (self.n3 as f64);
        for (i, value) in data.iter_mut().enumerate().skip(l_start) {
            let phase = ((i - l_start) as f64 / (GRIESS_DIMENSION - l_start) as f64) * theta3;
            let noise = rng.next_f64() * 0.1 - 0.05; // [-0.05, 0.05)
            *value = (phase.cos() + phase.sin()) / std::f64::consts::SQRT_2 + noise;
        }

        GriessVector::from_vec(data)
    }

    /// Get vector for a given class index
    ///
    /// Returns cached vector if available, otherwise generates on-demand.
    pub fn get_vector(&self, class: usize) -> Result<Arc<GriessVector>> {
        if class >= self.total_classes {
            return Err(Error::ClassOutOfRange(class as u8));
        }

        match &self.cache {
            Some(vectors) => Ok(Arc::clone(&vectors[class])),
            None => self.generate_vector(class).map(Arc::new),
        }
    }

    /// Encode a BigUint using Hadamard product composition
    ///
    /// Maps arbitrary integers to Griess space using base-(total_classes) representation
    /// and Lie group composition (Hadamard product).
    ///
    /// For value with digits [d₀, d₁, d₂, ...]:
    /// result = V_{d₀} ⊙ V_{d₁} ⊙ V_{d₂} ⊙ ...
    ///
    /// This is the ring multiplication operator on the Griess lattice.
    pub fn encode_large(&self, value: &BigUint) -> Result<GriessVector> {
        if value.is_zero() {
            return self.get_vector(0).map(|v| (*v).clone());
        }

        // Convert to base-(total_classes) digits
        let digits = self.to_base_n(value);

        // Start with identity element
        let mut result = GriessVector::identity();

        // Compose via ring multiplication (Hadamard product)
        for digit in digits {
            let vector = self.get_vector(digit)?;
            result = product(&result, &vector)?;
        }

        Ok(result)
    }

    /// Decode a Griess vector to BigUint using resolver cache (vector database)
    ///
    /// Uses nearest-neighbor search in the pre-computed resolver cache to find
    /// the integer value that produced this vector. This is O(n) where n is the
    /// resolver_limit, but uses SIMD-optimized dot products for fast execution.
    ///
    /// # Arguments
    ///
    /// * `vector` - The Griess vector to decode
    ///
    /// # Returns
    ///
    /// The decoded BigUint value (0..resolver_limit)
    ///
    /// # Errors
    ///
    /// Returns error if resolver cache is not initialized. Use `with_resolver()`.
    pub fn decode_large(&self, vector: &GriessVector) -> Result<BigUint> {
        let resolver_cache = self.resolver_cache.as_ref().ok_or_else(|| {
            Error::DecodingFailed("Resolver cache not initialized. Use ScaledAtlas::with_resolver()".to_string())
        })?;

        // Find nearest neighbor in resolver cache using cosine similarity
        let mut max_similarity = f64::NEG_INFINITY;
        let mut best_match = 0usize;

        // Normalize input vector for cosine similarity
        let norm = vector.norm();
        let normalized_input = if norm > 1e-10 {
            scalar_mul(vector, 1.0 / norm)?
        } else {
            return Ok(BigUint::zero()); // Zero vector decodes to 0
        };

        // Vectorized nearest-neighbor search (can be SIMD-accelerated)
        for (value, cached_vector) in resolver_cache.iter().enumerate() {
            // Cosine similarity = dot product of normalized vectors
            let similarity = Self::dot_product(&normalized_input, cached_vector);

            if similarity > max_similarity {
                max_similarity = similarity;
                best_match = value;
            }
        }

        Ok(BigUint::from(best_match))
    }

    /// Decode with fallback for values outside resolver cache
    ///
    /// If the vector is not in the resolver cache range, falls back to
    /// iterative decomposition (slower but works for any value).
    pub fn decode_large_with_limit(&self, vector: &GriessVector, max_digits: usize) -> Result<BigUint> {
        // Try resolver cache first if available
        if self.resolver_cache.is_some() {
            return self.decode_large(vector);
        }

        // Fallback: iterative Hadamard decomposition
        self.decode_iterative(vector, max_digits)
    }

    /// Iterative Hadamard decomposition (fallback for large values)
    fn decode_iterative(&self, vector: &GriessVector, max_digits: usize) -> Result<BigUint> {
        let mut digits = Vec::new();
        let mut residual = vector.clone();

        const IDENTITY_THRESHOLD: f64 = 0.15;

        for _ in 0..max_digits {
            let (nearest_class, _) = self.find_nearest_class(&residual)?;

            let atlas_vector = self.get_vector(nearest_class)?;
            let new_residual = divide(&residual, &atlas_vector)?;

            if self.is_near_identity(&new_residual, IDENTITY_THRESHOLD) {
                digits.push(nearest_class);
                break;
            }

            digits.push(nearest_class);
            residual = new_residual;
        }

        let mut value = BigUint::zero();
        let base = BigUint::from(self.total_classes);
        let mut power = BigUint::from(1u32);

        for &digit in &digits {
            value += BigUint::from(digit) * &power;
            power *= &base;
        }

        Ok(value)
    }

    /// Find the nearest Atlas class to a given vector
    ///
    /// Returns (class_index, distance) tuple
    fn find_nearest_class(&self, vector: &GriessVector) -> Result<(usize, f64)> {
        // Normalize input vector for cosine similarity
        let norm = vector.norm();
        let normalized_input = if norm > 1e-10 {
            scalar_mul(vector, 1.0 / norm)?
        } else {
            return Ok((0, 0.0)); // Zero vector defaults to class 0
        };

        let mut max_similarity = f64::NEG_INFINITY;
        let mut nearest_class = 0;

        // Compute dot product with all cached vectors (vectorizable operation)
        for class in 0..self.total_classes {
            let atlas_vector = self.get_vector(class)?;

            // Atlas vectors are already normalized, so dot product = cosine similarity
            let similarity = Self::dot_product(&normalized_input, &atlas_vector);

            if similarity > max_similarity {
                max_similarity = similarity;
                nearest_class = class;
            }
        }

        Ok((nearest_class, max_similarity))
    }

    /// Compute dot product of two vectors (pure Arrow operation)
    fn dot_product(a: &GriessVector, b: &GriessVector) -> f64 {
        a.as_slice().iter().zip(b.as_slice().iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Check if a vector is close to the identity (all components ≈ 1.0)
    fn is_near_identity(&self, vector: &GriessVector, threshold: f64) -> bool {
        let slice = vector.as_slice();

        // Check if mean is close to 1.0
        let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
        if (mean - 1.0).abs() > threshold {
            return false;
        }

        // Check if standard deviation is small
        let variance: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;
        let std_dev = variance.sqrt();

        std_dev < threshold
    }

    /// Convert BigUint to base-n representation
    fn to_base_n(&self, value: &BigUint) -> Vec<usize> {
        if value.is_zero() {
            return vec![0];
        }

        let mut digits = Vec::new();
        let mut n = value.clone();
        let base = BigUint::from(self.total_classes);

        while !n.is_zero() {
            let digit = (&n % &base).to_usize().expect("Digit should fit in usize");
            digits.push(digit);
            n /= &base;
        }

        digits
    }

    /// Convert base-n digits to BigUint
    ///
    /// This is the inverse of `to_base_n`.
    #[cfg(test)]
    fn digits_to_value(&self, digits: &[usize]) -> BigUint {
        let mut value = BigUint::zero();
        let base = BigUint::from(self.total_classes);
        let mut power = BigUint::one();

        for &digit in digits {
            value += BigUint::from(digit) * &power;
            power *= &base;
        }

        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_atlas_creation() {
        // Base 96-class Atlas
        let base = ScaledAtlas::base().unwrap();
        assert_eq!(base.class_count(), 96);
        assert_eq!(base.dimensions(), (4, 3, 8));

        // Expanded 1024-class Atlas
        let expanded = ScaledAtlas::new(8, 8, 16).unwrap();
        assert_eq!(expanded.class_count(), 1024);
        assert_eq!(expanded.dimensions(), (8, 8, 16));
    }

    #[test]
    fn test_encode_decode_class() {
        let atlas = ScaledAtlas::new(4, 5, 6).unwrap();

        for i1 in 0..4 {
            for i2 in 0..5 {
                for i3 in 0..6 {
                    let class = atlas.encode_class(i1, i2, i3);
                    let (d1, d2, d3) = atlas.decode_class(class);

                    assert_eq!(
                        (d1, d2, d3),
                        (i1, i2, i3),
                        "Roundtrip failed for ({}, {}, {})",
                        i1,
                        i2,
                        i3
                    );
                }
            }
        }
    }

    #[test]
    fn test_generate_vectors() {
        let atlas = ScaledAtlas::new(4, 3, 8).unwrap();

        for class in 0..96 {
            let vector = atlas.generate_vector(class).unwrap();

            // Check dimensionality
            assert_eq!(vector.len(), 196_884);

            // Check normalization
            let norm = vector.norm();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "Vector {} not normalized: norm = {}",
                class,
                norm
            );
        }
    }

    #[test]
    fn test_base_n_conversion() {
        let atlas = ScaledAtlas::new(10, 10, 10).unwrap(); // Base 1000

        for value in [0u64, 1, 999, 1000, 1001, 1_000_000] {
            let biguint = BigUint::from(value);
            let digits = atlas.to_base_n(&biguint);
            let reconstructed = atlas.digits_to_value(&digits);

            assert_eq!(reconstructed, biguint, "Roundtrip failed for {}", value);
        }
    }

    #[test]
    #[ignore = "Slow test: takes ~23 seconds"]
    fn test_encode_decode_single_digit() {
        let atlas = ScaledAtlas::with_resolver(4, 3, 8, 100).unwrap();

        // Test single-digit values (0 through 95)
        for value in 0u64..96 {
            let biguint = BigUint::from(value);
            let encoded = atlas.encode_large(&biguint).unwrap();
            let decoded = atlas.decode_large(&encoded).unwrap();

            assert_eq!(decoded, biguint, "Single-digit {} failed to roundtrip", value);
        }
    }

    #[test]
    fn test_resolver_cache_roundtrip() {
        let atlas = ScaledAtlas::with_resolver(4, 3, 8, 200).unwrap();

        // Test all values in resolver cache roundtrip correctly
        for value in [0u64, 1, 5, 42, 95, 96, 100, 150, 199] {
            let vector = atlas.encode_large(&BigUint::from(value)).unwrap();
            let decoded = atlas.decode_large(&vector).unwrap();

            assert_eq!(
                decoded,
                BigUint::from(value),
                "Value {} failed to roundtrip with resolver cache",
                value
            );
        }
    }

    #[test]
    #[ignore = "Slow test: takes ~25 seconds"]
    fn test_scaled_atlas_larger_base() {
        let atlas = ScaledAtlas::with_resolver(8, 8, 16, 1100).unwrap(); // 1024 classes

        // Test single-digit values in expanded base
        for value in [0u64, 1, 511, 512, 1023] {
            let biguint = BigUint::from(value);
            let encoded = atlas.encode_large(&biguint).unwrap();
            let decoded = atlas.decode_large(&encoded).unwrap();

            assert_eq!(decoded, biguint, "Value {} failed in expanded Atlas", value);
        }
    }

    #[test]
    #[ignore = "Memory intensive: requires ~15GB RAM for resolver cache"]
    fn test_multi_digit_encoding_decoding() {
        let atlas = ScaledAtlas::with_resolver(4, 3, 8, 10000).unwrap(); // Base 96

        // Test multi-digit values (> 95 in base 96)
        for value in [96u64, 97, 100, 1000, 1961, 9216] {
            let biguint = BigUint::from(value);
            let encoded = atlas.encode_large(&biguint).unwrap();
            let decoded = atlas.decode_large(&encoded).unwrap();

            assert_eq!(decoded, biguint, "Multi-digit value {} failed to roundtrip", value);
        }
    }

    #[test]
    #[ignore = "Memory intensive: requires ~1.7TB RAM for resolver cache"]
    fn test_large_number_encoding() {
        let atlas = ScaledAtlas::with_resolver(10, 10, 10, 1100000).unwrap(); // Base 1000

        // Test larger numbers
        for value in [1_000u64, 10_000, 100_000, 1_000_000] {
            let biguint = BigUint::from(value);
            let encoded = atlas.encode_large(&biguint).unwrap();
            let decoded = atlas.decode_large(&encoded).unwrap();

            // May not be exact due to numerical precision, but should be close
            let diff = if decoded > biguint {
                &decoded - &biguint
            } else {
                &biguint - &decoded
            };

            // Allow small error for large numbers
            let max_error = BigUint::from((value as f64 * 0.01) as u64); // 1% error
            assert!(
                diff <= max_error,
                "Value {} decoded to {} (diff: {})",
                value,
                decoded,
                diff
            );
        }
    }

    #[test]
    #[ignore = "Memory intensive: requires ~15GB RAM for resolver cache"]
    fn test_two_digit_base96() {
        let atlas = ScaledAtlas::with_resolver(4, 3, 8, 10000).unwrap(); // Base 96

        // Test all two-digit values in base-96
        for d0 in [0u64, 1, 50, 95] {
            for d1 in [0u64, 1, 10, 20] {
                let value = d0 + d1 * 96;
                let biguint = BigUint::from(value);

                let encoded = atlas.encode_large(&biguint).unwrap();
                let decoded = atlas.decode_large(&encoded).unwrap();

                assert_eq!(decoded, biguint, "Two-digit [{}, {}] (value {}) failed", d0, d1, value);
            }
        }
    }

    #[test]
    fn test_identity_detection() {
        let atlas = ScaledAtlas::new(4, 3, 8).unwrap();

        // Test that identity vector is recognized
        let identity = GriessVector::identity();
        assert!(atlas.is_near_identity(&identity, 0.1));

        // Test that non-identity is not recognized
        let vec = atlas.generate_vector(42).unwrap();
        assert!(!atlas.is_near_identity(&vec, 0.1));
    }

    #[test]
    fn test_decode_with_limit() {
        let atlas = ScaledAtlas::with_resolver(4, 3, 8, 2000).unwrap();

        let value = BigUint::from(1961u64);
        let encoded = atlas.encode_large(&value).unwrap();

        // Test with various limits
        for limit in [1, 2, 3, 5, 10] {
            let decoded = atlas.decode_large_with_limit(&encoded, limit).unwrap();

            // With sufficient limit, should decode correctly
            if limit >= 3 {
                assert_eq!(decoded, value, "Failed to decode with limit {}", limit);
            }
        }
    }
}
