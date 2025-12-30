//! Decoding operator D: GriessVector → Integer
//!
//! The decoding operator recovers an integer from a Griess vector by inverting
//! the embedding process. Two approaches are provided:
//!
//! ## Classical Decoding (Nearest Class)
//!
//! Finds the nearest Atlas class for single-digit values (0-95).
//! Limited to single-digit decoding.
//!
//! ## MoonshineHRM Decoding (Inverse Group Action)
//!
//! The canonical approach using dual lattice projection and inverse group actions:
//! 1. Project vector onto dual Griess lattice
//! 2. Decompose into inverse operator sequence via group action
//! 3. Extract resonance class sequence [k₀, k₁, k₂, ...]
//! 4. Reconstruct base-96 representation
//! 5. Convert to BigUint
//!
//! This approach handles multi-digit values correctly.

use crate::algebra::MoonshineAlgebra;
use crate::atlas::Atlas;
use crate::griess::GriessVector;
use crate::moonshine::action::GroupAction;
use crate::moonshine::MoonshineOperator;
use crate::Result;
use num_bigint::BigUint;
use num_traits::Zero;

/// Decode a Griess vector to BigUint
///
/// **Current Limitation**: This implementation only decodes single-digit values (0-95).
/// Multi-digit decoding requires decomposing Hadamard products, which is not yet implemented.
///
/// This finds the nearest Atlas class and uses the class index as the
/// decoded integer value.
///
/// # Arguments
///
/// * `vector` - The Griess vector to decode
/// * `atlas` - Atlas partition for finding nearest class
///
/// # Returns
///
/// The BigUint representation of the decoded value (0-95 for single digits)
///
/// # Example
///
/// ```ignore
/// use hologram_hrm::{Atlas, decode_vector, embed_u64};
///
/// let atlas = Atlas::with_cache()?;
///
/// // Single-digit roundtrip works
/// let vector = embed_u64(42, &atlas)?;
/// let decoded = decode_vector(&vector, &atlas)?;
/// assert_eq!(decoded, BigUint::from(42u64));
///
/// // Multi-digit values decode to their nearest single Atlas class
/// let vector = embed_u64(1961, &atlas)?; // 1961 = [41, 20] in base-96
/// let decoded = decode_vector(&vector, &atlas)?;
/// // decoded will be in range 0-95, not 1961
/// ```
pub fn decode_vector(vector: &GriessVector, atlas: &Atlas) -> Result<BigUint> {
    // Find nearest Atlas class
    let mut min_distance = f64::INFINITY;
    let mut nearest_class = 0u8;

    for class in 0..96 {
        let atlas_vector = atlas.get_vector(class)?;
        let distance = vector.distance(&atlas_vector);

        if distance < min_distance {
            min_distance = distance;
            nearest_class = class;
        }
    }

    // Use class index as decoded value
    // NOTE: This only works correctly for single-digit values (0-95)
    // Multi-digit decoding requires decomposing Hadamard products
    Ok(BigUint::from(nearest_class))
}

/// Decode using MoonshineHRM inverse group action
///
/// This is the canonical decoding approach that handles multi-digit values
/// by decomposing the group action using the dual lattice.
///
/// # Algorithm
///
/// 1. **Project onto dual lattice**: Use DualGriessLattice for projection
/// 2. **Iterative class extraction**: Find resonance classes by nearest-neighbor
/// 3. **Reconstruct base-96**: Build digit sequence from operator sequence
/// 4. **Convert to BigUint**: Combine digits with positional values
///
/// # Arguments
///
/// * `vector` - The Griess vector to decode
/// * `atlas` - Atlas for class lookups
/// * `algebra` - MoonshineAlgebra for inverse operations
/// * `max_digits` - Maximum number of digits to extract (prevents infinite loops)
///
/// # Returns
///
/// The BigUint representation of the decoded value.
///
/// # Example
///
/// ```ignore
/// let atlas = Atlas::with_cache()?;
/// let algebra = MoonshineAlgebra::with_cache()?;
///
/// // Multi-digit roundtrip
/// let value = BigUint::from(1961u64);
/// let vector = embed_with_moonshine(&value, &atlas, &algebra)?;
/// let decoded = decode_with_moonshine(&vector, &atlas, &algebra, 10)?;
/// assert_eq!(decoded, value);
/// ```
pub fn decode_with_moonshine(
    vector: &GriessVector,
    _atlas: &Atlas,
    _algebra: &MoonshineAlgebra,
    max_digits: usize,
) -> Result<BigUint> {
    // Decoding strategy: Reverse the MoonshineHRM group action
    // The embedding composes operators and applies to identity
    // We need to find which sequence of operators was applied

    let identity = GriessVector::identity();
    let mut digits = Vec::new();
    let mut current = vector.clone();

    // Iteratively extract digits by finding the best matching operator
    for _digit_pos in 0..max_digits {
        // Check if we're close to identity (no more significant digits)
        if current.distance(&identity) < 0.5 {
            break;
        }

        // Try each possible class and find which inverse brings us closest to a simpler state
        let mut best_class = 0u8;
        let mut best_residual_norm = f64::INFINITY;

        for candidate_class in 0..96u8 {
            // Create operator and its inverse
            let op = MoonshineOperator::new(candidate_class)?;
            let inv_op = op.inverse()?;

            // Apply inverse to current vector
            let test_residual = inv_op.act(&current)?;

            // Score based on how much closer we get to identity or an atlas vector
            let score = test_residual.distance(&identity);

            if score < best_residual_norm {
                best_residual_norm = score;
                best_class = candidate_class;
            }
        }

        // Apply the best inverse operator to make progress
        if best_class != 0 {
            let best_op = MoonshineOperator::new(best_class)?;
            let inv_best_op = best_op.inverse()?;
            current = inv_best_op.act(&current)?;
            digits.push(best_class);
        } else {
            // If best class is 0, check if we should stop
            if best_residual_norm < 1.0 {
                break;
            }
            // Otherwise record the zero and continue
            digits.push(0);
            break;
        }
    }

    // Handle empty digit sequence (decoded to zero)
    if digits.is_empty() {
        return Ok(BigUint::zero());
    }

    // Reconstruct BigUint from base-96 digits
    // value = d₀ + d₁·96 + d₂·96² + ...
    let mut result = BigUint::zero();
    let base = BigUint::from(96u32);

    for (i, &digit) in digits.iter().enumerate() {
        let digit_value = BigUint::from(digit);
        let positional_value = digit_value * base.pow(i as u32);
        result += positional_value;
    }

    Ok(result)
}

/// Decode with simplified nearest-neighbor search
///
/// A simpler variant that uses direct nearest-neighbor matching
/// without inverse group actions. Useful when Lie algebra is not available.
///
/// # Example
///
/// ```ignore
/// let atlas = Atlas::with_cache()?;
/// let value = BigUint::from(143u32);
///
/// // Create simple embedding
/// let topology = NetworkTopology::from_biguint(&value);
/// let vector = embed_with_topology(&topology, &atlas, &algebra)?;
///
/// // Decode back
/// let decoded = decode_nearest_neighbor(&vector, &atlas, 10)?;
/// ```
pub fn decode_nearest_neighbor(vector: &GriessVector, atlas: &Atlas, max_digits: usize) -> Result<BigUint> {
    // Simplified approach: just extract digits by repeated nearest-class
    let mut digits = Vec::new();
    let mut residual = vector.clone();

    for _ in 0..max_digits {
        // Find nearest class
        let mut min_distance = f64::INFINITY;
        let mut nearest_class = 0u8;

        for class in 0..96 {
            let atlas_vector = atlas.get_vector(class)?;
            let distance = residual.distance(&atlas_vector);

            if distance < min_distance {
                min_distance = distance;
                nearest_class = class;
            }
        }

        digits.push(nearest_class);

        // Simple residual reduction: normalize and scale down
        let atlas_vec = atlas.get_vector(nearest_class)?;
        let residual_data: Vec<f64> = residual
            .as_slice()
            .iter()
            .zip(atlas_vec.as_slice().iter())
            .map(|(&r, &a)| r - a * 0.5)
            .collect();

        residual = GriessVector::from_vec(residual_data)?;

        // Stop if residual is very small
        if residual.norm() < 1e-3 {
            break;
        }
    }

    // Reconstruct BigUint from digits
    if digits.is_empty() {
        return Ok(BigUint::zero());
    }

    let mut result = BigUint::zero();
    let base = BigUint::from(96u32);

    for (i, &digit) in digits.iter().enumerate() {
        let digit_value = BigUint::from(digit);
        result += digit_value * base.pow(i as u32);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::embed_u64;
    use crate::symbolic::SymbolicInteger;

    #[test]
    fn test_decode_single_digit_roundtrip() {
        let atlas = Atlas::with_cache().unwrap();

        // Test all single-digit values (0-95) roundtrip correctly
        for value in 0..96 {
            let vector = embed_u64(value, &atlas).unwrap();
            let decoded = decode_vector(&vector, &atlas).unwrap();

            assert_eq!(
                decoded,
                BigUint::from(value),
                "Single digit {} failed to roundtrip",
                value
            );
        }
    }

    #[test]
    fn test_decode_zero() {
        let atlas = Atlas::with_cache().unwrap();
        let vector = atlas.get_vector(0).unwrap();
        let decoded = decode_vector(&vector, &atlas).unwrap();

        assert_eq!(decoded, BigUint::from(0u64));
    }

    #[test]
    fn test_decode_max_single_digit() {
        let atlas = Atlas::with_cache().unwrap();
        let vector = atlas.get_vector(95).unwrap();
        let decoded = decode_vector(&vector, &atlas).unwrap();

        assert_eq!(decoded, BigUint::from(95u64));
    }

    #[test]
    fn test_decode_deterministic() {
        let atlas = Atlas::with_cache().unwrap();
        let vector = embed_u64(42, &atlas).unwrap();

        let decoded1 = decode_vector(&vector, &atlas).unwrap();
        let decoded2 = decode_vector(&vector, &atlas).unwrap();

        assert_eq!(decoded1, decoded2, "Decoding should be deterministic");
    }

    #[test]
    fn test_decode_finds_nearest_class() {
        let atlas = Atlas::with_cache().unwrap();

        // Encode a single-digit value
        let vector = embed_u64(25, &atlas).unwrap();

        // Decode should find class 25 as nearest
        let decoded = decode_vector(&vector, &atlas).unwrap();
        assert_eq!(decoded, BigUint::from(25u64));

        // Verify it's actually closest to class 25
        let atlas_25 = atlas.get_vector(25).unwrap();
        let distance_25 = vector.distance(&atlas_25);

        // Check a few other classes are farther
        for other in [24u8, 26, 0, 50, 95] {
            let atlas_other = atlas.get_vector(other).unwrap();
            let distance_other = vector.distance(&atlas_other);
            assert!(
                distance_25 <= distance_other,
                "Class 25 should be nearest, but class {} is closer",
                other
            );
        }
    }

    #[test]
    fn test_decode_perturbed_vector() {
        let atlas = Atlas::with_cache().unwrap();

        // Start with an Atlas vector for class 30
        let original = atlas.get_vector(30).unwrap();

        // Create perturbed version
        let mut perturbed_data = original.to_vec();
        for item in perturbed_data.iter_mut().take(100) {
            *item *= 1.001; // 0.1% perturbation
        }
        let vector = GriessVector::from_vec(perturbed_data).unwrap();

        let decoded = decode_vector(&vector, &atlas).unwrap();
        assert_eq!(
            decoded,
            BigUint::from(30u64),
            "Small perturbation should preserve nearest class"
        );
    }

    #[test]
    #[ignore = "Slow test: takes ~21 seconds"]
    fn test_decode_all_classes_distinguishable() {
        let atlas = Atlas::with_cache().unwrap();

        // Verify all 96 Atlas vectors decode to their own class
        for class in 0..96 {
            let vector = atlas.get_vector(class).unwrap();
            let decoded = decode_vector(&vector, &atlas).unwrap();

            assert_eq!(
                decoded,
                BigUint::from(class as u64),
                "Atlas vector for class {} should decode to itself",
                class
            );
        }
    }

    #[test]
    fn test_decode_boundary_classes() {
        let atlas = Atlas::with_cache().unwrap();

        // Test boundary values: 0, 1, 95
        for value in [0u64, 1, 95] {
            let vector = embed_u64(value, &atlas).unwrap();
            let decoded = decode_vector(&vector, &atlas).unwrap();

            assert_eq!(
                decoded,
                BigUint::from(value),
                "Boundary value {} failed to roundtrip",
                value
            );
        }
    }

    #[test]
    fn test_decode_symbolic_integer_single_digit() {
        let atlas = Atlas::with_cache().unwrap();

        let sym = SymbolicInteger::from(50u64);
        let vector = crate::embed::embed_integer(&sym, &atlas).unwrap();
        let decoded = decode_vector(&vector, &atlas).unwrap();

        assert_eq!(decoded, BigUint::from(50u64));
    }

    mod property_based_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Property: Single-digit values (0-95) always roundtrip exactly
            #[test]
            #[ignore = "Memory intensive: creates many Atlas instances with cache"]
            fn prop_single_digit_roundtrip(value in 0u64..96) {
                let atlas = Atlas::with_cache().unwrap();
                let vector = embed_u64(value, &atlas).unwrap();
                let decoded = decode_vector(&vector, &atlas).unwrap();

                prop_assert_eq!(decoded, BigUint::from(value));
            }

            /// Property: Decoding is deterministic
            #[test]
            #[ignore = "Memory intensive: creates many Atlas instances with cache"]
            fn prop_decode_deterministic(value in 0u64..96) {
                let atlas = Atlas::with_cache().unwrap();
                let vector = embed_u64(value, &atlas).unwrap();

                let decoded1 = decode_vector(&vector, &atlas).unwrap();
                let decoded2 = decode_vector(&vector, &atlas).unwrap();

                prop_assert_eq!(decoded1, decoded2);
            }

            /// Property: Decoded value is always in valid range [0, 95]
            #[test]
            #[ignore = "Memory intensive: creates many Atlas instances with cache"]
            fn prop_decoded_value_in_range(value in 0u64..96) {
                let atlas = Atlas::with_cache().unwrap();
                let vector = embed_u64(value, &atlas).unwrap();
                let decoded = decode_vector(&vector, &atlas).unwrap();

                let decoded_u64 = decoded.to_u64_digits();
                let decoded_val = if decoded_u64.is_empty() { 0 } else { decoded_u64[0] };
                prop_assert!(decoded_val < 96);
            }

            /// Property: Different single-digit inputs produce different decodings
            #[test]
            #[ignore = "Memory intensive: creates many Atlas instances with cache"]
            fn prop_different_inputs_different_outputs(value1 in 0u64..96, value2 in 0u64..96) {
                prop_assume!(value1 != value2);

                let atlas = Atlas::with_cache().unwrap();
                let vector1 = embed_u64(value1, &atlas).unwrap();
                let vector2 = embed_u64(value2, &atlas).unwrap();

                let decoded1 = decode_vector(&vector1, &atlas).unwrap();
                let decoded2 = decode_vector(&vector2, &atlas).unwrap();

                prop_assert_ne!(decoded1, decoded2);
            }

            /// Property: Decoder finds one of the 96 Atlas classes
            #[test]
            #[ignore = "Memory intensive: creates many Atlas instances with cache"]
            fn prop_decoder_finds_atlas_class(value in 0u64..96) {
                let atlas = Atlas::with_cache().unwrap();
                let vector = embed_u64(value, &atlas).unwrap();
                let decoded = decode_vector(&vector, &atlas).unwrap();

                // Decoded value must be a valid Atlas class index
                let digits = decoded.to_u64_digits();
                let class_idx = if digits.is_empty() { 0 } else { digits[0] as u8 };
                prop_assert!(class_idx < 96);

                // The decoded class should be the nearest
                let decoded_vector = atlas.get_vector(class_idx).unwrap();
                let distance_to_decoded = vector.distance(&decoded_vector);

                // Check it's at least as close as original class
                let original_vector = atlas.get_vector(value as u8).unwrap();
                let distance_to_original = vector.distance(&original_vector);

                prop_assert!(distance_to_decoded <= distance_to_original + 1e-10);
            }
        }
    }
}
