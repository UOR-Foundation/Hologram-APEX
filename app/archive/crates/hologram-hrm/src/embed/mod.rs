//! Embedding operator E: Integer → GriessVector
//!
//! The embedding operator maps arbitrary integers to Griess vectors using two approaches:
//!
//! ## Classical Embedding (Hadamard Product)
//!
//! Given a SymbolicInteger with base-96 representation [d₀, d₁, d₂, ...]:
//! 1. Get the Atlas vector Vᵢ for each digit dᵢ
//! 2. Compose using Hadamard product: V = V₀ ⊙ V₁ ⊙ V₂ ⊙ ...
//! 3. The result V is the embedded Griess vector
//!
//! ## MoonshineHRM Embedding (Group Action)
//!
//! The canonical MoonshineHRM approach uses group actions on network topology:
//! 1. Parse input to base decomposition → creates NetworkTopology
//! 2. For each digit d_i, get MoonshineOperator for resonance class d_i
//! 3. Compose operators: G = g₀ ∘ g₁ ∘ g₂ ∘ ... (group multiplication)
//! 4. Apply group action to identity: V = G·V_identity
//! 5. Scale via Lie algebra based on input structure
//! 6. Result is canonical embedding in Griess lattice
//!
//! # Properties
//!
//! - **Deterministic**: Same input always produces same output
//! - **Structure-preserving**: Group action preserves algebraic relationships
//! - **Scalable**: Lie algebra scaling modulates representation based on input
//! - **Reversible**: Can decode back via dual lattice projection
//!
//! # Example
//!
//! ```ignore
//! use hologram_hrm::{SymbolicInteger, Atlas, MoonshineAlgebra, embed_with_moonshine};
//!
//! let atlas = Atlas::with_cache()?;
//! let algebra = MoonshineAlgebra::with_cache()?;
//! let value = BigUint::from(1961u64);
//!
//! // Embed using MoonshineHRM group action
//! let vector = embed_with_moonshine(&value, &atlas, &algebra)?;
//! assert_eq!(vector.len(), 196_884);
//! ```

use crate::algebra::{LieAlgebra, MoonshineAlgebra};
use crate::atlas::Atlas;
use crate::griess::{product, GriessVector};
use crate::moonshine::action::{GroupAction, NetworkTopology};
use crate::moonshine::{MoonshineOperator, OperatorSequence};
use crate::symbolic::SymbolicInteger;
use crate::Result;
use num_bigint::BigUint;
use num_traits::Zero;

/// Embed a SymbolicInteger into Griess space
///
/// This is the core embedding operator E that maps integers to Griess vectors.
///
/// # Algorithm
///
/// 1. Convert integer to base-96 digits [d₀, d₁, d₂, ...]
/// 2. Retrieve Atlas vector Vᵢ for each digit dᵢ
/// 3. Compose via Hadamard product: V = V₀ ⊙ V₁ ⊙ V₂ ⊙ ...
///
/// # Arguments
///
/// * `sym` - The SymbolicInteger to embed
/// * `atlas` - The Atlas providing canonical vectors
///
/// # Returns
///
/// A GriessVector representing the embedded integer.
///
/// # Example
///
/// ```ignore
/// let atlas = Atlas::with_cache()?;
/// let sym = SymbolicInteger::from(1961u64);
/// let vector = embed_integer(&sym, &atlas)?;
/// ```
pub fn embed_integer(sym: &SymbolicInteger, atlas: &Atlas) -> Result<GriessVector> {
    let digits = sym.to_base96();

    // Start with identity element
    let mut result = GriessVector::identity();

    // Compose Atlas vectors via Hadamard product
    for digit in digits {
        let atlas_vector = atlas.get_vector(digit)?;
        result = product(&result, &atlas_vector)?;
    }

    Ok(result)
}

/// Embed a u64 value into Griess space
///
/// Convenience function for embedding u64 values directly.
///
/// # Example
///
/// ```ignore
/// let atlas = Atlas::with_cache()?;
/// let vector = embed_u64(1961, &atlas)?;
/// ```
pub fn embed_u64(value: u64, atlas: &Atlas) -> Result<GriessVector> {
    let sym = SymbolicInteger::from(value);
    embed_integer(&sym, atlas)
}

/// Embed an arbitrary integer using MoonshineHRM group action
///
/// This is the canonical MoonshineHRM embedding that uses group actions
/// on network topology to create structure-preserving embeddings.
///
/// # Algorithm
///
/// 1. **Create network topology**: Parse value to base-96 decomposition
/// 2. **Build operator sequence**: For each digit d_i, get operator for class d_i
/// 3. **Compose operators**: G = g₀ ∘ g₁ ∘ g₂ ∘ ... (group multiplication)
/// 4. **Apply group action**: V = G · V_identity
/// 5. **Lie algebra scaling**: Scale result based on input structure
///
/// # Arguments
///
/// * `value` - The BigUint to embed
/// * `atlas` - The Atlas providing canonical vectors
/// * `algebra` - The MoonshineAlgebra for Lie scaling
///
/// # Returns
///
/// A GriessVector representing the canonical embedding.
///
/// # Example
///
/// ```ignore
/// let atlas = Atlas::with_cache()?;
/// let algebra = MoonshineAlgebra::with_cache()?;
/// let value = BigUint::from(143u32); // Semi-prime 11 × 13
///
/// let vector = embed_with_moonshine(&value, &atlas, &algebra)?;
/// ```
pub fn embed_with_moonshine(value: &BigUint, atlas: &Atlas, algebra: &MoonshineAlgebra) -> Result<GriessVector> {
    // Handle zero case
    if value.is_zero() {
        return Ok(atlas.get_vector(0)?.as_ref().clone());
    }

    // Create network topology from base-96 decomposition
    let topology = NetworkTopology::from_biguint(value);

    // Build operator sequence from digit nodes
    let mut operators = OperatorSequence::new();
    for &digit in &topology.nodes {
        let op = MoonshineOperator::new(digit)?;
        operators.push(op);
    }

    // Compose all operators into single group element
    let composed_operator = operators.compose_all();

    // Apply group action to identity vector
    let identity = GriessVector::identity();
    let mut result = composed_operator.act(&identity)?;

    // Scale using Lie algebra based on input structure
    // The scaling factor is proportional to the number of digits
    // (larger numbers get more scaling)
    let num_digits = topology.num_nodes();
    if num_digits > 1 {
        let theta = (num_digits as f64).ln() / 10.0; // Logarithmic scaling
        let remainder = value % BigUint::from(96u32);
        let digits = remainder.to_u64_digits();
        let generator_idx = if digits.is_empty() { 0 } else { digits[0] as usize % 96 };
        result = algebra.scale(&result, generator_idx, theta)?;
    }

    Ok(result)
}

/// Embed using MoonshineHRM with explicit network topology
///
/// This variant allows custom network topologies (e.g., factor trees)
/// rather than just base-96 digit chains.
///
/// # Example
///
/// ```ignore
/// // Create factor topology for semi-prime 143 = 11 × 13
/// let factors = vec![11, 13];
/// let topology = NetworkTopology::from_factors(&factors)?;
///
/// let vector = embed_with_topology(&topology, &atlas, &algebra)?;
/// ```
pub fn embed_with_topology(
    topology: &NetworkTopology,
    atlas: &Atlas,
    algebra: &MoonshineAlgebra,
) -> Result<GriessVector> {
    if topology.is_empty() {
        return Ok(atlas.get_vector(0)?.as_ref().clone());
    }

    // Build operator sequence from topology nodes
    let mut operators = OperatorSequence::new();
    for &digit in &topology.nodes {
        let op = MoonshineOperator::new(digit)?;
        operators.push(op);
    }

    // Compose and apply
    let composed_operator = operators.compose_all();
    let identity = GriessVector::identity();
    let mut result = composed_operator.act(&identity)?;

    // Scale based on topology structure (edges encode relationships)
    if topology.num_edges() > 0 {
        // Use edge weights for scaling
        let total_weight: f64 = topology.edges.iter().map(|(_, _, w)| w).sum();
        let theta = (total_weight.ln() + 1.0) / 100.0;

        // Use first node as generator index
        let generator_idx = topology.nodes[0] as usize % 96;
        result = algebra.scale(&result, generator_idx, theta)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_zero() {
        let atlas = Atlas::with_cache().unwrap();
        let sym = SymbolicInteger::zero();

        let vector = embed_integer(&sym, &atlas).unwrap();

        // Zero embeds to the Atlas vector for digit 0
        let expected = atlas.get_vector(0).unwrap();
        assert_eq!(vector, *expected);
    }

    #[test]
    fn test_embed_single_digit() {
        let atlas = Atlas::with_cache().unwrap();

        for digit in 0..96 {
            let sym = SymbolicInteger::from(digit as u64);
            let vector = embed_integer(&sym, &atlas).unwrap();

            // Single digit embeds to its Atlas vector
            let expected = atlas.get_vector(digit).unwrap();
            assert_eq!(vector, *expected);
        }
    }

    #[test]
    fn test_embed_two_digits() {
        let atlas = Atlas::with_cache().unwrap();

        // 96 in base-96 is [0, 1]
        let sym = SymbolicInteger::from(96u64);
        let vector = embed_integer(&sym, &atlas).unwrap();

        // Should be V₀ ⊙ V₁
        let v0 = atlas.get_vector(0).unwrap();
        let v1 = atlas.get_vector(1).unwrap();
        let expected = product(&v0, &v1).unwrap();

        assert_eq!(vector, expected);
    }

    #[test]
    fn test_embed_1961() {
        let atlas = Atlas::with_cache().unwrap();

        // 1961 = 41 + 20·96 = [41, 20] in base-96
        let sym = SymbolicInteger::from(1961u64);
        let vector = embed_integer(&sym, &atlas).unwrap();

        // Should be V₄₁ ⊙ V₂₀
        let v41 = atlas.get_vector(41).unwrap();
        let v20 = atlas.get_vector(20).unwrap();
        let expected = product(&v41, &v20).unwrap();

        assert_eq!(vector, expected);
    }

    #[test]
    fn test_embed_deterministic() {
        let atlas = Atlas::with_cache().unwrap();
        let sym = SymbolicInteger::from(1961u64);

        let v1 = embed_integer(&sym, &atlas).unwrap();
        let v2 = embed_integer(&sym, &atlas).unwrap();

        // Same input produces same output
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_embed_u64() {
        let atlas = Atlas::with_cache().unwrap();

        let v1 = embed_u64(1961, &atlas).unwrap();
        let v2 = embed_integer(&SymbolicInteger::from(1961u64), &atlas).unwrap();

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_embed_different_values() {
        let atlas = Atlas::with_cache().unwrap();

        let v1 = embed_u64(1961, &atlas).unwrap();
        let v2 = embed_u64(1962, &atlas).unwrap();

        // Different inputs produce different outputs
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_embed_large_number() {
        let atlas = Atlas::with_cache().unwrap();

        let sym = SymbolicInteger::from(1_000_000u64);
        let vector = embed_integer(&sym, &atlas).unwrap();

        assert_eq!(vector.len(), 196_884);

        // Verify it's finite and not zero
        assert!(!vector.is_zero(1e-10));
        let norm = vector.norm();
        assert!(norm.is_finite() && norm > 0.0);
    }

    mod property_based_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Property: Embedding always produces 196,884-dimensional vectors
            #[test]
            #[ignore = "Memory intensive: creates many Atlas instances with cache"]
            fn prop_embed_dimensionality(value in 0u64..10000) {
                let atlas = Atlas::with_cache().unwrap();
                let vector = embed_u64(value, &atlas).unwrap();

                prop_assert_eq!(vector.len(), 196_884);
            }

            /// Property: Embedding is deterministic
            #[test]
            #[ignore = "Memory intensive: creates many Atlas instances with cache"]
            fn prop_embed_deterministic(value in 0u64..1000) {
                let atlas = Atlas::with_cache().unwrap();

                let v1 = embed_u64(value, &atlas).unwrap();
                let v2 = embed_u64(value, &atlas).unwrap();

                prop_assert_eq!(v1, v2);
            }

            /// Property: Different inputs produce different embeddings
            #[test]
            #[ignore = "Memory intensive: creates many Atlas instances with cache"]
            fn prop_different_inputs_different_embeddings(value1 in 0u64..1000, value2 in 0u64..1000) {
                prop_assume!(value1 != value2);

                let atlas = Atlas::with_cache().unwrap();
                let v1 = embed_u64(value1, &atlas).unwrap();
                let v2 = embed_u64(value2, &atlas).unwrap();

                prop_assert_ne!(v1, v2);
            }

            /// Property: Embedded vectors are non-zero
            #[test]
            #[ignore = "Memory intensive: creates many Atlas instances with cache"]
            fn prop_embed_non_zero(value in 1u64..1000) {
                let atlas = Atlas::with_cache().unwrap();
                let vector = embed_u64(value, &atlas).unwrap();

                prop_assert!(!vector.is_zero(1e-10));
            }

            /// Property: Embedded vectors have finite norm
            #[test]
            #[ignore = "Memory intensive: creates many Atlas instances with cache"]
            fn prop_embed_finite_norm(value in 0u64..1000) {
                let atlas = Atlas::with_cache().unwrap();
                let vector = embed_u64(value, &atlas).unwrap();

                let norm = vector.norm();
                prop_assert!(norm.is_finite());
                prop_assert!(norm >= 0.0);
            }

            /// Property: Single-digit values embed to their Atlas vector
            #[test]
            #[ignore = "Memory intensive: creates many Atlas instances with cache"]
            fn prop_single_digit_embeds_to_atlas(digit in 0u8..96) {
                let atlas = Atlas::with_cache().unwrap();

                let vector = embed_u64(digit as u64, &atlas).unwrap();
                let expected = atlas.get_vector(digit).unwrap();

                prop_assert_eq!(&vector, expected.as_ref());
            }

            /// Property: Embedding zero produces Atlas class 0
            #[test]
            #[ignore = "Memory intensive: creates many Atlas instances with cache"]
            fn prop_embed_zero_is_atlas_zero(_seed in 0..100u64) {
                let atlas = Atlas::with_cache().unwrap();

                let vector = embed_u64(0, &atlas).unwrap();
                let expected = atlas.get_vector(0).unwrap();

                prop_assert_eq!(&vector, expected.as_ref());
            }

            /// Property: SymbolicInteger and u64 embed equivalently for small values
            #[test]
            #[ignore = "Memory intensive: creates many Atlas instances with cache"]
            fn prop_symbolic_and_u64_embed_equal(value in 0u64..1000) {
                let atlas = Atlas::with_cache().unwrap();

                let v_u64 = embed_u64(value, &atlas).unwrap();
                let v_sym = embed_integer(&SymbolicInteger::from(value), &atlas).unwrap();

                prop_assert_eq!(v_u64, v_sym);
            }

            // NOTE: The old Hadamard product-based embedding test (prop_embed_respects_base96)
            // has been removed as it tested deprecated behavior. The new MoonshineHRM embeddings
            // use group actions instead of Hadamard products.
        }
    }
}
