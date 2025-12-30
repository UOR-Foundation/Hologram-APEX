//! Integration tests for MoonshineHRM embeddings model
//!
//! These tests verify that the MoonshineHRM group action properly operates on
//! the embeddings, and that encoding/decoding is working correctly.

use hologram_hrm::{
    algebra::{LieAlgebra, MoonshineAlgebra},
    atlas::Atlas,
    embed::{embed_with_moonshine, embed_with_topology},
    moonshine::action::{GroupAction, NetworkTopology},
    moonshine::{MoonshineOperator, OperatorSequence},
    GriessVector, Result,
};
use num_bigint::BigUint;

#[test]
fn test_moonshine_embedding_properties() -> Result<()> {
    let atlas = Atlas::with_cache()?;
    let algebra = MoonshineAlgebra::with_cache()?;

    // Test with different values
    let values = [143u32, 1961, 42, 96, 255];
    let mut embeddings = Vec::new();

    for &value in &values {
        let input = BigUint::from(value);

        // Embed using MoonshineHRM group action
        let embedded = embed_with_moonshine(&input, &atlas, &algebra)?;

        // Verify correct dimension
        assert_eq!(embedded.len(), 196_884);

        // Verify non-zero (except for value 0)
        if value != 0 {
            assert!(!embedded.is_zero(1e-10));
        }

        embeddings.push(embedded);
    }

    // Verify different inputs produce different embeddings
    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let distance = embeddings[i].distance(&embeddings[j]);
            assert!(
                distance > 1.0,
                "Embeddings too similar for values {} and {}: distance = {}",
                values[i],
                values[j],
                distance
            );
        }
    }

    Ok(())
}

#[test]
fn test_moonshine_operator_group_action() -> Result<()> {
    // Test that MoonshineOperator acts on vectors correctly
    let op1 = MoonshineOperator::new(5)?;
    let op2 = MoonshineOperator::new(7)?;

    let identity = GriessVector::identity();

    // Apply operators sequentially
    let v1 = op1.act(&identity)?;
    let v2 = op2.act(&v1)?;

    // Apply composed operator
    let composed = op1.compose(&op2);
    let v_composed = composed.act(&identity)?;

    // Results should be similar
    let distance = v2.distance(&v_composed);

    // Allow some numerical tolerance
    assert!(distance < 1.0, "Composed operator distance too large: {}", distance);

    Ok(())
}

#[test]
fn test_network_topology_from_number() -> Result<()> {
    // Test that network topology is created correctly
    let value = BigUint::from(1961u32);
    let topology = NetworkTopology::from_biguint(&value);

    // 1961 in base-96 is [41, 20] (41 + 20*96 = 1961)
    assert_eq!(topology.num_nodes(), 2);
    assert_eq!(topology.nodes[0], 41);
    assert_eq!(topology.nodes[1], 20);
    assert_eq!(topology.num_edges(), 1); // Chain has n-1 edges

    Ok(())
}

#[test]
fn test_operator_sequence_composition() -> Result<()> {
    // Test that operator sequences compose correctly
    let mut seq = OperatorSequence::new();
    seq.push(MoonshineOperator::new(3)?);
    seq.push(MoonshineOperator::new(5)?);
    seq.push(MoonshineOperator::new(7)?);

    // Compose all operators: 3 + 5 + 7 = 15 mod 96 = 15
    let composed = seq.compose_all();
    assert_eq!(composed.class, 15);

    Ok(())
}

#[test]
fn test_lie_algebra_scaling() -> Result<()> {
    let algebra = MoonshineAlgebra::with_cache()?;
    let identity = GriessVector::identity();

    // Scale using generator 5 with theta = 0.5
    let scaled = algebra.scale(&identity, 5, 0.5)?;

    // Scaled vector should be different from identity
    let distance = scaled.distance(&identity);
    assert!(distance > 1e-6, "Scaling had no effect: distance = {}", distance);

    // Verify result has correct dimension
    assert_eq!(scaled.len(), 196_884);

    Ok(())
}

#[test]
fn test_embed_with_topology() -> Result<()> {
    let atlas = Atlas::with_cache()?;
    let algebra = MoonshineAlgebra::with_cache()?;

    // Create topology from factorization: 143 = 11 * 13
    let factors = vec![11, 13];
    let topology = NetworkTopology::from_factors(&factors)?;

    // Embed using the topology
    let embedded = embed_with_topology(&topology, &atlas, &algebra)?;

    // Verify result has correct dimension
    assert_eq!(embedded.len(), 196_884);

    // Verify it's not zero
    assert!(!embedded.is_zero(1e-10));

    Ok(())
}

#[test]
fn test_group_action_identity() -> Result<()> {
    let identity_op = MoonshineOperator::identity();
    let v = GriessVector::identity();

    // Identity operator should leave vector unchanged
    let result = identity_op.act(&v)?;

    let distance = v.distance(&result);
    assert!(
        distance < 1e-10,
        "Identity operator changed the vector: distance = {}",
        distance
    );

    Ok(())
}

#[test]
fn test_moonshine_deterministic() -> Result<()> {
    let atlas = Atlas::with_cache()?;
    let algebra = MoonshineAlgebra::with_cache()?;

    let value = BigUint::from(1961u32);

    // Embed twice
    let v1 = embed_with_moonshine(&value, &atlas, &algebra)?;
    let v2 = embed_with_moonshine(&value, &atlas, &algebra)?;

    // Results should be identical
    let distance = v1.distance(&v2);
    assert!(distance < 1e-10, "Embedding not deterministic: distance = {}", distance);

    Ok(())
}

#[test]
fn test_moonshine_generators() -> Result<()> {
    let algebra = MoonshineAlgebra::with_cache()?;

    // Test that all 96 generators can be accessed
    for i in 0..96 {
        let generator = algebra.generator(i)?;
        assert_eq!(generator.len(), 196_884);
        assert!(!generator.is_zero(1e-10));
    }

    Ok(())
}

#[test]
fn test_lie_bracket() -> Result<()> {
    let algebra = MoonshineAlgebra::with_cache()?;

    // Test Lie bracket antisymmetry: [g_i, g_j] = -[g_j, g_i]
    let bracket_ij = algebra.bracket(5, 10)?;
    let bracket_ji = algebra.bracket(10, 5)?;

    let ij_data = bracket_ij.as_slice();
    let ji_data = bracket_ji.as_slice();

    // Check antisymmetry
    for i in 0..196_884 {
        let sum = ij_data[i] + ji_data[i];
        assert!(
            sum.abs() < 1e-6,
            "Bracket not antisymmetric at index {}: {} + {} = {}",
            i,
            ij_data[i],
            ji_data[i],
            sum
        );
    }

    Ok(())
}
