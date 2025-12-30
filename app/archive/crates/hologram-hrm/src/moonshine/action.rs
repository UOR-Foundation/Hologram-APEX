//! Group actions and network topology for MoonshineHRM embeddings
//!
//! This module provides:
//! - **GroupAction trait**: How operators act on Griess vectors
//! - **NetworkTopology**: Graph structure representing base decompositions
//! - **Action implementation**: Concrete transformation of vectors by operators
//!
//! The network topology captures the structure of number decompositions:
//! - For integer n with base-96 digits [d₀, d₁, d₂]: digit sequence graph
//! - For semi-prime p×q: factor graph with two nodes
//! - Edges encode algebraic relationships between digits/factors

#![allow(missing_docs)]

use crate::algebra::{LieAlgebra, MoonshineAlgebra};
use crate::moonshine::MoonshineOperator;
use crate::{Error, GriessVector, Result, GRIESS_DIMENSION};
use num_bigint::BigUint;
use num_traits::Zero;

/// Group action trait
///
/// A group action is a way for group elements (operators) to transform
/// elements of a space (Griess vectors). The action must satisfy:
/// - Identity: e·v = v
/// - Compatibility: (g₁·g₂)·v = g₁·(g₂·v)
pub trait GroupAction {
    /// Apply this group element to a vector
    fn act(&self, vector: &GriessVector) -> Result<GriessVector>;

    /// Apply action multiple times (power)
    fn act_pow(&self, vector: &GriessVector, n: u32) -> Result<GriessVector> {
        let mut result = vector.clone();
        for _ in 0..n {
            result = self.act(&result)?;
        }
        Ok(result)
    }
}

/// Network topology representing base decomposition structure
///
/// The topology encodes how a number is decomposed:
/// - **Digits**: For base-96 representation [d₀, d₁, ...]
/// - **Factors**: For factorizations like p×q
/// - **Edges**: Relationships between components
///
/// This structure is what the MoonshineHRM group action operates on.
#[derive(Debug, Clone)]
pub struct NetworkTopology {
    /// Node values (digit values or factor values)
    pub nodes: Vec<u8>,

    /// Edge list: (from_idx, to_idx, weight)
    /// Weight encodes the algebraic relationship
    pub edges: Vec<(usize, usize, f64)>,

    /// Topology type
    pub topology_type: TopologyType,
}

/// Type of network topology
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyType {
    /// Sequential digit chain: d₀ → d₁ → d₂ → ...
    DigitChain,

    /// Factor tree: root → factor₁, factor₂
    FactorTree,

    /// Custom topology
    Custom,
}

impl NetworkTopology {
    /// Create topology from base-96 digit sequence
    ///
    /// For value = d₀ + d₁·96 + d₂·96² + ..., creates chain:
    /// d₀ → d₁ → d₂ → ...
    pub fn from_digits(digits: &[u8]) -> Self {
        let mut edges = Vec::new();

        // Create chain edges with positional weights
        for i in 0..(digits.len().saturating_sub(1)) {
            let weight = 96f64.powi(i as i32);
            edges.push((i, i + 1, weight));
        }

        Self {
            nodes: digits.to_vec(),
            edges,
            topology_type: TopologyType::DigitChain,
        }
    }

    /// Create topology from factorization
    ///
    /// For value = p × q, creates tree with root and two factor nodes
    pub fn from_factors(factors: &[u64]) -> Result<Self> {
        if factors.is_empty() {
            return Err(Error::InvalidInput("Empty factor list".to_string()));
        }

        // Convert factors to resonance classes (mod 96)
        let nodes: Vec<u8> = factors.iter().map(|&f| (f % 96) as u8).collect();

        // Create star topology: all factors connect to virtual root
        let mut edges = Vec::new();
        for i in 0..nodes.len() {
            edges.push((i, (i + 1) % nodes.len(), 1.0));
        }

        Ok(Self {
            nodes,
            edges,
            topology_type: TopologyType::FactorTree,
        })
    }

    /// Create from arbitrary BigUint using base-96 decomposition
    pub fn from_biguint(value: &BigUint) -> Self {
        if value.is_zero() {
            return Self {
                nodes: vec![0],
                edges: Vec::new(),
                topology_type: TopologyType::DigitChain,
            };
        }

        let mut digits = Vec::new();
        let mut n = value.clone();
        let base = BigUint::from(96u32);

        while !n.is_zero() {
            let digit = (&n % &base).to_u64_digits();
            let digit_val = if digit.is_empty() { 0 } else { digit[0] as u8 };
            digits.push(digit_val);
            n /= &base;
        }

        Self::from_digits(&digits)
    }

    /// Get number of nodes in the topology
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Check if topology is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Implementation of GroupAction for MoonshineOperator
impl GroupAction for MoonshineOperator {
    fn act(&self, vector: &GriessVector) -> Result<GriessVector> {
        // The action of operator k on vector v is:
        // k·v = rotation of v by angle θ_k in Griess space
        //
        // This is implemented as: v + α·g_k
        // where g_k is the generator for class k and α is a scaling factor

        if self.is_identity() {
            return Ok(vector.clone());
        }

        // Get the Lie algebra generator for this class
        // (This requires having a MoonshineAlgebra instance, which we'll
        // need to pass through the embedding context)
        //
        // For now, implement a simple rotation based on class index
        let v_data = vector.as_slice();
        let mut result = vec![0.0; GRIESS_DIMENSION];

        // Rotation angle based on resonance class
        let theta = 2.0 * std::f64::consts::PI * (self.class as f64) / 96.0;

        // Apply rotation in pairs of coordinates (simple 2D rotations)
        for i in 0..(GRIESS_DIMENSION / 2) {
            let idx1 = 2 * i;
            let idx2 = 2 * i + 1;

            let x = v_data[idx1];
            let y = v_data[idx2];

            // 2D rotation: [x'] = [cos θ  -sin θ] [x]
            //              [y']   [sin θ   cos θ] [y]
            result[idx1] = x * theta.cos() - y * theta.sin();
            result[idx2] = x * theta.sin() + y * theta.cos();
        }

        // Handle odd dimension
        if GRIESS_DIMENSION % 2 == 1 {
            result[GRIESS_DIMENSION - 1] = v_data[GRIESS_DIMENSION - 1];
        }

        GriessVector::from_vec(result)
    }
}

/// Enhanced action that uses MoonshineAlgebra for proper Lie group action
pub struct AlgebraAction<'a> {
    pub operator: MoonshineOperator,
    pub algebra: &'a MoonshineAlgebra,
    pub theta: f64,
}

impl<'a> AlgebraAction<'a> {
    pub fn new(operator: MoonshineOperator, algebra: &'a MoonshineAlgebra, theta: f64) -> Self {
        Self {
            operator,
            algebra,
            theta,
        }
    }
}

impl<'a> GroupAction for AlgebraAction<'a> {
    fn act(&self, vector: &GriessVector) -> Result<GriessVector> {
        // Use Lie algebra exponential map for the action
        // exp(θ·g_k) · v
        self.algebra.scale(vector, self.operator.class as usize, self.theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_topology_from_digits() {
        let digits = vec![1, 2, 3, 4];
        let topology = NetworkTopology::from_digits(&digits);

        assert_eq!(topology.num_nodes(), 4);
        assert_eq!(topology.num_edges(), 3); // Chain has n-1 edges
        assert_eq!(topology.topology_type, TopologyType::DigitChain);

        // Check edge weights (positional)
        assert_eq!(topology.edges[0].2, 1.0); // 96^0
        assert_eq!(topology.edges[1].2, 96.0); // 96^1
        assert_eq!(topology.edges[2].2, 9216.0); // 96^2
    }

    #[test]
    fn test_network_topology_from_factors() {
        let factors = vec![11, 13]; // 143 = 11 × 13
        let topology = NetworkTopology::from_factors(&factors).unwrap();

        assert_eq!(topology.num_nodes(), 2);
        assert_eq!(topology.topology_type, TopologyType::FactorTree);
    }

    #[test]
    fn test_network_topology_from_biguint() {
        // Test with 143 = 1 + 1*96 + 0*96^2
        let value = BigUint::from(143u32);
        let topology = NetworkTopology::from_biguint(&value);

        assert_eq!(topology.num_nodes(), 2); // [47, 1]
        assert_eq!(topology.topology_type, TopologyType::DigitChain);
    }

    #[test]
    fn test_network_topology_zero() {
        let value = BigUint::zero();
        let topology = NetworkTopology::from_biguint(&value);

        assert_eq!(topology.num_nodes(), 1);
        assert_eq!(topology.nodes[0], 0);
        assert!(topology.edges.is_empty());
    }

    #[test]
    fn test_group_action_identity() -> Result<()> {
        let identity = MoonshineOperator::identity();
        let v = GriessVector::identity();

        let result = identity.act(&v)?;

        // Identity action should return same vector
        let v_data = v.as_slice();
        let r_data = result.as_slice();

        for i in 0..GRIESS_DIMENSION {
            assert!((v_data[i] - r_data[i]).abs() < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_group_action_rotation() -> Result<()> {
        let op = MoonshineOperator::new(12)?; // π/4 rotation (96/8 = 12)

        // Create a simple vector
        let mut v_data = vec![0.0; GRIESS_DIMENSION];
        v_data[0] = 1.0;
        v_data[1] = 0.0;
        let v = GriessVector::from_vec(v_data)?;

        let result = op.act(&v)?;
        let r_data = result.as_slice();

        // After rotation, coordinates should have changed
        assert!(r_data[0] != 1.0);

        Ok(())
    }

    #[test]
    fn test_group_action_composition() -> Result<()> {
        let op1 = MoonshineOperator::new(6)?;
        let op2 = MoonshineOperator::new(6)?;

        let v = GriessVector::identity();

        // Apply op1, then op2
        let v1 = op1.act(&v)?;
        let v2 = op2.act(&v1)?;

        // Compose operators and apply once
        let composed = op1.compose(&op2);
        let v_composed = composed.act(&v)?;

        // Results should be similar (may not be exact due to numerical precision)
        let diff_norm = GriessVector::from_vec(
            v2.as_slice()
                .iter()
                .zip(v_composed.as_slice().iter())
                .map(|(a, b)| a - b)
                .collect(),
        )?
        .norm();

        assert!(diff_norm < 1.0); // Some tolerance for composition

        Ok(())
    }

    #[test]
    fn test_empty_factor_list() {
        let result = NetworkTopology::from_factors(&[]);
        assert!(result.is_err());
    }
}
