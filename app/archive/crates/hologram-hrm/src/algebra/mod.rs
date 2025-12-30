//! Algebraic structures for HRM embeddings
//!
//! This module provides the mathematical foundations for the MoonshineHRM embeddings model:
//!
//! - **Ring**: Basic algebraic structure with addition and multiplication
//! - **Lattice**: Discrete subgroup of vector space with basis operations
//! - **LieAlgebra**: Non-associative algebra with Lie bracket and scaling
//!
//! These structures enable the encoder/decoder to embed symmetrical rings of two lattices,
//! with Lie algebra operations providing the scaling mechanism.

#![allow(clippy::type_complexity)]

pub mod division;
pub mod lattice;
pub mod lie;

// Re-export key types
pub use lie::MoonshineAlgebra;

use crate::{GriessVector, Result};

/// Ring algebraic structure
///
/// A ring is a set equipped with two binary operations (addition and multiplication)
/// satisfying ring axioms. Used as the foundation for lattice embeddings.
pub trait Ring: Clone {
    /// Additive identity (zero element)
    fn zero() -> Self;

    /// Multiplicative identity (one element)
    fn one() -> Self;

    /// Ring addition
    fn add(&self, other: &Self) -> Result<Self>;

    /// Ring multiplication
    fn mul(&self, other: &Self) -> Result<Self>;

    /// Additive inverse (negation)
    fn neg(&self) -> Result<Self>;

    /// Check if element is zero
    fn is_zero(&self) -> bool;
}

/// Lattice structure extending Ring
///
/// A lattice is a discrete subgroup of a vector space. In HRM, lattices provide
/// the geometric structure for embedding integers as points in high-dimensional space.
///
/// The dual lattice structure enables symmetrical encoding/decoding operations.
pub trait Lattice: Ring {
    /// Dimension of the ambient vector space
    fn dimension(&self) -> usize;

    /// Get basis vector at index i
    fn basis_vector(&self, i: usize) -> Result<GriessVector>;

    /// Project a Griess vector onto this lattice
    fn project(&self, vector: &GriessVector) -> Result<GriessVector>;

    /// Check if a point is in the lattice
    fn contains(&self, vector: &GriessVector) -> bool;
}

/// Lie algebra structure for scaling operations
///
/// A Lie algebra is a vector space with a bracket operation [·,·] (the Lie bracket).
/// In HRM, the Lie algebra provides the scaling mechanism that modulates the
/// representation space based on input structure.
///
/// The MoonshineHRM uses 96 generators (one per resonance class) forming a
/// Lie algebra that acts on the Griess space.
pub trait LieAlgebra {
    /// Number of generators in this Lie algebra
    fn num_generators(&self) -> usize;

    /// Get generator at index i
    fn generator(&self, i: usize) -> Result<GriessVector>;

    /// Lie bracket: [g_i, g_j]
    ///
    /// The Lie bracket measures the non-commutativity of the algebra.
    /// Satisfies: [x,y] = -[y,x] and Jacobi identity.
    fn bracket(&self, i: usize, j: usize) -> Result<GriessVector>;

    /// Exponential map: exp(θ · g_i)
    ///
    /// Maps Lie algebra element to group element via matrix exponential.
    /// Used for scaling: exp(θ·g_i) acts on vectors to scale coordinates.
    fn exp_map(&self, generator_idx: usize, theta: f64) -> Result<Box<dyn Fn(&GriessVector) -> Result<GriessVector>>>;

    /// Scale a vector using Lie algebra generator
    ///
    /// Applies exp(θ·g_i) to the input vector, scaling its coordinates
    /// in the direction of generator g_i.
    fn scale(&self, vector: &GriessVector, generator_idx: usize, theta: f64) -> Result<GriessVector> {
        let exp_g = self.exp_map(generator_idx, theta)?;
        exp_g(vector)
    }

    /// Adjoint representation: ad(g_i)(g_j) = [g_i, g_j]
    ///
    /// The adjoint maps generators to linear transformations of the algebra.
    fn adjoint(&self, i: usize, j: usize) -> Result<GriessVector> {
        self.bracket(i, j)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_ring_trait_exists() {
        // This test just verifies the trait compiles
        // Actual implementations will be tested in division.rs
    }

    #[test]
    fn test_lattice_trait_exists() {
        // This test just verifies the trait compiles
        // Actual implementations will be tested in lattice.rs
    }

    #[test]
    fn test_lie_algebra_trait_exists() {
        // This test just verifies the trait compiles
        // Actual implementations will be tested in lie.rs
    }
}
