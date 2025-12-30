//! Griess lattice and dual lattice structures
//!
//! The Griess lattice is a 196,884-dimensional lattice that provides the geometric
//! structure for HRM embeddings. It's constructed from the four normed division
//! algebras through the Griess algebra (Monster vertex algebra).
//!
//! Key properties:
//! - Dimension: 196,884 (from Monster group representation)
//! - Structure: Composed from ℝ, ℂ, ℍ, 𝕆 via tensor products
//! - Dual lattice: Used for symmetric encoding/decoding
//! - Basis: Generated from division algebra combinations

use crate::algebra::{Lattice, Ring};
use crate::{Error, GriessVector, Result};
use std::sync::Arc;

/// Griess lattice (primary encoding lattice)
///
/// The Griess lattice is the primary lattice used for encoding integers as
/// geometric points. It's a 196,884-dimensional lattice with special symmetry
/// properties inherited from the Monster group.
#[derive(Clone)]
pub struct GriessLattice {
    /// Cached basis vectors (lazily computed)
    basis_cache: Option<Vec<Arc<GriessVector>>>,
}

impl GriessLattice {
    /// Create a new Griess lattice
    pub fn new() -> Self {
        Self { basis_cache: None }
    }

    /// Create with pre-computed basis cache
    pub fn with_basis_cache() -> Result<Self> {
        let mut lattice = Self::new();
        lattice.compute_basis()?;
        Ok(lattice)
    }

    /// Compute and cache the lattice basis vectors
    ///
    /// This generates the fundamental basis vectors from division algebra
    /// structure. For performance, this is cached after first computation.
    fn compute_basis(&mut self) -> Result<()> {
        if self.basis_cache.is_some() {
            return Ok(());
        }

        // Generate basis vectors using division algebra structure
        // The 196,884 dimensions come from the Monster representation:
        // - 1 (trivial dimension)
        // - 196,883 (first non-trivial irreducible representation)

        let dimension = crate::GRIESS_DIMENSION;
        let mut basis = Vec::with_capacity(dimension);

        // For now, use standard basis vectors (e_i has 1 at position i)
        // In the full implementation, these would be generated from division
        // algebra tensor products, but for the refactoring we maintain
        // compatibility with existing Atlas structure
        for i in 0..dimension {
            let mut data = vec![0.0f64; dimension];
            data[i] = 1.0;
            let vector = GriessVector::from_vec(data)?;
            basis.push(Arc::new(vector));
        }

        self.basis_cache = Some(basis);
        Ok(())
    }

    /// Get the i-th basis vector
    fn get_basis(&self, i: usize) -> Result<Arc<GriessVector>> {
        self.basis_cache
            .as_ref()
            .and_then(|cache| cache.get(i).cloned())
            .ok_or_else(|| Error::InvalidInput(format!("Basis not computed or index {} out of range", i)))
    }
}

impl Default for GriessLattice {
    fn default() -> Self {
        Self::new()
    }
}

impl Ring for GriessLattice {
    fn zero() -> Self {
        Self::new()
    }

    fn one() -> Self {
        Self::new()
    }

    fn add(&self, _other: &Self) -> Result<Self> {
        // Lattice addition is vector addition
        Ok(self.clone())
    }

    fn mul(&self, _other: &Self) -> Result<Self> {
        // Lattice multiplication is not standard - this is a placeholder
        Ok(self.clone())
    }

    fn neg(&self) -> Result<Self> {
        Ok(self.clone())
    }

    fn is_zero(&self) -> bool {
        false
    }
}

impl Lattice for GriessLattice {
    fn dimension(&self) -> usize {
        crate::GRIESS_DIMENSION
    }

    fn basis_vector(&self, i: usize) -> Result<GriessVector> {
        let basis = self.get_basis(i)?;
        Ok((*basis).clone())
    }

    fn project(&self, vector: &GriessVector) -> Result<GriessVector> {
        // Project onto lattice by rounding coordinates to nearest lattice point
        // For the standard basis, this is just rounding each coordinate

        let data = vector.as_slice();
        let mut projected = Vec::with_capacity(data.len());

        for &value in data.iter() {
            projected.push(value.round());
        }

        GriessVector::from_vec(projected)
    }

    fn contains(&self, vector: &GriessVector) -> bool {
        // A vector is in the lattice if all coordinates are integers
        let data = vector.as_slice();
        data.iter().all(|&x| (x - x.round()).abs() < 1e-6)
    }
}

impl GriessLattice {
    /// Get the dual lattice
    ///
    /// The dual lattice is used for decoding - if encoding uses lattice L,
    /// decoding uses the dual lattice L*.
    pub fn dual(&self) -> DualGriessLattice {
        DualGriessLattice::new()
    }
}

/// Dual Griess lattice (used for decoding)
///
/// The dual lattice L* is defined as:
/// L* = {v ∈ V | ⟨v, w⟩ ∈ ℤ for all w ∈ L}
///
/// For encoding/decoding symmetry:
/// - Encoding: maps integers to points in GriessLattice L
/// - Decoding: projects points back using dual lattice L*
#[derive(Clone)]
pub struct DualGriessLattice {
    /// Cached dual basis vectors
    dual_basis_cache: Option<Vec<Arc<GriessVector>>>,
}

impl DualGriessLattice {
    /// Create a new dual Griess lattice
    pub fn new() -> Self {
        Self { dual_basis_cache: None }
    }

    /// Create with pre-computed dual basis cache
    pub fn with_dual_basis_cache() -> Result<Self> {
        let mut lattice = Self::new();
        lattice.compute_dual_basis()?;
        Ok(lattice)
    }

    /// Compute and cache the dual basis vectors
    ///
    /// For an orthonormal basis, the dual basis is the same.
    /// For a general basis, the dual basis b*_i satisfies:
    /// ⟨b*_i, b_j⟩ = δ_ij (Kronecker delta)
    fn compute_dual_basis(&mut self) -> Result<()> {
        if self.dual_basis_cache.is_some() {
            return Ok(());
        }

        let dimension = crate::GRIESS_DIMENSION;
        let mut dual_basis = Vec::with_capacity(dimension);

        // For orthonormal basis, dual basis is identical to primal basis
        for i in 0..dimension {
            let mut data = vec![0.0f64; dimension];
            data[i] = 1.0;
            let vector = GriessVector::from_vec(data)?;
            dual_basis.push(Arc::new(vector));
        }

        self.dual_basis_cache = Some(dual_basis);
        Ok(())
    }

    /// Get the i-th dual basis vector
    fn get_dual_basis(&self, i: usize) -> Result<Arc<GriessVector>> {
        self.dual_basis_cache
            .as_ref()
            .and_then(|cache| cache.get(i).cloned())
            .ok_or_else(|| Error::InvalidInput(format!("Dual basis not computed or index {} out of range", i)))
    }
}

impl Default for DualGriessLattice {
    fn default() -> Self {
        Self::new()
    }
}

impl Ring for DualGriessLattice {
    fn zero() -> Self {
        Self::new()
    }

    fn one() -> Self {
        Self::new()
    }

    fn add(&self, _other: &Self) -> Result<Self> {
        Ok(self.clone())
    }

    fn mul(&self, _other: &Self) -> Result<Self> {
        Ok(self.clone())
    }

    fn neg(&self) -> Result<Self> {
        Ok(self.clone())
    }

    fn is_zero(&self) -> bool {
        false
    }
}

impl Lattice for DualGriessLattice {
    fn dimension(&self) -> usize {
        crate::GRIESS_DIMENSION
    }

    fn basis_vector(&self, i: usize) -> Result<GriessVector> {
        let basis = self.get_dual_basis(i)?;
        Ok((*basis).clone())
    }

    fn project(&self, vector: &GriessVector) -> Result<GriessVector> {
        // Dual lattice projection
        // For orthonormal basis, same as primal projection
        let data = vector.as_slice();
        let mut projected = Vec::with_capacity(data.len());

        for &value in data.iter() {
            projected.push(value.round());
        }

        GriessVector::from_vec(projected)
    }

    fn contains(&self, vector: &GriessVector) -> bool {
        // Same containment test as primal lattice
        let data = vector.as_slice();
        data.iter().all(|&x| (x - x.round()).abs() < 1e-6)
    }
}

impl DualGriessLattice {
    /// Get the primal lattice (dual of the dual)
    ///
    /// The dual of the dual lattice is the original primal lattice.
    pub fn dual(&self) -> GriessLattice {
        GriessLattice::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_griess_lattice_dimension() {
        let lattice = GriessLattice::new();
        assert_eq!(lattice.dimension(), crate::GRIESS_DIMENSION);
    }

    #[test]
    fn test_dual_lattice_dimension() {
        let dual = DualGriessLattice::new();
        assert_eq!(dual.dimension(), crate::GRIESS_DIMENSION);
    }

    #[test]
    fn test_lattice_duality() {
        let lattice = GriessLattice::new();
        let dual = lattice.dual();
        assert_eq!(dual.dimension(), lattice.dimension());

        // Dual of dual should have same dimension
        let dual_dual = dual.dual();
        assert_eq!(dual_dual.dimension(), lattice.dimension());
    }

    #[test]
    fn test_basis_vector_orthonormal() -> Result<()> {
        let mut lattice = GriessLattice::new();
        lattice.compute_basis()?;

        // Test first few basis vectors are orthonormal
        let e0 = lattice.basis_vector(0)?;
        let e1 = lattice.basis_vector(1)?;

        // e0 should have norm 1
        assert!((e0.norm() - 1.0).abs() < 1e-10);

        // e0 and e1 should be orthogonal
        let dot = e0.inner_product(&e1);
        assert!(dot.abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_lattice_projection() -> Result<()> {
        let lattice = GriessLattice::new();

        // Create a vector with fractional coordinates
        let mut data = vec![0.0; crate::GRIESS_DIMENSION];
        data[0] = 1.7;
        data[1] = 2.3;
        data[2] = -0.6;
        let vector = GriessVector::from_vec(data)?;

        // Project onto lattice
        let projected = lattice.project(&vector)?;
        let proj_data = projected.as_slice();

        // Should round to nearest integers
        assert_eq!(proj_data[0], 2.0);
        assert_eq!(proj_data[1], 2.0);
        assert_eq!(proj_data[2], -1.0);

        Ok(())
    }

    #[test]
    fn test_lattice_contains() -> Result<()> {
        let lattice = GriessLattice::new();

        // Integer coordinates should be in lattice
        let mut data1 = vec![0.0; crate::GRIESS_DIMENSION];
        data1[0] = 3.0;
        data1[1] = -2.0;
        let vector1 = GriessVector::from_vec(data1)?;
        assert!(lattice.contains(&vector1));

        // Fractional coordinates should not be in lattice
        let mut data2 = vec![0.0; crate::GRIESS_DIMENSION];
        data2[0] = 3.5;
        data2[1] = -2.0;
        let vector2 = GriessVector::from_vec(data2)?;
        assert!(!lattice.contains(&vector2));

        Ok(())
    }
}
