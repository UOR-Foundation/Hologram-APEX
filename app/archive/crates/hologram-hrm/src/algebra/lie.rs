//! Lie algebra for MoonshineHRM scaling operations
//!
//! The MoonshineAlgebra provides the scaling mechanism for HRM embeddings.
//! It consists of 96 generators (one per resonance class k ∈ ℤ₉₆) that
//! form a Lie algebra structure.
//!
//! Key operations:
//! - Lie bracket: [g_i, g_j] measures non-commutativity
//! - Exponential map: exp(θ·g_i) creates group element from algebra element
//! - Scaling: Applies exp(θ·g_i) to vectors to modulate representation space
//!
//! The Lie algebra structure corresponds to the (h₂, d, ℓ) decomposition:
//! - h₂ ∈ ℤ₄: Quaternionic generators (4 directions)
//! - d ∈ ℤ₃: Octonionic generators (3 triality states)
//! - ℓ ∈ ℤ₈: Clifford generators (8 context slots)

#![allow(clippy::needless_range_loop)]

use crate::algebra::LieAlgebra;
use crate::atlas::prng::SplitMix64;
use crate::{Error, GriessVector, Result, GRIESS_DIMENSION};
use std::sync::Arc;

/// Number of generators in the Moonshine Lie algebra
pub const NUM_GENERATORS: usize = 96;

/// MoonshineAlgebra: 96-dimensional Lie algebra for HRM scaling
///
/// The algebra has structure ℤ₄ × ℤ₃ × ℤ₈ = 96, corresponding to:
/// - 4 quaternionic directions (h₂)
/// - 3 octonionic triality states (d)
/// - 8 Clifford context slots (ℓ)
pub struct MoonshineAlgebra {
    /// Cached generators (lazily computed)
    generators: Option<Vec<Arc<GriessVector>>>,

    /// Structure constants: c^k_ij where [g_i, g_j] = Σ_k c^k_ij g_k
    /// Stored as sparse map for efficiency
    structure_constants: Option<Vec<Vec<Vec<f64>>>>,
}

impl MoonshineAlgebra {
    /// Create a new Moonshine Lie algebra
    pub fn new() -> Self {
        Self {
            generators: None,
            structure_constants: None,
        }
    }

    /// Create with pre-computed generators and structure constants
    pub fn with_cache() -> Result<Self> {
        let mut algebra = Self::new();
        algebra.compute_generators()?;
        algebra.compute_structure_constants()?;
        Ok(algebra)
    }

    /// Compute and cache the 96 generators
    ///
    /// Each generator g_i corresponds to a resonance class i ∈ [0, 95].
    /// Generators are constructed from the (h₂, d, ℓ) decomposition.
    fn compute_generators(&mut self) -> Result<()> {
        if self.generators.is_some() {
            return Ok(());
        }

        let mut generators = Vec::with_capacity(NUM_GENERATORS);

        for class in 0..NUM_GENERATORS {
            let generator = self.generate_lie_vector(class as u8)?;
            generators.push(Arc::new(generator));
        }

        self.generators = Some(generators);
        Ok(())
    }

    /// Generate a Lie algebra generator for a given resonance class
    ///
    /// The generator is constructed based on the (h₂, d, ℓ) decomposition:
    /// - class = 24·h₂ + 8·d + ℓ
    ///
    /// The generator encodes directional information in three regions:
    /// - H-region (dims 0..65,628): h₂ influence (quaternionic)
    /// - D-region (dims 65,628..131,256): d influence (octonionic triality)
    /// - L-region (dims 131,256..196,884): ℓ influence (Clifford context)
    fn generate_lie_vector(&self, class: u8) -> Result<GriessVector> {
        // Decompose class into (h₂, d, ℓ)
        let h2 = class / 24;
        let remainder = class % 24;
        let d = remainder / 8;
        let ell = remainder % 8;

        let mut data = vec![0.0f64; GRIESS_DIMENSION];
        let mut rng = SplitMix64::new(class as u64 + 1000);

        // Region boundaries (roughly 1/3 each)
        let h_end = GRIESS_DIMENSION / 3;
        let d_start = h_end;
        let d_end = 2 * GRIESS_DIMENSION / 3;
        let l_start = d_end;

        // H-region: Quaternionic influence (h₂ ∈ {0,1,2,3})
        let theta_h = 2.0 * std::f64::consts::PI * (h2 as f64) / 4.0;
        for (i, value) in data.iter_mut().enumerate().take(h_end) {
            let phase = (i as f64 / h_end as f64) * theta_h;
            let base = phase.cos();
            let noise = rng.next_f64() * 0.1 - 0.05;
            *value = base + noise;
        }

        // D-region: Octonionic triality (d ∈ {0,1,2})
        let theta_d = 2.0 * std::f64::consts::PI * (d as f64) / 3.0;
        for (i, value) in data.iter_mut().enumerate().skip(d_start).take(d_end - d_start) {
            let idx = i - d_start;
            let phase = (idx as f64 / (d_end - d_start) as f64) * theta_d;
            let base = phase.sin();
            let noise = rng.next_f64() * 0.1 - 0.05;
            *value = base + noise;
        }

        // L-region: Clifford context (ℓ ∈ {0,1,2,3,4,5,6,7})
        let theta_l = 2.0 * std::f64::consts::PI * (ell as f64) / 8.0;
        for (i, value) in data.iter_mut().enumerate().skip(l_start) {
            let idx = i - l_start;
            let phase = (idx as f64 / (GRIESS_DIMENSION - l_start) as f64) * theta_l;
            let base = (phase.cos() + phase.sin()) / std::f64::consts::SQRT_2;
            let noise = rng.next_f64() * 0.05 - 0.025;
            *value = base + noise;
        }

        GriessVector::from_vec(data)
    }

    /// Compute structure constants c^k_ij for Lie bracket
    ///
    /// The Lie bracket is bilinear and antisymmetric:
    /// [g_i, g_j] = Σ_k c^k_ij g_k
    ///
    /// For computational efficiency, we use an approximation based on
    /// the (h₂, d, ℓ) structure rather than computing all 96³ constants.
    fn compute_structure_constants(&mut self) -> Result<()> {
        if self.structure_constants.is_some() {
            return Ok(());
        }

        // Initialize 96×96×96 tensor (sparse representation)
        let mut constants = vec![vec![vec![0.0; NUM_GENERATORS]; NUM_GENERATORS]; NUM_GENERATORS];

        // Compute structure constants based on (h₂, d, ℓ) commutation relations
        for i in 0..NUM_GENERATORS {
            for j in (i + 1)..NUM_GENERATORS {
                // Decompose indices
                let (h2_i, d_i, ell_i) = self.decompose_class(i as u8);
                let (h2_j, d_j, ell_j) = self.decompose_class(j as u8);

                // Compute bracket based on component differences
                // The bracket is stronger when components differ more
                let h_diff = ((h2_i as i32 - h2_j as i32).abs()) as f64 / 4.0;
                let d_diff = ((d_i as i32 - d_j as i32).abs()) as f64 / 3.0;
                let ell_diff = ((ell_i as i32 - ell_j as i32).abs()) as f64 / 8.0;

                let strength = h_diff + d_diff + ell_diff;

                // Result class is modular combination
                let k = ((h2_i + h2_j) % 4) * 24 + ((d_i + d_j) % 3) * 8 + ((ell_i + ell_j) % 8);

                // [g_i, g_j] = strength * g_k
                constants[k as usize][i][j] = strength;
                constants[k as usize][j][i] = -strength; // Antisymmetry
            }
        }

        self.structure_constants = Some(constants);
        Ok(())
    }

    /// Decompose class index into (h₂, d, ℓ) coordinates
    fn decompose_class(&self, class: u8) -> (u8, u8, u8) {
        let h2 = class / 24;
        let remainder = class % 24;
        let d = remainder / 8;
        let ell = remainder % 8;
        (h2, d, ell)
    }

    /// Get cached generator
    fn get_generator(&self, i: usize) -> Result<Arc<GriessVector>> {
        self.generators
            .as_ref()
            .and_then(|gens| gens.get(i).cloned())
            .ok_or_else(|| Error::InvalidInput(format!("Generators not computed or index {} out of range", i)))
    }
}

impl Default for MoonshineAlgebra {
    fn default() -> Self {
        Self::new()
    }
}

impl LieAlgebra for MoonshineAlgebra {
    fn num_generators(&self) -> usize {
        NUM_GENERATORS
    }

    fn generator(&self, i: usize) -> Result<GriessVector> {
        if i >= NUM_GENERATORS {
            return Err(Error::InvalidInput(format!(
                "Generator index {} out of range [0, {})",
                i, NUM_GENERATORS
            )));
        }

        let gen = self.get_generator(i)?;
        Ok((*gen).clone())
    }

    fn bracket(&self, i: usize, j: usize) -> Result<GriessVector> {
        if i >= NUM_GENERATORS || j >= NUM_GENERATORS {
            return Err(Error::InvalidInput(format!(
                "Generator indices ({}, {}) out of range [0, {})",
                i, j, NUM_GENERATORS
            )));
        }

        // [g_i, g_j] = Σ_k c^k_ij g_k
        let constants = self
            .structure_constants
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("Structure constants not computed".to_string()))?;

        let mut result_data = vec![0.0; GRIESS_DIMENSION];

        // Sum over all generators weighted by structure constants
        for k in 0..NUM_GENERATORS {
            let c_k_ij = constants[k][i][j];
            if c_k_ij.abs() > 1e-10 {
                let g_k = self.get_generator(k)?;
                let g_k_data = g_k.as_slice();

                for (idx, value) in result_data.iter_mut().enumerate() {
                    *value += c_k_ij * g_k_data[idx];
                }
            }
        }

        GriessVector::from_vec(result_data)
    }

    fn exp_map(&self, generator_idx: usize, theta: f64) -> Result<Box<dyn Fn(&GriessVector) -> Result<GriessVector>>> {
        if generator_idx >= NUM_GENERATORS {
            return Err(Error::InvalidInput(format!(
                "Generator index {} out of range",
                generator_idx
            )));
        }

        let generator = self.get_generator(generator_idx)?;

        // The exponential map exp(θ·g) acting on vector v
        // We use the approximation: exp(θ·g)·v ≈ v + θ·⟨g, v⟩·g
        // This is valid for small θ or when g is small compared to v
        Ok(Box::new(move |v: &GriessVector| -> Result<GriessVector> {
            let v_data = v.as_slice();
            let g_data = generator.as_slice();

            // Compute inner product ⟨g, v⟩
            let mut dot_product = 0.0;
            for i in 0..GRIESS_DIMENSION {
                dot_product += g_data[i] * v_data[i];
            }

            // Result: v + θ·⟨g,v⟩·g
            let mut result_data = vec![0.0; GRIESS_DIMENSION];
            for i in 0..GRIESS_DIMENSION {
                result_data[i] = v_data[i] + theta * dot_product * g_data[i];
            }

            GriessVector::from_vec(result_data)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moonshine_algebra_creation() {
        let algebra = MoonshineAlgebra::new();
        assert_eq!(algebra.num_generators(), NUM_GENERATORS);
    }

    #[test]
    fn test_generator_computation() -> Result<()> {
        let mut algebra = MoonshineAlgebra::new();
        algebra.compute_generators()?;

        // Should have 96 generators
        assert_eq!(algebra.generators.as_ref().unwrap().len(), 96);

        // Each generator should have correct dimension
        let g0 = algebra.generator(0)?;
        assert_eq!(g0.as_slice().len(), GRIESS_DIMENSION);

        Ok(())
    }

    #[test]
    fn test_bracket_antisymmetry() -> Result<()> {
        let algebra = MoonshineAlgebra::with_cache()?;

        // Test [g_i, g_j] = -[g_j, g_i]
        let bracket_ij = algebra.bracket(5, 10)?;
        let bracket_ji = algebra.bracket(10, 5)?;

        let ij_data = bracket_ij.as_slice();
        let ji_data = bracket_ji.as_slice();

        for i in 0..GRIESS_DIMENSION {
            assert!((ij_data[i] + ji_data[i]).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_bracket_self_zero() -> Result<()> {
        let algebra = MoonshineAlgebra::with_cache()?;

        // Test [g_i, g_i] = 0
        let bracket = algebra.bracket(7, 7)?;
        let data = bracket.as_slice();

        for &value in data.iter() {
            assert!(value.abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_exponential_map() -> Result<()> {
        let algebra = MoonshineAlgebra::with_cache()?;

        // Create a test vector
        let mut v_data = vec![0.0; GRIESS_DIMENSION];
        v_data[0] = 1.0;
        v_data[1] = 2.0;
        let v = GriessVector::from_vec(v_data)?;

        // Apply exp map
        let theta = 0.1;
        let exp_g = algebra.exp_map(5, theta)?;
        let result = exp_g(&v)?;

        // Result should have same dimension
        assert_eq!(result.as_slice().len(), GRIESS_DIMENSION);

        Ok(())
    }

    #[test]
    fn test_scale_operation() -> Result<()> {
        let algebra = MoonshineAlgebra::with_cache()?;

        // Create identity-like vector
        let v = GriessVector::identity();

        // Scale using generator 0
        let scaled = algebra.scale(&v, 0, 0.5)?;

        // Scaled vector should be different from original
        let v_norm = v.norm();
        let scaled_norm = scaled.norm();
        assert!((v_norm - scaled_norm).abs() > 1e-6);

        Ok(())
    }

    #[test]
    fn test_decompose_class() {
        let algebra = MoonshineAlgebra::new();

        // Test class 0: (0, 0, 0)
        assert_eq!(algebra.decompose_class(0), (0, 0, 0));

        // Test class 1: (0, 0, 1)
        assert_eq!(algebra.decompose_class(1), (0, 0, 1));

        // Test class 8: (0, 1, 0)
        assert_eq!(algebra.decompose_class(8), (0, 1, 0));

        // Test class 24: (1, 0, 0)
        assert_eq!(algebra.decompose_class(24), (1, 0, 0));

        // Test class 95: (3, 2, 7)
        assert_eq!(algebra.decompose_class(95), (3, 2, 7));
    }
}
