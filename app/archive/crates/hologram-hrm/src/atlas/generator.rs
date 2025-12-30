//! Atlas vector generation
//!
//! This module generates the 96 canonical Atlas vectors deterministically
//! based on the (h₂, d, ℓ) coordinate system from hologram-compiler.
//! Each vector is 196,884-dimensional and represents one equivalence class
//! in the (ℤ₄ × ℤ₃ × ℤ₈) quotient structure.
//!
//! # Generation Process
//!
//! For each class index (0..95):
//! 1. Decode class to (h₂, d, ℓ) components using hologram-compiler
//! 2. Compute deterministic seed from coordinates
//! 3. Generate structured vector where each coordinate influences specific dimensions:
//!    - h₂ ∈ {0,1,2,3} influences dimensions 0..49,221 (quaternionic structure)
//!    - d ∈ {0,1,2} influences dimensions 49,221..131,072 (octonionic/triality structure)
//!    - ℓ ∈ {0,1,2,3,4,5,6,7} influences dimensions 131,072..196,884 (Clifford/context structure)
//! 4. Normalize to unit length (L2 norm = 1)
//!
//! # Properties
//!
//! - **Deterministic**: Same class always produces same vector
//! - **Reproducible**: Generated from mathematical formula, not random
//! - **Canonical**: Each vector represents one equivalence class
//! - **Structured**: Respects (h₂, d, ℓ) coordinate system

use crate::griess::{scalar_mul, GriessVector};
use crate::{Error, Result, ATLAS_CLASSES, GRIESS_DIMENSION};

use super::prng::SplitMix64;

// Import hologram-compiler types for (h₂, d, ℓ) decoding
// Note: This creates a dependency on hologram-compiler
// We use dynamic loading via function pointer to avoid circular deps
// For now, we implement the decoding directly to avoid dependency issues

/// Modality component (d ∈ {0, 1, 2})
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Modality {
    /// Neutral modality (d = 0)
    Neutral = 0,
    /// Produce modality (d = 1)
    Produce = 1,
    /// Consume modality (d = 2)
    Consume = 2,
}

impl Modality {
    /// Convert modality to u8 value
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Create modality from u8 value
    ///
    /// # Panics
    ///
    /// Panics if value is not 0, 1, or 2
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Modality::Neutral,
            1 => Modality::Produce,
            2 => Modality::Consume,
            _ => panic!("Invalid modality value: {}", value),
        }
    }
}

/// Sigil components: (h₂, d, ℓ) triple
///
/// Represents the three coordinates in the (ℤ₄ × ℤ₃ × ℤ₈) quotient structure:
/// - h₂ ∈ {0,1,2,3}: Quaternionic scope quadrant
/// - d ∈ {0,1,2}: Triality/octonionic modality
/// - ℓ ∈ {0,1,2,3,4,5,6,7}: Clifford context slot
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SigilComponents {
    /// Scope quadrant (h₂ ∈ {0,1,2,3})
    pub h2: u8,
    /// Modality (d ∈ {Neutral, Produce, Consume})
    pub d: Modality,
    /// Context slot (ℓ ∈ {0,1,2,3,4,5,6,7})
    pub l: u8,
}

impl SigilComponents {
    /// Create new SigilComponents
    ///
    /// # Panics
    ///
    /// Panics if h2 >= 4 or l >= 8
    pub fn new(h2: u8, d: Modality, l: u8) -> Self {
        assert!(h2 < 4, "h2 must be in [0, 3], got {}", h2);
        assert!(l < 8, "l must be in [0, 7], got {}", l);
        Self { h2, d, l }
    }

    /// Encode components to class index
    ///
    /// Formula: class = 24*h₂ + 8*d + ℓ
    pub fn to_class_index(&self) -> u8 {
        24 * self.h2 + 8 * self.d.as_u8() + self.l
    }

    /// Decode class index to components
    ///
    /// Formula inverse: h₂ = class / 24, d = (class % 24) / 8, ℓ = class % 8
    pub fn from_class_index(class_index: u8) -> Self {
        decode_class_index(class_index)
    }
}

/// Decode class index to (h₂, d, ℓ) components
///
/// Formula: class = 24*h₂ + 8*d + ℓ
/// Inverse: h₂ = class / 24, d = (class % 24) / 8, ℓ = class % 8
fn decode_class_index(class_index: u8) -> SigilComponents {
    assert!(class_index < 96, "Class index {} out of range [0..95]", class_index);

    let h2 = class_index / 24;
    let remainder = class_index % 24;
    let d_val = remainder / 8;
    let l = remainder % 8;

    let d = match d_val {
        0 => Modality::Neutral,
        1 => Modality::Produce,
        2 => Modality::Consume,
        _ => unreachable!("Invalid modality value: {}", d_val),
    };

    SigilComponents { h2, d, l }
}

// Dimension boundaries for coordinate influence regions
const H2_END: usize = 49_221;
const D_START: usize = H2_END;
const D_END: usize = 131_072;
const L_START: usize = D_END;
const L_END: usize = GRIESS_DIMENSION;

/// Generate a canonical Atlas vector for the given class
///
/// This generates a 196,884-dimensional vector deterministically based on
/// the (h₂, d, ℓ) coordinate structure. Each coordinate influences specific
/// dimensions of the vector, creating a structured canonical representative.
///
/// # Arguments
///
/// * `class` - The base-96 class index (must be in [0, 95])
///
/// # Returns
///
/// A normalized GriessVector representing the canonical vector for this class.
///
/// # Errors
///
/// Returns `Error::ClassOutOfRange` if class >= 96.
///
/// # Example
///
/// ```ignore
/// use hologram_hrm::atlas::generate_atlas_vector;
///
/// // Generate the canonical vector for class 42
/// let vector = generate_atlas_vector(42)?;
/// assert_eq!(vector.len(), 196_884);
///
/// // Vectors are normalized (unit length)
/// let norm = vector.norm();
/// assert!((norm - 1.0).abs() < 1e-6);
/// ```
pub fn generate_atlas_vector(class: u8) -> Result<GriessVector> {
    if class >= ATLAS_CLASSES {
        return Err(Error::ClassOutOfRange(class));
    }

    // Decode class to (h₂, d, ℓ) components
    let comp = decode_class_index(class);

    // Compute deterministic seed from coordinates
    let seed = compute_coordinate_seed(&comp);

    // Generate structured vector based on (h₂, d, ℓ)
    let vector = generate_structured_vector(seed, &comp)?;

    // Normalize to unit length (L2 norm = 1)
    let norm = vector.norm();
    if norm < 1e-10 {
        return Err(Error::DecodingFailed(format!(
            "Generated zero vector for class {} (h₂={}, d={}, ℓ={})",
            class,
            comp.h2,
            comp.d.as_u8(),
            comp.l
        )));
    }

    // Return normalized vector
    scalar_mul(&vector, 1.0 / norm)
}

/// Compute deterministic seed from (h₂, d, ℓ) coordinates
///
/// This packs the three coordinates into a single u64 seed that uniquely
/// identifies the equivalence class.
fn compute_coordinate_seed(comp: &SigilComponents) -> u64 {
    // Pack coordinates into seed: h₂ in high bits, d in middle, ℓ in low
    ((comp.h2 as u64) << 32) | ((comp.d.as_u8() as u64) << 16) | (comp.l as u64)
}

/// Generate a structured vector based on (h₂, d, ℓ) coordinates
///
/// Each coordinate influences its designated region of the 196,884-dimensional space:
/// - h₂ influences dimensions 0..49,221 (quaternionic structure)
/// - d influences dimensions 49,221..131,072 (octonionic structure)
/// - ℓ influences dimensions 131,072..196,884 (Clifford structure)
fn generate_structured_vector(seed: u64, comp: &SigilComponents) -> Result<GriessVector> {
    let mut data = vec![0.0; GRIESS_DIMENSION];
    let mut rng = SplitMix64::new(seed);

    // Generate base random values
    for (i, value) in data.iter_mut().enumerate() {
        let base_value = rng.next_f64();

        // Apply coordinate-specific weights
        let weight = if i < H2_END {
            // Quaternionic region: influenced by h₂
            weight_quaternionic(i, comp.h2)
        } else if i < D_END {
            // Octonionic region: influenced by d
            weight_octonionic(i, comp.d.as_u8())
        } else {
            // Clifford region: influenced by ℓ
            weight_clifford(i, comp.l)
        };

        *value = base_value * weight;
    }

    GriessVector::from_vec(data)
}

/// Compute weight for quaternionic region based on h₂ coordinate
///
/// h₂ ∈ {0, 1, 2, 3} creates distinct patterns in the quaternionic subspace
fn weight_quaternionic(dim_index: usize, h2: u8) -> f64 {
    debug_assert!(dim_index < H2_END);
    debug_assert!(h2 < 4);

    // Create phase shift based on h₂ value
    // This makes vectors for different h₂ values distinguishable
    let phase = (h2 as f64) * std::f64::consts::PI / 2.0;
    let position = (dim_index as f64) / (H2_END as f64);

    // Sinusoidal modulation creates structure
    1.0 + 0.5 * (position * 2.0 * std::f64::consts::PI + phase).sin()
}

/// Compute weight for octonionic region based on d coordinate
///
/// d ∈ {0, 1, 2} creates triality structure in the octonionic subspace
fn weight_octonionic(dim_index: usize, d: u8) -> f64 {
    debug_assert!((D_START..D_END).contains(&dim_index));
    debug_assert!(d < 3);

    // Create triality structure (3-fold symmetry)
    let phase = (d as f64) * 2.0 * std::f64::consts::PI / 3.0;
    let position = ((dim_index - D_START) as f64) / ((D_END - D_START) as f64);

    // Triality modulation
    1.0 + 0.5 * (position * 3.0 * std::f64::consts::PI + phase).sin()
}

/// Compute weight for Clifford region based on ℓ coordinate
///
/// ℓ ∈ {0, 1, ..., 7} creates 8-fold structure in the Clifford subspace
fn weight_clifford(dim_index: usize, l: u8) -> f64 {
    debug_assert!((L_START..L_END).contains(&dim_index));
    debug_assert!(l < 8);

    // Create 8-fold context structure
    let phase = (l as f64) * std::f64::consts::PI / 4.0;
    let position = ((dim_index - L_START) as f64) / ((L_END - L_START) as f64);

    // Clifford modulation (8-fold symmetry)
    1.0 + 0.5 * (position * 4.0 * std::f64::consts::PI + phase).sin()
}

/// Generate all 96 Atlas vectors
///
/// This generates the complete Atlas partition: all 96 canonical vectors.
/// Each vector is 196,884-dimensional and normalized to unit length.
///
/// # Returns
///
/// A Vec of 96 GriessVectors, indexed by class (0-95).
///
/// # Example
///
/// ```ignore
/// use hologram_hrm::atlas::generate_all_atlas_vectors;
///
/// let atlas = generate_all_atlas_vectors()?;
/// assert_eq!(atlas.len(), 96);
///
/// // Each vector is normalized
/// for vector in &atlas {
///     let norm = vector.norm();
///     assert!((norm - 1.0).abs() < 1e-6);
/// }
/// ```
pub fn generate_all_atlas_vectors() -> Result<Vec<GriessVector>> {
    (0..ATLAS_CLASSES).map(generate_atlas_vector).collect()
}

/// Check if two Atlas vectors are orthogonal (within tolerance)
///
/// # Example
///
/// ```ignore
/// let v1 = generate_atlas_vector(0)?;
/// let v2 = generate_atlas_vector(1)?;
///
/// // Atlas vectors are not necessarily orthogonal
/// let is_orth = are_orthogonal(&v1, &v2, 1e-6);
/// ```
pub fn are_orthogonal(a: &GriessVector, b: &GriessVector, tolerance: f64) -> bool {
    let dot = a.inner_product(b);
    dot.abs() < tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_atlas_vector_valid_class() {
        let vector = generate_atlas_vector(0).unwrap();
        assert_eq!(vector.len(), GRIESS_DIMENSION);
    }

    #[test]
    fn test_generate_atlas_vector_invalid_class() {
        let result = generate_atlas_vector(96);
        assert!(matches!(result, Err(Error::ClassOutOfRange(96))));
    }

    #[test]
    fn test_atlas_vector_normalized() {
        let vector = generate_atlas_vector(42).unwrap();
        let norm = vector.norm();
        assert!((norm - 1.0).abs() < 1e-6, "Vector not normalized: norm = {}", norm);
    }

    #[test]
    fn test_atlas_vector_deterministic() {
        let v1 = generate_atlas_vector(10).unwrap();
        let v2 = generate_atlas_vector(10).unwrap();

        // Same class should produce identical vectors
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_different_classes_different_vectors() {
        let v1 = generate_atlas_vector(0).unwrap();
        let v2 = generate_atlas_vector(1).unwrap();

        // Different classes should produce different vectors
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_generate_all_atlas_vectors() {
        let atlas = generate_all_atlas_vectors().unwrap();

        assert_eq!(atlas.len(), 96);

        // All vectors should be normalized
        for (i, vector) in atlas.iter().enumerate() {
            let norm = vector.norm();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "Vector {} not normalized: norm = {}",
                i,
                norm
            );
        }
    }

    #[test]
    fn test_atlas_vectors_distinct() {
        let atlas = generate_all_atlas_vectors().unwrap();

        // Check that all vectors are distinct (pairwise different)
        for i in 0..96 {
            for j in (i + 1)..96 {
                assert_ne!(atlas[i], atlas[j], "Vectors {} and {} are identical", i, j);
            }
        }
    }

    #[test]
    fn test_atlas_vector_values_in_valid_range() {
        let vector = generate_atlas_vector(0).unwrap();

        // After normalization, values should be reasonable (not NaN, not infinite)
        for &value in vector.as_slice() {
            assert!(value.is_finite(), "Vector contains non-finite value: {}", value);
        }
    }

    #[test]
    fn test_sigil_components_roundtrip() {
        // Test all 96 classes roundtrip correctly
        for class in 0..96 {
            let components = SigilComponents::from_class_index(class);
            let recovered = components.to_class_index();
            assert_eq!(recovered, class, "Class {} failed roundtrip", class);
        }
    }

    #[test]
    fn test_sigil_components_bounds() {
        // Test that all components stay in bounds
        for class in 0..96 {
            let comp = SigilComponents::from_class_index(class);
            assert!(comp.h2 < 4, "h2 out of bounds for class {}: {}", class, comp.h2);
            assert!(
                comp.d.as_u8() < 3,
                "d out of bounds for class {}: {}",
                class,
                comp.d.as_u8()
            );
            assert!(comp.l < 8, "l out of bounds for class {}: {}", class, comp.l);
        }
    }

    #[test]
    fn test_sigil_components_formula() {
        // Verify formula: class = 24*h₂ + 8*d + ℓ
        let test_cases = vec![
            (0, Modality::Neutral, 0, 0),  // class 0
            (0, Modality::Neutral, 7, 7),  // class 7
            (0, Modality::Produce, 0, 8),  // class 8
            (0, Modality::Consume, 0, 16), // class 16
            (1, Modality::Neutral, 0, 24), // class 24
            (3, Modality::Consume, 7, 95), // class 95
        ];

        for (h2, d_mod, l, expected_class) in test_cases {
            let comp = SigilComponents::new(h2, d_mod, l);
            let actual = comp.to_class_index();
            assert_eq!(actual, expected_class, "Failed for h2={}, d={:?}, l={}", h2, d_mod, l);
        }
    }

    #[test]
    fn test_modality_conversion() {
        assert_eq!(Modality::from_u8(0), Modality::Neutral);
        assert_eq!(Modality::from_u8(1), Modality::Produce);
        assert_eq!(Modality::from_u8(2), Modality::Consume);

        assert_eq!(Modality::Neutral.as_u8(), 0);
        assert_eq!(Modality::Produce.as_u8(), 1);
        assert_eq!(Modality::Consume.as_u8(), 2);
    }

    #[test]
    #[should_panic(expected = "Invalid modality value")]
    fn test_modality_invalid() {
        Modality::from_u8(3);
    }
}
