//! UOR (Universal Object Repository) mathematical primitives
//!
//! This module provides the core mathematical operations from atlas-12288:
//! - R96 resonance classification
//! - Φ (Phi) boundary encoding/decoding
//! - Conservation and truth semantics

use crate::{constants::*, AtlasError, Result};

type CoreResonanceClass = atlas_core::ResonanceClass;

/// Classify a byte value to its R96 resonance class [0, 96)
///
/// This implements the R96 classifier from atlas-12288, which maps
/// 256 byte values to 96 resonance classes with 3/8 compression.
///
/// # Example
///
/// ```
/// use atlas_isa::r96_classify;
///
/// let byte_val = 42u8;
/// let class = r96_classify(byte_val);
/// assert!(class.as_u8() < 96);
/// ```
pub fn r96_classify(byte: u8) -> ResonanceClass {
    ResonanceClass::classify(byte)
}

/// Encode (page, byte) boundary coordinates into a single u32 value
///
/// Uses Φ (Phi) boundary encoding for compact representation of
/// coordinates on the (48 × 256) torus.
///
/// # Arguments
///
/// * `page` - Page index in [0, 48)
/// * `byte` - Byte index in [0, 256)
///
/// # Returns
///
/// Encoded coordinate as u32
///
/// # Panics
///
/// Panics if page >= 48. Byte is always valid since it's u8 [0, 255] ⊂ [0, 256).
pub fn phi_encode(page: u8, byte: u8) -> u32 {
    atlas_core::phi_encode(page, byte)
}

/// Decode a Φ-encoded coordinate back to (page, byte)
///
/// # Arguments
///
/// * `encoded` - The encoded coordinate from `phi_encode`
///
/// # Returns
///
/// Tuple of (page, byte)
pub fn phi_decode(encoded: u32) -> (u8, u8) {
    let pair = atlas_core::phi_decode(encoded);
    (pair.page, pair.byte)
}

/// Test truth semantics at budget 0 (conservation law)
///
/// Returns true if the value represents truth at zero budget,
/// i.e., perfect conservation.
pub fn truth_zero(budget: u32) -> bool {
    budget == 0
}

/// Test truth semantics for addition (conservation under addition)
///
/// Returns true if adding a and b conserves truth.
pub fn truth_add(a: u32, b: u32) -> bool {
    // Wrapping addition: overflow to 0 conserves truth
    let (result, overflow) = a.overflowing_add(b);
    result == 0 || overflow
}

/// Resonance class [0, 96)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ResonanceClass(pub(crate) CoreResonanceClass);

impl ResonanceClass {
    /// Create a new resonance class, validating range
    pub fn new(value: u8) -> Result<Self> {
        CoreResonanceClass::new(value).map(Self).map_err(AtlasError::from)
    }

    /// Create resonance class without validation (unsafe)
    ///
    /// # Safety
    ///
    /// Caller must ensure value < 96
    pub unsafe fn new_unchecked(value: u8) -> Self {
        Self(CoreResonanceClass::new_unchecked(value))
    }

    /// Get the underlying u8 value
    pub const fn as_u8(self) -> u8 {
        self.0.as_u8()
    }

    /// Classify a byte to its resonance class
    pub fn classify(byte: u8) -> Self {
        Self(CoreResonanceClass::classify(byte))
    }

    /// Get the mirror pair of this class
    ///
    /// Mirror pairing follows the canonical atlas-core data.
    pub fn mirror(self) -> Self {
        Self(self.0.mirror())
    }

    /// Check if this class is in the unity set
    ///
    /// Delegates to atlas-core unity mask maintained by atlas-embeddings.
    pub fn is_unity(self) -> bool {
        self.0.is_unity()
    }
}

impl From<CoreResonanceClass> for ResonanceClass {
    fn from(inner: CoreResonanceClass) -> Self {
        Self(inner)
    }
}

impl From<ResonanceClass> for CoreResonanceClass {
    fn from(class: ResonanceClass) -> Self {
        class.0
    }
}

/// Boundary coordinate (page, byte) on the (48 × 256) torus
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhiCoordinate {
    pub page: u8,
    pub byte: u8,
}

impl PhiCoordinate {
    /// Create a new coordinate
    pub fn new(page: u8, byte: u8) -> Result<Self> {
        if page >= PAGES as u8 {
            return Err(AtlasError::InvalidPage(page as u32));
        }
        Ok(Self { page, byte })
    }

    /// Create coordinate without validation (unsafe)
    ///
    /// # Safety
    ///
    /// Caller must ensure page < 48 and byte < 256
    pub const unsafe fn new_unchecked(page: u8, byte: u8) -> Self {
        Self { page, byte }
    }

    /// Encode this coordinate using Φ encoding
    pub fn encode(self) -> u32 {
        phi_encode(self.page, self.byte)
    }

    /// Decode from Φ-encoded value
    pub fn decode(encoded: u32) -> Self {
        let (page, byte) = phi_decode(encoded);
        Self { page, byte }
    }

    /// Get linear index into the 12,288-element array
    pub fn linear_index(self) -> usize {
        atlas_core::linear_index(self.page, self.byte)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r96_classify_range() {
        for byte in 0..=255u8 {
            let class = r96_classify(byte);
            assert!(
                class.as_u8() < 96,
                "class {} out of range for byte {}",
                class.as_u8(),
                byte
            );
        }
    }

    #[test]
    fn test_r96_classify_coverage() {
        use std::collections::HashSet;
        let mut classes = HashSet::new();
        for byte in 0..=255u8 {
            classes.insert(r96_classify(byte).as_u8());
        }
        // Should have good coverage (at least 80 unique classes)
        assert!(classes.len() >= 80, "only {} unique classes", classes.len());
    }

    #[test]
    fn test_phi_encode_decode_roundtrip() {
        for page in 0..PAGES as u8 {
            for byte in (0..BYTES_PER_PAGE as u8).step_by(17) {
                let encoded = phi_encode(page, byte);
                let (decoded_page, decoded_byte) = phi_decode(encoded);
                assert_eq!(page, decoded_page, "page mismatch");
                assert_eq!(byte, decoded_byte, "byte mismatch");
            }
        }
    }

    #[test]
    fn test_phi_coordinate() {
        let coord = PhiCoordinate::new(10, 128).unwrap();
        let encoded = coord.encode();
        let decoded = PhiCoordinate::decode(encoded);
        assert_eq!(coord, decoded);
    }

    #[test]
    fn test_phi_coordinate_linear_index() {
        let coord = PhiCoordinate::new(0, 0).unwrap();
        assert_eq!(coord.linear_index(), 0);

        let coord = PhiCoordinate::new(0, 255).unwrap();
        assert_eq!(coord.linear_index(), 255);

        let coord = PhiCoordinate::new(1, 0).unwrap();
        assert_eq!(coord.linear_index(), 256);

        let coord = PhiCoordinate::new(47, 255).unwrap();
        assert_eq!(coord.linear_index(), 12287);
    }

    #[test]
    fn test_truth_zero() {
        assert!(truth_zero(0));
        assert!(!truth_zero(1));
        assert!(!truth_zero(u32::MAX));
    }

    #[test]
    fn test_truth_add() {
        assert!(truth_add(0, 0));
        assert!(truth_add(u32::MAX, 1)); // Wraps to 0
        assert!(!truth_add(1, 1)); // = 2
        assert!(!truth_add(10, 20)); // = 30
    }

    #[test]
    fn test_resonance_class_mirror() {
        let class = ResonanceClass::new(10).unwrap();
        let mirror = class.mirror();

        // Mirror of mirror should return original
        assert_eq!(class, mirror.mirror());

        // Mirror should be different (unless at boundary)
        if class.as_u8() != 47 && class.as_u8() != 48 {
            assert_ne!(class, mirror);
        }
    }

    #[test]
    fn test_resonance_class_validation() {
        assert!(ResonanceClass::new(0).is_ok());
        assert!(ResonanceClass::new(95).is_ok());
        assert!(ResonanceClass::new(96).is_err());
        assert!(ResonanceClass::new(255).is_err());
    }

    #[test]
    fn test_coordinate_validation() {
        assert!(PhiCoordinate::new(0, 0).is_ok());
        assert!(PhiCoordinate::new(47, 255).is_ok());
        assert!(PhiCoordinate::new(48, 0).is_err());
        // Note: byte is u8, so values [0, 255] are all valid by type
        // The function checks byte >= 256 which can never be true for u8
    }

    #[test]
    fn test_phi_encode_uniqueness() {
        use std::collections::HashSet;
        let mut encodings = HashSet::new();

        for page in 0..PAGES as u8 {
            // Note: byte is u8, so we can iterate 0..=255 to cover all 256 values
            for byte in 0..=255u8 {
                let encoded = phi_encode(page, byte);
                assert!(encodings.insert(encoded), "duplicate encoding for ({}, {})", page, byte);
            }
        }

        assert_eq!(encodings.len(), TOTAL_ELEMENTS);
    }
}
