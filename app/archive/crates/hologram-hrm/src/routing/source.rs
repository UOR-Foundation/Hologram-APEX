//! Source Router: Input Value → (h₂, d, ℓ) Coordinate Mapping
//!
//! The SourceRouter pre-computes mappings from input values to their canonical
//! Atlas coordinates. This enables O(1) lookups at runtime.
//!
//! ## Architecture
//!
//! ```text
//! Input Value (f32/f64)
//!     ↓ hash bits
//! Hash Table Lookup
//!     ↓ O(1)
//! (h₂, d, ℓ) Coordinates
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use hologram_hrm::routing::SourceRouter;
//! use hologram_hrm::Atlas;
//!
//! let atlas = Atlas::with_cache()?;
//!
//! // Build router for specific input patterns
//! let patterns = vec![1.0, 2.0, 3.0, 5.0, 10.0];
//! let router = SourceRouter::build_for_patterns(&patterns, atlas)?;
//!
//! // O(1) lookup at runtime
//! let coords = router.route(2.0)?;
//! println!("Value 2.0 maps to: h2={}, d={:?}, l={}", coords.h2, coords.d, coords.l);
//! ```

use crate::atlas::{Atlas, SigilComponents};
use crate::embed::embed_integer;
use crate::griess::GriessVector;
use crate::symbolic::SymbolicInteger;
use crate::{Error, Result};
use num_bigint::BigUint;
use std::collections::HashMap;
use std::sync::Arc;

/// Source Router: Pre-computed input → coordinate mappings
///
/// This router maintains a cache of input values to their (h₂, d, ℓ) coordinates.
/// All mappings are computed at build time, enabling O(1) runtime lookups.
#[derive(Clone)]
pub struct SourceRouter {
    /// Atlas partition for vector operations
    #[allow(dead_code)]
    atlas: Arc<Atlas>,

    /// Pattern cache: input bits → (h₂, d, ℓ)
    ///
    /// We use bit representation (u64 for f64, u32 for f32) as keys
    /// because floating-point equality is unreliable.
    cache: Arc<HashMap<u64, SigilComponents>>,
}

impl SourceRouter {
    /// Create new SourceRouter with Atlas
    pub fn new(atlas: Arc<Atlas>) -> Self {
        Self {
            atlas,
            cache: Arc::new(HashMap::new()),
        }
    }

    /// Build router for specific input patterns
    ///
    /// This pre-computes (h₂, d, ℓ) coordinates for all input patterns.
    /// Runtime lookups will be O(1) hash table access.
    ///
    /// # Arguments
    ///
    /// * `patterns` - Input values to pre-compute
    /// * `atlas` - Atlas partition to use
    ///
    /// # Example
    ///
    /// ```ignore
    /// let patterns = vec![1.0, 2.0, 3.0, 5.0];
    /// let router = SourceRouter::build_for_patterns(&patterns, atlas)?;
    /// ```
    pub fn build_for_patterns(patterns: &[f64], atlas: Arc<Atlas>) -> Result<Self> {
        let mut cache = HashMap::new();

        for &value in patterns {
            // Convert f64 to symbolic integer via BigUint
            let symbolic = f64_to_symbolic(value)?;

            // Embed symbolic integer → Griess vector
            let vector = embed_integer(&symbolic, &atlas)?;

            // Find nearest Atlas class
            let class = find_nearest_class(&vector, &atlas)?;

            // Decode class to (h₂, d, ℓ)
            let coords = SigilComponents::from_class_index(class);

            // Cache using bit representation
            let bits = value.to_bits();
            cache.insert(bits, coords);
        }

        Ok(Self {
            atlas,
            cache: Arc::new(cache),
        })
    }

    /// Route input value to (h₂, d, ℓ) coordinates
    ///
    /// # Errors
    ///
    /// Returns `Error::PatternNotFound` if value was not pre-computed
    pub fn route(&self, value: f64) -> Result<SigilComponents> {
        let bits = value.to_bits();

        self.cache
            .get(&bits)
            .copied()
            .ok_or_else(|| Error::PatternNotFound(format!("Value {} not in pattern cache", value)))
    }

    /// Check if router has pattern
    pub fn has_pattern(&self, value: f64) -> bool {
        self.cache.contains_key(&value.to_bits())
    }

    /// Get number of cached patterns
    pub fn pattern_count(&self) -> usize {
        self.cache.len()
    }
}

/// Convert f64 to SymbolicInteger
///
/// This converts a floating-point value to a BigUint representation
/// suitable for embedding in the Griess algebra.
fn f64_to_symbolic(value: f64) -> Result<SymbolicInteger> {
    // Handle special cases
    if !value.is_finite() {
        return Err(Error::InvalidInput(format!(
            "Cannot convert non-finite value {} to symbolic integer",
            value
        )));
    }

    // Convert f64 to integer representation
    // For now, we use the bit pattern as a BigUint
    // This ensures each distinct float maps to a distinct integer
    let bits = value.to_bits();
    let big_uint = BigUint::from(bits);

    Ok(SymbolicInteger::from_biguint(big_uint))
}

/// Find nearest Atlas class to a Griess vector
///
/// This finds the class index (0..95) whose canonical vector is closest
/// to the given vector using L2 distance.
fn find_nearest_class(vector: &GriessVector, atlas: &Atlas) -> Result<u8> {
    let mut min_distance = f64::INFINITY;
    let mut nearest_class = 0u8;

    for class in 0..96 {
        let atlas_vector = atlas.get_vector(class)?;

        // Compute L2 distance
        let distance = vector.distance(&atlas_vector);

        if distance < min_distance {
            min_distance = distance;
            nearest_class = class;
        }
    }

    Ok(nearest_class)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_to_symbolic_basic() {
        let sym = f64_to_symbolic(42.0).unwrap();
        // Should convert to some BigUint representation
        assert!(sym.digit_count() > 0);
    }

    #[test]
    fn test_f64_to_symbolic_special_values() {
        // NaN should error
        assert!(f64_to_symbolic(f64::NAN).is_err());

        // Infinity should error
        assert!(f64_to_symbolic(f64::INFINITY).is_err());
        assert!(f64_to_symbolic(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn test_f64_to_symbolic_deterministic() {
        let sym1 = f64_to_symbolic(std::f64::consts::PI).unwrap();
        let sym2 = f64_to_symbolic(std::f64::consts::PI).unwrap();

        // Same input should produce same symbolic integer
        assert_eq!(sym1.digit_count(), sym2.digit_count());
    }

    #[test]
    fn test_source_router_build() {
        let atlas = Arc::new(Atlas::with_cache().unwrap());

        let patterns = vec![1.0, 2.0, 3.0];
        let router = SourceRouter::build_for_patterns(&patterns, atlas).unwrap();

        assert_eq!(router.pattern_count(), 3);
        assert!(router.has_pattern(1.0));
        assert!(router.has_pattern(2.0));
        assert!(router.has_pattern(3.0));
        assert!(!router.has_pattern(4.0));
    }

    #[test]
    fn test_source_router_route() {
        let atlas = Arc::new(Atlas::with_cache().unwrap());

        let patterns = vec![1.0, 2.0];
        let router = SourceRouter::build_for_patterns(&patterns, atlas).unwrap();

        // Cached patterns should route successfully
        let coords1 = router.route(1.0).unwrap();
        assert!(coords1.h2 < 4);
        assert!(coords1.l < 8);

        let coords2 = router.route(2.0).unwrap();
        assert!(coords2.h2 < 4);
        assert!(coords2.l < 8);

        // Uncached pattern should error
        assert!(router.route(3.0).is_err());
    }

    #[test]
    fn test_source_router_deterministic() {
        let atlas = Arc::new(Atlas::with_cache().unwrap());

        let patterns = vec![42.0];
        let router = SourceRouter::build_for_patterns(&patterns, atlas).unwrap();

        // Same input should always produce same coordinates
        let coords1 = router.route(42.0).unwrap();
        let coords2 = router.route(42.0).unwrap();

        assert_eq!(coords1, coords2);
    }
}
