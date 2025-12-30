//! Destination Router: (h₂, d, ℓ) Coordinate → Output Value Mapping
//!
//! The DestinationRouter pre-computes mappings from Atlas coordinates to their
//! corresponding output values. This enables O(1) lookups at runtime.
//!
//! ## Architecture
//!
//! ```text
//! (h₂, d, ℓ) Coordinates
//!     ↓ encode to class index
//! Hash Table Lookup
//!     ↓ O(1)
//! Output Value (f64)
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use hologram_hrm::routing::DestinationRouter;
//! use hologram_hrm::Atlas;
//! use hologram_hrm::atlas::SigilComponents;
//!
//! let atlas = Atlas::with_cache()?;
//!
//! // Build destination router
//! let router = DestinationRouter::build(atlas)?;
//!
//! // O(1) lookup at runtime
//! let coords = SigilComponents::from_class_index(42);
//! let output = router.route(coords)?;
//! println!("Coordinates map to value: {}", output);
//! ```

use crate::atlas::{Atlas, SigilComponents};
use crate::decode::decode_vector;
use crate::{Error, Result};
use num_bigint::BigUint;
use std::collections::HashMap;
use std::sync::Arc;

/// Destination Router: Pre-computed coordinate → value mappings
///
/// This router maintains a cache of (h₂, d, ℓ) coordinates to output values.
/// All mappings are computed at build time, enabling O(1) runtime lookups.
#[derive(Clone)]
pub struct DestinationRouter {
    /// Atlas partition for vector operations
    #[allow(dead_code)]
    atlas: Arc<Atlas>,

    /// Coordinate cache: class index → output value
    ///
    /// We pre-compute all 96 mappings at build time.
    cache: Arc<HashMap<u8, f64>>,
}

impl DestinationRouter {
    /// Create new DestinationRouter with Atlas
    pub fn new(atlas: Arc<Atlas>) -> Self {
        Self {
            atlas,
            cache: Arc::new(HashMap::new()),
        }
    }

    /// Build router by pre-computing all 96 coordinate mappings
    ///
    /// This computes output values for all possible (h₂, d, ℓ) coordinates.
    /// Runtime lookups will be O(1) hash table access.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let router = DestinationRouter::build(atlas)?;
    /// ```
    pub fn build(atlas: Arc<Atlas>) -> Result<Self> {
        let mut cache = HashMap::new();

        // Pre-compute all 96 coordinate → value mappings
        for class in 0..96 {
            // Get canonical Atlas vector for this class
            let vector = atlas.get_vector(class)?;

            // Decode vector → BigUint → f64
            let big_uint = decode_vector(&vector, &atlas)?;
            let value = biguint_to_f64(&big_uint)?;

            // Cache the mapping
            cache.insert(class, value);
        }

        Ok(Self {
            atlas,
            cache: Arc::new(cache),
        })
    }

    /// Route (h₂, d, ℓ) coordinates to output value
    ///
    /// # Errors
    ///
    /// Returns `Error::ClassNotFound` if coordinates are invalid
    pub fn route(&self, coords: SigilComponents) -> Result<f64> {
        let class = coords.to_class_index();

        self.cache
            .get(&class)
            .copied()
            .ok_or_else(|| Error::ClassNotFound(class))
    }

    /// Get number of cached mappings (should always be 96)
    pub fn mapping_count(&self) -> usize {
        self.cache.len()
    }
}

/// Convert BigUint to f64
///
/// This reverses the f64_to_symbolic conversion in the source router.
/// We convert the BigUint back to f64 using the bit pattern.
fn biguint_to_f64(value: &BigUint) -> Result<f64> {
    // Convert BigUint to u64 (bit pattern)
    let bytes = value.to_bytes_le();

    // Pad or truncate to 8 bytes
    let mut bits_array = [0u8; 8];
    let copy_len = bytes.len().min(8);
    bits_array[..copy_len].copy_from_slice(&bytes[..copy_len]);

    // Convert bytes to u64
    let bits = u64::from_le_bytes(bits_array);

    // Reconstruct f64 from bits
    let float = f64::from_bits(bits);

    // Validate result
    if !float.is_finite() {
        return Err(Error::DecodingFailed(format!(
            "Decoded non-finite f64 from BigUint: {}",
            float
        )));
    }

    Ok(float)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::Modality;

    #[test]
    fn test_biguint_to_f64_basic() {
        // Test roundtrip: f64 → BigUint → f64
        let original = 42.0f64;
        let bits = original.to_bits();
        let big_uint = BigUint::from(bits);
        let recovered = biguint_to_f64(&big_uint).unwrap();

        assert_eq!(original, recovered);
    }

    #[test]
    fn test_biguint_to_f64_special_values() {
        // Test various f64 values
        let test_values: Vec<f64> = vec![0.0, 1.0, -1.0, std::f64::consts::PI, -273.15, 1e10, 1e-10];

        for original in test_values {
            let bits = original.to_bits();
            let big_uint = BigUint::from(bits);
            let recovered = biguint_to_f64(&big_uint).unwrap();

            assert_eq!(original, recovered, "Failed for value {}", original);
        }
    }

    #[test]
    fn test_destination_router_build() {
        let atlas = Arc::new(Atlas::with_cache().unwrap());
        let router = DestinationRouter::build(atlas).unwrap();

        // Should have all 96 mappings
        assert_eq!(router.mapping_count(), 96);
    }

    #[test]
    fn test_destination_router_route() {
        let atlas = Arc::new(Atlas::with_cache().unwrap());
        let router = DestinationRouter::build(atlas).unwrap();

        // Test routing various coordinates
        let coords = SigilComponents::new(0, Modality::Neutral, 0);
        let value = router.route(coords).unwrap();
        assert!(value.is_finite());

        let coords = SigilComponents::new(2, Modality::Produce, 5);
        let value = router.route(coords).unwrap();
        assert!(value.is_finite());
    }

    #[test]
    fn test_destination_router_deterministic() {
        let atlas = Arc::new(Atlas::with_cache().unwrap());
        let router = DestinationRouter::build(atlas).unwrap();

        // Same coordinates should always produce same value
        let coords = SigilComponents::new(1, Modality::Consume, 3);
        let value1 = router.route(coords).unwrap();
        let value2 = router.route(coords).unwrap();

        assert_eq!(value1, value2);
    }

    #[test]
    fn test_destination_router_all_classes() {
        let atlas = Arc::new(Atlas::with_cache().unwrap());
        let router = DestinationRouter::build(atlas).unwrap();

        // All 96 classes should route successfully
        for class in 0..96 {
            let coords = SigilComponents::from_class_index(class);
            let value = router.route(coords).unwrap();
            assert!(value.is_finite(), "Class {} produced non-finite value", class);
        }
    }
}
