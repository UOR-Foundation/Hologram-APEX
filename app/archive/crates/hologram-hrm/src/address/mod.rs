//! Address resolution: Integer → (class, page, byte)
//!
//! This module maps arbitrary integers to addresses in the Atlas address space.
//! Each address is represented by a PhiCoordinate (φ-coordinate) consisting of:
//! - **class**: One of 96 classes [0, 95]
//! - **page**: One of 48 pages [0, 47]
//! - **byte**: One of 256 bytes [0, 255]
//!
//! Total address space: 96 × 48 × 256 = 1,179,648 addresses
//!
//! # Algorithm
//!
//! 1. Convert integer to SymbolicInteger
//! 2. Embed into Griess space using embedding operator E
//! 3. Project Griess vector to find closest Atlas vector (class)
//! 4. Compute page and byte from residual
//!
//! # Example
//!
//! ```ignore
//! use hologram_hrm::{HrmStore, resolve_address};
//!
//! let store = HrmStore::new();
//! let (class, page, byte) = resolve_address(1961, &store)?;
//! println!("1961 → class={}, page={}, byte={}", class, page, byte);
//! ```

use crate::atlas::Atlas;
use crate::embed::embed_integer;
use crate::griess::GriessVector;
use crate::storage::HrmStore;
use crate::symbolic::SymbolicInteger;
use crate::{Result, ATLAS_CLASSES};

/// A φ-coordinate (phi-coordinate) in the Atlas address space
///
/// Represents a unique address as (class, page, byte):
/// - **class**: [0, 95] - Which of the 96 Atlas classes
/// - **page**: [0, 47] - Which of 48 pages within the class
/// - **byte**: [0, 255] - Which of 256 bytes within the page
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhiCoordinate {
    /// The class index [0, 95]
    pub class: u8,
    /// The page index [0, 47]
    pub page: u8,
    /// The byte index [0, 255]
    pub byte: u8,
}

impl PhiCoordinate {
    /// Create a new PhiCoordinate
    ///
    /// # Arguments
    ///
    /// * `class` - Class index [0, 95]
    /// * `page` - Page index [0, 47]
    /// * `byte` - Byte index [0, 255]
    pub const fn new(class: u8, page: u8, byte: u8) -> Self {
        Self { class, page, byte }
    }

    /// Convert to a linear address in [0, 1,179,647]
    ///
    /// Linear address = class × (48 × 256) + page × 256 + byte
    pub fn to_linear_address(&self) -> usize {
        (self.class as usize) * (48 * 256) + (self.page as usize) * 256 + (self.byte as usize)
    }

    /// Create from a linear address
    ///
    /// # Arguments
    ///
    /// * `addr` - Linear address in [0, 1,179,647]
    ///
    /// # Returns
    ///
    /// The PhiCoordinate, or None if address is out of range.
    pub fn from_linear_address(addr: usize) -> Option<Self> {
        const TOTAL_ADDRESS_SPACE: usize = 96 * 48 * 256;
        if addr >= TOTAL_ADDRESS_SPACE {
            return None;
        }

        let class = (addr / (48 * 256)) as u8;
        let remainder = addr % (48 * 256);
        let page = (remainder / 256) as u8;
        let byte = (remainder % 256) as u8;

        Some(Self { class, page, byte })
    }
}

/// Resolve an integer to an address (class, page, byte)
///
/// This is the core address resolution function that maps arbitrary integers
/// to addresses in the Atlas address space.
///
/// # Algorithm
///
/// 1. Convert input to SymbolicInteger
/// 2. Check if address is cached in the store
/// 3. If not cached:
///    - Embed into Griess space
///    - Find nearest Atlas vector (class)
///    - Compute page and byte from remaining information
///    - Cache the result
///
/// # Arguments
///
/// * `input` - The input value to resolve
/// * `store` - The HRM store (for caching and Atlas vectors)
///
/// # Returns
///
/// A tuple (class, page, byte) representing the address.
///
/// # Example
///
/// ```ignore
/// let store = HrmStore::new();
/// let (class, page, byte) = resolve_address(1961, &store)?;
/// ```
pub fn resolve_address(input: u64, store: &HrmStore) -> Result<(u8, u8, u8)> {
    let sym = SymbolicInteger::from(input);
    let hash = sym.hash_value();

    // Check cache first
    if let Some((class, page, byte)) = store.get_address(hash) {
        return Ok((class, page, byte));
    }

    // Not cached - compute address
    let address = compute_address(&sym, store)?;

    // Cache the result
    store.insert_address(hash, address.class, address.page, address.byte);

    Ok((address.class, address.page, address.byte))
}

/// Compute the address for a SymbolicInteger
///
/// This performs the actual address computation without caching.
fn compute_address(sym: &SymbolicInteger, store: &HrmStore) -> Result<PhiCoordinate> {
    // Create Atlas from the store
    let atlas = Atlas::from_store(store.clone())?;

    // Embed the integer into Griess space
    let embedded = embed_integer(sym, &atlas)?;

    // Find the nearest Atlas vector (class)
    let class = find_nearest_class(&embedded, &atlas)?;

    // Get the remaining information for page and byte
    // For now, use a deterministic hash-based approach
    let hash = sym.hash_value();
    let page = ((hash >> 8) % 48) as u8;
    let byte = (hash & 0xFF) as u8;

    Ok(PhiCoordinate { class, page, byte })
}

/// Find the nearest Atlas class to the given vector
///
/// This computes the distance from the input vector to all 96 Atlas vectors
/// and returns the index of the closest one.
fn find_nearest_class(vector: &GriessVector, atlas: &Atlas) -> Result<u8> {
    let mut min_distance = f64::INFINITY;
    let mut nearest_class = 0;

    for class in 0..ATLAS_CLASSES {
        let atlas_vector = atlas.get_vector(class)?;
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

    /// Create a populated HrmStore for testing
    fn create_test_store() -> HrmStore {
        let atlas = Atlas::with_cache().unwrap();
        let mut store = HrmStore::new();
        atlas.save_to_store(&mut store).unwrap();
        store
    }

    #[test]
    fn test_phi_coordinate_new() {
        let phi = PhiCoordinate::new(42, 10, 128);
        assert_eq!(phi.class, 42);
        assert_eq!(phi.page, 10);
        assert_eq!(phi.byte, 128);
    }

    #[test]
    fn test_phi_coordinate_linear_address() {
        let phi = PhiCoordinate::new(0, 0, 0);
        assert_eq!(phi.to_linear_address(), 0);

        let phi = PhiCoordinate::new(0, 0, 1);
        assert_eq!(phi.to_linear_address(), 1);

        let phi = PhiCoordinate::new(0, 1, 0);
        assert_eq!(phi.to_linear_address(), 256);

        let phi = PhiCoordinate::new(1, 0, 0);
        assert_eq!(phi.to_linear_address(), 48 * 256);
    }

    #[test]
    fn test_phi_coordinate_roundtrip() {
        for class in [0, 10, 42, 95] {
            for page in [0, 10, 47] {
                for byte in [0, 1, 128, 255] {
                    let phi = PhiCoordinate::new(class, page, byte);
                    let linear = phi.to_linear_address();
                    let reconstructed = PhiCoordinate::from_linear_address(linear).unwrap();
                    assert_eq!(phi, reconstructed);
                }
            }
        }
    }

    #[test]
    fn test_phi_coordinate_from_linear_out_of_range() {
        let result = PhiCoordinate::from_linear_address(96 * 48 * 256);
        assert_eq!(result, None);
    }

    #[test]
    fn test_resolve_address_deterministic() {
        let store = create_test_store();

        let (c1, p1, b1) = resolve_address(1961, &store).unwrap();
        let (c2, p2, b2) = resolve_address(1961, &store).unwrap();

        // Same input should produce same output
        assert_eq!((c1, p1, b1), (c2, p2, b2));
    }

    #[test]
    fn test_resolve_address_different_inputs() {
        let store = create_test_store();

        let addr1 = resolve_address(1961, &store).unwrap();
        let addr2 = resolve_address(1962, &store).unwrap();

        // Different inputs should (likely) produce different outputs
        // Note: There's a small chance of collision, but it's very unlikely
        assert_ne!(addr1, addr2);
    }

    #[test]
    fn test_resolve_address_valid_range() {
        let store = create_test_store();

        for value in [0, 1, 42, 96, 1961, 65536] {
            let (class, page, _byte) = resolve_address(value, &store).unwrap();

            assert!(class < 96, "Class {} out of range for input {}", class, value);
            assert!(page < 48, "Page {} out of range for input {}", page, value);
            // byte can be any value 0-255, so no assertion needed
        }
    }

    #[test]
    fn test_find_nearest_class() {
        let atlas = Atlas::with_cache().unwrap();

        // An Atlas vector should be nearest to itself
        for class in [0, 10, 42, 95] {
            let vector = atlas.get_vector(class).unwrap();
            let nearest = find_nearest_class(&vector, &atlas).unwrap();
            assert_eq!(nearest, class);
        }
    }
}
