//! Hash-based factorization for deterministic address mapping
//!
//! This module provides a hash-based factorization approach that deterministically
//! maps input patterns to addresses in the extended address space without requiring
//! mathematical factorization of Griess vectors.
//!
//! # Algorithm
//!
//! Given an input hash (u64) and operation ID:
//! 1. Combine hash with operation ID to avoid collisions
//! 2. Extract address components using modular arithmetic
//! 3. Return deterministic ExtendedAddress
//!
//! # Properties
//!
//! - **Deterministic**: Same input always maps to same address
//! - **Fast**: O(1) computation (just arithmetic)
//! - **Collision-resistant**: Uses FNV-1a mixing for good distribution
//! - **Scalable**: Supports 773B addresses in extended address space
//!
//! # Example
//!
//! ```rust,ignore
//! use hologram_hrm::extract::hash_factorization::factorize_hash;
//!
//! let input_hash: u64 = 0x123456789ABCDEF0;
//! let op_id: usize = 42;
//!
//! let address = factorize_hash(input_hash, op_id);
//! println!("Address: class={}, page={}, byte={}, sub={}",
//!     address.class, address.page, address.byte, address.sub_index);
//! ```

use ahash::AHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

/// Extended address with sub-byte granularity
///
/// Supports 773B addresses: 96 classes × 480 pages × 256 bytes × 65536 sub-indices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExtendedAddress {
    /// Class index (0-95)
    pub class: u8,
    /// Page index (0-479)
    pub page: u16,
    /// Byte index (0-255)
    pub byte: u8,
    /// Sub-index (0-65535)
    pub sub_index: u16,
}

impl ExtendedAddress {
    /// Create a new extended address with validation
    pub fn new(class: u8, page: u16, byte: u8, sub_index: u16) -> Result<Self, String> {
        if class >= 96 {
            return Err(format!("Class must be < 96, got {}", class));
        }
        if page >= 480 {
            return Err(format!("Page must be < 480, got {}", page));
        }
        // byte and sub_index are u8/u16, so they're automatically in range

        Ok(Self {
            class,
            page,
            byte,
            sub_index,
        })
    }

    /// Create without validation (unsafe but fast)
    ///
    /// # Safety
    ///
    /// Caller must ensure class < 96 and page < 480
    pub const unsafe fn new_unchecked(class: u8, page: u16, byte: u8, sub_index: u16) -> Self {
        Self {
            class,
            page,
            byte,
            sub_index,
        }
    }
}

/// Factorize a hash to an extended address
///
/// Uses FNV-1a-style mixing for good distribution across the address space.
///
/// # Algorithm
///
/// 1. Mix input_hash with op_id using FNV-1a prime (0x100000001b3)
/// 2. Extract address components using modular arithmetic
/// 3. Ensure uniform distribution across all components
///
/// # Arguments
///
/// * `input_hash` - Hash of the input pattern (e.g., from AHash)
/// * `op_id` - Operation identifier to avoid cross-operation collisions
///
/// # Returns
///
/// Deterministic ExtendedAddress for the input
///
/// # Example
///
/// ```rust,ignore
/// let hash = compute_input_hash(&input_data);
/// let address = factorize_hash(hash, 0);
/// ```
pub fn factorize_hash(input_hash: u64, op_id: usize) -> ExtendedAddress {
    // FNV-1a mixing: combine input_hash with op_id
    // This ensures different operations map to different address regions
    const FNV_PRIME: u64 = 0x100000001b3;
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;

    let mut hash = FNV_OFFSET;
    hash ^= input_hash;
    hash = hash.wrapping_mul(FNV_PRIME);
    hash ^= op_id as u64;
    hash = hash.wrapping_mul(FNV_PRIME);

    // Extract address components with good distribution
    // Use different bit ranges to minimize correlation

    // Class: use lower 7 bits (0-127) then mod 96
    let class = (hash & 0x7F) % 96;

    // Page: use bits 7-16, mod 480
    let page = ((hash >> 7) & 0x3FF) % 480;

    // Byte: use bits 16-24 (naturally 0-255)
    let byte = ((hash >> 16) & 0xFF) as u8;

    // Sub-index: use upper 16 bits (naturally 0-65535)
    let sub_index = ((hash >> 32) & 0xFFFF) as u16;

    ExtendedAddress {
        class: class as u8,
        page: page as u16,
        byte,
        sub_index,
    }
}

/// Compute hash of input data using AHash
///
/// Provides fast, high-quality hashing for pattern recognition.
///
/// # Arguments
///
/// * `data` - Input data to hash (typically flattened tensor values)
///
/// # Returns
///
/// 64-bit hash suitable for factorize_hash
///
/// # Example
///
/// ```rust,ignore
/// let input = vec![1.0f32, 2.0, 3.0];
/// let hash = compute_data_hash(&input);
/// let address = factorize_hash(hash, 0);
/// ```
pub fn compute_data_hash<T: Hash>(data: &T) -> u64 {
    let mut hasher = AHasher::default();
    data.hash(&mut hasher);
    hasher.finish()
}

/// Factorize input data directly to an address
///
/// Convenience function combining hash computation and factorization.
///
/// # Arguments
///
/// * `data` - Input data to factorize
/// * `op_id` - Operation identifier
///
/// # Example
///
/// ```rust,ignore
/// let input = vec![1.0f32, 2.0, 3.0];
/// let address = factorize_data(&input, 0);
/// ```
pub fn factorize_data<T: Hash>(data: &T, op_id: usize) -> ExtendedAddress {
    let hash = compute_data_hash(data);
    factorize_hash(hash, op_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extended_address_validation() {
        // Valid addresses
        assert!(ExtendedAddress::new(0, 0, 0, 0).is_ok());
        assert!(ExtendedAddress::new(95, 479, 255, 65535).is_ok());

        // Invalid addresses
        assert!(ExtendedAddress::new(96, 0, 0, 0).is_err());
        assert!(ExtendedAddress::new(0, 480, 0, 0).is_err());
    }

    #[test]
    fn test_factorize_hash_deterministic() {
        let hash = 0x123456789ABCDEF0;
        let op_id = 42;

        let addr1 = factorize_hash(hash, op_id);
        let addr2 = factorize_hash(hash, op_id);

        assert_eq!(addr1, addr2, "Same input should produce same address");
    }

    #[test]
    fn test_factorize_hash_different_inputs() {
        let hash1 = 0x123456789ABCDEF0;
        let hash2 = 0x123456789ABCDEF1;
        let op_id = 42;

        let addr1 = factorize_hash(hash1, op_id);
        let addr2 = factorize_hash(hash2, op_id);

        assert_ne!(addr1, addr2, "Different hashes should produce different addresses");
    }

    #[test]
    fn test_factorize_hash_different_ops() {
        let hash = 0x123456789ABCDEF0;

        let addr1 = factorize_hash(hash, 0);
        let addr2 = factorize_hash(hash, 1);

        assert_ne!(addr1, addr2, "Different operations should produce different addresses");
    }

    #[test]
    fn test_factorize_hash_range() {
        // Test many hashes to ensure components stay in range
        for i in 0u64..1000 {
            let hash = i.wrapping_mul(0x123456789ABCDEF0);
            let addr = factorize_hash(hash, 0);

            assert!(addr.class < 96, "Class out of range: {}", addr.class);
            assert!(addr.page < 480, "Page out of range: {}", addr.page);
            // byte and sub_index are u8/u16, so they're automatically in range
        }
    }

    #[test]
    fn test_compute_data_hash() {
        let data1 = vec![1u64, 2, 3];
        let data2 = vec![1u64, 2, 3];
        let data3 = vec![1u64, 2, 4];

        let hash1 = compute_data_hash(&data1);
        let hash2 = compute_data_hash(&data2);
        let hash3 = compute_data_hash(&data3);

        assert_eq!(hash1, hash2, "Same data should produce same hash");
        assert_ne!(hash1, hash3, "Different data should produce different hash");
    }

    #[test]
    fn test_factorize_data() {
        let input = vec![1u64, 2, 3];
        let op_id = 0;

        let addr1 = factorize_data(&input, op_id);
        let addr2 = factorize_data(&input, op_id);

        assert_eq!(addr1, addr2, "Same data should produce same address");

        // Verify address is valid
        assert!(addr1.class < 96);
        assert!(addr1.page < 480);
    }

    #[test]
    fn test_distribution_across_classes() {
        // Test that hashes distribute well across classes
        let mut class_counts = vec![0; 96];

        for i in 0u64..9600 {
            let hash = i.wrapping_mul(0x9E3779B97F4A7C15); // Good mixing constant
            let addr = factorize_hash(hash, 0);
            class_counts[addr.class as usize] += 1;
        }

        // Each class should have roughly 100 entries (9600 / 96)
        // Allow 40-160 range for statistical variation
        for (class, &count) in class_counts.iter().enumerate() {
            assert!(
                (40..=160).contains(&count),
                "Class {} has poor distribution: {} entries",
                class,
                count
            );
        }
    }

    #[test]
    fn test_distribution_across_pages() {
        // Test that hashes distribute well across pages
        let mut page_counts = vec![0; 480];

        for i in 0u64..48000 {
            let hash = i.wrapping_mul(0x9E3779B97F4A7C15);
            let addr = factorize_hash(hash, 0);
            page_counts[addr.page as usize] += 1;
        }

        // Each page should have roughly 100 entries (48000 / 480)
        // Allow 40-160 range for statistical variation
        for (page, &count) in page_counts.iter().enumerate() {
            assert!(
                (40..=160).contains(&count),
                "Page {} has poor distribution: {} entries",
                page,
                count
            );
        }
    }
}
