//! Data schemas for HRM storage
//!
//! This module defines the type-safe schemas for storing Atlas vectors
//! and address mappings. These are simple Rust structs that can be serialized
//! to/from Apache Arrow RecordBatches and Parquet files.

use serde::{Deserialize, Serialize};

/// Atlas vector entry
///
/// Stores one of the 96 canonical Griess vectors that form the Atlas partition.
/// These vectors are generated deterministically using SplitMix64 PRNG seeded
/// with the class index.
///
/// # Storage
///
/// - **Total**: 96 entries (one per class)
/// - **Size per entry**: ~1.5 MB (196,884 × 8 bytes)
/// - **Total Atlas size**: ~151 MB uncompressed, ~15 MB in Parquet
///
/// # Schema
///
/// ```text
/// class: u8          [PRIMARY KEY] - Resonance class [0, 95]
/// vector: Vec<f64>                 - 196,884-dimensional Griess vector
/// checksum: [u8; 32]               - SHA-256 checksum for integrity
/// metadata: String                 - JSON metadata (generated_at, norm, etc.)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasVector {
    /// Resonance class [0, 95]
    pub class: u8,

    /// GriessVector: 196,884 float64 values
    ///
    /// This is the canonical vector for this class, generated deterministically
    /// using SplitMix64 PRNG seeded with `class` value.
    pub vector: Vec<f64>,

    /// SHA-256 checksum for integrity verification
    ///
    /// Computed over the vector data to ensure deterministic generation
    /// and detect corruption.
    pub checksum: [u8; 32],

    /// Metadata in JSON format
    ///
    /// Contains:
    /// - `generated_at`: ISO-8601 timestamp
    /// - `norm`: Normalization type ("unit" or "raw")
    /// - `prng`: PRNG algorithm ("SplitMix64")
    /// - `seed`: Seed value (equal to class)
    pub metadata: String,
}

/// Address mapping entry
///
/// Maps input values to their canonical memory addresses in the Atlas
/// address space (1,179,648 total addresses = 96 classes × 12,288 per class).
///
/// # Storage
///
/// - **Compile-time**: First 10,000 addresses precomputed
/// - **Runtime**: Additional addresses computed on-demand (with warning)
/// - **Size per entry**: 20 bytes (8 + 1 + 1 + 1 + variable)
///
/// # Schema
///
/// ```text
/// input_hash: u64       [PRIMARY KEY] - Hash of input value
/// class: u8                           - Resonance class [0, 95]
/// page: u8                            - Page within class [0, 47]
/// byte: u8                            - Byte within page [0, 255]
/// input_value: Vec<u8>                - Original input (BigUint bytes)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddressMapping {
    /// Input value hash (used as lookup key)
    ///
    /// For small integers (<2^64), this is the integer itself.
    /// For larger BigUint values, this is a hash of the bytes.
    pub input_hash: u64,

    /// Resonance class [0, 95]
    ///
    /// First component of the address, determines which of the 96
    /// canonical vector spaces this input maps to.
    pub class: u8,

    /// Page within class [0, 47]
    ///
    /// Second component of the address, determines which of 48 pages
    /// within the class space.
    pub page: u8,

    /// Byte within page [0, 255]
    ///
    /// Third component of the address, determines which of 256 bytes
    /// within the page.
    pub byte: u8,

    /// Original input value (for verification)
    ///
    /// Stored as BigUint bytes (little-endian). This allows round-trip
    /// verification: hash → address → original value.
    pub input_value: Vec<u8>,
}

impl AtlasVector {
    /// Create a new Atlas vector entry
    pub fn new(class: u8, vector: Vec<f64>, checksum: [u8; 32], metadata: String) -> Self {
        Self {
            class,
            vector,
            checksum,
            metadata,
        }
    }

    /// Get the vector dimension
    pub fn dimension(&self) -> usize {
        self.vector.len()
    }

    /// Verify the checksum matches the vector data
    pub fn verify_checksum(&self) -> bool {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for &val in &self.vector {
            hasher.update(val.to_le_bytes());
        }
        let computed: [u8; 32] = hasher.finalize().into();
        computed == self.checksum
    }
}

impl AddressMapping {
    /// Create a new address mapping from u64 input
    pub fn from_u64(input: u64, class: u8, page: u8, byte: u8) -> Self {
        Self {
            input_hash: input,
            class,
            page,
            byte,
            input_value: input.to_le_bytes().to_vec(),
        }
    }

    /// Create a new address mapping from BigUint bytes
    pub fn from_bytes(hash: u64, input_bytes: Vec<u8>, class: u8, page: u8, byte: u8) -> Self {
        Self {
            input_hash: hash,
            class,
            page,
            byte,
            input_value: input_bytes,
        }
    }

    /// Get the linear offset within the class (0-12,287)
    pub fn linear_offset(&self) -> usize {
        (self.page as usize) * 256 + (self.byte as usize)
    }

    /// Get the global linear offset across all classes (0-1,179,647)
    pub fn global_offset(&self) -> usize {
        (self.class as usize) * 12_288 + self.linear_offset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atlas_vector_creation() {
        let vector = vec![1.0; 196_884];
        let checksum = [0u8; 32];
        let metadata = r#"{"generated_at":"2025-01-01T00:00:00Z"}"#.to_string();

        let av = AtlasVector::new(42, vector, checksum, metadata);
        assert_eq!(av.class, 42);
        assert_eq!(av.dimension(), 196_884);
    }

    #[test]
    fn test_atlas_vector_checksum() {
        use sha2::{Digest, Sha256};

        let vector: Vec<f64> = vec![1.0; 196_884];
        let mut hasher = Sha256::new();
        for &val in &vector {
            hasher.update(val.to_le_bytes());
        }
        let checksum: [u8; 32] = hasher.finalize().into();

        let av = AtlasVector::new(0, vector, checksum, "{}".to_string());
        assert!(av.verify_checksum());
    }

    #[test]
    fn test_address_mapping_from_u64() {
        let mapping = AddressMapping::from_u64(1961, 5, 10, 128);
        assert_eq!(mapping.input_hash, 1961);
        assert_eq!(mapping.class, 5);
        assert_eq!(mapping.page, 10);
        assert_eq!(mapping.byte, 128);
        assert_eq!(mapping.input_value, 1961u64.to_le_bytes());
    }

    #[test]
    fn test_address_mapping_offsets() {
        let mapping = AddressMapping::from_u64(42, 1, 2, 3);

        // Linear offset within class: page*256 + byte = 2*256 + 3 = 515
        assert_eq!(mapping.linear_offset(), 515);

        // Global offset: class*12288 + linear = 1*12288 + 515 = 12803
        assert_eq!(mapping.global_offset(), 12_803);
    }

    #[test]
    fn test_address_mapping_from_bytes() {
        let bytes = vec![1, 2, 3, 4];
        let mapping = AddressMapping::from_bytes(12345, bytes.clone(), 10, 20, 30);

        assert_eq!(mapping.input_hash, 12345);
        assert_eq!(mapping.input_value, bytes);
        assert_eq!(mapping.class, 10);
    }
}
