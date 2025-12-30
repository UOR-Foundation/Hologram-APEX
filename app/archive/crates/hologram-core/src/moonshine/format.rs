//! .mshr Binary Format Specification
//!
//! Defines the memory-mapped binary format for compiled MoonshineHRM operations.
//!
//! ## File Structure
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │ Header (64 bytes)                   │  Magic, version, offsets
//! ├─────────────────────────────────────┤
//! │ Manifest (JSON, variable)           │  Operation metadata
//! ├─────────────────────────────────────┤
//! │ Hash Table (variable)               │  Input hash → result index
//! ├─────────────────────────────────────┤
//! │ Result Data (page-aligned)          │  Pre-computed results
//! └─────────────────────────────────────┘
//! ```

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::mem;

/// Magic bytes for .mshr files: "MSHRFMT\0"
pub const MSHR_MAGIC: [u8; 8] = [b'M', b'S', b'H', b'R', b'F', b'M', b'T', 0];

/// Current .mshr format version
pub const MSHR_VERSION: u32 = 1;

/// .mshr file header (64 bytes, fixed size)
///
/// Stored at the beginning of every .mshr file. Provides offsets to
/// other sections for zero-copy memory-mapped access.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MshrHeader {
    /// Magic bytes: "MSHRFMT\0" (8 bytes)
    pub magic: [u8; 8],

    /// Format version (currently 1)
    pub version: u32,

    /// Reserved flags for future use
    pub flags: u32,

    /// Byte offset to manifest JSON
    pub manifest_offset: u64,

    /// Size of manifest in bytes
    pub manifest_size: u64,

    /// Byte offset to hash table
    pub hash_table_offset: u64,

    /// Size of hash table in bytes
    pub hash_table_size: u64,

    /// Byte offset to result data
    pub result_data_offset: u64,

    /// Size of result data in bytes
    pub result_data_size: u64,
}

impl MshrHeader {
    /// Size of header in bytes (must be 64)
    pub const SIZE: usize = 64;

    /// Create a new header with given offsets and sizes
    pub fn new(
        manifest_offset: u64,
        manifest_size: u64,
        hash_table_offset: u64,
        hash_table_size: u64,
        result_data_offset: u64,
        result_data_size: u64,
    ) -> Self {
        Self {
            magic: MSHR_MAGIC,
            version: MSHR_VERSION,
            flags: 0,
            manifest_offset,
            manifest_size,
            hash_table_offset,
            hash_table_size,
            result_data_offset,
            result_data_size,
        }
    }

    /// Validate header magic and version
    pub fn validate(&self) -> Result<()> {
        if self.magic != MSHR_MAGIC {
            return Err(crate::Error::InvalidOperation(
                "Invalid .mshr file: bad magic bytes".to_string(),
            ));
        }

        if self.version != MSHR_VERSION {
            return Err(crate::Error::InvalidOperation(format!(
                "Unsupported .mshr version: {} (expected {})",
                self.version, MSHR_VERSION
            )));
        }

        Ok(())
    }

    /// Parse header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(crate::Error::InvalidOperation(format!(
                "Header too small: {} bytes (expected {})",
                bytes.len(),
                Self::SIZE
            )));
        }

        // Safety: We've verified the size, and MshrHeader is repr(C)
        let header = unsafe { std::ptr::read(bytes.as_ptr() as *const MshrHeader) };

        header.validate()?;
        Ok(header)
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let bytes =
            unsafe { std::slice::from_raw_parts(self as *const MshrHeader as *const u8, mem::size_of::<MshrHeader>()) };
        bytes.to_vec()
    }
}

/// Hash table entry (16 bytes)
///
/// Maps input pattern hash to result index
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HashEntry {
    /// FNV-1a hash of input pattern
    pub key_hash: u64,

    /// Index into result data array
    pub result_index: u32,

    /// Padding for 16-byte alignment
    pub _padding: u32,
}

impl HashEntry {
    /// Size of entry in bytes
    pub const SIZE: usize = 16;

    /// Create new hash entry
    pub fn new(key_hash: u64, result_index: u32) -> Self {
        Self {
            key_hash,
            result_index,
            _padding: 0,
        }
    }
}

/// Operation manifest metadata (JSON)
///
/// Stored as JSON for human readability and extensibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Operation name (e.g., "vector_add")
    pub operation: String,

    /// Operation version
    pub version: String,

    /// Number of cached input patterns
    pub input_patterns: usize,

    /// Size of each output result
    pub output_size: usize,

    /// Data type of results
    pub data_type: DataType,

    /// Hash function used (currently "fnv1a_64")
    pub hash_function: String,

    /// When this was compiled
    pub compilation_date: String,

    /// Atlas version used for compilation
    pub atlas_version: String,
}

/// Supported data types for results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
}

impl DataType {
    /// Size of this data type in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F64 => 8,
            DataType::I32 => 4,
            DataType::I64 => 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_size() {
        // Header must be exactly 64 bytes
        assert_eq!(mem::size_of::<MshrHeader>(), 64);
    }

    #[test]
    fn test_hash_entry_size() {
        // Hash entry must be exactly 16 bytes
        assert_eq!(mem::size_of::<HashEntry>(), 16);
    }

    #[test]
    fn test_header_roundtrip() {
        let header = MshrHeader::new(64, 512, 576, 1024, 1600, 4096);

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), MshrHeader::SIZE);

        let parsed = MshrHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.magic, MSHR_MAGIC);
        assert_eq!(parsed.version, MSHR_VERSION);
        assert_eq!(parsed.manifest_offset, 64);
        assert_eq!(parsed.manifest_size, 512);
    }

    #[test]
    fn test_header_validation() {
        let mut header = MshrHeader::new(64, 512, 576, 1024, 1600, 4096);

        // Valid header
        assert!(header.validate().is_ok());

        // Invalid magic
        header.magic = [0; 8];
        assert!(header.validate().is_err());

        // Invalid version
        header.magic = MSHR_MAGIC;
        header.version = 999;
        assert!(header.validate().is_err());
    }

    #[test]
    fn test_data_type_sizes() {
        assert_eq!(DataType::F32.size_bytes(), 4);
        assert_eq!(DataType::F64.size_bytes(), 8);
        assert_eq!(DataType::I32.size_bytes(), 4);
        assert_eq!(DataType::I64.size_bytes(), 8);
    }

    #[test]
    fn test_manifest_json_roundtrip() {
        let manifest = Manifest {
            operation: "vector_add".to_string(),
            version: "1.0.0".to_string(),
            input_patterns: 1000,
            output_size: 256,
            data_type: DataType::F32,
            hash_function: "fnv1a_64".to_string(),
            compilation_date: "2025-11-14T12:00:00Z".to_string(),
            atlas_version: "1.0.0".to_string(),
        };

        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: Manifest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.operation, "vector_add");
        assert_eq!(parsed.input_patterns, 1000);
        assert_eq!(parsed.data_type, DataType::F32);
    }
}
