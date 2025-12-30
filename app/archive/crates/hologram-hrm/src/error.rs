//! Error types for hologram-hrm

use thiserror::Error;

/// Result type alias for hologram-hrm operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in hologram-hrm operations
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid vector dimension
    #[error("Invalid vector dimension: expected 196884, got {0}")]
    InvalidDimension(usize),

    /// Class index out of range [0, 95]
    #[error("Class index out of range: {0} (must be in [0, 95])")]
    ClassOutOfRange(u8),

    /// Address not found in cache
    #[error("Address not found for input: {0}")]
    AddressNotFound(u64),

    /// Class not found in Atlas
    #[error("Class not found in Atlas: {0}")]
    ClassNotFound(u8),

    /// Factorization failed
    #[error("Factorization failed: could not find factor pair")]
    FactorizationFailed,

    /// Decoding failed
    #[error("Decoding failed: {0}")]
    DecodingFailed(String),

    /// Invalid base-96 digit
    #[error("Invalid base-96 digit: {0} (must be in [0, 95])")]
    InvalidBase96Digit(u8),

    /// Checksum mismatch
    #[error("Checksum mismatch: expected {expected:?}, got {actual:?}")]
    ChecksumMismatch {
        /// Expected checksum
        expected: [u8; 32],
        /// Actual checksum computed
        actual: [u8; 32],
    },

    /// Storage error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Atlas-core error
    #[error("Atlas-core error: {0}")]
    AtlasCore(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Arrow error
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// Parquet error
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    /// Tonbo error
    #[error("Tonbo error: {0}")]
    Tonbo(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Pattern not found in router cache
    #[error("Pattern not found: {0}")]
    PatternNotFound(String),

    /// Invalid input value
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}
