//! Error types for the ONNX compiler

use std::io;
use thiserror::Error;

/// Result type for compiler operations
pub type Result<T> = std::result::Result<T, CompilerError>;

/// Errors that can occur during ONNX model compilation
#[derive(Debug, Error)]
pub enum CompilerError {
    #[error("Failed to parse ONNX model: {0}")]
    ParseError(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOp(String),

    #[error("Invalid model structure: {0}")]
    InvalidModel(String),

    #[error("Shape inference failed: {0}")]
    ShapeInferenceError(String),

    #[error("Graph optimization failed: {0}")]
    OptimizationError(String),

    #[error("Binary generation failed: {0}")]
    BinaryGenerationError(String),

    #[error("Weight extraction failed: {0}")]
    WeightExtractionError(String),

    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    #[error("Core error: {0}")]
    CoreError(#[from] hologram_core::Error),

    #[error("Safetensors error: {0}")]
    SafetensorsError(String),

    #[error("Missing required attribute: {0}")]
    MissingAttribute(String),

    #[error("Invalid attribute value: {0}")]
    InvalidAttribute(String),

    #[error("Missing hash entry: {0}")]
    MissingHashEntry(u64),

    #[error("Invalid address: {0}")]
    InvalidAddress(String),

    #[error("HRM error: {0}")]
    HrmError(#[from] hologram_hrm::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Protobuf decode error: {0}")]
    DecodeError(#[from] prost::DecodeError),

    #[error("Graph error: {0}")]
    AnyhowError(String),
}

// Manual From impl for anyhow::Error
impl From<anyhow::Error> for CompilerError {
    fn from(err: anyhow::Error) -> Self {
        CompilerError::AnyhowError(err.to_string())
    }
}
