//! Error types for Atlas backends

use thiserror::Error;

/// Result type for backend operations
pub type Result<T> = std::result::Result<T, BackendError>;

/// Errors that can occur in backend operations
#[derive(Debug, Error)]
pub enum BackendError {
    /// Memory allocation failed
    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    /// Cache pinning failed (insufficient privileges)
    #[error("Cache pinning failed: {0}")]
    CachePinningFailed(String),

    /// Invalid topology descriptor
    #[error("Invalid topology: {0}")]
    InvalidTopology(String),

    /// Operation execution failed
    #[error("Operation execution failed: {0}")]
    ExecutionFailed(String),

    /// Invalid phase value
    #[error("Invalid phase: {0} (must be < 768)")]
    InvalidPhase(u16),

    /// Invalid resonance class
    #[error("Invalid resonance class: {0} (must be < 96)")]
    InvalidClass(u8),

    /// Invalid backend handle
    #[error("Invalid backend handle: {}", .0.0)]
    InvalidHandle(crate::types::BackendHandle),

    /// Backend not initialized
    #[error("Backend not initialized")]
    NotInitialized,

    /// Hardware not available
    #[error("Hardware not available: {0}")]
    HardwareUnavailable(String),

    /// Synchronization failed
    #[error("Synchronization failed: {0}")]
    SynchronizationFailed(String),

    /// Register type mismatch
    #[error("Register r{register} type mismatch: expected {expected}, got {actual:?}")]
    TypeMismatch {
        register: u8,
        expected: atlas_isa::Type,
        actual: Option<atlas_isa::Type>,
    },

    /// Uninitialized register access
    #[error("Register r{register} accessed before initialization")]
    UninitializedRegister { register: u8 },

    /// Unsupported instruction
    #[error("Unsupported instruction: {0}")]
    UnsupportedInstruction(String),

    /// Empty call stack (RET without CALL)
    #[error("Call stack empty (RET without matching CALL)")]
    EmptyCallStack,

    /// Atlas runtime error
    #[error("Atlas runtime error: {0}")]
    Runtime(#[from] atlas_runtime::AtlasError),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
