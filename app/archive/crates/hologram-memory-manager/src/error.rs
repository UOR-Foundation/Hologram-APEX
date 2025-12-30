//! Error types for the processor crate

/// Errors that can occur during stream processing
#[derive(Debug, thiserror::Error)]
pub enum ProcessorError {
    /// Chunking failed
    #[error("Chunking failed: {0}")]
    ChunkingError(String),

    /// Memory pool operation failed
    #[error("Memory pool error: {0}")]
    MemoryPoolError(String),

    /// Domain head not found
    #[error("Domain head not found: {0}")]
    DomainHeadNotFound(String),

    /// Gauge construction failed
    #[error("Gauge construction failed: {0}")]
    GaugeError(String),

    /// Stream operation failed
    #[error("Stream error: {0}")]
    StreamError(String),

    /// Circuit compilation failed
    #[error("Circuit compilation failed: {0}")]
    CompilationError(String),

    /// Execution failed
    #[error("Execution failed: {0}")]
    ExecutionError(String),

    /// Invalid chunk size for type conversion
    #[error("Invalid chunk size {chunk_size} for type size {type_size}")]
    InvalidChunkSize { chunk_size: usize, type_size: usize },

    /// Backend allocation failed
    #[error("Backend allocation failed: {0}")]
    BackendAllocation(String),

    /// Backend copy operation failed
    #[error("Backend copy failed: {0}")]
    BackendCopy(String),

    /// Backend operation failed
    #[error("Backend error: {0}")]
    BackendError(String),

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

/// Result type for processor operations
pub type Result<T> = std::result::Result<T, ProcessorError>;
