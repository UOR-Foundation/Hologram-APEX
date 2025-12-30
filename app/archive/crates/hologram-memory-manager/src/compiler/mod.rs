//! Circuit Compilation Module
//!
//! Compiles circuit circuits to stream processing contexts.
//! This module provides the integration point between circuit canonical
//! compilation and the processor's stream execution model.

use crate::{constants::DEFAULT_MAX_CHUNK_LEVELS, stream::StreamContext, ProcessorError, Result};

/// Circuit stream compiler
///
/// Compiles circuit circuit expressions into stream processing contexts.
pub struct CircuitStreamCompiler {
    /// Maximum primorial levels for chunking
    max_levels: usize,
}

impl CircuitStreamCompiler {
    /// Create new compiler with default settings
    ///
    /// Uses [`DEFAULT_MAX_CHUNK_LEVELS`] for chunking. To customize,
    /// use [`CircuitStreamCompiler::with_levels`].
    pub fn new() -> Self {
        Self {
            max_levels: DEFAULT_MAX_CHUNK_LEVELS,
        }
    }

    /// Create new compiler with specific chunk levels
    pub fn with_levels(max_levels: usize) -> Self {
        Self { max_levels }
    }

    /// Get max levels
    pub fn max_levels(&self) -> usize {
        self.max_levels
    }

    /// Set max levels
    pub fn set_max_levels(&mut self, levels: usize) {
        self.max_levels = levels;
    }

    /// Compile circuit to stream (integration point for future implementation)
    ///
    /// Future implementation will:
    /// 1. Parse circuit expression via circuit
    /// 2. Canonicalize circuit
    /// 3. Map to stream operations
    /// 4. Return compiled representation
    pub fn compile_to_stream(&self, _circuit: &str) -> Result<CompiledCircuitStream> {
        // Integration point for future implementation
        // This will connect to circuit::CircuitCompiler
        Err(ProcessorError::CompilationError(
            "Circuit compilation integration not yet implemented - see compiler/mod.rs".to_string(),
        ))
    }
}

impl Default for CircuitStreamCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiled circuit stream
///
/// Represents a compiled circuit circuit ready for stream execution.
pub struct CompiledCircuitStream {
    /// Canonical circuit form
    pub canonical: String,

    /// Stream context (if embedded)
    pub context: Option<StreamContext>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_creation() {
        let compiler = CircuitStreamCompiler::new();
        assert_eq!(compiler.max_levels, DEFAULT_MAX_CHUNK_LEVELS);

        let compiler2 = CircuitStreamCompiler::with_levels(10);
        assert_eq!(compiler2.max_levels, 10);
    }
}
