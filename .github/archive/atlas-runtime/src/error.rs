//! Error types for Atlas Runtime
//!
//! Matches the error model defined in Atlas Runtime Spec §15

use thiserror::Error;

/// Result type for Atlas Runtime operations
pub type Result<T> = std::result::Result<T, AtlasError>;

/// Atlas Runtime errors
///
/// These errors correspond to the error codes defined in the Atlas Runtime
/// Specification §15. Each error includes context for debugging and validation.
#[derive(Debug, Error)]
pub enum AtlasError {
    /// Launch attempted outside kernel's phase window
    #[error("Phase window violation: current phase {current} not in [{begin}, {end})")]
    PhaseWindow { current: u16, begin: u16, end: u16 },

    /// Boundary coordinates outside declared window
    #[error("Boundary violation: ({x}, {y}) outside window ([{x_min}, {x_max}), [{y_min}, {y_max}))")]
    BoundaryViolation {
        x: u8,
        y: u8,
        x_min: u8,
        x_max: u8,
        y_min: u8,
        y_max: u8,
    },

    /// Illegal neighbor traversal (not in 1-skeleton)
    #[error("Illegal neighbor traversal: class {from} → {to} not in 1-skeleton")]
    NeighborIllegal { from: u8, to: u8 },

    /// Declared non-aliasing violated
    #[error("Aliasing violation: buffers {buf1} and {buf2} overlap")]
    Aliasing { buf1: String, buf2: String },

    /// Invalid class ID (must be < 96)
    #[error("Invalid class ID: {0} (must be < 96)")]
    InvalidClass(u8),

    /// Invalid page ID (must be < 48)
    #[error("Invalid page ID: {0} (must be < 48)")]
    InvalidPage(u8),

    /// Unity neutrality check failed
    #[error("Unity neutrality violation: net resonance delta non-zero")]
    UnityNeutralityViolation,

    /// Mirror safety check failed
    #[error("Mirror safety violation: class {class} and mirror {mirror} produced different results")]
    MirrorSafetyViolation { class: u8, mirror: u8 },

    /// Resonance ratio provided with zero denominator
    #[error("Resonance ratio has zero denominator")]
    ResonanceZeroDenominator,

    /// Resonance ratio not in canonical form (denominator positive, reduced fraction)
    #[error("Resonance ratio non-canonical: {numer}/{denom}")]
    ResonanceNonCanonical { numer: i64, denom: i64 },

    /// Resonance accumulation overflowed representable range (emits `tracing::warn`)
    #[error("Resonance arithmetic overflow: {numer}/{denom} exceeds i64 range")]
    ResonanceOverflow { numer: i128, denom: i128 },

    /// Backend-specific error
    #[error("Backend error: {0}")]
    Backend(String),

    /// Invalid metadata
    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),

    /// Kernel execution failed
    #[error("Kernel execution failed: {0}")]
    ExecutionFailed(String),

    /// Out of memory
    #[error("Out of memory: requested {requested} bytes, {available} available")]
    OutOfMemory { requested: usize, available: usize },

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

impl AtlasError {
    /// Create a phase window error
    pub fn phase_window(current: u16, begin: u16, span: u16) -> Self {
        Self::PhaseWindow {
            current,
            begin,
            end: (begin + span) % 768,
        }
    }

    /// Create a boundary violation error
    pub fn boundary_violation(x: u8, y: u8, x_range: (u8, u8), y_range: (u8, u8)) -> Self {
        Self::BoundaryViolation {
            x,
            y,
            x_min: x_range.0,
            x_max: x_range.1,
            y_min: y_range.0,
            y_max: y_range.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages() {
        let err = AtlasError::phase_window(100, 50, 20);
        assert!(err.to_string().contains("100"));

        let err = AtlasError::InvalidClass(100);
        assert!(err.to_string().contains("100"));

        let err = AtlasError::boundary_violation(50, 255, (0, 48), (0, 255));
        assert!(err.to_string().contains("50"));
        assert!(err.to_string().contains("255"));
    }
}
