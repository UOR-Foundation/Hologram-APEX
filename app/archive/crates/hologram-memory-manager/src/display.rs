//! Display Trait Implementations
//!
//! Provides human-readable Display implementations for processor types.

use crate::{
    domain::Modality,
    memory::{EmbeddedBlock, EmbeddingResult},
};
use std::fmt;

// Note: ProcessorError Display is provided by thiserror::Error derive

impl fmt::Display for EmbeddedBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Block #{} [primorial={}, gauge={}, {} bytes]",
            self.index,
            self.primorial,
            self.gauge_name(),
            self.len()
        )
    }
}

impl fmt::Display for EmbeddingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Embedded {} blocks ({} bytes) with {} gauges (memory: {} bytes)",
            self.blocks_embedded, self.total_bytes, self.gauges_constructed, self.memory_used
        )
    }
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Modality::Factors(p, q) => write!(f, "Factors({} × {})", p, q),
            Modality::Spectrum(spec) => write!(f, "Spectrum({} points)", spec.len()),
            Modality::Compressed(data) => write!(f, "Compressed({} bytes)", data.len()),
            Modality::Raw(data) => write!(f, "Raw({} bytes)", data.len()),
            Modality::Normalized(data) => write!(f, "Normalized({} f32)", data.len()),
            Modality::Aggregated { min, max, sum, count } => write!(
                f,
                "Aggregated(min={:.2}, max={:.2}, sum={:.2}, count={})",
                min, max, sum, count
            ),
            Modality::Filtered(data) => write!(f, "Filtered({} f32)", data.len()),
            Modality::TextEmbedding(vec) => write!(f, "TextEmbedding({} dims)", vec.len()),
            Modality::TextCompletion { text, tokens } => {
                if text.len() > 50 {
                    write!(f, "TextCompletion('{}...', {} tokens)", &text[..50], tokens.len())
                } else {
                    write!(f, "TextCompletion('{}', {} tokens)", text, tokens.len())
                }
            }
        }
    }
}
