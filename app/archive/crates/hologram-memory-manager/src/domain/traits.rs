//! Domain Head Traits and Modality Types
//!
//! Domain heads act as interpreters that extract specific modalities from
//! universal memory pools. They're like "codecs" for canonical compute.

use crate::memory::UniversalMemoryPool;
use crate::Result;

/// Domain head trait - interprets embedded data
///
/// A domain head extracts a specific modality from a universal memory pool.
/// Different heads can interpret the same pool in different ways, like
/// different "mediatypes" (similar to MIME types for web content).
pub trait DomainHead {
    /// Mediatype identifier (like MIME type)
    ///
    /// Examples:
    /// - `application/octet-stream`
    /// - `application/semiprime-factors`
    /// - `application/frequency-spectrum`
    fn mediatype(&self) -> &str;

    /// Extract modality from embedded pool
    ///
    /// Interprets the embedded blocks using their gauges to extract
    /// domain-specific information.
    fn extract(&self, pool: &UniversalMemoryPool) -> Result<Modality>;
}

/// Modality - output format of domain head
///
/// Represents different ways of interpreting the same embedded data.
#[derive(Debug, Clone)]
pub enum Modality {
    /// Prime factors (p, q) of semiprime
    Factors(u64, u64),

    /// Frequency spectrum [(frequency, magnitude)]
    Spectrum(Vec<(f32, f32)>),

    /// Compressed data
    Compressed(Vec<u8>),

    /// Raw data (pass-through reconstruction)
    Raw(Vec<u8>),

    /// Normalized f32 data in [0, 1] range
    Normalized(Vec<f32>),

    /// Aggregated statistics (min, max, mean, stddev)
    Aggregated { min: f32, max: f32, sum: f32, count: usize },

    /// Filtered f32 data (subset passing predicate)
    Filtered(Vec<f32>),

    /// Text embedding vector (for model serving)
    TextEmbedding(Vec<f32>),

    /// Generated text completion (for model serving)
    TextCompletion { text: String, tokens: Vec<usize> },
}
