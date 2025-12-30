//! Stream Execution Module
//!
//! Executes stream processing operations with automatic gauge construction.

use crate::{
    constants::DEFAULT_MAX_CHUNK_LEVELS,
    domain::{DomainHead, DomainHeadRegistry, Modality},
    stream::{Stream, StreamContext},
    Result,
};

/// Stream processor
///
/// Main execution engine for stream processing with domain heads.
pub struct StreamProcessor {
    /// Registry of domain heads
    registry: DomainHeadRegistry,

    /// Maximum primorial levels for chunking
    max_chunk_levels: usize,
}

impl StreamProcessor {
    /// Create new stream processor with default settings
    ///
    /// Uses [`DEFAULT_MAX_CHUNK_LEVELS`] for chunking. To customize,
    /// use [`StreamProcessor::with_chunk_levels`].
    pub fn new() -> Self {
        Self {
            registry: DomainHeadRegistry::new(),
            max_chunk_levels: DEFAULT_MAX_CHUNK_LEVELS,
        }
    }

    /// Register a domain head
    pub fn register_domain_head<H: DomainHead + Send + Sync + 'static>(&mut self, head: H) {
        self.registry.register(head);
    }

    /// Process input data
    pub fn process<T: bytemuck::Pod>(&self, input: Vec<T>) -> Result<StreamContext> {
        let stream = Stream::new(input);
        let chunked = stream.chunk(self.max_chunk_levels)?;
        chunked.embed()
    }

    /// Extract modality from context
    pub fn extract_modality(&self, context: &StreamContext, mediatype: &str) -> Result<Modality> {
        self.registry.extract(mediatype, &context.pool)
    }

    /// List available mediatypes
    pub fn list_mediatypes(&self) -> Vec<String> {
        self.registry.list_mediatypes()
    }

    /// Set maximum chunk levels
    pub fn set_max_chunk_levels(&mut self, levels: usize) {
        self.max_chunk_levels = levels;
    }

    /// Get maximum chunk levels
    pub fn max_chunk_levels(&self) -> usize {
        self.max_chunk_levels
    }

    /// Builder: Create processor with specific chunk levels
    pub fn with_chunk_levels(mut self, levels: usize) -> Self {
        self.max_chunk_levels = levels;
        self
    }

    /// Builder: Register domain head and return self
    pub fn with_domain_head<H: DomainHead + Send + Sync + 'static>(mut self, head: H) -> Self {
        self.register_domain_head(head);
        self
    }

    /// Check if a mediatype is registered
    pub fn has_mediatype(&self, mediatype: &str) -> bool {
        self.list_mediatypes().contains(&mediatype.to_string())
    }

    /// Get number of registered domain heads
    pub fn domain_head_count(&self) -> usize {
        self.list_mediatypes().len()
    }
}

impl Default for StreamProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::RawDomainHead;

    #[test]
    fn test_processor_creation() {
        let processor = StreamProcessor::new();
        assert_eq!(processor.max_chunk_levels, DEFAULT_MAX_CHUNK_LEVELS);
        assert!(processor.list_mediatypes().is_empty());
    }

    #[test]
    fn test_processor_with_domain_heads() {
        let mut processor = StreamProcessor::new();
        processor.register_domain_head(RawDomainHead);

        let mediatypes = processor.list_mediatypes();
        assert_eq!(mediatypes.len(), 1);
    }

    #[test]
    fn test_process_and_extract() {
        let mut processor = StreamProcessor::new();
        processor.register_domain_head(RawDomainHead);

        let input: Vec<u8> = (0..100).collect();
        let context = processor.process(input.clone()).unwrap();

        // Extract raw
        let raw = processor
            .extract_modality(&context, "application/octet-stream")
            .unwrap();
        match raw {
            Modality::Raw(data) => {
                assert_eq!(data, input);
            }
            _ => panic!("Wrong modality"),
        }
    }
}
