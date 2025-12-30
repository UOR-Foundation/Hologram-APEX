//! Domain Head Registry
//!
//! Runtime registry for domain heads, enabling dynamic registration and lookup.

use super::traits::{DomainHead, Modality};
use crate::{memory::UniversalMemoryPool, ProcessorError, Result};
use std::collections::HashMap;

/// Registry for domain heads
///
/// Allows runtime registration of domain heads by mediatype,
/// enabling dynamic dispatch to different interpreters.
pub struct DomainHeadRegistry {
    heads: HashMap<String, Box<dyn DomainHead + Send + Sync>>,
}

impl DomainHeadRegistry {
    /// Create new empty registry
    pub fn new() -> Self {
        Self { heads: HashMap::new() }
    }

    /// Register a domain head
    ///
    /// # Example
    ///
    /// ```
    /// use hologram_memory_manager::{DomainHeadRegistry, RawDomainHead};
    ///
    /// let mut registry = DomainHeadRegistry::new();
    /// registry.register(RawDomainHead);
    /// ```
    pub fn register<H: DomainHead + Send + Sync + 'static>(&mut self, head: H) {
        let mediatype = head.mediatype().to_string();
        self.heads.insert(mediatype, Box::new(head));
    }

    /// Get domain head by mediatype
    pub fn get(&self, mediatype: &str) -> Option<&(dyn DomainHead + Send + Sync)> {
        self.heads.get(mediatype).map(|b| b.as_ref())
    }

    /// Extract modality using registered domain head
    pub fn extract(&self, mediatype: &str, pool: &UniversalMemoryPool) -> Result<Modality> {
        let head = self
            .get(mediatype)
            .ok_or_else(|| ProcessorError::DomainHeadNotFound(mediatype.to_string()))?;

        head.extract(pool)
    }

    /// List all registered mediatypes
    pub fn list_mediatypes(&self) -> Vec<String> {
        self.heads.keys().cloned().collect()
    }

    /// Check if mediatype is registered
    pub fn has_mediatype(&self, mediatype: &str) -> bool {
        self.heads.contains_key(mediatype)
    }

    /// Number of registered domain heads
    pub fn len(&self) -> usize {
        self.heads.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }
}

impl Default for DomainHeadRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::RawDomainHead;

    #[test]
    fn test_registry_operations() {
        let mut registry = DomainHeadRegistry::new();

        // Initially empty
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        // Register heads
        registry.register(RawDomainHead);

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());

        // Check registered mediatypes
        assert!(registry.has_mediatype("application/octet-stream"));
        assert!(!registry.has_mediatype("application/nonexistent"));
    }

    #[test]
    fn test_registry_extraction() {
        let mut registry = DomainHeadRegistry::new();
        registry.register(RawDomainHead);

        // Create pool with data
        let mut pool = UniversalMemoryPool::new();
        let input: Vec<u8> = (0..100).collect();
        pool.embed(input, 5).unwrap();

        // Extract via registry
        let raw = registry.extract("application/octet-stream", &pool).unwrap();

        match raw {
            Modality::Raw(_) => println!("✓ Raw data extracted via registry"),
            _ => panic!("Wrong modality"),
        }
    }

    #[test]
    fn test_missing_mediatype() {
        let registry = DomainHeadRegistry::new();
        let mut pool = UniversalMemoryPool::new();
        pool.embed(b"test".to_vec(), 3).unwrap();

        let result = registry.extract("application/nonexistent", &pool);
        assert!(result.is_err());
    }
}
