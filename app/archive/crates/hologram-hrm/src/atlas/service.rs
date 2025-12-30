//! Atlas service for managing canonical vectors
//!
//! This module provides the high-level Atlas service that manages the 96
//! canonical vectors. It integrates with HrmStore for persistence and
//! provides efficient access to Atlas vectors.
//!
//! # Design
//!
//! The Atlas can operate in two modes:
//! 1. **On-demand generation**: Generate vectors as needed (no storage)
//! 2. **Cached mode**: Pre-generate and store all 96 vectors for fast access
//!
//! # Example
//!
//! ```ignore
//! use hologram_hrm::atlas::Atlas;
//!
//! // Create Atlas (on-demand generation)
//! let atlas = Atlas::new();
//!
//! // Get a canonical vector
//! let vector = atlas.get_vector(42)?;
//!
//! // Or create with pre-generated cache
//! let atlas = Atlas::with_cache()?;
//! ```

use crate::griess::GriessVector;
use crate::storage::HrmStore;
use crate::{Error, Result, ATLAS_CLASSES};
use std::sync::Arc;

use super::generator::generate_atlas_vector;

/// Atlas service providing access to the 96 canonical vectors
///
/// The Atlas manages the canonical vectors that form the basis of the
/// HRM address space. It can operate on-demand (generating vectors as needed)
/// or with a pre-generated cache for faster access.
pub struct Atlas {
    /// Optional cache of pre-generated vectors
    /// If None, vectors are generated on-demand
    cache: Option<Vec<Arc<GriessVector>>>,
}

impl Atlas {
    /// Create a new Atlas service (on-demand generation mode)
    ///
    /// Vectors are generated lazily as needed. This has lower memory overhead
    /// but slightly higher latency on first access.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let atlas = Atlas::new();
    /// let vector = atlas.get_vector(0)?;
    /// ```
    pub fn new() -> Self {
        Self { cache: None }
    }

    /// Create an Atlas with pre-generated cache
    ///
    /// All 96 vectors are generated upfront and cached in memory.
    /// This provides O(1) access with ~155MB memory overhead.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let atlas = Atlas::with_cache()?;
    /// let vector = atlas.get_vector(42)?;  // Instant access
    /// ```
    pub fn with_cache() -> Result<Self> {
        let vectors: Result<Vec<Arc<GriessVector>>> = (0..ATLAS_CLASSES)
            .map(|class| generate_atlas_vector(class).map(Arc::new))
            .collect();

        Ok(Self { cache: Some(vectors?) })
    }

    /// Create an Atlas from an existing HrmStore
    ///
    /// This loads the Atlas partition from a pre-built store.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use hologram_hrm::storage::HrmStore;
    /// use std::path::Path;
    ///
    /// let store = HrmStore::load_atlas_only(Path::new("atlas.parquet"))?;
    /// let atlas = Atlas::from_store(store)?;
    /// ```
    pub fn from_store(store: HrmStore) -> Result<Self> {
        let mut vectors = Vec::new();

        for class in 0..ATLAS_CLASSES {
            let vector = store.get_atlas_vector(class)?;
            vectors.push(vector);
        }

        Ok(Self { cache: Some(vectors) })
    }

    /// Get the canonical vector for the given class
    ///
    /// # Arguments
    ///
    /// * `class` - The class index (must be in [0, 95])
    ///
    /// # Returns
    ///
    /// An Arc<GriessVector> for zero-copy sharing.
    ///
    /// # Errors
    ///
    /// Returns `Error::ClassOutOfRange` if class >= 96.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let atlas = Atlas::new();
    /// let vector = atlas.get_vector(42)?;
    /// assert_eq!(vector.len(), 196_884);
    /// ```
    pub fn get_vector(&self, class: u8) -> Result<Arc<GriessVector>> {
        if class >= ATLAS_CLASSES {
            return Err(Error::ClassOutOfRange(class));
        }

        match &self.cache {
            Some(vectors) => {
                // Cached mode: return Arc clone (cheap)
                Ok(Arc::clone(&vectors[class as usize]))
            }
            None => {
                // On-demand mode: generate the vector
                generate_atlas_vector(class).map(Arc::new)
            }
        }
    }

    /// Check if the Atlas is operating in cached mode
    pub fn is_cached(&self) -> bool {
        self.cache.is_some()
    }

    /// Get the number of cached vectors (0 if on-demand mode)
    pub fn cached_count(&self) -> usize {
        self.cache.as_ref().map(|v| v.len()).unwrap_or(0)
    }

    /// Save the Atlas partition to an HrmStore
    ///
    /// This generates (or retrieves) all 96 vectors and stores them
    /// in the given HrmStore for persistence.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use hologram_hrm::storage::HrmStore;
    /// use std::path::Path;
    ///
    /// let atlas = Atlas::with_cache()?;
    /// let mut store = HrmStore::new();
    ///
    /// atlas.save_to_store(&mut store)?;
    /// store.save_atlas_to_parquet(Path::new("atlas.parquet"))?;
    /// ```
    pub fn save_to_store(&self, store: &mut HrmStore) -> Result<()> {
        for class in 0..ATLAS_CLASSES {
            let vector = self.get_vector(class)?;
            // Dereference Arc to get GriessVector, then clone the data
            store.insert_atlas_vector(class, (*vector).clone())?;
        }
        Ok(())
    }
}

impl Default for Atlas {
    fn default() -> Self {
        Self::new()
    }
}

// Cheap cloning via Arc
impl Clone for Atlas {
    fn clone(&self) -> Self {
        Self {
            cache: self.cache.clone(), // Arc clones are cheap
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atlas_new() {
        let atlas = Atlas::new();
        assert!(!atlas.is_cached());
        assert_eq!(atlas.cached_count(), 0);
    }

    #[test]
    fn test_atlas_with_cache() {
        let atlas = Atlas::with_cache().unwrap();
        assert!(atlas.is_cached());
        assert_eq!(atlas.cached_count(), 96);
    }

    #[test]
    fn test_get_vector_on_demand() {
        let atlas = Atlas::new();
        let vector = atlas.get_vector(42).unwrap();
        assert_eq!(vector.len(), 196_884);
    }

    #[test]
    fn test_get_vector_cached() {
        let atlas = Atlas::with_cache().unwrap();
        let vector = atlas.get_vector(42).unwrap();
        assert_eq!(vector.len(), 196_884);
    }

    #[test]
    fn test_get_vector_invalid_class() {
        let atlas = Atlas::new();
        let result = atlas.get_vector(96);
        assert!(matches!(result, Err(Error::ClassOutOfRange(96))));
    }

    #[test]
    fn test_cached_vectors_identical_to_generated() {
        let atlas_cached = Atlas::with_cache().unwrap();
        let atlas_on_demand = Atlas::new();

        for class in 0..96 {
            let v_cached = atlas_cached.get_vector(class).unwrap();
            let v_on_demand = atlas_on_demand.get_vector(class).unwrap();

            assert_eq!(*v_cached, *v_on_demand, "Mismatch for class {}", class);
        }
    }

    #[test]
    fn test_save_to_store() {
        let atlas = Atlas::with_cache().unwrap();
        let mut store = HrmStore::new();

        atlas.save_to_store(&mut store).unwrap();

        // Verify all vectors were saved
        assert_eq!(store.atlas_count(), 96);

        // Verify vectors match
        for class in 0..96 {
            let from_atlas = atlas.get_vector(class).unwrap();
            let from_store = store.get_atlas_vector(class).unwrap();
            assert_eq!(*from_atlas, *from_store);
        }
    }

    #[test]
    fn test_atlas_from_store() {
        // Create and populate a store
        let atlas1 = Atlas::with_cache().unwrap();
        let mut store = HrmStore::new();
        atlas1.save_to_store(&mut store).unwrap();

        // Create Atlas from store
        let atlas2 = Atlas::from_store(store).unwrap();

        assert!(atlas2.is_cached());
        assert_eq!(atlas2.cached_count(), 96);

        // Verify vectors match
        for class in 0..96 {
            let v1 = atlas1.get_vector(class).unwrap();
            let v2 = atlas2.get_vector(class).unwrap();
            assert_eq!(*v1, *v2);
        }
    }

    #[test]
    fn test_atlas_clone() {
        let atlas1 = Atlas::with_cache().unwrap();
        let atlas2 = atlas1.clone();

        assert!(atlas2.is_cached());
        assert_eq!(atlas2.cached_count(), 96);

        // Verify vectors are shared (Arc pointers equal)
        let v1 = atlas1.get_vector(0).unwrap();
        let v2 = atlas2.get_vector(0).unwrap();
        assert!(Arc::ptr_eq(&v1, &v2));
    }
}
