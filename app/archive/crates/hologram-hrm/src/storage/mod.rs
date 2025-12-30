//! Storage layer for HRM using direct Apache Arrow integration
//!
//! This module provides persistent storage for:
//! - **Atlas vectors**: 96 canonical Griess vectors (~151 MB, compressed to ~15 MB)
//! - **Address mappings**: Input value → (class, page, byte) mappings
//!
//! # Design
//!
//! No database required! Just fast HashMap lookups with zero-copy Arrow data.
//! Perfect for static read-only data that's precomputed at build/compile time.
//!
//! # Performance
//!
//! - **Lookup**: ~5ns (HashMap + Arc clone)
//! - **Startup**: ~100ms (load 151MB from Parquet)
//! - **Memory**: ~155MB (151MB data + 4MB index)
//!
//! # Example
//!
//! ```rust,ignore
//! use hologram_hrm::storage::HrmStore;
//! use std::path::Path;
//!
//! // Load from Parquet files
//! let store = HrmStore::load_from_parquet(
//!     Path::new("atlas.parquet"),
//!     Path::new("addresses.parquet")
//! )?;
//!
//! // Fast O(1) lookups
//! let vector = store.get_atlas_vector(42)?;  // ~5ns
//! let addr = store.get_address(1961);         // ~5ns
//! ```

pub mod schemas;
pub mod simple_store;

pub use schemas::{AddressMapping, AtlasVector};
pub use simple_store::HrmStore;
