//! Domain Heads Module
//!
//! Domain heads act as "mediatypes" that interpret embedded data in different ways.
//! Like MIME types for canonical compute, they enable the same universal memory pool
//! to be interpreted through different lenses.
//!
//! ## Concept
//!
//! A domain head is like a "codec" for canonical compute:
//! - **Input**: Universal embedded memory pool (content-agnostic)
//! - **Process**: Interpret blocks using their gauges
//! - **Output**: Domain-specific modality (factors, spectrum, etc.)
//!
//! ## Examples
//!
//! - **Shor's Domain Head**: Extracts semiprime factors → `application/semiprime-factors`
//! - **FFT Domain Head**: Extracts frequency spectrum → `application/frequency-spectrum`
//! - **Raw Domain Head**: Reconstructs original data → `application/octet-stream`
//!
//! Same pool, different mediatypes!
//!
//! ## Usage
//!
//! ```
//! use hologram_memory_manager::{UniversalMemoryPool, DomainHeadRegistry, RawDomainHead};
//!
//! // Create registry and register heads
//! let mut registry = DomainHeadRegistry::new();
//! registry.register(RawDomainHead);
//!
//! // Create and populate pool
//! let mut pool = UniversalMemoryPool::new();
//! let data: Vec<u8> = vec![1, 2, 3, 4, 5];
//! pool.embed(data, 7).unwrap();
//!
//! // Extract modality via registry
//! let raw = registry.extract("application/octet-stream", &pool).unwrap();
//! ```

pub mod aggregate;
pub mod filter;
pub mod normalize;
pub mod raw;
pub mod registry;
pub mod traits;

pub use aggregate::AggregateDomainHead;
pub use filter::{FilterDomainHead, FilterType};
pub use normalize::NormalizeDomainHead;
pub use raw::RawDomainHead;
pub use registry::DomainHeadRegistry;
pub use traits::{DomainHead, Modality};
