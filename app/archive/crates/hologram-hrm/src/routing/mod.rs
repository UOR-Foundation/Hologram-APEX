//! Source and Destination Routing for MoonshineHRM
//!
//! This module implements the two-router architecture for compiled operations:
//!
//! ```text
//! Input Values
//!     ↓
//! [SOURCE ROUTER] - Maps inputs to (h₂, d, ℓ) coordinates
//!     ↓
//! Atlas Space (96 classes)
//!     ↓
//! [DESTINATION ROUTER] - Maps coordinates back to output values
//!     ↓
//! Output Values
//! ```
//!
//! ## Source Router
//!
//! Pre-computes input patterns → (h₂, d, ℓ) coordinate mappings at compile-time.
//! Runtime lookup is O(1) via hash table.
//!
//! ## Destination Router
//!
//! Pre-computes (h₂, d, ℓ) coordinates → output value mappings.
//! All 96 coordinate mappings are computed once at initialization.
//!
//! ## Example
//!
//! ```ignore
//! use hologram_hrm::routing::{SourceRouter, DestinationRouter};
//! use hologram_hrm::Atlas;
//!
//! let atlas = Atlas::with_cache()?;
//!
//! // Build routers for specific patterns
//! let patterns = vec![1.0, 2.0, 3.0, 5.0, 10.0];
//! let source = SourceRouter::build_for_patterns(&patterns, atlas.clone())?;
//! let dest = DestinationRouter::build(atlas.clone())?;
//!
//! // Route a value through Atlas space
//! let coords = source.route(2.0)?;  // O(1) lookup
//! let output = dest.route(coords)?;  // O(1) lookup
//! ```

pub mod destination;
pub mod source;

pub use destination::DestinationRouter;
pub use source::SourceRouter;
