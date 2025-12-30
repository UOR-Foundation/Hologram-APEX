//! # Hologram HRM (Hierarchical Representation Model)
//!
//! Compile-time memory address resolution using Griess algebra embeddings.
//!
//! ## Overview
//!
//! hologram-hrm provides a deterministic mapping from arbitrary integers to memory
//! addresses in the Atlas 12,288 × 96 address space. This is achieved through:
//!
//! 1. **Griess Algebra**: 196,884-dimensional vector space for embedding integers
//! 2. **Atlas Partition**: 96 canonical vectors generated deterministically (SplitMix64)
//! 3. **Embedding Operator E**: Maps integers → Griess vectors via composition
//! 4. **Address Projection**: Maps Griess vectors → (class, page, byte) addresses
//! 5. **Tonbo Storage**: Fast KV store with Arrow/Parquet for zero-copy reads
//!
//! ## Architecture
//!
//! ```text
//! Input Integer (BigUint)
//!     ↓ base-96 conversion
//! Symbolic Integer (digits)
//!     ↓ embedding operator E
//! Griess Vector (196,884-dim)
//!     ↓ projection
//! Address (class ∈ [0,95], page ∈ [0,47], byte ∈ [0,255])
//!     ↓ stored in Tonbo
//! Fast O(1) lookup at runtime
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use hologram_hrm::{HrmStore, resolve_address};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Open Tonbo store
//! let store = HrmStore::open_memory().await?;
//!
//! // Resolve address (compile-time or runtime)
//! let (class, phi) = resolve_address(1961, &store).await?;
//!
//! println!("Address: class={}, page={}, byte={}", class.as_u8(), phi.page, phi.byte);
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

// Public modules
pub mod address;
pub mod algebra;
pub mod arrow;
pub mod atlas;
pub mod compiler;
pub mod decode;
pub mod embed;
pub mod extract;
pub mod griess;
pub mod moonshine;
pub mod routing;
pub mod storage;
pub mod symbolic;

// Internal modules
mod error;
mod types;

// Re-exports
pub use error::{Error, Result};
pub use types::*;

// Re-export key types
pub use address::resolve_address;
pub use algebra::{Lattice, LieAlgebra, MoonshineAlgebra, Ring};
pub use atlas::{Atlas, ScaledAtlas};
pub use griess::GriessVector;
pub use moonshine::action::{GroupAction, NetworkTopology};
pub use moonshine::{MoonshineOperator, OperatorSequence};
pub use storage::HrmStore;
pub use symbolic::SymbolicInteger;
