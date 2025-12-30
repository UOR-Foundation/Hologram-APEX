//! Primordial-Based Chunking Module
//!
//! This module provides automatic gauge construction via primorial chunking.
//! The chunking scheme IS the gauge generator - primorial-based chunk sizes
//! naturally encode prime structure, and gauges are constructed from those primes.
//!
//! # Core Concept
//!
//! ```text
//! Primordial Sequence:    [1, 2, 6, 30, 210, 2310, ...]
//!         ↓
//! Extract Primes:         [[], [2], [2,3], [2,3,5], [2,3,5,7], ...]
//!         ↓
//! Construct Gauges:       [G{2,3}, G{2,3}, G{2,3}, G{2,3,5}, G{2,3,5,7}, ...]
//! ```
//!
//! This ensures that gauge structure emerges AUTOMATICALLY from the chunking scheme,
//! with no manual selection or hardcoded enums.

pub mod gauge_map;
pub mod period_driven;
pub mod primorial;

pub use gauge_map::{gauge_for_index, gauge_for_primorial, PRIMORDIAL_GAUGE_MAP};
pub use period_driven::{ChunkWithGauge, PeriodDrivenChunker, PeriodInfo};
pub use primorial::{factor_primorial, generate_n_primorials, generate_primorial_sequence};
