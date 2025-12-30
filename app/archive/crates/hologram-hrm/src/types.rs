//! Common type definitions for hologram-hrm

/// Griess algebra vector dimension (196,884 = 196883 + 1)
pub const GRIESS_DIMENSION: usize = 196_884;

/// Number of resonance classes in Atlas
pub const ATLAS_CLASSES: u8 = 96;

/// Base for symbolic integer representation
pub const SYMBOLIC_BASE: u32 = 96;

/// Number of pages per class (from atlas-core)
pub const PAGES_PER_CLASS: u8 = 48;

/// Bytes per page (from atlas-core)
pub const BYTES_PER_PAGE: u16 = 256;

/// Total elements per class
pub const ELEMENTS_PER_CLASS: usize = 12_288; // 48 × 256

/// Total address space across all classes
pub const TOTAL_ADDRESS_SPACE: usize = 1_179_648; // 96 × 12_288
