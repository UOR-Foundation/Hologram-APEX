//! Address resolution and boundary lens (Φ) implementation
//!
//! This module implements BOUND.MAP - the core addressing primitive that transforms
//! Atlas coordinates (class, page, byte) into linear offsets in the computational
//! memory space.

/// Size of each page in bytes (matches boundary Y dimension)
pub const PAGE_SIZE: usize = 256;

/// Number of pages per resonance class (matches boundary X dimension)
pub const PAGES_PER_CLASS: usize = 48;

/// Stride between resonance classes in bytes
/// Each class has 48 pages × 256 bytes = 12,288 bytes (12 KiB)
pub const CLASS_STRIDE: usize = PAGES_PER_CLASS * PAGE_SIZE; // 12_288

/// Total size of Atlas space: 96 classes × 12 KiB
pub const TOTAL_SPACE: usize = 96 * CLASS_STRIDE; // 1_179_648 bytes

/// BOUND.MAP: Transform Atlas coordinates to linear offset
///
/// This is the fundamental addressing primitive. It maps logical Atlas coordinates
/// `(class, page, byte)` to a linear offset in the contiguous memory slab.
///
/// # Memory Layout Guarantees
///
/// - **Byte dimension (256)**: Contiguous, stride-1 access
///   - Perfect for cache lines (64 bytes)
///   - Perfect for SIMD (32/64/128 byte lanes)
/// - **Page dimension (48)**: Stride-256 access
///   - Each page is a contiguous 256-byte row
/// - **Class dimension (96)**: Stride-12288 access
///   - Each class region fits in L1 cache (12 KiB)
///
/// # Performance
///
/// This function is marked `#[inline(always)]` and compiles to 2-3 instructions:
/// - `class * 12_288`: 1 mul (or shift if compiler optimizes)
/// - `page * 256`: 1 shift (8 bits)
/// - Final add: 1 add
///
/// # Example
///
/// ```
/// use atlas_runtime::bound_map;
///
/// // Access first byte of first page in class 0
/// assert_eq!(bound_map(0, 0, 0), 0);
///
/// // Access first byte of second page in class 0
/// assert_eq!(bound_map(0, 1, 0), 256);
///
/// // Access first byte of first page in class 1
/// assert_eq!(bound_map(1, 0, 0), 12_288);
/// ```
#[inline(always)]
pub const fn bound_map(class: u8, page: u8, byte: u8) -> usize {
    debug_assert!(class < 96, "class must be < 96");
    debug_assert!(page < 48, "page must be < 48");

    (class as usize) * CLASS_STRIDE + (page as usize) * PAGE_SIZE + (byte as usize)
}

/// Alias for bound_map (matches Atlas ISA spec naming)
pub const BOUND_MAP: fn(u8, u8, u8) -> usize = bound_map;

/// Inverse of BOUND.MAP: linear offset → (class, page, byte)
///
/// Decomposes a linear offset back into Atlas coordinates.
/// Useful for debugging, validation, and address space introspection.
///
/// # Example
///
/// ```
/// use atlas_runtime::{bound_map, bound_unmap};
///
/// let offset = bound_map(42, 10, 128);
/// let (class, page, byte) = bound_unmap(offset);
/// assert_eq!((class, page, byte), (42, 10, 128));
/// ```
#[inline(always)]
pub const fn bound_unmap(offset: usize) -> (u8, u8, u8) {
    let class = (offset / CLASS_STRIDE) as u8;
    let rem = offset % CLASS_STRIDE;
    let page = (rem / PAGE_SIZE) as u8;
    let byte = (rem % PAGE_SIZE) as u8;
    (class, page, byte)
}

/// Boundary Lens (Φ) Descriptor
///
/// Describes how to map boundary coordinates (x, y) to global addresses.
/// Per Atlas Runtime Spec §4.3, this enables flexible memory layouts while
/// maintaining the logical structure.
///
/// # Layout Strategies
///
/// - **Identity/Affine**: Direct linear mapping (default for contiguous tensors)
/// - **Tiled**: Explicit 48×256 tiles per class
/// - **Swizzled**: Morton/Z-order for 2D locality
/// - **Per-Class**: Different base pointers per resonance class
#[derive(Debug, Clone, Copy)]
pub struct PhiDesc {
    /// Base pointer (global address)
    pub base_ptr: u64,

    /// Address delta for x increment (page dimension)
    pub stride_x: i64,

    /// Address delta for y increment (byte dimension)
    pub stride_y: i64,

    /// Optional swizzle pattern ID (0 = none)
    pub swizzle_id: u16,

    /// Boundary window X range [x_min, x_max)
    pub window_x: (u8, u8),

    /// Boundary window Y range [y_min, y_max)
    pub window_y: (u16, u16),

    /// Per-class offset (added to base_ptr)
    pub class_offset: u64,
}

impl PhiDesc {
    /// Create a simple identity mapping for contiguous memory
    ///
    /// This is the most common case: tensor data is contiguous in row-major order.
    pub const fn identity(base_ptr: u64) -> Self {
        Self {
            base_ptr,
            stride_x: PAGE_SIZE as i64, // Page stride
            stride_y: 1,                // Byte stride (contiguous)
            swizzle_id: 0,
            window_x: (0, 48),
            window_y: (0, 256),
            class_offset: 0,
        }
    }

    /// Create a class-tiled mapping
    ///
    /// Each class has its own 12 KiB region at `base_ptr + class * CLASS_STRIDE`.
    pub const fn class_tiled(base_ptr: u64) -> Self {
        Self {
            base_ptr,
            stride_x: PAGE_SIZE as i64,
            stride_y: 1,
            swizzle_id: 0,
            window_x: (0, 48),
            window_y: (0, 256),
            class_offset: CLASS_STRIDE as u64,
        }
    }

    /// Map boundary coordinates to global address
    ///
    /// Implements: `addr = base_ptr + class_offset*class + stride_x*x + stride_y*y`
    #[inline]
    pub fn map(&self, class: u8, x: u8, y: u8) -> u64 {
        let offset = (self.class_offset * class as u64) as i64 + self.stride_x * x as i64 + self.stride_y * y as i64;

        (self.base_ptr as i64 + offset) as u64
    }

    /// Check if coordinates are within window
    #[inline]
    pub fn in_window(&self, x: u8, y: u8) -> bool {
        x >= self.window_x.0 && x < self.window_x.1 && (y as u16) >= self.window_y.0 && (y as u16) < self.window_y.1
    }
}

impl Default for PhiDesc {
    fn default() -> Self {
        Self::identity(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bound_map_first_element() {
        assert_eq!(bound_map(0, 0, 0), 0);
    }

    #[test]
    fn test_bound_map_page_stride() {
        // Second page in class 0
        assert_eq!(bound_map(0, 1, 0), PAGE_SIZE);
        assert_eq!(bound_map(0, 2, 0), PAGE_SIZE * 2);
    }

    #[test]
    fn test_bound_map_class_stride() {
        // First byte of class 1
        assert_eq!(bound_map(1, 0, 0), CLASS_STRIDE);
        // First byte of class 2
        assert_eq!(bound_map(2, 0, 0), CLASS_STRIDE * 2);
    }

    #[test]
    fn test_bound_map_roundtrip() {
        for class in [0, 42, 95] {
            for page in [0, 10, 47] {
                for byte in [0, 128, 255] {
                    let offset = bound_map(class, page, byte);
                    let (c2, p2, b2) = bound_unmap(offset);
                    assert_eq!((class, page, byte), (c2, p2, b2));
                }
            }
        }
    }

    #[test]
    fn test_bound_map_max_offset() {
        // Last byte of last page in last class
        let max = bound_map(95, 47, 255);
        assert_eq!(max, TOTAL_SPACE - 1);
    }

    #[test]
    fn test_phi_desc_identity() {
        let phi = PhiDesc::identity(0x1000);

        // Map corner cases
        assert_eq!(phi.map(0, 0, 0), 0x1000);
        assert_eq!(phi.map(0, 1, 0), 0x1000 + PAGE_SIZE as u64);
        assert_eq!(phi.map(0, 0, 1), 0x1000 + 1);
    }

    #[test]
    fn test_phi_desc_class_tiled() {
        let phi = PhiDesc::class_tiled(0x1000);

        assert_eq!(phi.map(0, 0, 0), 0x1000);
        assert_eq!(phi.map(1, 0, 0), 0x1000 + CLASS_STRIDE as u64);
        assert_eq!(phi.map(2, 0, 0), 0x1000 + CLASS_STRIDE as u64 * 2);
    }

    #[test]
    fn test_phi_window() {
        let mut phi = PhiDesc::identity(0);
        phi.window_x = (5, 15);
        phi.window_y = (100u16, 200u16);

        assert!(!phi.in_window(0, 128)); // x out of range
        assert!(!phi.in_window(10, 50)); // y out of range
        assert!(phi.in_window(10, 150)); // both in range
        assert!(phi.in_window(10, 199));
    }

    #[test]
    fn test_phi_window_allows_full_byte_range() {
        let phi = PhiDesc::identity(0);
        assert_eq!(phi.window_y, (0u16, 256u16));
        assert!(phi.in_window(10, 0));
        assert!(phi.in_window(10, 255));
    }

    #[test]
    fn test_total_space_size() {
        // Verify the space is ~1.125 MiB as documented
        assert_eq!(TOTAL_SPACE, 1_179_648);
        assert_eq!(TOTAL_SPACE, 96 * 48 * 256);
    }

    #[test]
    fn test_class_stride_fits_l1() {
        // Each class is 12 KiB, should fit in modern L1 cache (32-64 KiB)
        assert_eq!(CLASS_STRIDE, 12_288);
        assert!(CLASS_STRIDE <= 64 * 1024); // Typical L1 size
    }
}
