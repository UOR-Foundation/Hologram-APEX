//! Atlas computational memory space
//!
//! Implements the core 96×48×256 boundary memory structure as specified
//! in Atlas Runtime Spec §4.

use crate::addressing::{bound_map, TOTAL_SPACE};
use crate::buffer::{BufferAllocator, BufferHandle, MemoryPool};
use crate::error::{AtlasError, Result};
use crate::phase::PhaseCounter;
use crate::resonance::ResonanceAccumulator;
use crate::topology::{MirrorTable, NeighborTable};

/// Atlas Space - The Computational Memory Model
///
/// This is the core data structure: a contiguous 1.125 MiB slab with logical
/// tiling into 96 resonance classes, each containing a 48×256 boundary lens.
///
/// ## Physical Layout
///
/// ```text
/// [Class 0: 12,288 bytes][Class 1: 12,288 bytes]...[Class 95: 12,288 bytes]
/// ```
///
/// Each class region:
/// ```text
/// [Page 0: 256 bytes][Page 1: 256 bytes]...[Page 47: 256 bytes]
/// ```
///
/// ## Why This is Fast
///
/// 1. **Tiny working sets**: Each class (12 KiB) fits in L1 cache
/// 2. **Predictable strides**: Pages are contiguous 256-byte rows
/// 3. **SIMD-friendly**: 256-byte pages divide evenly by vector widths
/// 4. **Spatial locality**: Boundary lens ensures related data is nearby
/// 5. **No coordination**: Pure address resolution, no locks needed
///
/// ## Example Usage
///
/// ```
/// use atlas_runtime::AtlasSpace;
///
/// let mut space = AtlasSpace::new();
///
/// // Write to boundary coordinate
/// space.write_boundary(0, 0, 0, 42);
///
/// // Read back
/// let value = space.read_boundary(0, 0, 0);
/// assert_eq!(value, 42);
/// ```
pub struct AtlasSpace {
    /// The F₄-derived boundary structure (FIXED at 1.125 MiB)
    /// 96 resonance classes from F₄'s 48 roots + involution
    /// This is DISCRETE mathematics - cannot be extended
    /// Aligned to 64 bytes for cache-friendly access
    boundary: Box<AlignedBoundary>,

    /// Linear memory pool (UNLIMITED)
    /// For operations that don't use the F₄ boundary structure
    /// This is just storage with no Atlas topology
    /// Grows dynamically as needed
    linear: parking_lot::RwLock<Vec<u8>>,

    /// Phase counter (mod 768) for temporal ordering
    phase: PhaseCounter,

    /// Resonance accumulator R[96] for the 96 F₄-derived classes
    resonance: ResonanceAccumulator,

    /// Mirror involution table (F₄-derived)
    mirrors: MirrorTable,

    /// Neighbor adjacency table / 1-skeleton (F₄-derived)
    neighbors: NeighborTable,

    /// Buffer allocator for managing allocations
    allocator: BufferAllocator,
}

/// Aligned boundary memory
///
/// The `#[repr(align(64))]` ensures the boundary starts at a cache-line boundary,
/// improving performance by preventing false sharing and enabling better prefetching.
#[repr(align(64))]
struct AlignedBoundary([u8; TOTAL_SPACE]);

impl AtlasSpace {
    /// Create a new Atlas space
    ///
    /// Initializes:
    /// - 1.125 MiB F₄-derived boundary memory (fixed, zeroed)
    /// - Unlimited linear memory pool (starts empty, grows as needed)
    /// - Phase counter at 0
    /// - Resonance accumulator (all zeros)
    /// - Topology tables (mirrors, neighbors from F₄)
    pub fn new() -> Self {
        // Allocate F₄ boundary structure on heap to avoid stack overflow
        let boundary = unsafe {
            let layout = std::alloc::Layout::new::<AlignedBoundary>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut AlignedBoundary;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr)
        };

        Self {
            boundary,
            linear: parking_lot::RwLock::new(Vec::new()),
            phase: PhaseCounter::new(),
            resonance: ResonanceAccumulator::new(),
            mirrors: MirrorTable::new(),
            neighbors: NeighborTable::new(),
            allocator: BufferAllocator::new(),
        }
    }

    /// Get phase counter
    pub fn phase(&self) -> &PhaseCounter {
        &self.phase
    }

    /// Get resonance accumulator
    pub fn resonance(&self) -> &ResonanceAccumulator {
        &self.resonance
    }

    /// Get resonance snapshot (R\[96\])
    ///
    /// Returns a snapshot of all 96 resonance class values.
    /// Used for trace capture and validation (ISA spec §13).
    ///
    /// # Example
    ///
    /// ```
    /// use atlas_runtime::AtlasSpace;
    ///
    /// let space = AtlasSpace::new();
    /// let snapshot = space.resonance_snapshot();
    /// assert_eq!(snapshot.len(), 96);
    /// ```
    pub fn resonance_snapshot(&self) -> [atlas_core::AtlasRatio; 96] {
        self.resonance.snapshot()
    }

    /// Get mirror table
    pub fn mirrors(&self) -> &MirrorTable {
        &self.mirrors
    }

    /// Get neighbor table
    pub fn neighbors(&self) -> &NeighborTable {
        &self.neighbors
    }

    /// Get neighbor class by direction index
    pub fn get_neighbor(&self, class: u8, direction: u8) -> u8 {
        let neighbors = self.neighbors.neighbors(class);
        neighbors[direction as usize]
    }

    /// Get mirror of a class
    pub fn get_mirror(&self, class: u8) -> u8 {
        self.mirrors.mirror(class)
    }

    /// Accumulate resonance delta for a class
    pub fn accumulate_resonance(&mut self, class: u8, numer: i64, denom: i64) -> Result<()> {
        use atlas_core::AtlasRatio;
        let delta = AtlasRatio::new_raw(numer, denom);
        self.resonance.add(class, delta)
    }

    /// Read a single byte from boundary coordinates
    ///
    /// # Performance
    ///
    /// This is a simple array access after address calculation (2-3 instructions).
    ///
    /// # Example
    ///
    /// ```
    /// use atlas_runtime::AtlasSpace;
    ///
    /// let space = AtlasSpace::new();
    /// let byte = space.read_boundary(0, 0, 0);
    /// ```
    #[inline]
    pub fn read_boundary(&self, class: u8, page: u8, byte: u8) -> u8 {
        let offset = bound_map(class, page, byte);
        self.boundary.0[offset]
    }

    /// Write a single byte to boundary coordinates
    #[inline]
    pub fn write_boundary(&mut self, class: u8, page: u8, byte: u8, value: u8) {
        let offset = bound_map(class, page, byte);
        self.boundary.0[offset] = value;
    }

    /// Get a slice of a page (256 bytes)
    ///
    /// Returns the contiguous 256-byte page, perfect for SIMD operations.
    ///
    /// # Example
    ///
    /// ```
    /// use atlas_runtime::AtlasSpace;
    ///
    /// let space = AtlasSpace::new();
    /// let page_slice = space.get_page(0, 0);
    /// assert_eq!(page_slice.len(), 256);
    /// ```
    pub fn get_page(&self, class: u8, page: u8) -> &[u8] {
        let offset = bound_map(class, page, 0);
        &self.boundary.0[offset..offset + 256]
    }

    /// Get a mutable slice of a page
    pub fn get_page_mut(&mut self, class: u8, page: u8) -> &mut [u8] {
        let offset = bound_map(class, page, 0);
        &mut self.boundary.0[offset..offset + 256]
    }

    /// Get a slice of an entire class (12,288 bytes = 48 pages)
    ///
    /// Returns the full 12 KiB class region.
    pub fn get_class(&self, class: u8) -> &[u8] {
        let offset = bound_map(class, 0, 0);
        &self.boundary.0[offset..offset + 12_288]
    }

    /// Get a mutable slice of an entire class
    pub fn get_class_mut(&mut self, class: u8) -> &mut [u8] {
        let offset = bound_map(class, 0, 0);
        &mut self.boundary.0[offset..offset + 12_288]
    }

    /// Get raw pointer to boundary memory
    ///
    /// Useful for zero-copy interfacing with external libraries.
    ///
    /// # Safety
    ///
    /// Caller must ensure the pointer is not used after the AtlasSpace is dropped.
    pub fn as_ptr(&self) -> *const u8 {
        self.boundary.0.as_ptr()
    }

    /// Get raw mutable pointer to boundary memory
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - No concurrent access
    /// - Pointer is not used after AtlasSpace is dropped
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.boundary.0.as_mut_ptr()
    }

    /// Get the entire boundary as a slice
    pub fn as_slice(&self) -> &[u8] {
        &self.boundary.0
    }

    /// Get the entire boundary as a mutable slice
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        &mut self.boundary.0
    }

    /// Copy data from a source buffer into the boundary
    ///
    /// Copies `src` into the boundary starting at the given coordinates.
    /// Useful for initializing boundary memory from tensors.
    ///
    /// # Errors
    ///
    /// Returns error if the copy would exceed boundary limits.
    pub fn copy_from_slice(&mut self, class: u8, page: u8, byte: u8, src: &[u8]) -> Result<()> {
        let offset = bound_map(class, page, byte);
        let end = offset + src.len();

        if end > TOTAL_SPACE {
            return Err(AtlasError::InvalidMetadata(format!(
                "copy would exceed boundary: offset {} + len {} > {}",
                offset,
                src.len(),
                TOTAL_SPACE
            )));
        }

        self.boundary.0[offset..end].copy_from_slice(src);
        Ok(())
    }

    /// Copy data from the boundary to a destination buffer
    pub fn copy_to_slice(&self, class: u8, page: u8, byte: u8, dst: &mut [u8]) -> Result<()> {
        let offset = bound_map(class, page, byte);
        let end = offset + dst.len();

        if end > TOTAL_SPACE {
            return Err(AtlasError::InvalidMetadata(format!(
                "copy would exceed boundary: offset {} + len {} > {}",
                offset,
                dst.len(),
                TOTAL_SPACE
            )));
        }

        dst.copy_from_slice(&self.boundary.0[offset..end]);
        Ok(())
    }

    /// Fill a region with a value
    pub fn fill(&mut self, class: u8, page: u8, byte: u8, len: usize, value: u8) -> Result<()> {
        let offset = bound_map(class, page, byte);
        let end = offset + len;

        if end > TOTAL_SPACE {
            return Err(AtlasError::InvalidMetadata(format!(
                "fill would exceed boundary: offset {} + len {} > {}",
                offset, len, TOTAL_SPACE
            )));
        }

        self.boundary.0[offset..end].fill(value);
        Ok(())
    }

    /// Reset all boundary memory to zero
    pub fn clear(&mut self) {
        self.boundary.0.fill(0);
    }

    /// Reset entire space (boundary, phase, resonance)
    pub fn reset(&mut self) {
        self.clear();
        self.phase.reset();
        self.resonance.reset();
    }

    // ========== Buffer Management ==========

    /// Allocate a buffer in Atlas space from the specified memory pool
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes
    /// * `alignment` - Alignment requirement (must be power of 2)
    /// * `pool` - Which memory pool to allocate from
    ///
    /// # Returns
    ///
    /// A buffer handle on success, or an error if out of memory.
    ///
    /// # Example
    ///
    /// ```
    /// use atlas_runtime::{AtlasSpace, buffer::MemoryPool};
    ///
    /// let mut space = AtlasSpace::new();
    /// // Allocate from unlimited linear pool (default for most operations)
    /// let handle = space.allocate_buffer(1024, 64, MemoryPool::Linear).unwrap();
    /// assert_eq!(handle.size(), 1024);
    /// ```
    pub fn allocate_buffer(&mut self, size: usize, alignment: usize, pool: MemoryPool) -> Result<BufferHandle> {
        let handle = self.allocator.allocate(size, alignment, pool)?;

        // If allocating from linear pool, ensure we have enough physical memory
        if matches!(pool, MemoryPool::Linear) {
            let required_size = handle.end();
            let mut linear = self.linear.write();
            if linear.len() < required_size {
                // Grow with some headroom to avoid frequent reallocations
                let new_size = required_size.next_power_of_two().max(required_size);
                linear.resize(new_size, 0);
            }
        }

        Ok(handle)
    }

    /// Copy data from host slice to buffer
    ///
    /// # Arguments
    ///
    /// * `handle` - Buffer handle
    /// * `src` - Source slice
    ///
    /// # Errors
    ///
    /// Returns error if buffer size doesn't match slice length.
    pub fn buffer_copy_from_slice(&mut self, handle: &BufferHandle, src: &[u8]) -> Result<()> {
        if src.len() != handle.size() {
            return Err(AtlasError::InvalidOperation(format!(
                "Buffer size mismatch: buffer is {} bytes, slice is {} bytes",
                handle.size(),
                src.len()
            )));
        }

        let offset = handle.offset();
        match handle.pool() {
            MemoryPool::Boundary => {
                self.boundary.0[offset..offset + handle.size()].copy_from_slice(src);
            }
            MemoryPool::Linear => {
                let mut linear = self.linear.write();
                linear[offset..offset + handle.size()].copy_from_slice(src);
            }
        }
        Ok(())
    }

    /// Copy data from buffer to host slice
    ///
    /// # Arguments
    ///
    /// * `handle` - Buffer handle
    /// * `dst` - Destination slice
    ///
    /// # Errors
    ///
    /// Returns error if buffer size doesn't match slice length.
    pub fn buffer_copy_to_slice(&self, handle: &BufferHandle, dst: &mut [u8]) -> Result<()> {
        if dst.len() != handle.size() {
            return Err(AtlasError::InvalidOperation(format!(
                "Buffer size mismatch: buffer is {} bytes, slice is {} bytes",
                handle.size(),
                dst.len()
            )));
        }

        let offset = handle.offset();
        match handle.pool() {
            MemoryPool::Boundary => {
                dst.copy_from_slice(&self.boundary.0[offset..offset + handle.size()]);
            }
            MemoryPool::Linear => {
                let linear = self.linear.read();
                dst.copy_from_slice(&linear[offset..offset + handle.size()]);
            }
        }
        Ok(())
    }

    /// Get buffer as slice (read-only)
    pub fn buffer_as_slice(&self, handle: &BufferHandle) -> Result<Vec<u8>> {
        let offset = handle.offset();
        let size = handle.size();
        match handle.pool() {
            MemoryPool::Boundary => Ok(self.boundary.0[offset..offset + size].to_vec()),
            MemoryPool::Linear => {
                let linear = self.linear.read();
                Ok(linear[offset..offset + size].to_vec())
            }
        }
    }

    /// Get buffer as mutable slice - only for boundary pool
    /// Linear pool buffers require locking and cannot return raw slices
    pub fn buffer_as_slice_mut(&mut self, handle: &BufferHandle) -> Result<&mut [u8]> {
        if !matches!(handle.pool(), MemoryPool::Boundary) {
            return Err(AtlasError::InvalidOperation(
                "Cannot get mutable slice for linear pool buffers - use copy operations".to_string(),
            ));
        }
        let offset = handle.offset();
        Ok(&mut self.boundary.0[offset..offset + handle.size()])
    }

    // ========== Raw Memory Access ==========

    /// Read bytes from Atlas space at the given offset from the specified pool
    ///
    /// Used by the HGIR interpreter for memory load operations.
    pub fn read_bytes(&self, pool: MemoryPool, offset: usize, len: usize) -> Result<Vec<u8>> {
        match pool {
            MemoryPool::Boundary => {
                if offset + len > TOTAL_SPACE {
                    return Err(AtlasError::InvalidOperation(format!(
                        "Read would exceed boundary: offset {} + len {} > {}",
                        offset, len, TOTAL_SPACE
                    )));
                }
                Ok(self.boundary.0[offset..offset + len].to_vec())
            }
            MemoryPool::Linear => {
                let linear = self.linear.read();
                if offset + len > linear.len() {
                    return Err(AtlasError::InvalidOperation(format!(
                        "Read would exceed linear pool: offset {} + len {} > {}",
                        offset,
                        len,
                        linear.len()
                    )));
                }
                Ok(linear[offset..offset + len].to_vec())
            }
        }
    }

    /// Write bytes to Atlas space at the given offset in the specified pool
    ///
    /// Used by the HGIR interpreter for memory store operations.
    pub fn write_bytes(&mut self, pool: MemoryPool, offset: usize, bytes: &[u8]) -> Result<()> {
        let end = offset + bytes.len();
        match pool {
            MemoryPool::Boundary => {
                if end > TOTAL_SPACE {
                    return Err(AtlasError::InvalidOperation(format!(
                        "Write would exceed boundary: offset {} + len {} > {}",
                        offset,
                        bytes.len(),
                        TOTAL_SPACE
                    )));
                }
                self.boundary.0[offset..end].copy_from_slice(bytes);
                Ok(())
            }
            MemoryPool::Linear => {
                let mut linear = self.linear.write();
                if end > linear.len() {
                    return Err(AtlasError::InvalidOperation(format!(
                        "Write would exceed linear pool: offset {} + len {} > {}",
                        offset,
                        bytes.len(),
                        linear.len()
                    )));
                }
                linear[offset..end].copy_from_slice(bytes);
                Ok(())
            }
        }
    }
}

impl Default for AtlasSpace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atlas_space_new() {
        let space = AtlasSpace::new();

        // Boundary should be zeroed
        assert_eq!(space.read_boundary(0, 0, 0), 0);
        assert_eq!(space.read_boundary(95, 47, 255), 0);

        // Phase should be 0
        assert_eq!(space.phase().get(), 0);
    }

    #[test]
    fn test_read_write_boundary() {
        let mut space = AtlasSpace::new();

        space.write_boundary(10, 20, 128, 42);
        assert_eq!(space.read_boundary(10, 20, 128), 42);

        // Other locations should still be zero
        assert_eq!(space.read_boundary(10, 20, 127), 0);
        assert_eq!(space.read_boundary(10, 20, 129), 0);
    }

    #[test]
    fn test_get_page() {
        let mut space = AtlasSpace::new();

        // Write to first byte of a page
        space.write_boundary(5, 10, 0, 100);

        let page = space.get_page(5, 10);
        assert_eq!(page.len(), 256);
        assert_eq!(page[0], 100);
        assert_eq!(page[1], 0);
    }

    #[test]
    fn test_get_page_mut() {
        let mut space = AtlasSpace::new();

        {
            let page = space.get_page_mut(5, 10);
            page[0] = 42;
            page[255] = 99;
        }

        assert_eq!(space.read_boundary(5, 10, 0), 42);
        assert_eq!(space.read_boundary(5, 10, 255), 99);
    }

    #[test]
    fn test_get_class() {
        let mut space = AtlasSpace::new();

        space.write_boundary(3, 0, 0, 11);
        space.write_boundary(3, 47, 255, 22);

        let class_slice = space.get_class(3);
        assert_eq!(class_slice.len(), 12_288);
        assert_eq!(class_slice[0], 11);
        assert_eq!(class_slice[12_287], 22); // Last byte of class 3
    }

    #[test]
    fn test_copy_from_slice() {
        let mut space = AtlasSpace::new();

        let data = [1, 2, 3, 4, 5];
        space.copy_from_slice(0, 0, 0, &data).unwrap();

        assert_eq!(space.read_boundary(0, 0, 0), 1);
        assert_eq!(space.read_boundary(0, 0, 1), 2);
        assert_eq!(space.read_boundary(0, 0, 4), 5);
    }

    #[test]
    fn test_copy_to_slice() {
        let mut space = AtlasSpace::new();

        space.write_boundary(0, 0, 0, 10);
        space.write_boundary(0, 0, 1, 20);
        space.write_boundary(0, 0, 2, 30);

        let mut dst = [0u8; 3];
        space.copy_to_slice(0, 0, 0, &mut dst).unwrap();

        assert_eq!(dst, [10, 20, 30]);
    }

    #[test]
    fn test_fill() {
        let mut space = AtlasSpace::new();

        space.fill(0, 0, 0, 256, 0xFF).unwrap();

        // Check first page of class 0 is filled
        let page = space.get_page(0, 0);
        assert!(page.iter().all(|&b| b == 0xFF));

        // Check next page is still zero
        let page = space.get_page(0, 1);
        assert!(page.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_clear() {
        let mut space = AtlasSpace::new();

        // Write some data
        space.write_boundary(10, 20, 30, 42);
        assert_eq!(space.read_boundary(10, 20, 30), 42);

        // Clear
        space.clear();
        assert_eq!(space.read_boundary(10, 20, 30), 0);
    }

    #[test]
    fn test_reset() {
        let mut space = AtlasSpace::new();

        // Modify state
        space.write_boundary(0, 0, 0, 42);
        space.phase().advance(100);

        // Reset
        space.reset();

        assert_eq!(space.read_boundary(0, 0, 0), 0);
        assert_eq!(space.phase().get(), 0);
    }

    #[test]
    fn test_alignment() {
        let space = AtlasSpace::new();
        let ptr = space.as_ptr();

        // Should be 64-byte aligned
        assert_eq!(ptr as usize % 64, 0);
    }

    #[test]
    fn test_boundary_size() {
        use std::mem::size_of_val;
        let space = AtlasSpace::new();

        assert_eq!(size_of_val(&space.boundary.0), TOTAL_SPACE);
        assert_eq!(TOTAL_SPACE, 1_179_648);
    }
}
