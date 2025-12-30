//! Buffer management for Atlas space
//!
//! Provides simple linear buffer allocation within the Atlas computational memory.

use crate::addressing::TOTAL_SPACE;
use crate::error::{AtlasError, Result};
use parking_lot::Mutex;

/// Memory pool type for buffer allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryPool {
    /// F₄-derived boundary structure (fixed 1.125 MiB)
    /// Used for boundary-addressed operations with Atlas topology
    Boundary,
    /// Linear memory pool (unlimited)
    /// Used for general operations without Atlas topology
    Linear,
}

/// Buffer handle identifying an allocation in Atlas space
///
/// A buffer handle tracks the location, size, and memory pool of an allocated region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle {
    /// Starting offset in bytes from the beginning of the memory pool
    pub(crate) offset: usize,
    /// Size in bytes
    pub(crate) size: usize,
    /// Which memory pool this buffer is allocated from
    pub(crate) pool: MemoryPool,
}

impl BufferHandle {
    /// Create a new buffer handle
    pub const fn new(offset: usize, size: usize, pool: MemoryPool) -> Self {
        Self { offset, size, pool }
    }

    /// Get buffer offset
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Get buffer size in bytes
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Get memory pool
    pub const fn pool(&self) -> MemoryPool {
        self.pool
    }

    /// Get the end offset (exclusive)
    pub const fn end(&self) -> usize {
        self.offset + self.size
    }

    /// Check if this buffer overlaps with another in the same pool
    pub fn overlaps(&self, other: &BufferHandle) -> bool {
        // Buffers in different pools cannot overlap
        if !matches!(
            (self.pool, other.pool),
            (MemoryPool::Boundary, MemoryPool::Boundary) | (MemoryPool::Linear, MemoryPool::Linear)
        ) {
            return false;
        }
        !(self.end() <= other.offset() || other.end() <= self.offset())
    }
}
/// Buffer Allocator
///
/// This allocator maintains separate "high-water mark" for two memory pools:
/// - Boundary pool: Fixed F₄ structure (1.125 MiB limit)
/// - Linear pool: Unlimited growth
///
/// Deallocation is not supported - buffers live for the entire duration
/// of the Atlas space.
///
/// ## Design Rationale
///
/// A simple linear allocator is appropriate because:
/// - Buffers are typically allocated once and kept
/// - Simplicity aligns with Atlas philosophy
/// - No fragmentation concerns for typical ML workloads
#[derive(Debug)]
pub struct BufferAllocator {
    /// Next free offset in boundary pool (F₄ structure, fixed size)
    next_boundary_offset: Mutex<usize>,
    /// Next free offset in linear pool (unlimited growth)
    next_linear_offset: Mutex<usize>,
}

impl BufferAllocator {
    /// Create a new buffer allocator
    pub fn new() -> Self {
        Self {
            next_boundary_offset: Mutex::new(0),
            next_linear_offset: Mutex::new(0),
        }
    }

    /// Allocate a buffer from the specified memory pool
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
    /// use atlas_runtime::buffer::{BufferAllocator, MemoryPool};
    ///
    /// let alloc = BufferAllocator::new();
    /// let handle = alloc.allocate(1024, 64, MemoryPool::Linear).unwrap();
    /// assert_eq!(handle.size(), 1024);
    /// assert_eq!(handle.offset() % 64, 0); // Aligned
    /// ```
    pub fn allocate(&self, size: usize, alignment: usize, pool: MemoryPool) -> Result<BufferHandle> {
        if size == 0 {
            return Err(AtlasError::InvalidOperation(
                "Cannot allocate zero-sized buffer".to_string(),
            ));
        }

        if !alignment.is_power_of_two() {
            return Err(AtlasError::InvalidOperation(format!(
                "Alignment {} is not a power of 2",
                alignment
            )));
        }

        match pool {
            MemoryPool::Boundary => self.allocate_boundary(size, alignment),
            MemoryPool::Linear => self.allocate_linear(size, alignment),
        }
    }

    /// Allocate from boundary pool (F₄ structure, fixed 1.125 MiB)
    fn allocate_boundary(&self, size: usize, alignment: usize) -> Result<BufferHandle> {
        let mut next = self.next_boundary_offset.lock();

        // Align the offset
        let aligned_offset = (*next + alignment - 1) & !(alignment - 1);

        // Check if we have space in the fixed boundary
        if aligned_offset + size > TOTAL_SPACE {
            return Err(AtlasError::OutOfMemory {
                requested: size,
                available: TOTAL_SPACE.saturating_sub(aligned_offset),
            });
        }

        // Create handle and update next offset
        let handle = BufferHandle::new(aligned_offset, size, MemoryPool::Boundary);
        *next = aligned_offset + size;

        Ok(handle)
    }

    /// Allocate from linear pool (unlimited)
    fn allocate_linear(&self, size: usize, alignment: usize) -> Result<BufferHandle> {
        let mut next = self.next_linear_offset.lock();

        // Align the offset
        let aligned_offset = (*next + alignment - 1) & !(alignment - 1);

        // No space check - linear pool is unlimited
        // Create handle and update next offset
        let handle = BufferHandle::new(aligned_offset, size, MemoryPool::Linear);
        *next = aligned_offset + size;

        Ok(handle)
    }

    /// Get amount of allocated space in boundary pool
    pub fn boundary_allocated(&self) -> usize {
        *self.next_boundary_offset.lock()
    }

    /// Get amount of allocated space in linear pool
    pub fn linear_allocated(&self) -> usize {
        *self.next_linear_offset.lock()
    }

    /// Get amount of free space in boundary pool
    pub fn boundary_free(&self) -> usize {
        TOTAL_SPACE.saturating_sub(self.boundary_allocated())
    }

    /// Reset allocator (deallocates all buffers from both pools)
    ///
    /// # Safety
    ///
    /// This should only be called when no buffer handles are in use.
    pub fn reset(&self) {
        *self.next_boundary_offset.lock() = 0;
        *self.next_linear_offset.lock() = 0;
    }
}

impl Default for BufferAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_handle_creation() {
        let handle = BufferHandle::new(0, 1024, MemoryPool::Linear);
        assert_eq!(handle.offset(), 0);
        assert_eq!(handle.size(), 1024);
        assert_eq!(handle.end(), 1024);
        assert_eq!(handle.pool(), MemoryPool::Linear);
    }

    #[test]
    fn test_buffer_handle_overlap() {
        let h1 = BufferHandle::new(0, 100, MemoryPool::Linear);
        let h2 = BufferHandle::new(50, 100, MemoryPool::Linear);
        let h3 = BufferHandle::new(200, 100, MemoryPool::Linear);
        let h4 = BufferHandle::new(0, 100, MemoryPool::Boundary);

        assert!(h1.overlaps(&h2));
        assert!(h2.overlaps(&h1));
        assert!(!h1.overlaps(&h3));
        assert!(!h3.overlaps(&h1));
        // Different pools don't overlap
        assert!(!h1.overlaps(&h4));
        assert!(!h4.overlaps(&h1));
    }

    #[test]
    fn test_allocator_simple() {
        let alloc = BufferAllocator::new();
        let h1 = alloc.allocate(1024, 1, MemoryPool::Linear).unwrap();
        assert_eq!(h1.offset(), 0);
        assert_eq!(h1.size(), 1024);
        assert_eq!(h1.pool(), MemoryPool::Linear);
    }

    #[test]
    fn test_allocator_alignment() {
        let alloc = BufferAllocator::new();
        let h1 = alloc.allocate(10, 1, MemoryPool::Linear).unwrap(); // offset 0
        let h2 = alloc.allocate(10, 64, MemoryPool::Linear).unwrap(); // offset aligned to 64
        assert_eq!(h2.offset() % 64, 0);
        assert!(h2.offset() >= h1.end());
    }

    #[test]
    fn test_allocator_sequential() {
        let alloc = BufferAllocator::new();
        let h1 = alloc.allocate(100, 1, MemoryPool::Linear).unwrap();
        let h2 = alloc.allocate(200, 1, MemoryPool::Linear).unwrap();
        let h3 = alloc.allocate(300, 1, MemoryPool::Linear).unwrap();

        assert_eq!(h1.offset(), 0);
        assert_eq!(h2.offset(), 100);
        assert_eq!(h3.offset(), 300);
        assert!(!h1.overlaps(&h2));
        assert!(!h2.overlaps(&h3));
    }

    #[test]
    fn test_allocator_boundary_out_of_memory() {
        let alloc = BufferAllocator::new();
        let result = alloc.allocate(TOTAL_SPACE + 1, 1, MemoryPool::Boundary);
        assert!(result.is_err());
    }

    #[test]
    fn test_allocator_linear_large() {
        let alloc = BufferAllocator::new();
        // Linear pool is unlimited - can allocate beyond TOTAL_SPACE
        let result = alloc.allocate(TOTAL_SPACE * 10, 1, MemoryPool::Linear);
        assert!(result.is_ok());
    }

    #[test]
    fn test_allocator_zero_size() {
        let alloc = BufferAllocator::new();
        let result = alloc.allocate(0, 1, MemoryPool::Linear);
        assert!(result.is_err());
    }

    #[test]
    fn test_allocator_invalid_alignment() {
        let alloc = BufferAllocator::new();
        let result = alloc.allocate(100, 3, MemoryPool::Linear); // Not power of 2
        assert!(result.is_err());
    }

    #[test]
    fn test_allocator_boundary_free_space() {
        let alloc = BufferAllocator::new();
        assert_eq!(alloc.boundary_free(), TOTAL_SPACE);
        alloc.allocate(1000, 1, MemoryPool::Boundary).unwrap();
        assert_eq!(alloc.boundary_free(), TOTAL_SPACE - 1000);
    }

    #[test]
    fn test_allocator_reset() {
        let alloc = BufferAllocator::new();
        alloc.allocate(1000, 1, MemoryPool::Boundary).unwrap();
        alloc.allocate(2000, 1, MemoryPool::Linear).unwrap();
        assert_eq!(alloc.boundary_allocated(), 1000);
        assert_eq!(alloc.linear_allocated(), 2000);
        alloc.reset();
        assert_eq!(alloc.boundary_allocated(), 0);
        assert_eq!(alloc.linear_allocated(), 0);
    }

    #[test]
    fn test_dual_pools_independent() {
        let alloc = BufferAllocator::new();
        let h1 = alloc.allocate(1000, 1, MemoryPool::Boundary).unwrap();
        let h2 = alloc.allocate(1000, 1, MemoryPool::Linear).unwrap();

        // Both start at offset 0 in their respective pools
        assert_eq!(h1.offset(), 0);
        assert_eq!(h2.offset(), 0);
        // But they're in different pools so they don't overlap
        assert!(!h1.overlaps(&h2));
    }
}
