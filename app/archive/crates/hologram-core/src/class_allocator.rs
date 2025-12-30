//! Class allocation and deallocation management
//!
//! Provides a shared allocator for the 96-class system with automatic
//! deallocation when buffers are dropped.

use crate::sync::Mutex;
use std::collections::HashSet;
use std::sync::Arc;

/// Shared class allocator for the 96-class system
///
/// This allocator manages class lifecycle:
/// - Allocates classes from a free list
/// - Tracks allocated classes
/// - Returns classes to free list when buffers are dropped
///
/// The allocator is thread-safe and can be shared between
/// Executor and multiple Buffer instances.
pub struct ClassAllocator {
    /// Classes currently allocated
    allocated: HashSet<u8>,
    /// Classes available for allocation (LIFO stack for cache locality)
    free_list: Vec<u8>,
}

impl ClassAllocator {
    /// Create a new class allocator with all 96 classes available
    pub fn new() -> Self {
        // Initialize free list with all classes [0, 96) in reverse order
        // This gives us LIFO semantics (better cache locality)
        let free_list: Vec<u8> = (0..96).rev().collect();

        Self {
            allocated: HashSet::new(),
            free_list,
        }
    }

    /// Allocate a class
    ///
    /// Returns the class index, or None if all classes are exhausted.
    pub fn allocate(&mut self) -> Option<u8> {
        if let Some(class) = self.free_list.pop() {
            self.allocated.insert(class);
            Some(class)
        } else {
            None
        }
    }

    /// Allocate N consecutive classes
    ///
    /// Returns the starting class index, or None if N consecutive classes cannot be allocated.
    /// This is used for multi-class buffers that span multiple classes.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of consecutive classes to allocate
    ///
    /// # Returns
    ///
    /// Starting class index if successful, None if insufficient consecutive classes available
    pub fn allocate_consecutive(&mut self, count: usize) -> Option<u8> {
        if count == 0 {
            return None;
        }

        if count == 1 {
            return self.allocate();
        }

        // Find a contiguous range of free classes
        // We need to search through the free_list to find consecutive indices
        // Build a sorted list of free classes for easier range-finding
        let mut free_classes: Vec<u8> = self.free_list.to_vec();
        free_classes.sort_unstable();

        // Find first range of 'count' consecutive free classes
        let mut start_class = None;
        for window in free_classes.windows(count) {
            // Check if this window represents consecutive indices
            let is_consecutive = window.windows(2).all(|pair| pair[1] == pair[0] + 1);
            if is_consecutive {
                start_class = Some(window[0]);
                break;
            }
        }

        if let Some(start) = start_class {
            // Remove the consecutive classes from free_list and add to allocated
            for i in 0..count {
                let class = start + i as u8;
                self.allocated.insert(class);
                // Remove from free_list
                if let Some(pos) = self.free_list.iter().position(|&c| c == class) {
                    self.free_list.swap_remove(pos);
                }
            }

            tracing::debug!(
                start_class = start,
                count = count,
                free_count = self.free_list.len(),
                allocated_count = self.allocated.len(),
                "consecutive_classes_allocated"
            );

            Some(start)
        } else {
            None
        }
    }

    /// Free a class, returning it to the free list
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The class was previously allocated
    /// - No references to this class remain
    pub fn free(&mut self, class: u8) {
        if self.allocated.remove(&class) {
            self.free_list.push(class);
            tracing::debug!(
                class = class,
                free_count = self.free_list.len(),
                allocated_count = self.allocated.len(),
                "class_deallocated"
            );
        } else {
            tracing::warn!(class = class, "Attempted to free class that was not allocated");
        }
    }

    /// Free N consecutive classes starting from start_class
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - All classes in the range [start_class, start_class + count) were previously allocated
    /// - No references to these classes remain
    ///
    /// # Arguments
    ///
    /// * `start_class` - Starting class index
    /// * `count` - Number of consecutive classes to free
    pub fn free_consecutive(&mut self, start_class: u8, count: usize) {
        for i in 0..count {
            let class = start_class + i as u8;
            if self.allocated.remove(&class) {
                self.free_list.push(class);
            } else {
                tracing::warn!(
                    class = class,
                    start_class = start_class,
                    count = count,
                    "Attempted to free class that was not allocated (in consecutive free)"
                );
            }
        }

        tracing::debug!(
            start_class = start_class,
            count = count,
            free_count = self.free_list.len(),
            allocated_count = self.allocated.len(),
            "consecutive_classes_deallocated"
        );
    }

    /// Get number of allocated classes
    pub fn allocated_count(&self) -> usize {
        self.allocated.len()
    }

    /// Get number of free classes
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Check if a class is currently allocated
    pub fn is_allocated(&self, class: u8) -> bool {
        self.allocated.contains(&class)
    }
}

impl Default for ClassAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe wrapper for ClassAllocator
pub type SharedClassAllocator = Arc<Mutex<ClassAllocator>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_initialization() {
        let allocator = ClassAllocator::new();
        assert_eq!(allocator.free_count(), 96);
        assert_eq!(allocator.allocated_count(), 0);
    }

    #[test]
    fn test_allocate_and_free() {
        let mut allocator = ClassAllocator::new();

        // Allocate a class
        let class = allocator.allocate().unwrap();
        assert!(class < 96);
        assert_eq!(allocator.free_count(), 95);
        assert_eq!(allocator.allocated_count(), 1);
        assert!(allocator.is_allocated(class));

        // Free the class
        allocator.free(class);
        assert_eq!(allocator.free_count(), 96);
        assert_eq!(allocator.allocated_count(), 0);
        assert!(!allocator.is_allocated(class));
    }

    #[test]
    fn test_exhaust_all_classes() {
        let mut allocator = ClassAllocator::new();
        let mut classes = Vec::new();

        // Allocate all 96 classes
        for _ in 0..96 {
            let class = allocator.allocate().unwrap();
            classes.push(class);
        }

        assert_eq!(allocator.free_count(), 0);
        assert_eq!(allocator.allocated_count(), 96);

        // Attempt to allocate one more should fail
        assert!(allocator.allocate().is_none());

        // Free all classes
        for class in classes {
            allocator.free(class);
        }

        assert_eq!(allocator.free_count(), 96);
        assert_eq!(allocator.allocated_count(), 0);
    }

    #[test]
    fn test_lifo_semantics() {
        let mut allocator = ClassAllocator::new();

        // Allocate and free a class
        let class1 = allocator.allocate().unwrap();
        allocator.free(class1);

        // Next allocation should return the same class (LIFO)
        let class2 = allocator.allocate().unwrap();
        assert_eq!(class1, class2);
    }

    #[test]
    fn test_shared_allocator() {
        let allocator = Arc::new(Mutex::new(ClassAllocator::new()));

        // Simulate multiple threads using the allocator
        let allocator_clone = Arc::clone(&allocator);
        let class = {
            let mut guard = allocator_clone.lock();
            guard.allocate().unwrap()
        };

        // Verify allocation from different reference
        {
            let guard = allocator.lock();
            assert!(guard.is_allocated(class));
            assert_eq!(guard.allocated_count(), 1);
        }
    }

    #[test]
    fn test_allocate_consecutive_basic() {
        let mut allocator = ClassAllocator::new();

        // Allocate 4 consecutive classes
        let start = allocator.allocate_consecutive(4).unwrap();
        assert!(start <= 92); // Must fit within 96 classes (92 + 4 = 96)

        // Verify all 4 classes are allocated
        for i in 0..4 {
            assert!(allocator.is_allocated(start + i));
        }

        assert_eq!(allocator.allocated_count(), 4);
        assert_eq!(allocator.free_count(), 92);
    }

    #[test]
    fn test_allocate_consecutive_and_free() {
        let mut allocator = ClassAllocator::new();

        // Allocate 8 consecutive classes
        let start = allocator.allocate_consecutive(8).unwrap();
        assert_eq!(allocator.allocated_count(), 8);

        // Free them
        allocator.free_consecutive(start, 8);
        assert_eq!(allocator.allocated_count(), 0);
        assert_eq!(allocator.free_count(), 96);
    }

    #[test]
    fn test_allocate_consecutive_large() {
        let mut allocator = ClassAllocator::new();

        // Allocate large consecutive block (for large buffers like VAE weights)
        let start = allocator.allocate_consecutive(10).unwrap();
        assert_eq!(allocator.allocated_count(), 10);

        // Verify all 10 are consecutive
        for i in 0..10 {
            assert!(allocator.is_allocated(start + i));
        }
    }

    #[test]
    fn test_allocate_consecutive_after_fragmentation() {
        let mut allocator = ClassAllocator::new();

        // Create fragmentation by allocating and freeing non-consecutive classes
        let c1 = allocator.allocate().unwrap();
        let _c2 = allocator.allocate().unwrap();
        let _c3 = allocator.allocate().unwrap();
        let c4 = allocator.allocate().unwrap();

        // Free c1 and c4 to create gaps
        allocator.free(c1);
        allocator.free(c4);

        // Should still be able to find consecutive range
        let _start = allocator.allocate_consecutive(5).unwrap();
        assert_eq!(allocator.allocated_count(), 7); // 2 + 5 = 7 (_c2, _c3 from before, plus 5 new)
    }

    #[test]
    fn test_allocate_consecutive_zero() {
        let mut allocator = ClassAllocator::new();

        // Allocating 0 classes should return None
        assert!(allocator.allocate_consecutive(0).is_none());
    }

    #[test]
    fn test_allocate_consecutive_one() {
        let mut allocator = ClassAllocator::new();

        // Allocating 1 class should work like regular allocate
        let class = allocator.allocate_consecutive(1).unwrap();
        assert!(class < 96);
        assert_eq!(allocator.allocated_count(), 1);
    }

    #[test]
    fn test_allocate_consecutive_exhaustion() {
        let mut allocator = ClassAllocator::new();

        // Allocate most classes, leaving gaps
        for i in (0..90).step_by(2) {
            // Allocate every other class to create fragmentation
            let start = i as u8;
            if !allocator.is_allocated(start) {
                // Manually allocate specific class by removing from free_list
                if let Some(pos) = allocator.free_list.iter().position(|&c| c == start) {
                    allocator.free_list.remove(pos);
                    allocator.allocated.insert(start);
                }
            }
        }

        // Now try to allocate 10 consecutive - should fail due to fragmentation
        assert!(allocator.allocate_consecutive(10).is_none());
    }
}
