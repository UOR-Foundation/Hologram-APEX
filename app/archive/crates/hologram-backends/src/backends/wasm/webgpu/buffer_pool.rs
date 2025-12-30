//! GPU Buffer pooling for reduced allocation overhead
//!
//! Maintains a pool of reusable GPU buffers to minimize allocation/deallocation
//! costs and reduce CPU-GPU synchronization overhead.

use crate::sync::{lock_mutex, Mutex};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Buffer, BufferUsages, Device, Queue};

/// Buffer pool for GPU memory reuse
///
/// Maintains pools of buffers organized by size to enable efficient reuse
/// without repeated allocation/deallocation overhead.
///
/// # Performance Benefits
///
/// - **Reduced allocations**: Reuse existing buffers instead of creating new ones
/// - **Lower latency**: Avoid GPU memory allocation overhead
/// - **Better memory locality**: Buffers stay warm in GPU cache
/// - **Fewer sync points**: Persistent buffers reduce CPU-GPU synchronization
pub struct BufferPool {
    device: Arc<Device>,
    queue: Arc<Queue>,
    /// Pools organized by buffer size in bytes
    /// Key: size in bytes, Value: list of available buffers
    storage_pools: Mutex<HashMap<usize, Vec<Arc<Buffer>>>>,
    staging_pools: Mutex<HashMap<usize, Vec<Arc<Buffer>>>>,
    /// Statistics for monitoring pool efficiency
    stats: Mutex<PoolStats>,
}

/// Buffer pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total number of allocations requested
    pub total_allocations: usize,
    /// Number of allocations served from pool (cache hits)
    pub pool_hits: usize,
    /// Number of new allocations created (cache misses)
    pub pool_misses: usize,
    /// Total number of buffers returned to pool
    pub total_returns: usize,
    /// Current number of storage buffers in pool
    pub storage_buffers_pooled: usize,
    /// Current number of staging buffers in pool
    pub staging_buffers_pooled: usize,
}

impl PoolStats {
    /// Calculate hit rate (percentage of allocations served from pool)
    pub fn hit_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            return 0.0;
        }
        (self.pool_hits as f64) / (self.total_allocations as f64) * 100.0
    }
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device,
            queue,
            storage_pools: Mutex::new(HashMap::new()),
            staging_pools: Mutex::new(HashMap::new()),
            stats: Mutex::new(PoolStats::default()),
        }
    }

    /// Acquire a storage buffer from the pool
    ///
    /// Returns an existing buffer if available, otherwise creates a new one.
    /// Storage buffers are used for compute shader data (STORAGE | COPY_SRC | COPY_DST).
    pub fn acquire_storage(&self, size: usize) -> Arc<Buffer> {
        // WebGPU requires buffer sizes to be multiples of 4 for copy operations
        let padded_size = ((size + 3) / 4) * 4;

        let mut stats = lock_mutex(&self.stats);
        stats.total_allocations += 1;

        // Try to get from pool (using padded size for pooling key)
        let mut pools = lock_mutex(&self.storage_pools);
        if let Some(pool) = pools.get_mut(&padded_size) {
            if let Some(buffer) = pool.pop() {
                stats.pool_hits += 1;
                drop(pools);
                drop(stats);
                return buffer;
            }
        }
        drop(pools);

        // Pool miss - create new buffer with padded size
        stats.pool_misses += 1;
        drop(stats);

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pooled Storage Buffer"),
            size: padded_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Arc::new(buffer)
    }

    /// Acquire a read staging buffer from the pool (GPU → CPU)
    ///
    /// Read staging buffers are used for copying from GPU to CPU.
    /// Usage: MAP_READ | COPY_DST (WebGPU requires MAP_READ can only combine with COPY_DST)
    pub fn acquire_staging(&self, size: usize) -> Arc<Buffer> {
        let mut stats = lock_mutex(&self.stats);
        stats.total_allocations += 1;

        // Try to get from pool
        let mut pools = lock_mutex(&self.staging_pools);
        if let Some(pool) = pools.get_mut(&size) {
            if let Some(buffer) = pool.pop() {
                stats.pool_hits += 1;
                drop(pools);
                drop(stats);
                return buffer;
            }
        }
        drop(pools);

        // Pool miss - create new buffer
        stats.pool_misses += 1;
        drop(stats);

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pooled Read Staging Buffer"),
            size: size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Arc::new(buffer)
    }

    /// Acquire a write staging buffer from the pool (CPU → GPU)
    ///
    /// Write staging buffers are used for copying from CPU to GPU.
    /// Usage: COPY_DST | COPY_SRC (no mapping - uses write_buffer instead)
    pub fn acquire_staging_write(&self, size: usize) -> Arc<Buffer> {
        let mut stats = lock_mutex(&self.stats);
        stats.total_allocations += 1;

        // Try to get from pool (reuse same staging_pools for now)
        let mut pools = lock_mutex(&self.staging_pools);
        if let Some(pool) = pools.get_mut(&size) {
            if let Some(buffer) = pool.pop() {
                stats.pool_hits += 1;
                drop(pools);
                drop(stats);
                return buffer;
            }
        }
        drop(pools);

        // Pool miss - create new buffer
        stats.pool_misses += 1;
        drop(stats);

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pooled Write Staging Buffer"),
            size: size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Arc::new(buffer)
    }

    /// Return a storage buffer to the pool for reuse
    ///
    /// The buffer will be available for future `acquire_storage()` calls.
    pub fn release_storage(&self, buffer: Arc<Buffer>) {
        let size = buffer.size() as usize;

        let mut pools = lock_mutex(&self.storage_pools);
        pools.entry(size).or_default().push(buffer);

        let mut stats = lock_mutex(&self.stats);
        stats.total_returns += 1;
        stats.storage_buffers_pooled = pools.values().map(|v| v.len()).sum();
    }

    /// Return a staging buffer to the pool for reuse
    pub fn release_staging(&self, buffer: Arc<Buffer>) {
        let size = buffer.size() as usize;

        let mut pools = lock_mutex(&self.staging_pools);
        pools.entry(size).or_default().push(buffer);

        let mut stats = lock_mutex(&self.stats);
        stats.total_returns += 1;
        stats.staging_buffers_pooled = pools.values().map(|v| v.len()).sum();
    }

    /// Clear all buffers from the pool
    ///
    /// Useful for freeing GPU memory when not actively computing.
    pub fn clear(&self) {
        let mut storage_pools = lock_mutex(&self.storage_pools);
        let mut staging_pools = lock_mutex(&self.staging_pools);

        storage_pools.clear();
        staging_pools.clear();

        let mut stats = lock_mutex(&self.stats);
        stats.storage_buffers_pooled = 0;
        stats.staging_buffers_pooled = 0;
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        lock_mutex(&self.stats).clone()
    }

    /// Reset statistics counters
    pub fn reset_stats(&self) {
        let mut stats = lock_mutex(&self.stats);
        *stats = PoolStats::default();
    }

    /// Get device reference
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get queue reference
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BufferPool(allocs: {}, hits: {}, misses: {}, hit_rate: {:.1}%, pooled: {} storage + {} staging)",
            self.total_allocations,
            self.pool_hits,
            self.pool_misses,
            self.hit_rate(),
            self.storage_buffers_pooled,
            self.staging_buffers_pooled
        )
    }
}

/// Pooled buffer handle with automatic return-to-pool on drop
///
/// RAII wrapper that automatically returns buffer to pool when dropped.
pub struct PooledBuffer {
    buffer: Option<Arc<Buffer>>,
    pool: Arc<BufferPool>,
    is_staging: bool,
}

impl PooledBuffer {
    /// Create a new pooled storage buffer
    pub fn new_storage(pool: Arc<BufferPool>, size: usize) -> Self {
        let buffer = pool.acquire_storage(size);
        Self {
            buffer: Some(buffer),
            pool,
            is_staging: false,
        }
    }

    /// Create a new pooled staging buffer
    pub fn new_staging(pool: Arc<BufferPool>, size: usize) -> Self {
        let buffer = pool.acquire_staging(size);
        Self {
            buffer: Some(buffer),
            pool,
            is_staging: true,
        }
    }

    /// Get reference to underlying buffer
    pub fn buffer(&self) -> &Buffer {
        self.buffer.as_ref().unwrap()
    }

    /// Get buffer size
    pub fn size(&self) -> u64 {
        self.buffer().size()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            if self.is_staging {
                self.pool.release_staging(buffer);
            } else {
                self.pool.release_storage(buffer);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests would require a real WebGPU device
    // They are structured to show the expected behavior

    #[test]
    fn test_pool_stats_hit_rate() {
        let stats = PoolStats {
            total_allocations: 100,
            pool_hits: 80,
            pool_misses: 20,
            total_returns: 80,
            storage_buffers_pooled: 20,
            staging_buffers_pooled: 10,
        };

        assert_eq!(stats.hit_rate(), 80.0);
    }

    #[test]
    fn test_pool_stats_display() {
        let stats = PoolStats {
            total_allocations: 100,
            pool_hits: 75,
            pool_misses: 25,
            total_returns: 70,
            storage_buffers_pooled: 15,
            staging_buffers_pooled: 5,
        };

        let display = format!("{}", stats);
        assert!(display.contains("allocs: 100"));
        assert!(display.contains("hits: 75"));
        assert!(display.contains("hit_rate: 75.0%"));
    }

    #[test]
    fn test_pool_stats_zero_allocations() {
        let stats = PoolStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }
}
