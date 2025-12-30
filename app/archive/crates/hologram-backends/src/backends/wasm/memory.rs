//! Memory manager for WASM backend
//!
//! Manages buffers and pools in WebAssembly linear memory.
//! Uses DashMap for lock-free concurrent access with per-buffer locking.

use crate::backend::{BufferHandle, PoolHandle};
use crate::backends::common::memory::MemoryStorage;
use crate::error::{BackendError, Result};
use crate::pool::LinearPool;
use crate::sync::{lock_mutex, read_lock, write_lock, Mutex, RwLock};
use dashmap::DashMap;
use std::sync::Arc;

/// Memory manager for WASM backend
///
/// Uses linear memory for buffer and pool storage.
/// Provides thread-safe access via DashMap with per-buffer locking.
pub struct MemoryManager {
    /// Buffers storage (linear memory)
    /// LOCK-FREE CONCURRENT ACCESS: DashMap enables parallel get() operations
    /// PER-BUFFER LOCKING: Each buffer has its own RwLock for fine-grained parallelism
    buffers: Arc<DashMap<u64, Arc<RwLock<Vec<u8>>>>>,

    /// Pools storage (linear pools)
    /// LOCK-FREE CONCURRENT ACCESS: DashMap for thread-safe pool management
    /// PER-POOL LOCKING: Each pool has its own RwLock for fine-grained parallelism
    pools: Arc<DashMap<u64, Arc<RwLock<LinearPool>>>>,

    /// Next buffer handle ID
    next_buffer_id: Arc<Mutex<u64>>,

    /// Next pool handle ID
    next_pool_id: Arc<Mutex<u64>>,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new() -> Self {
        Self {
            buffers: Arc::new(DashMap::new()),
            pools: Arc::new(DashMap::new()),
            next_buffer_id: Arc::new(Mutex::new(1)),
            next_pool_id: Arc::new(Mutex::new(1)),
        }
    }

    /// Allocate a new buffer
    pub fn allocate_buffer(&self, size: usize) -> Result<BufferHandle> {
        let id = {
            let mut next_id = lock_mutex(&self.next_buffer_id);
            let id = *next_id;
            *next_id += 1;
            id
        };

        let buffer = vec![0u8; size];
        self.buffers.insert(id, Arc::new(RwLock::new(buffer)));

        Ok(BufferHandle(id))
    }

    /// Free a buffer
    pub fn free_buffer(&self, handle: BufferHandle) -> Result<()> {
        self.buffers
            .remove(&handle.0)
            .ok_or(BackendError::InvalidBufferHandle(handle.0))?;
        Ok(())
    }

    /// Copy data to buffer
    pub fn copy_to_buffer(&self, handle: BufferHandle, data: &[u8]) -> Result<()> {
        let buffer = self
            .buffers
            .get(&handle.0)
            .ok_or(BackendError::InvalidBufferHandle(handle.0))?;
        let mut buffer_guard = write_lock(&buffer);

        if data.len() > buffer_guard.len() {
            return Err(BackendError::BufferOutOfBounds {
                offset: 0,
                size: data.len(),
                buffer_size: buffer_guard.len(),
            });
        }

        buffer_guard[..data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Copy data from buffer
    pub fn copy_from_buffer(&self, handle: BufferHandle, data: &mut [u8]) -> Result<()> {
        let buffer = self
            .buffers
            .get(&handle.0)
            .ok_or(BackendError::InvalidBufferHandle(handle.0))?;
        let buffer_guard = read_lock(&buffer);

        if data.len() > buffer_guard.len() {
            return Err(BackendError::BufferOutOfBounds {
                offset: 0,
                size: data.len(),
                buffer_size: buffer_guard.len(),
            });
        }

        data.copy_from_slice(&buffer_guard[..data.len()]);
        Ok(())
    }

    /// Get buffer size
    pub fn buffer_size(&self, handle: BufferHandle) -> Result<usize> {
        let buffer = self
            .buffers
            .get(&handle.0)
            .ok_or(BackendError::InvalidBufferHandle(handle.0))?;
        let guard = read_lock(&buffer);
        Ok(guard.len())
    }

    /// Allocate a new pool
    pub fn allocate_pool(&self, size: usize) -> Result<PoolHandle> {
        let id = {
            let mut next_id = lock_mutex(&self.next_pool_id);
            let id = *next_id;
            *next_id += 1;
            id
        };

        let pool = LinearPool::new(size);
        self.pools.insert(id, Arc::new(RwLock::new(pool)));

        Ok(PoolHandle(id))
    }

    /// Free a pool
    pub fn free_pool(&self, handle: PoolHandle) -> Result<()> {
        self.pools
            .remove(&handle.0)
            .ok_or(BackendError::InvalidPoolHandle(handle.0))?;
        Ok(())
    }

    /// Copy data to pool
    pub fn copy_to_pool(&self, handle: PoolHandle, offset: usize, data: &[u8]) -> Result<()> {
        let pool = self
            .pools
            .get(&handle.0)
            .ok_or(BackendError::InvalidPoolHandle(handle.0))?;
        let mut pool_guard = write_lock(&pool);

        if offset + data.len() > pool_guard.capacity() {
            return Err(BackendError::PoolOutOfBounds {
                offset,
                size: data.len(),
                pool_size: pool_guard.capacity(),
            });
        }

        pool_guard.store_bytes(offset, data)?;
        Ok(())
    }

    /// Copy data from pool
    pub fn copy_from_pool(&self, handle: PoolHandle, offset: usize, data: &mut [u8]) -> Result<()> {
        let pool = self
            .pools
            .get(&handle.0)
            .ok_or(BackendError::InvalidPoolHandle(handle.0))?;
        let pool_guard = read_lock(&pool);

        if offset + data.len() > pool_guard.capacity() {
            return Err(BackendError::PoolOutOfBounds {
                offset,
                size: data.len(),
                pool_size: pool_guard.capacity(),
            });
        }

        pool_guard.load_bytes(offset, data)?;
        Ok(())
    }

    /// Get pool size
    pub fn pool_size(&self, handle: PoolHandle) -> Result<usize> {
        let pool = self
            .pools
            .get(&handle.0)
            .ok_or(BackendError::InvalidPoolHandle(handle.0))?;
        let guard = read_lock(&pool);
        Ok(guard.capacity())
    }

    /// Get raw const pointer to buffer memory (for SIMD kernels)
    ///
    /// # Safety
    ///
    /// The returned pointer is valid as long as:
    /// - The buffer handle is valid
    /// - No mutable operations occur on the buffer
    /// - The memory manager is not dropped
    #[allow(dead_code)] // Used in SIMD operations (Phase 7.3)
    pub fn buffer_as_ptr(&self, handle: BufferHandle) -> Result<*const u8> {
        let buffer = self
            .buffers
            .get(&handle.0)
            .ok_or(BackendError::InvalidBufferHandle(handle.0))?;
        let buffer_guard = read_lock(&buffer);
        Ok(buffer_guard.as_ptr())
    }

    /// Get raw mutable pointer to buffer memory (for SIMD kernels)
    ///
    /// # Safety
    ///
    /// The returned pointer is valid as long as:
    /// - The buffer handle is valid
    /// - No concurrent access occurs
    /// - The memory manager is not dropped
    #[allow(dead_code)] // Used in SIMD operations (Phase 7.3)
    pub fn buffer_as_mut_ptr(&self, handle: BufferHandle) -> Result<*mut u8> {
        let buffer = self
            .buffers
            .get(&handle.0)
            .ok_or(BackendError::InvalidBufferHandle(handle.0))?;
        let mut buffer_guard = write_lock(&buffer);
        Ok(buffer_guard.as_mut_ptr())
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

// Implement MemoryStorage trait for WASM backend
impl MemoryStorage for MemoryManager {
    fn allocate_buffer(&self, size: usize) -> Result<BufferHandle> {
        self.allocate_buffer(size)
    }

    fn free_buffer(&self, handle: BufferHandle) -> Result<()> {
        self.free_buffer(handle)
    }

    fn copy_to_buffer(&self, handle: BufferHandle, data: &[u8]) -> Result<()> {
        self.copy_to_buffer(handle, data)
    }

    fn copy_from_buffer(&self, handle: BufferHandle, data: &mut [u8]) -> Result<()> {
        self.copy_from_buffer(handle, data)
    }

    fn buffer_size(&self, handle: BufferHandle) -> Result<usize> {
        self.buffer_size(handle)
    }

    fn allocate_pool(&self, size: usize) -> Result<PoolHandle> {
        self.allocate_pool(size)
    }

    fn free_pool(&self, handle: PoolHandle) -> Result<()> {
        self.free_pool(handle)
    }

    fn copy_to_pool(&self, handle: PoolHandle, offset: usize, data: &[u8]) -> Result<()> {
        self.copy_to_pool(handle, offset, data)
    }

    fn copy_from_pool(&self, handle: PoolHandle, offset: usize, data: &mut [u8]) -> Result<()> {
        self.copy_from_pool(handle, offset, data)
    }

    fn pool_size(&self, handle: PoolHandle) -> Result<usize> {
        self.pool_size(handle)
    }

    /// Zero-copy buffer read (WASM: falls back to copy)
    fn with_buffer_read<F, R>(&self, handle: BufferHandle, f: F) -> Result<R>
    where
        F: FnOnce(&[u8]) -> Result<R>,
    {
        let size = self.buffer_size(handle)?;
        let mut data = vec![0u8; size];
        self.copy_from_buffer(handle, &mut data)?;
        f(&data)
    }

    /// Zero-copy buffer write (WASM: falls back to copy)
    fn with_buffer_write<F, R>(&self, handle: BufferHandle, f: F) -> Result<R>
    where
        F: FnOnce(&mut [u8]) -> Result<R>,
    {
        let size = self.buffer_size(handle)?;
        let mut data = vec![0u8; size];
        self.copy_from_buffer(handle, &mut data)?;
        let result = f(&mut data)?;
        self.copy_to_buffer(handle, &data)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_allocation() {
        let manager = MemoryManager::new();
        let buffer = manager.allocate_buffer(1024).unwrap();
        assert_eq!(manager.buffer_size(buffer).unwrap(), 1024);
        manager.free_buffer(buffer).unwrap();
    }

    #[test]
    fn test_buffer_copy() {
        let manager = MemoryManager::new();
        let buffer = manager.allocate_buffer(16).unwrap();

        let data = b"Hello, WASM!";
        manager.copy_to_buffer(buffer, data).unwrap();

        let mut result = vec![0u8; data.len()];
        manager.copy_from_buffer(buffer, &mut result).unwrap();

        assert_eq!(result, data);
        manager.free_buffer(buffer).unwrap();
    }

    #[test]
    fn test_pool_allocation() {
        let manager = MemoryManager::new();
        let pool = manager.allocate_pool(4096).unwrap();
        assert_eq!(manager.pool_size(pool).unwrap(), 4096);
        manager.free_pool(pool).unwrap();
    }

    #[test]
    fn test_pool_copy() {
        let manager = MemoryManager::new();
        let pool = manager.allocate_pool(1024).unwrap();

        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes = bytemuck::cast_slice(&data);
        manager.copy_to_pool(pool, 0, bytes).unwrap();

        let mut result = [0.0f32; 4];
        let result_bytes = bytemuck::cast_slice_mut(&mut result);
        manager.copy_from_pool(pool, 0, result_bytes).unwrap();

        assert_eq!(result, data);
        manager.free_pool(pool).unwrap();
    }

    #[test]
    fn test_invalid_buffer() {
        let manager = MemoryManager::new();
        let invalid = BufferHandle(999);
        assert!(manager.free_buffer(invalid).is_err());
    }

    #[test]
    fn test_invalid_pool() {
        let manager = MemoryManager::new();
        let invalid = PoolHandle(999);
        assert!(manager.free_pool(invalid).is_err());
    }
}
