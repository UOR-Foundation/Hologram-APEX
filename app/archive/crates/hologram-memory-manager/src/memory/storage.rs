//! Memory Storage Abstractions
//!
//! Provides hybrid storage strategy for chunks:
//! - **CpuShared**: Arc-based zero-copy for CPU (Rayon parallelism)
//! - **DevicePool**: Backend pool storage for GPU/WASM (zero-copy within device)
//!
//! # Architecture
//!
//! ```text
//! MemoryStorage
//!   ├─ CpuShared(Arc<[u8]>)     → Fast CPU access, Rayon parallelism
//!   └─ DevicePool {              → GPU/WASM backend pools
//!        backend: Arc<dyn Backend>,
//!        pool: PoolHandle,
//!        offset: usize,
//!        len: usize,
//!      }
//! ```
//!
//! This enables:
//! - Zero-copy within each storage type
//! - Optimal parallelism (Rayon for CPU, SIMT for GPU)
//! - O(1) space streaming through pool reuse

use hologram_backends::{Backend, PoolHandle};
use parking_lot::Mutex;
use std::sync::Arc;

/// Hybrid memory storage strategy
///
/// Provides optimal storage for different execution contexts:
/// - **CPU**: Arc-based for Rayon parallelism and zero-copy sharing
/// - **GPU/WASM**: Backend pool storage for device memory
#[derive(Clone)]
pub enum MemoryStorage {
    /// CPU-resident shared memory (Arc-based zero-copy)
    ///
    /// **Benefits**:
    /// - Zero-copy access via Arc cloning (just refcount increment)
    /// - Perfect for Rayon parallelism (par_iter over chunks)
    /// - SIMD-friendly: direct slice access for vectorization
    ///
    /// **Use when**: Data resides in CPU RAM and needs parallel processing
    CpuShared {
        /// Shared source data
        source: Arc<[u8]>,
        /// Offset within source (bytes)
        offset: usize,
        /// Length of data (bytes)
        len: usize,
    },

    /// Device pool storage (GPU/WASM/accelerator)
    ///
    /// **Benefits**:
    /// - Zero-copy within device (no CPU ↔ device transfers during execution)
    /// - O(1) space streaming: fixed pool reused for arbitrary data
    /// - Gauge-aware optimizations via backend
    ///
    /// **Use when**: Data resides in device memory (GPU VRAM, WASM linear memory)
    DevicePool {
        /// Backend managing the pool (CPU, Metal, CUDA, WASM)
        /// Wrapped in Mutex for interior mutability
        backend: Arc<Mutex<dyn Backend + Send + Sync>>,

        /// Handle to the allocated pool
        pool: PoolHandle,

        /// Offset within pool (byte offset)
        offset: usize,

        /// Length of data (bytes)
        len: usize,
    },
}

impl MemoryStorage {
    /// Create CPU-shared storage from data
    pub fn new_cpu_shared(data: Vec<u8>) -> Self {
        let len = data.len();
        Self::CpuShared {
            source: Arc::from(data.into_boxed_slice()),
            offset: 0,
            len,
        }
    }

    /// Create CPU-shared storage from Arc (entire buffer)
    pub fn from_arc(arc: Arc<[u8]>) -> Self {
        let len = arc.len();
        Self::CpuShared {
            source: arc,
            offset: 0,
            len,
        }
    }

    /// Create CPU-shared storage from Arc with offset and length
    pub fn from_arc_slice(source: Arc<[u8]>, offset: usize, len: usize) -> Self {
        debug_assert!(
            offset + len <= source.len(),
            "Slice bounds out of range: offset={}, len={}, source={}",
            offset,
            len,
            source.len()
        );
        Self::CpuShared { source, offset, len }
    }

    /// Create device pool storage
    pub fn new_device_pool(
        backend: Arc<Mutex<dyn Backend + Send + Sync>>,
        pool: PoolHandle,
        offset: usize,
        len: usize,
    ) -> Self {
        Self::DevicePool {
            backend,
            pool,
            offset,
            len,
        }
    }

    /// Check if storage is CPU-resident
    pub fn is_cpu_resident(&self) -> bool {
        matches!(self, Self::CpuShared { .. })
    }

    /// Check if storage is device-resident
    pub fn is_device_resident(&self) -> bool {
        matches!(self, Self::DevicePool { .. })
    }

    /// Get data length
    pub fn len(&self) -> usize {
        match self {
            Self::CpuShared { len, .. } => *len,
            Self::DevicePool { len, .. } => *len,
        }
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get data as slice (zero-copy for CPU, None for device)
    ///
    /// Returns `Some(&[u8])` for CPU-resident data (zero-copy Arc access).
    /// Returns `None` for device-resident data (would require copy from device).
    ///
    /// # Example
    ///
    /// ```rust
    /// use hologram_memory_manager::memory::MemoryStorage;
    ///
    /// let storage = MemoryStorage::new_cpu_shared(vec![1, 2, 3, 4]);
    /// if let Some(data) = storage.as_slice() {
    ///     // Zero-copy access for CPU data
    ///     assert_eq!(data.len(), 4);
    /// }
    /// ```
    pub fn as_slice(&self) -> Option<&[u8]> {
        match self {
            Self::CpuShared { source, offset, len } => Some(&source[*offset..*offset + *len]),
            Self::DevicePool { .. } => None, // Requires copy from device
        }
    }

    /// Copy data to Vec (always succeeds, may copy from device)
    ///
    /// For CPU-resident data, this creates a copy (not zero-copy).
    /// For device-resident data, this copies from device memory to CPU.
    ///
    /// # Errors
    ///
    /// Returns error if device copy fails.
    pub fn to_vec(&self) -> crate::Result<Vec<u8>> {
        match self {
            Self::CpuShared { source, offset, len } => Ok(source[*offset..*offset + *len].to_vec()),
            Self::DevicePool {
                backend,
                pool,
                offset,
                len,
            } => {
                let mut result = vec![0u8; *len];
                backend
                    .lock()
                    .copy_from_pool(*pool, *offset, &mut result)
                    .map_err(|e| crate::ProcessorError::BackendError(format!("{}", e)))?;
                Ok(result)
            }
        }
    }

    /// Write data to storage (for device pools)
    ///
    /// For CPU-resident storage, this is a no-op (data is immutable Arc).
    /// For device-resident storage, this writes to the device pool.
    ///
    /// # Errors
    ///
    /// Returns error if device write fails or if data size doesn't match.
    pub fn write_data(&mut self, data: &[u8]) -> crate::Result<()> {
        match self {
            Self::CpuShared { .. } => {
                // CPU storage is immutable (Arc) - would need to replace the Arc
                Err(crate::ProcessorError::InvalidOperation(
                    "Cannot write to immutable CPU-shared storage".to_string(),
                ))
            }
            Self::DevicePool {
                backend,
                pool,
                offset,
                len,
            } => {
                if data.len() != *len {
                    return Err(crate::ProcessorError::InvalidOperation(format!(
                        "Data size mismatch: expected {}, got {}",
                        len,
                        data.len()
                    )));
                }
                backend
                    .lock()
                    .copy_to_pool(*pool, *offset, data)
                    .map_err(|e| crate::ProcessorError::BackendError(format!("{}", e)))?;
                Ok(())
            }
        }
    }

    /// Get pool handle (if device-resident)
    pub fn pool_handle(&self) -> Option<PoolHandle> {
        match self {
            Self::CpuShared { .. } => None,
            Self::DevicePool { pool, .. } => Some(*pool),
        }
    }

    /// Get backend (if device-resident)
    pub fn backend(&self) -> Option<Arc<Mutex<dyn Backend + Send + Sync>>> {
        match self {
            Self::CpuShared { .. } => None,
            Self::DevicePool { backend, .. } => Some(Arc::clone(backend)),
        }
    }

    /// Get offset within pool (if device-resident)
    pub fn pool_offset(&self) -> Option<usize> {
        match self {
            Self::CpuShared { .. } => None,
            Self::DevicePool { offset, .. } => Some(*offset),
        }
    }
}

impl std::fmt::Debug for MemoryStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CpuShared {
                offset, len, source, ..
            } => write!(f, "CpuShared(offset={}, len={}, source={})", offset, len, source.len()),
            Self::DevicePool { pool, offset, len, .. } => {
                write!(f, "DevicePool(pool={}, offset={}, len={})", pool, offset, len)
            }
        }
    }
}
