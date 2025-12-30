//! Embedded Block Type
//!
//! Represents a block of data embedded in memory pool with its constructed gauge.
//! Supports hybrid storage: CPU Arc (zero-copy) or backend pools (device memory).

use crate::gauge::Gauge;
use crate::memory::MemoryStorage;

/// Embedded block with constructed gauge (hybrid storage)
///
/// Supports two storage modes:
/// - **CPU**: Arc-based zero-copy for Rayon parallelism
/// - **Device**: Backend pool storage for GPU/WASM execution
#[derive(Clone)]
pub struct EmbeddedBlock {
    /// Hybrid storage (CPU Arc or device pool)
    storage: MemoryStorage,

    /// Gauge constructed during embedding
    pub gauge: Gauge,

    /// Primordial used for this block
    pub primorial: u64,

    /// Block index in sequence
    pub index: usize,
}

impl EmbeddedBlock {
    /// Create a new block with hybrid storage
    pub fn new(storage: MemoryStorage, gauge: Gauge, primorial: u64, index: usize) -> Self {
        Self {
            storage,
            gauge,
            primorial,
            index,
        }
    }

    /// Create from Arc (CPU storage) - convenience constructor
    pub fn from_arc(source: std::sync::Arc<[u8]>, gauge: Gauge, primorial: u64, index: usize) -> Self {
        Self::new(MemoryStorage::from_arc(source), gauge, primorial, index)
    }

    /// Get block data as slice (zero-copy for CPU, None for device)
    ///
    /// Returns `Some(&[u8])` for CPU-resident blocks (zero-copy Arc access).
    /// Returns `None` for device-resident blocks (would require copy from device).
    pub fn data(&self) -> Option<&[u8]> {
        self.storage.as_slice()
    }

    /// Get owned copy of data (works for all storage types)
    pub fn data_owned(&self) -> crate::Result<Vec<u8>> {
        self.storage.to_vec()
    }

    /// Get length of block
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if block is empty
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Get storage reference
    pub fn storage(&self) -> &MemoryStorage {
        &self.storage
    }

    /// Check if block is CPU-resident
    pub fn is_cpu_resident(&self) -> bool {
        self.storage.is_cpu_resident()
    }

    /// Check if block is device-resident
    pub fn is_device_resident(&self) -> bool {
        self.storage.is_device_resident()
    }

    /// Get gauge name
    pub fn gauge_name(&self) -> String {
        self.gauge.name().to_string()
    }

    /// Get gauge cycle length
    pub fn cycle_length(&self) -> u64 {
        self.gauge.cycle_length
    }

    /// Get gauge class count
    pub fn class_count(&self) -> u16 {
        self.gauge.class_count
    }

    /// Get gauge metadata for backend execution
    ///
    /// This converts the block's gauge and primorial into the metadata structure
    /// that backends use for gauge-aware kernel execution.
    ///
    /// # Example
    ///
    /// ```
    /// use hologram_memory_manager::{EmbeddedBlock, Gauge};
    /// use hologram_memory_manager::memory::MemoryStorage;
    ///
    /// let storage = MemoryStorage::new_cpu_shared(vec![1, 2, 3, 4]);
    /// let block = EmbeddedBlock::new(storage, Gauge::GAUGE_235, 30, 0);
    ///
    /// let metadata = block.gauge_metadata();
    /// assert_eq!(metadata.cycle_length, 3840);
    /// assert_eq!(metadata.class_count, 120);
    /// assert_eq!(metadata.period, 30);
    /// ```
    pub fn gauge_metadata(&self) -> hologram_backends::GaugeMetadata {
        self.gauge.to_gauge_metadata(self.primorial)
    }
}
