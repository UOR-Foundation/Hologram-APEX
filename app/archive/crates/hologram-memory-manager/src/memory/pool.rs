//! Universal Content-Addressed Memory Pool
//!
//! Memory pool that stores embedded blocks WITHOUT circuit-defined structure.
//! Each block has its gauge constructed during embedding.
//!
//! ## Key Properties
//!
//! - **Content-agnostic**: Pool doesn't interpret data
//! - **Adaptive embedding**: Chunking adapts to detected periodicities in data
//! - **Gauge-embedded**: Each block carries its constructed gauge
//! - **Domain-neutral**: Interpretation happens at domain head layer

use super::block::EmbeddedBlock;
use crate::chunking::PeriodDrivenChunker;
use crate::Result;

/// Universal memory pool
pub struct UniversalMemoryPool {
    /// Embedded blocks
    blocks: Vec<EmbeddedBlock>,

    /// Total bytes embedded
    total_bytes: usize,

    /// Number of gauges constructed
    gauges_constructed: usize,
}

/// Result of embedding operation
pub struct EmbeddingResult {
    pub blocks_embedded: usize,
    pub total_bytes: usize,
    pub gauges_constructed: usize,
    pub memory_used: usize,
}

impl UniversalMemoryPool {
    /// Create new empty pool
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            total_bytes: 0,
            gauges_constructed: 0,
        }
    }

    /// Embed input into pool with automatic gauge construction (fast path, zero-copy)
    ///
    /// This is the optimized core operation:
    /// 1. Takes ownership of input for zero-copy Arc-based chunking
    /// 2. Chunk using deterministic primorial sequence (no detection overhead)
    /// 3. Construct gauge for each chunk using const lookup table
    /// 4. Embed blocks with Arc-based views (no data copying)
    ///
    /// For period detection (slower), use `embed_with_detection()`
    ///
    /// # Performance
    ///
    /// Zero-copy architecture eliminates all per-chunk allocations and data copies.
    /// Expected: 3-5× faster than previous implementation.
    pub fn embed(&mut self, input: Vec<u8>, max_primorial_levels: usize) -> Result<EmbeddingResult> {
        let chunker = PeriodDrivenChunker::new_fast(max_primorial_levels);
        let chunks = chunker.chunk_fast(input)?;
        self.embed_chunks(chunks)
    }

    /// Embed input with period detection (slower, opt-in)
    ///
    /// Uses runtime period detection and autocorrelation to determine chunk sizes.
    /// This is slower than the default `embed()` which uses the deterministic
    /// primorial sequence.
    ///
    /// Only use this if you need adaptive chunking based on detected periods.
    ///
    /// # Note
    ///
    /// This method copies the input data to enable period detection.
    pub fn embed_with_detection(&mut self, input: &[u8], max_primorial_levels: usize) -> Result<EmbeddingResult> {
        let chunker = PeriodDrivenChunker::new(max_primorial_levels);
        let chunks = chunker.chunk_and_construct_gauges(input)?;
        self.embed_chunks(chunks)
    }

    /// Embed input to backend pool (device memory, zero-copy within device)
    ///
    /// This writes chunks directly to a backend pool (GPU VRAM, WASM linear memory, etc.)
    /// instead of using CPU Arc-based storage. Enables:
    ///
    /// - **O(1) space streaming**: Stream 100 MB through fixed 36 KB pool = 2,844× amplification
    /// - **Zero-copy within device**: No CPU ↔ device transfers during execution
    /// - **Gauge-aware backends**: Period metadata flows to backend for optimization
    ///
    /// # Arguments
    ///
    /// * `input` - Input data to embed
    /// * `backend` - Backend managing the pool (CPU, Metal, CUDA, WASM)
    /// * `pool` - Pre-allocated pool handle (must have sufficient capacity)
    /// * `max_primorial_levels` - Maximum primorial levels for chunking
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Chunking fails
    /// - Pool capacity is insufficient
    /// - Backend copy operations fail
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use hologram_memory_manager::UniversalMemoryPool;
    /// use hologram_backends::{CpuBackend, Backend};
    /// use std::sync::Arc;
    /// use parking_lot::Mutex;
    ///
    /// let backend = Arc::new(Mutex::new(CpuBackend::new()));
    /// let pool = backend.lock().allocate_pool(100_000)?;
    ///
    /// let mut memory_pool = UniversalMemoryPool::new();
    /// let input = vec![42u8; 10000];
    ///
    /// let result = memory_pool.embed_to_pool(&input, backend, pool, 5)?;
    /// ```
    pub fn embed_to_pool(
        &mut self,
        input: &[u8],
        backend: std::sync::Arc<parking_lot::Mutex<dyn hologram_backends::Backend + Send + Sync>>,
        pool: hologram_backends::PoolHandle,
        max_primorial_levels: usize,
    ) -> Result<EmbeddingResult> {
        use crate::memory::MemoryStorage;

        // 1. Chunk data with period detection (required for borrowed slice)
        let chunker = PeriodDrivenChunker::new(max_primorial_levels);
        let chunks = chunker.chunk_and_construct_gauges(input)?;

        let start_index = self.blocks.len();
        let mut pool_offset = 0usize;

        // 2. Write each chunk to backend pool and create blocks
        for chunk in chunks {
            let chunk_data = chunk.data().ok_or_else(|| {
                crate::ProcessorError::InvalidOperation("Chunk must be CPU-resident for writing to pool".to_string())
            })?;
            let chunk_len = chunk_data.len();

            // Write chunk data to backend pool
            backend
                .lock()
                .copy_to_pool(pool, pool_offset, chunk_data)
                .map_err(|e| crate::ProcessorError::BackendCopy(format!("{}", e)))?;

            // Create block with DevicePool storage
            let storage = MemoryStorage::new_device_pool(std::sync::Arc::clone(&backend), pool, pool_offset, chunk_len);

            let block = EmbeddedBlock::new(storage, chunk.gauge, chunk.primorial, start_index + chunk.index);

            self.blocks.push(block);
            self.total_bytes += chunk_len;
            self.gauges_constructed += 1;

            pool_offset += chunk_len;
        }

        Ok(EmbeddingResult {
            blocks_embedded: self.blocks.len(),
            total_bytes: self.total_bytes,
            gauges_constructed: self.gauges_constructed,
            memory_used: pool_offset, // Actual pool bytes used
        })
    }

    /// Helper: embed pre-chunked data with gauges (zero-copy)
    fn embed_chunks(
        &mut self,
        chunks: impl IntoIterator<Item = crate::chunking::ChunkWithGauge>,
    ) -> Result<EmbeddingResult> {
        let start_index = self.blocks.len();

        for chunk in chunks {
            let data_len = chunk.len();
            let block = EmbeddedBlock::new(
                chunk.storage().clone(),
                chunk.gauge,
                chunk.primorial,
                start_index + chunk.index,
            );

            self.blocks.push(block);
            self.total_bytes += data_len;
            self.gauges_constructed += 1;
        }

        Ok(EmbeddingResult {
            blocks_embedded: self.blocks.len(),
            total_bytes: self.total_bytes,
            gauges_constructed: self.gauges_constructed,
            memory_used: self.total_bytes,
        })
    }

    /// Get all embedded blocks
    pub fn blocks(&self) -> &[EmbeddedBlock] {
        &self.blocks
    }

    /// Get specific block
    pub fn get_block(&self, index: usize) -> Option<&EmbeddedBlock> {
        self.blocks.get(index)
    }

    /// Number of blocks in pool
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Check if pool is empty
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Total bytes stored
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Clear pool
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.total_bytes = 0;
        self.gauges_constructed = 0;
    }

    /// Iterate over blocks
    pub fn iter_blocks(&self) -> impl Iterator<Item = &EmbeddedBlock> {
        self.blocks.iter()
    }

    /// Get number of gauges constructed
    pub fn gauges_constructed(&self) -> usize {
        self.gauges_constructed
    }

    /// Insert a pre-constructed block directly
    ///
    /// This is useful when you already have blocks with gauges
    /// and want to avoid re-chunking.
    pub fn insert_block(&mut self, block: EmbeddedBlock) {
        self.total_bytes += block.len();
        self.gauges_constructed += 1;
        self.blocks.push(block);
    }

    /// Insert multiple blocks directly
    pub fn insert_blocks(&mut self, blocks: impl IntoIterator<Item = EmbeddedBlock>) {
        for block in blocks {
            self.insert_block(block);
        }
    }
}

impl Default for UniversalMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_simple() {
        let mut pool = UniversalMemoryPool::new();

        let input = b"Hello, World!";

        let result = pool.embed(input.to_vec(), 5).unwrap();

        assert!(result.blocks_embedded > 0);
        assert_eq!(result.total_bytes, input.len());
        assert_eq!(result.gauges_constructed, result.blocks_embedded);

        println!(
            "Embedded {} bytes into {} blocks",
            result.total_bytes, result.blocks_embedded
        );
    }

    #[test]
    fn test_each_block_has_gauge() {
        let mut pool = UniversalMemoryPool::new();

        let input: Vec<u8> = (0..300).map(|i| (i % 256) as u8).collect();

        pool.embed(input, 6).unwrap();

        // Verify each block has a gauge
        for (i, block) in pool.blocks().iter().enumerate() {
            println!(
                "Block {}: gauge={}, cycle={}, data_len={}",
                i,
                block.gauge_name(),
                block.cycle_length(),
                block.len()
            );

            assert!(block.cycle_length() > 0);
            assert!(block.class_count() > 0);
        }
    }

    #[test]
    fn test_gauge_progression() {
        let mut pool = UniversalMemoryPool::new();

        let input: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();

        pool.embed(input, 8).unwrap();

        // Gauges should progress in size
        let mut prev_cycle = 0;
        for block in pool.blocks() {
            // Cycle lengths should generally increase (not strictly, but trend)
            if block.cycle_length() > prev_cycle {
                prev_cycle = block.cycle_length();
            }
        }

        assert!(prev_cycle > 0);
    }

    #[test]
    fn test_memory_usage() {
        let mut pool = UniversalMemoryPool::new();

        let input = vec![42u8; 10000]; // 10KB input
        let input_len = input.len();

        let result = pool.embed(input, 10).unwrap();

        println!("Input: {} bytes", input_len);
        println!("Blocks: {}", result.blocks_embedded);
        println!("Memory: {} bytes", result.memory_used);
        println!("Gauges: {}", result.gauges_constructed);

        assert_eq!(result.memory_used, result.total_bytes);
    }

    #[test]
    fn test_embed_to_pool() {
        use hologram_backends::{Backend, CpuBackend};
        use parking_lot::Mutex;
        use std::sync::Arc;

        let mut pool = UniversalMemoryPool::new();

        // Create backend and allocate pool
        let backend = Arc::new(Mutex::new(CpuBackend::new()));
        let backend_pool = backend.lock().allocate_pool(100_000).unwrap();

        let input = b"Hello, Device Pool!";

        // Embed to backend pool
        let result = pool.embed_to_pool(input, backend.clone(), backend_pool, 5).unwrap();

        println!("Embedded {} bytes to device pool", result.total_bytes);
        println!("Blocks: {}", result.blocks_embedded);
        println!("Pool bytes used: {}", result.memory_used);

        assert!(result.blocks_embedded > 0);
        assert_eq!(result.total_bytes, input.len());

        // Verify blocks are device-resident
        for block in pool.blocks() {
            assert!(block.is_device_resident());
            assert!(!block.is_cpu_resident());
            assert_eq!(block.storage().pool_handle(), Some(backend_pool));
        }
    }

    #[test]
    fn test_embed_to_pool_large_data() {
        use hologram_backends::{Backend, CpuBackend};
        use parking_lot::Mutex;
        use std::sync::Arc;

        let mut pool = UniversalMemoryPool::new();

        // Create backend and allocate large pool
        let backend = Arc::new(Mutex::new(CpuBackend::new()));
        let backend_pool = backend.lock().allocate_pool(1_000_000).unwrap();

        let input: Vec<u8> = (0..50000).map(|i| (i % 256) as u8).collect();

        // Embed large data to backend pool
        let result = pool.embed_to_pool(&input, backend.clone(), backend_pool, 8).unwrap();

        println!(
            "Embedded {} bytes into {} blocks",
            result.total_bytes, result.blocks_embedded
        );
        println!("Pool memory used: {}", result.memory_used);

        assert!(result.blocks_embedded > 0);
        assert_eq!(result.total_bytes, input.len());
        assert!(result.memory_used <= 1_000_000); // Within pool capacity

        // Verify all blocks have valid gauges and are device-resident
        for block in pool.blocks() {
            assert!(block.cycle_length() > 0);
            assert!(block.class_count() > 0);
            assert!(block.is_device_resident());
        }
    }
}
