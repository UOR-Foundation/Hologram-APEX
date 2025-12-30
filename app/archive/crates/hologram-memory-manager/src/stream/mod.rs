//! Stream Processing Module
//!
//! Provides high-level stream processing API with automatic chunking and gauge construction.
//!
//! ## Usage
//!
//! ```
//! use hologram_memory_manager::Stream;
//!
//! let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
//! let stream = Stream::new(data);
//! let chunked = stream.chunk(7).unwrap();
//! let context = chunked.embed().unwrap();
//! ```

use crate::{
    chunking::{ChunkWithGauge, PeriodDrivenChunker},
    memory::UniversalMemoryPool,
    Result,
};

/// Stream processing context
///
/// Contains the embedded memory pool and metadata about the embedding.
pub struct StreamContext {
    /// Universal memory pool with embedded blocks
    pub pool: UniversalMemoryPool,

    /// Number of gauges constructed
    pub gauges_count: usize,

    /// Total bytes embedded
    pub total_bytes: usize,
}

/// Lazy stream interface
///
/// Provides chainable operations on data before chunking.
pub struct Stream<T> {
    data: Vec<T>,
}

impl<T> Stream<T> {
    /// Create new stream from data
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Chunk stream with period-driven detection (requires T: bytemuck::Pod)
    pub fn chunk(self, levels: usize) -> Result<ChunkedStream<T>>
    where
        T: bytemuck::Pod,
    {
        // Convert to bytes
        let bytes = bytemuck::cast_slice::<T, u8>(&self.data).to_vec();

        // Create chunker
        let chunker = PeriodDrivenChunker::new(levels);

        // Chunk with automatic gauge construction
        let chunks = chunker.chunk_and_construct_gauges(&bytes)?;

        Ok(ChunkedStream {
            chunks,
            original_len: self.data.len(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Map operation
    pub fn map<U, F>(self, f: F) -> Stream<U>
    where
        F: Fn(T) -> U,
    {
        Stream {
            data: self.data.into_iter().map(f).collect(),
        }
    }

    /// Filter operation
    pub fn filter<F>(self, f: F) -> Stream<T>
    where
        F: Fn(&T) -> bool,
    {
        Stream {
            data: self.data.into_iter().filter(f).collect(),
        }
    }

    /// Get data length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Chunked stream with automatic gauge construction
///
/// Represents data that has been chunked via primorial sequence
/// with gauges automatically constructed.
pub struct ChunkedStream<T> {
    chunks: Vec<ChunkWithGauge>,
    original_len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ChunkedStream<T> {
    /// Embed chunks into universal memory pool
    ///
    /// This efficiently inserts pre-chunked blocks with their gauges
    /// directly into the pool without re-processing.
    pub fn embed(self) -> Result<StreamContext> {
        let mut pool = UniversalMemoryPool::new();

        let mut total_bytes = 0;

        // Insert blocks directly (more efficient than re-chunking)
        for (index, chunk) in self.chunks.into_iter().enumerate() {
            total_bytes += chunk.len();

            let block = crate::memory::EmbeddedBlock::new(chunk.storage().clone(), chunk.gauge, chunk.primorial, index);

            pool.insert_block(block);
        }

        Ok(StreamContext {
            gauges_count: pool.gauges_constructed(),
            total_bytes,
            pool,
        })
    }

    /// Get chunks
    pub fn chunks(&self) -> &[ChunkWithGauge] {
        &self.chunks
    }

    /// Number of chunks
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Get original data length before chunking
    pub fn original_len(&self) -> usize {
        self.original_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_creation() {
        let data: Vec<u8> = (0..100).collect();
        let stream = Stream::new(data);
        assert_eq!(stream.len(), 100);
        assert!(!stream.is_empty());
    }

    #[test]
    fn test_stream_map() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5];
        let stream = Stream::new(data);
        let mapped = stream.map(|x| x * 2);
        assert_eq!(mapped.data, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_stream_filter() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let stream = Stream::new(data);
        let filtered = stream.filter(|&x| x % 2 == 0);
        assert_eq!(filtered.data, vec![2, 4, 6]);
    }

    #[test]
    fn test_stream_chunking() {
        let data: Vec<u8> = (0..200).collect();
        let stream = Stream::new(data);
        let chunked = stream.chunk(5).unwrap();

        assert!(!chunked.is_empty());
    }

    #[test]
    fn test_chunked_stream_embedding() {
        let data: Vec<u8> = (0..100).collect();
        let stream = Stream::new(data);
        let chunked = stream.chunk(6).unwrap();
        let context = chunked.embed().unwrap();

        assert_eq!(context.total_bytes, 100);
        assert!(context.gauges_count > 0);
        assert!(!context.pool.is_empty());
    }
}
