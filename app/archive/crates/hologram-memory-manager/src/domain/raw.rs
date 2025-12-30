//! Raw Domain Head
//!
//! Pass-through domain head that reconstructs the original input data.
//!
//! ## Performance
//!
//! Uses parallel block copying via Rayon for improved throughput on multi-core systems.
//! Future optimization: SIMD-accelerated buffer concatenation (Phase 4).

use super::traits::{DomainHead, Modality};
use crate::{memory::UniversalMemoryPool, Result};
use rayon::prelude::*;

/// Raw data domain head - reconstructs original input
pub struct RawDomainHead;

impl DomainHead for RawDomainHead {
    fn mediatype(&self) -> &str {
        "application/octet-stream"
    }

    fn extract(&self, pool: &UniversalMemoryPool) -> Result<Modality> {
        let total_size = pool.total_bytes();

        // Threshold: use parallel copying for large total sizes
        // For small data, sequential is faster (avoids thread spawn overhead)
        const PARALLEL_THRESHOLD: usize = 64 * 1024; // 64 KB

        if total_size < PARALLEL_THRESHOLD || pool.blocks().len() == 1 {
            // Sequential path for small data or single block
            let mut data = Vec::with_capacity(total_size);
            for block in pool.blocks() {
                data.extend_from_slice(block.data().expect("CPU-resident block required for raw modality"));
            }
            return Ok(Modality::Raw(data));
        }

        // Parallel path for large data with multiple blocks
        // Process blocks in parallel, then concatenate sequentially
        // Each block is copied independently, enabling parallel memory access
        let block_copies: Vec<Vec<u8>> = pool
            .blocks()
            .par_iter()
            .map(|block| {
                block
                    .data()
                    .expect("CPU-resident block required for raw modality")
                    .to_vec()
            })
            .collect();

        // Concatenate the parallel copies sequentially
        let mut data = Vec::with_capacity(total_size);
        for block_data in block_copies {
            data.extend_from_slice(&block_data);
        }

        Ok(Modality::Raw(data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_reconstruction() {
        let mut pool = UniversalMemoryPool::new();

        let input = b"Test data for raw reconstruction";
        pool.embed(input.to_vec(), 5).unwrap();

        // Extract raw data
        let head = RawDomainHead;
        let modality = head.extract(&pool).unwrap();

        match modality {
            Modality::Raw(data) => {
                assert_eq!(data, input);
                println!("✓ Successfully reconstructed {} bytes", data.len());
            }
            _ => panic!("Wrong modality"),
        }
    }

    #[test]
    fn test_large_input_reconstruction() {
        let mut pool = UniversalMemoryPool::new();

        let input: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let expected = input.clone(); // Save for assertion
        pool.embed(input, 7).unwrap();

        let head = RawDomainHead;
        let modality = head.extract(&pool).unwrap();

        match modality {
            Modality::Raw(data) => {
                assert_eq!(data, expected);
                assert_eq!(data.len(), 1000);
            }
            _ => panic!("Wrong modality"),
        }
    }

    #[test]
    fn test_parallel_reconstruction() {
        let mut pool = UniversalMemoryPool::new();

        // Create data larger than PARALLEL_THRESHOLD (64 KB)
        // Use 128 KB to trigger parallel path
        let input: Vec<u8> = (0..128 * 1024).map(|i| (i % 256) as u8).collect();
        let expected = input.clone();
        pool.embed(input, 10).unwrap(); // Use more levels to create multiple blocks

        let head = RawDomainHead;
        let modality = head.extract(&pool).unwrap();

        match modality {
            Modality::Raw(data) => {
                assert_eq!(data, expected);
                assert_eq!(data.len(), 128 * 1024);
                println!(
                    "✓ Successfully reconstructed {} KB using parallel path with {} blocks",
                    data.len() / 1024,
                    pool.blocks().len()
                );
            }
            _ => panic!("Wrong modality"),
        }
    }

    #[test]
    fn test_small_data_sequential_path() {
        let mut pool = UniversalMemoryPool::new();

        // Small data (less than 64 KB) should use sequential path
        let input = b"Small test data";
        pool.embed(input.to_vec(), 5).unwrap();

        let head = RawDomainHead;
        let modality = head.extract(&pool).unwrap();

        match modality {
            Modality::Raw(data) => {
                assert_eq!(&data[..], &input[..]);
            }
            _ => panic!("Wrong modality"),
        }
    }
}
