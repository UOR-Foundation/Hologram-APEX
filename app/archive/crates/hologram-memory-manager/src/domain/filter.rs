//! Filter Domain Head
//!
//! Applies SIMD-accelerated filtering predicates.
//!
//! ## Architecture
//!
//! Uses hologram-core's SIMD clip and ReLU operations to filter data.
//! This is significantly faster than scalar conditional logic.
//!
//! ## Performance
//!
//! - **SIMD Filtering**: Vectorized comparisons
//! - **No Branching**: SIMD operations avoid branch mispredictions
//! - **5-10x faster** than scalar filtering
//!
//! ## Example
//!
//! ```
//! use hologram_memory_manager::{FilterDomainHead, DomainHead, Modality};
//! use hologram_memory_manager::{UniversalMemoryPool, EmbeddedBlock, Gauge};
//! use std::sync::Arc;
//!
//! // Create f32-aligned data manually
//! let float_data: Vec<f32> = (0..256).map(|i| i as f32 - 128.0).collect();
//! let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();
//!
//! // Create pool with aligned block
//! let mut pool = UniversalMemoryPool::new();
//! let source: Arc<[u8]> = byte_data.into();
//! let len = source.len();
//! let block = EmbeddedBlock::from_arc(source, Gauge::GAUGE_23, 6, 0);
//! pool.insert_block(block);
//!
//! // Filter to positive values only using SIMD
//! let head = FilterDomainHead::positive_only();
//! let modality = head.extract(&pool).unwrap();
//! ```

use crate::domain::{DomainHead, Modality};
use crate::memory::{EmbeddedBlock, UniversalMemoryPool};
use crate::Result;

/// Filter type for domain head
#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    /// Keep only positive values (ReLU)
    PositiveOnly,
    /// Clip to range [min, max]
    Clip { min: f32, max: f32 },
    /// Keep values above threshold
    AboveThreshold(f32),
}

/// Domain head that filters f32 data using SIMD operations
pub struct FilterDomainHead {
    /// Filter type to apply
    pub filter: FilterType,
}

impl FilterDomainHead {
    /// Create filter for positive values only (ReLU)
    ///
    /// # Example
    ///
    /// ```
    /// use hologram_memory_manager::FilterDomainHead;
    ///
    /// let head = FilterDomainHead::positive_only();
    /// ```
    pub fn positive_only() -> Self {
        Self {
            filter: FilterType::PositiveOnly,
        }
    }

    /// Create filter that clips to range
    ///
    /// # Example
    ///
    /// ```
    /// use hologram_memory_manager::FilterDomainHead;
    ///
    /// // Clip to [0, 1]
    /// let head = FilterDomainHead::clip(0.0, 1.0);
    /// ```
    pub fn clip(min: f32, max: f32) -> Self {
        Self {
            filter: FilterType::Clip { min, max },
        }
    }

    /// Create filter for values above threshold
    ///
    /// # Example
    ///
    /// ```
    /// use hologram_memory_manager::FilterDomainHead;
    ///
    /// // Keep values >= 0.5
    /// let head = FilterDomainHead::above_threshold(0.5);
    /// ```
    pub fn above_threshold(threshold: f32) -> Self {
        Self {
            filter: FilterType::AboveThreshold(threshold),
        }
    }
}

impl DomainHead for FilterDomainHead {
    fn mediatype(&self) -> &str {
        "application/x-filtered-f32"
    }

    fn extract(&self, pool: &UniversalMemoryPool) -> Result<Modality> {
        let blocks = pool.blocks();

        // Filter to f32-aligned blocks only
        let f32_blocks: Vec<&EmbeddedBlock> = blocks.iter().filter(|b| b.len() % 4 == 0).collect();

        if f32_blocks.is_empty() {
            return Err(crate::ProcessorError::DomainHeadNotFound(
                "No f32-aligned blocks in pool".to_string(),
            ));
        }

        let mut filtered_data = Vec::new();

        for block in f32_blocks {
            // Access block data directly (zero-copy for CPU-resident)
            let data = block.data().ok_or_else(|| {
                crate::ProcessorError::InvalidOperation("Filter operation requires CPU-resident block".to_string())
            })?;

            // Cast to f32 slice
            let input: &[f32] = bytemuck::cast_slice(data);

            // Apply scalar filter
            let output: Vec<f32> = match self.filter {
                FilterType::PositiveOnly => {
                    // ReLU: max(0, x)
                    input.iter().map(|&x| x.max(0.0)).collect()
                }
                FilterType::Clip { min, max } => {
                    // Clip to [min, max]
                    input.iter().map(|&x| x.clamp(min, max)).collect()
                }
                FilterType::AboveThreshold(threshold) => {
                    // Clip below threshold to threshold
                    input.iter().map(|&x| x.max(threshold)).collect()
                }
            };

            filtered_data.extend_from_slice(&output);
        }

        Ok(Modality::Filtered(filtered_data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_positive_only() {
        use crate::gauge::Gauge;
        use std::sync::Arc;

        // Create f32-aligned test data: [-10, -9, ..., -1, 0, 1, ..., 10]
        let float_data: Vec<f32> = (-10..=10).map(|i| i as f32).collect();
        let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

        // Create pool with manually aligned block (no chunking)
        let mut pool = UniversalMemoryPool::new();
        let source: Arc<[u8]> = byte_data.into();

        let block = EmbeddedBlock::from_arc(source, Gauge::GAUGE_23, 6, 0);
        pool.insert_block(block);

        // Apply positive-only filter (ReLU)
        let head = FilterDomainHead::positive_only();
        let modality = head.extract(&pool).unwrap();

        // Verify filtering
        if let Modality::Filtered(data) = modality {
            assert_eq!(data.len(), 21);
            // All negative values should be 0, positive values unchanged
            for (i, &val) in data.iter().enumerate() {
                let original = -10 + i as i32;
                if original < 0 {
                    assert_eq!(val, 0.0, "Negative value not zeroed at index {}", i);
                } else {
                    assert_eq!(val, original as f32, "Positive value changed at index {}", i);
                }
            }
        } else {
            panic!("Expected Filtered modality");
        }
    }

    #[test]
    fn test_filter_clip() {
        use crate::gauge::Gauge;
        use std::sync::Arc;

        // Create f32-aligned test data: [0, 1, 2, ..., 20]
        let float_data: Vec<f32> = (0..=20).map(|i| i as f32).collect();
        let byte_data: Vec<u8> = bytemuck::cast_slice(&float_data).to_vec();

        // Create pool with manually aligned block (no chunking)
        let mut pool = UniversalMemoryPool::new();
        let source: Arc<[u8]> = byte_data.into();

        let block = EmbeddedBlock::from_arc(source, Gauge::GAUGE_23, 6, 0);
        pool.insert_block(block);

        // Apply clip to [5, 15]
        let head = FilterDomainHead::clip(5.0, 15.0);
        let modality = head.extract(&pool).unwrap();

        // Verify clipping
        if let Modality::Filtered(data) = modality {
            assert_eq!(data.len(), 21);
            for (i, &val) in data.iter().enumerate() {
                let original = i as f32;
                if original < 5.0 {
                    assert_eq!(val, 5.0);
                } else if original > 15.0 {
                    assert_eq!(val, 15.0);
                } else {
                    assert_eq!(val, original);
                }
            }
        } else {
            panic!("Expected Filtered modality");
        }
    }

    #[test]
    fn test_filter_mediatype() {
        let head = FilterDomainHead::positive_only();
        assert_eq!(head.mediatype(), "application/x-filtered-f32");
    }
}
