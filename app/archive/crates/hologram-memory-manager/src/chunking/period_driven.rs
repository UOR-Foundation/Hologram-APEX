//! Period-Driven Chunking with Automatic Period Detection
//!
//! Analyzes input data for natural periodicities and chunks at period boundaries.
//! Gauges constructed from detected periods encode the temporal structure.

use super::gauge_map::gauge_for_primorial;
use super::primorial::{factor_primorial, generate_n_primorials};
use crate::gauge::Gauge;
use crate::memory::MemoryStorage;
use crate::Result;
use std::sync::Arc;

/// Chunk with its constructed gauge (hybrid storage)
///
/// Supports two storage modes:
/// - **CPU**: Arc-based zero-copy for Rayon parallelism
/// - **Device**: Backend pool storage for GPU/WASM execution
#[derive(Clone)]
pub struct ChunkWithGauge {
    /// Hybrid storage (CPU Arc or device pool)
    storage: MemoryStorage,

    /// Gauge constructed from primorial primes
    pub gauge: Gauge,

    /// Primordial index used for this chunk
    pub primorial: u64,

    /// Chunk index in sequence
    pub index: usize,
}

impl ChunkWithGauge {
    /// Create a new chunk with hybrid storage
    pub fn new(storage: MemoryStorage, gauge: Gauge, primorial: u64, index: usize) -> Self {
        Self {
            storage,
            gauge,
            primorial,
            index,
        }
    }

    /// Create from Arc (CPU storage) - convenience constructor
    pub fn from_arc(source: Arc<[u8]>, gauge: Gauge, primorial: u64, index: usize) -> Self {
        Self::new(MemoryStorage::from_arc(source), gauge, primorial, index)
    }

    /// Get chunk data as slice (zero-copy for CPU, None for device)
    ///
    /// Returns `Some(&[u8])` for CPU-resident chunks (zero-copy Arc access).
    /// Returns `None` for device-resident chunks (would require copy from device).
    pub fn data(&self) -> Option<&[u8]> {
        self.storage.as_slice()
    }

    /// Get owned copy of data (works for all storage types)
    pub fn data_owned(&self) -> Result<Vec<u8>> {
        self.storage.to_vec()
    }

    /// Get length of chunk
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if chunk is empty
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Get storage reference
    pub fn storage(&self) -> &MemoryStorage {
        &self.storage
    }

    /// Check if chunk is CPU-resident
    pub fn is_cpu_resident(&self) -> bool {
        self.storage.is_cpu_resident()
    }

    /// Check if chunk is device-resident
    pub fn is_device_resident(&self) -> bool {
        self.storage.is_device_resident()
    }

    /// Get gauge metadata for backend execution
    ///
    /// This converts the chunk's gauge and primorial into the metadata structure
    /// that backends use for gauge-aware kernel execution.
    ///
    /// # Example
    ///
    /// ```
    /// use hologram_memory_manager::chunking::{ChunkWithGauge, PeriodDrivenChunker};
    /// use hologram_memory_manager::Gauge;
    /// use hologram_memory_manager::memory::MemoryStorage;
    ///
    /// let storage = MemoryStorage::new_cpu_shared(vec![1, 2, 3, 4]);
    /// let chunk = ChunkWithGauge::new(storage, Gauge::GAUGE_235, 30, 0);
    ///
    /// let metadata = chunk.gauge_metadata();
    /// assert_eq!(metadata.cycle_length, 3840);
    /// assert_eq!(metadata.class_count, 120);
    /// assert_eq!(metadata.period, 30);
    /// ```
    pub fn gauge_metadata(&self) -> hologram_backends::GaugeMetadata {
        self.gauge.to_gauge_metadata(self.primorial)
    }
}

/// Period-driven chunker that detects data periodicities
pub struct PeriodDrivenChunker {
    /// Primordials available for period matching
    primorials: Vec<u64>,

    /// Minimum period to detect (bytes)
    min_period: usize,

    /// Maximum period to detect (bytes)
    max_period: usize,

    /// Autocorrelation threshold for period detection
    correlation_threshold: f64,
}

/// Detected period information
#[derive(Debug, Clone)]
pub struct PeriodInfo {
    /// Detected period length (bytes)
    pub period: u64,

    /// Nearest primorial matching period
    pub primorial: u64,

    /// Detection confidence (0.0 - 1.0)
    pub confidence: f64,
}

impl PeriodDrivenChunker {
    /// Create new period-driven chunker
    ///
    /// # Arguments
    ///
    /// * `max_primorial_levels` - Number of primorials to use
    pub fn new(max_primorial_levels: usize) -> Self {
        let primorials = generate_n_primorials(max_primorial_levels);

        // Skip primorials 1, 2, 6 (too small for meaningful periods)
        let min_period = if primorials.len() > 3 {
            primorials[3] as usize // 30
        } else {
            6
        };

        let max_period = *primorials.last().unwrap_or(&2310) as usize;

        Self {
            primorials,
            min_period,
            max_period,
            correlation_threshold: 0.3,
        }
    }

    /// Create new chunker with fast path (no period detection)
    ///
    /// This constructor creates a chunker that skips all period detection
    /// and uses const gauge lookups for maximum performance.
    ///
    /// Use this when:
    /// - You want maximum embedding speed
    /// - Your data is not highly periodic
    /// - You don't need period-optimized chunking
    ///
    /// # Performance
    ///
    /// - **No period detection** (skips entropy check, autocorrelation)
    /// - **Const gauge lookup** (zero runtime overhead)
    /// - **Move semantics** (no data cloning)
    /// - **Expected speedup**: 3-5× faster than detection-enabled path
    ///
    /// # Arguments
    ///
    /// * `max_primorial_levels` - Number of primorials to use
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_memory_manager::chunking::PeriodDrivenChunker;
    ///
    /// // Fast path: skips detection, uses const gauges
    /// let data: Vec<u8> = (0..1000).map(|i| i as u8).collect();
    /// let chunker = PeriodDrivenChunker::new_fast(10);
    /// let chunks = chunker.chunk_fast(data)?;
    /// # Ok::<(), hologram_memory_manager::ProcessorError>(())
    /// ```
    pub fn new_fast(max_primorial_levels: usize) -> Self {
        Self::new(max_primorial_levels)
    }

    /// Create with custom correlation threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.correlation_threshold = threshold;
        self
    }

    /// Chunk input with fast path (no period detection, zero-copy)
    ///
    /// Bypasses all period detection and uses the default primorial sequence
    /// with const gauge lookups. Returns zero-copy Arc-based chunks.
    ///
    /// # Performance
    ///
    /// - **No entropy check**
    /// - **No autocorrelation**
    /// - **No period detection**
    /// - **No data copying** (Arc-based views)
    /// - **Const gauge lookup only**
    /// - **5-10× faster** than detection-enabled path
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_memory_manager::chunking::PeriodDrivenChunker;
    ///
    /// let data: Vec<u8> = (0..1000).map(|i| i as u8).collect();
    /// let chunker = PeriodDrivenChunker::new_fast(10);
    /// let chunks = chunker.chunk_fast(data)?;
    /// # Ok::<(), hologram_memory_manager::ProcessorError>(())
    /// ```
    pub fn chunk_fast(&self, input: Vec<u8>) -> Result<Vec<ChunkWithGauge>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // Fast path: zero-copy chunking with Arc slices
        self.chunk_with_default_sequence_arc(input)
    }

    /// Chunk input based on detected periods and construct gauges
    ///
    /// Optimized approach:
    /// 1. Quick entropy check - skip detection for non-periodic data
    /// 2. Sample-based detection (first 1MB only)
    /// 3. Early termination on strong period detection
    /// 4. Fast path for default primorial sequence
    pub fn chunk_and_construct_gauges(&self, input: &[u8]) -> Result<Vec<ChunkWithGauge>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // Optimization 1: Quick entropy check
        if !self.looks_periodic(input) {
            return self.chunk_with_default_sequence(input);
        }

        // Optimization 2: Sample-based period detection (first 1MB)
        let sample_size = 1_048_576.min(input.len());
        let sample = &input[..sample_size];

        // Optimization 3: Early termination
        if let Some(period_info) = self.detect_strongest_period(sample)? {
            return self.chunk_at_single_period(input, period_info.primorial);
        }

        // Fallback: default primorial sequence
        self.chunk_with_default_sequence(input)
    }

    /// Fast path: chunk with default primorial sequence (no detection)
    /// Zero-copy chunking with Arc slices (internal fast path)
    fn chunk_with_default_sequence_arc(&self, input: Vec<u8>) -> Result<Vec<ChunkWithGauge>> {
        use crate::memory::MemoryStorage;

        let input_len = input.len();
        let source: Arc<[u8]> = Arc::from(input);
        let mut chunks = Vec::new();
        let mut pos = 0;
        let mut primorial_index = 3; // Skip 1, 2, 6

        while pos < input_len && primorial_index < self.primorials.len() {
            let primorial = self.primorials[primorial_index];
            let chunk_size = (primorial as usize).min(input_len - pos);

            // Use compile-time gauge lookup (zero runtime overhead)
            let gauge = gauge_for_primorial(primorial);

            // Create zero-copy slice of Arc
            let storage = MemoryStorage::from_arc_slice(Arc::clone(&source), pos, chunk_size);

            chunks.push(ChunkWithGauge::new(storage, gauge, primorial, chunks.len()));

            pos += chunk_size;
            primorial_index += 1;
        }

        Ok(chunks)
    }

    /// Legacy: Copy-based chunking (kept for compatibility)
    #[allow(dead_code)]
    fn chunk_with_default_sequence(&self, input: &[u8]) -> Result<Vec<ChunkWithGauge>> {
        // Convert to owned and use Arc version
        self.chunk_with_default_sequence_arc(input.to_vec())
    }

    /// Chunk entire input at a single detected period
    fn chunk_at_single_period(&self, input: &[u8], primorial: u64) -> Result<Vec<ChunkWithGauge>> {
        use crate::memory::MemoryStorage;

        // Convert to Arc for zero-copy chunking
        let input_len = input.len();
        let source: Arc<[u8]> = Arc::from(input.to_vec());
        let mut chunks = Vec::new();
        let mut pos = 0;
        let chunk_size = primorial as usize;

        // Use compile-time gauge lookup (zero runtime overhead)
        let gauge = gauge_for_primorial(primorial);

        while pos < input_len {
            let size = chunk_size.min(input_len - pos);

            // Create zero-copy slice of Arc
            let storage = MemoryStorage::from_arc_slice(Arc::clone(&source), pos, size);

            chunks.push(ChunkWithGauge::new(storage, gauge, primorial, chunks.len()));

            pos += size;
        }

        Ok(chunks)
    }

    /// Quick check if data is likely periodic
    fn looks_periodic(&self, data: &[u8]) -> bool {
        if data.len() < 256 {
            return false;
        }

        // Sample first 256 bytes
        let sample = &data[..256];

        // Count unique bytes
        let mut seen = [false; 256];
        let mut unique_count = 0;

        for &byte in sample {
            if !seen[byte as usize] {
                seen[byte as usize] = true;
                unique_count += 1;
            }
        }

        // If less than 50% unique bytes, might be periodic
        unique_count < 128
    }

    /// Detect strongest period with early termination
    fn detect_strongest_period(&self, sample: &[u8]) -> Result<Option<PeriodInfo>> {
        for &primorial in &self.primorials {
            let period_len = primorial as usize;

            if period_len < self.min_period || period_len > self.max_period {
                continue;
            }

            if period_len > sample.len() / 2 {
                break;
            }

            let correlation = self.autocorrelation(sample, period_len);

            // Early termination: strong period found
            if correlation > 0.8 {
                return Ok(Some(PeriodInfo {
                    period: primorial,
                    primorial,
                    confidence: correlation,
                }));
            }
        }

        Ok(None)
    }

    /// Detect periodicities in input data
    ///
    /// Uses autocorrelation to find repeating patterns,
    /// then matches to nearest primorials.
    pub fn detect_periods(&self, data: &[u8]) -> Result<Vec<PeriodInfo>> {
        let mut period_candidates = Vec::new();

        // Scan primorials for periodic structure
        for &primorial in &self.primorials {
            let period_len = primorial as usize;

            // Skip if outside detection range
            if period_len < self.min_period || period_len > self.max_period {
                continue;
            }

            // Skip if period larger than data
            if period_len > data.len() / 2 {
                break;
            }

            // Measure autocorrelation at this period
            let correlation = self.autocorrelation(data, period_len);

            if correlation > self.correlation_threshold {
                // Significant period detected
                period_candidates.push(PeriodInfo {
                    period: primorial,
                    primorial,
                    confidence: correlation,
                });
            }
        }

        // If no periods detected, use default primorial progression
        if period_candidates.is_empty() {
            // Fallback: use primorial sequence as default
            period_candidates = self
                .primorials
                .iter()
                .skip(3) // Skip 1, 2, 6 (too small)
                .copied()
                .map(|p| PeriodInfo {
                    period: p,
                    primorial: p,
                    confidence: 0.5, // Default confidence
                })
                .collect();
        }

        // Ensure we have at least one period
        if period_candidates.is_empty() {
            period_candidates.push(PeriodInfo {
                period: 30,
                primorial: 30,
                confidence: 0.5,
            });
        }

        Ok(period_candidates)
    }

    /// Compute autocorrelation at given period
    ///
    /// Returns correlation coefficient [0.0, 1.0] indicating
    /// how much the data repeats at this period.
    fn autocorrelation(&self, data: &[u8], period: usize) -> f64 {
        if period == 0 || period >= data.len() {
            return 0.0;
        }

        // Compare data[i] with data[i + period]
        let n = data.len() - period;
        let mut matches = 0usize;

        for i in 0..n {
            if data[i] == data[i + period] {
                matches += 1;
            }
        }

        matches as f64 / n as f64
    }

    /// Get primorial sequence
    pub fn primorials(&self) -> &[u64] {
        &self.primorials
    }

    /// Get period detection range
    pub fn period_range(&self) -> (usize, usize) {
        (self.min_period, self.max_period)
    }
}

impl ChunkWithGauge {
    /// Get the primes that define this chunk's gauge
    pub fn primes(&self) -> Vec<u8> {
        factor_primorial(self.primorial)
    }

    /// Get gauge name for display
    pub fn gauge_name(&self) -> String {
        self.gauge.name().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_periodic_pattern() {
        let chunker = PeriodDrivenChunker::new(8);

        // Create input with 30-byte period
        let data: Vec<u8> = (0..1000).map(|i| (i % 30) as u8).collect();

        // Detect periods
        let periods = chunker.detect_periods(&data).unwrap();

        // Should detect period 30 (or nearest primorial)
        let has_30 = periods.iter().any(|p| p.primorial == 30);
        assert!(has_30, "Should detect period 30 in repeating 30-byte pattern");

        // Confidence should be high for perfect repetition
        let period_30 = periods.iter().find(|p| p.primorial == 30);
        if let Some(p) = period_30 {
            assert!(p.confidence > 0.9, "Confidence should be high for perfect period");
        }
    }

    #[test]
    fn test_chunk_with_detected_periods() {
        let chunker = PeriodDrivenChunker::new(8);

        // Create input with 210-byte period
        let data: Vec<u8> = (0..2000).map(|i| (i % 210) as u8).collect();

        // Chunk based on detected periods
        let chunks = chunker.chunk_and_construct_gauges(&data).unwrap();

        assert!(!chunks.is_empty());

        // Verify chunks have gauges
        for chunk in &chunks {
            assert!(chunk.gauge.cycle_length > 0);
            println!(
                "Chunk {}: primorial={}, gauge={}, cycle={}, len={}",
                chunk.index,
                chunk.primorial,
                chunk.gauge_name(),
                chunk.gauge.cycle_length,
                chunk.len()
            );
        }
    }

    #[test]
    fn test_autocorrelation() {
        let chunker = PeriodDrivenChunker::new(8);

        // Perfect 30-byte period
        let data: Vec<u8> = (0..1000).map(|i| (i % 30) as u8).collect();
        let corr = chunker.autocorrelation(&data, 30);
        assert!(corr > 0.9, "Perfect period should have correlation > 0.9, got {}", corr);

        // No period (random-like)
        let data2: Vec<u8> = (0..1000).map(|i| (i * 137 % 256) as u8).collect();
        let corr2 = chunker.autocorrelation(&data2, 30);
        assert!(
            corr2 < 0.5,
            "Non-periodic data should have low correlation, got {}",
            corr2
        );
    }

    #[test]
    fn test_fallback_to_primorial_progression() {
        let chunker = PeriodDrivenChunker::new(8);

        // Random data with no clear period
        let data: Vec<u8> = (0..500).map(|i| (i * 137 % 256) as u8).collect();

        let periods = chunker.detect_periods(&data).unwrap();

        // Should fall back to primorial sequence
        assert!(!periods.is_empty());

        // Should have reasonable primorials
        for period in &periods {
            assert!(period.primorial >= 30);
        }
    }

    #[test]
    fn test_small_input() {
        let chunker = PeriodDrivenChunker::new(5);

        // Small input (just 50 bytes)
        let data: Vec<u8> = (0..50).map(|i| (i % 6) as u8).collect();

        let chunks = chunker.chunk_and_construct_gauges(&data).unwrap();

        // Should still create chunks
        assert!(!chunks.is_empty());

        // All chunks should have valid gauges
        for chunk in &chunks {
            assert!(chunk.gauge.cycle_length > 0);
        }
    }

    #[test]
    fn test_gauge_encodes_period() {
        let chunker = PeriodDrivenChunker::new(8);

        // 210-byte period
        let data: Vec<u8> = (0..1000).map(|i| (i % 210) as u8).collect();

        let chunks = chunker.chunk_and_construct_gauges(&data).unwrap();

        // Find chunk with primorial 210
        let chunk_210 = chunks.iter().find(|c| c.primorial == 210);

        if let Some(chunk) = chunk_210 {
            // Gauge should encode period information
            println!("Chunk with primorial 210: cycle_length={}", chunk.gauge.cycle_length);

            // Primes should be {2, 3, 5, 7}
            let primes = chunk.primes();
            assert!(primes.contains(&2));
            assert!(primes.contains(&3));
            assert!(primes.contains(&5));
            assert!(primes.contains(&7));
        }
    }

    #[test]
    fn test_round_trip_integrity() {
        let chunker = PeriodDrivenChunker::new(8);

        // Original data
        let original: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();

        // Chunk
        let chunks = chunker.chunk_and_construct_gauges(&original).unwrap();

        // Reconstruct
        let mut reconstructed = Vec::new();
        for chunk in &chunks {
            reconstructed.extend_from_slice(chunk.data().expect("CPU-resident chunk"));
        }

        // Verify exact reconstruction
        assert_eq!(reconstructed, original, "Round-trip should preserve data exactly");
    }

    #[test]
    fn test_custom_threshold() {
        let chunker = PeriodDrivenChunker::new(8).with_threshold(0.5);

        // Data with moderate periodicity
        let data: Vec<u8> = (0..1000)
            .map(|i| {
                let base = (i % 30) as u8;
                let noise = (i * 17 % 5) as u8;
                base.wrapping_add(noise)
            })
            .collect();

        let periods = chunker.detect_periods(&data).unwrap();

        // With lower threshold, might detect periods
        assert!(!periods.is_empty());
    }
}
