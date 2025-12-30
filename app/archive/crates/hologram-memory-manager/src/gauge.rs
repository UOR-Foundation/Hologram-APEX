//! Gauge System for Processor (Vendored from quantum-768)
//!
//! This module provides the gauge type used for primorial chunking.
//! Vendored from the quantum-768 crate to avoid dependency on the full quantum computing stack.
//!
//! ## Gauge Theory
//!
//! A gauge is defined by a set of prime factors P, which determines:
//! - **Cycle Length**: L = (∏ p∈P\{2} p) × 256
//! - **Class Count**: R = projector-determined resonance classes
//! - **Compatible Periods**: Periods r where L % r == 0
//!
//! ### Standard Gauges
//!
//! | Gauge | Primes | Cycle (L) | Classes (R) | Supports Periods |
//! |-------|--------|-----------|-------------|------------------|
//! | {2,3} | [2,3] | 768 | 96 | {1,2,3,4,6,8,12,16,...} |
//! | {2,3,5} | [2,3,5] | 3,840 | 120 | + {5,10,15,20,...} |
//! | {2,3,5,7} | [2,3,5,7] | 26,880 | 168 | + {7,14,21,28,...} |
//! | {2,3,5,7,11} | [2,3,5,7,11] | 295,680 | 264 | + {11,22,33,44,...} |

/// Gauge configuration for geometric period encoding
///
/// A gauge defines the cycle length, class count, and supported prime factors
/// for encoding periodicities in the processor's chunking system.
///
/// # Memory Model
///
/// **Logical vs Physical Addressing**:
/// - `cycle_length`: Logical address space for modular arithmetic
/// - `class_count`: Number of canonical classes for measurement projection
/// - `primes`: Prime factors that determine compatible periods
///
/// For processor use, gauges are selected based on detected periods in input data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gauge {
    /// Cycle length: logical address space for position encoding
    pub cycle_length: u64,

    /// Class count: number of resonance classes
    pub class_count: u16,

    /// Prime factors included in this gauge (excluding 2 from cycle calculation)
    pub primes: &'static [u8],
}

impl Gauge {
    /// Get human-readable name for this gauge
    ///
    /// # Example
    ///
    /// ```
    /// use hologram_memory_manager::Gauge;
    ///
    /// assert_eq!(Gauge::GAUGE_23.name(), "Gauge{2,3}");
    /// assert_eq!(Gauge::GAUGE_235.name(), "Gauge{2,3,5}");
    /// ```
    pub fn name(&self) -> &'static str {
        match (self.cycle_length, self.class_count) {
            (768, 96) => "Gauge{2,3}",
            (3840, 120) => "Gauge{2,3,5}",
            (26880, 168) => "Gauge{2,3,5,7}",
            (295680, 264) => "Gauge{2,3,5,7,11}",
            _ => "CustomGauge",
        }
    }

    /// Convert to GaugeMetadata for backend execution
    ///
    /// This creates the metadata structure that backends use to optimize
    /// gauge-aware operations during kernel execution.
    ///
    /// # Arguments
    ///
    /// * `period` - The detected period (primorial) for this chunk (e.g., 30, 210, 2310)
    ///
    /// # Example
    ///
    /// ```
    /// use hologram_memory_manager::Gauge;
    ///
    /// let gauge = Gauge::GAUGE_235;
    /// let metadata = gauge.to_gauge_metadata(30);  // primorial = 2×3×5 = 30
    ///
    /// assert_eq!(metadata.cycle_length, 3840);
    /// assert_eq!(metadata.class_count, 120);
    /// assert_eq!(metadata.period, 30);
    /// ```
    pub fn to_gauge_metadata(&self, period: u64) -> hologram_backends::GaugeMetadata {
        hologram_backends::GaugeMetadata::new(self.cycle_length, self.class_count, period)
    }

    /// Standard gauge {2,3}: 768-cycle, 96-classes
    ///
    /// **Cycle**: 768 = 3 × 256 = 2^8 × 3
    /// **Classes**: 96 = 32 × 3
    /// **Supports**: Periods with prime factors from {2, 3}
    ///
    /// Compatible periods: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768
    pub const GAUGE_23: Gauge = Gauge {
        cycle_length: 768,
        class_count: 96,
        primes: &[2, 3],
    };

    /// Extended gauge {2,3,5}: 3,840-cycle, 120-classes
    ///
    /// **Cycle**: 3,840 = 15 × 256 = 2^8 × 3 × 5
    /// **Classes**: 120 = 40 × 3
    /// **Supports**: Periods with prime factors from {2, 3, 5}
    ///
    /// Adds support for: 5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 640, 960, ...
    pub const GAUGE_235: Gauge = Gauge {
        cycle_length: 3840,
        class_count: 120,
        primes: &[2, 3, 5],
    };

    /// Extended gauge {2,3,5,7}: 26,880-cycle, 168-classes
    ///
    /// **Cycle**: 26,880 = 105 × 256 = 2^8 × 3 × 5 × 7
    /// **Classes**: 168 = 56 × 3
    /// **Supports**: Periods with prime factors from {2, 3, 5, 7}
    ///
    /// Adds support for: 7, 14, 21, 28, 35, 42, 56, 70, 84, 105, 112, 140, 168, 210, ...
    pub const GAUGE_2357: Gauge = Gauge {
        cycle_length: 26880,
        class_count: 168,
        primes: &[2, 3, 5, 7],
    };

    /// Extended gauge {2,3,5,7,11}: 295,680-cycle, 264-classes
    ///
    /// **Cycle**: 295,680 = 1155 × 256 = 2^8 × 3 × 5 × 7 × 11
    /// **Classes**: 264 = 88 × 3
    /// **Supports**: Periods with prime factors from {2, 3, 5, 7, 11}
    ///
    /// Adds support for: 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 132, 154, 165, 176, ...
    pub const GAUGE_235711: Gauge = Gauge {
        cycle_length: 295680,
        class_count: 264,
        primes: &[2, 3, 5, 7, 11],
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauge_constants() {
        // Verify gauge definitions
        assert_eq!(Gauge::GAUGE_23.cycle_length, 768);
        assert_eq!(Gauge::GAUGE_23.class_count, 96);
        assert_eq!(Gauge::GAUGE_23.primes, &[2, 3]);

        assert_eq!(Gauge::GAUGE_235.cycle_length, 3840);
        assert_eq!(Gauge::GAUGE_235.class_count, 120);
        assert_eq!(Gauge::GAUGE_235.primes, &[2, 3, 5]);

        assert_eq!(Gauge::GAUGE_2357.cycle_length, 26880);
        assert_eq!(Gauge::GAUGE_2357.class_count, 168);
        assert_eq!(Gauge::GAUGE_2357.primes, &[2, 3, 5, 7]);

        assert_eq!(Gauge::GAUGE_235711.cycle_length, 295680);
        assert_eq!(Gauge::GAUGE_235711.class_count, 264);
        assert_eq!(Gauge::GAUGE_235711.primes, &[2, 3, 5, 7, 11]);
    }

    #[test]
    fn test_gauge_equality() {
        assert_eq!(Gauge::GAUGE_23, Gauge::GAUGE_23);
        assert_ne!(Gauge::GAUGE_23, Gauge::GAUGE_235);
    }

    #[test]
    fn test_gauge_copy() {
        let g1 = Gauge::GAUGE_23;
        let g2 = g1; // Should copy, not move
        assert_eq!(g1, g2);
    }

    #[test]
    fn test_to_gauge_metadata() {
        // Test conversion for each standard gauge
        let metadata_23 = Gauge::GAUGE_23.to_gauge_metadata(6);
        assert_eq!(metadata_23.cycle_length, 768);
        assert_eq!(metadata_23.class_count, 96);
        assert_eq!(metadata_23.period, 6);

        let metadata_235 = Gauge::GAUGE_235.to_gauge_metadata(30);
        assert_eq!(metadata_235.cycle_length, 3840);
        assert_eq!(metadata_235.class_count, 120);
        assert_eq!(metadata_235.period, 30);

        let metadata_2357 = Gauge::GAUGE_2357.to_gauge_metadata(210);
        assert_eq!(metadata_2357.cycle_length, 26880);
        assert_eq!(metadata_2357.class_count, 168);
        assert_eq!(metadata_2357.period, 210);

        let metadata_235711 = Gauge::GAUGE_235711.to_gauge_metadata(2310);
        assert_eq!(metadata_235711.cycle_length, 295680);
        assert_eq!(metadata_235711.class_count, 264);
        assert_eq!(metadata_235711.period, 2310);
    }
}
