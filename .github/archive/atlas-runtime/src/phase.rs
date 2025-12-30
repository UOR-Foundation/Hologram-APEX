//! Phase counter and temporal ordering
//!
//! Implements the phase counter (mod 768) and epoch management as specified
//! in Atlas Runtime Spec §5 and §13.

use parking_lot::RwLock;
use std::sync::atomic::{AtomicU16, Ordering};

/// Phase modulus: 768 = 96 classes × 8 sub-phases
pub const PHASE_MODULUS: u16 = 768;

/// Sub-phases per class (768 / 96 = 8)
pub const SUB_PHASES_PER_CLASS: u16 = 8;

/// Pages per sub-phase (48 / 8 = 6)
pub const PAGES_PER_SUB_PHASE: u8 = 6;

/// Phase Counter (mod 768)
///
/// Provides temporal ordering for kernel launches. The phase counter advances
/// deterministically, enabling:
/// - **Temporal scheduling**: Kernels declare phase windows
/// - **Spatial locality**: Sub-phases select page blocks (6 pages each)
/// - **Reproducibility**: Deterministic execution order
///
/// # Phase Schedule Example
///
/// ```text
/// Phase 0-7:   Class 0, pages [0-5], [6-11], [12-17], ...
/// Phase 8-15:  Class 1, pages [0-5], [6-11], [12-17], ...
/// ...
/// Phase 760-767: Class 95, pages [0-5], [6-11], [12-17], ...
/// ```
///
/// This schedule ensures:
/// - Working set stays hot (6 pages = 1536 bytes per sub-phase)
/// - Prefetcher can predict patterns
/// - All pages touched exactly once per 768-phase sweep
///
/// # Example
///
/// ```
/// use atlas_runtime::PhaseCounter;
///
/// let phase = PhaseCounter::new();
/// assert_eq!(phase.get(), 0);
///
/// phase.advance(1);
/// assert_eq!(phase.get(), 1);
///
/// // Modular arithmetic
/// phase.set(767);
/// phase.advance(1);
/// assert_eq!(phase.get(), 0); // Wraps to 0
/// ```
pub struct PhaseCounter {
    /// Current phase (mod 768)
    /// Uses atomic for lock-free reads
    current: AtomicU16,

    /// Epoch counter (increments on each full 768-phase cycle)
    /// Protected by lock for less frequent access
    epoch: RwLock<u64>,
}

impl PhaseCounter {
    /// Create a new phase counter starting at phase 0
    pub fn new() -> Self {
        Self {
            current: AtomicU16::new(0),
            epoch: RwLock::new(0),
        }
    }

    /// Get current phase (lock-free)
    #[inline(always)]
    pub fn get(&self) -> u16 {
        self.current.load(Ordering::Acquire)
    }

    /// Set phase to a specific value
    ///
    /// The phase is automatically taken modulo 768.
    pub fn set(&self, phase: u16) {
        let normalized = phase % PHASE_MODULUS;
        self.current.store(normalized, Ordering::Release);
    }

    /// Advance phase by delta
    ///
    /// Returns the new phase value. If the phase wraps around 768,
    /// the epoch counter is incremented by the number of full cycles.
    ///
    /// Thread-safe using compare-and-swap loop.
    pub fn advance(&self, delta: u16) -> u16 {
        loop {
            let old = self.current.load(Ordering::Acquire);
            let new = (old + delta) % PHASE_MODULUS;

            // Try to update atomically
            match self
                .current
                .compare_exchange_weak(old, new, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    // Success - now update epoch if we wrapped
                    let cycles = (old + delta) / PHASE_MODULUS;
                    if cycles > 0 {
                        let mut epoch = self.epoch.write();
                        *epoch += cycles as u64;
                    }
                    return new;
                }
                Err(_) => {
                    // Another thread updated, retry
                    continue;
                }
            }
        }
    }

    /// Get current epoch
    ///
    /// The epoch increments each time the phase counter wraps from 767 to 0.
    /// Useful for:
    /// - Long-running profiling
    /// - Detecting phase cycles
    /// - Temporal validation
    pub fn epoch(&self) -> u64 {
        *self.epoch.read()
    }

    /// Reset to phase 0, epoch 0
    pub fn reset(&self) {
        self.current.store(0, Ordering::Release);
        *self.epoch.write() = 0;
    }

    /// Check if phase is within a window
    ///
    /// Handles modular arithmetic correctly. A window wraps if `begin + span >= 768`.
    ///
    /// # Example
    ///
    /// ```
    /// use atlas_runtime::PhaseCounter;
    ///
    /// let phase = PhaseCounter::new();
    /// phase.set(10);
    ///
    /// // Non-wrapping window
    /// assert!(phase.in_window(5, 20));
    /// assert!(!phase.in_window(20, 10));
    ///
    /// // Wrapping window: [760, 10) = [760..768) ∪ [0..10)
    /// phase.set(5);
    /// assert!(phase.in_window(760, 250)); // 760 + 250 = 1010 mod 768 = 242
    /// ```
    pub fn in_window(&self, begin: u16, span: u16) -> bool {
        let phase = self.get();
        phase_in_window(phase, begin, span)
    }

    /// Get class and sub-phase from current phase
    ///
    /// Decomposes phase into:
    /// - Class ID (phase / 8)
    /// - Sub-phase within class (phase % 8)
    ///
    /// # Example
    ///
    /// ```
    /// use atlas_runtime::PhaseCounter;
    ///
    /// let phase = PhaseCounter::new();
    /// phase.set(42);
    ///
    /// let (class, sub_phase) = phase.decompose();
    /// assert_eq!(class, 5);      // 42 / 8
    /// assert_eq!(sub_phase, 2);  // 42 % 8
    /// ```
    pub fn decompose(&self) -> (u8, u8) {
        let phase = self.get();
        decompose_phase(phase)
    }

    /// Get page block for current phase
    ///
    /// Returns the range of pages [start, end) that should be processed
    /// in the current sub-phase.
    ///
    /// # Example
    ///
    /// ```
    /// use atlas_runtime::PhaseCounter;
    ///
    /// let phase = PhaseCounter::new();
    /// phase.set(0);
    ///
    /// let (start, end) = phase.page_block();
    /// assert_eq!((start, end), (0, 6)); // Pages [0..6)
    ///
    /// phase.set(1);
    /// let (start, end) = phase.page_block();
    /// assert_eq!((start, end), (6, 12)); // Pages [6..12)
    /// ```
    pub fn page_block(&self) -> (u8, u8) {
        let (_class, sub_phase) = self.decompose();
        let start = sub_phase * PAGES_PER_SUB_PHASE;
        let end = start + PAGES_PER_SUB_PHASE;
        (start, end)
    }
}

impl Default for PhaseCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a phase is within a window (modular arithmetic)
///
/// Handles wrapping windows correctly. A window `[begin, begin+span)` wraps
/// if `begin + span >= 768`.
#[inline]
pub fn phase_in_window(phase: u16, begin: u16, span: u16) -> bool {
    let begin = begin % PHASE_MODULUS;
    let phase = phase % PHASE_MODULUS;

    if span >= PHASE_MODULUS {
        // Full window - always true
        return true;
    }

    let end = (begin + span) % PHASE_MODULUS;

    if begin < end {
        // Non-wrapping: [begin, end)
        phase >= begin && phase < end
    } else {
        // Wrapping: [begin, 768) ∪ [0, end)
        phase >= begin || phase < end
    }
}

/// Decompose phase into (class, sub_phase)
#[inline]
pub fn decompose_phase(phase: u16) -> (u8, u8) {
    let phase = phase % PHASE_MODULUS;
    let class = (phase / SUB_PHASES_PER_CLASS) as u8;
    let sub_phase = (phase % SUB_PHASES_PER_CLASS) as u8;
    (class, sub_phase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_modulus() {
        assert_eq!(PHASE_MODULUS, 768);
        assert_eq!(PHASE_MODULUS, 96 * 8);
    }

    #[test]
    fn test_phase_counter_basic() {
        let phase = PhaseCounter::new();
        assert_eq!(phase.get(), 0);

        phase.advance(1);
        assert_eq!(phase.get(), 1);

        phase.set(100);
        assert_eq!(phase.get(), 100);
    }

    #[test]
    fn test_phase_counter_wrap() {
        let phase = PhaseCounter::new();
        phase.set(767);
        assert_eq!(phase.epoch(), 0);

        phase.advance(1);
        assert_eq!(phase.get(), 0);
        assert_eq!(phase.epoch(), 1);

        phase.advance(768);
        assert_eq!(phase.get(), 0);
        assert_eq!(phase.epoch(), 2);
    }

    #[test]
    fn test_phase_counter_reset() {
        let phase = PhaseCounter::new();
        phase.advance(100);
        phase.reset();

        assert_eq!(phase.get(), 0);
        assert_eq!(phase.epoch(), 0);
    }

    #[test]
    fn test_phase_in_window_non_wrapping() {
        // Window [10, 30)
        assert!(phase_in_window(15, 10, 20));
        assert!(phase_in_window(10, 10, 20)); // Inclusive start
        assert!(!phase_in_window(30, 10, 20)); // Exclusive end
        assert!(!phase_in_window(5, 10, 20));
    }

    #[test]
    fn test_phase_in_window_wrapping() {
        // Window [760, 10) = [760..768) ∪ [0..10)
        assert!(phase_in_window(765, 760, 18));
        assert!(phase_in_window(0, 760, 18));
        assert!(phase_in_window(5, 760, 18));
        assert!(!phase_in_window(10, 760, 18));
        assert!(!phase_in_window(100, 760, 18));
    }

    #[test]
    fn test_phase_in_window_full() {
        // Full window (span >= 768)
        for phase in 0..768 {
            assert!(phase_in_window(phase, 0, 768));
            assert!(phase_in_window(phase, 100, 800));
        }
    }

    #[test]
    fn test_decompose_phase() {
        assert_eq!(decompose_phase(0), (0, 0));
        assert_eq!(decompose_phase(7), (0, 7));
        assert_eq!(decompose_phase(8), (1, 0));
        assert_eq!(decompose_phase(42), (5, 2)); // 42 / 8 = 5, 42 % 8 = 2
        assert_eq!(decompose_phase(767), (95, 7));
    }

    #[test]
    fn test_page_block() {
        let phase = PhaseCounter::new();

        // Sub-phase 0: pages [0, 6)
        phase.set(0);
        assert_eq!(phase.page_block(), (0, 6));

        // Sub-phase 1: pages [6, 12)
        phase.set(1);
        assert_eq!(phase.page_block(), (6, 12));

        // Sub-phase 7: pages [42, 48)
        phase.set(7);
        assert_eq!(phase.page_block(), (42, 48));

        // Class 1, sub-phase 0
        phase.set(8);
        assert_eq!(phase.page_block(), (0, 6));
    }

    #[test]
    fn test_phase_counter_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let phase = Arc::new(PhaseCounter::new());
        let mut handles = vec![];

        for _ in 0..4 {
            let phase_clone = phase.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    phase_clone.advance(1);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have advanced by 400 total
        assert_eq!(phase.get(), 400);
    }
}
