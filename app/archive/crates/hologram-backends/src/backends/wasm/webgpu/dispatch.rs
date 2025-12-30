//! Hybrid dispatch logic for automatic GPU vs CPU selection
//!
//! This module implements intelligent operation dispatching based on:
//! - Operation size (number of elements)
//! - Operation type (complexity)
//! - GPU availability
//! - Performance characteristics

use std::sync::atomic::{AtomicBool, Ordering};

/// Dispatch decision for an operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchTarget {
    /// Execute on GPU (WebGPU)
    Gpu,
    /// Execute on CPU (scalar WASM)
    Cpu,
}

/// Operation type for dispatch heuristics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Binary element-wise operations (add, mul, min, max, sub, div)
    BinaryElementwise,
    /// Unary element-wise operations (abs, exp, log, sqrt, sigmoid, tanh)
    UnaryElementwise,
    /// Reduction operations (sum, min, max)
    Reduction,
    /// Complex multi-pass operations (softmax)
    MultiPass,
    /// Memory operations (copy, mark, swap)
    Memory,
}

/// Dispatch configuration with tunable thresholds
#[derive(Debug, Clone)]
pub struct DispatchConfig {
    /// Minimum size for GPU dispatch (binary operations)
    pub binary_gpu_threshold: usize,
    /// Minimum size for GPU dispatch (unary operations)
    pub unary_gpu_threshold: usize,
    /// Minimum size for GPU dispatch (reductions)
    pub reduction_gpu_threshold: usize,
    /// Minimum size for GPU dispatch (multi-pass)
    pub multipass_gpu_threshold: usize,
    /// Minimum size for GPU dispatch (memory ops)
    pub memory_gpu_threshold: usize,
    /// Whether to enable adaptive tuning
    pub adaptive_tuning: bool,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self {
            // Based on research: ~0.1ms GPU overhead vs compute time
            binary_gpu_threshold: 1024,   // 1K elements: ~20x speedup
            unary_gpu_threshold: 1024,    // 1K elements: ~18x speedup
            reduction_gpu_threshold: 512, // 512 elements: reductions benefit more
            multipass_gpu_threshold: 256, // 256 elements: higher fixed cost
            memory_gpu_threshold: 2048,   // 2K elements: memory bandwidth bound
            adaptive_tuning: false,       // Disabled by default (future work)
        }
    }
}

impl DispatchConfig {
    /// Create a conservative configuration (higher thresholds)
    pub fn conservative() -> Self {
        Self {
            binary_gpu_threshold: 2048,
            unary_gpu_threshold: 2048,
            reduction_gpu_threshold: 1024,
            multipass_gpu_threshold: 512,
            memory_gpu_threshold: 4096,
            adaptive_tuning: false,
        }
    }

    /// Create an aggressive configuration (lower thresholds)
    pub fn aggressive() -> Self {
        Self {
            binary_gpu_threshold: 512,
            unary_gpu_threshold: 512,
            reduction_gpu_threshold: 256,
            multipass_gpu_threshold: 128,
            memory_gpu_threshold: 1024,
            adaptive_tuning: false,
        }
    }
}

/// Hybrid dispatcher for GPU/CPU selection
pub struct HybridDispatcher {
    config: DispatchConfig,
    gpu_available: AtomicBool,
}

impl HybridDispatcher {
    /// Create a new dispatcher with default configuration
    pub fn new() -> Self {
        Self {
            config: DispatchConfig::default(),
            gpu_available: AtomicBool::new(false),
        }
    }

    /// Create a new dispatcher with custom configuration
    pub fn with_config(config: DispatchConfig) -> Self {
        Self {
            config,
            gpu_available: AtomicBool::new(false),
        }
    }

    /// Set GPU availability
    pub fn set_gpu_available(&self, available: bool) {
        self.gpu_available.store(available, Ordering::Relaxed);
    }

    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available.load(Ordering::Relaxed)
    }

    /// Decide where to execute an operation
    ///
    /// # Arguments
    ///
    /// * `op_type` - Type of operation
    /// * `size` - Number of elements to process
    ///
    /// # Returns
    ///
    /// Returns `DispatchTarget::Gpu` if GPU execution is beneficial,
    /// `DispatchTarget::Cpu` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hologram_backends::backends::wasm::webgpu::dispatch::{HybridDispatcher, OperationType};
    ///
    /// let dispatcher = HybridDispatcher::new();
    /// dispatcher.set_gpu_available(true);
    ///
    /// // Large operation -> GPU
    /// let target = dispatcher.dispatch(OperationType::BinaryElementwise, 10000);
    /// // Small operation -> CPU
    /// let target = dispatcher.dispatch(OperationType::BinaryElementwise, 100);
    /// ```
    pub fn dispatch(&self, op_type: OperationType, size: usize) -> DispatchTarget {
        // If GPU not available, always use CPU
        if !self.is_gpu_available() {
            return DispatchTarget::Cpu;
        }

        // Size-based dispatch based on operation type
        let threshold = match op_type {
            OperationType::BinaryElementwise => self.config.binary_gpu_threshold,
            OperationType::UnaryElementwise => self.config.unary_gpu_threshold,
            OperationType::Reduction => self.config.reduction_gpu_threshold,
            OperationType::MultiPass => self.config.multipass_gpu_threshold,
            OperationType::Memory => self.config.memory_gpu_threshold,
        };

        if size >= threshold {
            DispatchTarget::Gpu
        } else {
            DispatchTarget::Cpu
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &DispatchConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: DispatchConfig) {
        self.config = config;
    }
}

impl Default for HybridDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance statistics for adaptive tuning (future work)
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Number of GPU dispatches
    pub gpu_dispatches: usize,
    /// Number of CPU dispatches
    pub cpu_dispatches: usize,
    /// Average GPU execution time (microseconds)
    pub avg_gpu_time_us: f64,
    /// Average CPU execution time (microseconds)
    pub avg_cpu_time_us: f64,
}

impl PerformanceStats {
    /// Create empty statistics
    pub fn new() -> Self {
        Self {
            gpu_dispatches: 0,
            cpu_dispatches: 0,
            avg_gpu_time_us: 0.0,
            avg_cpu_time_us: 0.0,
        }
    }
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_without_gpu() {
        let dispatcher = HybridDispatcher::new();
        // GPU not available
        assert_eq!(
            dispatcher.dispatch(OperationType::BinaryElementwise, 10000),
            DispatchTarget::Cpu
        );
    }

    #[test]
    fn test_dispatch_with_gpu_small_size() {
        let dispatcher = HybridDispatcher::new();
        dispatcher.set_gpu_available(true);

        // Small size -> CPU (below threshold)
        assert_eq!(
            dispatcher.dispatch(OperationType::BinaryElementwise, 100),
            DispatchTarget::Cpu
        );
    }

    #[test]
    fn test_dispatch_with_gpu_large_size() {
        let dispatcher = HybridDispatcher::new();
        dispatcher.set_gpu_available(true);

        // Large size -> GPU (above threshold)
        assert_eq!(
            dispatcher.dispatch(OperationType::BinaryElementwise, 2048),
            DispatchTarget::Gpu
        );
    }

    #[test]
    fn test_reduction_lower_threshold() {
        let dispatcher = HybridDispatcher::new();
        dispatcher.set_gpu_available(true);

        // Reductions have lower threshold (512 vs 1024)
        assert_eq!(dispatcher.dispatch(OperationType::Reduction, 600), DispatchTarget::Gpu);
        assert_eq!(
            dispatcher.dispatch(OperationType::BinaryElementwise, 600),
            DispatchTarget::Cpu
        );
    }

    #[test]
    fn test_conservative_config() {
        let mut dispatcher = HybridDispatcher::new();
        dispatcher.set_gpu_available(true);
        dispatcher.set_config(DispatchConfig::conservative());

        // Conservative config has higher thresholds
        assert_eq!(
            dispatcher.dispatch(OperationType::BinaryElementwise, 1500),
            DispatchTarget::Cpu // Below 2048 threshold
        );
    }

    #[test]
    fn test_aggressive_config() {
        let mut dispatcher = HybridDispatcher::new();
        dispatcher.set_gpu_available(true);
        dispatcher.set_config(DispatchConfig::aggressive());

        // Aggressive config has lower thresholds
        assert_eq!(
            dispatcher.dispatch(OperationType::BinaryElementwise, 600),
            DispatchTarget::Gpu // Above 512 threshold
        );
    }

    #[test]
    fn test_operation_type_thresholds() {
        let dispatcher = HybridDispatcher::new();
        dispatcher.set_gpu_available(true);
        let config = dispatcher.config();

        // Verify default thresholds
        assert_eq!(config.binary_gpu_threshold, 1024);
        assert_eq!(config.unary_gpu_threshold, 1024);
        assert_eq!(config.reduction_gpu_threshold, 512);
        assert_eq!(config.multipass_gpu_threshold, 256);
        assert_eq!(config.memory_gpu_threshold, 2048);
    }
}
