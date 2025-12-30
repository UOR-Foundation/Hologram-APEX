//! Workgroup size optimization and configuration
//!
//! Provides configurable workgroup sizes for GPU compute operations.
//! Workgroup size affects occupancy, memory coalescing, and overall performance.

use std::sync::atomic::{AtomicU32, Ordering};
use wgpu::Device;

/// Workgroup configuration for compute operations
///
/// Different operations may benefit from different workgroup sizes.
/// Typical values: 64, 128, 256, 512
///
/// # Performance Considerations
///
/// - **Small workgroups** (64): Better for small data, lower occupancy
/// - **Medium workgroups** (128-256): Balanced for most operations
/// - **Large workgroups** (512+): Maximum occupancy, best for large data
///
/// # GPU Limits
///
/// WebGPU guarantees:
/// - `max_compute_workgroup_size_x >= 256`
/// - `max_compute_invocations_per_workgroup >= 256`
///
/// Actual limits vary by device. Query with `WorkgroupConfig::from_device()`.
#[derive(Debug, Clone)]
pub struct WorkgroupConfig {
    /// Workgroup size for binary operations (add, mul, min, max, sub, div)
    pub binary_ops_size: u32,
    /// Workgroup size for unary operations (abs, exp, log, sqrt, sigmoid, tanh)
    pub unary_ops_size: u32,
    /// Workgroup size for reduction operations (sum, min, max)
    pub reduction_size: u32,
    /// Workgroup size for multi-pass operations (softmax)
    pub multipass_size: u32,
    /// Workgroup size for memory operations (copy, mark, swap)
    pub memory_ops_size: u32,
}

impl Default for WorkgroupConfig {
    fn default() -> Self {
        Self {
            binary_ops_size: 256,
            unary_ops_size: 256,
            reduction_size: 256,
            multipass_size: 256,
            memory_ops_size: 256,
        }
    }
}

impl WorkgroupConfig {
    /// Create configuration from device limits
    ///
    /// Queries GPU capabilities and sets optimal workgroup sizes
    /// based on hardware limits.
    pub fn from_device(device: &Device) -> Self {
        let limits = device.limits();

        // Use maximum supported workgroup size, capped at reasonable values
        let max_size = limits.max_compute_workgroup_size_x.min(512); // Cap at 512 for memory/occupancy balance

        Self {
            binary_ops_size: max_size,
            unary_ops_size: max_size,
            reduction_size: max_size,  // Reductions benefit from larger workgroups
            multipass_size: 256,       // Multi-pass limited by shared memory
            memory_ops_size: max_size, // Memory ops benefit from coalescing
        }
    }

    /// Create a conservative configuration (smaller workgroups)
    ///
    /// Better for small operations and devices with limited GPU resources.
    pub fn conservative() -> Self {
        Self {
            binary_ops_size: 128,
            unary_ops_size: 128,
            reduction_size: 128,
            multipass_size: 128,
            memory_ops_size: 128,
        }
    }

    /// Create an aggressive configuration (larger workgroups)
    ///
    /// Maximizes occupancy for large operations on capable GPUs.
    pub fn aggressive() -> Self {
        Self {
            binary_ops_size: 512,
            unary_ops_size: 512,
            reduction_size: 512,
            multipass_size: 256, // Limited by shared memory
            memory_ops_size: 512,
        }
    }

    /// Get workgroup size for operation type
    pub fn size_for_operation(&self, op_type: WorkgroupOperation) -> u32 {
        match op_type {
            WorkgroupOperation::Binary => self.binary_ops_size,
            WorkgroupOperation::Unary => self.unary_ops_size,
            WorkgroupOperation::Reduction => self.reduction_size,
            WorkgroupOperation::MultiPass => self.multipass_size,
            WorkgroupOperation::Memory => self.memory_ops_size,
        }
    }

    /// Validate configuration against device limits
    ///
    /// Returns `Err` if any workgroup size exceeds device capabilities
    pub fn validate(&self, device: &Device) -> Result<(), String> {
        let limits = device.limits();

        let configs = [
            ("binary", self.binary_ops_size),
            ("unary", self.unary_ops_size),
            ("reduction", self.reduction_size),
            ("multipass", self.multipass_size),
            ("memory", self.memory_ops_size),
        ];

        for (name, size) in configs.iter() {
            if *size > limits.max_compute_workgroup_size_x {
                return Err(format!(
                    "{} workgroup size {} exceeds device limit {}",
                    name, size, limits.max_compute_workgroup_size_x
                ));
            }

            if *size > limits.max_compute_invocations_per_workgroup {
                return Err(format!(
                    "{} workgroup size {} exceeds max invocations {}",
                    name, size, limits.max_compute_invocations_per_workgroup
                ));
            }
        }

        Ok(())
    }
}

/// Operation type for workgroup size selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkgroupOperation {
    /// Binary element-wise operations (add, mul, min, max, sub, div)
    Binary,
    /// Unary element-wise operations (abs, exp, log, sqrt, sigmoid, tanh)
    Unary,
    /// Reduction operations (sum, min, max)
    Reduction,
    /// Multi-pass operations (softmax)
    MultiPass,
    /// Memory operations (copy, mark, swap)
    Memory,
}

/// Device limits query result
#[derive(Debug, Clone)]
pub struct DeviceLimits {
    /// Maximum workgroup size in X dimension
    pub max_workgroup_size_x: u32,
    /// Maximum workgroup size in Y dimension
    pub max_workgroup_size_y: u32,
    /// Maximum workgroup size in Z dimension
    pub max_workgroup_size_z: u32,
    /// Maximum total invocations per workgroup
    pub max_invocations_per_workgroup: u32,
    /// Maximum number of workgroups in X dimension
    pub max_workgroups_x: u32,
    /// Maximum number of workgroups in Y dimension
    pub max_workgroups_y: u32,
    /// Maximum number of workgroups in Z dimension
    pub max_workgroups_z: u32,
}

impl DeviceLimits {
    /// Query device limits from WebGPU device
    pub fn from_device(device: &Device) -> Self {
        let limits = device.limits();

        Self {
            max_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_workgroup_size_y: limits.max_compute_workgroup_size_y,
            max_workgroup_size_z: limits.max_compute_workgroup_size_z,
            max_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_workgroups_x: limits.max_compute_workgroups_per_dimension,
            max_workgroups_y: limits.max_compute_workgroups_per_dimension,
            max_workgroups_z: limits.max_compute_workgroups_per_dimension,
        }
    }

    /// Calculate optimal workgroup count for given data size
    ///
    /// Returns (workgroups_x, total_threads)
    pub fn calculate_dispatch(&self, data_size: usize, workgroup_size: u32) -> (u32, u32) {
        let num_workgroups = ((data_size as u32).div_ceil(workgroup_size)).min(self.max_workgroups_x);

        let total_threads = num_workgroups * workgroup_size;

        (num_workgroups, total_threads)
    }
}

/// Runtime workgroup size cache
///
/// Caches shader compilations for different workgroup sizes
/// to avoid recompilation overhead.
pub struct WorkgroupCache {
    /// Currently active workgroup size
    active_size: AtomicU32,
}

impl WorkgroupCache {
    /// Create new workgroup cache
    pub fn new() -> Self {
        Self {
            active_size: AtomicU32::new(256), // Default size
        }
    }

    /// Get active workgroup size
    pub fn active_size(&self) -> u32 {
        self.active_size.load(Ordering::Relaxed)
    }

    /// Set active workgroup size
    pub fn set_active_size(&self, size: u32) {
        self.active_size.store(size, Ordering::Relaxed);
    }
}

impl Default for WorkgroupCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate shader source with specific workgroup size
///
/// Takes base shader source and replaces workgroup_size parameter.
/// WGSL requires compile-time workgroup sizes, so we generate variants.
///
/// # Example
///
/// ```
/// let base_shader = "@compute @workgroup_size(WORKGROUP_SIZE)\nfn main() { }";
/// let shader_256 = generate_shader_variant(base_shader, 256);
/// assert!(shader_256.contains("@workgroup_size(256)"));
/// ```
pub fn generate_shader_variant(base_shader: &str, workgroup_size: u32) -> String {
    // Replace WORKGROUP_SIZE placeholder with actual value
    base_shader.replace("WORKGROUP_SIZE", &workgroup_size.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WorkgroupConfig::default();
        assert_eq!(config.binary_ops_size, 256);
        assert_eq!(config.unary_ops_size, 256);
        assert_eq!(config.reduction_size, 256);
    }

    #[test]
    fn test_conservative_config() {
        let config = WorkgroupConfig::conservative();
        assert_eq!(config.binary_ops_size, 128);
        assert_eq!(config.unary_ops_size, 128);
    }

    #[test]
    fn test_aggressive_config() {
        let config = WorkgroupConfig::aggressive();
        assert_eq!(config.binary_ops_size, 512);
        assert_eq!(config.reduction_size, 512);
    }

    #[test]
    fn test_operation_type_selection() {
        let config = WorkgroupConfig {
            binary_ops_size: 128,
            unary_ops_size: 256,
            reduction_size: 512,
            multipass_size: 256,
            memory_ops_size: 256,
        };

        assert_eq!(config.size_for_operation(WorkgroupOperation::Binary), 128);
        assert_eq!(config.size_for_operation(WorkgroupOperation::Unary), 256);
        assert_eq!(config.size_for_operation(WorkgroupOperation::Reduction), 512);
    }

    #[test]
    fn test_shader_variant_generation() {
        let base = "@compute @workgroup_size(WORKGROUP_SIZE)\nfn main() { }";
        let shader_128 = generate_shader_variant(base, 128);
        let shader_256 = generate_shader_variant(base, 256);

        assert!(shader_128.contains("@workgroup_size(128)"));
        assert!(shader_256.contains("@workgroup_size(256)"));
    }

    #[test]
    fn test_workgroup_cache() {
        let cache = WorkgroupCache::new();
        assert_eq!(cache.active_size(), 256); // Default

        cache.set_active_size(512);
        assert_eq!(cache.active_size(), 512);
    }
}
