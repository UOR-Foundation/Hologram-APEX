//! Kernel launch configuration and parameters
//!
//! This module provides launch descriptors and parameter types for kernel execution.
//! Kernel compilation, caching, and execution are handled by backend implementations.

/// Launch configuration for kernel execution
#[derive(Debug, Clone, Copy)]
pub struct LaunchDesc {
    /// Grid dimensions (number of blocks)
    pub grid: (u32, u32, u32),

    /// Block dimensions (threads per block)
    pub block: (u32, u32, u32),

    /// Total number of threads
    pub total_threads: u32,
}

impl LaunchDesc {
    /// Create a 1D launch configuration
    pub fn one_dimensional(n: u32) -> Self {
        Self {
            grid: (1, 1, 1),
            block: (n, 1, 1),
            total_threads: n,
        }
    }

    /// Create a 2D launch configuration
    pub fn two_dimensional(width: u32, height: u32) -> Self {
        Self {
            grid: (1, 1, 1),
            block: (width, height, 1),
            total_threads: width * height,
        }
    }

    /// Create a 3D launch configuration
    pub fn three_dimensional(width: u32, height: u32, depth: u32) -> Self {
        Self {
            grid: (1, 1, 1),
            block: (width, height, depth),
            total_threads: width * height * depth,
        }
    }

    /// Create a custom launch configuration
    pub fn custom(grid_x: u32, grid_y: u32, grid_z: u32, block_x: u32, block_y: u32, block_z: u32) -> Self {
        let total_threads = grid_x * grid_y * grid_z * block_x * block_y * block_z;
        Self {
            grid: (grid_x, grid_y, grid_z),
            block: (block_x, block_y, block_z),
            total_threads,
        }
    }
}

/// Kernel parameter types
///
/// These represent the ABI-level parameter types that can be passed to kernels.
/// Backends are responsible for marshalling these into their target execution model.
#[derive(Debug, Clone)]
pub enum KernelParam {
    /// Device pointer (offset + memory pool)
    /// Contains offset into the specified memory pool (Boundary or Linear)
    DevicePtr(usize, crate::MemoryPool),

    /// Scalar u32
    U32(u32),

    /// Scalar i32
    I32(i32),

    /// Scalar f32
    F32(f32),

    /// Scalar f64
    F64(f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_launch_desc_1d() {
        let launch = LaunchDesc::one_dimensional(256);
        assert_eq!(launch.grid, (1, 1, 1));
        assert_eq!(launch.block, (256, 1, 1));
        assert_eq!(launch.total_threads, 256);
    }

    #[test]
    fn test_launch_desc_2d() {
        let launch = LaunchDesc::two_dimensional(16, 16);
        assert_eq!(launch.grid, (1, 1, 1));
        assert_eq!(launch.block, (16, 16, 1));
        assert_eq!(launch.total_threads, 256);
    }

    #[test]
    fn test_launch_desc_3d() {
        let launch = LaunchDesc::three_dimensional(8, 8, 4);
        assert_eq!(launch.grid, (1, 1, 1));
        assert_eq!(launch.block, (8, 8, 4));
        assert_eq!(launch.total_threads, 256);
    }

    #[test]
    fn test_launch_desc_custom() {
        let launch = LaunchDesc::custom(2, 2, 1, 64, 1, 1);
        assert_eq!(launch.grid, (2, 2, 1));
        assert_eq!(launch.block, (64, 1, 1));
        assert_eq!(launch.total_threads, 256);
    }
}
