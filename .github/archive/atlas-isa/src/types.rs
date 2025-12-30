//! Core types for the Atlas ISA execution model

use crate::{constants::*, uor::ResonanceClass};

use crate::uor::PhiCoordinate;
use std::fmt;

/// Grid dimensions for kernel launch
///
/// Defines the 3D iteration space of blocks that execute the kernel.
/// Each block is identified by its position (x, y, z) within the grid.
///
/// # Atlas ISA Mapping (§2.1, §2.2)
///
/// - The grid defines the outer iteration space
/// - `class_id = blockIdx.x mod 96` maps each block to a resonance class
/// - Target implementations map Grid to their native parallelism (GPU blocks, CPU thread pools, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridDim {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl GridDim {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub fn linear_size(&self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }
}

impl Default for GridDim {
    fn default() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }
}

/// Block dimensions
///
/// Defines the 3D arrangement of lanes within a block.
/// All lanes in a block can synchronize via BAR.SYNC (§2.4).
///
/// # Atlas ISA Mapping (§2.1, §2.2)
///
/// - `boundary = (laneIdx.x mod 48, laneIdx.y mod 256)` for boundary lens addressing
/// - Blocks share per-block scratch memory (Shared address space, §5.1)
/// - Target implementations map to native workgroups (GPU thread blocks, CPU task groups, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockDim {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl BlockDim {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub fn linear_size(&self) -> u32 {
        self.x * self.y * self.z
    }
}

impl Default for BlockDim {
    fn default() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }
}

/// Block index within grid
///
/// Identifies a block's position in the 3D grid iteration space.
///
/// # Atlas Semantics (§2.2)
///
/// - `class_id = blockIdx.x mod 96` maps each block to its resonance class
/// - This mapping enables class-aware scheduling and mirror-safe execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockIdx {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl BlockIdx {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Compute linear index within grid
    pub fn linear_index(&self, grid: GridDim) -> u64 {
        (self.z as u64 * grid.y as u64 * grid.x as u64) + (self.y as u64 * grid.x as u64) + self.x as u64
    }

    /// Compute resonance class from block X coordinate (§2.2)
    ///
    /// Maps the block to one of 96 resonance classes:
    /// `class_id = blockIdx.x mod 96`
    ///
    /// This enables class-based scheduling and mirror-safe execution patterns.
    pub fn resonance_class(&self) -> ResonanceClass {
        let value = (self.x % RESONANCE_CLASSES) as u8;
        ResonanceClass::new(value).expect("blockIdx.x mod 96 must produce valid resonance class")
    }
}

/// Lane index within block
///
/// A Lane is the minimal execution agent in the Atlas ISA (§2.1).
/// Lanes within a block share per-block resources and can synchronize
/// via BAR.SYNC instructions.
///
/// # Atlas Semantics (§2.2)
///
/// - `boundary = (laneIdx.x mod 48, laneIdx.y mod 256)` - Boundary lens addressing
/// - Lanes execute in SIMT/SIMD fashion on target hardware
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaneIdx {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl LaneIdx {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Compute linear index within block
    pub fn linear_index(&self, block: BlockDim) -> u32 {
        (self.z * block.y * block.x) + (self.y * block.x) + self.x
    }

    /// Compute boundary coordinates from lane index (§2.2)
    ///
    /// Maps lane position to boundary lens coordinates:
    /// - `page = laneIdx.x mod 48`
    /// - `byte = laneIdx.y mod 256`
    ///
    /// This enables direct access to the 48 × 256 boundary torus.
    pub fn boundary_coords(&self) -> (u8, u8) {
        let coord = PhiCoordinate::new((self.x % PAGES) as u8, (self.y % BYTES_PER_PAGE) as u8)
            .expect("modulo projection must produce in-range boundary coordinates");
        (coord.page, coord.byte)
    }
}

/// Ergonomic alias for LaneIdx
///
/// While the Atlas ISA spec uses "Lane", "Thread" is a widely understood
/// term across parallel programming models. This alias provides familiar
/// naming for users coming from CUDA, OpenCL, or other threading models.
pub type ThreadIdx = LaneIdx;

/// Launch configuration for an Atlas kernel
///
/// Specifies the iteration space (grid × block) and resources required
/// for kernel execution.
///
/// # Fields
///
/// - `grid`: 3D grid dimensions (number of blocks in each dimension)
/// - `block`: 3D block dimensions (number of lanes per block in each dimension)
/// - `shared_mem_bytes`: Per-block scratch memory allocation (§5.1: Shared address space)
///
/// # Atlas ISA Semantics
///
/// Targets map LaunchConfig to their native execution models:
/// - **GPU**: Grid → thread blocks, Block → threads per block
/// - **CPU**: Grid → task batches, Block → parallel lanes via SIMD/threading
/// - **TPU**: Grid → tile groups, Block → tensor cores
#[derive(Debug, Clone, Copy, Default)]
pub struct LaunchConfig {
    pub grid: GridDim,
    pub block: BlockDim,
    pub shared_mem_bytes: usize,
}

impl LaunchConfig {
    pub fn new(grid: GridDim, block: BlockDim, shared_mem_bytes: usize) -> Self {
        Self {
            grid,
            block,
            shared_mem_bytes,
        }
    }

    /// Total number of blocks in the grid
    pub fn total_blocks(&self) -> u64 {
        self.grid.linear_size()
    }

    /// Total number of lanes across all blocks
    pub fn total_lanes(&self) -> u64 {
        self.grid.linear_size() * self.block.linear_size() as u64
    }
}

/// Address space in Atlas memory model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressSpace {
    /// Global memory (64-bit flat, accessible by all lanes)
    Global,
    /// Shared memory (per-block scratch, lifetime = block)
    Shared,
    /// Constant memory (read-only, initialized at module load)
    Const,
    /// Local memory (per-lane spill space)
    Local,
}

impl fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AddressSpace::Global => write!(f, "global"),
            AddressSpace::Shared => write!(f, "shared"),
            AddressSpace::Const => write!(f, "const"),
            AddressSpace::Local => write!(f, "local"),
        }
    }
}

/// Memory access pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Read access
    Read,
    /// Write access
    Write,
    /// Read-modify-write access
    ReadWrite,
}

/// Synchronization scope
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncScope {
    /// Block-level synchronization
    Block,
    /// Device-level synchronization
    Device,
    /// System-level synchronization
    System,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_dim_linear_size() {
        let grid = GridDim::new(2, 3, 4);
        assert_eq!(grid.linear_size(), 24);
    }

    #[test]
    fn test_block_dim_linear_size() {
        let block = BlockDim::new(8, 8, 1);
        assert_eq!(block.linear_size(), 64);
    }

    #[test]
    fn test_block_idx_linear_index() {
        let grid = GridDim::new(4, 4, 2);
        let block = BlockIdx::new(1, 2, 0);
        assert_eq!(block.linear_index(grid), 9); // 0*4*4 + 2*4 + 1
    }

    #[test]
    fn test_block_idx_resonance_class() {
        // class_id = blockIdx.x mod 96
        let block = BlockIdx::new(100, 0, 0);
        assert_eq!(block.resonance_class().as_u8(), 4); // 100 % 96

        let block = BlockIdx::new(0, 0, 0);
        assert_eq!(block.resonance_class().as_u8(), 0);

        let block = BlockIdx::new(95, 0, 0);
        assert_eq!(block.resonance_class().as_u8(), 95);
    }

    #[test]
    fn test_lane_idx_boundary_coords() {
        let lane = LaneIdx::new(50, 300, 0);
        let (bx, by) = lane.boundary_coords();
        assert_eq!(bx, 2); // 50 % 48
        assert_eq!(by, 44); // 300 % 256
    }

    #[test]
    fn test_launch_config_totals() {
        let config = LaunchConfig::new(GridDim::new(2, 2, 1), BlockDim::new(8, 8, 1), 1024);

        assert_eq!(config.total_blocks(), 4);
        assert_eq!(config.total_lanes(), 256); // 4 blocks * 64 lanes/block
    }
}
