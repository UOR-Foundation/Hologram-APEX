//! WebGPU compute acceleration for WASM backend
//!
//! This module provides GPU-accelerated execution of hologram operations
//! through WebGPU compute shaders in browser environments.
//!
//! ## Architecture
//!
//! - **Device Management**: WebGPU device, queue, and adapter initialization
//! - **Compute Pipelines**: WGSL shader compilation and caching
//! - **Hybrid Execution**: Automatic dispatch (GPU for large ops, CPU fallback)
//! - **Generator Mapping**: 7 fundamental generators → WGSL compute kernels
//!
//! ## Usage
//!
//! ```rust,ignore
//! use hologram_backends::backends::wasm::webgpu::WebGpuDevice;
//!
//! async fn example() -> Result<(), String> {
//!     // Initialize WebGPU device (async in browser)
//!     let device = WebGpuDevice::new().await?;
//!
//!     // Check availability
//!     if WebGpuDevice::is_available() {
//!         // Use GPU acceleration
//!     } else {
//!         // Fallback to scalar WASM
//!     }
//!     Ok(())
//! }
//! ```

#[cfg(feature = "webgpu")]
mod device;

#[cfg(feature = "webgpu")]
pub mod pipeline;

#[cfg(feature = "webgpu")]
pub mod buffer;

#[cfg(feature = "webgpu")]
mod executor;

#[cfg(feature = "webgpu")]
pub mod dispatch;

#[cfg(feature = "webgpu")]
pub mod workgroup;

#[cfg(feature = "webgpu")]
pub mod buffer_pool;

#[cfg(feature = "webgpu")]
mod backend;

#[cfg(feature = "webgpu")]
#[cfg(feature = "webgpu")]
pub mod isa_translator;

#[cfg(feature = "webgpu")]
pub use device::WebGpuDevice;

#[cfg(feature = "webgpu")]
pub use executor::WebGpuExecutor;

#[cfg(feature = "webgpu")]
pub use backend::WebGpuBackend;

#[cfg(not(feature = "webgpu"))]
pub struct WebGpuDevice;

#[cfg(not(feature = "webgpu"))]
impl WebGpuDevice {
    /// WebGPU is not available (feature not enabled)
    pub fn is_available() -> bool {
        false
    }
}
