//! Backend implementations for different execution targets
//!
//! This module contains:
//! - `common` - Shared backend infrastructure (registers, memory, execution state, Atlas ops)
//! - `cpu` - CPU backend (reference implementation)
//! - `metal` - Metal GPU backend (Apple Silicon)
//! - `cuda` - CUDA GPU backend (NVIDIA GPUs)
//! - `wasm` - WASM backend (WebAssembly)
//! - `tpu` - TPU backend via PJRT (TODO)
//! - `fpga` - FPGA backend (TODO)

pub mod common;
pub mod cpu;
pub mod cuda;
pub mod metal;
pub mod wasm;

// Re-export backends
pub use cpu::CpuBackend;
pub use cuda::CudaBackend;
pub use metal::MetalBackend;
pub use wasm::WasmBackend;

// Re-export WebGPU backend (WASM GPU acceleration)
#[cfg(feature = "webgpu")]
pub use wasm::WebGpuBackend;
