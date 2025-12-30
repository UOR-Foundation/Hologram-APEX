//! DLPack protocol implementation for zero-copy tensor exchange
//!
//! DLPack is a standard for zero-copy tensor exchange between frameworks (PyTorch, TensorFlow, JAX, etc.).
//! This module provides FFI-safe structures and methods to export/import Hologram tensors via DLPack.
//!
//! # DLPack Specification
//!
//! DLPack defines a common in-memory tensor structure with:
//! - Data pointer (device or host memory)
//! - Shape and strides
//! - Data type
//! - Device type (CPU, CUDA, etc.)
//!
//! # Performance Characteristics
//!
//! - **Export (Hologram → External)**: Zero-copy (~100-200ns overhead, shares memory)
//! - **Import (External → Hologram)**: Copy-based (O(n), copies data into 96-class system)
//!
//! # Usage Example
//!
//! ```no_run
//! use hologram_core::{Executor, Tensor};
//!
//! # fn main() -> hologram_core::Result<()> {
//! let mut exec = Executor::new()?;
//!
//! // Create Hologram tensor
//! let buffer = exec.allocate::<f32>(12)?;
//! let tensor = Tensor::from_buffer(buffer, vec![3, 4])?;
//!
//! // Export to DLPack (zero-copy)
//! let dlpack = exec.tensor_to_dlpack(&tensor)?;
//!
//! // Pass to PyTorch/JAX via Python bindings
//! // let pytorch_tensor = torch.from_dlpack(dlpack_capsule);
//! # Ok(())
//! # }
//! ```
//!
//! # Import from External Framework
//!
//! ```no_run
//! use hologram_core::{Executor, Tensor};
//!
//! # fn main() -> hologram_core::Result<()> {
//! let mut exec = Executor::new()?;
//! # let dlpack_ptr = 0x1234 as u64; // Simulated pointer
//!
//! // Import from PyTorch/JAX (copy-based)
//! // let dlpack_ptr = extract_from_pytorch_capsule(...);
//! let tensor: Tensor<f32> = exec.tensor_from_dlpack(dlpack_ptr)?;
//!
//! println!("Imported tensor shape: {:?}", tensor.shape());
//! # Ok(())
//! # }
//! ```
//!
//! # Safety
//!
//! DLPack tensors must outlive the DLPack struct. The deleter function
//! is called when the consuming framework (e.g., PyTorch) is done with the data.
//!
//! For complete documentation, see [DLPack Integration Guide](../../docs/DLPACK_INTEGRATION.md).

use std::ffi::c_void;
use std::os::raw::c_int;

/// DLPack device types
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DLDeviceType {
    /// CPU memory
    CPU = 1,
    /// CUDA GPU memory
    CUDA = 2,
    /// CUDA pinned memory
    CUDAHost = 3,
    /// OpenCL
    OpenCL = 4,
    /// Vulkan
    Vulkan = 7,
    /// Metal (Apple GPU)
    Metal = 8,
    /// ROCm (AMD GPU)
    ROCM = 10,
}

/// DLPack device context
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDevice {
    /// Device type (CPU, CUDA, etc.)
    pub device_type: DLDeviceType,
    /// Device ID (e.g., GPU 0, GPU 1, etc.)
    pub device_id: c_int,
}

impl DLDevice {
    /// Create CPU device
    pub fn cpu() -> Self {
        Self {
            device_type: DLDeviceType::CPU,
            device_id: 0,
        }
    }

    /// Create CUDA device
    pub fn cuda(device_id: i32) -> Self {
        Self {
            device_type: DLDeviceType::CUDA,
            device_id,
        }
    }
}

/// DLPack data type codes
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DLDataTypeCode {
    Int = 0,
    UInt = 1,
    Float = 2,
    /// OpaqueHandle (used for custom types)
    Handle = 3,
}

/// DLPack data type descriptor
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDataType {
    /// Type code (int, float, etc.)
    pub code: DLDataTypeCode,
    /// Number of bits per element (e.g., 32 for f32)
    pub bits: u8,
    /// Number of lanes for vector types (1 for scalars)
    pub lanes: u16,
}

impl DLDataType {
    /// Float32 type
    pub fn float32() -> Self {
        Self {
            code: DLDataTypeCode::Float,
            bits: 32,
            lanes: 1,
        }
    }

    /// Float64 type
    pub fn float64() -> Self {
        Self {
            code: DLDataTypeCode::Float,
            bits: 64,
            lanes: 1,
        }
    }

    /// Int32 type
    pub fn int32() -> Self {
        Self {
            code: DLDataTypeCode::Int,
            bits: 32,
            lanes: 1,
        }
    }

    /// Int64 type
    pub fn int64() -> Self {
        Self {
            code: DLDataTypeCode::Int,
            bits: 64,
            lanes: 1,
        }
    }

    /// UInt8 type
    pub fn uint8() -> Self {
        Self {
            code: DLDataTypeCode::UInt,
            bits: 8,
            lanes: 1,
        }
    }
}

/// DLPack tensor descriptor
///
/// This is the core structure that describes a tensor's memory layout.
/// It's compatible with the DLPack C ABI specification.
#[repr(C)]
pub struct DLTensor {
    /// Pointer to tensor data (device or host memory)
    pub data: *mut c_void,
    /// Device context (CPU, CUDA, etc.)
    pub device: DLDevice,
    /// Number of dimensions
    pub ndim: c_int,
    /// Data type descriptor
    pub dtype: DLDataType,
    /// Shape array (pointer to ndim elements)
    pub shape: *mut i64,
    /// Strides array (pointer to ndim elements, may be null for row-major)
    pub strides: *mut i64,
    /// Byte offset from data pointer to first element
    pub byte_offset: u64,
}

/// DLPack managed tensor
///
/// This structure wraps DLTensor with ownership and deletion semantics.
/// When a framework consumes a DLManagedTensor, it must call the deleter
/// when done to free resources.
#[repr(C)]
pub struct DLManagedTensor {
    /// The tensor descriptor
    pub dl_tensor: DLTensor,
    /// Opaque manager context (framework-specific data)
    pub manager_ctx: *mut c_void,
    /// Deleter function called when consumer is done
    /// Signature: fn(*mut DLManagedTensor)
    pub deleter: Option<extern "C" fn(*mut DLManagedTensor)>,
}

// Safety: DLManagedTensor is an FFI type that can be safely sent between threads
// The deleter ensures proper cleanup
unsafe impl Send for DLManagedTensor {}
unsafe impl Sync for DLManagedTensor {}

impl DLManagedTensor {
    /// Create a new managed tensor with deleter
    ///
    /// # Safety
    ///
    /// - `dl_tensor` must point to valid memory for its lifetime
    /// - `manager_ctx` must be valid for the deleter
    /// - `deleter` must properly free all allocated resources
    pub unsafe fn new(
        dl_tensor: DLTensor,
        manager_ctx: *mut c_void,
        deleter: extern "C" fn(*mut DLManagedTensor),
    ) -> Box<Self> {
        Box::new(Self {
            dl_tensor,
            manager_ctx,
            deleter: Some(deleter),
        })
    }
}

/// Helper function to infer DLDataType from Rust type
pub trait DLPackType {
    fn dlpack_dtype() -> DLDataType;
}

impl DLPackType for f32 {
    fn dlpack_dtype() -> DLDataType {
        DLDataType::float32()
    }
}

impl DLPackType for f64 {
    fn dlpack_dtype() -> DLDataType {
        DLDataType::float64()
    }
}

impl DLPackType for i32 {
    fn dlpack_dtype() -> DLDataType {
        DLDataType::int32()
    }
}

impl DLPackType for i64 {
    fn dlpack_dtype() -> DLDataType {
        DLDataType::int64()
    }
}

impl DLPackType for u8 {
    fn dlpack_dtype() -> DLDataType {
        DLDataType::uint8()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dldevice_creation() {
        let cpu = DLDevice::cpu();
        assert_eq!(cpu.device_type, DLDeviceType::CPU);
        assert_eq!(cpu.device_id, 0);

        let cuda0 = DLDevice::cuda(0);
        assert_eq!(cuda0.device_type, DLDeviceType::CUDA);
        assert_eq!(cuda0.device_id, 0);

        let cuda1 = DLDevice::cuda(1);
        assert_eq!(cuda1.device_type, DLDeviceType::CUDA);
        assert_eq!(cuda1.device_id, 1);
    }

    #[test]
    fn test_dldatatype_creation() {
        let f32_type = DLDataType::float32();
        assert_eq!(f32_type.code, DLDataTypeCode::Float);
        assert_eq!(f32_type.bits, 32);
        assert_eq!(f32_type.lanes, 1);

        let i32_type = DLDataType::int32();
        assert_eq!(i32_type.code, DLDataTypeCode::Int);
        assert_eq!(i32_type.bits, 32);
        assert_eq!(i32_type.lanes, 1);
    }

    #[test]
    fn test_dlpack_type_trait() {
        let f32_dtype = f32::dlpack_dtype();
        assert_eq!(f32_dtype.code, DLDataTypeCode::Float);
        assert_eq!(f32_dtype.bits, 32);

        let i64_dtype = i64::dlpack_dtype();
        assert_eq!(i64_dtype.code, DLDataTypeCode::Int);
        assert_eq!(i64_dtype.bits, 64);
    }
}
