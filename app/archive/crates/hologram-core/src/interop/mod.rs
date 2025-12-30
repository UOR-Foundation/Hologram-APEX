//! Interoperability with other frameworks
//!
//! This module provides zero-copy tensor exchange protocols:
//! - **DLPack**: Standard protocol for PyTorch, TensorFlow, JAX, etc.
//! - Future: Arrow, ONNX Runtime

pub mod dlpack;

pub use dlpack::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensor, DLPackType, DLTensor};
