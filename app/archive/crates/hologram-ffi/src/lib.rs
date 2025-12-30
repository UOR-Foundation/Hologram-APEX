//! # Hologram FFI
//!
//! Cross-language Foreign Function Interface for Hologram
//!
//! This crate provides a unified FFI interface using UniFFI to expose Hologram
//! Circuit-based compute functionality across multiple programming languages
//! including Python, TypeScript, Swift, Kotlin, and WebAssembly.
//!
//! ## Architecture
//!
//! The FFI uses handle-based object management to safely expose Rust objects
//! across language boundaries:
//!
//! - **Executor** - Circuit circuit executor (u64 handle)
//! - **Buffer<T>** - Typed memory buffers (u64 handle)
//! - **Tensor<T>** - Multi-dimensional arrays (u64 handle)
//!
//! All operations compile to canonical Circuit circuits under the hood,
//! providing lowest-latency execution through pattern-based canonicalization.

// Allow clippy warnings in generated UniFFI code
#![allow(clippy::empty_line_after_doc_comments)]

// Module declarations
mod buffer;
mod buffer_ext;
mod buffer_zerocopy;
mod c_api; // C-compatible API for direct C++ linkage
mod executor;
mod executor_ext;
mod handles;
#[cfg(feature = "model-server")]
mod model_server; // Model server FFI bindings
#[cfg(not(feature = "model-server"))]
mod model_server_stub; // Model server stubs when feature is disabled
mod tensor;

// Re-export all public functions for UniFFI
pub use buffer::{
    buffer_canonicalize_all, buffer_cleanup, buffer_copy, buffer_copy_from_canonical_slice, buffer_copy_from_slice,
    buffer_copy_to_slice, buffer_fill, buffer_length, buffer_to_vec, buffer_verify_canonical,
};
pub use buffer_ext::{
    buffer_class_index, buffer_element_size, buffer_is_boundary, buffer_is_empty, buffer_is_linear, buffer_pool,
    buffer_size_bytes,
};
pub use buffer_zerocopy::{buffer_as_mut_ptr, buffer_as_ptr, buffer_copy_from_bytes, buffer_to_bytes};
pub use executor::{
    executor_allocate_buffer, executor_cleanup, new_executor, new_executor_auto, new_executor_with_backend,
};
pub use executor_ext::executor_allocate_boundary_buffer;
pub use handles::clear_all_registries;

// Model server functions (real implementation when feature enabled, stubs otherwise)
#[cfg(feature = "model-server")]
pub use model_server::{
    inference_count_tokens, inference_engine_create, inference_engine_destroy, inference_generate_completion,
    inference_generate_embedding, model_list_loaded, model_load_bert, model_load_from_config, model_load_gpt2,
    model_unload,
};

#[cfg(not(feature = "model-server"))]
pub use model_server_stub::{
    inference_count_tokens, inference_engine_create, inference_engine_destroy, inference_generate_completion,
    inference_generate_embedding, model_list_loaded, model_load_bert, model_load_from_config, model_load_gpt2,
    model_unload,
};

pub use tensor::{
    tensor_broadcast_shapes, tensor_buffer, tensor_cleanup, tensor_contiguous, tensor_dlpack_device_id,
    tensor_dlpack_device_type, tensor_from_buffer, tensor_from_buffer_with_strides, tensor_from_dlpack_capsule,
    tensor_is_broadcast_compatible_with, tensor_is_contiguous, tensor_matmul, tensor_narrow, tensor_ndim, tensor_numel,
    tensor_offset, tensor_permute, tensor_reshape, tensor_select, tensor_shape, tensor_slice, tensor_strides,
    tensor_to_dlpack, tensor_transpose, tensor_view_1d,
};

// NOTE: ISA-based operation bindings (math, activation, reduce, loss, linalg) removed
// Will be replaced with CompiledOperation bindings in Phase 2.2

/// Get the version of the hologram-ffi library
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// Include the UDL file for UniFFI scaffolding
uniffi::include_scaffolding!("hologram_ffi");
