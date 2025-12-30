//! # Hologram ONNX
//!
//! ONNX toolchain for Hologram - compile and run ONNX models as `.holo` binaries.
//!
//! ## Overview
//!
//! This library provides both a CLI tool (`hologram-onnx`) and a library API for compiling
//! ONNX models into Hologram's `.holo` format, which enables O(1) lookup-based inference
//! using pre-computed results stored in perfect hash tables.
//!
//! ## CLI Usage
//!
//! ```bash
//! # Compile an ONNX model
//! hologram-onnx compile --input model.onnx --output model.holo --parallel
//!
//! # Display model information
//! hologram-onnx info --input model.holo
//!
//! # Run inference (coming soon)
//! hologram-onnx run --model model.holo --input data.json
//! ```
//!
//! ## Library Usage
//!
//! ```no_run
//! use hologram_onnx::Compiler;
//!
//! let compiler = Compiler::new()
//!     .with_memory_budget(2048)
//!     .with_verbose(true)
//!     .with_parallel(true);
//!
//! compiler.compile("model.onnx", "model.holo")?;
//! # Ok::<(), hologram_onnx::CompilerError>(())
//! ```
//!
//! ## Architecture
//!
//! The Hologram approach uses:
//! - Griess algebra embedding (196,884 dimensions)
//! - Pre-computation of all operation results
//! - Perfect hash tables for O(1) pattern lookup
//! - SIMD-aligned binary format (.holo)
//! - Zero-copy runtime execution

// Re-export everything from the compiler crate
pub use hologram_onnx_compiler::*;

// Re-export everything from the downloader crate
pub use hologram_onnx_downloader as downloader;
pub use hologram_onnx_downloader::{
    download_model_with_options, download_onnx_model, parse_hf_model_spec, DownloadOptions,
    FileFilter,
};

// Re-export everything from the converter crate
pub use hologram_onnx_converter as converter;
pub use hologram_onnx_converter::{
    check_python_dependencies, convert_pytorch_to_onnx, ConversionConfig, ModelType,
    get_python_executable, setup_python_environment,
};

/// Get the version of hologram-onnx
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let ver = version();
        assert!(!ver.is_empty());
        assert!(ver.contains('.'));
    }

    #[test]
    fn test_compiler_accessible() {
        // Verify compiler is accessible through re-exports
        let _compiler_type = std::any::type_name::<Compiler>();
        let _stats_type = std::any::type_name::<CompilationStats>();
    }
}
