//! # Hologram ONNX Converter
//!
//! PyTorch to ONNX converter with automatic Python environment management.
//!
//! This crate provides a Rust interface for converting PyTorch models to ONNX format,
//! with automatic Python virtual environment creation and package installation.
//!
//! ## Features
//!
//! - **Automatic venv management**: Creates and manages Python virtual environments
//! - **Package installation**: Automatically installs required packages (torch, onnx, etc.)
//! - **Multiple model types**: Supports Stable Diffusion, Transformers (BERT, GPT, etc.)
//! - **External data format**: Optional conversion to ONNX external data format
//! - **Cross-platform**: Works on Windows, Linux, and macOS
//!
//! ## Quick Start
//!
//! ```no_run
//! use hologram_onnx_converter::{convert_pytorch_to_onnx, ConversionConfig, ModelType};
//! use std::path::PathBuf;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = ConversionConfig {
//!     model_dir: PathBuf::from("models/my-model"),
//!     output_onnx: PathBuf::from("model.onnx"),
//!     model_type: ModelType::Auto,
//!     ..Default::default()
//! };
//!
//! let onnx_path = convert_pytorch_to_onnx(&config).await?;
//! println!("ONNX model saved to: {}", onnx_path.display());
//! # Ok(())
//! # }
//! ```

/// Python environment manager
pub mod python_env;

/// PyTorch to ONNX converter
pub mod pytorch_converter;

// Re-export commonly used types
pub use pytorch_converter::{
    check_python_dependencies, convert_pytorch_to_onnx, install_python_dependencies,
    ConversionConfig, ModelType,
};
pub use python_env::{
    check_missing_packages, get_python_executable, setup_python_environment,
};
