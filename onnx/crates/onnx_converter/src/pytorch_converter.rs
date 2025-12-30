//! PyTorch to ONNX Converter (Rust wrapper around Python)
//!
//! This module provides a Rust interface for converting PyTorch models to ONNX format.
//! It wraps Python tooling (torch.onnx, optimum, etc.) in a type-safe Rust API.
//!
//! ## Why Hybrid Approach?
//!
//! While we prefer pure Rust, PyTorch → ONNX conversion requires:
//! - Dynamic tracing of PyTorch execution
//! - Access to PyTorch's C++ internals
//! - Complex graph construction logic
//!
//! The Python ecosystem (torch.onnx, optimum) is mature and battle-tested.
//! This wrapper provides a seamless Rust interface while leveraging that ecosystem.

use anyhow::{Context, Result};
use std::path::PathBuf;
use tokio::process::Command;

use crate::python_env;

/// Embedded Python conversion script
/// Compiled into the binary at build time
const PYTORCH_TO_ONNX_SCRIPT: &str = include_str!("../scripts/pytorch_to_onnx.py");

/// Configuration for PyTorch to ONNX conversion
#[derive(Debug, Clone)]
pub struct ConversionConfig {
    /// Input model directory (contains config.json, model.safetensors, etc.)
    pub model_dir: PathBuf,

    /// Output ONNX file path
    pub output_onnx: PathBuf,

    /// Model architecture type (e.g., "stable-diffusion", "bert", "gpt2")
    pub model_type: ModelType,

    /// Opset version for ONNX (default: 17)
    pub opset_version: u32,

    /// Whether to export with dynamic axes (for variable batch size, sequence length)
    pub dynamic_axes: bool,

    /// Optimize the ONNX graph after export
    pub optimize: bool,

    /// Simplify the ONNX graph (removes redundant nodes)
    pub simplify: bool,

    /// Use external data format (weights stored separately)
    pub external_data: bool,

    /// Verbose output
    pub verbose: bool,
}

/// Supported model architecture types
#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    /// Stable Diffusion components (UNet, VAE, TextEncoder)
    StableDiffusion,
    /// BERT-style transformers
    Bert,
    /// GPT-style transformers
    Gpt,
    /// Vision transformers
    Vit,
    /// Generic (auto-detect)
    Auto,
}

impl ModelType {
    pub fn as_str(&self) -> &str {
        match self {
            ModelType::StableDiffusion => "stable-diffusion",
            ModelType::Bert => "bert",
            ModelType::Gpt => "gpt",
            ModelType::Vit => "vit",
            ModelType::Auto => "auto",
        }
    }
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            output_onnx: PathBuf::new(),
            model_type: ModelType::Auto,
            opset_version: 17,
            dynamic_axes: true,
            optimize: true,
            simplify: false,
            external_data: false,
            verbose: false,
        }
    }
}

/// Convert PyTorch model to ONNX
///
/// This function wraps the Python conversion script and provides a clean Rust API.
/// It automatically sets up a Python virtual environment with required packages on first use.
///
/// # Arguments
///
/// * `config` - Conversion configuration
///
/// # Returns
///
/// Path to the generated ONNX file
///
/// # Example
///
/// ```no_run
/// use hologram_onnx_converter::{convert_pytorch_to_onnx, ConversionConfig, ModelType};
/// use std::path::PathBuf;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = ConversionConfig {
///     model_dir: PathBuf::from("models/stable-diffusion/unet"),
///     output_onnx: PathBuf::from("unet.onnx"),
///     model_type: ModelType::StableDiffusion,
///     ..Default::default()
/// };
///
/// let onnx_path = convert_pytorch_to_onnx(&config).await?;
/// println!("ONNX model saved to: {}", onnx_path.display());
/// # Ok(())
/// # }
/// ```
pub async fn convert_pytorch_to_onnx(config: &ConversionConfig) -> Result<PathBuf> {
    // Setup Python environment (installs packages if needed)
    let python_exe = python_env::get_python_executable(true, config.verbose).await?;
    // Verify input directory exists
    if !config.model_dir.exists() {
        anyhow::bail!("Model directory does not exist: {}", config.model_dir.display());
    }

    // Create output directory if needed
    if let Some(parent) = config.output_onnx.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    if config.verbose {
        println!("Converting PyTorch model to ONNX...");
        println!("  Model dir: {}", config.model_dir.display());
        println!("  Output: {}", config.output_onnx.display());
        println!("  Model type: {:?}", config.model_type);
    }

    // Write embedded script to temp file
    let temp_script = std::env::temp_dir().join("hologram_pytorch_to_onnx.py");
    tokio::fs::write(&temp_script, PYTORCH_TO_ONNX_SCRIPT)
        .await
        .context("Failed to write conversion script to temp directory")?;

    // Build command using venv python
    let mut cmd = Command::new(&python_exe);
    cmd.arg(&temp_script)
        .arg("--model-dir")
        .arg(&config.model_dir)
        .arg("--output")
        .arg(&config.output_onnx)
        .arg("--model-type")
        .arg(config.model_type.as_str())
        .arg("--opset")
        .arg(config.opset_version.to_string());

    if config.dynamic_axes {
        cmd.arg("--dynamic-axes");
    }

    if config.optimize {
        cmd.arg("--optimize");
    }

    if config.simplify {
        cmd.arg("--simplify");
    }

    if config.external_data {
        cmd.arg("--external-data");
    }

    if config.verbose {
        cmd.arg("--verbose");
    }

    // Execute conversion
    if config.verbose {
        println!("  Executing Python conversion script...");
    }

    let output = cmd.output().await.context("Failed to execute Python conversion script")?;

    // Clean up temp script
    let _ = tokio::fs::remove_file(&temp_script).await;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Conversion failed:\n{}", stderr);
    }

    if config.verbose {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.is_empty() {
            println!("{}", stdout);
        }
    }

    // Verify output was created
    if !config.output_onnx.exists() {
        anyhow::bail!("Conversion script completed but ONNX file was not created");
    }

    if config.verbose {
        let metadata = tokio::fs::metadata(&config.output_onnx).await?;
        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        println!("  ✓ Created ONNX model ({:.2} MB)", size_mb);
    }

    Ok(config.output_onnx.clone())
}

/// Check if Python dependencies are installed
///
/// Verifies that required Python packages (torch, onnx, etc.) are available.
pub async fn check_python_dependencies() -> Result<Vec<String>> {
    let mut missing = Vec::new();

    // Check Python version
    let output = Command::new("python3")
        .arg("--version")
        .output()
        .await
        .context("Python3 not found. Please install Python 3.8+")?;

    if !output.status.success() {
        anyhow::bail!("Python3 not available");
    }

    // Check required packages
    let required_packages = vec!["torch", "onnx", "onnxruntime", "transformers", "diffusers", "safetensors"];

    for package in required_packages {
        let output = Command::new("python3")
            .arg("-c")
            .arg(format!("import {}", package))
            .output()
            .await?;

        if !output.status.success() {
            missing.push(package.to_string());
        }
    }

    Ok(missing)
}

/// Install missing Python dependencies
pub async fn install_python_dependencies(packages: &[String], verbose: bool) -> Result<()> {
    if packages.is_empty() {
        return Ok(());
    }

    if verbose {
        println!("Installing missing Python packages: {}", packages.join(", "));
    }

    let output = Command::new("pip3")
        .arg("install")
        .arg("-q")
        .args(packages)
        .output()
        .await
        .context("Failed to run pip3. Please install pip manually.")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Failed to install Python packages:\n{}", stderr);
    }

    if verbose {
        println!("  ✓ Packages installed successfully");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_check_python_dependencies() {
        // This test requires Python to be installed
        match check_python_dependencies().await {
            Ok(missing) => {
                println!("Missing packages: {:?}", missing);
            }
            Err(e) => {
                println!("Python not available: {}", e);
            }
        }
    }

    #[test]
    fn test_model_type_as_str() {
        assert_eq!(ModelType::StableDiffusion.as_str(), "stable-diffusion");
        assert_eq!(ModelType::Bert.as_str(), "bert");
        assert_eq!(ModelType::Auto.as_str(), "auto");
    }
}
