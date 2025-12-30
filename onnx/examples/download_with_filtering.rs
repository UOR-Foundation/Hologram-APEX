//! Example: Download HuggingFace models with advanced filtering
//!
//! This example demonstrates how to use the enhanced downloader with:
//! - Include/exclude patterns (like `hf download --include/--exclude`)
//! - File type filtering
//! - Subdirectory filtering
//! - External data format conversion
//!
//! Run with:
//! ```bash
//! cargo run --example download_with_filtering
//! ```

use hologram_onnx::{download_model_with_options, DownloadOptions, FileFilter};
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    println!("=== HuggingFace Download Examples ===\n");

    // Example 1: Download only ONNX files from specific components
    // Similar to: hf download onnxruntime/sd-turbo --include "text_encoder/*" --include "unet/*" --exclude "*.md"
    println!("Example 1: Download SD-Turbo with filtering");
    let options = DownloadOptions {
        file_filter: FileFilter::OnnxWithConfig,
        subdirs: Some(vec!["text_encoder".to_string(), "unet".to_string()]),
        include_patterns: None,
        exclude_patterns: Some(vec!["*.md".to_string(), "*.txt".to_string()]),
        skip_lfs: false,
        auto_convert_pytorch: false,
        convert_to_external: false,
        verbose: true,
    };

    let output_dir = Path::new("downloads/sd-turbo-filtered");
    std::fs::create_dir_all(output_dir)?;

    println!("\nDownloading onnxruntime/sd-turbo...");
    let _repo_path = download_model_with_options(
        "onnxruntime/sd-turbo",
        None,
        Some(output_dir),
        &options,
    )
    .await?;
    println!("✓ Downloaded to: {}\n", output_dir.display());

    // Example 2: Download with include patterns only
    println!("Example 2: Download only specific file patterns");
    let options = DownloadOptions {
        file_filter: FileFilter::All,
        subdirs: None,
        include_patterns: Some(vec![
            "text_encoder/*".to_string(),
            "*/model.onnx".to_string(),
        ]),
        exclude_patterns: None,
        skip_lfs: false,
        auto_convert_pytorch: false,
        convert_to_external: false,
        verbose: true,
    };

    let output_dir = Path::new("downloads/sd-turbo-patterns");
    std::fs::create_dir_all(output_dir)?;

    println!("\nDownloading with patterns...");
    let _repo_path = download_model_with_options(
        "onnxruntime/sd-turbo",
        None,
        Some(output_dir),
        &options,
    )
    .await?;
    println!("✓ Downloaded to: {}\n", output_dir.display());

    // Example 3: Download and convert to external data format
    println!("Example 3: Download with external data conversion");
    let options = DownloadOptions {
        file_filter: FileFilter::OnnxOnly,
        subdirs: Some(vec!["vae_decoder".to_string()]),
        include_patterns: None,
        exclude_patterns: Some(vec!["*.md".to_string()]),
        skip_lfs: false,
        auto_convert_pytorch: false,
        convert_to_external: true, // This will create separate .bin files for weights
        verbose: true,
    };

    let output_dir = Path::new("downloads/vae-external");
    std::fs::create_dir_all(output_dir)?;

    println!("\nDownloading and converting VAE decoder...");
    let _repo_path = download_model_with_options(
        "onnxruntime/sd-turbo",
        None,
        Some(output_dir),
        &options,
    )
    .await?;
    println!("✓ Downloaded and converted to: {}\n", output_dir.display());

    // Example 4: Download only configuration files (lightweight)
    println!("Example 4: Download only config files");
    let options = DownloadOptions {
        file_filter: FileFilter::ConfigOnly,
        subdirs: None,
        include_patterns: None,
        exclude_patterns: None,
        skip_lfs: true, // Skip LFS files (large model weights)
        auto_convert_pytorch: false,
        convert_to_external: false,
        verbose: true,
    };

    let output_dir = Path::new("downloads/sd-turbo-config");
    std::fs::create_dir_all(output_dir)?;

    println!("\nDownloading only config files...");
    let _repo_path = download_model_with_options(
        "onnxruntime/sd-turbo",
        None,
        Some(output_dir),
        &options,
    )
    .await?;
    println!("✓ Downloaded to: {}\n", output_dir.display());

    println!("=== All examples completed! ===");

    Ok(())
}
