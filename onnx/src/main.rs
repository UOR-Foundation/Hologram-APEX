//! Hologram ONNX Compiler CLI
//!
//! Command-line interface for compiling ONNX models to Hologram .holo format
//! and downloading models from HuggingFace Hub.

use anyhow::Context;
use clap::{Parser, Subcommand};
use hologram::common::helpers::system::get_available_memory_mb;
use hologram_onnx::{convert_pytorch_to_onnx, ConversionConfig, ModelType};
use hologram_onnx_compiler::Compiler;
use hologram_onnx_downloader::{download_onnx_model, parse_hf_model_spec};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process;

/// Hologram ONNX - Compile and manage ONNX models for Hologram
///
/// This tool provides two main commands:
/// - `compile`: Compile ONNX models to .holo format for O(1) lookup-based inference
/// - `download`: Download ONNX models from HuggingFace Hub
#[derive(Parser, Debug)]
#[command(name = "hologram-onnx")]
#[command(author = "Hologram Team")]
#[command(version)]
#[command(about = "Compile and manage ONNX models for Hologram", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Compile an ONNX model to Hologram .holo format
    ///
    /// The compiler uses a simplified 4-step flow:
    /// 1. Load ONNX → HologramGraph (petgraph-based IR)
    /// 2. Optimize Graph (fusion, dead code elimination)
    /// 3. Execute Operators (on sample patterns with Atlas)
    /// 4. Serialize Binary (hash tables + address space)
    Compile(CompileArgs),

    /// Download an ONNX model from HuggingFace Hub
    Download(DownloadArgs),

    /// Convert PyTorch models to ONNX format
    ///
    /// Converts PyTorch models (from HuggingFace or local) to ONNX format.
    /// Supports Stable Diffusion, BERT, GPT, and other architectures.
    /// Uses Python tooling (torch.onnx) via Rust wrapper.
    Convert(ConvertArgs),

    /// Download and convert a model from HuggingFace in one step
    ///
    /// Combines download and convert operations for convenience.
    /// Downloads a PyTorch model from HuggingFace Hub, then converts it to ONNX.
    /// Optionally cleans up downloaded files to save disk space.
    AutoConvert(AutoConvertArgs),
}

#[derive(Parser, Debug)]
struct CompileArgs {
    /// Path to input ONNX model file (can be set in config file)
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// HuggingFace model repository (e.g., "deepseek-ai/DeepSeek-OCR")
    ///
    /// If specified, the model will be downloaded from HuggingFace Hub before compilation.
    #[arg(long = "hf-model", conflicts_with = "input")]
    hf_model: Option<String>,

    /// Output path for compiled .holo file
    #[arg(short, long)]
    output: PathBuf,

    /// Enable parallel processing (recommended, 100x faster)
    #[arg(short, long)]
    parallel: bool,

    /// Memory budget in MB (default: 8192 MB = 8 GB)
    #[arg(short, long, default_value = "8192")]
    memory_budget: usize,

    /// Accuracy target (0.0-1.0, default: 0.95 = 95%)
    #[arg(short, long, default_value = "0.95")]
    accuracy: f64,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Export debug artifacts to directory (optional)
    #[arg(long = "debug-export")]
    debug_export: Option<PathBuf>,

    /// Checkpoint directory for resumable compilation (optional)
    #[arg(long = "checkpoint-dir")]
    checkpoint_dir: Option<PathBuf>,

    /// Checkpoint interval (save every N operations, default: 10)
    #[arg(long = "checkpoint-interval", default_value = "10")]
    checkpoint_interval: usize,

    /// Path to config file (TOML format)
    ///
    /// If not specified, searches for config in:
    /// 1. ./config.toml (current dir)
    /// 2. ./.hologram-onnx-compiler.toml (dotfile in current dir)
    /// 3. ./hologram-onnx-compiler.toml (current dir)
    /// 4. ~/.config/hologram/onnx-compiler.toml (user config)
    #[arg(short = 'c', long = "config")]
    config: Option<PathBuf>,

    /// Input shapes for shape-specific compilation (enables 100% pre-compilation)
    ///
    /// Format: "name:dim1,dim2,..." (e.g., "input_ids:1,77")
    /// Can be specified multiple times for different inputs.
    /// When specified, model will be optimized for these exact shapes.
    ///
    /// Example:
    ///   --input-shape "input_ids:1,77" --input-shape "attention_mask:1,77"
    ///
    /// Benefits:
    /// - 100% pre-compilation (zero runtime-JIT)
    /// - All MatMul/LayerNorm operations use pre-computed results
    /// - Faster inference (~35ns per op vs ~1-100µs JIT)
    ///
    /// Trade-off:
    /// - Model only works for specified shapes
    /// - Falls back to runtime-JIT for different shapes
    #[arg(long = "input-shape")]
    input_shapes: Vec<String>,

    /// Path to SafeTensors file(s) containing model weights (optional)
    ///
    /// Allows loading ONNX graph structure separately from weights.
    /// Useful when you have:
    /// - ONNX file with graph structure only (no weights or placeholder weights)
    /// - SafeTensors file(s) with actual trained weights
    ///
    /// Can be specified multiple times to load weights from multiple files.
    /// Weights are merged into the graph's initializers, combining tensor space
    /// from ONNX with actual weight values from SafeTensors.
    ///
    /// Example:
    ///   --input model.onnx --weights model.safetensors
    ///   --input model.onnx --weights encoder.safetensors --weights decoder.safetensors
    #[arg(long = "weights")]
    weights: Vec<PathBuf>,
}

#[derive(Parser, Debug)]
struct DownloadArgs {
    /// HuggingFace model repository (e.g., "deepseek-ai/DeepSeek-OCR")
    ///
    /// Can include optional filename after colon:
    ///   - "repo/model" (auto-detect ONNX file)
    ///   - "repo/model:specific.onnx" (explicit file)
    #[arg(value_name = "REPO[:FILE]")]
    model: String,

    /// Output directory for downloaded model (default: current directory)
    #[arg(short, long, default_value = "./models")]
    output: PathBuf,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Parser, Debug)]
struct ConvertArgs {
    /// Path to PyTorch model directory (contains config.json, model.safetensors, etc.)
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Output ONNX file path
    #[arg(short, long)]
    output: PathBuf,

    /// Model type (auto-detect if not specified)
    #[arg(long, default_value = "auto")]
    model_type: String,

    /// ONNX opset version
    #[arg(long, default_value = "17")]
    opset: u32,

    /// Enable dynamic axes (variable batch size, sequence length)
    #[arg(long)]
    dynamic_axes: bool,

    /// Optimize ONNX graph after export
    #[arg(long)]
    optimize: bool,

    /// Simplify ONNX graph (requires onnx-simplifier)
    #[arg(long)]
    simplify: bool,

    /// Use external data format (weights stored separately)
    #[arg(long)]
    external_data: bool,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Skip automatic Python environment setup (use system Python)
    #[arg(long)]
    no_auto_install: bool,
}

#[derive(Parser, Debug)]
struct AutoConvertArgs {
    /// HuggingFace model repository (e.g., "IDKiro/sdxs-512-0.9")
    #[arg(value_name = "REPO")]
    repo: String,

    /// Model component to convert (e.g., "unet", "vae", "text_encoder")
    /// If not specified, tries to convert the root model directory
    #[arg(short, long)]
    component: Option<String>,

    /// Output ONNX file path
    #[arg(short, long)]
    output: PathBuf,

    /// Temporary directory for downloaded model (default: ./models/temp)
    #[arg(long, default_value = "./models/temp")]
    download_dir: PathBuf,

    /// Model type (auto-detect if not specified)
    #[arg(long, default_value = "auto")]
    model_type: String,

    /// ONNX opset version
    #[arg(long, default_value = "17")]
    opset: u32,

    /// Enable dynamic axes (variable batch size, sequence length)
    #[arg(long)]
    dynamic_axes: bool,

    /// Optimize ONNX graph after export
    #[arg(long)]
    optimize: bool,

    /// Simplify ONNX graph (requires onnx-simplifier)
    #[arg(long)]
    simplify: bool,

    /// Use external data format (weights stored separately)
    #[arg(long)]
    external_data: bool,

    /// Keep downloaded model files after conversion
    #[arg(long)]
    keep_download: bool,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    // Run appropriate command
    let result = match cli.command {
        Commands::Compile(args) => run_compile(args).await,
        Commands::Download(args) => run_download(args).await,
        Commands::Convert(args) => run_convert(args).await,
        Commands::AutoConvert(args) => run_auto_convert(args).await,
    };

    if let Err(e) = result {
        eprintln!("Error: {:#}", e);
        process::exit(1);
    }
}

/// Format bytes as human-readable string
fn human_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

/// Parse input shape specifications from CLI arguments
///
/// Format: "name:dim1,dim2,..." (e.g., "input_ids:1,77")
///
/// Returns a HashMap mapping input names to their dimensions.
fn parse_input_shapes(shape_specs: &[String]) -> anyhow::Result<HashMap<String, Vec<i64>>> {
    let mut shapes = HashMap::new();

    for spec in shape_specs {
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() != 2 {
            anyhow::bail!(
                "Invalid shape specification '{}'. Expected format: 'name:dim1,dim2,...'",
                spec
            );
        }

        let name = parts[0].to_string();
        let dims: Result<Vec<i64>, _> = parts[1]
            .split(',')
            .map(|d| d.trim().parse::<i64>())
            .collect();

        match dims {
            Ok(dimensions) => {
                if dimensions.is_empty() {
                    anyhow::bail!("Shape specification '{}' has no dimensions", spec);
                }
                if dimensions.iter().any(|&d| d <= 0) {
                    anyhow::bail!(
                        "Shape specification '{}' contains non-positive dimension",
                        spec
                    );
                }
                shapes.insert(name, dimensions);
            }
            Err(e) => {
                anyhow::bail!("Invalid dimension in shape specification '{}': {}", spec, e);
            }
        }
    }

    Ok(shapes)
}

async fn run_compile(args: CompileArgs) -> anyhow::Result<()> {
    use hologram_onnx_compiler::config::CompilerConfig;

    // Load config file (if specified or found in standard locations)
    let config = if let Some(config_path) = &args.config {
        // Use specified config file
        if args.verbose {
            println!("Loading config from: {}", config_path.display());
        }
        match CompilerConfig::load(config_path) {
            Ok(cfg) => Some(cfg),
            Err(e) => {
                eprintln!("Warning: Failed to load config file: {}", e);
                None
            }
        }
    } else {
        // Try to find config in standard locations
        match CompilerConfig::find_and_load() {
            Ok(Some(cfg)) => {
                if args.verbose {
                    println!("Loaded config from standard location");
                }
                Some(cfg)
            }
            Ok(None) => None,
            Err(e) => {
                if args.verbose {
                    println!("Note: No config file found ({})", e);
                }
                None
            }
        }
    };

    // Merge config with CLI args (CLI takes precedence)
    let merged = if let Some(cfg) = config {
        cfg.merge_with_cli(
            args.input.clone(),
            if args.memory_budget != 8192 {
                Some(args.memory_budget)
            } else {
                None
            },
            if (args.accuracy - 0.95).abs() > 0.001 {
                Some(args.accuracy)
            } else {
                None
            },
            if args.parallel { Some(true) } else { None },
            if args.verbose { Some(true) } else { None },
            args.debug_export.clone(),
            args.checkpoint_dir.clone(),
            if args.checkpoint_interval != 10 {
                Some(args.checkpoint_interval)
            } else {
                None
            },
        )
    } else {
        // No config file, use CLI args with defaults
        CompilerConfig::default().merge_with_cli(
            args.input.clone(),
            Some(args.memory_budget),
            Some(args.accuracy),
            Some(args.parallel),
            Some(args.verbose),
            args.debug_export.clone(),
            args.checkpoint_dir.clone(),
            Some(args.checkpoint_interval),
        )
    };

    // Use merged config for all settings
    let memory_budget = merged.memory_budget;
    let accuracy = merged.accuracy;
    let parallel = merged.parallel;
    let verbose = merged.verbose;
    let _debug_export = merged.debug_export; // Not used in simplified compiler yet

    // Validate arguments
    if !(0.0..=1.0).contains(&accuracy) {
        anyhow::bail!("Accuracy must be between 0.0 and 1.0");
    }

    // Get input from merged config (output is always from CLI)
    let input_path = merged.input.or(args.input);
    let output_path = args.output;

    // Validate input is specified
    if input_path.is_none() && args.hf_model.is_none() {
        anyhow::bail!("Either --input or --hf-model must be specified (or set in config file)");
    }

    // Determine input source
    let input_file_path = if let Some(input_path) = input_path {
        input_path
    } else if let Some(hf_model) = args.hf_model {
        // Parse HuggingFace model specification
        let (repo_id, filename) = parse_hf_model_spec(&hf_model)
            .with_context(|| format!("Invalid HuggingFace model specification: {}", hf_model))?;

        if verbose {
            println!("Downloading model from HuggingFace Hub...");
            println!("  Repository: {}", repo_id);
            if let Some(ref fname) = filename {
                println!("  File: {}", fname);
            } else {
                println!("  File: (auto-detect)");
            }
        }

        // Download the model
        let model_path = download_onnx_model(&repo_id, filename.as_deref(), None)
            .await
            .with_context(|| format!("Failed to download model from {}", repo_id))?;

        if verbose {
            println!("  Downloaded to: {}", model_path.display());
            println!();
        }

        model_path
    } else {
        anyhow::bail!("Either --input or --hf-model must be specified");
    };

    // Ensure output has .holo extension
    let output_path = if output_path.extension().is_none_or(|e| e != "holo") {
        output_path.with_extension("holo")
    } else {
        output_path
    };

    // Check system memory and warn if budget is too high
    if let Some(available_mb) = get_available_memory_mb() {
        if memory_budget > available_mb {
            eprintln!(
                "⚠ WARNING: Memory budget ({} MB) exceeds available system memory ({} MB)",
                memory_budget, available_mb
            );
            eprintln!("⚠ Compilation may fail due to out-of-memory (OOM).");
            eprintln!(
                "⚠ Recommended: Set --memory-budget to at most {} MB",
                available_mb / 2
            );
            eprintln!();
        }
    }

    // Parse input shapes if provided
    let input_shapes = if !args.input_shapes.is_empty() {
        Some(parse_input_shapes(&args.input_shapes)?)
    } else {
        None
    };

    // Build compiler
    let mut compiler = Compiler::new()
        .with_memory_budget(memory_budget)
        .with_verbose(verbose)
        .with_parallel(parallel);

    if let Some(shapes) = input_shapes {
        compiler = compiler.with_input_shapes(shapes);
    }

    // Add SafeTensors weights if provided
    if !args.weights.is_empty() {
        if verbose {
            println!(
                "Will load weights from {} SafeTensors file(s)",
                args.weights.len()
            );
            for weight_path in &args.weights {
                println!("  • {}", weight_path.display());
            }
            println!();
        }
        compiler = compiler.with_safetensors(args.weights.clone());
    }

    // Compile model
    let stats = compiler.compile(&input_file_path, &output_path)?;

    // Print success message
    println!("\n┌─────────────────────────────────────────────┐");
    println!("│ ✓ Compilation Successful!                   │");
    println!("└─────────────────────────────────────────────┘");
    println!();
    println!("Output:");
    println!("  .holo binary: {}", output_path.display());

    // Print binary size
    if let Ok(metadata) = std::fs::metadata(&output_path) {
        let size_mb = metadata.len() / (1024 * 1024);
        println!("  Binary size: {} MB ({} bytes)", size_mb, metadata.len());
    }

    println!("  Total time: {:.2}s", stats.compilation_time.as_secs_f64());
    println!();
    println!("Statistics:");
    println!(
        "  Operations: {} → {} ({:.1}% reduction)",
        stats.original_operations,
        stats.optimized_operations,
        100.0 * (1.0 - stats.optimized_operations as f64 / stats.original_operations.max(1) as f64)
    );
    println!("  Total patterns: {}", stats.total_patterns);
    println!(
        "  Hash table size: {:.2} MB",
        stats.hash_table_size as f64 / 1_000_000.0
    );
    println!(
        "  Address space size: {:.2} MB",
        stats.address_space_size as f64 / 1_000_000.0
    );
    println!();
    println!("Next steps:");
    println!("  1. Load .holo binary at: {}", output_path.display());
    println!("  2. Expected loading time: <100µs (native) or ~10ms (WASM)");
    println!("  3. Expected inference latency: O(1) lookup-based execution");

    Ok(())
}

async fn run_download(args: DownloadArgs) -> anyhow::Result<()> {
    use std::fs;

    // Parse model specification
    let (repo_id, filename) = parse_hf_model_spec(&args.model)
        .with_context(|| format!("Invalid HuggingFace model specification: {}", args.model))?;

    if args.verbose {
        println!("┌─────────────────────────────────────────────┐");
        println!("│ Downloading from HuggingFace Hub            │");
        println!("└─────────────────────────────────────────────┘");
        println!();
        println!("Repository: {}", repo_id);
        if let Some(ref fname) = filename {
            println!("File: {}", fname);
        } else {
            println!("Auto-detect: ONNX files");
        }
        println!();
    }

    // Download the model to output directory
    let repo_path = download_onnx_model(&repo_id, None, Some(&args.output)).await?;

    println!("┌─────────────────────────────────────────────┐");
    println!("│ ✓ Download Successful!                      │");
    println!("└─────────────────────────────────────────────┘");
    println!();
    println!("Repository downloaded to: {}", repo_path.display());
    println!();

    // Find all ONNX files in the downloaded repository
    let onnx_files = find_onnx_files(&repo_path)?;

    if onnx_files.is_empty() {
        println!("⚠ Warning: No ONNX files found in repository");
    } else {
        println!("Found {} ONNX file(s):", onnx_files.len());
        for onnx_file in &onnx_files {
            println!("  • {}", onnx_file.display());

            if let Ok(metadata) = fs::metadata(onnx_file) {
                let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
                println!("    ({:.2} MB)", size_mb);
            }
        }
        println!();

        // Show compilation command for the first ONNX file
        if let Some(first_onnx) = onnx_files.first() {
            println!("Next steps:");
            println!("  Compile the model:");
            println!(
                "    hologram-onnx compile --input {} --output model.holo --parallel",
                first_onnx.display()
            );
        }
    }

    Ok(())
}

/// Find all ONNX files in a directory (recursively)
fn find_onnx_files(dir: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    use std::fs;

    let mut onnx_files = Vec::new();

    fn visit_dir(dir: &std::path::Path, onnx_files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
        if !dir.is_dir() {
            return Ok(());
        }

        for entry in fs::read_dir(dir).context("Failed to read directory")? {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            if path.is_dir() {
                // Skip .git directory
                if path.file_name().and_then(|n| n.to_str()) != Some(".git") {
                    visit_dir(&path, onnx_files)?;
                }
            } else if let Some(ext) = path.extension() {
                if ext == "onnx" {
                    onnx_files.push(path);
                }
            }
        }
        Ok(())
    }

    visit_dir(dir, &mut onnx_files)?;
    Ok(onnx_files)
}

async fn run_convert(args: ConvertArgs) -> anyhow::Result<()> {
    println!("┌─────────────────────────────────────────────┐");
    println!("│ PyTorch to ONNX Converter                   │");
    println!("└─────────────────────────────────────────────┘");
    println!();

    // Setup Python environment automatically (unless disabled)
    if !args.no_auto_install {
        use hologram_onnx::converter::python_env;

        if args.verbose {
            println!("Checking Python environment...");
        }

        // This will create venv and install packages if needed
        match python_env::setup_python_environment(args.verbose).await {
            Ok(_) => {
                if args.verbose {
                    println!("✓ Python environment ready");
                    println!();
                }
            }
            Err(e) => {
                eprintln!("Warning: Could not setup Python environment: {}", e);
                eprintln!("Falling back to system Python...");
                eprintln!();
            }
        }
    }

    // Parse model type
    let model_type = match args.model_type.to_lowercase().as_str() {
        "stable-diffusion" | "sd" => ModelType::StableDiffusion,
        "bert" => ModelType::Bert,
        "gpt" | "gpt2" => ModelType::Gpt,
        "vit" | "vision" => ModelType::Vit,
        "auto" => ModelType::Auto,
        _ => {
            eprintln!("Unknown model type: {}", args.model_type);
            eprintln!("Supported types: auto, stable-diffusion, bert, gpt, vit");
            anyhow::bail!("Invalid model type");
        }
    };

    // Build conversion config
    let config = ConversionConfig {
        model_dir: args.model_dir.clone(),
        output_onnx: args.output.clone(),
        model_type,
        opset_version: args.opset,
        dynamic_axes: args.dynamic_axes,
        optimize: args.optimize,
        simplify: args.simplify,
        external_data: args.external_data,
        verbose: args.verbose,
    };

    println!("Converting PyTorch model to ONNX...");
    println!("  Model dir: {}", config.model_dir.display());
    println!("  Output: {}", config.output_onnx.display());
    println!("  Model type: {:?}", config.model_type);
    println!("  Opset version: {}", config.opset_version);
    println!();

    // Convert
    let onnx_path = convert_pytorch_to_onnx(&config)
        .await
        .context("Conversion failed")?;

    println!("┌─────────────────────────────────────────────┐");
    println!("│ ✓ Conversion Successful!                    │");
    println!("└─────────────────────────────────────────────┘");
    println!();
    println!("ONNX model saved to: {}", onnx_path.display());

    if let Ok(metadata) = tokio::fs::metadata(&onnx_path).await {
        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        println!("File size: {:.2} MB", size_mb);
    }

    println!();
    println!("Next steps:");
    println!("  1. Compile to .holo format:");
    println!("     hologram-onnx compile \\");
    println!("       --input {} \\", onnx_path.display());
    println!("       --output model.holo \\");
    println!("       --parallel --verbose");
    println!();

    // If SafeTensors exist alongside the model, suggest using them
    let safetensors_file = args.model_dir.join("model.safetensors");
    if safetensors_file.exists() {
        println!("  💡 Tip: SafeTensors weights detected!");
        println!("     You can also compile with separate weights:");
        println!("     hologram-onnx compile \\");
        println!("       --input {} \\", onnx_path.display());
        println!("       --weights {} \\", safetensors_file.display());
        println!("       --output model.holo");
        println!();
    }

    Ok(())
}

async fn run_auto_convert(args: AutoConvertArgs) -> anyhow::Result<()> {
    use hologram_onnx::converter::python_env;
    use hologram_onnx_downloader::{download_model_with_options, DownloadOptions, FileFilter};

    println!("┌─────────────────────────────────────────────┐");
    println!("│ Auto-Convert: Download + Convert to ONNX   │");
    println!("└─────────────────────────────────────────────┘");
    println!();

    // Parse HuggingFace repo
    let repo_id = args.repo.clone();

    if args.verbose {
        println!("Step 1: Downloading model from HuggingFace...");
        println!("  Repository: {}", repo_id);
        if let Some(ref component) = args.component {
            println!("  Component: {}", component);
        }
        println!("  Download to: {}", args.download_dir.display());
        println!();
    }

    // Create download directory
    tokio::fs::create_dir_all(&args.download_dir).await?;

    // Download model with PyTorch files (config.json, safetensors, etc.)
    // Note: We download all LFS files to ensure safetensors weights are properly downloaded
    let download_opts = DownloadOptions {
        file_filter: FileFilter::PyTorchModel, // Get PyTorch model files
        subdirs: None, // Download all subdirectories to ensure LFS files are available
        include_patterns: None,
        exclude_patterns: None,
        skip_lfs: false, // Download LFS files (safetensors weights)
        auto_convert_pytorch: false, // We'll convert manually
        convert_to_external: false,
        verbose: args.verbose,
    };

    if args.verbose {
        println!("  Downloading repository with LFS files...");
    }

    let repo_path =
        download_model_with_options(&repo_id, None, Some(&args.download_dir), &download_opts)
            .await
            .context("Failed to download model from HuggingFace")?;

    if args.verbose {
        println!("  ✓ Downloaded to: {}", repo_path.display());
        println!();
    }

    // Determine model directory for conversion
    let model_dir = if let Some(component) = &args.component {
        // User specified a component (e.g., "unet", "vae")
        repo_path.join(component)
    } else {
        // Use root model directory
        repo_path.clone()
    };

    // Verify model directory exists
    if !model_dir.exists() {
        anyhow::bail!(
            "Model directory does not exist: {}\nAvailable components might include: unet, vae, text_encoder",
            model_dir.display()
        );
    }

    // Verify safetensors files are actual binary files, not LFS pointers
    if args.verbose {
        println!("  Verifying downloaded files...");

        // Check for common safetensors files
        for filename in &["diffusion_pytorch_model.safetensors", "model.safetensors", "pytorch_model.safetensors"] {
            let safetensors_path = model_dir.join(filename);
            if safetensors_path.exists() {
                if let Ok(contents) = std::fs::read_to_string(&safetensors_path) {
                    if contents.starts_with("version https://git-lfs.github.com/spec/v1") {
                        eprintln!("  ⚠️  Warning: {} appears to be a Git LFS pointer file, not the actual model weights!", filename);
                        eprintln!("  This usually means Git LFS files weren't downloaded correctly.");
                        eprintln!("  The conversion will likely fail. Please ensure:");
                        eprintln!("  1. git-lfs is installed on your system");
                        eprintln!("  2. The download completed successfully");
                    } else {
                        println!("  ✓ {} verified ({})", filename, human_bytes(safetensors_path.metadata()?.len()));
                    }
                }
            }
        }
    }

    if args.verbose {
        println!("Step 2: Setting up Python environment...");
    }

    // Setup Python environment (auto-install packages)
    match python_env::setup_python_environment(args.verbose).await {
        Ok(_) => {
            if args.verbose {
                println!("  ✓ Python environment ready");
                println!();
            }
        }
        Err(e) => {
            eprintln!("Warning: Could not setup Python environment: {}", e);
            eprintln!("Falling back to system Python...");
            eprintln!();
        }
    }

    if args.verbose {
        println!("Step 3: Converting to ONNX...");
    }

    // Parse model type
    let model_type = match args.model_type.to_lowercase().as_str() {
        "stable-diffusion" | "sd" => ModelType::StableDiffusion,
        "bert" => ModelType::Bert,
        "gpt" | "gpt2" => ModelType::Gpt,
        "vit" | "vision" => ModelType::Vit,
        "auto" => ModelType::Auto,
        _ => {
            eprintln!("Unknown model type: {}", args.model_type);
            eprintln!("Supported types: auto, stable-diffusion, bert, gpt, vit");
            anyhow::bail!("Invalid model type");
        }
    };

    // Build conversion config
    let config = ConversionConfig {
        model_dir: model_dir.clone(),
        output_onnx: args.output.clone(),
        model_type,
        opset_version: args.opset,
        dynamic_axes: args.dynamic_axes,
        optimize: args.optimize,
        simplify: args.simplify,
        external_data: args.external_data,
        verbose: args.verbose,
    };

    // Convert to ONNX
    let onnx_path = convert_pytorch_to_onnx(&config)
        .await
        .context("Conversion failed")?;

    // Cleanup downloaded files if requested
    if !args.keep_download {
        if args.verbose {
            println!();
            println!("Step 4: Cleaning up downloaded files...");
        }

        if let Err(e) = tokio::fs::remove_dir_all(&repo_path).await {
            eprintln!("Warning: Could not remove downloaded files: {}", e);
        } else if args.verbose {
            println!("  ✓ Removed: {}", repo_path.display());
        }
    }

    println!();
    println!("┌─────────────────────────────────────────────┐");
    println!("│ ✓ Auto-Convert Successful!                  │");
    println!("└─────────────────────────────────────────────┘");
    println!();
    println!("ONNX model saved to: {}", onnx_path.display());

    if let Ok(metadata) = tokio::fs::metadata(&onnx_path).await {
        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        println!("File size: {:.2} MB", size_mb);
    }

    println!();
    println!("Next steps:");
    println!("  1. Compile to .holo format:");
    println!("     hologram-onnx compile \\");
    println!("       --input {} \\", onnx_path.display());
    println!("       --output model.holo \\");
    println!("       --parallel --verbose");
    println!();

    // If SafeTensors exist alongside the model, suggest using them
    let safetensors_file = model_dir.join("model.safetensors");
    if safetensors_file.exists() && args.keep_download {
        println!("  💡 Tip: SafeTensors weights detected!");
        println!("     You can also compile with separate weights:");
        println!("     hologram-onnx compile \\");
        println!("       --input {} \\", onnx_path.display());
        println!("       --weights {} \\", safetensors_file.display());
        println!("       --output model.holo");
        println!();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_args_parsing() {
        let cli = Cli::parse_from([
            "hologram-onnx",
            "compile",
            "--input",
            "model.onnx",
            "--output",
            "model.holo",
        ]);

        match cli.command {
            Commands::Compile(args) => {
                assert!(args.input.is_some());
                assert_eq!(args.output, PathBuf::from("model.holo"));
                assert_eq!(args.memory_budget, 8192);
                assert_eq!(args.accuracy, 0.95);
            }
            _ => panic!("Expected Compile command"),
        }
    }

    #[test]
    fn test_compile_with_hf_model() {
        let cli = Cli::parse_from([
            "hologram-onnx",
            "compile",
            "--hf-model",
            "deepseek-ai/DeepSeek-OCR",
            "--output",
            "model.holo",
        ]);

        match cli.command {
            Commands::Compile(args) => {
                assert!(args.input.is_none());
                assert_eq!(args.hf_model, Some("deepseek-ai/DeepSeek-OCR".to_string()));
            }
            _ => panic!("Expected Compile command"),
        }
    }

    #[test]
    fn test_download_args_parsing() {
        let cli = Cli::parse_from([
            "hologram-onnx",
            "download",
            "deepseek-ai/DeepSeek-OCR",
            "--output",
            "/tmp/models",
        ]);

        match cli.command {
            Commands::Download(args) => {
                assert_eq!(args.model, "deepseek-ai/DeepSeek-OCR");
                assert_eq!(args.output, PathBuf::from("/tmp/models"));
            }
            _ => panic!("Expected Download command"),
        }
    }

    #[test]
    fn test_parse_input_shapes() {
        let shapes = vec![
            "input_ids:1,77".to_string(),
            "attention_mask:1,77".to_string(),
        ];

        let result = parse_input_shapes(&shapes).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result.get("input_ids"), Some(&vec![1, 77]));
        assert_eq!(result.get("attention_mask"), Some(&vec![1, 77]));
    }

    #[test]
    fn test_parse_input_shapes_invalid() {
        let shapes = vec!["invalid".to_string()];
        assert!(parse_input_shapes(&shapes).is_err());

        let shapes = vec!["name:".to_string()];
        assert!(parse_input_shapes(&shapes).is_err());

        let shapes = vec!["name:-1,2".to_string()];
        assert!(parse_input_shapes(&shapes).is_err());
    }
}
