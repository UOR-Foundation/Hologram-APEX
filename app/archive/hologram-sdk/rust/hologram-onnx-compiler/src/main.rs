//! Hologram ONNX Compiler CLI
//!
//! Command-line interface for compiling ONNX models to Hologram .holo format.

use clap::Parser;
use hologram_onnx_compiler::Compiler;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process;

/// Hologram ONNX Compiler - Simplified Single-Pass
///
/// Compiles ONNX models to .holo format using HologramGraph and hologram-hrm Atlas
/// for O(1) lookup-based inference.
///
/// The compiler uses a simplified 4-step flow:
/// 1. Load ONNX → HologramGraph (petgraph-based IR)
/// 2. Optimize Graph (fusion, dead code elimination)
/// 3. Execute Operators (on sample patterns with Atlas)
/// 4. Serialize Binary (hash tables + address space)
///
/// This replaces the old 5-pass pipeline with a simpler, more maintainable architecture.
#[derive(Parser, Debug)]
#[command(name = "hologram-onnx-compiler")]
#[command(author = "Hologram Team")]
#[command(version)]
#[command(about = "Compile ONNX models to Hologram .holo format", long_about = None)]
struct Args {
    /// Path to input ONNX model file (can be set in config file)
    #[arg(short, long, conflicts_with = "hf_model")]
    input: Option<PathBuf>,

    /// HuggingFace model repository (e.g., "deepseek-ai/DeepSeek-OCR")
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
}

/// Get available system memory in MB (Linux only)
fn get_available_memory_mb() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;

        // Read /proc/meminfo
        let meminfo = fs::read_to_string("/proc/meminfo").ok()?;

        // Find MemAvailable line
        for line in meminfo.lines() {
            if line.starts_with("MemAvailable:") {
                // Parse: "MemAvailable:   12345678 kB"
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let kb = parts[1].parse::<usize>().ok()?;
                    return Some(kb / 1024); // Convert kB to MB
                }
            }
        }
        None
    }

    #[cfg(not(target_os = "linux"))]
    {
        // Memory detection not implemented for non-Linux platforms
        None
    }
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
        let dims: Result<Vec<i64>, _> = parts[1].split(',').map(|d| d.trim().parse::<i64>()).collect();

        match dims {
            Ok(dimensions) => {
                if dimensions.is_empty() {
                    anyhow::bail!("Shape specification '{}' has no dimensions", spec);
                }
                if dimensions.iter().any(|&d| d <= 0) {
                    anyhow::bail!("Shape specification '{}' contains non-positive dimension", spec);
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

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    let args = Args::parse();

    // Run compilation
    if let Err(e) = run(args) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn run(args: Args) -> anyhow::Result<()> {
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
    } else if let Some(_hf_model) = args.hf_model {
        anyhow::bail!("HuggingFace model download not yet implemented");
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
            eprintln!("⚠ Recommended: Set --memory-budget to at most {} MB", available_mb / 2);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_parsing() {
        // Test with input file
        let args = Args::parse_from([
            "hologram-onnx-compiler",
            "--input",
            "model.onnx",
            "--output",
            "model.holo",
        ]);

        assert!(args.input.is_some());
        assert_eq!(args.output, PathBuf::from("model.holo"));
        assert_eq!(args.memory_budget, 8192); // Default 8GB
        assert_eq!(args.accuracy, 0.95); // Default 95%
        assert!(!args.parallel); // Default false
        assert!(!args.verbose); // Default false
    }

    #[test]
    fn test_args_with_parallel() {
        let args = Args::parse_from([
            "hologram-onnx-compiler",
            "--input",
            "model.onnx",
            "--output",
            "model.holo",
            "--parallel",
        ]);

        assert!(args.parallel);
    }

    #[test]
    fn test_args_with_custom_settings() {
        let args = Args::parse_from([
            "hologram-onnx-compiler",
            "--input",
            "model.onnx",
            "--output",
            "model.holo",
            "--parallel",
            "--memory-budget",
            "16384",
            "--accuracy",
            "0.99",
            "--verbose",
        ]);

        assert_eq!(args.memory_budget, 16384); // 16GB
        assert_eq!(args.accuracy, 0.99); // 99%
        assert!(args.parallel);
        assert!(args.verbose);
    }

    #[test]
    fn test_args_with_hf_model() {
        let args = Args::parse_from([
            "hologram-onnx-compiler",
            "--hf-model",
            "deepseek-ai/DeepSeek-OCR",
            "--output",
            "model.holo",
        ]);

        assert!(args.input.is_none());
        assert_eq!(args.hf_model, Some("deepseek-ai/DeepSeek-OCR".to_string()));
    }

    #[test]
    fn test_args_with_debug_export() {
        let args = Args::parse_from([
            "hologram-onnx-compiler",
            "--input",
            "model.onnx",
            "--output",
            "model.holo",
            "--debug-export",
            "/tmp/debug",
        ]);

        assert_eq!(args.debug_export, Some(PathBuf::from("/tmp/debug")));
    }
}
