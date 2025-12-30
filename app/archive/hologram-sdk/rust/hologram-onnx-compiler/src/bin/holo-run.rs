//! Hologram Model Runner
//!
//! Loads and executes compiled .holo models
//!
//! # Usage
//!
//! ```bash
//! # Show model information
//! holo-run info model.holo
//!
//! # Run inference with JSON input
//! holo-run run model.holo --input input.json --output output.json
//!
//! # Run inference with generated test data
//! holo-run test model.holo
//! ```

use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

/// Hologram Model Runner - Execute compiled .holo models
#[derive(Parser)]
#[command(name = "holo-run")]
#[command(about = "Run compiled Hologram models", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show model information
    Info {
        /// Path to compiled .holo model
        model: PathBuf,
    },

    /// Run inference
    Run {
        /// Path to compiled .holo model
        model: PathBuf,

        /// Path to input JSON file
        #[arg(short, long)]
        input: Option<PathBuf>,

        /// Path to output JSON file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Test model with generated data
    Test {
        /// Path to compiled .holo model
        model: PathBuf,

        /// Number of test runs
        #[arg(short, long, default_value = "10")]
        iterations: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info { model } => {
            if let Err(e) = show_info(&model) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Run { model, input, output } => {
            if let Err(e) = run_inference(&model, input.as_deref(), output.as_deref()) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Test { model, iterations } => {
            if let Err(e) = test_model(&model, iterations) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

/// Show model information
fn show_info(model_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading model: {}", model_path.display());

    let model = load_model(model_path)?;

    println!("\n┌─────────────────────────────────────────────┐");
    println!("│ Model Information                            │");
    println!("└─────────────────────────────────────────────┘\n");

    println!("File size: {:.2} MB", model.total_bytes as f64 / 1_000_000.0);
    println!("Format version: {}", model.version);
    println!();
    println!("Sections:");
    println!(
        "  Manifest:      {:.2} MB ({} bytes)",
        model.manifest_size as f64 / 1_000_000.0,
        model.manifest_size
    );
    println!(
        "  Address space: {:.2} MB ({} bytes)",
        model.address_space_size as f64 / 1_000_000.0,
        model.address_space_size
    );
    println!(
        "  Hash tables:   {:.2} MB ({} bytes)",
        model.hash_tables_size as f64 / 1_000_000.0,
        model.hash_tables_size
    );
    println!(
        "  Metadata:      {:.2} MB ({} bytes)",
        model.metadata_size as f64 / 1_000_000.0,
        model.metadata_size
    );
    println!();

    // Parse manifest
    if let Ok(manifest) = serde_json::from_slice::<serde_json::Value>(&model.manifest_data) {
        if let Some(op_stats) = manifest.get("operation_stats").and_then(|v| v.as_array()) {
            println!("Operations: {}", op_stats.len());
            println!();
            println!("Operation breakdown:");
            for (i, op) in op_stats.iter().enumerate() {
                let op_type = op.get("op_type").and_then(|v| v.as_str()).unwrap_or("Unknown");
                let input_shapes = op
                    .get("input_shapes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().map(|s| format!("{:?}", s)).collect::<Vec<_>>().join(", "))
                    .unwrap_or_default();
                let output_shapes = op
                    .get("output_shapes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().map(|s| format!("{:?}", s)).collect::<Vec<_>>().join(", "))
                    .unwrap_or_default();

                println!("  {}: {} ({}) → ({})", i, op_type, input_shapes, output_shapes);
            }
        }

        if let Some(patterns) = manifest.get("patterns_per_operation").and_then(|v| v.as_array()) {
            let total: i64 = patterns.iter().filter_map(|v| v.as_i64()).sum();
            println!();
            println!("Total pre-computed patterns: {}", total);
        }
    }

    Ok(())
}

/// Run inference
fn run_inference(
    model_path: &PathBuf,
    input_path: Option<&std::path::Path>,
    output_path: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading model: {}", model_path.display());
    let model = load_model(model_path)?;

    // Load or generate input
    let input_data: Vec<f32> = if let Some(input_path) = input_path {
        println!("Loading input: {}", input_path.display());
        let input_json = fs::read_to_string(input_path)?;
        serde_json::from_str(&input_json)?
    } else {
        println!("Generating test input...");
        generate_test_input(256)
    };

    println!("Running inference...");
    let output = execute_model(&model, &input_data)?;

    // Save or display output
    if let Some(output_path) = output_path {
        println!("Saving output: {}", output_path.display());
        let output_json = serde_json::to_string_pretty(&output)?;
        fs::write(output_path, output_json)?;
    } else {
        println!("\nOutput ({} values):", output.len());
        println!("{:?}", &output[..output.len().min(10)]);
        if output.len() > 10 {
            println!("... ({} more values)", output.len() - 10);
        }
    }

    println!("\n✓ Inference complete!");

    Ok(())
}

/// Test model with multiple iterations
fn test_model(model_path: &PathBuf, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading model: {}", model_path.display());
    let model = load_model(model_path)?;

    println!("Running {} test iterations...\n", iterations);

    let mut total_time = std::time::Duration::ZERO;

    for i in 0..iterations {
        let input = generate_test_input(256);

        let start = std::time::Instant::now();
        let _output = execute_model(&model, &input)?;
        let elapsed = start.elapsed();

        total_time += elapsed;

        println!("Iteration {}: {:?}", i + 1, elapsed);
    }

    let avg_time = total_time / iterations as u32;
    println!("\n┌─────────────────────────────────────────────┐");
    println!("│ Performance Summary                          │");
    println!("└─────────────────────────────────────────────┘");
    println!();
    println!("Total time:   {:?}", total_time);
    println!("Average time: {:?}", avg_time);
    println!("Throughput:   {:.2} inferences/sec", 1.0 / avg_time.as_secs_f64());

    Ok(())
}

/// Loaded model structure
struct HoloModel {
    version: u32,
    manifest_data: Vec<u8>,
    address_space_data: Vec<u8>,
    hash_tables_data: Vec<u8>,
    #[allow(dead_code)]
    metadata_data: Vec<u8>,
    manifest_size: usize,
    address_space_size: usize,
    hash_tables_size: usize,
    metadata_size: usize,
    total_bytes: usize,
}

/// Load .holo model from disk
fn load_model(path: &PathBuf) -> Result<HoloModel, Box<dyn std::error::Error>> {
    let mut file = fs::File::open(path)?;

    // Read entire file
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    // Parse header (72 bytes)
    if data.len() < 72 {
        return Err("Invalid .holo file: too small".into());
    }

    // Check magic
    if &data[0..4] != b"HOLO" {
        return Err("Invalid .holo file: bad magic".into());
    }

    // Read version
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

    // Read section offsets (8 bytes each)
    let manifest_offset = u64::from_le_bytes(data[8..16].try_into()?) as usize;
    let address_space_offset = u64::from_le_bytes(data[16..24].try_into()?) as usize;
    let hash_tables_offset = u64::from_le_bytes(data[24..32].try_into()?) as usize;
    let metadata_offset = u64::from_le_bytes(data[32..40].try_into()?) as usize;

    // Read section sizes (8 bytes each)
    let manifest_size = u64::from_le_bytes(data[40..48].try_into()?) as usize;
    let address_space_size = u64::from_le_bytes(data[48..56].try_into()?) as usize;
    let hash_tables_size = u64::from_le_bytes(data[56..64].try_into()?) as usize;
    let metadata_size = u64::from_le_bytes(data[64..72].try_into()?) as usize;

    // Extract sections
    let manifest_data = data[manifest_offset..manifest_offset + manifest_size].to_vec();
    let address_space_data = data[address_space_offset..address_space_offset + address_space_size].to_vec();
    let hash_tables_data = data[hash_tables_offset..hash_tables_offset + hash_tables_size].to_vec();
    let metadata_data = data[metadata_offset..metadata_offset + metadata_size].to_vec();

    Ok(HoloModel {
        version,
        manifest_data,
        address_space_data,
        hash_tables_data,
        metadata_data,
        manifest_size,
        address_space_size,
        hash_tables_size,
        metadata_size,
        total_bytes: data.len(),
    })
}

/// Execute model on input data
fn execute_model(model: &HoloModel, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use std::hash::{Hash, Hasher};

    // Parse hash tables from JSON
    let hash_tables: Vec<HashMap<u64, ExtendedAddress>> = serde_json::from_slice(&model.hash_tables_data)?;

    if hash_tables.is_empty() {
        return Err("No hash tables in model".into());
    }

    // Hash the input pattern using ahash (same as compilation)
    let mut hasher = ahash::AHasher::default();
    for &val in input {
        val.to_bits().hash(&mut hasher);
    }
    let input_hash = hasher.finish();

    // Lookup in hash table (using first operation)
    let first_table = &hash_tables[0];
    if let Some(address) = first_table.get(&input_hash) {
        // Calculate byte offset from ExtendedAddress
        let offset = calculate_address_offset(address);

        // Read f32 values from address space
        let start = offset * 4; // f32 = 4 bytes
        let end = (start + (input.len() * 4)).min(model.address_space_data.len());

        let mut result = Vec::new();
        for chunk in model.address_space_data[start..end].chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into()?;
            result.push(f32::from_le_bytes(bytes));
        }

        Ok(result)
    } else {
        // Pattern not found in hash table
        Err(format!("Input pattern (hash: {}) not found in pre-computed results", input_hash).into())
    }
}

/// Extended address for hash table lookups
#[derive(Debug, Clone, serde::Deserialize)]
struct ExtendedAddress {
    class: u8,
    page: u8,
    byte: u8,
    sub_index: usize,
}

/// Calculate byte offset from ExtendedAddress
fn calculate_address_offset(address: &ExtendedAddress) -> usize {
    // Address space layout: class → page → byte → sub_index
    // Total size per class: 48 pages × 256 bytes = 12,288 bytes
    const PAGES_PER_CLASS: usize = 48;
    const BYTES_PER_PAGE: usize = 256;

    let class_offset = address.class as usize * PAGES_PER_CLASS * BYTES_PER_PAGE;
    let page_offset = address.page as usize * BYTES_PER_PAGE;
    let byte_offset = address.byte as usize;

    class_offset + page_offset + byte_offset + address.sub_index
}

/// Hash input pattern for lookup
#[allow(dead_code)]
fn hash_input(input: &[f32]) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for &val in input {
        val.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Generate test input data
fn generate_test_input(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32 * 0.01) % 2.0 - 1.0).collect()
}
