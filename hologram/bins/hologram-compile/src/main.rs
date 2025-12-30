//! Hologram Compiler - CLI tool for compiling Python kernels to Atlas ISA

mod cli;
mod compiler;
mod output;

use anyhow::{Context, Result};
use camino::Utf8PathBuf;
use clap::Parser;
use std::process;

use cli::{Cli, Command};
use compiler::Compiler;
use output::OutputFormatter;

fn main() {
    // Parse command-line arguments
    let cli = Cli::parse();

    // Initialize logging
    cli.init_logging();

    // Run the appropriate command
    if let Err(e) = run(cli) {
        eprintln!("Error: {:#}", e);
        process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    // Create output formatter
    let formatter = OutputFormatter::new(cli.format, !cli.quiet && atty::is(atty::Stream::Stdout));

    // Get effective command (including implicit compile)
    let command = cli
        .effective_command()
        .context("No input file or command specified. Use --help for usage information.")?;

    // Execute command
    match command {
        Command::Compile { input, output } => {
            compile_single(&cli, &formatter, input, output)?;
        }
        Command::CompileAll {
            input_dir,
            output_dir,
        } => {
            compile_directory(&cli, &formatter, input_dir, output_dir)?;
        }
        Command::Check { input } => {
            check_single(&cli, &formatter, input)?;
        }
        Command::Info { input } => {
            show_info(&formatter, input)?;
        }
        Command::Disasm { input } => {
            disassemble(&formatter, input)?;
        }
    }

    Ok(())
}

/// Compile a single kernel
fn compile_single(
    cli: &Cli,
    formatter: &OutputFormatter,
    input: std::path::PathBuf,
    output: Option<std::path::PathBuf>,
) -> Result<()> {
    let input_path = Utf8PathBuf::try_from(input.clone())
        .with_context(|| format!("Invalid UTF-8 path: {:?}", input))?;

    if !input_path.exists() {
        anyhow::bail!("Input file not found: {}", input_path);
    }

    // Create compiler
    let mut compiler = Compiler::new(cli.backend, cli.opt_level)
        .with_no_canonicalize(cli.no_canonicalize)
        .with_emit_circuit(cli.emit_circuit || cli.format == cli::OutputFormat::Circuit);

    // Compile
    let mut result = compiler.compile_file(&input_path)?;

    // Determine output path
    let output_path = if let Some(out) = output {
        Some(
            Utf8PathBuf::try_from(out.clone())
                .with_context(|| format!("Invalid UTF-8 path: {:?}", out))?,
        )
    } else if cli.format != cli::OutputFormat::Asm {
        // Default output: input.json
        let mut default_out = input_path.clone();
        default_out.set_extension(match cli.format {
            cli::OutputFormat::Json => "json",
            cli::OutputFormat::Binary => "bin",
            cli::OutputFormat::Circuit => "circuit",
            cli::OutputFormat::Asm => "asm",
        });
        Some(default_out)
    } else {
        None
    };

    // Write output
    if let Some(ref path) = output_path {
        formatter.write_to_file(&result, path)?;
        result.output_path = Some(path.to_owned());
    } else {
        // Write to stdout for asm format
        formatter.write_to_stdout(&result)?;
    }

    // Print success message
    if !cli.quiet {
        formatter.print_success(&result.kernel_name, output_path.as_deref());

        // Print statistics if requested
        if cli.stats {
            formatter.print_stats(&result.stats)?;
        }
    }

    Ok(())
}

/// Compile all kernels in a directory
fn compile_directory(
    cli: &Cli,
    formatter: &OutputFormatter,
    input_dir: std::path::PathBuf,
    output_dir: Option<std::path::PathBuf>,
) -> Result<()> {
    let input_path = Utf8PathBuf::try_from(input_dir.clone())
        .with_context(|| format!("Invalid UTF-8 path: {:?}", input_dir))?;

    if !input_path.exists() {
        anyhow::bail!("Input directory not found: {}", input_path);
    }

    let output_path = if let Some(out) = output_dir {
        Some(
            Utf8PathBuf::try_from(out.clone())
                .with_context(|| format!("Invalid UTF-8 path: {:?}", out))?,
        )
    } else {
        None
    };

    // Create output directory if it doesn't exist
    if let Some(ref out_dir) = output_path {
        std::fs::create_dir_all(out_dir)
            .with_context(|| format!("Failed to create output directory: {}", out_dir))?;
    }

    // Create compiler
    let mut compiler = Compiler::new(cli.backend, cli.opt_level)
        .with_no_canonicalize(cli.no_canonicalize)
        .with_emit_circuit(cli.emit_circuit);

    // Compile all kernels
    let results = compiler.compile_directory(&input_path, output_path.as_deref())?;

    // Write results
    for result in &results {
        if let Some(ref path) = result.output_path {
            formatter.write_to_file(result, path)?;

            if !cli.quiet {
                formatter.print_success(&result.kernel_name, Some(path));
            }
        }
    }

    // Print summary
    if !cli.quiet {
        println!();
        formatter.print_info(&format!("Compiled {} kernels", results.len()));

        if cli.stats {
            let total_time: u64 = results.iter().map(|r| r.stats.total_time_ms).sum();
            let total_instructions: usize = results.iter().map(|r| r.stats.isa_instructions).sum();

            println!("  Total time:        {} ms", total_time);
            println!("  Total instructions: {}", total_instructions);
        }
    }

    Ok(())
}

/// Check if a kernel compiles
fn check_single(cli: &Cli, _formatter: &OutputFormatter, input: std::path::PathBuf) -> Result<()> {
    let input_path = Utf8PathBuf::try_from(input.clone())
        .with_context(|| format!("Invalid UTF-8 path: {:?}", input))?;

    if !input_path.exists() {
        anyhow::bail!("Input file not found: {}", input_path);
    }

    let mut compiler =
        Compiler::new(cli.backend, cli.opt_level).with_no_canonicalize(cli.no_canonicalize);

    let success = compiler.check_file(&input_path)?;

    if !success {
        process::exit(1);
    }

    Ok(())
}

/// Show information about a compiled kernel
fn show_info(_formatter: &OutputFormatter, input: std::path::PathBuf) -> Result<()> {
    let input_path = Utf8PathBuf::try_from(input.clone())
        .with_context(|| format!("Invalid UTF-8 path: {:?}", input))?;

    if !input_path.exists() {
        anyhow::bail!("Input file not found: {}", input_path);
    }

    output::show_info(&input_path)
}

/// Disassemble a compiled ISA program
fn disassemble(_formatter: &OutputFormatter, input: std::path::PathBuf) -> Result<()> {
    let input_path = Utf8PathBuf::try_from(input.clone())
        .with_context(|| format!("Invalid UTF-8 path: {:?}", input))?;

    if !input_path.exists() {
        anyhow::bail!("Input file not found: {}", input_path);
    }

    output::disassemble_file(&input_path)
}
