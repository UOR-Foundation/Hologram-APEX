//! Output formatting for compilation results

use anyhow::{Context, Result};
use camino::Utf8Path;
use colored::Colorize;
use hologram_backends::{Instruction, Program};
use std::fs;
use std::io::{self, Write};

use crate::cli::OutputFormat;
use crate::compiler::{CompilationResult, CompilationStats};

/// Output formatter for compilation results
pub struct OutputFormatter {
    /// Output format
    format: OutputFormat,

    /// Colorize output
    colorize: bool,
}

impl OutputFormatter {
    /// Create a new output formatter
    pub fn new(format: OutputFormat, colorize: bool) -> Self {
        Self { format, colorize }
    }

    /// Write compilation result to file
    pub fn write_to_file(&self, result: &CompilationResult, output_path: &Utf8Path) -> Result<()> {
        let content = match self.format {
            OutputFormat::Json => self.format_json(result)?,
            OutputFormat::Binary => {
                return Err(anyhow::anyhow!(
                    "Binary format not yet implemented - use JSON"
                ))
            }
            OutputFormat::Asm => self.format_asm(&result.program),
            OutputFormat::Circuit => {
                if let Some(ref circuit) = result.circuit {
                    circuit.clone()
                } else {
                    return Err(anyhow::anyhow!(
                        "Circuit representation not available (use --emit-circuit)"
                    ));
                }
            }
        };

        fs::write(output_path, content)
            .with_context(|| format!("Failed to write output to {}", output_path))?;

        Ok(())
    }

    /// Write compilation result to stdout
    pub fn write_to_stdout(&self, result: &CompilationResult) -> Result<()> {
        let content = match self.format {
            OutputFormat::Json => self.format_json(result)?,
            OutputFormat::Binary => {
                return Err(anyhow::anyhow!(
                    "Binary format cannot be written to stdout - use --output"
                ))
            }
            OutputFormat::Asm => self.format_asm(&result.program),
            OutputFormat::Circuit => {
                if let Some(ref circuit) = result.circuit {
                    circuit.clone()
                } else {
                    return Err(anyhow::anyhow!(
                        "Circuit representation not available (use --emit-circuit)"
                    ));
                }
            }
        };

        println!("{}", content);
        Ok(())
    }

    /// Format result as JSON
    fn format_json(&self, result: &CompilationResult) -> Result<String> {
        serde_json::to_string_pretty(result).context("Failed to serialize to JSON")
    }

    /// Format program as human-readable assembly
    fn format_asm(&self, program: &Program) -> String {
        let mut output = String::new();

        output.push_str("# Atlas ISA Assembly\n");
        output.push_str(&format!(
            "# {} instructions\n\n",
            program.instructions.len()
        ));

        for (i, instr) in program.instructions.iter().enumerate() {
            output.push_str(&format!("{:4}: {}\n", i, self.format_instruction(instr)));
        }

        output
    }

    /// Format a single instruction (simplified - just use Debug format)
    fn format_instruction(&self, instr: &Instruction) -> String {
        format!("{:?}", instr)
    }

    /// Print compilation statistics
    pub fn print_stats(&self, stats: &CompilationStats) -> Result<()> {
        let mut stdout = io::stdout();

        if self.colorize {
            writeln!(stdout, "\n{}", "Compilation Statistics:".bold().green())?;
        } else {
            writeln!(stdout, "\nCompilation Statistics:")?;
        }

        writeln!(stdout, "  Total time:        {} ms", stats.total_time_ms)?;
        writeln!(stdout, "  Circuit time:      {} ms", stats.circuit_time_ms)?;
        writeln!(stdout, "  ISA gen time:      {} ms", stats.isa_time_ms)?;
        writeln!(stdout, "  Circuit nodes:     {}", stats.circuit_nodes)?;
        writeln!(stdout, "  ISA instructions:  {}", stats.isa_instructions)?;
        writeln!(stdout, "  Canonicalized:     {}", stats.canonicalized)?;
        writeln!(stdout, "  Opt level:         {}", stats.opt_level)?;

        stdout.flush()?;
        Ok(())
    }

    /// Print success message
    pub fn print_success(&self, kernel_name: &str, output_path: Option<&Utf8Path>) {
        if self.colorize {
            print!("{} ", "✓".green().bold());
        } else {
            print!("✓ ");
        }

        if let Some(path) = output_path {
            println!("Compiled {} → {}", kernel_name, path);
        } else {
            println!("Compiled {}", kernel_name);
        }
    }

    /// Print error message
    #[allow(dead_code)]
    pub fn print_error(&self, message: &str) {
        if self.colorize {
            eprintln!("{} {}", "✗".red().bold(), message);
        } else {
            eprintln!("✗ {}", message);
        }
    }

    /// Print info message
    pub fn print_info(&self, message: &str) {
        if self.colorize {
            println!("{} {}", "ℹ".blue().bold(), message);
        } else {
            println!("ℹ {}", message);
        }
    }
}

/// Disassemble a compiled ISA program from JSON file
pub fn disassemble_file(input_path: &Utf8Path) -> Result<()> {
    let content = fs::read_to_string(input_path)
        .with_context(|| format!("Failed to read file: {}", input_path))?;

    let result: CompilationResult =
        serde_json::from_str(&content).context("Failed to parse JSON")?;

    let formatter = OutputFormatter::new(OutputFormat::Asm, true);

    println!("{}", "═".repeat(80).bright_blue());
    println!(
        "{} {}",
        "Kernel:".bold(),
        result.kernel_name.bright_yellow()
    );
    println!(
        "{} {} instructions",
        "Instructions:".bold(),
        result.stats.isa_instructions
    );
    println!("{}", "═".repeat(80).bright_blue());
    println!();

    let asm = formatter.format_asm(&result.program);
    println!("{}", asm);

    Ok(())
}

/// Show information about a compiled kernel
pub fn show_info(input_path: &Utf8Path) -> Result<()> {
    let content = fs::read_to_string(input_path)
        .with_context(|| format!("Failed to read file: {}", input_path))?;

    let result: CompilationResult =
        serde_json::from_str(&content).context("Failed to parse JSON")?;

    println!("{}", "═".repeat(80).bright_blue());
    println!(
        "{} {}",
        "Kernel:".bold(),
        result.kernel_name.bright_yellow()
    );
    println!("{}", "═".repeat(80).bright_blue());
    println!();

    if let Some(ref output) = result.output_path {
        println!("Output: {}", output);
    }

    println!();
    println!("{}", "Statistics:".bold().green());
    println!("  Total time:        {} ms", result.stats.total_time_ms);
    println!("  Circuit time:      {} ms", result.stats.circuit_time_ms);
    println!("  ISA gen time:      {} ms", result.stats.isa_time_ms);
    println!("  Circuit nodes:     {}", result.stats.circuit_nodes);
    println!("  ISA instructions:  {}", result.stats.isa_instructions);
    println!("  Canonicalized:     {}", result.stats.canonicalized);
    println!("  Opt level:         {}", result.stats.opt_level);

    if let Some(ref circuit) = result.circuit {
        println!();
        println!("{}", "Circuit:".bold().cyan());
        println!("  {}", circuit);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_formatter_creation() {
        let formatter = OutputFormatter::new(OutputFormat::Json, true);
        assert!(matches!(formatter.format, OutputFormat::Json));
        assert!(formatter.colorize);
    }
}
