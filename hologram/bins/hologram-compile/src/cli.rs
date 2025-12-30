//! Command-line argument parsing for hologram-compile

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// Hologram Compiler - Compile Python kernels to Atlas ISA
#[derive(Parser, Debug)]
#[command(name = "hologram-compile")]
#[command(author, version, about = "Hologram Compiler - Compile Python kernels to Atlas ISA", long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Input Python kernel file or directory
    #[arg(value_name = "INPUT")]
    pub input: Option<PathBuf>,

    /// Output file or directory
    #[arg(short, long, value_name = "OUTPUT")]
    pub output: Option<PathBuf>,

    /// Output format
    #[arg(short = 'f', long, value_enum, default_value = "json")]
    pub format: OutputFormat,

    /// Target backend
    #[arg(short, long, value_enum, default_value = "cpu")]
    pub backend: Backend,

    /// Optimization level (0-3)
    #[arg(short = 'O', long, default_value = "2")]
    pub opt_level: u8,

    /// Enable verbose output
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Suppress all output except errors
    #[arg(short, long)]
    pub quiet: bool,

    /// Emit human-readable ISA assembly
    #[arg(long)]
    pub emit_asm: bool,

    /// Emit Circuit representation before ISA
    #[arg(long)]
    pub emit_circuit: bool,

    /// Skip canonicalization (for debugging)
    #[arg(long)]
    pub no_canonicalize: bool,

    /// Print compilation statistics
    #[arg(long)]
    pub stats: bool,

    /// Watch mode: recompile on file changes
    #[arg(short, long)]
    pub watch: bool,

    /// Subcommand
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    /// Compile a single kernel
    Compile {
        /// Input Python kernel file
        input: PathBuf,

        /// Output ISA file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Compile all kernels in a directory
    CompileAll {
        /// Input directory containing Python kernels
        input_dir: PathBuf,

        /// Output directory for compiled ISA files
        #[arg(short, long)]
        output_dir: Option<PathBuf>,
    },

    /// Check if a kernel compiles without generating output
    Check {
        /// Input Python kernel file
        input: PathBuf,
    },

    /// Show information about a compiled kernel
    Info {
        /// Compiled ISA file
        input: PathBuf,
    },

    /// Disassemble a compiled ISA program
    Disasm {
        /// Compiled ISA file (JSON)
        input: PathBuf,
    },
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// JSON format (default)
    Json,
    /// Binary format
    Binary,
    /// Human-readable assembly
    Asm,
    /// Circuit representation
    Circuit,
}

#[derive(ValueEnum, Debug, Clone, Copy)]
pub enum Backend {
    /// CPU backend (SIMD)
    Cpu,
    /// CUDA backend
    Cuda,
    /// Metal backend (macOS/iOS)
    Metal,
    /// WebGPU backend (WASM)
    Webgpu,
}

impl Backend {
    /// Convert to hologram_backends::BackendType
    #[allow(dead_code)]
    pub fn to_backend_type(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Webgpu => "webgpu",
        }
    }
}

impl Cli {
    /// Initialize logging based on verbosity level
    pub fn init_logging(&self) {
        use tracing_subscriber::{fmt, EnvFilter};

        if self.quiet {
            return;
        }

        let level = match self.verbose {
            0 => "warn",
            1 => "info",
            2 => "debug",
            _ => "trace",
        };

        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

        fmt().with_env_filter(filter).with_target(false).init();
    }

    /// Get the effective command (including implicit compile)
    pub fn effective_command(&self) -> Option<Command> {
        if let Some(ref cmd) = self.command {
            return Some(cmd.clone());
        }

        // If no subcommand, treat as implicit compile if input is provided
        self.input.as_ref().map(|input| Command::Compile {
            input: input.clone(),
            output: self.output.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing_basic() {
        let cli = Cli::parse_from(["hologram-compile", "kernel.py"]);
        assert_eq!(cli.input, Some(PathBuf::from("kernel.py")));
        assert!(matches!(cli.format, OutputFormat::Json));
        assert!(matches!(cli.backend, Backend::Cpu));
    }

    #[test]
    fn test_cli_parsing_with_options() {
        let cli = Cli::parse_from([
            "hologram-compile",
            "kernel.py",
            "-o",
            "output.json",
            "-f",
            "json",
            "-b",
            "cuda",
            "-O",
            "3",
            "-vv",
        ]);

        assert_eq!(cli.output, Some(PathBuf::from("output.json")));
        assert!(matches!(cli.format, OutputFormat::Json));
        assert!(matches!(cli.backend, Backend::Cuda));
        assert_eq!(cli.opt_level, 3);
        assert_eq!(cli.verbose, 2);
    }

    #[test]
    fn test_backend_conversion() {
        assert_eq!(Backend::Cpu.to_backend_type(), "cpu");
        assert_eq!(Backend::Cuda.to_backend_type(), "cuda");
        assert_eq!(Backend::Metal.to_backend_type(), "metal");
        assert_eq!(Backend::Webgpu.to_backend_type(), "webgpu");
    }
}
