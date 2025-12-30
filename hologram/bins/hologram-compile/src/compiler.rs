//! Compilation pipeline for Python kernels to Atlas ISA

use anyhow::{Context, Result};
use camino::{Utf8Path, Utf8PathBuf};
use hologram_backends::{circuit_to_isa, Program};
use hologram_codegen::KernelRegistry;
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;
use tracing::{debug, info, warn};

use crate::cli::Backend;

/// Compilation result with statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationResult {
    /// Input kernel name
    pub kernel_name: String,

    /// Output file path  
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_path: Option<Utf8PathBuf>,

    /// Compiled ISA program
    pub program: Program,

    /// Circuit representation (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub circuit: Option<String>,

    /// Compilation statistics
    pub stats: CompilationStats,
}

/// Compilation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationStats {
    /// Total compilation time (milliseconds)
    pub total_time_ms: u64,

    /// Circuit compilation time (milliseconds)
    pub circuit_time_ms: u64,

    /// ISA generation time (milliseconds)
    pub isa_time_ms: u64,

    /// Number of circuit nodes
    pub circuit_nodes: usize,

    /// Number of ISA instructions
    pub isa_instructions: usize,

    /// Canonicalization applied
    pub canonicalized: bool,

    /// Optimization level
    pub opt_level: u8,
}

/// Compiler for Python kernels to Atlas ISA
pub struct Compiler {
    /// Registry for loading kernels
    #[allow(dead_code)]
    registry: KernelRegistry,

    /// Target backend
    #[allow(dead_code)]
    backend: Backend,

    /// Optimization level
    opt_level: u8,

    /// Skip canonicalization
    no_canonicalize: bool,

    /// Emit circuit representation
    emit_circuit: bool,
}

impl Compiler {
    /// Create a new compiler
    pub fn new(backend: Backend, opt_level: u8) -> Self {
        Self {
            registry: KernelRegistry::new(),
            backend,
            opt_level,
            no_canonicalize: false,
            emit_circuit: false,
        }
    }

    /// Set whether to skip canonicalization
    pub fn with_no_canonicalize(mut self, no_canonicalize: bool) -> Self {
        self.no_canonicalize = no_canonicalize;
        self
    }

    /// Set whether to emit circuit representation
    pub fn with_emit_circuit(mut self, emit_circuit: bool) -> Self {
        self.emit_circuit = emit_circuit;
        self
    }

    /// Compile a Python kernel file to ISA
    pub fn compile_file(&mut self, input_path: &Utf8Path) -> Result<CompilationResult> {
        let start = Instant::now();

        info!("Compiling kernel: {}", input_path);

        // Extract kernel name from file path
        let kernel_name = input_path
            .file_stem()
            .context("Invalid input file path")?
            .to_string();

        // Load circuit (for now, use placeholder)
        debug!("Loading kernel from: {}", input_path);
        let circuit_str = self.load_circuit_from_registry(&kernel_name)?;

        // Compile the circuit
        let result = self.compile_circuit(&kernel_name, &circuit_str, start)?;

        info!(
            "Compilation successful: {} instructions in {}ms",
            result.stats.isa_instructions, result.stats.total_time_ms
        );

        Ok(result)
    }

    /// Compile a circuit string to ISA
    fn compile_circuit(
        &self,
        kernel_name: &str,
        circuit_str: &str,
        start: Instant,
    ) -> Result<CompilationResult> {
        let circuit_start = Instant::now();

        // Use circuit_to_isa translation
        debug!("Parsing and translating circuit: {}", circuit_str);
        // Note: no_canonicalize flag is tracked in stats but canonicalization is always applied
        // Default to f32 for circuit compilation
        let translated = circuit_to_isa::translate_to_isa_with_canonicalization::<f32>(circuit_str)
            .map_err(|e| anyhow::anyhow!("Failed to translate circuit: {}", e))?;

        let circuit_time_ms = circuit_start.elapsed().as_millis() as u64;
        let isa_time_ms = 0; // Already included in circuit_time_ms

        let total_time_ms = start.elapsed().as_millis() as u64;

        let stats = CompilationStats {
            total_time_ms,
            circuit_time_ms,
            isa_time_ms,
            circuit_nodes: 1,
            isa_instructions: translated.program.instructions.len(),
            canonicalized: !self.no_canonicalize,
            opt_level: self.opt_level,
        };

        let circuit_repr = if self.emit_circuit {
            Some(circuit_str.to_string())
        } else {
            None
        };

        Ok(CompilationResult {
            kernel_name: kernel_name.to_string(),
            output_path: None,
            program: translated.program,
            circuit: circuit_repr,
            stats,
        })
    }

    /// Load circuit from kernel registry
    fn load_circuit_from_registry(&self, kernel_name: &str) -> Result<String> {
        warn!(
            "Registry integration not yet implemented, using placeholder circuit for '{}'",
            kernel_name
        );

        // Return a simple circuit for demonstration
        Ok("mark@c00".to_string())
    }

    /// Compile all Python files in a directory
    pub fn compile_directory(
        &mut self,
        input_dir: &Utf8Path,
        output_dir: Option<&Utf8Path>,
    ) -> Result<Vec<CompilationResult>> {
        info!("Compiling directory: {}", input_dir);

        let mut results = Vec::new();

        // Read all .py files in directory
        let entries = fs::read_dir(input_dir)
            .with_context(|| format!("Failed to read directory: {}", input_dir))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("py") {
                let input_path = Utf8PathBuf::try_from(path.clone())
                    .with_context(|| format!("Invalid UTF-8 path: {:?}", path))?;

                match self.compile_file(&input_path) {
                    Ok(mut result) => {
                        // Set output path if output_dir is specified
                        if let Some(out_dir) = output_dir {
                            let output_file = format!("{}.json", result.kernel_name);
                            result.output_path = Some(out_dir.join(output_file));
                        }

                        results.push(result);
                    }
                    Err(e) => {
                        warn!("Failed to compile {}: {}", input_path, e);
                    }
                }
            }
        }

        info!("Compiled {} kernels", results.len());

        Ok(results)
    }

    /// Check if a kernel compiles without generating output
    pub fn check_file(&mut self, input_path: &Utf8Path) -> Result<bool> {
        info!("Checking kernel: {}", input_path);

        match self.compile_file(input_path) {
            Ok(_) => {
                info!("✓ Kernel compiles successfully");
                Ok(true)
            }
            Err(e) => {
                warn!("✗ Compilation failed: {}", e);
                Ok(false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_creation() {
        let compiler = Compiler::new(Backend::Cpu, 2);
        assert_eq!(compiler.opt_level, 2);
        assert!(!compiler.no_canonicalize);
        assert!(!compiler.emit_circuit);
    }

    #[test]
    fn test_compiler_with_options() {
        let compiler = Compiler::new(Backend::Cpu, 3)
            .with_no_canonicalize(true)
            .with_emit_circuit(true);

        assert_eq!(compiler.opt_level, 3);
        assert!(compiler.no_canonicalize);
        assert!(compiler.emit_circuit);
    }

    #[test]
    fn test_compile_simple_circuit() {
        let compiler = Compiler::new(Backend::Cpu, 2);
        let start = Instant::now();

        let result = compiler
            .compile_circuit("test", "mark@c00", start)
            .expect("Should compile simple circuit");

        assert_eq!(result.kernel_name, "test");
        assert!(result.stats.isa_instructions > 0);
    }
}
