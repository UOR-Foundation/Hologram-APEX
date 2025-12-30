# hologram-compile Binary Specification

**Status:** Approved
**Version:** 0.1.0
**Last Updated:** 2025-01-18

## Overview

`hologram-compile` is a command-line tool for compiling high-level kernel definitions to Atlas ISA programs. It bridges the gap between human-readable Python-based kernel definitions and the low-level ISA that hologram backends execute.

## Purpose

Core responsibilities:
- Compile Python kernel definitions to intermediate JSON representation
- Translate JSON to optimized Atlas ISA programs
- Validate kernel syntax and semantics
- Generate optimized ISA with pattern-based canonicalization
- Support batch compilation of kernel directories
- Provide detailed error messages for kernel issues

## Compilation Pipeline

```
Python Kernel (.py)
    ↓ parse & validate
Intermediate JSON (.json)
    ↓ translate & optimize
Atlas ISA Program (.isa)
    ↓ load & execute
hologram-core execution
```

## Command-Line Interface

### Basic Usage

```bash
# Compile single kernel to JSON
hologram-compile kernel.py

# Compile to JSON and ISA
hologram-compile kernel.py --isa

# Compile entire directory
hologram-compile --dir kernels/onnx/core/

# Output to specific location
hologram-compile kernel.py --output build/kernel.json

# Enable optimization
hologram-compile kernel.py --optimize --isa

# Verbose compilation with debug info
hologram-compile kernel.py --verbose --isa
```

### Command Structure

```
hologram-compile [OPTIONS] <INPUT>

ARGUMENTS:
    <INPUT>    Input kernel file (.py) or directory

OPTIONS:
    -o, --output <PATH>       Output file path (default: same as input with .json)
    --isa                     Generate ISA program (.isa) in addition to JSON
    --dir                     Treat input as directory and compile all kernels
    -O, --optimize            Enable canonicalization optimizations
    --validate-only           Only validate kernel syntax (no compilation)
    --verbose                 Enable verbose output
    --debug                   Include debug information in output
    -h, --help               Print help information
    -V, --version            Print version information
```

## Kernel Language Support

### Supported Python Constructs

The compiler supports a restricted subset of Python for kernel definitions:

**Allowed:**
- Variables and assignments: `x = get_global_id()`
- Constants: `42`, `3.14`, `True`, `False`
- Binary operations: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Comparison operations: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical operations: `and`, `or`, `not`
- Array subscripting: `Input[idx]`, `Output[i * N + j]`
- Function calls: `get_global_id()`, `abs(x)`, `max(a, b)`
- For loops: `for i in range(N):`
- If statements: `if condition:`
- Type annotations: `x: f32`, `Input: DeviceArray[f32]`

**Not Allowed:**
- Python lists/dicts/sets: `[1, 2, 3]`, `{key: value}`
- Complex data structures
- Lambdas or closures
- Class definitions (except kernel signatures)
- Import statements (except from atlas_kernel)
- Dynamic typing

### Example Kernel

```python
"""
Vector Addition Kernel

Adds two vectors element-wise: C = A + B

Shapes:
  - A: [N] (input)
  - B: [N] (input)
  - C: [N] (output)
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id

def vector_add(
    A: DeviceArray[f32],
    B: DeviceArray[f32],
    C: DeviceArray[f32],
    N: u32
):
    """
    Element-wise vector addition

    Parameters:
    - A: First input vector
    - B: Second input vector
    - C: Output vector
    - N: Vector length
    """
    idx = get_global_id()

    if idx < N:
        C[idx] = A[idx] + B[idx]
```

## JSON Intermediate Representation

### JSON Kernel Format

```json
{
  "name": "vector_add",
  "description": "Element-wise vector addition",
  "documentation": "Adds two vectors element-wise: C = A + B",
  "parameters": [
    {
      "name": "A",
      "type": "DeviceArray",
      "element_type": "f32",
      "direction": "input"
    },
    {
      "name": "B",
      "type": "DeviceArray",
      "element_type": "f32",
      "direction": "input"
    },
    {
      "name": "C",
      "type": "DeviceArray",
      "element_type": "f32",
      "direction": "output"
    },
    {
      "name": "N",
      "type": "u32",
      "direction": "input"
    }
  ],
  "body": [
    {
      "op": "get_global_id",
      "result": "idx"
    },
    {
      "op": "if",
      "condition": {
        "op": "less_than",
        "left": "idx",
        "right": "N"
      },
      "then": [
        {
          "op": "load",
          "array": "A",
          "index": "idx",
          "result": "a_val"
        },
        {
          "op": "load",
          "array": "B",
          "index": "idx",
          "result": "b_val"
        },
        {
          "op": "add",
          "left": "a_val",
          "right": "b_val",
          "result": "sum"
        },
        {
          "op": "store",
          "array": "C",
          "index": "idx",
          "value": "sum"
        }
      ]
    }
  ],
  "shapes": {
    "A": ["N"],
    "B": ["N"],
    "C": ["N"]
  }
}
```

## Atlas ISA Generation

### ISA Program Structure

The compiler translates JSON to Atlas ISA programs:

```isa
; Vector addition kernel
; Compiled from: kernels/onnx/core/add.py

PROGRAM vector_add
  INPUTS:
    buffer A f32[N]
    buffer B f32[N]
    buffer C f32[N]
    scalar N u32

  ENTRY:
    ; Get global thread ID
    tid = GLOBAL_ID()

    ; Bounds check
    valid = CMP_LT tid, N
    BRANCH_IF_NOT valid, END

    ; Load inputs
    a_val = LOAD A[tid]
    b_val = LOAD B[tid]

    ; Compute sum
    sum = ADD a_val, b_val

    ; Store result
    STORE C[tid], sum

  END:
    RETURN
END_PROGRAM
```

### Optimization Passes

When `--optimize` is enabled:

1. **Dead Code Elimination** - Remove unused variables and operations
2. **Constant Folding** - Evaluate constant expressions at compile time
3. **Strength Reduction** - Replace expensive operations with cheaper equivalents
4. **Loop Unrolling** - Unroll small loops for better performance
5. **Memory Coalescing** - Optimize memory access patterns
6. **Canonicalization** - Apply pattern-based simplification (via hologram-compiler)

## Implementation Architecture

### Crate Structure

```
hologram-compile/
├── Cargo.toml
├── src/
│   ├── main.rs              # CLI entry point
│   ├── parser.rs            # Python kernel parser
│   ├── validator.rs         # Kernel validation
│   ├── json_gen.rs          # JSON IR generation
│   ├── isa_gen.rs           # ISA code generation
│   ├── optimizer.rs         # Optimization passes
│   ├── error.rs             # Error types
│   └── lib.rs               # Library exports
└── tests/
    ├── parse_tests.rs       # Parser tests
    ├── compile_tests.rs     # End-to-end tests
    └── fixtures/            # Test kernels
```

### Dependencies

```toml
[dependencies]
hologram-compiler = { path = "../crates/compiler", version = "0.1.0" }
hologram-core = { path = "../crates/core", version = "0.1.0" }

# CLI
clap = { version = "4.4", features = ["derive"] }
anyhow = "1.0"
thiserror = "1.0"

# Parsing
rustpython-parser = "0.3"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
tempfile = "3.8"
assert_cmd = "2.0"
predicates = "3.0"
```

### Main Entry Point

```rust
// src/main.rs
use clap::Parser;
use hologram_compile::{compile_kernel, CompileOptions};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "hologram-compile")]
#[command(about = "Compile kernel definitions to Atlas ISA", long_about = None)]
struct Cli {
    /// Input kernel file or directory
    input: PathBuf,

    /// Output file path
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Generate ISA in addition to JSON
    #[arg(long)]
    isa: bool,

    /// Treat input as directory
    #[arg(long)]
    dir: bool,

    /// Enable optimizations
    #[arg(short = 'O', long)]
    optimize: bool,

    /// Validate only (no compilation)
    #[arg(long)]
    validate_only: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Debug mode
    #[arg(long)]
    debug: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Setup logging
    if cli.verbose {
        tracing_subscriber::fmt::init();
    }

    let options = CompileOptions {
        generate_isa: cli.isa,
        optimize: cli.optimize,
        validate_only: cli.validate_only,
        debug: cli.debug,
    };

    if cli.dir {
        compile_directory(&cli.input, &options)?;
    } else {
        let output = cli.output.unwrap_or_else(|| {
            let mut path = cli.input.clone();
            path.set_extension("json");
            path
        });

        compile_kernel(&cli.input, &output, &options)?;
    }

    Ok(())
}

fn compile_directory(dir: &PathBuf, options: &CompileOptions) -> anyhow::Result<()> {
    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "py"))
    {
        let input = entry.path();
        let mut output = input.to_path_buf();
        output.set_extension("json");

        println!("Compiling: {}", input.display());
        compile_kernel(input, &output, options)?;
    }
    Ok(())
}
```

### Core Compilation Logic

```rust
// src/lib.rs
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Code generation error: {0}")]
    CodeGenError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, CompileError>;

pub struct CompileOptions {
    pub generate_isa: bool,
    pub optimize: bool,
    pub validate_only: bool,
    pub debug: bool,
}

pub fn compile_kernel(
    input: &Path,
    output: &Path,
    options: &CompileOptions,
) -> Result<()> {
    // 1. Parse Python kernel
    let source = std::fs::read_to_string(input)?;
    let ast = parser::parse_kernel(&source)?;

    // 2. Validate kernel
    validator::validate(&ast)?;

    if options.validate_only {
        println!("✓ Kernel validation passed");
        return Ok(());
    }

    // 3. Generate JSON IR
    let json_ir = json_gen::generate(&ast)?;

    // 4. Optimize if requested
    let optimized = if options.optimize {
        optimizer::optimize(json_ir)?
    } else {
        json_ir
    };

    // 5. Write JSON output
    std::fs::write(output, serde_json::to_string_pretty(&optimized)?)?;
    println!("✓ Generated: {}", output.display());

    // 6. Generate ISA if requested
    if options.generate_isa {
        let isa_path = output.with_extension("isa");
        let isa_program = isa_gen::generate(&optimized)?;
        std::fs::write(&isa_path, isa_program)?;
        println!("✓ Generated: {}", isa_path.display());
    }

    Ok(())
}
```

## Usage Examples

### Compiling ONNX Operations

```bash
# Compile single ONNX operation
cd /workspace/hologram/kernels
hologram-compile onnx/core/add.py --isa --optimize

# Output:
# ✓ Generated: onnx/core/add.json
# ✓ Generated: onnx/core/add.isa

# Compile all ONNX core operations
hologram-compile --dir onnx/core/ --isa --optimize

# Compile with verbose logging
hologram-compile onnx/activation/relu.py --isa --optimize --verbose
```

### Validating Kernels

```bash
# Validate syntax without compiling
hologram-compile kernel.py --validate-only

# Output:
# ✓ Kernel validation passed
```

### Custom Output Location

```bash
# Compile to specific output directory
hologram-compile kernels/custom_op.py \
  --output build/kernels/custom_op.json \
  --isa \
  --optimize
```

## Error Handling

### Parse Errors

```
Error: Parse error at line 15, column 8
    C[idx] = A[idx] + B[idx
                          ^
Expected: closing bracket ']'
```

### Validation Errors

```
Error: Validation error in function 'vector_add'
  Parameter 'A' type mismatch:
    Expected: DeviceArray[f32]
    Found: DeviceArray[i32]
```

### Type Errors

```
Error: Type error at line 20
    result = A[idx] + "string"
                      ^^^^^^^^
Cannot add f32 and str
```

## Integration with hologram-core

### Loading Compiled Programs

```rust
use hologram_core::{Executor, Program};

// Load compiled ISA program
let program = Program::from_file("kernels/onnx/core/add.isa")?;

// Execute on backend
let exec = Executor::new()?;
exec.execute_program(&program, &launch_config)?;
```

### Runtime Compilation

```rust
use hologram_compile::compile_kernel;

// Compile at runtime if needed
let options = CompileOptions {
    generate_isa: true,
    optimize: true,
    validate_only: false,
    debug: false,
};

compile_kernel(
    "custom_kernel.py",
    "custom_kernel.json",
    &options
)?;

// Load and execute
let program = Program::from_file("custom_kernel.isa")?;
exec.execute_program(&program, &config)?;
```

## Build and Installation

### Building the Binary

```bash
# Build in release mode
cargo build --release --bin hologram-compile

# Binary location
./target/release/hologram-compile

# Install system-wide
cargo install --path crates/hologram-compile
```

### Cross-Compilation

```bash
# Linux
cargo build --release --target x86_64-unknown-linux-gnu

# macOS
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin

# Windows
cargo build --release --target x86_64-pc-windows-msvc
```

## Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_kernel() {
        let source = r#"
from atlas_kernel import DeviceArray, f32, get_global_id

def add(A: DeviceArray[f32], B: DeviceArray[f32], C: DeviceArray[f32], N: u32):
    idx = get_global_id()
    if idx < N:
        C[idx] = A[idx] + B[idx]
"#;
        let ast = parser::parse_kernel(source).unwrap();
        assert_eq!(ast.name, "add");
        assert_eq!(ast.parameters.len(), 4);
    }

    #[test]
    fn test_validation_catches_type_errors() {
        let source = r#"
def bad_add(A: DeviceArray[f32], B: DeviceArray[i32], C: DeviceArray[f32]):
    C[0] = A[0] + B[0]  # Type error: f32 + i32
"#;
        let ast = parser::parse_kernel(source).unwrap();
        assert!(validator::validate(&ast).is_err());
    }
}
```

### Integration Tests

```rust
// tests/compile_tests.rs
use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

#[test]
fn test_compile_onnx_add() {
    let temp = TempDir::new().unwrap();

    Command::cargo_bin("hologram-compile")
        .unwrap()
        .arg("kernels/onnx/core/add.py")
        .arg("--output")
        .arg(temp.path().join("add.json"))
        .arg("--isa")
        .assert()
        .success()
        .stdout(predicate::str::contains("✓ Generated"));

    // Verify JSON exists
    assert!(temp.path().join("add.json").exists());
    assert!(temp.path().join("add.isa").exists());
}

#[test]
fn test_validate_only() {
    Command::cargo_bin("hologram-compile")
        .unwrap()
        .arg("kernels/onnx/core/add.py")
        .arg("--validate-only")
        .assert()
        .success()
        .stdout(predicate::str::contains("✓ Kernel validation passed"));
}
```

## Performance Considerations

### Compilation Speed

- **Target**: Compile 100 kernels/second on modern hardware
- **Optimization**: Parallel compilation for directory mode
- **Caching**: Cache parsed ASTs to avoid re-parsing

### Memory Usage

- **Streaming**: Process large kernels in streaming fashion
- **Memory limit**: < 100 MB for typical kernel compilation

## Future Enhancements

- [ ] Watch mode for auto-recompilation: `hologram-compile --watch kernels/`
- [ ] Language server protocol (LSP) for IDE integration
- [ ] Kernel documentation generation (markdown/HTML)
- [ ] Performance profiling of generated ISA
- [ ] Cross-kernel optimization (inline across kernels)
- [ ] WASM target for browser-based compilation
- [ ] Interactive kernel REPL
- [ ] Kernel package manager (import/publish kernels)

## References

- [Atlas Kernel Language Specification](../crates/compiler.md)
- [Atlas ISA Reference](../crates/backends.md)
- [Python Parser Documentation](https://rustpython.github.io/rustpython-parser/)
