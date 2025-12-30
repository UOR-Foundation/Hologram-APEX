# hologram-compile

CLI tool for compiling Python kernels to Atlas ISA.

## Installation

```bash
cargo install --path .
```

Or from the workspace root:

```bash
cargo build --release -p hologram-compile
# Binary will be at: target/release/hologram-compile
```

## Usage

### Compile a single kernel

```bash
hologram-compile kernel.py
hologram-compile kernel.py -o output.json
```

### Compile all kernels in a directory

```bash
hologram-compile compile-all kernels/ -o out/
```

### Check if a kernel compiles

```bash
hologram-compile check kernel.py
```

### Show information about a compiled kernel

```bash
hologram-compile info compiled.json
```

### Disassemble a compiled ISA program

```bash
hologram-compile disasm compiled.json
```

## Options

### Output Format

- `--format json` - JSON format (default)
- `--format asm` - Human-readable assembly
- `--format circuit` - Circuit representation
- `--format binary` - Binary format (not yet implemented)

### Backend Selection

- `--backend cpu` - CPU backend with SIMD (default)
- `--backend cuda` - CUDA backend for NVIDIA GPUs
- `--backend metal` - Metal backend for Apple Silicon
- `--backend webgpu` - WebGPU backend for WASM

### Optimization

- `-O 0` - No optimization
- `-O 1` - Basic optimization
- `-O 2` - Standard optimization (default)
- `-O 3` - Aggressive optimization

### Output Control

- `-v, -vv, -vvv` - Increase verbosity (info, debug, trace)
- `--quiet` - Suppress all output except errors
- `--stats` - Print compilation statistics
- `--emit-asm` - Emit human-readable ISA assembly
- `--emit-circuit` - Emit Circuit representation before ISA
- `--no-canonicalize` - Skip canonicalization (for debugging)

## Examples

### Basic compilation

```bash
# Compile with default settings
hologram-compile vector_add.py

# Compile for CUDA backend
hologram-compile vector_add.py -b cuda

# Compile with maximum optimization
hologram-compile vector_add.py -O 3
```

### Advanced usage

```bash
# Emit both circuit and ISA assembly
hologram-compile kernel.py --emit-circuit --emit-asm -o kernel.asm

# Compile with statistics
hologram-compile kernel.py --stats

# Compile directory with verbose output
hologram-compile compile-all kernels/ -o out/ -vv --stats
```

### Inspecting compiled kernels

```bash
# Show detailed information
hologram-compile info kernel.json

# Disassemble to human-readable assembly
hologram-compile disasm kernel.json
```

## Output Format

### JSON Format (Default)

```json
{
  "kernel_name": "vector_add",
  "output_path": "vector_add.json",
  "program": {
    "instructions": [...]
  },
  "stats": {
    "total_time_ms": 42,
    "circuit_time_ms": 15,
    "isa_time_ms": 27,
    "circuit_nodes": 5,
    "isa_instructions": 12,
    "canonicalized": true,
    "opt_level": 2
  }
}
```

### Assembly Format

```asm
# Atlas ISA Assembly
# 12 instructions

   0: ADD.F32    r0, r1, r2
   1: MUL.F32    r3, r0, r4
   2: RELU.F32   r5, r3
   ...
```

## Development

### Run tests

```bash
cargo test -p hologram-compile
```

### Run integration tests

```bash
cargo test -p hologram-compile --test integration_test
```

### Build for release

```bash
cargo build --release -p hologram-compile
```

## Architecture

The compiler consists of four main modules:

1. **cli.rs** - Command-line argument parsing with `clap`
2. **compiler.rs** - Compilation pipeline (Python → Circuit → ISA)
3. **output.rs** - Output formatting (JSON, ASM, Circuit)
4. **main.rs** - Entry point and command routing

### Compilation Pipeline

```
Python Kernel
    ↓
[Parse & Load]
    ↓
Circuit Representation
    ↓
[Canonicalization]
    ↓
Atlas ISA Program
    ↓
[Output Formatting]
    ↓
JSON / ASM / Binary
```

## License

MIT OR Apache-2.0
