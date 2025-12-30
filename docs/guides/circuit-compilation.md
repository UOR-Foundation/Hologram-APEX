# Circuit Compilation Guide

Hologram provides a powerful circuit compiler that translates high-level circuit expressions into optimized backend operations. This guide covers the circuit compilation pipeline, expression syntax, and optimization techniques.

## Table of Contents

- [Overview](#overview)
- [Circuit Expression Syntax](#circuit-expression-syntax)
- [Generators](#generators)
- [Compilation Pipeline](#compilation-pipeline)
- [Canonicalization](#canonicalization)
- [Range Operations](#range-operations)
- [Transforms](#transforms)
- [Using the Compiler](#using-the-compiler)
- [CLI Tool](#cli-tool)
- [Advanced Patterns](#advanced-patterns)
- [Best Practices](#best-practices)

## Overview

The Hologram circuit compiler translates circuit expressions (written in **sigil notation**) into optimized sequences of backend generator calls. The compiler performs automatic pattern recognition and canonicalization to reduce operation count.

### Key Features

- **Sigil notation** - Concise, composable circuit expression language
- **7 generators** - mark, copy, swap, merge, split, quote, evaluate
- **Automatic optimization** - Pattern-based canonicalization (H²=I, X²=I, etc.)
- **96-class system** - Canonical forms with efficient addressing
- **Range operations** - Multi-class vectors for large data
- **Transform algebra** - Rotate (R), Twist (T), and Mirror (M) operations

### Compilation Pipeline

```
Circuit Expression → Parse → Canonicalize → Translate → Backend Execution
     (sigil)         (AST)   (optimize)    (calls)      (execute)
```

## Circuit Expression Syntax

Circuit expressions use **sigil notation** with class targets and generator operations.

### Basic Syntax

```
generator@target
```

Examples:
```rust
mark@c00          // Mark operation at class 0
copy@c05->c06     // Copy from class 5 to class 6
mark@c21          // Mark operation at class 21
```

### Sequential Composition

Use `.` to chain operations sequentially:

```rust
mark@c00 . copy@c00->c01 . merge@c01->c02
```

### Parallel Composition

Use `||` for parallel operations:

```rust
mark@c00 || mark@c01 || mark@c02
```

### Grouping

Use parentheses for grouping:

```rust
(mark@c00 . copy@c00->c01) || mark@c10
```

## Generators

Hologram provides seven fundamental generators that operate on the 96-class system.

### 1. Mark Generator

Introduce or remove a mark at a class.

**Syntax**: `mark@c<class>`

**Properties**:
- `mark . mark = I` (involution)
- Equivalent to Hadamard gate in quantum computing

**Example**:
```rust
use hologram::{CircuitCompiler, Result};

let circuit = "mark@c21";
let compiled = CircuitCompiler::compile(circuit)?;
assert_eq!(compiled.calls.len(), 1);
```

### 2. Copy Generator

Copy data from source class to destination class.

**Syntax**: `copy@c<src>->c<dst>`

**Example**:
```rust
let circuit = "copy@c05->c06";
let compiled = CircuitCompiler::compile(circuit)?;
```

### 3. Swap Generator

Exchange data between two classes.

**Syntax**: `swap@c<a><->c<b>`

**Example**:
```rust
let circuit = "swap@c10<->c20";
let compiled = CircuitCompiler::compile(circuit)?;
```

### 4. Merge Generator

Combine source and context to produce destination.

**Syntax**: `merge@c<src>[c<context>,c<dst>]`

**Variants**:
- Add (default): `dst = src + context`
- Mul: `dst = src * context`
- Min: `dst = min(src, context)`
- Max: `dst = max(src, context)`

**Unary variants** (no context):
- Abs: `dst = abs(src)`
- Exp: `dst = exp(src)`
- Log: `dst = log(src)`
- Sqrt: `dst = sqrt(src)`
- Sigmoid: `dst = sigmoid(src)`
- Tanh: `dst = tanh(src)`
- Gelu: `dst = gelu(src)`

**Example**:
```rust
let circuit = "merge@c00[c01,c02]";  // c02 = c00 + c01
let compiled = CircuitCompiler::compile(circuit)?;
```

### 5. Split Generator

Decompose source minus context to produce destination.

**Syntax**: `split@c<src>[c<context>,c<dst>]`

**Variants**:
- Sub (default): `dst = src - context`
- Div: `dst = src / context`

**Example**:
```rust
let circuit = "split@c10[c11,c12]";  // c12 = c10 - c11
let compiled = CircuitCompiler::compile(circuit)?;
```

### 6. Quote Generator

Suspend computation at a class (delay evaluation).

**Syntax**: `quote@c<class>`

**Example**:
```rust
let circuit = "quote@c15";
let compiled = CircuitCompiler::compile(circuit)?;
```

### 7. Evaluate Generator

Force evaluation of suspended computation.

**Syntax**: `evaluate@c<class>`

**Example**:
```rust
let circuit = "quote@c15 . evaluate@c15";
let compiled = CircuitCompiler::compile(circuit)?;
```

## Compilation Pipeline

### Step 1: Parse

Convert circuit expression to Abstract Syntax Tree (AST).

```rust
use hologram::lang::parser::parse;

let circuit = "mark@c00 . copy@c00->c01";
let ast = parse(circuit)?;
```

### Step 2: Canonicalize

Apply rewrite rules to optimize the circuit.

```rust
use hologram::Canonicalizer;

// H² circuit - should optimize to identity
let circuit = "copy@c05->c06 . mark@c21 . copy@c05->c06 . mark@c21";
let result = Canonicalizer::parse_and_canonicalize(circuit)?;

println!("Rewrites applied: {}", result.rewrite_count);
println!("Rules: {:?}", result.applied_rules);
```

### Step 3: Compile

Translate canonical form to backend generator calls.

```rust
use hologram::CircuitCompiler;

let circuit = "mark@c00 . copy@c00->c01";
let compiled = CircuitCompiler::compile(circuit)?;

println!("Original ops: {}", compiled.original_ops);
println!("Canonical ops: {}", compiled.canonical_ops);
println!("Reduction: {:.1}%", compiled.reduction_pct);
println!("ISA instructions: {}", compiled.calls.len());
```

### Complete Example

```rust
use hologram::{CircuitCompiler, Result};

fn main() -> Result<()> {
    let circuit = "copy@c05->c06 . mark@c21 . copy@c05->c06 . mark@c21";

    let compiled = CircuitCompiler::compile(circuit)?;

    println!("Compilation Statistics:");
    println!("  Original expression: {}", compiled.original_expr);
    println!("  Canonical expression: {}", compiled.canonical_expr);
    println!("  Operations before: {}", compiled.original_ops);
    println!("  Operations after: {}", compiled.canonical_ops);
    println!("  Optimization: {:.1}% reduction", compiled.reduction_pct);
    println!("  Backend calls: {}", compiled.calls.len());

    Ok(())
}
```

## Canonicalization

Canonicalization automatically optimizes circuits by applying pattern-based rewrite rules.

### Supported Patterns

| Pattern | Rule | Example |
|---------|------|---------|
| **H² = I** | Hadamard squared | `(copy@c05->c06 . mark@c21)²` → `mark@c00` |
| **X² = I** | Pauli-X squared | `mark@c21 . mark@c21` → `mark@c00` |
| **Z² = I** | Pauli-Z squared | `merge@c00->c01 . merge@c00->c01` → identity |
| **S² = Z** | S gate squared | `(specific sequence)²` → Z gate |
| **HXH = Z** | Hadamard conjugation | `H . X . H` → Z gate |
| **I·I = I** | Identity composition | `mark@c00 . mark@c00` → `mark@c00` |

### Using the Canonicalizer

```rust
use hologram::Canonicalizer;

// Example 1: H² = I
let h_squared = "copy@c05->c06 . mark@c21 . copy@c05->c06 . mark@c21";
let result = Canonicalizer::parse_and_canonicalize(h_squared)?;

println!("Changed: {}", result.changed);        // true
println!("Rewrites: {}", result.rewrite_count); // 1
println!("Rules: {:?}", result.applied_rules);  // ["H² = I"]

// Example 2: X² = I
let x_squared = "mark@c21 . mark@c21";
let result = Canonicalizer::parse_and_canonicalize(x_squared)?;

println!("Rewrites: {}", result.rewrite_count); // 1
println!("Rules: {:?}", result.applied_rules);  // ["X² = I"]

// Get canonical form as string
let canonical_form = Canonicalizer::canonical_form(h_squared)?;
println!("Canonical: {}", canonical_form);
```

### Automatic Optimization Benefits

```rust
use hologram::CircuitCompiler;

// Without canonicalization (raw compilation)
let circuit = "copy@c05->c06 . mark@c21 . copy@c05->c06 . mark@c21";
let raw = CircuitCompiler::compile_raw(circuit)?;
println!("Raw operations: {}", raw.len()); // 4

// With canonicalization (default)
let optimized = CircuitCompiler::compile(circuit)?;
println!("Optimized operations: {}", optimized.canonical_ops); // 1
println!("Reduction: {:.1}%", optimized.reduction_pct);        // 75.0%
```

## Range Operations

Range operations apply generators across multiple classes efficiently.

### Syntax

```
generator@c[start..end]
```

### Supported Generators

- `mark@c[start..end]` - Mark all classes in range
- `merge@c[start..end]` - Merge operation across range
- `quote@c[start..end]` - Quote all classes in range
- `evaluate@c[start..end]` - Evaluate all classes in range

### Examples

```rust
use hologram::CircuitCompiler;

// Mark classes 0-9 (10 classes)
let circuit = "mark@c[0..9]";
let compiled = CircuitCompiler::compile(circuit)?;
assert_eq!(compiled.calls.len(), 1); // Compiled to single range call

// Merge across range
let circuit = "merge@c[5..14]";
let compiled = CircuitCompiler::compile(circuit)?;
// Handles 100K elements across 10 classes efficiently

// Quote range
let circuit = "quote@c[10..15]";
let compiled = CircuitCompiler::compile(circuit)?;

// Evaluate range
let circuit = "evaluate@c[20..25]";
let compiled = CircuitCompiler::compile(circuit)?;
```

### Mixed Operations

Combine single-class and range operations:

```rust
let circuit = "mark@c0 . merge@c[5..9] . mark@c20";
let compiled = CircuitCompiler::compile(circuit)?;
assert_eq!(compiled.calls.len(), 3);
```

### Unsupported on Ranges

These generators require explicit source/destination and don't support ranges:

- ❌ `copy@c[0..5]` - Copy requires specific source and destination
- ❌ `swap@c[0..5]` - Swap requires two specific classes
- ❌ `split@c[0..5]` - Split requires explicit context/destination

## Transforms

Transforms modify class addressing using algebraic operations.

### Three Transform Types

| Transform | Symbol | Description |
|-----------|--------|-------------|
| **Rotate** | `R` | Rotate h₂ component (mod 4) |
| **Twist** | `T` | Twist ℓ component (mod 8) |
| **Mirror** | `M` or `~` | Mirror dihedral component |

### Class Structure

Each class has three components:
- **h₂** - Higher order (0-3)
- **d** - Dihedral element (0=Neutral, 1=Produce, 2=Consume)
- **ℓ** - Lower order (0-7)

Class index = `24*h₂ + 8*d + ℓ`

### Prefix Transforms

Apply before the operation:

```rust
// Rotate by +1
let circuit = "R+1@ mark@c21";
let compiled = CircuitCompiler::compile(circuit)?;
// Class 21: h₂=0, d=2, ℓ=5 → R+1 → h₂=1, d=2, ℓ=5 → class 45

// Mirror
let circuit = "~@ mark@c21";
let compiled = CircuitCompiler::compile(circuit)?;
// Class 21: h₂=0, d=2 (Consume), ℓ=5 → M → h₂=0, d=1 (Produce), ℓ=5 → class 13

// Twist by +3
let circuit = "T+3@ mark@c00";
let compiled = CircuitCompiler::compile(circuit)?;
// Class 0: h₂=0, d=0, ℓ=0 → T+3 → h₂=0, d=0, ℓ=3 → class 3
```

### Postfix Transforms

Apply after the class specification:

```rust
// Postfix rotation
let circuit = "mark@c21+2";  // Rotate class 21 by +2

// Postfix twist
let circuit = "mark@c21^+3"; // Twist class 21 by +3

// Postfix mirror
let circuit = "mark@c21~";   // Mirror class 21
```

### Combined Transforms

Apply both prefix and postfix:

```rust
// R+1@ mark@c21^+2~
// Class 21: h₂=0, d=2, ℓ=5
// 1. Apply postfix T+2: h₂=0, d=2, ℓ=7
// 2. Apply postfix M: h₂=0, d=1, ℓ=7 → class 15
// 3. Apply prefix R+1: h₂=1, d=1, ℓ=7 → class 39

let circuit = "R+1@ mark@c21^+2~";
let compiled = CircuitCompiler::compile(circuit)?;
// Result: mark@c39
```

### Transforms on Ranges

Transforms apply to range boundaries:

```rust
// Mirror range
let circuit = "~@ merge@c[0..9]";
// Start: class 0 → M → class 0
// End: class 9 → M → class 17

// Rotate range
let circuit = "R+2@ quote@c[10..15]";
// Start: class 10 → R+2 → class 58
// End: class 15 → R+2 → class 63

// Combined transforms on range
let circuit = "~@ mark@c[5..10]^+3";
let compiled = CircuitCompiler::compile(circuit)?;
```

## Using the Compiler

### Rust API

```rust
use hologram::{CircuitCompiler, Canonicalizer, Result};

fn compile_circuit(circuit: &str) -> Result<()> {
    // Method 1: Full compilation with canonicalization
    let compiled = CircuitCompiler::compile(circuit)?;

    println!("Compilation Statistics:");
    println!("  Original: {} ops", compiled.original_ops);
    println!("  Canonical: {} ops", compiled.canonical_ops);
    println!("  Reduction: {:.1}%", compiled.reduction_pct);
    println!("  Backend calls: {}", compiled.calls.len());

    // Access generated calls
    for (i, call) in compiled.calls.iter().enumerate() {
        println!("  Call {}: {:?}", i, call);
    }

    Ok(())
}

// Method 2: Raw compilation (no optimization)
fn compile_raw(circuit: &str) -> Result<()> {
    let calls = CircuitCompiler::compile_raw(circuit)?;
    println!("Generated {} calls", calls.len());
    Ok(())
}

// Method 3: Canonicalization only
fn canonicalize_only(circuit: &str) -> Result<()> {
    let result = Canonicalizer::parse_and_canonicalize(circuit)?;

    if result.changed {
        println!("Applied {} rewrites", result.rewrite_count);
        println!("Rules: {:?}", result.applied_rules);
    } else {
        println!("Already in canonical form");
    }

    Ok(())
}
```

### Full Example

```rust
use hologram::{CircuitCompiler, Result};

fn main() -> Result<()> {
    // Example circuit: H² pattern
    let h_squared = "copy@c05->c06 . mark@c21 . copy@c05->c06 . mark@c21";

    println!("Compiling circuit: {}\n", h_squared);

    let compiled = CircuitCompiler::compile(h_squared)?;

    println!("Results:");
    println!("  Original operations: {}", compiled.original_ops);   // 4
    println!("  Canonical operations: {}", compiled.canonical_ops); // 1
    println!("  Reduction: {:.1}%", compiled.reduction_pct);        // 75.0%
    println!("  ISA instructions: {}", compiled.calls.len());

    println!("\nOriginal: {}", compiled.original_expr);
    println!("Canonical: {}", compiled.canonical_expr);

    println!("\nGenerated calls:");
    for (i, call) in compiled.calls.iter().enumerate() {
        println!("  {}: {:?}", i, call);
    }

    Ok(())
}
```

## CLI Tool

The `hologram-compile` binary provides command-line circuit compilation.

### Installation

```bash
cargo install hologram-compile
```

Or build from source:

```bash
cd binaries/hologram-compile
cargo build --release
```

### Usage

```bash
# Compile a circuit expression
hologram-compile "mark@c00 . copy@c00->c01"

# With canonicalization statistics
hologram-compile "copy@c05->c06 . mark@c21 . copy@c05->c06 . mark@c21"

# Show generated ISA calls
hologram-compile --show-calls "mark@c00 . merge@c[5..9]"

# Output to file
hologram-compile "mark@c00" --output circuit.isa

# Read from file
hologram-compile --input circuit.txt

# Verbose output
hologram-compile -v "mark@c00 . copy@c00->c01"
```

### Example Output

```bash
$ hologram-compile "copy@c05->c06 . mark@c21 . copy@c05->c06 . mark@c21"

=== Hologram Circuit Compiler ===

Input: copy@c05->c06 . mark@c21 . copy@c05->c06 . mark@c21

Compilation Statistics:
  Original operations: 4
  Canonical operations: 1
  Reduction: 75.0%
  ISA instructions: 1

Canonical form: Mark { generator: Mark, class_index: 0 }

Applied rewrite rules:
  - H² = I

✓ Compilation successful!
```

## Advanced Patterns

### 1. Bell State Preparation

```rust
// Prepare |Φ⁺⟩ Bell state
let bell_phi_plus = "mark@c00 . copy@c00->c01";
let compiled = CircuitCompiler::compile(bell_phi_plus)?;
```

### 2. GHZ State (3-qubit)

```rust
// |GHZ⟩ = (|000⟩ + |111⟩) / √2
let ghz_3 = "mark@c00 . copy@c00->c01 . copy@c00->c02";
let compiled = CircuitCompiler::compile(ghz_3)?;
```

### 3. Quantum Teleportation Circuit

```rust
let teleportation = "
    mark@c00 . copy@c00->c01 .
    copy@c02->c00 .
    mark@c02 .
    swap@c00<->c01
";
let compiled = CircuitCompiler::compile(teleportation)?;
```

### 4. Deutsch-Jozsa Algorithm

```rust
let deutsch_jozsa = "
    mark@c[0..3] .
    copy@c00->c10 .
    evaluate@c10 .
    mark@c[0..3]
";
let compiled = CircuitCompiler::compile(deutsch_jozsa)?;
```

### 5. Neural Network Layer

```rust
// Linear layer: y = Wx + b
let linear_layer = "
    merge@c00[c10,c20] .  // Matrix multiply
    merge@c20[c30,c20]    // Add bias
";
let compiled = CircuitCompiler::compile(linear_layer)?;
```

### 6. Softmax Activation

```rust
// Softmax across 32 elements
let softmax = "
    merge@c[0..31] .      // Compute exp
    evaluate@c[0..31] .   // Normalize
    quote@c[0..31]        // Cache result
";
let compiled = CircuitCompiler::compile(softmax)?;
```

## Best Practices

### 1. Let the Compiler Optimize

Always use `CircuitCompiler::compile()` instead of `compile_raw()`:

```rust
// ✅ Good: Automatic optimization
let compiled = CircuitCompiler::compile(circuit)?;

// ❌ Avoid: Manual optimization needed
let calls = CircuitCompiler::compile_raw(circuit)?;
```

### 2. Use Range Operations

For multi-class operations, use ranges:

```rust
// ✅ Good: Single range call
mark@c[0..31]

// ❌ Inefficient: 32 individual calls
mark@c00 . mark@c01 . mark@c02 . ... . mark@c31
```

### 3. Group Related Operations

Use grouping to clarify circuit structure:

```rust
// ✅ Clear structure
(mark@c00 . copy@c00->c01) || (mark@c10 . copy@c10->c11)

// ❌ Unclear structure
mark@c00 || mark@c10 . copy@c00->c01 . copy@c10->c11
```

### 4. Check Canonicalization Results

Verify optimization effectiveness:

```rust
let compiled = CircuitCompiler::compile(circuit)?;

if compiled.reduction_pct > 10.0 {
    println!("Significant optimization: {:.1}%", compiled.reduction_pct);
}

println!("Applied rules: {:?}", compiled.canonical_expr);
```

### 5. Use Transforms for Symmetry

Leverage transforms instead of duplicating circuits:

```rust
// ✅ Good: Use rotation transform
R+1@ (mark@c00 . copy@c00->c01)

// ❌ Inefficient: Manually specify rotated classes
mark@c24 . copy@c24->c25
```

### 6. Profile Complex Circuits

For large circuits, measure compilation time:

```rust
use std::time::Instant;

let start = Instant::now();
let compiled = CircuitCompiler::compile(complex_circuit)?;
let duration = start.elapsed();

println!("Compilation time: {:?}", duration);
println!("Operations reduced: {} → {}",
    compiled.original_ops, compiled.canonical_ops);
```

### 7. Validate Circuit Syntax

Handle parse errors gracefully:

```rust
match CircuitCompiler::compile(circuit) {
    Ok(compiled) => {
        println!("✓ Compiled successfully");
        println!("  ISA instructions: {}", compiled.calls.len());
    }
    Err(e) => {
        eprintln!("✗ Compilation failed: {}", e);
        eprintln!("  Check circuit syntax");
    }
}
```

## Troubleshooting

### Parse Error: Invalid Syntax

```
Error: Parse error: Expected class target, got ...
```

**Solution**: Verify circuit syntax follows sigil notation:
```rust
// ✅ Correct
mark@c00

// ❌ Wrong
mark c00
mark@00
```

### Generator Not Supported on Range

```
Error: Copy generator not supported on ranges
```

**Solution**: Use explicit source/destination for copy, swap, split:
```rust
// ✅ Correct
copy@c00->c01

// ❌ Wrong
copy@c[0..5]
```

### Transform Application Error

```
Error: Invalid class index after transform
```

**Solution**: Ensure transformed class index is within 0-95:
```rust
// Class indices must stay in valid range after transforms
// Check component values: h₂ (0-3), d (0-2), ℓ (0-7)
```

## Next Steps

- [Getting Started Guide](getting-started.md) - Learn Hologram basics
- [Multi-Backend Guide](multi-backend.md) - Execute on different hardware
- [API Reference](../api/) - Detailed API documentation
- [Examples](../../examples/) - More circuit examples

---

For complete circuit compilation examples, see [`examples/03_circuit_compilation.rs`](../../examples/03_circuit_compilation.rs).
