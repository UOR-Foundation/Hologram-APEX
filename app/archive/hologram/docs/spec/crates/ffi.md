# hologram-ffi Specification

**Status:** Approved
**Version:** 0.1.0
**Last Updated:** 2025-01-18

## Overview

`hologram-ffi` provides Foreign Function Interface (FFI) bindings for hologram-core, enabling usage from Python, Swift, Kotlin, Ruby, TypeScript, C++, and C.

## Purpose

Core responsibilities:
- UniFFI-based automatic bindings (Python, Swift, Kotlin, Ruby)
- Specialized bindings for unsupported languages (TypeScript, C++)
- C header generation via cbindgen
- Comprehensive workflow documentation
- Language-specific examples

## Architecture

```
hologram-ffi
├── UniFFI Bindings (Auto-generated)
│   ├── Python
│   ├── Swift
│   ├── Kotlin
│   └── Ruby
├── Specialized Bindings (Hand-crafted)
│   ├── TypeScript (Neon/N-API)
│   └── C++ (CXX)
├── C Headers (cbindgen)
└── Workflow Documentation
```

## Public API

### UniFFI Interface Definition

**File:** `hologram.udl`

```idl
namespace hologram {
    Executor executor_new(BackendType backend);
};

[Error]
enum HologramError {
    "AllocationFailed",
    "InvalidBuffer",
    "ExecutionFailed",
    "BackendUnavailable",
    "ConfigError",
};

enum BackendType {
    "Cpu",
    "Cuda",
    "Metal",
    "Wasm",
    "WebGpu",
};

interface Executor {
    constructor(BackendType backend);

    [Throws=HologramError]
    Buffer allocate(u64 size);

    [Throws=HologramError]
    void execute_operation([ByRef] Buffer input, [ByRef] Buffer output);

    BackendType backend_type();
};

interface Buffer {
    u64 size();

    [Throws=HologramError]
    void copy_from(sequence<u8> data);

    [Throws=HologramError]
    sequence<u8> to_vec();

    u64 len();
    boolean is_empty();
};

interface Tensor {
    [Throws=HologramError]
    constructor(Buffer buffer, sequence<u64> shape);

    sequence<u64> shape();
    sequence<i64> strides();
    u64 ndim();
    u64 numel();
    boolean is_contiguous();

    [Throws=HologramError]
    Tensor select(u64 dim, u64 index);

    [Throws=HologramError]
    Tensor narrow(u64 dim, u64 start, u64 length);

    [Throws=HologramError]
    Tensor transpose();

    [Throws=HologramError]
    Tensor matmul([ByRef] Executor exec, [ByRef] Tensor other);

    [Throws=HologramError]
    sequence<u8> to_vec();
};

namespace ops {
    [Throws=HologramError]
    void vector_add(
        [ByRef] Executor exec,
        [ByRef] Buffer a,
        [ByRef] Buffer b,
        [ByRef] Buffer c,
        u64 n
    );

    [Throws=HologramError]
    void vector_mul(
        [ByRef] Executor exec,
        [ByRef] Buffer a,
        [ByRef] Buffer b,
        [ByRef] Buffer c,
        u64 n
    );

    // ... more operations
};
```

### Rust FFI Wrappers

```rust
// src/executor.rs
use hologram_core::Executor as CoreExecutor;

pub struct Executor {
    inner: CoreExecutor,
}

impl Executor {
    pub fn new(backend: BackendType) -> Result<Self, HologramError> {
        let inner = CoreExecutor::new(backend.into())
            .map_err(|e| HologramError::BackendUnavailable(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn allocate(&mut self, size: u64) -> Result<Arc<Buffer>, HologramError> {
        let buffer = self.inner.allocate::<u8>(size as usize)
            .map_err(|e| HologramError::AllocationFailed(e.to_string()))?;
        Ok(Arc::new(Buffer::new(buffer)))
    }

    pub fn backend_type(&self) -> BackendType {
        self.inner.backend_type().into()
    }
}
```

## Internal Structure

```
crates/ffi/
├── Cargo.toml
├── build.rs                    # Generate bindings
├── README.md                   # 🔥 COMPREHENSIVE WORKFLOW GUIDE
├── hologram.udl                # UniFFI interface definition
├── src/
│   ├── lib.rs                  # Public FFI API
│   ├── executor.rs             # Executor FFI wrapper
│   ├── buffer.rs               # Buffer FFI wrapper
│   ├── tensor.rs               # Tensor FFI wrapper
│   ├── ops.rs                  # Operations FFI wrappers
│   ├── types.rs                # Type conversions
│   └── error.rs                # FFI error conversions
├── include/
│   └── hologram.h              # Generated C header (cbindgen)
├── bindings/                   # Auto-generated language bindings
│   ├── python/
│   │   ├── hologram.py
│   │   └── _hologram.so
│   ├── swift/
│   │   └── Hologram.swift
│   ├── kotlin/
│   │   └── hologram.kt
│   └── ruby/
│       └── hologram.rb
├── specialized/                # Hand-written bindings
│   ├── typescript/
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── index.ts
│   │   │   ├── executor.ts
│   │   │   ├── buffer.ts
│   │   │   └── tensor.ts
│   │   ├── native/
│   │   │   └── src/
│   │   │       └── lib.rs      # Neon bindings
│   │   └── README.md
│   └── cpp/
│       ├── include/
│       │   ├── hologram.hpp
│       │   ├── executor.hpp
│       │   ├── buffer.hpp
│       │   └── tensor.hpp
│       ├── src/
│       │   └── hologram.cpp    # CXX bindings
│       └── README.md
├── examples/
│   ├── example.py              # Python example
│   ├── example.swift           # Swift example
│   ├── example.ts              # TypeScript example
│   └── example.cpp             # C++ example
└── tests/
    ├── python_tests.py
    ├── swift_tests.swift
    ├── typescript_tests.ts
    └── cpp_tests.cpp
```

## Dependencies

```toml
[dependencies]
hologram-core = { path = "../core", version = "0.1.0" }

# UniFFI
uniffi = { version = "0.25", optional = true }

# Specialized bindings
neon = { version = "1.0", optional = true, default-features = false, features = ["napi-6"] }
cxx = { version = "1.0", optional = true }

[build-dependencies]
uniffi = { version = "0.25", features = ["build"], optional = true }
cbindgen = "0.26"

[features]
default = []
ffi = ["uniffi"]
ffi-python = ["ffi"]
ffi-swift = ["ffi"]
ffi-kotlin = ["ffi"]
ffi-ruby = ["ffi"]
ffi-typescript = ["neon"]
ffi-cpp = ["cxx"]

[lib]
crate-type = ["cdylib", "rlib"]
name = "hologram_ffi"
```

## README.md - Comprehensive Workflow Guide

**File:** `crates/ffi/README.md`

````markdown
# Hologram FFI Bindings

Foreign Function Interface (FFI) bindings for hologram-core.

## Architecture

### Binding Generation Strategy

**Hybrid approach:**
1. **UniFFI** (Primary): Auto-generates Python, Swift, Kotlin, Ruby
2. **Specialized Tools**: TypeScript (Neon), C++ (CXX)
3. **cbindgen**: C headers for maximum compatibility

### Single Source of Truth: `hologram.udl`

All FFI interfaces defined in `hologram.udl` (UniFFI Definition Language).

## Adding a New Language

### For UniFFI-Supported Languages (Python, Swift, Kotlin, Ruby)

**UniFFI automatically supports these.** No additional work needed!

**Steps:**
1. Ensure language in `hologram.udl`
2. Run: `cargo build --features ffi`
3. Bindings appear in `bindings/<language>/`
4. Write tests in `tests/<language>_tests.<ext>`
5. Add example in `examples/example.<ext>`

### For Unsupported Languages (TypeScript, C++, Go, etc.)

**Create specialized binding in `specialized/<language>/`**

#### Example: Adding Go Bindings

**Step 1: Create directory structure**
```bash
mkdir -p specialized/go/src
touch specialized/go/README.md
touch specialized/go/go.mod
```

**Step 2: Use cgo with C bindings**

`specialized/go/hologram.go`:
```go
package hologram

/*
#cgo LDFLAGS: -L../../target/release -lhologram_ffi
#include "../../include/hologram.h"
*/
import "C"
import "unsafe"

type Executor struct {
    handle *C.HologramExecutor
}

func NewExecutor(backend string) (*Executor, error) {
    cBackend := C.CString(backend)
    defer C.free(unsafe.Pointer(cBackend))

    handle := C.hologram_executor_new(cBackend)
    if handle == nil {
        return nil, errors.New("failed to create executor")
    }

    return &Executor{handle: handle}, nil
}
```

**Step 3: Write tests**
**Step 4: Add to build system**
**Step 5: Document in README**
**Step 6: Update CI**

## Workflow: Updating FFI When Core Changes

### Scenario: Adding `Tensor::reshape()` to hologram-core

**Step 1: Update hologram-core**
```rust
// crates/core/src/runtime/tensor.rs
impl<T> Tensor<T> {
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor<T>> {
        // Implementation
    }
}
```

**Step 2: Update hologram.udl**
```idl
interface Tensor {
    // Existing methods...

    [Throws=HologramError]
    Tensor reshape(sequence<u64> new_shape);  // 🆕 Add new method
};
```

**Step 3: Regenerate bindings**
```bash
cargo build --features ffi
```

**Automatic updates:**
- ✅ Python bindings updated
- ✅ Swift bindings updated
- ✅ Kotlin bindings updated
- ✅ Ruby bindings updated
- ✅ C header updated

**Step 4: Update specialized bindings**

TypeScript:
```typescript
export class Tensor {
  reshape(newShape: number[]): Tensor {
    const result = native.tensor_reshape(this.handle, newShape);
    return new Tensor(result);
  }
}
```

**Step 5: Update tests**

Python:
```python
def test_tensor_reshape():
    tensor = hologram.Tensor(buffer, [4, 8])
    reshaped = tensor.reshape([8, 4])
    assert reshaped.shape() == [8, 4]
```

**Step 6: Run all tests**
```bash
cargo test --features ffi --package hologram-ffi
pytest tests/python_tests.py
swift test
```

## Build Features

```bash
# Build specific language
cargo build --features ffi-python
cargo build --features ffi-typescript

# Build all
cargo build --features ffi
```

## Testing

```bash
# Rust tests
cargo test --features ffi --package hologram-ffi

# Python
pytest crates/ffi/tests/python_tests.py

# Swift
swift test --package-path crates/ffi/bindings/swift/

# TypeScript
npm test --prefix crates/ffi/specialized/typescript/
```

## Language Support Matrix

| Language   | Tool      | Status | Auto-Generated | Location |
|------------|-----------|--------|----------------|----------|
| Python     | UniFFI    | ✅ Ready | Yes | `bindings/python/` |
| Swift      | UniFFI    | ✅ Ready | Yes | `bindings/swift/` |
| Kotlin     | UniFFI    | ✅ Ready | Yes | `bindings/kotlin/` |
| Ruby       | UniFFI    | ✅ Ready | Yes | `bindings/ruby/` |
| C          | cbindgen  | ✅ Ready | Yes | `include/hologram.h` |
| TypeScript | Neon      | 🚧 Minimal | No | `specialized/typescript/` |
| C++        | CXX       | 🚧 Minimal | No | `specialized/cpp/` |
| Go         | cgo       | 📋 Planned | No | `specialized/go/` (future) |

## Examples

### Python
```python
import hologram

exec = hologram.Executor("cpu")
buf = exec.allocate(1024)
tensor = hologram.Tensor(buf, [32, 32])
result = tensor.matmul(other_tensor)
```

### Swift
```swift
import Hologram

let executor = try Executor(backend: .cpu)
let buffer = try executor.allocate(size: 1024)
let tensor = try Tensor(buffer: buffer, shape: [32, 32])
```

### TypeScript
```typescript
import { Executor, Tensor } from '@hologram/ffi';

const executor = new Executor('cpu');
const buffer = executor.allocate(1024);
const tensor = new Tensor(buffer, [32, 32]);
```

## Troubleshooting

### UniFFI bindings not generating
- Check `hologram.udl` syntax
- Run: `cargo clean && cargo build --features ffi`

### C header missing types
- Update `cbindgen.toml`
- Ensure types are `#[repr(C)]`

## Contributing

When adding new core functionality:
1. Update hologram-core API
2. Update `hologram.udl`
3. Regenerate bindings: `cargo build --features ffi`
4. Add tests for each language
5. Update examples
````

## Build Configuration

**File:** `build.rs`

```rust
fn main() {
    // Generate UniFFI bindings
    #[cfg(feature = "ffi")]
    {
        uniffi::generate_scaffolding("./hologram.udl")
            .expect("Failed to generate UniFFI scaffolding");
    }

    // Generate C header with cbindgen
    #[cfg(feature = "ffi")]
    {
        let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        cbindgen::Builder::new()
            .with_crate(&crate_dir)
            .with_language(cbindgen::Language::C)
            .generate()
            .expect("Unable to generate C bindings")
            .write_to_file("include/hologram.h");
    }
}
```

## Testing Requirements

### FFI Tests

Each language must have tests:

**Python:**
```python
# tests/python_tests.py
import hologram

def test_executor_creation():
    exec = hologram.Executor("cpu")
    assert exec.backend_type() == "cpu"

def test_buffer_allocation():
    exec = hologram.Executor("cpu")
    buf = exec.allocate(1024)
    assert buf.size() == 1024
```

**Swift:**
```swift
// tests/swift_tests.swift
import XCTest
@testable import Hologram

class HologramTests: XCTestCase {
    func testExecutorCreation() throws {
        let executor = try Executor(backend: .cpu)
        XCTAssertEqual(executor.backendType(), .cpu)
    }
}
```

## Performance Requirements

FFI overhead should be minimal:
- Function call overhead: < 10ns
- Data marshalling: O(n) where n = data size
- Zero-copy where possible (DLPack integration)

## Examples

### UniFFI Binding Usage

```rust
// Automatically generated for Python, Swift, Kotlin, Ruby
uniffi::include_scaffolding!("hologram");
```

### TypeScript Binding (Neon)

```rust
// specialized/typescript/native/src/lib.rs
use neon::prelude::*;

fn create_executor(mut cx: FunctionContext) -> JsResult<JsBox<Executor>> {
    let backend = cx.argument::<JsString>(0)?.value(&mut cx);
    let executor = Executor::new(backend.parse().unwrap())
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(executor))
}

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    cx.export_function("createExecutor", create_executor)?;
    Ok(())
}
```

## Migration from Current Codebase

Port any existing FFI code, or start fresh with UniFFI-based approach.

## Async Support

### Python Asyncio

```python
import asyncio
from hologram import AsyncExecutor

async def main():
    exec = await AsyncExecutor.create("cpu")

    # Async buffer operations
    buffer = await exec.allocate_async(1024, dtype="f32")
    await buffer.copy_from_async([1.0, 2.0, 3.0])

    # Async execution
    result = await exec.execute_async(operation)

    data = await buffer.to_list_async()
    print(data)

asyncio.run(main())
```

### Swift Async/Await

```swift
import Hologram

Task {
    let exec = try await Executor.create(backend: .cpu)

    // Async operations
    let buffer = try await exec.allocate(size: 1024, type: Float32.self)
    try await buffer.copyFrom([1.0, 2.0, 3.0])

    try await exec.execute(operation)

    let data = try await buffer.toArray()
    print(data)
}
```

### Kotlin Coroutines

```kotlin
import hologram.Executor
import kotlinx.coroutines.*

suspend fun main() = coroutineScope {
    val exec = Executor.create("cpu")

    // Suspend functions
    val buffer = exec.allocateAsync<Float>(1024)
    buffer.copyFromAsync(listOf(1.0f, 2.0f, 3.0f))

    exec.executeAsync(operation)

    val data = buffer.toListAsync()
    println(data)
}
```

## Callback Support

### Python Callbacks

```python
from hologram import Executor

def progress_callback(step: int, total: int):
    print(f"Progress: {step}/{total}")

exec = Executor("cpu")
exec.set_progress_callback(progress_callback)
exec.execute_with_progress(operation)
```

### Swift Closures

```swift
let exec = try Executor(backend: .cpu)

exec.setProgressCallback { step, total in
    print("Progress: \(step)/\(total)")
}

try exec.executeWithProgress(operation)
```

### TypeScript Callbacks

```typescript
import { Executor } from 'hologram';

const exec = new Executor('cpu');

exec.setProgressCallback((step: number, total: number) => {
  console.log(`Progress: ${step}/${total}`);
});

await exec.executeWithProgress(operation);
```

## Stream/Iterator Support

### Python Generators

```python
from hologram import Executor, StreamProcessor

exec = Executor("cpu")
processor = StreamProcessor(exec)

# Stream processing
for chunk in processor.stream_process(large_dataset, chunk_size=1024):
    # Process each chunk
    result = exec.process(chunk)
    yield result
```

### Swift AsyncSequence

```swift
import Hologram

let exec = try Executor(backend: .cpu)
let processor = StreamProcessor(executor: exec)

// Async iteration
for try await chunk in processor.streamProcess(dataset, chunkSize: 1024) {
    let result = try exec.process(chunk)
    // Handle result
}
```

### Rust Iterator

```rust
use hologram_core::{Executor, StreamProcessor};

let exec = Executor::new(BackendType::Cpu)?;
let processor = StreamProcessor::new(exec);

// Iterator-based streaming
for chunk in processor.stream_process(&dataset, 1024) {
    let result = processor.process(chunk)?;
    // Handle result
}
```

## Additional Language Bindings

### Dart FFI

```dart
import 'package:hologram/hologram.dart';

void main() async {
  final exec = await Executor.create(Backend.cpu);

  final buffer = await exec.allocate<double>(1024);
  await buffer.copyFrom([1.0, 2.0, 3.0]);

  await exec.execute(operation);

  final data = await buffer.toList();
  print(data);
}
```

**UniFFI Dart Support:**

```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
uniffi = { version = "0.25", features = ["dart"] }
```

### C# .NET Bindings

```csharp
using Hologram;

class Program
{
    static async Task Main(string[] args)
    {
        var exec = await Executor.CreateAsync(Backend.Cpu);

        var buffer = await exec.AllocateAsync<float>(1024);
        await buffer.CopyFromAsync(new float[] { 1.0f, 2.0f, 3.0f });

        await exec.ExecuteAsync(operation);

        var data = await buffer.ToArrayAsync();
        Console.WriteLine(string.Join(", ", data));
    }
}
```

**UniFFI C# Support:**

```toml
[dependencies]
uniffi = { version = "0.25", features = ["csharp"] }
```

### Java JNI Bindings

```java
import com.hologram.Executor;
import com.hologram.Buffer;

public class Main {
    public static void main(String[] args) {
        Executor exec = Executor.create("cpu");

        Buffer<Float> buffer = exec.allocate(1024, Float.class);
        buffer.copyFrom(new float[] { 1.0f, 2.0f, 3.0f });

        exec.execute(operation);

        float[] data = buffer.toArray();
        System.out.println(Arrays.toString(data));
    }
}
```

## Package Publishing

### Python (PyPI)

**Setup:**

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.0"]
build-backend = "maturin"

[project]
name = "hologram"
version = "0.1.0"
description = "Hologram Compute Acceleration"
requires-python = ">=3.8"
```

**Publishing:**

```bash
# Build wheel
maturin build --release

# Publish to PyPI
maturin publish
```

### npm (TypeScript/JavaScript)

**Setup:**

```json
{
  "name": "hologram",
  "version": "0.1.0",
  "description": "Hologram Compute Acceleration",
  "main": "index.node",
  "types": "index.d.ts",
  "napi": {
    "name": "hologram",
    "triples": {
      "defaults": true,
      "additional": [
        "aarch64-apple-darwin",
        "x86_64-unknown-linux-gnu",
        "x86_64-pc-windows-msvc"
      ]
    }
  }
}
```

**Publishing:**

```bash
# Build native module
npm run build

# Publish to npm
npm publish
```

### Swift Package Manager

**Setup:**

```swift
// Package.swift
let package = Package(
    name: "Hologram",
    products: [
        .library(
            name: "Hologram",
            targets: ["Hologram"]
        ),
    ],
    targets: [
        .binaryTarget(
            name: "HologramFFI",
            path: "./HologramFFI.xcframework"
        ),
        .target(
            name: "Hologram",
            dependencies: ["HologramFFI"]
        ),
    ]
)
```

### Maven Central (Java/Kotlin)

**Setup:**

```xml
<!-- pom.xml -->
<project>
  <groupId>com.hologram</groupId>
  <artifactId>hologram-core</artifactId>
  <version>0.1.0</version>

  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-jar-plugin</artifactId>
        <configuration>
          <archive>
            <manifest>
              <addClasspath>true</addClasspath>
            </manifest>
          </archive>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
```

### Cargo (Rust)

**Publishing:**

```bash
# Build and test
cargo build --release
cargo test --all

# Publish to crates.io
cargo publish -p hologram-ffi
```

### NuGet (C#/.NET)

**Setup:**

```xml
<!-- Hologram.csproj -->
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net6.0</TargetFramework>
    <PackageId>Hologram</PackageId>
    <Version>0.1.0</Version>
    <Authors>Hologram Team</Authors>
    <Description>Hologram Compute Acceleration</Description>
  </PropertyGroup>
</Project>
```

**Publishing:**

```bash
# Build package
dotnet pack -c Release

# Publish to NuGet
dotnet nuget push bin/Release/Hologram.0.1.0.nupkg --api-key <KEY> --source https://api.nuget.org/v3/index.json
```

## Future Enhancements

- [ ] Go bindings (via CGo)
- [ ] Lua FFI bindings
- [ ] R package (via Rcpp)
- [ ] Julia bindings
- [ ] WebAssembly Component Model support
- [ ] Automatic documentation generation for all languages
