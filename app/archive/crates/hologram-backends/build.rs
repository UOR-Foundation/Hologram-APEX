//! Build script for hologram-backends
//!
//! This script compiles CUDA kernels to PTX format when the 'cuda' feature is enabled.
//!
//! Pipeline:
//! 1. Detect nvcc compiler
//! 2. Compile src/backends/cuda/kernels.cu → PTX
//! 3. Embed PTX in binary via include_bytes!()
//!
//! Output: OUT_DIR/atlas_kernels.ptx (if CUDA available)

#[cfg(feature = "cuda")]
use std::env;
#[cfg(feature = "cuda")]
use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/backends/cuda/kernels.cu");
    println!("cargo:rerun-if-changed=build.rs");

    // Only compile CUDA kernels if the 'cuda' feature is enabled
    #[cfg(feature = "cuda")]
    compile_cuda_kernels();
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let cuda_src = Path::new("src/backends/cuda/kernels.cu");
    let ptx_output = Path::new(&out_dir).join("atlas_kernels.ptx");

    // Check if kernels.cu exists
    if !cuda_src.exists() {
        println!("cargo:warning=⚠️  CUDA kernels source not found: {:?}", cuda_src);
        return;
    }

    // Try to detect nvcc
    let nvcc = find_nvcc();
    if nvcc.is_none() {
        println!("cargo:warning=⚠️  nvcc not found in PATH. CUDA kernel compilation skipped.");
        println!("cargo:warning=⚠️  CUDA backend will fall back to CPU at runtime.");
        println!("cargo:warning=⚠️  Install CUDA Toolkit to enable GPU acceleration.");

        // Create empty PTX file to avoid compilation errors
        std::fs::write(&ptx_output, "// CUDA kernels not compiled - nvcc not found\n")
            .expect("Failed to write placeholder PTX");
        return;
    }

    let nvcc_path = nvcc.unwrap();
    println!("cargo:warning=🔨 Compiling CUDA kernels with nvcc...");
    println!("cargo:warning=   nvcc: {:?}", nvcc_path);
    println!("cargo:warning=   input: {:?}", cuda_src);
    println!("cargo:warning=   output: {:?}", ptx_output);

    // Compile kernels.cu to PTX
    let status = Command::new(&nvcc_path)
        .arg("--ptx") // Compile to PTX
        .arg("-o")
        .arg(&ptx_output) // Output file
        .arg(cuda_src) // Input file
        .arg("--std=c++11") // C++11 standard
        .arg("-O3") // Optimization level 3
        .arg("--use_fast_math") // Fast math operations
        .arg("-arch=sm_52") // Minimum compute capability 5.2 (Maxwell+)
        .arg("--gpu-architecture=compute_52") // Virtual architecture
        .arg("--gpu-code=sm_52,sm_60,sm_70,sm_75,sm_80,sm_86,sm_89,sm_90") // Target multiple architectures
        .status();

    match status {
        Ok(status) if status.success() => {
            println!("cargo:warning=✅ CUDA kernels compiled successfully");
            println!(
                "cargo:warning=   PTX size: {} bytes",
                std::fs::metadata(&ptx_output).map(|m| m.len()).unwrap_or(0)
            );
        }
        Ok(status) => {
            println!(
                "cargo:warning=❌ nvcc compilation failed with exit code: {:?}",
                status.code()
            );
            println!("cargo:warning=   CUDA backend will fall back to CPU at runtime.");

            // Create placeholder PTX
            std::fs::write(&ptx_output, "// CUDA kernel compilation failed\n")
                .expect("Failed to write placeholder PTX");
        }
        Err(e) => {
            println!("cargo:warning=❌ Failed to run nvcc: {}", e);
            println!("cargo:warning=   CUDA backend will fall back to CPU at runtime.");

            // Create placeholder PTX
            std::fs::write(&ptx_output, "// CUDA kernel compilation failed\n")
                .expect("Failed to write placeholder PTX");
        }
    }
}

#[cfg(feature = "cuda")]
fn find_nvcc() -> Option<PathBuf> {
    // Try to find nvcc in PATH
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout);
            let path = path_str.trim();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    // Try common CUDA installation locations
    let common_paths = ["/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc", "/opt/cuda/bin/nvcc"];

    for path in &common_paths {
        if Path::new(path).exists() {
            return Some(PathBuf::from(path));
        }
    }

    // Try CUDA_HOME environment variable
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let nvcc_path = PathBuf::from(cuda_home).join("bin").join("nvcc");
        if nvcc_path.exists() {
            return Some(nvcc_path);
        }
    }

    // Try CUDA_PATH environment variable (Windows)
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc_path = PathBuf::from(cuda_path).join("bin").join("nvcc.exe");
        if nvcc_path.exists() {
            return Some(nvcc_path);
        }
    }

    None
}
