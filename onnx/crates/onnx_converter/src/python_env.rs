//! Python Environment Manager
//!
//! Manages an isolated Python virtual environment for the converter.
//! Automatically installs required packages on first use.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tokio::process::Command;

/// Get the hologram cache directory for Python environment
pub fn get_python_cache_dir() -> Result<PathBuf> {
    let cache_dir = if let Some(home) = dirs::home_dir() {
        home.join(".cache/hologram-onnx/python")
    } else {
        // Fallback to temp directory if home not available
        std::env::temp_dir().join("hologram-onnx/python")
    };

    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("Failed to create cache directory: {}", cache_dir.display()))?;

    Ok(cache_dir)
}

/// Check if Python virtual environment exists and is valid
pub async fn check_venv_exists() -> Result<bool> {
    let venv_dir = get_python_cache_dir()?.join("venv");
    let python_bin = if cfg!(windows) {
        venv_dir.join("Scripts/python.exe")
    } else {
        venv_dir.join("bin/python")
    };

    Ok(python_bin.exists())
}

/// Create Python virtual environment
pub async fn create_venv(verbose: bool) -> Result<PathBuf> {
    let cache_dir = get_python_cache_dir()?;
    let venv_dir = cache_dir.join("venv");

    if venv_dir.exists() {
        if verbose {
            println!("  Virtual environment already exists");
        }
        return Ok(venv_dir);
    }

    if verbose {
        println!("  Creating Python virtual environment...");
    }

    // Create venv using python3 -m venv
    let output = Command::new("python3")
        .arg("-m")
        .arg("venv")
        .arg(&venv_dir)
        .output()
        .await
        .context("Failed to create virtual environment. Is python3-venv installed?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Failed to create venv:\n{}", stderr);
    }

    if verbose {
        println!("  ✓ Virtual environment created");
    }

    Ok(venv_dir)
}

/// Get path to Python executable in venv
pub fn get_venv_python(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts/python.exe")
    } else {
        venv_dir.join("bin/python")
    }
}

/// Get path to pip executable in venv
pub fn get_venv_pip(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts/pip.exe")
    } else {
        venv_dir.join("bin/pip")
    }
}

/// Install Python packages in virtual environment
pub async fn install_packages(packages: &[&str], verbose: bool) -> Result<()> {
    if packages.is_empty() {
        return Ok(());
    }

    // Create venv if it doesn't exist
    let venv_dir = create_venv(verbose).await?;
    let pip = get_venv_pip(&venv_dir);

    if verbose {
        println!("  Installing Python packages: {}", packages.join(", "));
        println!("  This may take a few minutes on first run...");
    }

    // Upgrade pip first
    let output = Command::new(&pip)
        .arg("install")
        .arg("--upgrade")
        .arg("pip")
        .output()
        .await?;

    if !output.status.success() && verbose {
        eprintln!("  Warning: Could not upgrade pip");
    }

    // Install packages
    let output = Command::new(&pip)
        .arg("install")
        .args(packages)
        .output()
        .await
        .context("Failed to install packages")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Failed to install packages:\n{}", stderr);
    }

    if verbose {
        println!("  ✓ Packages installed successfully");
    }

    Ok(())
}

/// Check which packages are missing from venv
pub async fn check_missing_packages(packages: &[&str]) -> Result<Vec<String>> {
    let venv_exists = check_venv_exists().await?;
    if !venv_exists {
        // No venv, all packages are missing
        return Ok(packages.iter().map(|s| s.to_string()).collect());
    }

    let venv_dir = get_python_cache_dir()?.join("venv");
    let python = get_venv_python(&venv_dir);

    let mut missing = Vec::new();

    for package in packages {
        let output = Command::new(&python)
            .arg("-c")
            .arg(format!("import {}", package))
            .output()
            .await?;

        if !output.status.success() {
            missing.push(package.to_string());
        }
    }

    Ok(missing)
}

/// Setup Python environment (create venv + install packages)
pub async fn setup_python_environment(verbose: bool) -> Result<PathBuf> {
    let packages = vec![
        "torch",
        "onnx",
        "onnxruntime",
        "transformers",
        "diffusers",
        "safetensors",
        "accelerate",
        "onnxscript",
    ];

    if verbose {
        println!("Setting up Python environment...");
    }

    // Check which packages are missing
    let missing = check_missing_packages(&packages).await?;

    if !missing.is_empty() {
        if verbose {
            println!("  Installing {} packages...", missing.len());
        }

        // Convert to &str for install_packages
        let missing_refs: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
        install_packages(&missing_refs, verbose).await?;
    } else if verbose {
        println!("  ✓ All packages already installed");
    }

    let venv_dir = get_python_cache_dir()?.join("venv");
    Ok(get_venv_python(&venv_dir))
}

/// Get Python executable path (using venv if available)
pub async fn get_python_executable(auto_setup: bool, verbose: bool) -> Result<PathBuf> {
    if auto_setup {
        // Setup venv and return its python
        setup_python_environment(verbose).await
    } else {
        // Just check if venv exists, otherwise use system python3
        if check_venv_exists().await? {
            let venv_dir = get_python_cache_dir()?.join("venv");
            Ok(get_venv_python(&venv_dir))
        } else {
            Ok(PathBuf::from("python3"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_dir() {
        let cache_dir = get_python_cache_dir().unwrap();
        assert!(cache_dir.to_string_lossy().contains("hologram-onnx"));
    }

    #[tokio::test]
    async fn test_check_venv() {
        // Should not error even if venv doesn't exist
        let exists = check_venv_exists().await.unwrap();
        println!("Venv exists: {}", exists);
    }
}
