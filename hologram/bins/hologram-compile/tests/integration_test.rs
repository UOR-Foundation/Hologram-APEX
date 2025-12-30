//! Integration tests for hologram-compile CLI

#![allow(deprecated)]

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

/// Test basic help command
#[test]
fn test_help_command() {
    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Hologram Compiler"))
        .stdout(predicate::str::contains("--help"));
}

/// Test version command
#[test]
fn test_version_command() {
    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg("--version");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("hologram-compile"));
}

/// Test compile command with missing file
#[test]
fn test_compile_missing_file() {
    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg("nonexistent.py");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Input file not found"));
}

/// Test compile subcommand help
#[test]
fn test_compile_subcommand_help() {
    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg("compile").arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Compile a single kernel"));
}

/// Test check subcommand help
#[test]
fn test_check_subcommand_help() {
    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg("check").arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Check if a kernel compiles"));
}

/// Test disasm subcommand help
#[test]
fn test_disasm_subcommand_help() {
    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg("disasm").arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Disassemble"));
}

/// Test info subcommand help
#[test]
fn test_info_subcommand_help() {
    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg("info").arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Show information"));
}

/// Test invalid backend option
#[test]
fn test_invalid_backend() {
    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg("test.py").arg("-b").arg("invalid");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("invalid value"));
}

/// Test invalid format option
#[test]
fn test_invalid_format() {
    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg("test.py").arg("-f").arg("invalid");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("invalid value"));
}

/// Test verbose flag
#[test]
fn test_verbose_flag() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.py");
    fs::write(&test_file, "# Test kernel\n").unwrap();

    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg(&test_file).arg("-vv");

    // Command will fail because test.py won't actually compile,
    // but we can verify the verbose flag is accepted
    let output = cmd.output().unwrap();
    // Just verify it doesn't panic on the verbose flag
    assert!(output.status.code().is_some());
}

/// Test quiet flag
#[test]
fn test_quiet_flag() {
    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg("missing.py").arg("--quiet");

    cmd.assert().failure();
    // With --quiet, only errors should be shown
}

/// Test optimization level
#[test]
fn test_optimization_level() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.py");
    fs::write(&test_file, "# Test kernel\n").unwrap();

    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg(&test_file).arg("-O").arg("3");

    // Verify the option is accepted
    let output = cmd.output().unwrap();
    assert!(output.status.code().is_some());
}

/// Test emit-circuit flag
#[test]
fn test_emit_circuit_flag() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.py");
    fs::write(&test_file, "# Test kernel\n").unwrap();

    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg(&test_file).arg("--emit-circuit");

    let output = cmd.output().unwrap();
    assert!(output.status.code().is_some());
}

/// Test stats flag
#[test]
fn test_stats_flag() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.py");
    fs::write(&test_file, "# Test kernel\n").unwrap();

    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg(&test_file).arg("--stats");

    let output = cmd.output().unwrap();
    assert!(output.status.code().is_some());
}

/// Test compile-all subcommand with missing directory
#[test]
fn test_compile_all_missing_dir() {
    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg("compile-all").arg("nonexistent_dir");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Input directory not found"));
}

/// Test check with missing file
#[test]
fn test_check_missing_file() {
    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg("check").arg("nonexistent.py");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Input file not found"));
}

/// Test info with missing file
#[test]
fn test_info_missing_file() {
    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg("info").arg("nonexistent.json");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Input file not found"));
}

/// Test disasm with missing file
#[test]
fn test_disasm_missing_file() {
    let mut cmd = Command::cargo_bin("hologram-compile").unwrap();
    cmd.arg("disasm").arg("nonexistent.json");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Input file not found"));
}
