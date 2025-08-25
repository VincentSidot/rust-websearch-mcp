//! Integration tests for the CLI

use std::process::Command;
use assert_cmd::prelude::*;
use predicates::prelude::*;

#[test]
fn test_cli_map_reduce_flags() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("cli")?;
    
    cmd.arg("summarize")
        .arg("--help");
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("--map-reduce"))
        .stdout(predicate::str::contains("--max-context-tokens"))
        .stdout(predicate::str::contains("--map-group-tokens"))
        .stdout(predicate::str::contains("--reduce-target-words"))
        .stdout(predicate::str::contains("--concurrency"));
    
    Ok(())
}