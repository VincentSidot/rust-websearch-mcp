#!/bin/bash

# Test the analyzer CLI command

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Set the path to the CLI binary
CLI_PATH="$PROJECT_ROOT/target/debug/websearch-cli"

# Check if the CLI binary exists
if [ ! -f "$CLI_PATH" ]; then
    echo "CLI binary not found at $CLI_PATH"
    echo "Please build the project first with 'cargo build'"
    exit 1
fi

# Run the analyzer on the sample document
echo "Running analyzer on sample document..."
"$CLI_PATH" analyze \
    --top-n 3 \
    --mmr-lambda 0.65 \
    "$PROJECT_ROOT/fixtures/documents/sample.json" \
    > "$PROJECT_ROOT/fixtures/analyze/sample_output.json"

# Check if the command succeeded
if [ $? -eq 0 ]; then
    echo "Analyzer CLI test passed"
    
    # Show the output
    echo "Output:"
    cat "$PROJECT_ROOT/fixtures/analyze/sample_output.json"
else
    echo "Analyzer CLI test failed"
    exit 1
fi