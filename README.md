# Rust Websearch MCP

⚠️ **Early Development** — This project is currently in active development and should be considered experimental.

A self-hosted pipeline for web scraping and smart summarization using local embeddings and LLM APIs.

## Overview

Rust Websearch MCP is a tool designed to extract and summarize content from web pages using a combination of local processing and LLM APIs. The current implementation focuses on providing a robust web scraping capability with intelligent content selection and summarization.

**Note**: This project is in early development. Features may change significantly as the project matures.

## Current Features

- **Web Scraping**: Clean and segment web pages into coherent chunks
- **Local Analysis**: Generate embeddings and rank content using centroid similarity and MMR (Maximal Marginal Relevance)
- **Optional Reranking**: Improve precision with cross-encoder reranker (off by default)
- **LLM Summarization**: Create concise summaries using OpenAI-compatible APIs
- **CLI Interface**: Run the full pipeline end-to-end
- **Caching**: Built-in caching for embedding computations to improve performance

## Architecture

The pipeline consists of several modular components:

1. **Scraper**: Fetches and parses web content
2. **Analyzer**: Processes content locally using embedding models to select relevant segments
3. **Summarizer**: Generates summaries using LLM APIs
4. **CLI**: Command-line interface to orchestrate the pipeline

## Prerequisites

- Rust 1.70 or later
- Cargo (Rust package manager)
- An API key for an OpenAI-compatible service (for summarization)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd rust-websearch-mcp

# Build the project
cargo build --release
```

## Environment & Credentials

The CLI automatically loads credentials from a `.qwen.env` file at the repository root. This file should contain:

```env
OPENAI_API_KEY=<your_key>
OPENAI_BASE_URL=<your_openai_compatible_base>  # e.g., http://localhost:1234/v1 or https://api.openai.com/v1
OPENAI_MODEL=<default_model_id>  # optional
```

**Loading order (env overrides config):**
1. Load `.qwen.env` (via `dotenvy`) at process start
2. Read environment variables
3. Use `config.toml` values only if the corresponding env var is **not** set

**Security Notes:**
- Do **not** commit `.qwen.env` to version control (it's in `.gitignore`)
- If `.qwen.env` is missing or `OPENAI_API_KEY` is empty, the CLI will print a clear error
- Secrets are redacted in logs

## Usage

### Running the Full Pipeline

```bash
# Run the full pipeline
cargo run --bin cli -- run <url>

# Run the full pipeline with reranking
cargo run --bin cli -- run --rerank <url>

# Run with custom output directory
cargo run --bin cli -- run --out-dir ./my-output <url>
```

## Configuration

Create a `config/config.toml` file based on `config/config.example.toml` to customize behavior.

Example configuration:
```toml
[analyzer]
backend = "onnx"
mmr_lambda = 0.65
top_n = 10
rerank = false

[summarizer]
base_url = "https://api.openai.com/v1"
model = "gpt-3.5-turbo"
timeout_ms = 30000
```

## Reranker

The analyzer supports an optional cross-encoder reranker that can improve the precision of selected segments. The reranker is off by default and can be enabled with the `--rerank` flag.

When enabled:
- The analyzer first selects a shortlist of M segments using centroid similarity and MMR
- The reranker scores each segment in the shortlist against a query derived from the document
- The top N segments are selected based on reranker scores

Configuration options:
- `--rerank-top-m <int>`: Number of segments to consider for reranking (default: 30)

## Output Files

When running the full pipeline, the following files are generated in the output directory:

- `document.json`: The scraped and segmented document
- `analysis.json`: The embedding analysis with selected top segments
- `summary.json`: The final summary with key points

## Development Status

⚠️ **This project is in early development.** While the core functionality is working, you may encounter:

- Breaking changes to APIs and command-line interfaces
- Unimplemented features
- Performance issues
- Bugs

Current development focus:
1. Web scraping functionality
2. Smart content selection using embeddings
3. LLM-based summarization

## Future Work

Planned future enhancements include:

- **Real Web Search**: Integration with search engines to automatically find relevant URLs to scrape based on queries
- **MCP Server**: Setting up a Model Context Protocol (MCP) server to expose the pipeline as a service
- **Enhanced Analysis**: Additional content analysis capabilities
- **Performance Improvements**: Optimizations for faster processing
- **Extended Models**: Support for additional embedding and reranking models

## About This Project

This project was built as a practical experiment using Qwen Code, an AI coding assistant. It serves as a real-world testbed to evaluate Qwen Code's capabilities in building a complete Rust application from scratch. The project demonstrates how AI-assisted development tools can help accelerate the development process while maintaining code quality and best practices.

## Development

```bash
# Build the project
cargo build

# Run tests
cargo test

# Check code quality
cargo clippy -- -D warnings
cargo fmt --check

# Run specific crate tests
cargo test -p websearch
cargo test -p analyzer
```

## Project Structure

```
crates/
├── analyzer/     # Content analysis with embeddings
├── cli/          # Command-line interface
├── kernel/       # Core data structures and traits
├── logger/       # Logging utilities
├── summarizer/   # LLM-based summarization
├── utils/        # Utility functions
└── websearch/    # Web scraping and related functionality
```

## Models

The pipeline uses the following models by default:

- **Embedding Model**: BAAI/bge-small-en-v1.5 (384 dimensions)
- **Reranker Model**: BAAI/bge-reranker-base (when enabled)

These models are automatically downloaded from Hugging Face when first used.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Please note that this project is in early development, so interfaces and functionality may change significantly.

## License

This project is licensed under the MIT License - see the LICENSE file for details.