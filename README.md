# Rust Websearch MCP

A self-hosted pipeline for web scraping, local analysis with embeddings, and LLM summarization.

## Features

- **Web Scraping**: Clean and segment web pages into coherent chunks
- **Local Analysis**: Generate embeddings and rank content using centroid similarity and MMR
- **Optional Reranking**: Improve precision with cross-encoder reranker (off by default)
- **LLM Summarization**: Create concise summaries using OpenAI-compatible APIs
- **CLI Interface**: Run each step individually or the full pipeline end-to-end

## Environment & Credentials

The CLI automatically loads credentials from a `.qwen.env` file at the repository root. This file should contain:

```
OPENAI_API_KEY=<your_key>
OPENAI_API_BASE=<your_openai_compatible_base>  # e.g., http://localhost:1234/v1 or https://api.openai.com/v1
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

```bash
# Run the full pipeline
cargo run --bin cli -- run <url>

# Run the full pipeline with reranking
cargo run --bin cli -- run --rerank <url>

# Scrape a webpage
cargo run --bin cli -- scrape <url>

# Analyze scraped content
cargo run --bin cli -- analyze <document.json>

# Analyze scraped content with reranking
cargo run --bin cli -- analyze --rerank <document.json>

# Summarize analyzed content
cargo run --bin cli -- summarize --analysis <analysis.json> --document <document.json>

# Cache management
cargo run --bin cli -- cache stats  # Show cache statistics
cargo run --bin cli -- cache clear  # Clear the cache
```

## Configuration

Create a `config/config.toml` file based on `config/config.example.toml` to customize behavior.

## Reranker

The analyzer supports an optional cross-encoder reranker that can improve the precision of selected segments. 
The reranker is off by default and can be enabled with the `--rerank` flag.

When enabled:
- The analyzer first selects a shortlist of M segments using centroid similarity and MMR
- The reranker scores each segment in the shortlist against a query derived from the document
- The top N segments are selected based on reranker scores

Configuration options:
- `--rerank-top-m <int>`: Number of segments to consider for reranking (default: 30)

## Development

```bash
# Build the project
cargo build

# Run tests
cargo test

# Check code quality
cargo clippy -D warnings
cargo fmt --check
```