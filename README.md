# Websearch Pipeline

A self-hosted pipeline for scraping, analyzing, and summarizing web content.

## Features

- **Scraping**: Fetch and clean web pages
- **Analysis**: Generate embeddings and rank content segments
- **Summarization**: Create concise summaries using LLMs

## Components

- `websearch`: Scrapes web pages and extracts structured content
- `analyzer`: Generates embeddings and ranks content segments
- `summarizer`: Creates summaries using LLMs
- `core`: Shared types and utilities
- `cli`: Command-line interface for the pipeline

## Getting Started

```bash
# Clone the repository
git clone <repository-url>
cd websearch-pipeline

# Build the project
cargo build

# Run the CLI
cargo run --bin cli --help
```

## Configuration

Copy `config/config.example.toml` to `config/config.toml` and modify as needed.

Copy `.env.example` to `.env` and set your API keys and other environment variables.

## License

MIT