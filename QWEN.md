# Rust Web Scraper MCP - Project Context

## Project Overview

This is a Rust-based web scraping tool designed for versatility. It can function as:

1.  A **Command-Line Interface (CLI)** application for direct scraping tasks.
2.  A **library** that can be integrated into other Rust projects.
3.  A **RESTful API server** for scraping webpages programmatically over HTTP.

The core functionality involves fetching a webpage, parsing its HTML content, and extracting key information such as the title, headings, hyperlinks, and the main text content. It supports multiple output formats (JSON, plain text, markdown, readable) and includes optional logging capabilities.

### Key Technologies & Dependencies

*   **Language:** Rust
*   **HTTP Client:** `reqwest` for making asynchronous HTTP requests.
*   **HTML Parsing:** `scraper` crate for parsing and querying HTML documents.
*   **Web Framework:** `axum` for building the REST API server.
*   **Serialization:** `serde` for serializing/deserializing data structures (e.g., to/from JSON).
*   **Async Runtime:** `tokio` for asynchronous operations.
*   **CLI Parser:** `clap` for parsing command-line arguments.
*   **Logging:** Optional `log` facade with `env_logger` implementation (enabled via the `logger` feature).
*   **Configuration:** `dotenvy` for loading environment variables from a `.env` file.

## Building and Running

### Prerequisites

*   Rust and Cargo installed (https://www.rust-lang.org/tools/install).

### Setup

1.  Clone the repository.
2.  Navigate to the project directory.
3.  Run `cargo build` to compile the project.

### CLI Usage

The main binary is defined in `bin/cli.rs` and provides two primary commands:

1.  **Scrape a Webpage:**
    ```bash
    cargo run -- scrape <URL> [--format <json|text|markdown|readable>]
    ```
    Example:
    ```bash
    cargo run -- scrape https://example.com --format json
    ```

2.  **Start the Web Server:**
    ```bash
    cargo run -- serve [--port <PORT>]
    ```
    Example:
    ```bash
    cargo run -- serve --port 8080
    ```

### Library Usage

Other Rust projects can use the scraping functionality by adding this crate as a dependency and calling the `scrape_webpage` function:

```rust
use rust_websearch_mcp::scrape_webpage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = scrape_webpage("https://example.com").await?;
    println!("Title: {:?}", data.title);
    Ok(())
}
```

### API Usage

Start the server using `cargo run -- serve`. By default, it listens on `http://127.0.0.1:3000`.

*   **Health Check:** `GET /`
*   **Scrape Endpoint:** `POST /scrape`
    *   Request Body: `{"url": "https://example.com"}`
    *   Response: A JSON object containing the scraped data (`title`, `headings`, `links`, `text_content`).

### Logger Feature

Logging is enabled by default via the `logger` feature. Control the log level using the `LOG_LEVEL` environment variable (e.g., `LOG_LEVEL=debug`). The logger also respects `dotenvy` for loading environment variables from a `.env` file.

```bash
LOG_LEVEL=debug cargo run --features logger -- scrape https://example.com
```

### Running Tests

Execute the test suite with:

```bash
cargo test
```