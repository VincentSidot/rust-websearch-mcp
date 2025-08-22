# Rust Web Scraper MCP

A Rust-based web scraping tool that can be used both as a CLI application and as a library. It also exposes an API for fetching and parsing webpage content.

## Features

- Scrape webpage titles, headings, links, and text content
- Command-line interface for direct usage
- RESTful API for easy integration
- CORS support for web applications
- Asynchronous processing for better performance
- Multiple output formats (JSON, plain text, markdown, readable)
- Optional logging support

## Dependencies

- `reqwest` - HTTP client
- `scraper` - HTML parsing library
- `axum` - Web framework
- `serde` - Serialization/deserialization
- `tokio` - Asynchronous runtime
- `clap` - Command line argument parser
- `log` - Logging facade (optional, enabled with `logger` feature)
- `env_logger` - Logger implementation (optional, enabled with `logger` feature)

## Setup

1. Clone the repository
2. Run `cargo build` to compile the project

## CLI Usage

The CLI provides two main commands:

### Scrape a webpage

```bash
cargo run -- scrape https://example.com
```

You can also get the results in different formats:

```bash
# JSON format
cargo run -- scrape https://example.com --format json

# Plain text format
cargo run -- scrape https://example.com --format text

# Markdown format
cargo run -- scrape https://example.com --format markdown

# Readable format (default)
cargo run -- scrape https://example.com --format readable
```

### Start the web server

```bash
cargo run -- serve
```

You can specify a port:

```bash
cargo run -- serve --port 8080
```

## Logger Feature

The project includes an optional logging feature that can be enabled with the `logger` feature flag:

```bash
# Enable logging with default level (info)
cargo run --features logger -- scrape https://example.com

# Control log level with environment variable
RUST_LOG=debug cargo run --features logger -- scrape https://example.com
```

The logger uses `env_logger` and respects the standard `RUST_LOG` environment variable for controlling log levels.

## API Usage

When running the server, it will start on `http://127.0.0.1:3000` by default.

### Endpoints

- `GET /` - Health check endpoint
- `POST /scrape` - Scrape a webpage

### Scrape a webpage

Send a POST request to `/scrape` with a JSON body:

```json
{
  "url": "https://example.com"
}
```

Response:

```json
{
  "title": "Example Domain",
  "headings": ["Example Domain"],
  "links": [
    {
      "url": "/",
      "text": "More information..."
    }
  ],
  "text_content": "This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission."
}
```

## Library Usage

You can also use the scraping functionality as a library:

```rust
use rust_websearch_mcp::scrape_webpage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = scrape_webpage("https://example.com").await?;
    println!("Title: {:?}", data.title);
    Ok(())
}
```

## Running Tests

Execute `cargo test` to run the test suite.