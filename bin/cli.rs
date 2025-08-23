use clap::{Parser, Subcommand};
use rust_websearch_mcp::formatter::{format_as_markdown, format_as_text};
use rust_websearch_mcp::logger;
use rust_websearch_mcp::scrape_webpage;
use rust_websearch_mcp::embedding;
use serde_json::to_string_pretty;
use std::net::SocketAddr;
use std::process;

#[derive(Parser)]
#[clap(name = "Web Scraper CLI")]
#[clap(about = "A CLI tool for scraping webpages", long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scrape a webpage and display the results
    Scrape {
        /// The URL to scrape
        #[clap(value_parser)]
        url: String,

        /// Output format (json, text, markdown, or readable)
        #[clap(short, long, value_parser, default_value = "readable")]
        format: String,

        /// Generate embedding for the content
        #[clap(long, action)]
        embed: bool,
    },

    /// Start the web server API
    Serve {
        /// Port to run the server on
        #[clap(short, long, value_parser, default_value = "3000")]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the logger
    logger::init_logger();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Scrape { url, format, embed } => {
            println!("Scraping {}...", url);

            match scrape_webpage(url).await {
                Ok(data) => {
                    // Generate embedding/summary if requested
                    if *embed {
                        println!("Generating summary...");
                        match embedding::summarize_content(data.title.as_ref().map(|s| s.as_str()), &data.text_content).await {
                            Ok(summary) => {
                                println!("Content Summary:");
                                println!("{}", summary);
                            }
                            Err(e) => {
                                eprintln!("Error generating summary: {}", e);
                            }
                        }
                    } else {
                        // Format output as requested
                        match format.as_str() {
                            "json" => {
                                println!("{}", to_string_pretty(&data)?);
                            }
                            "text" => {
                                println!("Title: {:?}", data.title);
                                println!("Headings: {:?}", data.headings);
                                println!("Number of links: {}", data.links.len());
                                println!(
                                    "Text content preview: {}",
                                    if data.text_content.len() > 200 {
                                        format!("{}...", &data.text_content[..200])
                                    } else {
                                        data.text_content.clone()
                                    }
                                );
                            }
                            "markdown" => {
                                println!("{}", format_as_markdown(&data));
                            }
                            "readable" | _ => {
                                println!("{}", format_as_text(&data));
                            }
                        }
                    }
                },
                Err(e) => {
                    eprintln!("Error scraping webpage: {}", e);
                    process::exit(1);
                }
            }
        }

        Commands::Serve { port } => {
            start_server(*port).await?;
        }
    }

    Ok(())
}

async fn start_server(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    use axum::{
        Router,
        http::StatusCode,
        response::Json,
        routing::{get, post},
    };
    use log::error;
    use serde::{Deserialize, Serialize};
    use tower_http::cors::CorsLayer;

    #[derive(Serialize, Deserialize, Debug)]
    struct ScrapeRequest {
        url: String,
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct EmbeddingRequest {
        text: String,
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct EmbeddingResponse {
        embedding: Vec<f32>,
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct SummaryRequest {
        title: Option<String>,
        content: String,
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct SummaryResponse {
        summary: String,
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct HealthCheckResponse {
        status: String,
    }

    async fn health_check() -> Json<HealthCheckResponse> {
        Json(HealthCheckResponse {
            status: "OK".to_string(),
        })
    }

    async fn scrape_handler(
        Json(payload): Json<ScrapeRequest>,
    ) -> Result<
        Json<rust_websearch_mcp::ScrapeResponse>,
        (StatusCode, Json<rust_websearch_mcp::ErrorResponse>),
    > {
        match scrape_webpage(&payload.url).await {
            Ok(data) => Ok(Json(data)),
            Err(e) => {
                error!("Error scraping webpage: {}", e);
                Err((
                    StatusCode::BAD_REQUEST,
                    Json(rust_websearch_mcp::ErrorResponse {
                        error: e.to_string(),
                    }),
                ))
            }
        }
    }

    async fn embedding_handler(
        Json(payload): Json<EmbeddingRequest>,
    ) -> Result<
        Json<EmbeddingResponse>,
        (StatusCode, Json<rust_websearch_mcp::ErrorResponse>),
    > {
        match embedding::generate_embedding(&payload.text).await {
            Ok(embedding) => Ok(Json(EmbeddingResponse { embedding })),
            Err(e) => {
                error!("Error generating embedding: {}", e);
                Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(rust_websearch_mcp::ErrorResponse {
                        error: e.to_string(),
                    }),
                ))
            }
        }
    }

    async fn summary_handler(
        Json(payload): Json<SummaryRequest>,
    ) -> Result<
        Json<SummaryResponse>,
        (StatusCode, Json<rust_websearch_mcp::ErrorResponse>),
    > {
        match embedding::summarize_content(
            payload.title.as_ref().map(|s| s.as_str()), 
            &payload.content
        ).await {
            Ok(summary) => Ok(Json(SummaryResponse { summary })),
            Err(e) => {
                error!("Error generating summary: {}", e);
                Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(rust_websearch_mcp::ErrorResponse {
                        error: e.to_string(),
                    }),
                ))
            }
        }
    }

    // Build our application with routes
    let app = Router::new()
        .route("/", get(health_check))
        .route("/scrape", post(scrape_handler))
        .route("/embed", post(embedding_handler))
        .route("/summarize", post(summary_handler))
        .layer(CorsLayer::permissive()); // Allow all CORS for demo purposes

    // Run the server
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    println!("Web Scraper API listening on http://{}", addr);
    println!("Press Ctrl+C to stop the server");

    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;

    Ok(())
}
