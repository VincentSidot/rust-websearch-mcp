use clap::{Parser, Subcommand};
use serde_json::to_string_pretty;
use std::net::SocketAddr;
use std::process;
use websearch::formatter::{format_as_markdown, format_as_text};
use websearch::logger;
use websearch::scrape_webpage;

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
        Commands::Scrape { url, format } => {
            println!("Scraping {}...", url);

            match scrape_webpage(url).await {
                Ok(data) => {
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
        http::StatusCode,
        response::Json,
        routing::{get, post},
        Router,
    };
    use log::error;
    use serde::{Deserialize, Serialize};
    use tower_http::cors::CorsLayer;

    #[derive(Serialize, Deserialize, Debug)]
    struct ScrapeRequest {
        url: String,
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
    ) -> Result<Json<websearch::ScrapeResponse>, (StatusCode, Json<websearch::ErrorResponse>)> {
        match scrape_webpage(&payload.url).await {
            Ok(data) => Ok(Json(data)),
            Err(e) => {
                error!("Error scraping webpage: {}", e);
                Err((
                    StatusCode::BAD_REQUEST,
                    Json(websearch::ErrorResponse {
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
        .layer(CorsLayer::permissive()); // Allow all CORS for demo purposes

    // Run the server
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    println!("Web Scraper API listening on http://{}", addr);
    println!("Press Ctrl+C to stop the server");

    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;

    Ok(())
}
