use clap::{Parser, Subcommand};
use core::{AnalyzeResponse, Document};
use dotenvy::dotenv;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use tokio;

/// CLI for the websearch pipeline
#[derive(Parser)]
#[clap(name = "websearch-cli")]
#[clap(about = "A CLI tool for web scraping, analysis, and summarization", long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scrape a webpage
    Scrape {
        /// The URL to scrape
        url: String,
    },
    /// Analyze scraped content
    Analyze {
        /// Path to the scraped document
        path: String,

        /// Number of top segments to select
        #[clap(long, default_value = "10")]
        top_n: usize,

        /// MMR lambda parameter (0.0 = centroid only, 1.0 = diversity only)
        #[clap(long, default_value = "0.65")]
        mmr_lambda: f32,

        /// Batch size for inference
        #[clap(long, default_value = "8")]
        batch_size: usize,

        /// Maximum sequence length
        #[clap(long, default_value = "512")]
        max_seq_len: usize,

        /// Output file path (stdout if not provided)
        #[clap(short, long)]
        output: Option<String>,
    },
    /// Summarize analyzed content
    Summarize {
        /// Path to the analyzed document (AnalyzeResponse JSON)
        #[clap(long)]
        analysis: String,

        /// Path to the original document (Document JSON)
        #[clap(long)]
        document: String,

        /// Style of summary to generate
        #[clap(long, default_value = "abstract_with_bullets")]
        style: String,

        /// Timeout for API requests (in milliseconds)
        #[clap(long, default_value = "30000")]
        timeout_ms: u64,

        /// Temperature for sampling
        #[clap(long, default_value = "0.2")]
        temperature: f32,
    },
    /// Run the full pipeline
    Run {
        /// The URL to process
        url: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    if let Err(err) = dotenv() {
        log::warn!("An error occured while loading .env file: {err}");
    }

    let cli = Cli::parse();

    match &cli.command {
        Commands::Scrape { url } => {
            println!("Scraping URL: {}", url);
            // TODO: Implement scraping
        }
        Commands::Analyze {
            path,
            top_n,
            mmr_lambda,
            batch_size,
            max_seq_len,
            output,
        } => {
            analyze_document(
                path,
                *top_n,
                *mmr_lambda,
                *batch_size,
                *max_seq_len,
                output.as_deref(),
            )?;
        }
        Commands::Summarize {
            analysis,
            document,
            style,
            timeout_ms,
            temperature,
        } => {
            summarize_document(analysis, document, style, *timeout_ms, *temperature).await?;
        }
        Commands::Run { url } => {
            println!("Running full pipeline for URL: {}", url);
            // TODO: Implement full pipeline
        }
    }

    Ok(())
}

fn analyze_document(
    path: &str,
    top_n: usize,
    mmr_lambda: f32,
    batch_size: usize,
    max_seq_len: usize,
    output: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Analyzing document: {}", path);

    // Load document
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let document: Document = serde_json::from_reader(reader)?;

    // Create analyzer configuration
    let config = analyzer::config::AnalyzerConfig {
        backend: "onnx".to_string(),
        model: analyzer::config::ModelConfig::HuggingFace(
            analyzer::config::HuggingFaceModelConfig {
                repo_id: "BAAI/bge-small-en-v1.5".to_string(),
                revision: "main".to_string(),
                files: vec![
                    "onnx/model.onnx".to_string(),
                    "tokenizer.json".to_string(),
                    "special_tokens_map.json".to_string(),
                ],
            },
        ),
        mmr_lambda,
        top_n,
        rerank: false,
        reranker_model_id: "".to_string(),
        allow_downloads: true,
    };

    // Create analyzer
    let mut analyzer = analyzer::Analyzer::new(config)?;

    // Log model info
    println!("Model ID: {}", analyzer.model_fingerprint());
    println!("Embedding dimension: 384");
    println!("Batch size: {}", batch_size);
    println!("Max sequence length: {}", max_seq_len);

    // Record start time
    let start_time = std::time::Instant::now();

    // Analyze document
    let response = analyzer.analyze(&document)?;

    // Record end time
    let duration = start_time.elapsed();
    println!("Document analysis completed in {:?}", duration);

    // Write output
    if let Some(output_path) = output {
        let file = File::create(output_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &response)?;
    } else {
        serde_json::to_writer_pretty(std::io::stdout(), &response)?;
    }

    println!("Analysis complete");
    Ok(())
}

async fn summarize_document(
    analysis_path: &str,
    document_path: &str,
    style: &str,
    timeout_ms: u64,
    temperature: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Summarizing document");
    println!("Analysis path: {}", analysis_path);
    println!("Document path: {}", document_path);

    // Load document
    let file = File::open(document_path)?;
    let reader = BufReader::new(file);
    let document: Document = serde_json::from_reader(reader)?;

    // Load analysis
    let file = File::open(analysis_path)?;
    let reader = BufReader::new(file);
    let analysis: AnalyzeResponse = serde_json::from_reader(reader)?;

    // Validate that the analysis matches the document
    if analysis.doc_id != document.doc_id {
        return Err("Document ID mismatch between document and analysis".into());
    }

    // Create summarizer configuration
    let style_enum = match style {
        "abstract_with_bullets" => summarizer::config::SummaryStyle::AbstractWithBullets,
        "tldr" => summarizer::config::SummaryStyle::TlDr,
        "extractive" => summarizer::config::SummaryStyle::Extractive,
        _ => summarizer::config::SummaryStyle::AbstractWithBullets, // Default
    };

    let config = summarizer::config::SummarizerConfig {
        base_url: std::env::var("OPENAI_BASE_URL").expect("Missing OPENAI_BASE_URL"),
        model: std::env::var("OPENAI_MODEL").expect("Missing OPENAI_MODEL"),
        timeout_ms,
        temperature,
        max_tokens: None, // Not configurable via CLI in this MVP
        style: style_enum,
        api_key: std::env::var("OPENAI_API_KEY").ok(),
    };

    // Log config info
    println!("Model: {}", config.model);
    println!("Timeout: {} ms", config.timeout_ms);
    println!("Temperature: {}", config.temperature);
    println!("Style: {:?}", config.style);

    // Create summarizer
    let summarizer = summarizer::Summarizer::new(config)?;

    // Log number of selected segments
    println!("Selected segments: {}", analysis.top_segments.len());

    // Record start time
    let start_time = std::time::Instant::now();

    // Summarize document
    let response = summarizer.summarize(&document, &analysis).await?;

    // Record end time
    let duration = start_time.elapsed();
    println!("Document summarization completed in {:?}", duration);

    // Write output to stdout
    serde_json::to_writer_pretty(std::io::stdout(), &response)?;

    println!("Summarization complete");
    Ok(())
}
