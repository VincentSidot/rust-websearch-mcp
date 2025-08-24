use clap::{Parser, Subcommand};
use kernel::{AnalyzeResponse, Document};
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
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

        /// Output directory (default: ./out)
        #[clap(long, default_value = "./out")]
        out_dir: String,

        /// Number of top segments to select
        #[clap(long, default_value = "10")]
        top_n: usize,

        /// MMR lambda parameter (0.0 = centroid only, 1.0 = diversity only)
        #[clap(long, default_value = "0.65")]
        mmr_lambda: f32,

        /// Style of summary to generate
        #[clap(long, default_value = "abstract_with_bullets")]
        style: String,

        /// Timeout for API requests (in milliseconds)
        #[clap(long, default_value = "30000")]
        timeout_ms: u64,

        /// Path to configuration file
        #[clap(long)]
        config: Option<String>,
    },
    /// Cache management commands
    Cache {
        #[clap(subcommand)]
        subcommand: CacheCommands,
    },
}

#[derive(Subcommand)]
enum CacheCommands {
    /// Show cache statistics
    Stats,
    /// Clear the cache
    Clear,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::env::load_env();
    logger::init_logger();

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
            )
            .await?;
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
        Commands::Run {
            url,
            out_dir,
            top_n,
            mmr_lambda,
            style,
            timeout_ms,
            config: _config,
        } => {
            run_pipeline(url, out_dir, *top_n, *mmr_lambda, style, *timeout_ms).await?;
        }
        Commands::Cache { subcommand } => match subcommand {
            CacheCommands::Stats => {
                cache_stats().await?;
            }
            CacheCommands::Clear => {
                cache_clear().await?;
            }
        }
    }

    Ok(())
}

async fn analyze_document(
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
        cache: analyzer::config::CacheConfig::default(),
    };

    // Create analyzer
    let mut analyzer = analyzer::Analyzer::new(config).await?;

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
    logger::init_logger(); // Setup logger
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

    log::trace!(
        "Using OPENAI_BASE_URL={:?}",
        std::env::var("OPENAI_BASE_URL")
    );

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

/// Run the full pipeline: scrape → analyze → summarize
async fn run_pipeline(
    url: &str,
    out_dir: &str,
    top_n: usize,
    mmr_lambda: f32,
    style: &str,
    timeout_ms: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    use websearch::{scrape_webpage, scraped_to_document};

    println!("Running full pipeline for URL: {}", url);
    println!("Output directory: {}", out_dir);
    println!("Top N segments: {}", top_n);
    println!("MMR lambda: {}", mmr_lambda);
    println!("Summary style: {}", style);
    println!("API timeout: {} ms", timeout_ms);

    // Create output directory if it doesn't exist
    fs::create_dir_all(out_dir)?;

    // Record start time
    let start_time = std::time::Instant::now();

    // Step 1: Scrape the webpage
    println!("\n--- Step 1: Scraping ---");
    let scrape_start = std::time::Instant::now();
    let scraped_data = scrape_webpage(url).await?;
    let scrape_duration = scrape_start.elapsed();
    println!("Scraping completed in {:?}", scrape_duration);

    // Convert scraped data to Document
    let document = scraped_to_document(url, &scraped_data);
    println!("Document ID: {}", document.doc_id);
    println!("Document title: {}", document.title);
    println!("Document language: {}", document.lang);
    println!("Number of segments: {}", document.segments.len());

    // Write document to file
    let document_path = Path::new(out_dir).join("document.json");
    let document_file = File::create(&document_path)?;
    let document_writer = BufWriter::new(document_file);
    serde_json::to_writer_pretty(document_writer, &document)?;
    println!("Document written to: {}", document_path.display());

    // Step 2: Analyze the document
    println!("\n--- Step 2: Analyzing ---");
    let analyze_start = std::time::Instant::now();

    // Create analyzer configuration
    let analyzer_config = analyzer::config::AnalyzerConfig {
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
        cache: analyzer::config::CacheConfig::default(),
    };

    // Create analyzer
    let mut analyzer = analyzer::Analyzer::new(analyzer_config).await?;

    // Log model info
    println!("Model ID: {}", analyzer.model_fingerprint());
    println!("Embedding dimension: 384");

    // Analyze document
    let analysis_response = analyzer.analyze(&document)?;
    let analyze_duration = analyze_start.elapsed();
    println!("Analysis completed in {:?}", analyze_duration);
    println!(
        "Selected segments: {}",
        analysis_response.top_segments.len()
    );

    // Write analysis to file
    let analysis_path = Path::new(out_dir).join("analysis.json");
    let analysis_file = File::create(&analysis_path)?;
    let analysis_writer = BufWriter::new(analysis_file);
    serde_json::to_writer_pretty(analysis_writer, &analysis_response)?;
    println!("Analysis written to: {}", analysis_path.display());

    // Step 3: Summarize the document
    println!("\n--- Step 3: Summarizing ---");
    let summarize_start = std::time::Instant::now();

    // Create summarizer configuration
    let style_enum = match style {
        "abstract_with_bullets" => summarizer::config::SummaryStyle::AbstractWithBullets,
        "tldr" => summarizer::config::SummaryStyle::TlDr,
        "extractive" => summarizer::config::SummaryStyle::Extractive,
        _ => summarizer::config::SummaryStyle::AbstractWithBullets, // Default
    };

    let summarizer_config = summarizer::config::SummarizerConfig {
        base_url: std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
        model: std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-3.5-turbo".to_string()),
        timeout_ms,
        temperature: 0.2, // Default temperature
        max_tokens: None, // Not configurable via CLI in this MVP
        style: style_enum,
        api_key: std::env::var("OPENAI_API_KEY").ok(),
    };

    // Log config info
    println!("Model: {}", summarizer_config.model);
    println!("Timeout: {} ms", summarizer_config.timeout_ms);
    println!("Style: {:?}", summarizer_config.style);

    // Create summarizer
    let summarizer = summarizer::Summarizer::new(summarizer_config)?;

    // Summarize document (with fallback on error)
    let summary_response = match summarizer.summarize(&document, &analysis_response).await {
        Ok(response) => response,
        Err(e) => {
            println!(
                "Warning: Summarization failed with error: {}. Using extractive fallback.",
                e
            );
            // Create a fallback summary response
            kernel::SummarizeResponse {
                summary_text: "Summary generation failed. Using extractive fallback.".to_string(),
                bullets: None,
                citations: None,
                guardrails: Some(kernel::GuardrailsInfo {
                    filtered: true,
                    reason: Some(format!("API error: {}", e)),
                }),
                metrics: kernel::SummarizationMetrics {
                    processing_time_ms: summarize_start.elapsed().as_millis() as u64,
                    input_tokens: 0,
                    output_tokens: 0,
                },
            }
        }
    };

    let summarize_duration = summarize_start.elapsed();
    println!("Summarization completed in {:?}", summarize_duration);

    // Write summary to file
    let summary_path = Path::new(out_dir).join("summary.json");
    let summary_file = File::create(&summary_path)?;
    let summary_writer = BufWriter::new(summary_file);
    serde_json::to_writer_pretty(summary_writer, &summary_response)?;
    println!("Summary written to: {}", summary_path.display());

    // Print final summary
    println!("\n--- Summary ---");
    println!("{}", summary_response.summary_text);
    if let Some(bullets) = &summary_response.bullets {
        println!("\nKey Points:");
        for (i, bullet) in bullets.iter().enumerate() {
            println!("{}. {}", i + 1, bullet);
        }
    }

    // Print completion info
    let total_duration = start_time.elapsed();
    println!("\n--- Pipeline completed in {:?} ---", total_duration);
    println!("Output files:");
    println!("  Document:  {}", document_path.display());
    println!("  Analysis:  {}", analysis_path.display());
    println!("  Summary:   {}", summary_path.display());

    Ok(())
}

/// Show cache statistics
async fn cache_stats() -> Result<(), Box<dyn std::error::Error>> {
    // Create default analyzer config to get cache path
    let config = analyzer::config::AnalyzerConfig::new();
    
    // Initialize cache
    let cache = analyzer::cache::EmbeddingCache::new(config.cache)?;
    
    // Get stats
    let stats = cache.get_stats()?;
    
    println!("Cache Statistics:");
    println!("  Entry Count: {}", stats.entry_count);
    println!("  Estimated Size: {} bytes", stats.total_size_bytes);
    
    Ok(())
}

/// Clear the cache
async fn cache_clear() -> Result<(), Box<dyn std::error::Error>> {
    // Create default analyzer config to get cache path
    let config = analyzer::config::AnalyzerConfig::new();
    
    // Initialize cache
    let cache = analyzer::cache::EmbeddingCache::new(config.cache)?;
    
    // Clear cache
    cache.clear()?;
    
    println!("Cache cleared successfully");
    
    Ok(())
}