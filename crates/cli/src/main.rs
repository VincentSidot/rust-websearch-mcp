use clap::{Parser, Subcommand};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use core::{Document};

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
        
        /// Output file path (stdout if not provided)
        #[clap(short, long)]
        output: Option<String>,
    },
    /// Summarize analyzed content
    Summarize {
        /// Path to the analyzed document
        path: String,
    },
    /// Run the full pipeline
    Run {
        /// The URL to process
        url: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Scrape { url } => {
            println!("Scraping URL: {}", url);
            // TODO: Implement scraping
        }
        Commands::Analyze { path, top_n, mmr_lambda, output } => {
            analyze_document(path, *top_n, *mmr_lambda, output.as_deref())?;
        }
        Commands::Summarize { path } => {
            println!("Summarizing document: {}", path);
            // TODO: Implement summarization
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
    output: Option<&str>
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Analyzing document: {}", path);
    
    // Load document
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let document: Document = serde_json::from_reader(reader)?;
    
    // Create analyzer configuration
    let config = analyzer::config::AnalyzerConfig {
        backend: "onnx".to_string(),
        model: analyzer::config::ModelConfig::HuggingFace(analyzer::config::HuggingFaceModelConfig {
            repo_id: "BAAI/bge-small-en-v1.5".to_string(),
            revision: "5c3b096d65c1aaa0213ced13dac076708b40c077".to_string(),
            files: vec![
                "onnx/model.onnx".to_string(),
                "tokenizer.json".to_string(),
                "special_tokens_map.json".to_string(),
            ],
        }),
        mmr_lambda,
        top_n,
        rerank: false,
        reranker_model_id: "".to_string(),
        allow_downloads: true,
    };
    
    // Create analyzer
    let analyzer = analyzer::Analyzer::new(config)?;
    
    // Analyze document
    let response = analyzer.analyze(&document)?;
    
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
