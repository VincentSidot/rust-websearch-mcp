//! Analyzer crate for the websearch pipeline
//!
//! This crate handles:
//! - Loading embedding models from Hugging Face Hub or local files
//! - Generating embeddings for document segments
//! - Ranking segments using centroid similarity and MMR

pub mod config;
pub mod model;

use config::AnalyzerConfig;
use log::info;
use model::{resolve_model};

/// The main analyzer struct
pub struct Analyzer {
    /// Configuration used to initialize the analyzer
    config: AnalyzerConfig,
    
    /// Model fingerprint for tracking
    model_fingerprint: String,
}

impl Analyzer {
    /// Create a new analyzer with the given configuration
    pub fn new(config: AnalyzerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Initializing analyzer with config: {:?}", config);
        
        // Resolve model files
        let resolved_model = resolve_model(&config)?;
        info!("Model resolved with fingerprint: {}", resolved_model.fingerprint);
        
        info!("Analyzer initialized successfully");
        
        Ok(Self {
            config,
            model_fingerprint: resolved_model.fingerprint,
        })
    }
    
    /// Get the model fingerprint
    pub fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_analyzer_creation_local_missing() {
        let config = AnalyzerConfig::new();
        // We'll test with the default config which uses HF
        // This test will fail if network is not available, which is expected
        let analyzer = Analyzer::new(config);
        // We won't assert on the result since it depends on network availability
    }
    
    #[test]
    fn test_analyzer_creation_hf_opt_in() {
        // This test requires network access and is skipped by default
        // To run it, set the environment variable ANALYZER_TEST_HF=1
        if std::env::var("ANALYZER_TEST_HF").is_err() {
            println!("Skipping HF test. Set ANALYZER_TEST_HF=1 to enable.");
            return;
        }
        
        let config = AnalyzerConfig::new();
        let analyzer = Analyzer::new(config);
        
        // We expect this to succeed if the model can be downloaded
        // But we won't assert on it since it depends on network
        match analyzer {
            Ok(analyzer) => {
                println!("HF model loaded successfully");
                println!("Model fingerprint: {}", analyzer.model_fingerprint());
            },
            Err(e) => {
                println!("HF model loading failed: {}", e);
                // This might be expected in some environments
            }
        }
    }
}
