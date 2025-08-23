//! Configuration for the analyzer crate
//!
//! This module defines the configuration structures for the analyzer,
//! including model configuration for both Hugging Face Hub and local models.

use core::Config as CoreConfig;
use serde::{Deserialize, Serialize};
use std::env;

/// Configuration for the analyzer
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnalyzerConfig {
    /// Backend to use for embeddings ("onnx" or "candle")
    pub backend: String,
    
    /// Model configuration
    #[serde(flatten)]
    pub model: ModelConfig,
    
    /// MMR lambda parameter (0.0 = centroid only, 1.0 = diversity only)
    pub mmr_lambda: f32,
    
    /// Number of top segments to select
    pub top_n: usize,
    
    /// Whether to use reranking
    pub rerank: bool,
    
    /// Model ID for reranking (if enabled)
    pub reranker_model_id: String,
    
    /// Whether to allow network downloads
    #[serde(default = "default_allow_downloads")]
    pub allow_downloads: bool,
}

/// Configuration for model loading
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ModelConfig {
    /// Hugging Face Hub model configuration
    HuggingFace(HuggingFaceModelConfig),
    
    /// Local model configuration
    Local(LocalModelConfig),
}

/// Configuration for a Hugging Face Hub model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HuggingFaceModelConfig {
    /// Repository ID on Hugging Face Hub
    pub repo_id: String,
    
    /// Revision (commit SHA) to use - required for reproducibility
    pub revision: String,
    
    /// Files to download from the repository
    pub files: Vec<String>,
}

/// Configuration for a local model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LocalModelConfig {
    /// Directory containing the model files
    pub model_dir: String,
}

impl AnalyzerConfig {
    /// Create a new AnalyzerConfig with default values
    pub fn new() -> Self {
        Self {
            backend: "onnx".to_string(),
            model: ModelConfig::HuggingFace(HuggingFaceModelConfig {
                repo_id: "BAAI/bge-small-en-v1.5".to_string(),
                revision: "5c3b096d65c1aaa0213ced13dac076708b40c077".to_string(), // pinned revision
                files: vec![
                    "onnx/model.onnx".to_string(),
                    "tokenizer.json".to_string(),
                    "special_tokens_map.json".to_string(),
                ],
            }),
            mmr_lambda: 0.5,
            top_n: 10,
            rerank: false,
            reranker_model_id: "".to_string(),
            allow_downloads: default_allow_downloads(),
        }
    }
    
    /// Create a new AnalyzerConfig from a CoreConfig
    pub fn from_core_config(core_config: &CoreConfig) -> Self {
        // Start with defaults
        let mut config = Self::new();
        
        // Override with values from core config
        for (key, value) in &core_config.settings {
            match key.as_str() {
                "backend" => config.backend = value.clone(),
                "mmr_lambda" => {
                    if let Ok(val) = value.parse::<f32>() {
                        config.mmr_lambda = val;
                    }
                },
                "top_n" => {
                    if let Ok(val) = value.parse::<usize>() {
                        config.top_n = val;
                    }
                },
                "rerank" => {
                    if let Ok(val) = value.parse::<bool>() {
                        config.rerank = val;
                    }
                },
                "reranker_model_id" => config.reranker_model_id = value.clone(),
                "allow_downloads" => {
                    if let Ok(val) = value.parse::<bool>() {
                        config.allow_downloads = val;
                    }
                },
                _ => {} // Ignore unknown keys
            }
        }
        
        config
    }
    
    /// Get the model fingerprint for this configuration
    pub fn model_fingerprint(&self) -> String {
        match &self.model {
            ModelConfig::HuggingFace(hf_config) => {
                format!("{}@{}", hf_config.repo_id, hf_config.revision)
            },
            ModelConfig::Local(local_config) => {
                format!("local:{}", local_config.model_dir)
            }
        }
    }
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Default value for allow_downloads - true unless explicitly disabled
fn default_allow_downloads() -> bool {
    // Check environment variable first
    if let Ok(val) = env::var("ANALYZER_ALLOW_DOWNLOADS") {
        val.parse::<bool>().unwrap_or(true)
    } else {
        // Default to true
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_default_config() {
        let config = AnalyzerConfig::new();
        assert_eq!(config.backend, "onnx");
        assert_eq!(config.mmr_lambda, 0.5);
        assert_eq!(config.top_n, 10);
        assert_eq!(config.rerank, false);
        assert_eq!(config.reranker_model_id, "");
        
        // Check HF model config
        match &config.model {
            ModelConfig::HuggingFace(hf_config) => {
                assert_eq!(hf_config.repo_id, "BAAI/bge-small-en-v1.5");
                assert_eq!(hf_config.revision, "5c3b096d65c1aaa0213ced13dac076708b40c077");
                assert_eq!(hf_config.files.len(), 3);
            },
            _ => panic!("Expected HuggingFace model config"),
        }
    }
    
    #[test]
    fn test_model_fingerprint_hf() {
        let config = AnalyzerConfig::new();
        let fingerprint = config.model_fingerprint();
        assert_eq!(fingerprint, "BAAI/bge-small-en-v1.5@5c3b096d65c1aaa0213ced13dac076708b40c077");
    }
    
    #[test]
    fn test_model_fingerprint_local() {
        let mut config = AnalyzerConfig::new();
        config.model = ModelConfig::Local(LocalModelConfig {
            model_dir: "/path/to/model".to_string(),
        });
        
        let fingerprint = config.model_fingerprint();
        assert_eq!(fingerprint, "local:/path/to/model");
    }
    
    #[test]
    fn test_from_core_config() {
        let mut core_settings = HashMap::new();
        core_settings.insert("backend".to_string(), "candle".to_string());
        core_settings.insert("mmr_lambda".to_string(), "0.7".to_string());
        core_settings.insert("top_n".to_string(), "15".to_string());
        core_settings.insert("rerank".to_string(), "true".to_string());
        core_settings.insert("reranker_model_id".to_string(), "my-reranker".to_string());
        
        let core_config = CoreConfig {
            settings: core_settings,
        };
        
        let config = AnalyzerConfig::from_core_config(&core_config);
        assert_eq!(config.backend, "candle");
        assert_eq!(config.mmr_lambda, 0.7);
        assert_eq!(config.top_n, 15);
        assert_eq!(config.rerank, true);
        assert_eq!(config.reranker_model_id, "my-reranker");
    }
}