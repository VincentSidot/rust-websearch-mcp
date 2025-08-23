//! Configuration for the summarizer crate
//!
//! This module defines the configuration structures for the summarizer,
//! including API configuration and style options.

use core::Config as CoreConfig;
use serde::{Deserialize, Serialize};
use std::env;

/// Configuration for the summarizer
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SummarizerConfig {
    /// Base URL for the LLM API
    pub base_url: String,

    /// Model to use for summarization
    pub model: String,

    /// Timeout for API requests (in milliseconds)
    pub timeout_ms: u64,

    /// Temperature for sampling (default: 0.2)
    pub temperature: f32,

    /// Maximum number of tokens to generate
    pub max_tokens: Option<u32>,

    /// Style of summary to generate
    pub style: SummaryStyle,

    /// API key for the LLM API (loaded from environment)
    #[serde(skip)]
    pub api_key: Option<String>,
}

/// Style of summary to generate
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SummaryStyle {
    /// Abstract summary with bullet points
    #[serde(rename = "abstract_with_bullets")]
    AbstractWithBullets,
    /// TL;DR style summary
    #[serde(rename = "tldr")]
    TlDr,
    /// Extractive summary (fallback)
    #[serde(rename = "extractive")]
    Extractive,
}

impl SummarizerConfig {
    /// Create a new SummarizerConfig with default values
    pub fn new() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            timeout_ms: 30000,
            temperature: 0.2,
            max_tokens: None,
            style: SummaryStyle::AbstractWithBullets,
            api_key: None,
        }
    }

    /// Create a new SummarizerConfig from a CoreConfig
    pub fn from_core_config(core_config: &CoreConfig) -> Self {
        // Start with defaults
        let mut config = Self::new();

        // Override with values from core config
        for (key, value) in &core_config.settings {
            match key.as_str() {
                "base_url" => config.base_url = value.clone(),
                "model" => config.model = value.clone(),
                "timeout_ms" => {
                    if let Ok(val) = value.parse::<u64>() {
                        config.timeout_ms = val;
                    }
                }
                "temperature" => {
                    if let Ok(val) = value.parse::<f32>() {
                        config.temperature = val;
                    }
                }
                "max_tokens" => {
                    if let Ok(val) = value.parse::<u32>() {
                        config.max_tokens = Some(val);
                    }
                }
                "style" => {
                    config.style = match value.as_str() {
                        "abstract_with_bullets" => SummaryStyle::AbstractWithBullets,
                        "tldr" => SummaryStyle::TlDr,
                        "extractive" => SummaryStyle::Extractive,
                        _ => SummaryStyle::AbstractWithBullets, // Default
                    }
                }
                _ => {} // Ignore unknown keys
            }
        }

        // Load API key from environment
        config.api_key = env::var("OPENAI_API_KEY").ok();

        config
    }
}

impl Default for SummarizerConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::Config as CoreConfig;
    use std::collections::HashMap;

    #[test]
    fn test_default_config() {
        let config = SummarizerConfig::new();
        assert_eq!(config.base_url, "https://api.openai.com/v1");
        assert_eq!(config.model, "gpt-3.5-turbo");
        assert_eq!(config.timeout_ms, 30000);
        assert_eq!(config.temperature, 0.2);
        assert_eq!(config.max_tokens, None);
        assert_eq!(config.style, SummaryStyle::AbstractWithBullets);
        assert_eq!(config.api_key, None);
    }

    #[test]
    fn test_config_from_core_config() {
        let mut core_settings = HashMap::new();
        core_settings.insert("base_url".to_string(), "http://localhost:8080/v1".to_string());
        core_settings.insert("model".to_string(), "llama3".to_string());
        core_settings.insert("timeout_ms".to_string(), "15000".to_string());
        core_settings.insert("temperature".to_string(), "0.5".to_string());
        core_settings.insert("max_tokens".to_string(), "500".to_string());
        core_settings.insert("style".to_string(), "tldr".to_string());
        
        let core_config = CoreConfig {
            settings: core_settings,
        };
        
        let config = SummarizerConfig::from_core_config(&core_config);
        assert_eq!(config.base_url, "http://localhost:8080/v1");
        assert_eq!(config.model, "llama3");
        assert_eq!(config.timeout_ms, 15000);
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.max_tokens, Some(500));
        assert_eq!(config.style, SummaryStyle::TlDr);
    }
}