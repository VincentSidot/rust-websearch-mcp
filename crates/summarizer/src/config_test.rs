//! Tests for the summarizer config module

use crate::config::{SummarizerConfig, SummaryStyle};
use kernel::Config as CoreConfig;
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