//! Model loading and resolution for the analyzer crate
//!
//! This module handles resolving model files from either the Hugging Face Hub
//! or a local directory, and provides absolute file paths for loading.

use crate::config::{AnalyzerConfig, ModelConfig};
use log::{debug, info};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

/// Resolved model files
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolvedModel {
    /// Absolute paths to the model files
    pub file_paths: Vec<PathBuf>,
    
    /// Model fingerprint for tracking
    pub fingerprint: String,
}

/// Error type for model resolution
#[derive(Debug)]
pub enum ModelResolutionError {
    DownloadsDisabled,
    LocalModelAccess(String),
    FileNotFound(String),
    DownloadFailed(String),
    HttpError(reqwest::Error),
    IoError(std::io::Error),
    EnvError(env::VarError),
}

impl fmt::Display for ModelResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelResolutionError::DownloadsDisabled => {
                write!(f, "Network downloads are disabled and model files are missing")
            }
            ModelResolutionError::LocalModelAccess(msg) => {
                write!(f, "Failed to access local model directory: {}", msg)
            }
            ModelResolutionError::FileNotFound(file) => {
                write!(f, "Required model file not found: {}", file)
            }
            ModelResolutionError::DownloadFailed(msg) => {
                write!(f, "Failed to download model file: {}", msg)
            }
            ModelResolutionError::HttpError(e) => {
                write!(f, "HTTP error: {}", e)
            }
            ModelResolutionError::IoError(e) => {
                write!(f, "IO error: {}", e)
            }
            ModelResolutionError::EnvError(e) => {
                write!(f, "Environment error: {}", e)
            }
        }
    }
}

impl std::error::Error for ModelResolutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelResolutionError::HttpError(e) => Some(e),
            ModelResolutionError::IoError(e) => Some(e),
            ModelResolutionError::EnvError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for ModelResolutionError {
    fn from(error: reqwest::Error) -> Self {
        ModelResolutionError::HttpError(error)
    }
}

impl From<std::io::Error> for ModelResolutionError {
    fn from(error: std::io::Error) -> Self {
        ModelResolutionError::IoError(error)
    }
}

impl From<env::VarError> for ModelResolutionError {
    fn from(error: env::VarError) -> Self {
        ModelResolutionError::EnvError(error)
    }
}

/// Resolve model files based on the configuration
pub fn resolve_model(config: &AnalyzerConfig) -> Result<ResolvedModel, ModelResolutionError> {
    match &config.model {
        ModelConfig::HuggingFace(hf_config) => {
            resolve_huggingface_model(hf_config, config.allow_downloads)
        },
        ModelConfig::Local(local_config) => {
            resolve_local_model(local_config)
        },
    }
}

/// Resolve a model from the Hugging Face Hub
fn resolve_huggingface_model(
    hf_config: &crate::config::HuggingFaceModelConfig,
    allow_downloads: bool,
) -> Result<ResolvedModel, ModelResolutionError> {
    info!(
        "Resolving model from Hugging Face Hub: {}@{}",
        hf_config.repo_id, hf_config.revision
    );
    
    // Get the Hugging Face cache directory
    let cache_dir = get_hf_cache_dir()?;
    let model_cache_dir = cache_dir
        .join("hub")
        .join(format!("models--{}--{}", hf_config.repo_id.replace('/', "--"), hf_config.revision));
    
    debug!("Model cache directory: {:?}", model_cache_dir);
    
    // Check if all files exist in cache
    let mut file_paths = Vec::new();
    let mut missing_files = Vec::new();
    
    for file in &hf_config.files {
        let file_path = model_cache_dir.join(file);
        if file_path.exists() && file_path.metadata()?.len() > 0 {
            file_paths.push(file_path);
        } else {
            missing_files.push(file.clone());
        }
    }
    
    // If we're missing files and downloads are not allowed, return an error
    if !missing_files.is_empty() && !allow_downloads {
        return Err(ModelResolutionError::DownloadsDisabled);
    }
    
    // Download missing files if needed
    if !missing_files.is_empty() {
        info!("Downloading {} missing model files", missing_files.len());
        download_hf_files(hf_config, &model_cache_dir, &missing_files)?;
        
        // Verify all files are now present
        file_paths.clear();
        for file in &hf_config.files {
            let file_path = model_cache_dir.join(file);
            if file_path.exists() && file_path.metadata()?.len() > 0 {
                file_paths.push(file_path);
            } else {
                return Err(ModelResolutionError::FileNotFound(file.clone()));
            }
        }
    }
    
    Ok(ResolvedModel {
        file_paths,
        fingerprint: format!("{}@{}", hf_config.repo_id, hf_config.revision),
    })
}

/// Resolve a model from a local directory
fn resolve_local_model(
    local_config: &crate::config::LocalModelConfig,
) -> Result<ResolvedModel, ModelResolutionError> {
    info!("Resolving model from local directory: {}", local_config.model_dir);
    
    let model_dir = Path::new(&local_config.model_dir);
    
    // Check if the directory exists
    if !model_dir.exists() {
        return Err(ModelResolutionError::LocalModelAccess(
            format!("Directory does not exist: {}", local_config.model_dir)
        ));
    }
    
    if !model_dir.is_dir() {
        return Err(ModelResolutionError::LocalModelAccess(
            format!("Path is not a directory: {}", local_config.model_dir)
        ));
    }
    
    // Expected files - for now we'll use the default HF file list
    let expected_files = vec![
        "onnx/model.onnx",
        "tokenizer.json",
        "special_tokens_map.json",
    ];
    
    let mut file_paths = Vec::new();
    
    for file in &expected_files {
        let file_path = model_dir.join(file);
        if file_path.exists() && file_path.metadata()?.len() > 0 {
            file_paths.push(file_path);
        } else {
            return Err(ModelResolutionError::FileNotFound(
                format!("Missing file: {}", file)
            ));
        }
    }
    
    Ok(ResolvedModel {
        file_paths,
        fingerprint: format!("local:{}", local_config.model_dir),
    })
}

/// Download files from the Hugging Face Hub
fn download_hf_files(
    hf_config: &crate::config::HuggingFaceModelConfig,
    cache_dir: &Path,
    files: &[String],
) -> Result<(), ModelResolutionError> {
    // Create the cache directory if it doesn't exist
    fs::create_dir_all(cache_dir)?;
    
    // Get Hugging Face token if available
    let token = env::var("HUGGING_FACE_HUB_TOKEN").ok();
    
    // Create HTTP client
    let client = Client::new();
    
    // Base URL for the repository
    let base_url = format!(
        "https://huggingface.co/{}/resolve/{}",
        hf_config.repo_id, hf_config.revision
    );
    
    for file in files {
        let url = format!("{}/{}", base_url, file);
        let file_path = cache_dir.join(file);
        
        info!("Downloading {} to {:?}", file, file_path);
        
        // Create parent directories if needed
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        // Make the HTTP request
        let mut request = client.get(&url);
        
        // Add authorization header if token is provided
        if let Some(ref token) = token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }
        
        let response = request.send()?;
        
        // Check if the request was successful
        if !response.status().is_success() {
            return Err(ModelResolutionError::DownloadFailed(
                format!("HTTP {}: {}", response.status(), url)
            ));
        }
        
        // Save the file
        let mut file_handle = fs::File::create(&file_path)?;
        let content = response.bytes()?;
        std::io::copy(&mut content.as_ref(), &mut file_handle)?;
        
        info!("Downloaded {} successfully", file);
    }
    
    Ok(())
}

/// Get the Hugging Face cache directory
fn get_hf_cache_dir() -> Result<PathBuf, ModelResolutionError> {
    // Check HF_HOME environment variable first
    if let Ok(hf_home) = env::var("HF_HOME") {
        return Ok(PathBuf::from(hf_home));
    }
    
    // Fall back to standard cache directories
    if let Some(cache_dir) = dirs::cache_dir() {
        return Ok(cache_dir.join("huggingface"));
    }
    
    // Last resort: current directory
    Ok(PathBuf::from(".cache/huggingface"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_get_hf_cache_dir() {
        // This test will use the default behavior since we can't easily mock env vars
        let cache_dir = get_hf_cache_dir().unwrap();
        // Should be a valid path, but we can't assert much about its exact value
        assert!(cache_dir.to_str().is_some());
    }
    
    #[test]
    fn test_resolve_local_model_missing_dir() {
        let local_config = crate::config::LocalModelConfig {
            model_dir: "/nonexistent/path".to_string(),
        };
        
        let result = resolve_local_model(&local_config);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ModelResolutionError::LocalModelAccess(_) => {}, // Expected
            _ => panic!("Expected LocalModelAccess error"),
        }
    }
    
    #[test]
    fn test_resolve_hf_model_no_downloads() {
        let hf_config = crate::config::HuggingFaceModelConfig {
            repo_id: "test/repo".to_string(),
            revision: "abcd1234".to_string(),
            files: vec!["model.onnx".to_string()],
        };
        
        // Create a temporary directory to use as a cache that will be empty
        let temp_dir = TempDir::new().unwrap();
        // Temporarily set HF_HOME to point to our temp dir
        env::set_var("HF_HOME", temp_dir.path());
        
        let result = resolve_huggingface_model(&hf_config, false);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ModelResolutionError::DownloadsDisabled => {}, // Expected
            _ => panic!("Expected DownloadsDisabled error"),
        }
        
        // Clean up
        env::remove_var("HF_HOME");
    }
}