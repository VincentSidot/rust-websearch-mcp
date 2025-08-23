//! Embedding module for generating text embeddings using OpenAI-compatible APIs.
//!
//! This module provides functionality to generate embeddings for text content
//! using models like "granite-embedding" through an OpenAI-compatible API endpoint.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;

use crate::config::get_config;

#[derive(Serialize, Deserialize, Debug)]
struct EmbeddingRequest {
    model: String,
    input: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: u32,
    object: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    model: String,
    object: String,
    usage: UsageData,
}

#[derive(Serialize, Deserialize, Debug)]
struct UsageData {
    prompt_tokens: u32,
    total_tokens: u32,
}

/// Generate an embedding for the given text using the configured API
///
/// # Arguments
///
/// * `text` - The text to generate an embedding for
///
/// # Returns
///
/// * `Result<Vec<f32>, Box<dyn Error>>` - The embedding vector or an error
pub async fn generate_embedding(text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let config = get_config();

    // Create HTTP client
    let client = Client::new();

    // Prepare request
    let request_body = EmbeddingRequest {
        model: config.embedding_model().to_string(),
        input: text.to_string(),
    };

    // Send request to API
    let response = client
        .post(format!("{}/embeddings", config.openai_api_base()))
        .header(
            "Authorization",
            format!("Bearer {}", config.openai_api_key()),
        )
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await?;

    // Check if request was successful
    if !response.status().is_success() {
        let error_text = response.text().await?;
        return Err(format!("API request failed: {}", error_text).into());
    }

    // Parse response
    let embedding_response: EmbeddingResponse = response.json().await?;

    // Return the first embedding
    if let Some(embedding_data) = embedding_response.data.first() {
        Ok(embedding_data.embedding.clone())
    } else {
        Err("No embedding data in response".into())
    }
}

/// Generate a summary of scraped content using embeddings
///
/// # Arguments
///
/// * `title` - The title of the page
/// * `content` - The text content of the page
///
/// # Returns
///
/// * `Result<String, Box<dyn Error>>` - A summary of the content or an error
pub async fn summarize_content(
    title: Option<&str>,
    content: &str,
) -> Result<String, Box<dyn Error>> {
    // For now, we'll just return a simple summary
    // In a more advanced implementation, we could use the embeddings for similarity search
    // or other NLP tasks

    let summary = if let Some(title) = title {
        format!(
            "Title: {}\
Content preview: {}",
            title,
            if content.len() > 200 {
                format!("{}...", &content[..200])
            } else {
                content.to_string()
            }
        )
    } else {
        format!(
            "Content preview: {}",
            if content.len() > 200 {
                format!("{}...", &content[..200])
            } else {
                content.to_string()
            }
        )
    };

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_embedding() {
        // This test requires a valid API key and endpoint
        // For now, we'll just test that the function compiles
        // In a real test, we would mock the HTTP client
        let _ = generate_embedding("test text").await;
    }

    #[tokio::test]
    async fn test_summarize_content() {
        let summary = summarize_content(Some("Test Title"), "This is a test content").await;
        assert!(summary.is_ok());
    }
}
