//! Simple example of using embedding functionality

use reqwest::Client;
use rust_websearch_mcp::config::get_config;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct EmbeddingRequest {
    model: String,
    input: String,
}

#[derive(Serialize, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

async fn generate_embedding(text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let config = get_config();
    let client = Client::new();

    let request_body = EmbeddingRequest {
        model: config.embedding_model().to_string(),
        input: text.to_string(),
    };

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

    if !response.status().is_success() {
        let error_text = response.text().await?;
        return Err(format!("API request failed: {}", error_text).into());
    }

    let embedding_response: EmbeddingResponse = response.json().await?;

    if let Some(embedding_data) = embedding_response.data.first() {
        Ok(embedding_data.embedding.clone())
    } else {
        Err("No embedding data in response".into())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = "This is a sample text for generating an embedding.";
    println!("Generating embedding for: {}", text);

    // Note: This will fail without a valid API key
    match generate_embedding(text).await {
        Ok(embedding) => {
            println!("Generated embedding with {} dimensions", embedding.len());
        }
        Err(e) => {
            eprintln!("Error generating embedding: {}", e);
        }
    }

    Ok(())
}
