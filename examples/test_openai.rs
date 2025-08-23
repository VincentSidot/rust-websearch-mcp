//! Test binary to check if OpenAI credentials are working by listing available models

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;

// Use the config from the main application
use rust_websearch_mcp::config::get_config;

#[derive(Serialize, Deserialize, Debug)]
struct Model {
    id: String,
    object: String,
    created: Option<u64>, // Make this optional as not all APIs include it
    owned_by: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ListModelsResponse {
    object: String,
    data: Vec<Model>,
}

async fn list_models() -> Result<ListModelsResponse, Box<dyn Error>> {
    let config = get_config();
    let client = Client::new();

    println!("Using API base: {}", config.openai_api_base());

    let response = client
        .get(format!("{}/models", config.openai_api_base()))
        .header(
            "Authorization",
            format!("Bearer {}", config.openai_api_key()),
        )
        .header("Content-Type", "application/json")
        .send()
        .await?;

    println!("Response status: {}", response.status());

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await?;
        eprintln!("Error response body: {}", error_text);
        return Err(format!("API request failed with status {}: {}", status, error_text).into());
    }

    let text = response.text().await?;

    let models_response: ListModelsResponse = match serde_json::from_str(&text) {
        Ok(response) => response,
        Err(e) => {
            eprintln!("Failed to parse JSON: {}", e);
            return Err(format!("Failed to parse JSON response: {}", e).into());
        }
    };

    Ok(models_response)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("Testing OpenAI credentials by listing available models...");

    match list_models().await {
        Ok(response) => {
            println!("Successfully connected to the API!");
            println!("Found {} models:", response.data.len());
            for model in response.data.iter().take(10) {
                println!("  - {} (owned by {})", model.id, model.owned_by);
            }
            if response.data.len() > 10 {
                println!("  ... and {} more models", response.data.len() - 10);
            }
        }
        Err(e) => {
            eprintln!("Error connecting to the API: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
