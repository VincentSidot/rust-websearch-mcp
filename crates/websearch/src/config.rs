use std::sync::OnceLock;

use dotenvy::dotenv;
use log::Level;

pub struct Config {
    log_level: Level,
    openai_api_key: String,
    openai_api_base: String,
    embedding_model: String,
}

static CONFIG: OnceLock<Config> = OnceLock::new();

pub fn get_config() -> &'static Config {
    CONFIG.get_or_init(|| Config::init())
}

impl Config {
    fn init() -> Self {
        // Load environment variables from .env file if it exists
        _ = dotenv().inspect_err(|e| log::warn!("Unable to load the .env file: {e}"));

        let log_level = std::env::var("LOG_LEVEL")
            .ok()
            .and_then(|level_str| level_str.to_uppercase().parse::<Level>().ok())
            .unwrap_or(Level::Info); // Default to INFO if parsing fails

        let openai_api_key =
            std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "YOUR_API_KEY_HERE".to_string());

        let openai_api_base = std::env::var("OPENAI_API_BASE")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

        let embedding_model = std::env::var("EMBEDDING_MODEL")
            .expect("EMBEDDING_MODEL must be set in the environment");

        Config {
            log_level,
            openai_api_key,
            openai_api_base,
            embedding_model,
        }
    }

    pub fn log_level(&self) -> Level {
        self.log_level
    }

    pub fn openai_api_key(&self) -> &str {
        &self.openai_api_key
    }

    pub fn openai_api_base(&self) -> &str {
        &self.openai_api_base
    }

    pub fn embedding_model(&self) -> &str {
        &self.embedding_model
    }
}
