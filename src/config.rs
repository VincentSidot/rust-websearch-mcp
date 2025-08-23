use std::sync::OnceLock;

use dotenvy::dotenv;
use log::Level;

pub struct Config {
    log_level: Level,
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
            .and_then(|level_str| level_str.to_uppercase().parse::<Level>().ok());

        Config {
            log_level: log_level.unwrap_or(Level::Info), // Default to INFO if parsing fails
        }
    }

    pub fn log_level(&self) -> Level {
        self.log_level
    }
}
