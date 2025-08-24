//! Logger module for the websearch pipeline
//!
//! This module provides logging functionality.
//! It sets up a logger with a default level of INFO.
//! The log level can be controlled via the RUST_LOG environment variable.

use colored::Colorize;
use std::sync::OnceLock;

use log::Log;

struct Logger {
    level: log::Level,
}

impl Log for Logger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= self.level
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let file = record.file().unwrap_or("unknown");
            let line = record.line().unwrap_or(0);

            let text = match record.level() {
                log::Level::Error => {
                    format!("{} {}:{} - {}", "[ERROR]".red(), file, line, record.args())
                }
                log::Level::Warn => format!(
                    "{} {}:{} - {}",
                    "[WARN]".yellow(),
                    file,
                    line,
                    record.args()
                ),
                log::Level::Info => {
                    format!("{} {}:{} - {}", "[INFO]".green(), file, line, record.args())
                }
                log::Level::Debug => {
                    format!("{} {}:{} - {}", "[DEBUG]".blue(), file, line, record.args())
                }
                log::Level::Trace => format!(
                    "{} {}:{} - {}",
                    "[TRACE]".purple(),
                    file,
                    line,
                    record.args()
                ),
            };
            // Print to stdout for simplicity; in a real application, consider using a more robust logging solution
            println!("{}", text);
        }
    }

    fn flush(&self) {}
}

/// Initializes the logger
///
/// This function sets up the logger with a default level of INFO.
/// The log level can be controlled via the RUST_LOG environment variable.
/// For example: `RUST_LOG=debug` or `RUST_LOG=websearch=trace`
pub fn init_logger() {
    static LOGGER: OnceLock<Logger> = OnceLock::new();

    log::set_logger(LOGGER.get_or_init(|| {
        utils::env::load_env();
        let level = std::env::var("LOG_LEVEL")
            .ok()
            .and_then(|level_str| level_str.to_uppercase().parse::<log::Level>().ok())
            .unwrap_or(log::Level::Info); // Default to INFO if parsing fails

        Logger { level }
    }))
    .expect("Failed to set logger");

    log::info!("Logger initialized");
}

/// Logs the start of a scraping operation
pub fn log_scraping_start(url: &str) {
    log::debug!("Starting to scrape URL: {}", url);
}

/// Logs the completion of a scraping operation
pub fn log_scraping_complete(url: &str, title: Option<&String>) {
    log::debug!(
        "Completed scraping URL: {}, title: {:?}",
        url,
        title.map(|s| &s[..])
    );
}
