//! Logger module for the web scraper
//!
//! This module provides logging functionality that can be enabled via the `logger` feature.
//! When enabled, it sets up a logger using env_logger with a default level of INFO.
//! The log level can be controlled via the RUST_LOG environment variable.

/// Initializes the logger
///
/// This function sets up the logger with a default level of INFO.
/// The log level can be controlled via the RUST_LOG environment variable.
/// For example: `RUST_LOG=debug` or `RUST_LOG=rust_websearch_mcp=trace`
///
/// This function only has an effect when the `logger` feature is enabled.

#[cfg(feature = "logger")]
mod logger_inner {
    use colored::Colorize;
    use std::sync::OnceLock;

    use log::Log;

    use crate::config::get_config;

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

    pub fn init_logger() {
        static LOGGER: OnceLock<Logger> = OnceLock::new();;

        log::set_logger(LOGGER.get_or_init(|| {
            let config = get_config();

            Logger {
                level: config.log_level(),
            }
        }))
        .expect("Failed to set logger");

        log::info!("Logger initialized");
    }
    pub fn log_scraping_start(url: &str) {
        log::debug!("Starting to scrape URL: {}", url);
    }
    pub fn log_scraping_complete(url: &str, title: Option<&String>) {
        log::debug!(
            "Completed scraping URL: {}, title: {:?}",
            url,
            title.map(|s| &s[..])
        );
    }
}

#[cfg(not(feature = "logger"))]
mod logger_inner {
    pub fn init_logger() {
        // This is a no-op when the logger feature is not enabled
    }
    pub fn log_scraping_start(_url: &str) {
        // No-op when logger feature is disabled
    }
    pub fn log_scraping_complete(_url: &str, _title: Option<&String>) {
        // No-op when logger feature is disabled
    }
}

pub use logger_inner::*;
