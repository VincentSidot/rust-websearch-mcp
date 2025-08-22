//! Logger module for the web scraper
//!
//! This module provides logging functionality that can be enabled via the `logger` feature.
//! When enabled, it sets up a logger using env_logger with a default level of INFO.
//! The log level can be controlled via the RUST_LOG environment variable.

#[cfg(feature = "logger")]
use log::{debug, error, info, warn};

/// Initializes the logger
///
/// This function sets up the logger with a default level of INFO.
/// The log level can be controlled via the RUST_LOG environment variable.
/// For example: `RUST_LOG=debug` or `RUST_LOG=rust_websearch_mcp=trace`
///
/// This function only has an effect when the `logger` feature is enabled.
#[cfg(feature = "logger")]
pub fn init_logger() {
    // Try to initialize the logger, but don't panic if it's already initialized
    let _ = env_logger::try_init();

    info!("Logger initialized");
}

/// Initializes the logger (no-op when logger feature is disabled)
#[cfg(not(feature = "logger"))]
pub fn init_logger() {
    // This is a no-op when the logger feature is not enabled
}

/// Logs a debug message about starting to scrape a URL
#[cfg(feature = "logger")]
pub fn log_scraping_start(url: &str) {
    debug!("Starting to scrape URL: {}", url);
}

#[cfg(not(feature = "logger"))]
pub fn log_scraping_start(_url: &str) {
    // No-op when logger feature is disabled
}

/// Logs a debug message about completing scraping
#[cfg(feature = "logger")]
pub fn log_scraping_complete(url: &str, title: Option<&String>) {
    debug!(
        "Completed scraping URL: {}, title: {:?}",
        url,
        title.as_ref().map(|s| &s[..])
    );
}

#[cfg(not(feature = "logger"))]
pub fn log_scraping_complete(_url: &str, _title: Option<&String>) {
    // No-op when logger feature is disabled
}
