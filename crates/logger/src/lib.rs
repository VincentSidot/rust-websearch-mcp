//! Logger module for the websearch pipeline
//!
//! This module provides logging functionality.
//! It sets up a logger with a default level of INFO.
//! The log level can be controlled via the RUST_LOG environment variable.
use std::sync::OnceLock;

use colored::Colorize;
use log::{Level, LevelFilter, Log};

struct Logger {
    default: LevelFilter,
    // (target_prefix, filter), e.g., ("html5ever", Off), ("reqwest", Warn), ("websearch", Trace)
    overrides: Vec<(String, LevelFilter)>,
}

impl Logger {
    fn effective_filter(&self, target: &str) -> LevelFilter {
        // Longest-prefix match like env_logger
        let mut best: Option<(&str, LevelFilter)> = None;
        for (prefix, lf) in &self.overrides {
            if target.starts_with(prefix) {
                match best {
                    None => best = Some((prefix, *lf)),
                    Some((prev, _)) if prefix.len() > prev.len() => best = Some((prefix, *lf)),
                    _ => {}
                }
            }
        }
        best.map(|(_, lf)| lf).unwrap_or(self.default)
    }

    fn level_allowed(level: Level, filter: LevelFilter) -> bool {
        match filter {
            LevelFilter::Off => false,
            LevelFilter::Error => level <= Level::Error,
            LevelFilter::Warn => level <= Level::Warn,
            LevelFilter::Info => level <= Level::Info,
            LevelFilter::Debug => level <= Level::Debug,
            LevelFilter::Trace => true,
        }
    }
}

impl Log for Logger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        let target = metadata.target();
        let filt = self.effective_filter(target);
        Self::level_allowed(metadata.level(), filt)
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

const DEFAULT_RUST_LOG: &str = "html5ever=warn,tokenizer=warn,reqwest=warn,ort=warn";

/// Parse RUST_LOG-like specs: e.g. "info,html5ever=off,reqwest=warn,websearch=trace"
fn parse_filters() -> (LevelFilter, Vec<(String, LevelFilter)>) {
    let mut default: LevelFilter = LevelFilter::Info;
    let mut overrides: Vec<(String, LevelFilter)> = Vec::new();

    for part in DEFAULT_RUST_LOG
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if let Some((target, lvl)) = part.split_once('=') {
            let lf = parse_level_filter(lvl).unwrap_or(LevelFilter::Info);
            let name = target.trim().to_string();

            if let Some(index) = overrides.iter().position(|raw| &raw.0 == &name) {
                overrides[index] = (name, lf);
            } else {
                overrides.push((name, lf));
            }
        }
    }

    if let Ok(spec) = std::env::var("RUST_LOG") {
        for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            if let Some((target, lvl)) = part.split_once('=') {
                let lf = parse_level_filter(lvl).unwrap_or(LevelFilter::Info);
                let name = target.trim().to_string();

                if let Some(index) = overrides.iter().position(|raw| &raw.0 == &name) {
                    overrides[index] = (name, lf);
                } else {
                    overrides.push((name, lf));
                }
            } else {
                default = parse_level_filter(part).unwrap_or(LevelFilter::Info);
            }
        }
    }

    (default, overrides)
}

fn parse_level_filter(s: &str) -> Option<LevelFilter> {
    match s.trim().to_ascii_lowercase().as_str() {
        "off" => Some(LevelFilter::Off),
        "error" => Some(LevelFilter::Error),
        "warn" | "warning" => Some(LevelFilter::Warn),
        "info" => Some(LevelFilter::Info),
        "debug" => Some(LevelFilter::Debug),
        "trace" => Some(LevelFilter::Trace),
        _ => None,
    }
}

/// Initializes the logger
///
/// This function sets up the logger with a default level of INFO.
/// The log level can be controlled via the RUST_LOG environment variable.
/// For example: `RUST_LOG=debug` or `RUST_LOG=websearch=trace`
pub fn init_logger() {
    static LOGGER: OnceLock<Logger> = OnceLock::new();

    utils::env::load_env();

    let (default, overrides) = parse_filters();

    // Set global max to the highest level we might emit (so overrides can work).
    let global_max = std::iter::once(default)
        .chain(overrides.iter().map(|(_, lf)| *lf))
        .max_by(|a, b| {
            // Manual ordering because LevelFilter doesn't guarantee Ord
            let rank = |lf: &LevelFilter| match lf {
                LevelFilter::Off => 0,
                LevelFilter::Error => 1,
                LevelFilter::Warn => 2,
                LevelFilter::Info => 3,
                LevelFilter::Debug => 4,
                LevelFilter::Trace => 5,
            };
            rank(a).cmp(&rank(b))
        })
        .unwrap_or(LevelFilter::Info);

    log::set_logger(LOGGER.get_or_init(|| Logger { default, overrides }))
        .map(|()| log::set_max_level(global_max))
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
