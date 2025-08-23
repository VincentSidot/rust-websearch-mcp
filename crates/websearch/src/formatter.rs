use crate::{LinkInfo, ScrapeResponse};
use std::fmt::Write;

/// Formats the scraped data into a more readable plain text format
pub fn format_as_text(scraped_data: &ScrapeResponse) -> String {
    let mut result = String::new();

    // Add title (try multiple selectors for Wikipedia)
    let title = if let Some(title) = &scraped_data.title {
        title.clone()
    } else {
        "(No title found)".to_string()
    };

    writeln!(result, "TITLE: {}", title).unwrap();
    writeln!(result, "{}", "=".repeat(50)).unwrap();

    // Add headings
    if !scraped_data.headings.is_empty() {
        writeln!(result, "HEADINGS:").unwrap();
        for (i, heading) in scraped_data.headings.iter().enumerate() {
            writeln!(result, "  {}. {}", i + 1, clean_text(heading)).unwrap();
        }
        writeln!(result).unwrap();
    }

    // Add text content
    if !scraped_data.text_content.is_empty() {
        writeln!(result, "CONTENT:").unwrap();
        // Clean and wrap text for better readability
        let clean_content = clean_text(&scraped_data.text_content);
        let wrapped_content = wrap_text(&clean_content, 80);
        writeln!(result, "{}", wrapped_content).unwrap();
        writeln!(result).unwrap();
    }

    // Add links
    if !scraped_data.links.is_empty() {
        writeln!(result, "LINKS ({} found):", scraped_data.links.len()).unwrap();
        // Limit to first 20 links for readability
        let link_limit = 20.min(scraped_data.links.len());
        for (i, link) in scraped_data.links.iter().take(link_limit).enumerate() {
            writeln!(result, "  {}. {}", i + 1, format_link(link)).unwrap();
        }
        if scraped_data.links.len() > link_limit {
            writeln!(
                result,
                "  ... and {} more links",
                scraped_data.links.len() - link_limit
            )
            .unwrap();
        }
    } else {
        writeln!(result, "LINKS: (No links found)").unwrap();
    }

    result
}

/// Formats the scraped data into a markdown format
pub fn format_as_markdown(scraped_data: &ScrapeResponse) -> String {
    let mut result = String::new();

    // Add title
    let title = if let Some(title) = &scraped_data.title {
        title.clone()
    } else {
        "(No title found)".to_string()
    };

    writeln!(result, "# {}", title).unwrap();
    writeln!(result).unwrap();

    // Add headings
    if !scraped_data.headings.is_empty() {
        writeln!(result, "## Headings").unwrap();
        for heading in &scraped_data.headings {
            writeln!(result, "- {}", clean_text(heading)).unwrap();
        }
        writeln!(result).unwrap();
    }

    // Add text content
    if !scraped_data.text_content.is_empty() {
        writeln!(result, "## Content").unwrap();
        writeln!(result, "{}", clean_text(&scraped_data.text_content)).unwrap();
        writeln!(result).unwrap();
    }

    // Add links
    if !scraped_data.links.is_empty() {
        writeln!(result, "## Links ({})", scraped_data.links.len()).unwrap();
        // Limit to first 20 links for readability
        let link_limit = 20.min(scraped_data.links.len());
        for link in scraped_data.links.iter().take(link_limit) {
            writeln!(result, "- {}", format_link_markdown(link)).unwrap();
        }
        if scraped_data.links.len() > link_limit {
            writeln!(
                result,
                "- ... and {} more links",
                scraped_data.links.len() - link_limit
            )
            .unwrap();
        }
    }

    result
}

/// Formats a single link for display in text format
fn format_link(link: &LinkInfo) -> String {
    let clean_url = clean_text(&link.url);
    let clean_link_text = clean_text(&link.text);

    if clean_link_text.trim().is_empty() || clean_link_text == clean_url {
        clean_url
    } else {
        format!("{} -> {}", clean_link_text.trim(), clean_url)
    }
}

/// Formats a single link for display in markdown format
fn format_link_markdown(link: &LinkInfo) -> String {
    let clean_url = clean_text(&link.url);
    let clean_link_text = clean_text(&link.text);

    if clean_link_text.trim().is_empty() || clean_link_text == clean_url {
        format!("[{}]({})", clean_url, clean_url)
    } else {
        format!("[{}]({})", clean_link_text.trim(), clean_url)
    }
}

/// Cleans HTML entities and extra whitespace from text
fn clean_text(text: &str) -> String {
    let mut cleaned = text.to_string();

    // Replace common HTML entities
    cleaned = cleaned.replace("&nbsp;", " ");
    cleaned = cleaned.replace("&amp;", "&");
    cleaned = cleaned.replace("&lt;", "<");
    cleaned = cleaned.replace("&gt;", ">");
    cleaned = cleaned.replace("&quot;", "\"");
    cleaned = cleaned.replace("&#39;", "'");

    // Remove extra whitespace
    cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");

    // Trim the result
    cleaned.trim().to_string()
}

/// Wraps text to a specified width for better readability
fn wrap_text(text: &str, width: usize) -> String {
    let mut result = String::new();
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        // If adding this word would exceed the width, start a new line
        if !current_line.is_empty() && current_line.len() + word.len() + 1 > width {
            result.push_str(&current_line);
            result.push('\n');
            current_line.clear();
        }

        // Add the word to the current line
        if !current_line.is_empty() {
            current_line.push(' ');
        }
        current_line.push_str(word);
    }

    // Add the last line
    if !current_line.is_empty() {
        result.push_str(&current_line);
    }

    result
}
