use chrono::Utc;
use reqwest::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::error::Error;

pub mod formatter;

#[derive(Serialize, Deserialize, Debug)]
pub struct ScrapeRequest {
    pub url: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ScrapeResponse {
    pub title: Option<String>,
    pub headings: Vec<String>,
    pub links: Vec<LinkInfo>,
    pub text_content: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LinkInfo {
    pub url: String,
    pub text: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ErrorResponse {
    pub error: String,
}

/// Converts scraped data to a Document
pub fn scraped_to_document(url: &str, scraped: &ScrapeResponse) -> kernel::Document {
    use kernel::{Document, Segment};
    use std::collections::HashMap;

    // Get current timestamp in ISO 8601 format
    let fetched_at = Utc::now().to_rfc3339();

    // Use the title from scraped data or derive from URL
    let title = scraped.title.clone().unwrap_or_else(|| {
        // Try to derive a title from the URL
        url.split('/').last().unwrap_or(url).to_string()
    });

    // Split text content into segments (paragraphs)
    let paragraphs: Vec<&str> = scraped
        .text_content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();

    // Create segments from paragraphs
    let segments: Vec<Segment> = paragraphs
        .iter()
        .enumerate()
        .map(|(i, &text)| {
            let segment_text = text.trim().to_string();
            if segment_text.is_empty() {
                return None;
            }

            // Create a segment ID based on the text content
            let segment_id = kernel::compute_segment_id(&segment_text);

            Some(Segment {
                segment_id,
                text: segment_text,
                path: format!("p[{}]", i),
                position: i,
            })
        })
        .flatten()
        .collect();

    // Compute document ID based on content
    let content_for_id = segments
        .iter()
        .map(|s| s.text.clone())
        .collect::<Vec<_>>()
        .join("\n");
    let doc_id = kernel::compute_doc_id(&content_for_id, "paragraphs", "websearch");

    Document {
        schema_version: kernel::SCHEMA_VERSION.to_string(),
        doc_id,
        url: url.to_string(),
        title,
        lang: "en".to_string(), // Default to English for now
        fetched_at,
        segments,
        hints: Some(HashMap::new()), // Empty hints for now
    }
}

/// Scrapes a webpage and returns structured data
///
/// # Arguments
///
/// * `url` - The URL of the webpage to scrape
///
/// # Returns
///
/// * `Result<ScrapeResponse, Box<dyn Error>>` - The scraped data or an error
///
/// # Example
///
/// ```rust
/// use websearch::scrape_webpage;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let data = scrape_webpage("https://example.com").await?;
///     println!("Title: {:?}", data.title);
///     Ok(())
/// }
/// ```
pub async fn scrape_webpage(url: &str) -> Result<ScrapeResponse, Box<dyn Error>> {
    // Create a client with a proper user-agent header
    let client = Client::builder()
        .user_agent("websearch/0.1 (https://github.com/your-username/rust-websearch-mcp)")
        .build()?;

    // Fetch the webpage content
    let response = client.get(url).send().await?;
    let html_content = response.text().await?;

    // Parse the HTML
    let document = Html::parse_document(&html_content);

    // Extract title (try multiple selectors for better compatibility)
    let title = extract_title(&document);

    // Extract headings (h1, h2, h3) with better filtering
    let headings = extract_headings(&document);

    // Extract links with better filtering
    let links = extract_links(&document);

    // Extract text content with better filtering for Wikipedia
    let text_content = extract_text_content(&document);

    Ok(ScrapeResponse {
        title,
        headings,
        links,
        text_content,
    })
}

/// Extracts title with multiple fallback selectors
fn extract_title(document: &Html) -> Option<String> {
    // Try the standard title tag first
    let title_selector = Selector::parse("title").unwrap();
    if let Some(title_element) = document.select(&title_selector).next() {
        let title = title_element.inner_html();
        if !title.trim().is_empty() {
            return Some(title);
        }
    }

    // Try Wikipedia-specific title selector
    let wiki_title_selector = Selector::parse("h1.firstHeading").unwrap();
    if let Some(title_element) = document.select(&wiki_title_selector).next() {
        let title = title_element.inner_html();
        if !title.trim().is_empty() {
            return Some(title);
        }
    }

    // Try any h1 as fallback
    let h1_selector = Selector::parse("h1").unwrap();
    if let Some(title_element) = document.select(&h1_selector).next() {
        let title = title_element.inner_html();
        if !title.trim().is_empty() {
            return Some(title);
        }
    }

    None
}

/// Extracts headings with filtering
fn extract_headings(document: &Html) -> Vec<String> {
    let mut headings = Vec::new();

    // For Wikipedia, we want to be more selective about headings
    // Include h1, h2, h3 but exclude some common navigation elements
    let heading_selectors = vec!["h1", "h2", "h3"];

    for selector_str in heading_selectors {
        let selector = Selector::parse(selector_str).unwrap();
        for element in document.select(&selector) {
            let text = element.inner_html();
            // Skip empty headings
            if !text.trim().is_empty() {
                headings.push(text);
            }
        }
    }

    headings
}

/// Extracts links with filtering
fn extract_links(document: &Html) -> Vec<LinkInfo> {
    let mut links = Vec::new();
    let link_selector = Selector::parse("a[href]").unwrap();

    for element in document.select(&link_selector) {
        if let Some(href) = element.value().attr("href") {
            let text = element.inner_html();

            // Filter out empty or obviously non-content links
            if !href.trim().is_empty() &&
               !href.starts_with("#") &&  // Skip anchor links
               !(text.trim().is_empty() && !href.starts_with("http"))
            {
                // Skip empty text internal links
                links.push(LinkInfo {
                    url: href.to_string(),
                    text: text,
                });
            }
        }
    }

    links
}

/// Extracts text content with better filtering for Wikipedia
fn extract_text_content(document: &Html) -> String {
    let mut paragraphs = Vec::new();

    // For Wikipedia, focus on the main content area
    let content_selectors = vec![
        "div#mw-content-text p",  // Wikipedia content paragraphs
        "div.mw-parser-output p", // Alternate Wikipedia content selector
        "div.content p",          // Generic content div
        "article p",              // Article tag paragraphs
        "main p",                 // Main tag paragraphs
        "p",                      // Fallback to all paragraphs
    ];

    for selector_str in content_selectors {
        let selector = Selector::parse(selector_str).unwrap();
        for element in document.select(&selector) {
            let text = element.text().collect::<Vec<_>>().join(" ");
            // Only include paragraphs with substantial content
            if text.trim().len() > 10 {
                paragraphs.push(text);
            }
        }

        // If we found content with this selector, break
        if !paragraphs.is_empty() {
            break;
        }
    }

    paragraphs.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scrape_webpage() {
        // This test requires an internet connection
        // You might want to use a mock HTTP client in a real test
        let result = scrape_webpage("https://example.com").await;
        assert!(result.is_ok());

        let data = result.unwrap();
        assert!(data.title.is_some());
        assert!(!data.headings.is_empty());
    }
}
