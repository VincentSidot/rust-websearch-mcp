//! Test binary to demonstrate web scraping functionality

use websearch::scrape_webpage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing web scraping functionality...");

    let url = "https://example.com";
    match scrape_webpage(url).await {
        Ok(data) => {
            println!("Successfully scraped: {}", url);
            println!("Title: {:?}", data.title);
            println!("Number of headings: {}", data.headings.len());
            println!("Number of links: {}", data.links.len());
        }
        Err(e) => {
            eprintln!("Error scraping {}: {}", url, e);
            std::process::exit(1);
        }
    }

    Ok(())
}
