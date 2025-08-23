//! Simple example of using the websearch functionality

use websearch::scrape_webpage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = "https://example.com";
    println!("Scraping {}...", url);

    match scrape_webpage(url).await {
        Ok(data) => {
            println!("Title: {:?}", data.title);
            println!("Number of headings: {}", data.headings.len());
            println!("Number of links: {}", data.links.len());
            println!(
                "Text content preview: {}",
                if data.text_content.len() > 200 {
                    format!("{}...", &data.text_content[..200])
                } else {
                    data.text_content.clone()
                }
            );
        }
        Err(e) => {
            eprintln!("Error scraping webpage: {}", e);
        }
    }

    Ok(())
}
