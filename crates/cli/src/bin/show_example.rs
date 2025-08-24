use kernel::{AnalyzeResponse, Document};
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load document
    let document_file = File::open("fixtures/example_document.json")?;
    let document_reader = BufReader::new(document_file);
    let document: Document = serde_json::from_reader(document_reader)?;

    // Load analysis
    let analysis_file = File::open("fixtures/example_analysis.json")?;
    let analysis_reader = BufReader::new(analysis_file);
    let analysis: AnalyzeResponse = serde_json::from_reader(analysis_reader)?;

    // Print information about the document and analysis
    println!("Document ID: {}", document.doc_id);
    println!("Document Title: {}", document.title);
    println!("Number of segments: {}", document.segments.len());
    println!("Selected segments: {}", analysis.top_segments.len());

    for (i, segment_score) in analysis.top_segments.iter().enumerate() {
        println!("  {}. Segment ID: {}", i + 1, segment_score.segment_id);
        // Find the segment text
        if let Some(segment) = document
            .segments
            .iter()
            .find(|s| s.segment_id == segment_score.segment_id)
        {
            println!("     Text: {}", segment.text);
        }
    }

    Ok(())
}
