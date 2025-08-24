//! Tests for the summarizer crate

use kernel::{AnalyzeResponse, Document, Segment, SegmentScore};
use serde_json::json;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use warp::Filter;

// Helper function to create a test document
fn create_test_document() -> Document {
    Document {
        schema_version: \"1.0.0\".to_string(),
        doc_id: \"test-doc-123\".to_string(),
        url: \"https://example.com\".to_string(),
        title: \"Test Document\".to_string(),
        lang: \"en\".to_string(),
        fetched_at: \"2023-01-01T00:00:00Z\".to_string(),
        segments: vec![
            Segment {
                segment_id: \"seg-1\".to_string(),
                text: \"This is the first segment.\".to_string(),
                path: \"body > p:nth-child(1)\".to_string(),
                position: 0,
            },
            Segment {
                segment_id: \"seg-2\".to_string(),
                text: \"This is the second segment.\".to_string(),
                path: \"body > p:nth-child(2)\".to_string(),
                position: 1,
            },
            Segment {
                segment_id: \"seg-3\".to_string(),
                text: \"This is the third segment.\".to_string(),
                path: \"body > p:nth-child(3)\".to_string(),
                position: 2,
            },
        ],
        hints: None,
    }
}

// Helper function to create a test analysis response
fn create_test_analysis() -> AnalyzeResponse {
    AnalyzeResponse {
        doc_id: \"test-doc-123\".to_string(),
        model_fingerprint: \"model456\".to_string(),
        top_segments: vec![
            SegmentScore {
                segment_id: \"seg-1\".to_string(),
                score_representative: 0.95,
                score_diversity: 0.85,
                reason: \"Highly central\".to_string(),
            },
            SegmentScore {
                segment_id: \"seg-2\".to_string(),
                score_representative: 0.85,
                score_diversity: 0.75,
                reason: \"Diverse content\".to_string(),
            },
        ],
        metrics: core::AnalysisMetrics {
            num_segments: 3,
            top_n: 2,
            mmr_lambda: 0.65,
            avg_pairwise_cosine: 0.3,
        },
    }
}

// Helper function to start a mock API server
async fn start_mock_api_server() -> (String, SocketAddr) {
    // Create a mock response
    let mock_response = warp::post()
        .and(warp::path(\"chat\"))
        .and(warp::path(\"completions\"))
        .map(|| {
            warp::reply::json(&json!({
                \"choices\": [{
                    \"message\": {
                        \"role\": \"assistant\",
                        \"content\": \"This is a test summary.\\n\\n- Point 1\\n- Point 2\"
                    }
                }]
            }))
        });

    // Start the server
    let server = warp::serve(mock_response);
    let listener = TcpListener::bind(\"127.0.0.1:0\").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_addr = format!(\"http://{}\", addr);

    tokio::spawn(async move {
        server.run(listener).await;
    });

    (server_addr, addr)
}

#[tokio::test]
async fn test_summarize_with_mock_api() {
    // Start mock API server
    let (server_addr, _addr) = start_mock_api_server().await;

    // Create test data
    let document = create_test_document();
    let analysis = create_test_analysis();

    // Create summarizer config
    let config = crate::config::SummarizerConfig {
        base_url: server_addr,
        model: \"test-model\".to_string(),
        timeout_ms: 5000,
        temperature: 0.2,
        max_tokens: None,
        style: crate::config::SummaryStyle::AbstractWithBullets,
        api_key: None,
    };

    // Create summarizer
    let summarizer = crate::Summarizer::new(config).unwrap();

    // Summarize
    let response = summarizer.summarize(&document, &analysis).await.unwrap();

    // Check response
    assert_eq!(response.summary_text, \"This is a test summary.\");
    assert!(response.bullets.is_some());
    assert_eq!(response.bullets.as_ref().unwrap().len(), 2);
    assert_eq!(response.bullets.as_ref().unwrap()[0], \"Point 1\");
    assert_eq!(response.bullets.as_ref().unwrap()[1], \"Point 2\");
    assert!(response.citations.is_some());
    assert_eq!(response.citations.as_ref().unwrap().len(), 2);
    assert!(response.guardrails.is_none());
    assert!(response.metrics.processing_time_ms > 0);
    assert!(response.metrics.input_tokens > 0);
    assert!(response.metrics.output_tokens > 0);
}

#[tokio::test]
async fn test_summarize_timeout_fallback() {
    // Create test data
    let document = create_test_document();
    let analysis = create_test_analysis();

    // Create summarizer config with a very short timeout and invalid URL
    let config = crate::config::SummarizerConfig {
        base_url: \"http://10.255.255.1\".to_string(), // Unreachable IP
        model: \"test-model\".to_string(),
        timeout_ms: 100, // Very short timeout
        temperature: 0.2,
        max_tokens: None,
        style: crate::config::SummaryStyle::AbstractWithBullets,
        api_key: None,
    };

    // Create summarizer
    let summarizer = crate::Summarizer::new(config).unwrap();

    // Summarize
    let response = summarizer.summarize(&document, &analysis).await.unwrap();

    // Check that we got a fallback response
    assert!(!response.summary_text.is_empty());
    assert!(response.bullets.is_none());
    assert!(response.citations.is_none());
    assert!(response.guardrails.is_some());
    assert_eq!(
        response.guardrails.as_ref().unwrap().reason,
        Some(\"API error or timeout, using extractive fallback\".to_string())
    );
    assert!(response.metrics.processing_time_ms > 0);
}