//! Core types and shared functionality for the websearch pipeline

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Current schema version for documents
pub const SCHEMA_VERSION: &str = "1.0.0";

/// Document schema
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Document {
    /// Schema version
    pub schema_version: String,
    /// Unique document ID
    pub doc_id: String,
    /// Source URL
    pub url: String,
    /// Document title
    pub title: String,
    /// Document language
    pub lang: String,
    /// Timestamp when fetched
    pub fetched_at: String, // Consider using chrono::DateTime<Utc> for a more robust type
    /// Segments of the document
    pub segments: Vec<Segment>,
    /// Optional hints for processing
    pub hints: Option<HashMap<String, String>>,
}

/// Segment schema
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Segment {
    /// Unique segment ID
    pub segment_id: String,
    /// Text content of the segment
    pub text: String,
    /// Path to the segment in the document (e.g., CSS selector or XPath)
    pub path: String,
    /// Position of the segment in the document
    pub position: usize,
}

/// Response from the analyzer
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnalyzeResponse {
    /// Document ID
    pub doc_id: String,
    /// Model fingerprint used for analysis
    pub model_fingerprint: String,
    /// Top segments with scores and reasons
    pub top_segments: Vec<SegmentScore>,
    /// Metrics about the analysis
    pub metrics: AnalysisMetrics,
}

/// Score and reason for a segment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SegmentScore {
    /// Segment ID
    pub segment_id: String,
    /// Score for representativeness (centroid cosine similarity)
    pub score_representative: f32,
    /// Score for diversity (MMR term)
    pub score_diversity: f32,
    /// Reason for selection
    pub reason: String,
}

/// Metrics about the analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnalysisMetrics {
    /// Total number of segments
    pub num_segments: usize,
    /// Number of top segments selected
    pub top_n: usize,
    /// MMR lambda parameter used
    pub mmr_lambda: f32,
    /// Average pairwise cosine similarity among selected segments (redundancy indicator)
    pub avg_pairwise_cosine: f32,
}

/// Coverage information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CoverageInfo {
    /// Total number of segments
    pub total_segments: usize,
    /// Number of segments analyzed
    pub analyzed_segments: usize,
}

/// Cache information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheInfo {
    /// Whether the result was served from cache
    pub hit: bool,
    /// Cache key used
    pub key: Option<String>,
}

/// Response from the summarizer
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SummarizeResponse {
    /// Summary text
    pub summary_text: String,
    /// Optional bullet points
    pub bullets: Option<Vec<String>>,
    /// Optional citations
    pub citations: Option<Vec<Citation>>,
    /// Optional guardrails information
    pub guardrails: Option<GuardrailsInfo>,
    /// Metrics about the summarization
    pub metrics: SummarizationMetrics,
}

/// Citation information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Citation {
    /// Segment ID cited
    pub segment_id: String,
    /// Start position in the summary text
    pub start: usize,
    /// End position in the summary text
    pub end: usize,
}

/// Guardrails information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GuardrailsInfo {
    /// Whether the summary was filtered
    pub filtered: bool,
    /// Reason for filtering (if applicable)
    pub reason: Option<String>,
}

/// Metrics about the summarization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SummarizationMetrics {
    /// Time taken to generate the summary (in milliseconds)
    pub processing_time_ms: u64,
    /// Number of input tokens
    pub input_tokens: usize,
    /// Number of output tokens
    pub output_tokens: usize,
}

/// Placeholder for error types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Error {
    /// Error message
    pub message: String,
    /// Error code
    pub code: String,
}

/// Placeholder for configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    /// Configuration key-value pairs
    pub settings: HashMap<String, String>,
}

/// Compute a document ID based on normalized text, chunking params, and model fingerprint
pub fn compute_doc_id(text: &str, chunk_params: &str, model_fingerprint: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(text.as_bytes());
    hasher.update(chunk_params.as_bytes());
    hasher.update(model_fingerprint.as_bytes());
    hasher.finalize().to_hex().to_string()
}

/// Compute a segment ID based on normalized segment text
pub fn compute_segment_id(text: &str) -> String {
    let hash = blake3::hash(text.as_bytes());
    hash.to_hex().to_string()
}

// Schema changelog
// 1.0.0 - Initial schema version

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_serde_json() {
        let doc = Document {
            schema_version: SCHEMA_VERSION.to_string(),
            doc_id: "doc123".to_string(),
            url: "https://example.com".to_string(),
            title: "Example".to_string(),
            lang: "en".to_string(),
            fetched_at: "2023-01-01T00:00:00Z".to_string(),
            segments: vec![],
            hints: None,
        };

        let json = serde_json::to_string(&doc).unwrap();
        let doc2: Document = serde_json::from_str(&json).unwrap();
        assert_eq!(doc, doc2);
    }

    #[test]
    fn test_document_serde_msgpack() {
        let doc = Document {
            schema_version: SCHEMA_VERSION.to_string(),
            doc_id: "doc123".to_string(),
            url: "https://example.com".to_string(),
            title: "Example".to_string(),
            lang: "en".to_string(),
            fetched_at: "2023-01-01T00:00:00Z".to_string(),
            segments: vec![],
            hints: None,
        };

        let bytes = rmp_serde::to_vec(&doc).unwrap();
        let doc2: Document = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(doc, doc2);
    }

    #[test]
    fn test_segment_serde_json() {
        let segment = Segment {
            segment_id: "seg123".to_string(),
            text: "This is a segment".to_string(),
            path: "body > p".to_string(),
            position: 0,
        };

        let json = serde_json::to_string(&segment).unwrap();
        let segment2: Segment = serde_json::from_str(&json).unwrap();
        assert_eq!(segment, segment2);
    }

    #[test]
    fn test_segment_serde_msgpack() {
        let segment = Segment {
            segment_id: "seg123".to_string(),
            text: "This is a segment".to_string(),
            path: "body > p".to_string(),
            position: 0,
        };

        let bytes = rmp_serde::to_vec(&segment).unwrap();
        let segment2: Segment = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(segment, segment2);
    }

    #[test]
    fn test_analyze_response_serde_json() {
        let response = AnalyzeResponse {
            doc_id: "doc123".to_string(),
            model_fingerprint: "model456".to_string(),
            top_segments: vec![SegmentScore {
                segment_id: "seg123".to_string(),
                score_representative: 0.95,
                score_diversity: 0.85,
                reason: "Highly central".to_string(),
            }],
            metrics: AnalysisMetrics {
                num_segments: 10,
                top_n: 5,
                mmr_lambda: 0.65,
                avg_pairwise_cosine: 0.3,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        let response2: AnalyzeResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(response, response2);
    }

    #[test]
    fn test_analyze_response_serde_msgpack() {
        let response = AnalyzeResponse {
            doc_id: "doc123".to_string(),
            model_fingerprint: "model456".to_string(),
            top_segments: vec![SegmentScore {
                segment_id: "seg123".to_string(),
                score_representative: 0.95,
                score_diversity: 0.85,
                reason: "Highly central".to_string(),
            }],
            metrics: AnalysisMetrics {
                num_segments: 10,
                top_n: 5,
                mmr_lambda: 0.65,
                avg_pairwise_cosine: 0.3,
            },
        };

        let bytes = rmp_serde::to_vec(&response).unwrap();
        let response2: AnalyzeResponse = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(response, response2);
    }

    #[test]
    fn test_summarize_response_serde_json() {
        let response = SummarizeResponse {
            summary_text: "This is a summary".to_string(),
            bullets: Some(vec!["Bullet 1".to_string(), "Bullet 2".to_string()]),
            citations: Some(vec![Citation {
                segment_id: "seg123".to_string(),
                start: 0,
                end: 4,
            }]),
            guardrails: Some(GuardrailsInfo {
                filtered: true,
                reason: Some("Contains sensitive content".to_string()),
            }),
            metrics: SummarizationMetrics {
                processing_time_ms: 100,
                input_tokens: 1000,
                output_tokens: 100,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        let response2: SummarizeResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(response, response2);
    }

    #[test]
    fn test_summarize_response_serde_msgpack() {
        let response = SummarizeResponse {
            summary_text: "This is a summary".to_string(),
            bullets: Some(vec!["Bullet 1".to_string(), "Bullet 2".to_string()]),
            citations: Some(vec![Citation {
                segment_id: "seg123".to_string(),
                start: 0,
                end: 4,
            }]),
            guardrails: Some(GuardrailsInfo {
                filtered: true,
                reason: Some("Contains sensitive content".to_string()),
            }),
            metrics: SummarizationMetrics {
                processing_time_ms: 100,
                input_tokens: 1000,
                output_tokens: 100,
            },
        };

        let bytes = rmp_serde::to_vec(&response).unwrap();
        let response2: SummarizeResponse = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(response, response2);
    }

    #[test]
    fn test_compute_doc_id() {
        let text = "This is a document";
        let chunk_params = "chunk_size=100";
        let model_fingerprint = "model789";
        let id1 = compute_doc_id(text, chunk_params, model_fingerprint);
        let id2 = compute_doc_id(text, chunk_params, model_fingerprint);
        assert_eq!(id1, id2);

        // Test that a small change in input produces a different ID
        let text2 = "This is a different document";
        let id3 = compute_doc_id(text2, chunk_params, model_fingerprint);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_compute_segment_id() {
        let text = "This is a segment";
        let id1 = compute_segment_id(text);
        let id2 = compute_segment_id(text);
        assert_eq!(id1, id2);

        // Test that a small change in input produces a different ID
        let text2 = "This is a different segment";
        let id3 = compute_segment_id(text2);
        assert_ne!(id1, id3);
    }
}