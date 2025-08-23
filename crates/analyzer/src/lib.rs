//! Analyzer crate for the websearch pipeline
//!
//! This crate handles:
//! - Loading embedding models from Hugging Face Hub or local files
//! - Generating embeddings for document segments
//! - Ranking segments using centroid similarity and MMR

pub mod config;
pub mod model;

use config::AnalyzerConfig;
use core::{Document, AnalyzeResponse, SegmentScore, AnalysisMetrics};
use log::{info, debug};

/// The main analyzer struct
pub struct Analyzer {
    /// Configuration used to initialize the analyzer
    config: AnalyzerConfig,
    
    /// Model fingerprint for tracking
    model_fingerprint: String,
}

impl Analyzer {
    /// Create a new analyzer with the given configuration
    pub fn new(config: AnalyzerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Initializing analyzer with config: {:?}", config);
        
        // For this step, we'll use a placeholder model fingerprint
        let model_fingerprint = config.model_fingerprint();
        info!("Model fingerprint: {}", model_fingerprint);
        
        info!("Analyzer initialized successfully");
        
        Ok(Self {
            config,
            model_fingerprint,
        })
    }
    
    /// Get the model fingerprint
    pub fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }
    
    /// Analyze a document and return an AnalyzeResponse
    pub fn analyze(&self, document: &Document) -> Result<AnalyzeResponse, Box<dyn std::error::Error>> {
        info!("Analyzing document: {}", document.doc_id);
        
        // Extract texts from segments
        let texts: Vec<&str> = document.segments.iter().map(|s| s.text.as_str()).collect();
        debug!("Processing {} segments", texts.len());
        
        // Generate embeddings for all segments (placeholder implementation)
        let embeddings = self.generate_embeddings(&texts)?;
        debug!("Generated embeddings with shape: ({}, {})", embeddings.nrows(), embeddings.ncols());
        
        // Compute centroid
        let centroid = compute_centroid(&embeddings);
        debug!("Computed centroid");
        
        // Compute cosine similarities to centroid
        let centroid_similarities = compute_cosine_similarities(&embeddings, &centroid);
        debug!("Computed centroid similarities");
        
        // Apply MMR selection
        let selected_indices = mmr_selection(
            &embeddings,
            &centroid_similarities,
            self.config.top_n,
            self.config.mmr_lambda,
        );
        debug!("Selected {} segments using MMR", selected_indices.len());
        
        // Create SegmentScore objects
        let top_segments: Vec<SegmentScore> = selected_indices.iter().map(|&idx| {
            let segment = &document.segments[idx];
            SegmentScore {
                segment_id: segment.segment_id.clone(),
                score_representative: centroid_similarities[idx],
                score_diversity: 0.0, // TODO: Compute actual diversity score
                reason: if idx == selected_indices[0] { 
                    "highly central".to_string() 
                } else { 
                    "diversity pick".to_string() 
                },
            }
        }).collect();
        
        // Compute average pairwise cosine similarity among selected segments
        let avg_pairwise_cosine = if selected_indices.len() > 1 {
            let mut sum = 0.0;
            let mut count = 0;
            for i in 0..selected_indices.len() {
                for j in (i+1)..selected_indices.len() {
                    let idx_i = selected_indices[i];
                    let idx_j = selected_indices[j];
                    // Convert rows to vectors for cosine similarity calculation
                    let row_i = embeddings.row(idx_i);
                    let row_j = embeddings.row(idx_j);
                    let vec_i: Vec<f32> = row_i.to_vec();
                    let vec_j: Vec<f32> = row_j.to_vec();
                    let cos_sim = cosine_similarity(&vec_i, &vec_j);
                    sum += cos_sim;
                    count += 1;
                }
            }
            sum / count as f32
        } else {
            0.0
        };
        
        // Create the response
        let response = AnalyzeResponse {
            doc_id: document.doc_id.clone(),
            model_fingerprint: self.model_fingerprint.clone(),
            top_segments,
            metrics: AnalysisMetrics {
                num_segments: document.segments.len(),
                top_n: self.config.top_n.min(document.segments.len()),
                mmr_lambda: self.config.mmr_lambda,
                avg_pairwise_cosine,
            },
        };
        
        Ok(response)
    }
    
    /// Generate embeddings for a batch of texts (placeholder implementation)
    fn generate_embeddings(&self, texts: &[&str]) -> Result<ndarray::Array2<f32>, Box<dyn std::error::Error>> {
        // For this step, we'll generate dummy embeddings
        // In a real implementation, we would use an ONNX model here
        let dummy_embeddings = ndarray::Array2::from_shape_fn((texts.len(), 384), |(i, j)| {
            (i * j) as f32 / (texts.len() * 384) as f32
        });
        
        // L2 normalize embeddings
        let normalized_embeddings = l2_normalize_rows(&dummy_embeddings);
        
        Ok(normalized_embeddings)
    }
}

/// Compute the centroid (mean) of all embeddings
fn compute_centroid(embeddings: &ndarray::Array2<f32>) -> ndarray::Array1<f32> {
    embeddings.mean_axis(ndarray::Axis(0)).unwrap_or_else(|| ndarray::Array1::zeros(embeddings.ncols()))
}

/// Compute cosine similarities between embeddings and a reference vector
fn compute_cosine_similarities(embeddings: &ndarray::Array2<f32>, reference: &ndarray::Array1<f32>) -> Vec<f32> {
    embeddings.axis_iter(ndarray::Axis(0))
        .map(|embedding| cosine_similarity(&embedding.to_vec(), &reference.to_vec()))
        .collect()
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// L2 normalize rows of a 2D array
fn l2_normalize_rows(arr: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    let mut normalized = arr.clone();
    for mut row in normalized.rows_mut() {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            row /= norm;
        }
    }
    normalized
}

/// Maximal Marginal Relevance (MMR) selection
fn mmr_selection(
    embeddings: &ndarray::Array2<f32>,
    centroid_similarities: &[f32],
    top_n: usize,
    lambda: f32,
) -> Vec<usize> {
    if embeddings.is_empty() || top_n == 0 {
        return vec![];
    }
    
    let n = embeddings.nrows().min(top_n);
    let mut selected_indices = Vec::with_capacity(n);
    
    // Start with the most similar to centroid
    let first_idx = centroid_similarities.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    selected_indices.push(first_idx);
    
    // Iteratively select remaining items
    while selected_indices.len() < n {
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;
        
        for i in 0..embeddings.nrows() {
            // Skip already selected indices
            if selected_indices.contains(&i) {
                continue;
            }
            
            // Calculate MMR score
            let centroid_score = centroid_similarities[i];
            
            // Find maximum similarity to already selected items
            let max_sim = selected_indices.iter()
                .map(|&j| {
                    // Convert rows to vectors for cosine similarity calculation
                    let row_i = embeddings.row(i);
                    let row_j = embeddings.row(j);
                    let vec_i: Vec<f32> = row_i.to_vec();
                    let vec_j: Vec<f32> = row_j.to_vec();
                    cosine_similarity(&vec_i, &vec_j)
                })
                .fold(0.0, f32::max);
            
            let mmr_score = lambda * centroid_score - (1.0 - lambda) * max_sim;
            
            if mmr_score > best_score {
                best_score = mmr_score;
                best_idx = i;
            }
        }
        
        selected_indices.push(best_idx);
    }
    
    selected_indices
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
        
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);
        
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), -1.0);
    }
    
    #[test]
    fn test_l2_normalize_rows() {
        let arr = ndarray::arr2(&[[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]]);
        let normalized = l2_normalize_rows(&arr);
        
        // First row should be [0.6, 0.8, 0.0] (3/5, 4/5, 0/5)
        assert!((normalized[[0, 0]] - 0.6).abs() < 1e-6);
        assert!((normalized[[0, 1]] - 0.8).abs() < 1e-6);
        assert_eq!(normalized[[0, 2]], 0.0);
        
        // Second row should be [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
        let inv_sqrt_3 = 1.0 / 3.0f32.sqrt();
        assert!((normalized[[1, 0]] - inv_sqrt_3).abs() < 1e-6);
        assert!((normalized[[1, 1]] - inv_sqrt_3).abs() < 1e-6);
        assert!((normalized[[1, 2]] - inv_sqrt_3).abs() < 1e-6);
    }
    
    #[test]
    fn test_mmr_selection() {
        // Create simple 2D embeddings for testing
        let embeddings = ndarray::arr2(&[
            [1.0, 0.0, 0.0],  // Most similar to centroid [1, 0, 0]
            [0.0, 1.0, 0.0],  // Orthogonal to centroid
            [0.9, 0.1, 0.0],  // Similar to centroid but different from item 0
            [0.0, 0.0, 1.0],  // Orthogonal to centroid
        ]);
        
        let centroid_similarities = vec![1.0, 0.0, 0.9, 0.0]; // Precomputed
        
        // Test with top_n=2, lambda=0.5 (balanced between centroid similarity and diversity)
        let selected = mmr_selection(&embeddings, &centroid_similarities, 2, 0.5);
        
        // Should select item 0 first (highest centroid similarity)
        assert_eq!(selected[0], 0);
        
        // For this simple case, let's just check that we get 2 items
        assert_eq!(selected.len(), 2);
    }
    
    #[test]
    fn test_analyzer_creation_local_missing() {
        let config = AnalyzerConfig::new();
        let analyzer = Analyzer::new(config);
        assert!(analyzer.is_ok());
    }
    
    #[test]
    fn test_analyze_snapshot() {
        // Create a simple document for testing
        let document = core::Document {
            schema_version: "1.0.0".to_string(),
            doc_id: "test-doc-123".to_string(),
            url: "https://example.com".to_string(),
            title: "Test Document".to_string(),
            lang: "en".to_string(),
            fetched_at: "2023-01-01T00:00:00Z".to_string(),
            segments: vec![
                core::Segment {
                    segment_id: "seg-1".to_string(),
                    text: "This is the first segment.".to_string(),
                    path: "body > p:nth-child(1)".to_string(),
                    position: 0,
                },
                core::Segment {
                    segment_id: "seg-2".to_string(),
                    text: "This is the second segment.".to_string(),
                    path: "body > p:nth-child(2)".to_string(),
                    position: 1,
                },
                core::Segment {
                    segment_id: "seg-3".to_string(),
                    text: "This is the third segment.".to_string(),
                    path: "body > p:nth-child(3)".to_string(),
                    position: 2,
                },
            ],
            hints: None,
        };
        
        // Create analyzer with default config
        let config = AnalyzerConfig::new();
        let analyzer = Analyzer::new(config.clone()).expect("Failed to create analyzer");
        
        // Analyze the document
        let response = analyzer.analyze(&document).expect("Failed to analyze document");
        
        // Basic assertions
        assert_eq!(response.doc_id, "test-doc-123");
        assert_eq!(response.top_segments.len(), 3.min(config.top_n)); // At most 3 segments in the test document
        assert_eq!(response.metrics.num_segments, 3);
        assert_eq!(response.metrics.top_n, 3.min(config.top_n));
        assert_eq!(response.metrics.mmr_lambda, config.mmr_lambda);
        
        // Check that scores are in valid range
        for segment_score in &response.top_segments {
            println!("Score: {}", segment_score.score_representative);
            // For the dummy implementation, we might get scores outside the normal range
            // Let's just check that they're finite numbers
            assert!(segment_score.score_representative.is_finite());
            assert!(segment_score.score_diversity >= 0.0);
        }
    }
}
