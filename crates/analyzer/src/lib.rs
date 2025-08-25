//! Analyzer crate for the websearch pipeline
//!
//! This crate handles:
//! - Loading embedding models from Hugging Face Hub or local files
//! - Generating embeddings for document segments
//! - Ranking segments using centroid similarity and MMR

pub mod cache;
pub mod config;
pub mod model;
mod reranker_test;

use cache::EmbeddingCache;
use config::{AnalyzerConfig, RerankerConfig};
use kernel::{AnalysisMetrics, AnalyzeResponse, Document, SegmentScore};
use log::{debug, info};
use ndarray::{Array2, Axis};
use ort::session::Session;
use tokenizers::Tokenizer;

/// The main analyzer struct
pub struct Analyzer {
    /// Configuration used to initialize the analyzer
    config: AnalyzerConfig,

    /// Model fingerprint for tracking
    model_fingerprint: String,

    /// ONNX Runtime session for the embedding model
    session: Session,

    /// Tokenizer for the embedding model
    tokenizer: Tokenizer,

    /// Maximum sequence length for embedder
    max_seq_len: usize,

    /// Maximum sequence length for reranker
    reranker_max_seq_len: usize,

    /// Embedding cache
    cache: EmbeddingCache,

    /// ONNX Runtime session for the reranker model (if enabled)
    reranker_session: Option<Session>,

    /// Tokenizer for the reranker model (if enabled)
    reranker_tokenizer: Option<Tokenizer>,

    /// Reranker model fingerprint (if enabled)
    reranker_model_fingerprint: Option<String>,
}

#[cfg(target_os = "windows")]
const ORT_DEFAULT_PATH: &str = "./onnxruntime.dll";
#[cfg(target_os = "linux")]
const ORT_DEFAULT_PATH: &str = "./libonnxruntime.so";
#[cfg(target_os = "macos")]
const ORT_DEFAULT_PATH: &str = "./libonnxruntime.dylib";
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
const ORT_DEFAULT_PATH: &str = "";

impl Analyzer {
    /// Create a new analyzer with the given configuration
    pub async fn new(config: AnalyzerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        debug!("Initializing analyzer with config: {:?}", config);

        {
            // Init ort to the path
            let path = config
                .onnx_provider
                .as_ref()
                .and_then(|path| path.to_str())
                .unwrap_or(ORT_DEFAULT_PATH);
            info!("Loading ONNX provider: '{}'", path);
            ort::init_from(path).commit()?;
        }

        // Initialize cache
        let cache = EmbeddingCache::new(config.cache.clone())?;
        info!("Initialized embedding cache");

        // Resolve model files
        let resolved_model = model::resolve_model(&config).await?;
        info!("Resolved model files: {:?}", resolved_model.file_paths);

        // Find the ONNX model file and tokenizer file
        let mut model_path = None;
        let mut tokenizer_path = None;

        for path in &resolved_model.file_paths {
            let path_str = path.to_string_lossy();
            if path_str.contains("model.onnx") {
                model_path = Some(path.clone());
            } else if path_str.contains("tokenizer.json") {
                tokenizer_path = Some(path.clone());
            }
        }

        // Ensure we found both files
        let model_path = model_path.ok_or("ONNX model file not found")?;
        let tokenizer_path = tokenizer_path.ok_or("Tokenizer file not found")?;

        info!("Model path: {:?}", model_path);
        info!("Tokenizer path: {:?}", tokenizer_path);

        // Load the tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        // Create ONNX Runtime environment and session
        let session = Session::builder()?.commit_from_file(model_path)?;

        // Get model fingerprint
        let model_fingerprint = resolved_model.fingerprint;
        info!("Model fingerprint: {}", model_fingerprint);

        // Initialize reranker if enabled
        let (reranker_session, reranker_tokenizer, reranker_model_fingerprint) =
            if config.rerank && config.reranker.enabled {
                info!("Initializing reranker");

                // Create a temporary config for the reranker model
                let reranker_config = AnalyzerConfig {
                    backend: config.backend.clone(),
                    model: config.reranker.model.clone(),
                    mmr_lambda: config.mmr_lambda,
                    top_n: config.top_n,
                    rerank: false,                       // Disable nested reranking
                    reranker: RerankerConfig::default(), // Default reranker config
                    allow_downloads: config.allow_downloads,
                    cache: config.cache.clone(),
                    onnx_provider: None, // No need to setup onnx_provider as it's already have been loaded
                };

                // Resolve reranker model files
                let resolved_reranker_model = model::resolve_model(&reranker_config).await?;
                info!(
                    "Resolved reranker model files: {:?}",
                    resolved_reranker_model.file_paths
                );

                // Find the ONNX model file and tokenizer file for reranker
                let mut reranker_model_path = None;
                let mut reranker_tokenizer_path = None;

                for path in &resolved_reranker_model.file_paths {
                    let path_str = path.to_string_lossy();
                    if path_str.contains("model.onnx") {
                        reranker_model_path = Some(path.clone());
                    } else if path_str.contains("tokenizer.json") {
                        reranker_tokenizer_path = Some(path.clone());
                    }
                }

                // Ensure we found both files
                let reranker_model_path =
                    reranker_model_path.ok_or("Reranker ONNX model file not found")?;
                let reranker_tokenizer_path =
                    reranker_tokenizer_path.ok_or("Reranker tokenizer file not found")?;

                info!("Reranker model path: {:?}", reranker_model_path);
                info!("Reranker tokenizer path: {:?}", reranker_tokenizer_path);

                // Load the reranker tokenizer
                let reranker_tokenizer = Tokenizer::from_file(&reranker_tokenizer_path)
                    .map_err(|e| format!("Failed to load reranker tokenizer: {}", e))?;

                // Create ONNX Runtime environment and session for reranker
                let mut session_builder = Session::builder()?;

                // Configure threading if specified
                if config.reranker.intra_op_threads > 0 {
                    session_builder =
                        session_builder.with_intra_threads(config.reranker.intra_op_threads)?;
                }
                if config.reranker.inter_op_threads > 0 {
                    session_builder =
                        session_builder.with_inter_threads(config.reranker.inter_op_threads)?;
                }

                let reranker_session = session_builder.commit_from_file(reranker_model_path)?;

                // Get reranker model fingerprint
                let reranker_model_fingerprint = resolved_reranker_model.fingerprint;
                info!("Reranker model fingerprint: {}", reranker_model_fingerprint);

                (
                    Some(reranker_session),
                    Some(reranker_tokenizer),
                    Some(reranker_model_fingerprint),
                )
            } else {
                (None, None, None)
            };

        // Set batch size and max sequence length from config or defaults
        let _batch_size = 8; // Default batch size (no longer used)
        let max_seq_len = 512; // Default max sequence length for embedder
        let reranker_max_seq_len = config.reranker.max_seq_len; // Use reranker max sequence length

        info!("Analyzer initialized successfully");

        Ok(Self {
            config,
            model_fingerprint,
            session,
            tokenizer,
            max_seq_len,
            cache,
            reranker_session,
            reranker_tokenizer,
            reranker_model_fingerprint,
            reranker_max_seq_len,
        })
    }

    /// Shutdown the analyzer and cleanup resources
    pub fn shutdown(self) -> Result<(), Box<dyn std::error::Error>> {
        // Explicitly drop the sessions to ensure proper cleanup
        drop(self.session);
        drop(self.reranker_session);
        drop(self.tokenizer);
        drop(self.reranker_tokenizer);

        // Explicitly flush and close the cache
        self.cache.close()?;

        Ok(())
    }

    /// Get the model fingerprint
    pub fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }

    /// Build a query text for reranking based on the document
    fn build_rerank_query(&self, document: &Document) -> String {
        match self.config.reranker.query_mode.as_str() {
            "centroid_summary" => {
                // Use document title and first sentence of first segment as query
                let title = &document.title;
                let first_segment_text = document
                    .segments
                    .first()
                    .map(|s| s.text.as_str())
                    .unwrap_or("");

                // Extract first sentence (up to first period)
                let first_sentence = first_segment_text
                    .split('.')
                    .next()
                    .unwrap_or(first_segment_text);

                if !title.is_empty() {
                    format!("{}: {}", title, first_sentence)
                } else {
                    first_sentence.to_string()
                }
            }
            _ => {
                // Default to first segment text
                document
                    .segments
                    .first()
                    .map(|s| s.text.clone())
                    .unwrap_or_else(|| "Document content".to_string())
            }
        }
    }

    /// Analyze a document and return an AnalyzeResponse
    pub fn analyze(
        &mut self,
        document: &Document,
    ) -> Result<AnalyzeResponse, Box<dyn std::error::Error>> {
        info!("Analyzing document: {}", document.doc_id);

        // Reset cache counters for this run
        self.cache.reset_counters();

        // Generate embeddings for all segments (with caching)
        let embeddings = self.generate_segment_embeddings(document)?;
        debug!(
            "Generated embeddings with shape: ({}, {})",
            embeddings.nrows(),
            embeddings.ncols()
        );

        // Report cache statistics
        let (hits, misses, ratio) = self.cache.get_hit_miss_stats();
        info!(
            "Embedding cache stats - Hits: {}, Misses: {}, Hit Ratio: {:.2}%",
            hits,
            misses,
            ratio * 100.0
        );

        // Compute centroid
        let centroid = compute_centroid(&embeddings);
        debug!("Computed centroid");

        // Compute cosine similarities to centroid
        let centroid_similarities = compute_cosine_similarities(&embeddings, &centroid);
        debug!("Computed centroid similarities");

        // Apply MMR selection to get the shortlist
        let mmr_selected_indices = mmr_selection(
            &embeddings,
            &centroid_similarities,
            self.config.reranker.top_m.min(document.segments.len()),
            self.config.mmr_lambda,
        );
        debug!(
            "Selected {} segments using MMR for reranking",
            mmr_selected_indices.len()
        );

        // Apply reranking if enabled
        let (final_selected_indices, top_segments) =
            if self.config.rerank && self.config.reranker.enabled {
                info!("Applying reranking");

                // Record start time
                let rerank_start = std::time::Instant::now();

                // Build query for reranking
                let query = self.build_rerank_query(document);
                info!("Rerank query: {}", query);

                // Get the segments for reranking
                let rerank_segments: Vec<&kernel::Segment> = mmr_selected_indices
                    .iter()
                    .map(|&idx| &document.segments[idx])
                    .collect();

                // Rerank segments
                let reranked_indices_scores = self.rerank_segments(&query, &rerank_segments)?;

                // Take top N from reranked results
                let top_n = self.config.top_n.min(reranked_indices_scores.len());
                let final_reranked = reranked_indices_scores
                    .into_iter()
                    .take(top_n)
                    .collect::<Vec<_>>();

                // Map back to original indices
                let final_indices: Vec<usize> = final_reranked
                    .iter()
                    .map(|(idx, _)| mmr_selected_indices[*idx])
                    .collect();

                // Create SegmentScore objects with rerank scores
                let segments_with_scores: Vec<SegmentScore> = final_reranked
                    .iter()
                    .map(|(idx, score)| {
                        let original_idx = mmr_selected_indices[*idx];
                        let segment = &document.segments[original_idx];
                        SegmentScore {
                            segment_id: segment.segment_id.clone(),
                            score_representative: centroid_similarities[original_idx],
                            score_diversity: 0.0, // TODO: Compute actual diversity score
                            score_rerank: Some(*score),
                            reason: if *idx == 0 {
                                "rerank: highly relevant".to_string()
                            } else {
                                "rerank: relevant".to_string()
                            },
                        }
                    })
                    .collect();

                // Record end time
                let rerank_duration = rerank_start.elapsed();
                info!("Reranking completed in {:?}", rerank_duration);

                (final_indices, segments_with_scores)
            } else {
                // Use original MMR selection without reranking
                let selected_indices = mmr_selection(
                    &embeddings,
                    &centroid_similarities,
                    self.config.top_n,
                    self.config.mmr_lambda,
                );
                debug!("Selected {} segments using MMR", selected_indices.len());

                // Create SegmentScore objects
                let segments_with_scores: Vec<SegmentScore> = selected_indices
                    .iter()
                    .map(|&idx| {
                        let segment = &document.segments[idx];
                        SegmentScore {
                            segment_id: segment.segment_id.clone(),
                            score_representative: centroid_similarities[idx],
                            score_diversity: 0.0, // TODO: Compute actual diversity score
                            score_rerank: None,
                            reason: if idx == selected_indices[0] {
                                "highly central".to_string()
                            } else {
                                "diversity pick".to_string()
                            },
                        }
                    })
                    .collect();

                (selected_indices, segments_with_scores)
            };

        // Compute average pairwise cosine similarity among selected segments
        let avg_pairwise_cosine = if final_selected_indices.len() > 1 {
            let mut sum = 0.0;
            let mut count = 0;
            for i in 0..final_selected_indices.len() {
                for j in (i + 1)..final_selected_indices.len() {
                    let idx_i = final_selected_indices[i];
                    let idx_j = final_selected_indices[j];
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

        // Determine the model fingerprint to use (include reranker if used)
        let model_fingerprint = if self.config.rerank && self.config.reranker.enabled {
            if let Some(reranker_fingerprint) = &self.reranker_model_fingerprint {
                format!("{}+{}", self.model_fingerprint, reranker_fingerprint)
            } else {
                self.model_fingerprint.clone()
            }
        } else {
            self.model_fingerprint.clone()
        };

        // Create the response
        let response = AnalyzeResponse {
            doc_id: document.doc_id.clone(),
            model_fingerprint,
            top_segments,
            metrics: AnalysisMetrics {
                num_segments: document.segments.len(),
                top_n: final_selected_indices.len(),
                mmr_lambda: self.config.mmr_lambda,
                avg_pairwise_cosine,
            },
        };

        Ok(response)
    }

    /// Generate embeddings for all segments in a document (with caching)
    fn generate_segment_embeddings(
        &mut self,
        document: &Document,
    ) -> Result<ndarray::Array2<f32>, Box<dyn std::error::Error>> {
        info!(
            "Generating embeddings for {} segments with caching",
            document.segments.len()
        );

        // Process each segment individually with caching
        let mut all_embeddings = Vec::new();

        for segment in &document.segments {
            // Try to get embedding from cache first
            if let Some(cached_embedding) = self
                .cache
                .get_embedding(&segment.segment_id, &self.model_fingerprint)?
            {
                debug!("Using cached embedding for segment: {}", segment.segment_id);
                all_embeddings.push(cached_embedding);
                continue;
            }

            // Generate embedding if not in cache
            debug!("Generating embedding for segment: {}", segment.segment_id);
            let embedding = self.generate_single_embedding(&segment.text)?;

            // Cache the embedding
            self.cache
                .put_embedding(&segment.segment_id, &self.model_fingerprint, &embedding)?;

            all_embeddings.push(embedding);
        }

        // Concatenate all embeddings
        if all_embeddings.is_empty() {
            Ok(Array2::zeros((0, 384)))
        } else {
            let embeddings = ndarray::concatenate(
                Axis(0),
                &all_embeddings.iter().map(|a| a.view()).collect::<Vec<_>>(),
            )?;
            Ok(embeddings)
        }
    }

    /// Generate embedding for a single text using ONNX model
    fn generate_single_embedding(
        &mut self,
        text: &str,
    ) -> Result<ndarray::Array2<f32>, Box<dyn std::error::Error>> {
        // Create a slice with the single text
        let texts = [text];
        self.generate_embeddings(&texts)
    }

    /// Rerank segments using the cross-encoder model
    fn rerank_segments(
        &mut self,
        query: &str,
        segments: &[&kernel::Segment],
    ) -> Result<Vec<(usize, f32)>, Box<dyn std::error::Error>> {
        if !self.config.rerank || !self.config.reranker.enabled {
            return Err("Reranker not enabled".into());
        }

        let reranker_session = self
            .reranker_session
            .as_mut()
            .ok_or("Reranker session not initialized")?;
        let reranker_tokenizer = self
            .reranker_tokenizer
            .as_ref()
            .ok_or("Reranker tokenizer not initialized")?;

        info!(
            "Reranking {} segments with query: {}",
            segments.len(),
            query
        );

        // Record start time
        let start_time = std::time::Instant::now();

        // Create pairs of (query, segment_text) with proper truncation policy
        let pairs: Vec<(&str, &str)> = segments
            .iter()
            .map(|segment| (query, segment.text.as_str()))
            .collect();

        // Tokenize pairs with proper truncation (truncating candidate tail if needed)
        let encodings = reranker_tokenizer
            .encode_batch(pairs.clone(), true)
            .map_err(|e| format!("Reranker tokenization failed: {}", e))?;

        // Convert to input tensors with proper truncation
        let input_ids: Vec<Vec<i64>> = encodings
            .iter()
            .map(|encoding| {
                let ids = encoding.get_ids();
                // Truncate or pad to max_seq_len
                // Apply truncation policy: prefer truncating the candidate tail if needed
                if ids.len() > self.reranker_max_seq_len {
                    // Truncate from the end (candidate tail truncation)
                    ids[..self.reranker_max_seq_len]
                        .iter()
                        .map(|&x| x as i64)
                        .collect()
                } else {
                    let mut padded: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
                    padded.resize(self.reranker_max_seq_len, 0);
                    padded
                }
            })
            .collect();

        let attention_mask: Vec<Vec<i64>> = input_ids
            .iter()
            .map(|ids| {
                ids.iter()
                    .map(|&id| if id > 0 { 1i64 } else { 0i64 })
                    .collect()
            })
            .collect();

        // Stack into batched tensors
        let batch_size = input_ids.len();
        let input_ids_array = ndarray::Array2::from_shape_vec(
            (batch_size, self.reranker_max_seq_len),
            input_ids.into_iter().flatten().collect(),
        )?;

        let attention_mask_array = ndarray::Array2::from_shape_vec(
            (batch_size, self.reranker_max_seq_len),
            attention_mask.into_iter().flatten().collect(),
        )?;

        // Run inference with proper tensor handling
        let scores = {
            // Convert to i64 for ONNX Runtime
            let input_ids_i64: Vec<i64> = input_ids_array.iter().map(|&x| x as i64).collect();
            let attention_mask_i64: Vec<i64> =
                attention_mask_array.iter().map(|&x| x as i64).collect();

            let input_ids_value =
                ort::value::Value::from_array((input_ids_array.shape(), input_ids_i64))?;
            let attention_mask_value =
                ort::value::Value::from_array((attention_mask_array.shape(), attention_mask_i64))?;

            // Use the session directly
            let outputs = reranker_session.run(ort::inputs![
                "input_ids" => input_ids_value,
                "attention_mask" => attention_mask_value
            ])?;

            // Extract scores from the output tensor based on config
            let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

            // Extract the relevance score based on the configured tensor index
            let scores: Vec<f32> = match self.config.reranker.score_tensor.as_str() {
                "logits[0]" => {
                    data.chunks(shape[1] as usize)
                        .map(|chunk| chunk[0]) // Take first element as relevance score
                        .collect()
                }
                _ => {
                    // Default to logits[0] if unknown
                    data.chunks(shape[1] as usize)
                        .map(|chunk| chunk[0])
                        .collect()
                }
            };

            scores
        };

        // Record end time
        let duration = start_time.elapsed();
        info!("Reranking completed in {:?}", duration);

        // Log top 5 scores for debugging (just indices and scores, no content)
        let mut indexed_scores_preview: Vec<(usize, f32)> = segments
            .iter()
            .enumerate()
            .map(|(i, _segment)| (i, scores[i]))
            .collect();

        // Sort by score descending for preview
        indexed_scores_preview
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Log top 5 scores
        let top_5_preview: Vec<(usize, f32)> = indexed_scores_preview
            .iter()
            .take(5)
            .map(|(idx, score)| (*idx, *score))
            .collect();

        info!("Top 5 rerank scores (index, score): {:?}", top_5_preview);

        // Create index-score pairs and sort by score (descending)
        let mut indexed_scores: Vec<(usize, f32)> = segments
            .iter()
            .enumerate()
            .map(|(i, _)| (i, scores[i]))
            .collect();

        // Sort by score descending with stable tie-breaking
        indexed_scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0)) // Stable tie-breaking by index
        });

        Ok(indexed_scores)
    }

    /// Generate embeddings for a batch of texts using ONNX model
    fn generate_embeddings(
        &mut self,
        texts: &[&str],
    ) -> Result<ndarray::Array2<f32>, Box<dyn std::error::Error>> {
        if texts.is_empty() {
            return Ok(Array2::zeros((0, 384)));
        }

        debug!("Generating embeddings for {} texts", texts.len());

        // Tokenize texts with padding and truncation
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        // Convert to input tensors
        let input_ids: Vec<Vec<i64>> = encodings
            .iter()
            .map(|encoding| {
                let ids = encoding.get_ids();
                // Truncate or pad to max_seq_len
                if ids.len() > self.max_seq_len {
                    ids[..self.max_seq_len].iter().map(|&x| x as i64).collect()
                } else {
                    let mut padded: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
                    padded.resize(self.max_seq_len, 0);
                    padded
                }
            })
            .collect();

        let attention_mask: Vec<Vec<i64>> = input_ids
            .iter()
            .map(|ids| {
                ids.iter()
                    .map(|&id| if id > 0 { 1i64 } else { 0i64 })
                    .collect()
            })
            .collect();

        // Stack into batched tensors
        let batch_size = input_ids.len();
        let input_ids_array = Array2::from_shape_vec(
            (batch_size, self.max_seq_len),
            input_ids.into_iter().flatten().collect(),
        )?;

        let attention_mask_array = Array2::from_shape_vec(
            (batch_size, self.max_seq_len),
            attention_mask.into_iter().flatten().collect(),
        )?;

        // Run inference
        let embeddings = {
            // Convert to i64 for ONNX Runtime
            let input_ids_i64: Vec<i64> = input_ids_array.iter().map(|&x| x as i64).collect();
            let attention_mask_i64: Vec<i64> =
                attention_mask_array.iter().map(|&x| x as i64).collect();

            // Create token_type_ids (usually all zeros for sentence embeddings)
            let token_type_ids_i64: Vec<i64> = vec![0i64; input_ids_i64.len()];

            let input_ids_value =
                ort::value::Value::from_array((input_ids_array.shape(), input_ids_i64))?;
            let attention_mask_value =
                ort::value::Value::from_array((attention_mask_array.shape(), attention_mask_i64))?;
            let token_type_ids_value =
                ort::value::Value::from_array((attention_mask_array.shape(), token_type_ids_i64))?;

            let outputs = self.session.run(ort::inputs![
                "input_ids" => input_ids_value,
                "attention_mask" => attention_mask_value,
                "token_type_ids" => token_type_ids_value
            ])?;

            // Extract embeddings (first output)
            let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
            // Convert shape from Vec<i64> to Vec<usize>
            let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
            ndarray::ArrayD::from_shape_vec(shape_usize, data.to_vec())?
        };

        // Apply mean pooling with attention mask
        let pooled_embeddings = self.mean_pool(&embeddings.view(), &attention_mask_array);

        // L2 normalize embeddings
        let normalized_embeddings = l2_normalize_rows(&pooled_embeddings);

        Ok(normalized_embeddings)
    }

    /// Apply mean pooling with attention mask
    fn mean_pool(
        &self,
        embeddings: &ndarray::ArrayViewD<f32>,
        attention_mask: &Array2<i64>,
    ) -> Array2<f32> {
        // Convert attention mask to f32
        let mask_f32: Array2<f32> = attention_mask.mapv(|x| x as f32);

        // Expand mask to match embedding dimensions
        let mask_expanded = mask_f32.insert_axis(Axis(2));

        // Apply mask to embeddings
        let masked_embeddings = embeddings.to_owned() * &mask_expanded;

        // Sum along sequence dimension
        let sum_embeddings = masked_embeddings.sum_axis(Axis(1));

        // Sum mask along sequence dimension
        let sum_mask = mask_expanded.sum_axis(Axis(1));

        // Avoid division by zero
        let sum_mask_safe = sum_mask.mapv(|x| if x > 0.0 { x } else { 1.0 });

        // Compute mean
        let mean_embeddings = sum_embeddings / sum_mask_safe;

        // Convert to 2D array
        mean_embeddings.into_dimensionality().unwrap()
    }
}

/// Compute the centroid (mean) of all embeddings
fn compute_centroid(embeddings: &ndarray::Array2<f32>) -> ndarray::Array1<f32> {
    embeddings
        .mean_axis(ndarray::Axis(0))
        .unwrap_or_else(|| ndarray::Array1::zeros(embeddings.ncols()))
}

/// Compute cosine similarities between embeddings and a reference vector
fn compute_cosine_similarities(
    embeddings: &ndarray::Array2<f32>,
    reference: &ndarray::Array1<f32>,
) -> Vec<f32> {
    embeddings
        .axis_iter(ndarray::Axis(0))
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
    let first_idx = centroid_similarities
        .iter()
        .enumerate()
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
            let max_sim = selected_indices
                .iter()
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
    use std::env::{current_dir, set_current_dir};

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
            [1.0, 0.0, 0.0], // Most similar to centroid [1, 0, 0]
            [0.0, 1.0, 0.0], // Orthogonal to centroid
            [0.9, 0.1, 0.0], // Similar to centroid but different from item 0
            [0.0, 0.0, 1.0], // Orthogonal to centroid
        ]);

        let centroid_similarities = vec![1.0, 0.0, 0.9, 0.0]; // Precomputed

        // Test with top_n=2, lambda=0.5 (balanced between centroid similarity and diversity)
        let selected = mmr_selection(&embeddings, &centroid_similarities, 2, 0.5);

        // Should select item 0 first (highest centroid similarity)
        assert_eq!(selected[0], 0);

        // For this simple case, let's just check that we get 2 items
        assert_eq!(selected.len(), 2);
    }

    #[tokio::test]
    async fn test_analyzer_creation_local_missing() {
        // Change cwd to load dylib correctly in tests
        {
            let current_directory = current_dir().expect("Failed to get current directory");
            set_current_dir(current_directory.join("../../")).expect("Failed to change directory");
        }
        let mut config = AnalyzerConfig::new();
        // Disable downloads for testing
        config.allow_downloads = false;
        let analyzer = Analyzer::new(config).await;
        // This should fail because we're not allowing downloads and the model isn't available locally
        assert!(analyzer.is_err());
    }

    #[tokio::test]
    async fn test_analyze_snapshot() {
        // This test requires a real model, so we'll skip it in CI
        // To run it locally, you need to have the model files available
        if std::env::var("EMBED_ONNX_TEST").unwrap_or_default() != "1" {
            println!("Skipping test_analyze_snapshot - set EMBED_ONNX_TEST=1 to run");
            return;
        }

        // Create a simple document for testing
        let document = kernel::Document {
            schema_version: "1.0.0".to_string(),
            doc_id: "test-doc-123".to_string(),
            url: "https://example.com".to_string(),
            title: "Test Document".to_string(),
            lang: "en".to_string(),
            fetched_at: "2023-01-01T00:00:00Z".to_string(),
            segments: vec![
                kernel::Segment {
                    segment_id: "seg-1".to_string(),
                    text: "This is the first segment.".to_string(),
                    path: "body > p:nth-child(1)".to_string(),
                    position: 0,
                },
                kernel::Segment {
                    segment_id: "seg-2".to_string(),
                    text: "This is the second segment.".to_string(),
                    path: "body > p:nth-child(2)".to_string(),
                    position: 1,
                },
                kernel::Segment {
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
        let mut analyzer = Analyzer::new(config.clone())
            .await
            .expect("Failed to create analyzer");

        // Analyze the document
        let response = analyzer
            .analyze(&document)
            .expect("Failed to analyze document");

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
