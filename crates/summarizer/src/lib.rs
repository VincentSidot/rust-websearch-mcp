//! Summarizer crate for the websearch pipeline
//!
//! This crate handles:
//! - Loading a Document and AnalyzeResponse
//! - Building a guarded summarization request using selected segments
//! - Calling an OpenAI-compatible endpoint to produce a concise summary
//! - Returning a SummarizeResponse JSON
//! - Providing an extractive fallback on timeout/error

pub mod config;

use config::{SummarizerConfig, SummaryStyle};
use kernel::{
    AnalyzeResponse, Citation, Document, GuardrailsInfo, Segment, SummarizationMetrics,
    SummarizeResponse,
};
use log::{debug, info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tokio::time::{timeout, Duration};

/// The main summarizer struct
pub struct Summarizer {
    /// Configuration used to initialize the summarizer
    config: SummarizerConfig,
    /// HTTP client for API requests
    client: Client,
}

/// Request structure for the OpenAI-compatible API
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    max_tokens: Option<u32>,
}

/// Message structure for the OpenAI-compatible API
#[derive(Debug, Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

/// Response structure from the OpenAI-compatible API
#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

/// Choice structure from the OpenAI-compatible API
#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

/// Micro-summary structure for map-reduce
#[derive(Debug, Clone)]
struct MicroSummary {
    text: String,
    used_segment_ids: Vec<String>,
}

impl Summarizer {
    /// Create a new summarizer with the given configuration
    pub fn new(config: SummarizerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        info!("Initializing summarizer with config: {:?}", config);

        // Create HTTP client with timeout
        let client = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()?;

        Ok(Self { config, client })
    }

    /// Summarize a document using an AnalyzeResponse
    pub async fn summarize(
        &self,
        document: &Document,
        analysis: &AnalyzeResponse,
    ) -> Result<SummarizeResponse, Box<dyn std::error::Error>> {
        info!("Summarizing document: {}", document.doc_id);

        // Record start time
        let start_time = Instant::now();

        // Resolve selected segment texts by matching IDs
        let selected_segments = self.resolve_selected_segments(document, analysis)?;
        debug!("Resolved {} selected segments", selected_segments.len());

        // Estimate total tokens
        let total_tokens = self.estimate_total_tokens(&selected_segments);
        info!(
            "Estimated total tokens: {}, max context tokens: {}",
            total_tokens, self.config.map_reduce.max_context_tokens
        );

        // Decide whether to use map-reduce or single-pass
        let use_map_reduce = self.config.map_reduce.enabled
            && total_tokens > self.config.map_reduce.max_context_tokens;

        // Log decision
        if use_map_reduce {
            info!(
                "Using map-reduce path ({} tokens > {} threshold)",
                total_tokens, self.config.map_reduce.max_context_tokens
            );
        } else {
            info!(
                "Using single-pass path ({} tokens <= {} threshold)",
                total_tokens, self.config.map_reduce.max_context_tokens
            );
        }

        // Generate summary based on decision
        let result = if use_map_reduce {
            self.summarize_map_reduce(document, &selected_segments, start_time)
                .await
        } else {
            self.summarize_single_pass(document, &selected_segments, start_time)
                .await
        };

        result
    }

    /// Resolve selected segment texts by matching IDs from the analysis to segments in the document
    fn resolve_selected_segments<'a>(
        &self,
        document: &'a Document,
        analysis: &AnalyzeResponse,
    ) -> Result<Vec<&'a Segment>, Box<dyn std::error::Error>> {
        // Create a map of segment_id to segment for quick lookup
        let segment_map: HashMap<&str, &Segment> = document
            .segments
            .iter()
            .map(|segment| (segment.segment_id.as_str(), segment))
            .collect();

        // Resolve selected segments in order
        let mut selected_segments = Vec::new();
        for segment_score in &analysis.top_segments {
            if let Some(segment) = segment_map.get(segment_score.segment_id.as_str()) {
                selected_segments.push(*segment);
            } else {
                return Err(format!(
                    "Segment ID {} not found in document",
                    segment_score.segment_id
                )
                .into());
            }
        }

        Ok(selected_segments)
    }

    /// Estimate total tokens for selected segments
    fn estimate_total_tokens(&self, selected_segments: &[&Segment]) -> usize {
        selected_segments
            .iter()
            .map(|segment| estimate_tokens(&segment.text))
            .sum()
    }

    /// Summarize using single-pass approach
    async fn summarize_single_pass(
        &self,
        document: &Document,
        selected_segments: &[&Segment],
        start_time: Instant,
    ) -> Result<SummarizeResponse, Box<dyn std::error::Error>> {
        // Build the prompt
        let prompt = self.build_prompt(document, selected_segments)?;
        debug!("Built prompt with {} characters", prompt.len());

        // Try to generate summary via API
        let api_result = self
            .generate_summary_via_api(&prompt, selected_segments)
            .await;

        // Handle API result or fallback
        let (summary_text, bullets, citations, guardrails, metrics) = match api_result {
            Ok((summary_text, bullets, citations)) => {
                let duration = start_time.elapsed();
                let metrics = SummarizationMetrics {
                    processing_time_ms: duration.as_millis() as u64,
                    input_tokens: estimate_tokens(&prompt),
                    output_tokens: estimate_tokens(&summary_text),
                };
                (summary_text, bullets, Some(citations), None, metrics)
            }
            Err(e) => {
                warn!("API call failed: {}. Using extractive fallback.", e);
                let (summary_text, metrics) = self.extractive_fallback(selected_segments, start_time);
                (
                    summary_text,
                    None, // No bullets in extractive fallback
                    None, // No citations in extractive fallback
                    Some(GuardrailsInfo {
                        filtered: true,
                        reason: Some("API error or timeout, using extractive fallback".to_string()),
                    }),
                    metrics,
                )
            }
        };

        // Create the response
        let response = SummarizeResponse {
            summary_text,
            bullets,
            citations,
            guardrails,
            metrics,
        };

        Ok(response)
    }

    /// Build a guarded summarization prompt
    fn build_prompt(
        &self,
        document: &Document,
        selected_segments: &[&Segment],
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut prompt = String::new();

        // Add document title and language if present
        if !document.title.is_empty() {
            prompt.push_str(&format!("Document Title: {}\n", document.title));
        }
        if !document.lang.is_empty() {
            prompt.push_str(&format!("Language: {}\n", document.lang));
        }

        prompt.push_str("\nSelected Passages:\n");

        // Add selected passages
        for (i, segment) in selected_segments.iter().enumerate() {
            prompt.push_str(&format!("{}. {}\n", i + 1, segment.text));
        }

        // Add instructions based on style
        match self.config.style {
            SummaryStyle::AbstractWithBullets => {
                prompt.push_str("\nInstructions:\n");
                prompt.push_str("Please provide a concise summary of the selected passages above in 150-250 words.\n");
                prompt
                    .push_str("Also include 3-6 bullet points highlighting the key information.\n");
                prompt.push_str("Do not include any information that is not present in the selected passages.\n");
                prompt.push_str("Cite the passage numbers inline or list the sources used.\n");
                prompt.push_str("If there is insufficient information to create a meaningful summary, please include a short \"Not in sources\" note.\n");
            }
            SummaryStyle::TlDr => {
                prompt.push_str("\nInstructions:\n");
                prompt.push_str("Please provide a TL;DR (Too Long; Didn't Read) summary of the selected passages above in 50-100 words.\n");
                prompt.push_str("Do not include any information that is not present in the selected passages.\n");
                prompt.push_str("If there is insufficient information to create a meaningful summary, please include a short \"Not in sources\" note.\n");
            }
            SummaryStyle::Extractive => {
                // For extractive style, we don't need complex instructions
                // This is handled by the extractive_fallback method
            }
        }

        Ok(prompt)
    }

    /// Generate summary via OpenAI-compatible API
    async fn generate_summary_via_api(
        &self,
        prompt: &str,
        selected_segments: &[&Segment],
    ) -> Result<(String, Option<Vec<String>>, Vec<Citation>), Box<dyn std::error::Error>> {
        // Create the request
        let request = ChatCompletionRequest {
            model: self.config.model.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        // Build the API URL
        let api_url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );

        // Prepare headers
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(api_key) = &self.config.api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", api_key).parse()?,
            );
        }
        headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse()?);

        // Make the API call with timeout
        let timeout_duration = Duration::from_millis(self.config.timeout_ms);
        let response_result = timeout(
            timeout_duration,
            self.client
                .post(&api_url)
                .headers(headers)
                .json(&request)
                .send(),
        )
        .await;

        // Handle timeout
        let response = match response_result {
            Ok(Ok(response)) => response,
            Ok(Err(e)) => return Err(format!("API request failed: {}", e).into()),
            Err(_) => return Err("API request timed out".into()),
        };

        // Check status
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(
                format!("API request failed with status {}: {}", status, error_text).into(),
            );
        }

        // Parse response
        let api_response: ChatCompletionResponse = response.json().await?;

        // Extract content
        if api_response.choices.is_empty() {
            return Err("API returned no choices".into());
        }

        let content = &api_response.choices[0].message.content;

        // For this MVP, we'll do a simple parse of the response to extract bullets
        // In a more sophisticated implementation, we might use a more structured approach
        let (summary_text, bullets) = if self.config.style == SummaryStyle::AbstractWithBullets {
            // Try to split into summary and bullets
            if let Some(pos) = content.find("\n-") {
                let summary = content[..pos].trim().to_string();
                let bullets_str = &content[pos + 1..];
                let bullet_points: Vec<String> = bullets_str
                    .lines()
                    .filter_map(|line| {
                        let clean_line = line.trim();
                        if clean_line.starts_with("- ") {
                            Some(clean_line[2..].to_string())
                        } else if clean_line.starts_with("* ") {
                            Some(clean_line[2..].to_string())
                        } else {
                            None
                        }
                    })
                    .collect();
                (summary, Some(bullet_points))
            } else {
                // If no bullets found, return entire content as summary
                (content.clone(), None)
            }
        } else {
            // For other styles, return entire content as summary
            (content.clone(), None)
        };

        // For citations, we'll just list all selected segment IDs
        // A more sophisticated implementation might try to map parts of the summary to specific segments
        let citations: Vec<Citation> = selected_segments
            .iter()
            .map(|segment| Citation {
                segment_id: segment.segment_id.clone(),
                start: 0, // Placeholder values
                end: 0,   // Placeholder values
            })
            .collect();

        Ok((summary_text, bullets, citations))
    }

    /// Generate extractive fallback summary
    fn extractive_fallback(
        &self,
        selected_segments: &[&Segment],
        start_time: Instant,
    ) -> (String, SummarizationMetrics) {
        // For extractive fallback, concatenate 1-2 strong sentences from the first few selected segments
        let mut summary = String::new();
        let target_length = 200; // Target length in characters

        for segment in selected_segments.iter().take(3) {
            // For simplicity, we'll just take the first sentence or a chunk of text
            // A more sophisticated implementation might use sentence tokenization
            let text = &segment.text;
            if summary.len() + text.len() > target_length {
                // Add a substring to reach target length
                let remaining_chars = target_length - summary.len();
                if remaining_chars > 20 {
                    // Only add if it's a meaningful addition
                    summary.push_str(&text[..remaining_chars]);
                    summary.push_str("...");
                }
                break;
            } else {
                summary.push_str(text);
                summary.push(' ');
            }
        }

        // Trim and ensure we don't exceed target length
        summary = summary.trim().to_string();
        if summary.len() > target_length {
            summary.truncate(target_length - 3);
            summary.push_str("...");
        }

        let duration = start_time.elapsed();
        let metrics = SummarizationMetrics {
            processing_time_ms: duration.as_millis() as u64,
            input_tokens: selected_segments
                .iter()
                .map(|s| estimate_tokens(&s.text))
                .sum(),
            output_tokens: estimate_tokens(&summary),
        };

        (summary, metrics)
    }

    /// Summarize using map-reduce approach
    async fn summarize_map_reduce(
        &self,
        document: &Document,
        selected_segments: &[&Segment],
        start_time: Instant,
    ) -> Result<SummarizeResponse, Box<dyn std::error::Error>> {
        // Partition segments into groups
        let groups = self.partition_segments(selected_segments)?;
        info!("Partitioned into {} groups for map stage", groups.len());

        // Map stage: generate micro-summaries
        let map_start = Instant::now();
        let micro_summaries = self.generate_micro_summaries(document, &groups).await?;
        let map_duration = map_start.elapsed();
        info!("Generated {} micro-summaries in {:?}", micro_summaries.len(), map_duration);

        // Reduce stage: merge micro-summaries
        let reduce_start = Instant::now();
        let (summary_text, bullets, citations) = self.merge_micro_summaries(document, &micro_summaries).await?;
        let reduce_duration = reduce_start.elapsed();
        info!("Merged micro-summaries in {:?}", reduce_duration);

        // Calculate metrics
        let total_duration = start_time.elapsed();
        let metrics = SummarizationMetrics {
            processing_time_ms: total_duration.as_millis() as u64,
            input_tokens: selected_segments
                .iter()
                .map(|s| estimate_tokens(&s.text))
                .sum(),
            output_tokens: estimate_tokens(&summary_text),
        };

        // Create response
        let response = SummarizeResponse {
            summary_text,
            bullets,
            citations: Some(citations),
            guardrails: None,
            metrics,
        };

        Ok(response)
    }

    /// Partition segments into groups based on token limit
    fn partition_segments<'a>(&self, selected_segments: &[&'a Segment]) -> Result<Vec<Vec<&'a Segment>>, Box<dyn std::error::Error>> {
        let mut groups: Vec<Vec<&'a Segment>> = Vec::new();
        let mut current_group: Vec<&'a Segment> = Vec::new();
        let mut current_tokens = 0;

        for segment in selected_segments {
            let segment_tokens = estimate_tokens(&segment.text);
            
            // If adding this segment would exceed the limit and we already have segments in the current group
            if current_tokens + segment_tokens > self.config.map_reduce.map_group_tokens && !current_group.is_empty() {
                // Save the current group and start a new one
                groups.push(current_group);
                current_group = Vec::new();
                current_tokens = 0;
            }
            
            // Add the segment to the current group
            current_group.push(segment);
            current_tokens += segment_tokens;
            
            // If a single segment exceeds the limit, it forms its own group
            if current_tokens > self.config.map_reduce.map_group_tokens {
                groups.push(current_group);
                current_group = Vec::new();
                current_tokens = 0;
            }
        }
        
        // Don't forget the last group
        if !current_group.is_empty() {
            groups.push(current_group);
        }

        Ok(groups)
    }

    /// Generate micro-summaries for each group
    async fn generate_micro_summaries(
        &self,
        document: &Document,
        groups: &[Vec<&Segment>],
    ) -> Result<Vec<MicroSummary>, Box<dyn std::error::Error>> {
        // Limit concurrency
        let mut micro_summaries = Vec::new();
        
        // Process groups with limited concurrency
        for chunk in groups.chunks(self.config.map_reduce.concurrency) {
            let mut futures = Vec::new();
            
            for group in chunk {
                let summary_future = self.generate_micro_summary(document, group);
                futures.push(summary_future);
            }
            
            // Wait for all futures in this chunk to complete
            let results = futures::future::join_all(futures).await;
            
            // Collect results
            for result in results {
                micro_summaries.push(result?);
            }
        }

        Ok(micro_summaries)
    }

    /// Generate a micro-summary for a group of segments
    async fn generate_micro_summary(
        &self,
        document: &Document,
        group: &[&Segment],
    ) -> Result<MicroSummary, Box<dyn std::error::Error>> {
        // Build the map prompt
        let prompt = self.build_map_prompt(document, group)?;
        debug!("Built map prompt with {} characters", prompt.len());

        // Try to generate micro-summary via API
        let api_result = self
            .generate_micro_summary_via_api(&prompt, group)
            .await;

        match api_result {
            Ok((text, used_segment_ids)) => {
                Ok(MicroSummary {
                    text,
                    used_segment_ids,
                })
            }
            Err(e) => {
                warn!("Map API call failed: {}. Using fallback.", e);
                // Fallback: concatenate first sentences of segments
                let mut fallback_text = String::new();
                for segment in group.iter().take(2) {
                    fallback_text.push_str(&segment.text);
                    fallback_text.push(' ');
                }
                fallback_text = fallback_text.trim().to_string();
                
                let used_segment_ids: Vec<String> = group
                    .iter()
                    .map(|segment| segment.segment_id.clone())
                    .collect();
                
                Ok(MicroSummary {
                    text: fallback_text,
                    used_segment_ids,
                })
            }
        }
    }

    /// Build a guarded map prompt
    fn build_map_prompt(
        &self,
        document: &Document,
        group: &[&Segment],
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut prompt = String::new();

        // Add document title and language if present
        if !document.title.is_empty() {
            prompt.push_str(&format!("Document Title: {}\n", document.title));
        }
        if !document.lang.is_empty() {
            prompt.push_str(&format!("Language: {}\n", document.lang));
        }

        prompt.push_str("\nSelected Passages:\n");

        // Add selected passages with their IDs
        for (i, segment) in group.iter().enumerate() {
            prompt.push_str(&format!("{}. {} (ID: {})\n", i + 1, segment.text, segment.segment_id));
        }

        prompt.push_str("\nInstructions:\n");
        prompt.push_str("Please provide a concise summary of the selected passages above in 50-100 words.\n");
        prompt.push_str("Do not include any information that is not present in the selected passages.\n");
        prompt.push_str("List the passage IDs that were used in your summary.\n");
        prompt.push_str("If there is insufficient information to create a meaningful summary, please include a short \"Not in sources\" note.\n");

        Ok(prompt)
    }

    /// Generate micro-summary via OpenAI-compatible API
    async fn generate_micro_summary_via_api(
        &self,
        prompt: &str,
        group: &[&Segment],
    ) -> Result<(String, Vec<String>), Box<dyn std::error::Error>> {
        // Create the request
        let request = ChatCompletionRequest {
            model: self.config.model.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        // Build the API URL
        let api_url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );

        // Prepare headers
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(api_key) = &self.config.api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", api_key).parse()?,
            );
        }
        headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse()?);

        // Make the API call with timeout
        let timeout_duration = Duration::from_millis(self.config.timeout_ms);
        let response_result = timeout(
            timeout_duration,
            self.client
                .post(&api_url)
                .headers(headers)
                .json(&request)
                .send(),
        )
        .await;

        // Handle timeout
        let response = match response_result {
            Ok(Ok(response)) => response,
            Ok(Err(e)) => return Err(format!("API request failed: {}", e).into()),
            Err(_) => return Err("API request timed out".into()),
        };

        // Check status
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(
                format!("API request failed with status {}: {}", status, error_text).into(),
            );
        }

        // Parse response
        let api_response: ChatCompletionResponse = response.json().await?;

        // Extract content
        if api_response.choices.is_empty() {
            return Err("API returned no choices".into());
        }

        let content = &api_response.choices[0].message.content;
        
        // Extract used segment IDs (simple approach - in a real implementation, we might parse them more carefully)
        let used_segment_ids: Vec<String> = group
            .iter()
            .map(|segment| segment.segment_id.clone())
            .collect();

        Ok((content.clone(), used_segment_ids))
    }

    /// Merge micro-summaries into a final summary
    async fn merge_micro_summaries(
        &self,
        document: &Document,
        micro_summaries: &[MicroSummary],
    ) -> Result<(String, Option<Vec<String>>, Vec<Citation>), Box<dyn std::error::Error>> {
        // Build the reduce prompt
        let prompt = self.build_reduce_prompt(document, micro_summaries)?;
        debug!("Built reduce prompt with {} characters", prompt.len());

        // Try to generate final summary via API
        let api_result = self
            .generate_final_summary_via_api(&prompt, micro_summaries)
            .await;

        match api_result {
            Ok((summary_text, bullets, citations)) => {
                Ok((summary_text, bullets, citations))
            }
            Err(e) => {
                warn!("Reduce API call failed: {}. Using fallback.", e);
                // Fallback: concatenate micro-summaries
                let mut fallback_text = String::new();
                for summary in micro_summaries.iter().take(3) {
                    fallback_text.push_str(&summary.text);
                    fallback_text.push(' ');
                }
                fallback_text = fallback_text.trim().to_string();
                
                // Simple citations from all micro-summaries
                let mut all_segment_ids = Vec::new();
                for summary in micro_summaries {
                    all_segment_ids.extend(summary.used_segment_ids.clone());
                }
                
                // Deduplicate while preserving order
                let mut seen = std::collections::HashSet::new();
                let citations: Vec<Citation> = all_segment_ids
                    .into_iter()
                    .filter(|id| seen.insert(id.clone()))
                    .map(|segment_id| Citation {
                        segment_id,
                        start: 0,
                        end: 0,
                    })
                    .collect();
                
                Ok((fallback_text, None, citations))
            }
        }
    }

    /// Build a guarded reduce prompt
    fn build_reduce_prompt(
        &self,
        document: &Document,
        micro_summaries: &[MicroSummary],
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut prompt = String::new();

        // Add document title and language if present
        if !document.title.is_empty() {
            prompt.push_str(&format!("Document Title: {}\n", document.title));
        }
        if !document.lang.is_empty() {
            prompt.push_str(&format!("Language: {}\n", document.lang));
        }

        prompt.push_str("\nMicro-summaries:\n");

        // Add micro-summaries
        for (i, summary) in micro_summaries.iter().enumerate() {
            prompt.push_str(&format!("{}. {}\n", i + 1, summary.text));
        }

        prompt.push_str("\nInstructions:\n");
        prompt.push_str(&format!("Please provide a concise summary of the micro-summaries above in {} words.\n", self.config.map_reduce.reduce_target_words));
        prompt.push_str("Also include 3-6 bullet points highlighting the key information.\n");
        prompt.push_str("Do not include any information that is not present in the micro-summaries.\n");
        prompt.push_str("If there is insufficient information to create a meaningful summary, please include a short \"Not in sources\" note.\n");

        Ok(prompt)
    }

    /// Generate final summary via OpenAI-compatible API
    async fn generate_final_summary_via_api(
        &self,
        prompt: &str,
        micro_summaries: &[MicroSummary],
    ) -> Result<(String, Option<Vec<String>>, Vec<Citation>), Box<dyn std::error::Error>> {
        // Create the request
        let request = ChatCompletionRequest {
            model: self.config.model.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        // Build the API URL
        let api_url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );

        // Prepare headers
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(api_key) = &self.config.api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", api_key).parse()?,
            );
        }
        headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse()?);

        // Make the API call with timeout
        let timeout_duration = Duration::from_millis(self.config.timeout_ms);
        let response_result = timeout(
            timeout_duration,
            self.client
                .post(&api_url)
                .headers(headers)
                .json(&request)
                .send(),
        )
        .await;

        // Handle timeout
        let response = match response_result {
            Ok(Ok(response)) => response,
            Ok(Err(e)) => return Err(format!("API request failed: {}", e).into()),
            Err(_) => return Err("API request timed out".into()),
        };

        // Check status
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(
                format!("API request failed with status {}: {}", status, error_text).into(),
            );
        }

        // Parse response
        let api_response: ChatCompletionResponse = response.json().await?;

        // Extract content
        if api_response.choices.is_empty() {
            return Err("API returned no choices".into());
        }

        let content = &api_response.choices[0].message.content;

        // For this implementation, we'll do a simple parse of the response to extract bullets
        let (summary_text, bullets) = if self.config.style == SummaryStyle::AbstractWithBullets {
            // Try to split into summary and bullets
            if let Some(pos) = content.find("\n-") {
                let summary = content[..pos].trim().to_string();
                let bullets_str = &content[pos + 1..];
                let bullet_points: Vec<String> = bullets_str
                    .lines()
                    .filter_map(|line| {
                        let clean_line = line.trim();
                        if clean_line.starts_with("- ") {
                            Some(clean_line[2..].to_string())
                        } else if clean_line.starts_with("* ") {
                            Some(clean_line[2..].to_string())
                        } else {
                            None
                        }
                    })
                    .collect();
                (summary, Some(bullet_points))
            } else {
                // If no bullets found, return entire content as summary
                (content.clone(), None)
            }
        } else {
            // For other styles, return entire content as summary
            (content.clone(), None)
        };

        // Collect all used segment IDs and create citations
        let mut all_segment_ids = Vec::new();
        for summary in micro_summaries {
            all_segment_ids.extend(summary.used_segment_ids.clone());
        }
        
        // Deduplicate while preserving order
        let mut seen = std::collections::HashSet::new();
        let citations: Vec<Citation> = all_segment_ids
            .into_iter()
            .filter(|id| seen.insert(id.clone()))
            .map(|segment_id| Citation {
                segment_id,
                start: 0, // Placeholder values
                end: 0,   // Placeholder values
            })
            .collect();

        Ok((summary_text, bullets, citations))
    }
}

/// Estimate the number of tokens in a text
/// This is a very rough estimation, a real implementation would use a tokenizer
fn estimate_tokens(text: &str) -> usize {
    // Very rough estimation: assume average of 4 characters per token
    text.chars().count() / 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel::{AnalyzeResponse, Document, Segment, SegmentScore};

    // Helper function to create a test document
    fn create_test_document() -> Document {
        Document {
            schema_version: "1.0.0".to_string(),
            doc_id: "test-doc-123".to_string(),
            url: "https://example.com".to_string(),
            title: "Test Document".to_string(),
            lang: "en".to_string(),
            fetched_at: "2023-01-01T00:00:00Z".to_string(),
            segments: vec![
                Segment {
                    segment_id: "seg-1".to_string(),
                    text: "This is the first segment.".to_string(),
                    path: "body > p:nth-child(1)".to_string(),
                    position: 0,
                },
                Segment {
                    segment_id: "seg-2".to_string(),
                    text: "This is the second segment.".to_string(),
                    path: "body > p:nth-child(2)".to_string(),
                    position: 1,
                },
                Segment {
                    segment_id: "seg-3".to_string(),
                    text: "This is the third segment.".to_string(),
                    path: "body > p:nth-child(3)".to_string(),
                    position: 2,
                },
            ],
            hints: None,
        }
    }

    // Helper function to create a test analysis response
    fn create_test_analysis() -> AnalyzeResponse {
        AnalyzeResponse {
            doc_id: "test-doc-123".to_string(),
            model_fingerprint: "model456".to_string(),
            top_segments: vec![
                SegmentScore {
                    segment_id: "seg-1".to_string(),
                    score_representative: 0.95,
                    score_diversity: 0.85,
                    reason: "Highly central".to_string(),
                    score_rerank: None,
                },
                SegmentScore {
                    segment_id: "seg-2".to_string(),
                    score_representative: 0.85,
                    score_diversity: 0.75,
                    reason: "Diverse content".to_string(),
                    score_rerank: None,
                },
            ],
            metrics: kernel::AnalysisMetrics {
                num_segments: 3,
                top_n: 2,
                mmr_lambda: 0.65,
                avg_pairwise_cosine: 0.3,
            },
        }
    }

    #[test]
    fn test_resolve_selected_segments() {
        let document = create_test_document();
        let analysis = create_test_analysis();
        let _analysis = &analysis; // Silence unused variable warning

        let config = SummarizerConfig::new();
        let summarizer = Summarizer::new(config).unwrap();

        let selected_segments = summarizer
            .resolve_selected_segments(&document, &analysis)
            .unwrap();

        assert_eq!(selected_segments.len(), 2);
        assert_eq!(selected_segments[0].segment_id, "seg-1");
        assert_eq!(selected_segments[1].segment_id, "seg-2");
    }

    #[test]
    fn test_build_prompt() {
        let document = create_test_document();
        let _analysis = create_test_analysis();
        let selected_segments = vec![&document.segments[0], &document.segments[1]];

        let config = SummarizerConfig::new();
        let summarizer = Summarizer::new(config).unwrap();

        let prompt = summarizer
            .build_prompt(&document, &selected_segments)
            .unwrap();

        assert!(prompt.contains("Document Title: Test Document"));
        assert!(prompt.contains("Language: en"));
        assert!(prompt.contains("1. This is the first segment."));
        assert!(prompt.contains("2. This is the second segment."));
        assert!(prompt.contains("Please provide a concise summary"));
    }

    #[test]
    fn test_extractive_fallback() {
        let document = create_test_document();
        let selected_segments = vec![&document.segments[0], &document.segments[1]];

        let config = SummarizerConfig::new();
        let summarizer = Summarizer::new(config).unwrap();

        let start_time = std::time::Instant::now();
        let (summary, metrics) = summarizer.extractive_fallback(&selected_segments, start_time);

        assert!(!summary.is_empty());
        // Remove the useless comparison warning
        // assert!(metrics.processing_time_ms >= 0); // u64 is always >= 0
        assert!(metrics.input_tokens > 0);
        assert!(metrics.output_tokens > 0);
    }

    #[test]
    fn test_partition_segments() {
        let document = create_test_document();
        let selected_segments = vec![&document.segments[0], &document.segments[1], &document.segments[2]];

        let config = SummarizerConfig::new();
        let summarizer = Summarizer::new(config).unwrap();

        // Test partitioning with a small group size to force multiple groups
        let groups = summarizer.partition_segments(&selected_segments).unwrap();
        
        // With our token estimation, all segments should fit in one group with default settings
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
    }
}
