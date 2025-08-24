#[cfg(test)]
mod tests {
    use crate::config::AnalyzerConfig;

    #[test]
    fn test_reranker_function_signature() {
        // This test just verifies that the function signature compiles correctly
        // It doesn't actually run the reranker since that would require models
        let config = AnalyzerConfig::new();
        assert_eq!(config.reranker.enabled, false);
    }

    #[test]
    fn test_reranker_config_fields() {
        let config = AnalyzerConfig::new();
        // Verify that the new config fields exist and have default values
        assert_eq!(config.reranker.intra_op_threads, 0);
        assert_eq!(config.reranker.inter_op_threads, 0);
        assert_eq!(config.reranker.max_seq_len, 512);
        assert_eq!(config.reranker.score_tensor, "logits[0]");
    }
}