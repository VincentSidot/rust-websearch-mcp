//! Embedding cache implementation using sled

use crate::config::CacheConfig;
use log::{debug, info, warn};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use sled::Db;
use std::path::Path;
use std::sync::Arc;

/// Cache key format: emb:{segment_id}:{embedding_model_fingerprint}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheKey {
    pub segment_id: String,
    pub model_fingerprint: String,
}

impl CacheKey {
    /// Create a new cache key
    pub fn new(segment_id: String, model_fingerprint: String) -> Self {
        Self {
            segment_id,
            model_fingerprint,
        }
    }

    /// Convert to string key for sled
    pub fn to_string_key(&self) -> String {
        format!("emb:{}:{}", self.segment_id, self.model_fingerprint)
    }

    /// Parse from string key
    pub fn from_string_key(key: &str) -> Option<Self> {
        if let Some(stripped) = key.strip_prefix("emb:") {
            let parts: Vec<&str> = stripped.split(':').collect();
            if parts.len() == 2 {
                return Some(Self {
                    segment_id: parts[0].to_string(),
                    model_fingerprint: parts[1].to_string(),
                });
            }
        }
        None
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entry_count: u64,
    pub total_size_bytes: u64,
}

/// Embedding cache using sled
pub struct EmbeddingCache {
    /// Sled database instance
    db: Arc<Db>,
    
    /// Cache configuration
    config: CacheConfig,
    
    /// Cache hit counter
    hit_count: std::sync::atomic::AtomicU64,
    
    /// Cache miss counter
    miss_count: std::sync::atomic::AtomicU64,
}

impl EmbeddingCache {
    /// Create a new embedding cache
    pub fn new(config: CacheConfig) -> Result<Self, Box<dyn std::error::Error>> {
        if !config.enabled {
            info!("Embedding cache is disabled");
            return Ok(Self {
                db: Arc::new(sled::Config::new().temporary(true).open()?),
                config,
                hit_count: std::sync::atomic::AtomicU64::new(0),
                miss_count: std::sync::atomic::AtomicU64::new(0),
            });
        }

        // Ensure cache directory exists
        let cache_path = Path::new(&config.path);
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        info!("Initializing embedding cache at: {:?}", cache_path);
        let db = sled::Config::new().path(cache_path).open()?;

        Ok(Self {
            db: Arc::new(db),
            config,
            hit_count: std::sync::atomic::AtomicU64::new(0),
            miss_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Get an embedding from the cache
    pub fn get_embedding(
        &self,
        segment_id: &str,
        model_fingerprint: &str,
    ) -> Result<Option<Array2<f32>>, Box<dyn std::error::Error>> {
        if !self.config.enabled {
            return Ok(None);
        }

        let key = CacheKey::new(segment_id.to_string(), model_fingerprint.to_string());
        let string_key = key.to_string_key();

        match self.db.get(&string_key)? {
            Some(value) => {
                // Try to deserialize the value
                match bincode::deserialize::<Vec<f32>>(&value) {
                    Ok(data) => {
                        // Convert to Array2 assuming a single row
                        let array = Array2::from_shape_vec((1, data.len()), data)?;
                        self.hit_count
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        debug!("Cache hit for key: {}", string_key);
                        Ok(Some(array))
                    }
                    Err(e) => {
                        // Corrupted entry, treat as miss and warn
                        warn!("Corrupted cache entry for key: {} - {:?}", string_key, e);
                        self.db.remove(&string_key)?;
                        self.miss_count
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        Ok(None)
                    }
                }
            }
            None => {
                self.miss_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                debug!("Cache miss for key: {}", string_key);
                Ok(None)
            }
        }
    }

    /// Put an embedding into the cache
    pub fn put_embedding(
        &self,
        segment_id: &str,
        model_fingerprint: &str,
        embedding: &Array2<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.config.enabled {
            return Ok(());
        }

        let key = CacheKey::new(segment_id.to_string(), model_fingerprint.to_string());
        let string_key = key.to_string_key();

        // Convert Array2 to Vec<f32> (flatten)
        let data: Vec<f32> = embedding.iter().cloned().collect();
        
        // Serialize to bytes
        let serialized = bincode::serialize(&data)?;
        
        self.db.insert(&string_key, serialized)?;
        debug!("Cached embedding for key: {}", string_key);
        Ok(())
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> Result<CacheStats, Box<dyn std::error::Error>> {
        let entry_count = self.db.len() as u64;
        
        // Estimate total size by sampling a few entries
        let mut total_size_bytes = 0u64;
        let mut sampled_entries = 0u64;
        
        for result in self.db.iter().take(100) {
            let (_key, value) = result?;
            total_size_bytes += value.len() as u64;
            sampled_entries += 1;
        }
        
        // Extrapolate total size if we sampled entries
        if sampled_entries > 0 && entry_count > 0 {
            let avg_size = total_size_bytes / sampled_entries;
            total_size_bytes = avg_size * entry_count;
        }
        
        Ok(CacheStats {
            entry_count,
            total_size_bytes,
        })
    }

    /// Clear the cache
    pub fn clear(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.db.clear()?;
        info!("Cleared embedding cache");
        Ok(())
    }

    /// Get hit/miss counts and ratio
    pub fn get_hit_miss_stats(&self) -> (u64, u64, f64) {
        let hits = self.hit_count.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.miss_count.load(std::sync::atomic::Ordering::Relaxed);
        let ratio = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        };
        (hits, misses, ratio)
    }

    /// Reset hit/miss counters
    pub fn reset_counters(&self) {
        self.hit_count.store(0, std::sync::atomic::Ordering::Relaxed);
        self.miss_count.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cache_key_serialization() {
        let key = CacheKey::new("seg-123".to_string(), "model-abc@v1".to_string());
        let string_key = key.to_string_key();
        assert_eq!(string_key, "emb:seg-123:model-abc@v1");
        
        let parsed = CacheKey::from_string_key(&string_key).unwrap();
        assert_eq!(parsed.segment_id, "seg-123");
        assert_eq!(parsed.model_fingerprint, "model-abc@v1");
    }

    #[test]
    fn test_cache_operations() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let cache_path = temp_dir.path().join("test_cache");
        
        let config = CacheConfig {
            enabled: true,
            path: cache_path.to_string_lossy().to_string(),
            ttl_days: None,
        };
        
        let cache = EmbeddingCache::new(config)?;
        
        let segment_id = "test-segment";
        let model_fingerprint = "test-model@v1";
        let embedding = Array2::from_shape_vec((1, 384), vec![0.1; 384])?;
        
        // Test cache miss
        let result = cache.get_embedding(segment_id, model_fingerprint)?;
        assert!(result.is_none());
        
        // Test cache put
        cache.put_embedding(segment_id, model_fingerprint, &embedding)?;
        
        // Test cache hit
        let result = cache.get_embedding(segment_id, model_fingerprint)?;
        assert!(result.is_some());
        let cached_embedding = result.unwrap();
        assert_eq!(cached_embedding.shape(), &[1, 384]);
        
        // Test cache stats
        let stats = cache.get_stats()?;
        assert_eq!(stats.entry_count, 1);
        
        // Test cache clear
        cache.clear()?;
        let stats = cache.get_stats()?;
        assert_eq!(stats.entry_count, 0);
        
        Ok(())
    }

    #[test]
    fn test_disabled_cache() -> Result<(), Box<dyn std::error::Error>> {
        let config = CacheConfig {
            enabled: false,
            path: "./.cache/analyzer".to_string(),
            ttl_days: None,
        };
        
        let cache = EmbeddingCache::new(config)?;
        let segment_id = "test-segment";
        let model_fingerprint = "test-model@v1";
        let embedding = Array2::from_shape_vec((1, 384), vec![0.1; 384])?;
        
        // Operations should succeed but not actually cache
        cache.put_embedding(segment_id, model_fingerprint, &embedding)?;
        let result = cache.get_embedding(segment_id, model_fingerprint)?;
        assert!(result.is_none());
        
        Ok(())
    }
}