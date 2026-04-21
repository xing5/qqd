// Copyright 2025 Embellama Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Prefix caching module for KV cache optimization.
//!
//! This module provides session-based prefix caching to reuse KV cache computations
//! for common text prefixes, achieving significant speedup for repetitive patterns.

use crate::cache::CacheMetrics;
use crate::error::{Error, Result};
use dashmap::DashMap;
use lru::LruCache;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tracing::info;

/// Minimum prefix length in tokens to consider for caching
/// Shorter prefixes have too much overhead relative to benefit
const MIN_PREFIX_LENGTH: usize = 100;

/// Maximum number of prefixes to track for frequency analysis
const MAX_TRACKED_PREFIXES: usize = 1000;

// Default frequency threshold for automatic caching
// const DEFAULT_FREQUENCY_THRESHOLD: usize = 5;

// Thread-local storage for prefix cache access
thread_local! {
    static LOCAL_PREFIX_CACHE: RefCell<LruCache<String, Arc<SessionData>>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(100).unwrap()));
}

/// Represents a saved KV cache session state
#[derive(Debug)]
pub struct SessionData {
    /// The prefix text this session represents
    pub prefix: String,
    /// Number of tokens in the prefix
    pub token_count: usize,
    /// Path to the saved session file (if persisted)
    pub file_path: Option<PathBuf>,
    /// In-memory session state (if available)
    pub memory_state: Option<Vec<u8>>,
    /// Creation timestamp
    pub created_at: Instant,
    /// Last accessed timestamp
    pub last_accessed: AtomicU64,
    /// Access count
    pub access_count: AtomicUsize,
}

impl SessionData {
    /// Create a new session data instance
    pub fn new(prefix: String, token_count: usize) -> Self {
        Self {
            prefix,
            token_count,
            file_path: None,
            memory_state: None,
            created_at: Instant::now(),
            last_accessed: AtomicU64::new(0),
            access_count: AtomicUsize::new(0),
        }
    }

    /// Update access statistics
    pub fn touch(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        let now = Instant::now().elapsed().as_secs();
        self.last_accessed.store(now, Ordering::Relaxed);
    }

    /// Get the age of this session in seconds
    pub fn age_seconds(&self) -> u64 {
        self.created_at.elapsed().as_secs()
    }
}

/// Trie node for efficient prefix matching
#[derive(Default)]
struct TrieNode {
    /// Child nodes indexed by token
    children: HashMap<i32, Box<TrieNode>>,
    /// Session data if this node represents a cached prefix
    session: Option<Arc<SessionData>>,
    /// Frequency count for this prefix
    #[allow(dead_code)]
    frequency: usize,
}

/// Prefix detector that analyzes text patterns for caching opportunities
pub struct PrefixDetector {
    /// Root of the prefix trie
    root: RwLock<TrieNode>,
    /// Frequency tracking for potential prefixes
    frequency_map: DashMap<String, usize>,
    /// Minimum frequency threshold for automatic caching
    frequency_threshold: usize,
    /// Maximum tracked prefixes
    max_tracked: usize,
}

impl PrefixDetector {
    /// Create a new prefix detector
    pub fn new(frequency_threshold: usize) -> Self {
        Self {
            root: RwLock::new(TrieNode::default()),
            frequency_map: DashMap::new(),
            frequency_threshold,
            max_tracked: MAX_TRACKED_PREFIXES,
        }
    }

    /// Analyze a token sequence and track prefix patterns
    pub fn analyze_tokens(&self, tokens: &[i32]) -> Option<usize> {
        if tokens.len() < MIN_PREFIX_LENGTH {
            return None;
        }

        // Track various prefix lengths
        let prefix_lengths = [
            MIN_PREFIX_LENGTH,
            MIN_PREFIX_LENGTH * 2,
            tokens.len() / 2,
            tokens.len() * 3 / 4,
        ];

        let mut best_prefix_len = None;
        let mut best_frequency = 0;

        for &len in &prefix_lengths {
            if len > tokens.len() {
                continue;
            }

            let prefix_key = Self::tokens_to_key(&tokens[..len]);

            let mut count = self.frequency_map.entry(prefix_key.clone()).or_insert(0);
            *count += 1;

            if *count >= self.frequency_threshold && *count > best_frequency {
                best_frequency = *count;
                best_prefix_len = Some(len);
            }
        }

        // Clean up old entries if we're tracking too many
        if self.frequency_map.len() > self.max_tracked {
            self.cleanup_infrequent();
        }

        best_prefix_len
    }

    /// Find the longest matching prefix in the trie
    ///
    /// # Panics
    ///
    /// May panic if the detector lock is poisoned
    pub fn find_longest_prefix(&self, tokens: &[i32]) -> Option<(usize, Arc<SessionData>)> {
        let root = self.root.read().unwrap();
        let mut node = &*root;
        let mut best_match = None;

        for (idx, &token) in tokens.iter().enumerate() {
            match node.children.get(&token) {
                Some(child) => {
                    node = child;
                    // Check if this node has a cached session
                    if let Some(session) = &node.session {
                        best_match = Some((idx + 1, session.clone()));
                    }
                }
                None => break,
            }
        }

        best_match
    }

    /// Insert a cached prefix into the trie
    ///
    /// # Panics
    ///
    /// May panic if the detector lock is poisoned
    pub fn insert_prefix(&self, tokens: &[i32], session: Arc<SessionData>) {
        if tokens.len() < MIN_PREFIX_LENGTH {
            return;
        }

        let mut root = self.root.write().unwrap();
        let mut node = &mut *root;

        for &token in tokens {
            node = node
                .children
                .entry(token)
                .or_insert_with(|| Box::new(TrieNode::default()));
        }

        node.session = Some(session);
    }

    /// Convert tokens to a cache key
    fn tokens_to_key(tokens: &[i32]) -> String {
        let mut hasher = Sha256::new();
        for &token in tokens {
            hasher.update(token.to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    /// Clean up infrequently accessed prefixes
    fn cleanup_infrequent(&self) {
        let threshold = self.frequency_threshold / 2;
        self.frequency_map.retain(|_, count| *count >= threshold);
    }
}

/// Main prefix cache manager
pub struct PrefixCache {
    /// Shared cache of session data
    sessions: Arc<DashMap<String, Arc<SessionData>>>,
    /// Prefix detector for pattern analysis
    detector: Arc<PrefixDetector>,
    /// Maximum number of cached sessions
    max_sessions: usize,
    /// TTL for cached sessions
    ttl_seconds: u64,
    /// Cache metrics
    metrics: Arc<CacheMetrics>,
    /// Directory for persistent session files
    session_dir: Option<PathBuf>,
}

impl PrefixCache {
    /// Create a new prefix cache
    ///
    /// # Errors
    ///
    /// Returns an error if the session directory cannot be created
    pub fn new(
        max_sessions: usize,
        ttl_seconds: u64,
        frequency_threshold: usize,
        session_dir: Option<PathBuf>,
    ) -> Result<Self> {
        // Create session directory if specified
        if let Some(ref dir) = session_dir {
            std::fs::create_dir_all(dir).map_err(|e| Error::ConfigurationError {
                message: format!("Failed to create session directory: {e}"),
            })?;
        }

        Ok(Self {
            sessions: Arc::new(DashMap::new()),
            detector: Arc::new(PrefixDetector::new(frequency_threshold)),
            max_sessions,
            ttl_seconds,
            metrics: Arc::new(CacheMetrics::default()),
            session_dir,
        })
    }

    /// Check if a text has a cached prefix and return the session
    pub fn find_prefix_session(
        &self,
        text: &str,
        tokens: &[i32],
    ) -> Option<(usize, Arc<SessionData>)> {
        // First check thread-local cache
        let key = Self::compute_key(text);
        let local_hit = LOCAL_PREFIX_CACHE.with(|cache| cache.borrow_mut().get(&key).cloned());

        if let Some(session) = local_hit {
            self.metrics.hits.fetch_add(1, Ordering::Relaxed);
            session.touch();
            return Some((session.token_count, session));
        }

        // Check shared trie for longest prefix match
        if let Some((prefix_len, session)) = self.detector.find_longest_prefix(tokens) {
            // Validate session is still valid
            if session.age_seconds() < self.ttl_seconds {
                self.metrics.hits.fetch_add(1, Ordering::Relaxed);
                session.touch();

                // Update thread-local cache
                LOCAL_PREFIX_CACHE.with(|cache| {
                    cache.borrow_mut().put(key, session.clone());
                });

                return Some((prefix_len, session));
            }
        }

        self.metrics.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Register a new prefix for caching
    ///
    /// # Errors
    ///
    /// Returns an error if the cache is full and cannot evict enough entries
    pub fn register_prefix(&self, text: &str, tokens: &[i32], session_data: Vec<u8>) -> Result<()> {
        if tokens.len() < MIN_PREFIX_LENGTH {
            return Err(Error::InvalidInput {
                message: format!(
                    "Prefix too short: {} tokens (minimum: {})",
                    tokens.len(),
                    MIN_PREFIX_LENGTH
                ),
            });
        }

        // Check capacity
        if self.sessions.len() >= self.max_sessions {
            self.evict_oldest();
        }

        let key = Self::compute_key(text);
        let mut session = SessionData::new(text.to_string(), tokens.len());
        session.memory_state = Some(session_data);

        // Optionally save to disk
        if let Some(ref dir) = self.session_dir {
            let file_path = dir.join(format!("{key}.session"));
            // > TODO: Implement actual file saving with llama-cpp session API
            session.file_path = Some(file_path);
        }

        let session = Arc::new(session);

        // Insert into shared cache and trie
        self.sessions.insert(key.clone(), session.clone());
        self.detector.insert_prefix(tokens, session.clone());

        info!(
            "Registered prefix cache for {} tokens (key: {})",
            tokens.len(),
            &key[..8]
        );

        Ok(())
    }

    /// Analyze text for caching opportunities
    pub fn analyze(&self, tokens: &[i32]) -> Option<usize> {
        self.detector.analyze_tokens(tokens)
    }

    /// Clear all cached sessions
    ///
    /// # Panics
    ///
    /// May panic if the detector lock is poisoned
    pub fn clear(&self) {
        self.sessions.clear();
        LOCAL_PREFIX_CACHE.with(|cache| cache.borrow_mut().clear());

        // Clear trie
        *self.detector.root.write().unwrap() = TrieNode::default();

        self.metrics
            .evictions
            .store(self.sessions.len() as u64, Ordering::Relaxed);
    }

    /// Get cache statistics
    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            session_count: self.sessions.len(),
            total_hits: self.metrics.hits.load(Ordering::Relaxed),
            total_misses: self.metrics.misses.load(Ordering::Relaxed),
            total_evictions: self.metrics.evictions.load(Ordering::Relaxed),
            memory_usage_bytes: self.estimate_memory_usage(),
        }
    }

    /// Compute cache key for text
    fn compute_key(text: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Evict oldest sessions when capacity is reached
    fn evict_oldest(&self) {
        let mut oldest_key = None;
        let mut oldest_time = u64::MAX;

        for entry in self.sessions.iter() {
            let last_accessed = entry.value().last_accessed.load(Ordering::Relaxed);
            if last_accessed < oldest_time {
                oldest_time = last_accessed;
                oldest_key = Some(entry.key().clone());
            }
        }

        if let Some(key) = oldest_key {
            self.sessions.remove(&key);
            self.metrics.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> u64 {
        let mut total = 0u64;

        for entry in self.sessions.iter() {
            let session = entry.value();

            // Estimate: prefix string + memory state + overhead
            total += session.prefix.len() as u64;
            if let Some(ref state) = session.memory_state {
                total += state.len() as u64;
            }
            total += 256; // Overhead estimate
        }

        total
    }
}

/// Statistics for prefix cache
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PrefixCacheStats {
    /// Number of cached sessions
    pub session_count: usize,
    /// Total cache hits
    pub total_hits: u64,
    /// Total cache misses
    pub total_misses: u64,
    /// Total evictions due to capacity
    pub total_evictions: u64,
    /// Estimated memory usage in bytes
    pub memory_usage_bytes: u64,
}

impl PrefixCacheStats {
    /// Calculate hit rate percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                (self.total_hits as f64 / total as f64) * 100.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_detector() {
        let detector = PrefixDetector::new(3);

        // Use the same tokens for the prefix - first 100 tokens are the shared prefix
        let tokens1: Vec<i32> = (0..150).collect();
        let tokens2: Vec<i32> = (0..150).collect();
        let tokens3: Vec<i32> = (0..150).collect();

        // First two accesses won't trigger caching
        assert_eq!(detector.analyze_tokens(&tokens1), None);
        assert_eq!(detector.analyze_tokens(&tokens2), None);

        // Third access of same prefix should trigger - MIN_PREFIX_LENGTH is 100
        let result = detector.analyze_tokens(&tokens3);
        assert!(result.is_some(), "Expected Some(_), got None");
        assert_eq!(result.unwrap(), MIN_PREFIX_LENGTH);
    }

    #[test]
    fn test_prefix_trie() {
        let detector = PrefixDetector::new(1);
        let tokens: Vec<i32> = (0..200).collect();

        let session = Arc::new(SessionData::new("test".to_string(), 150));
        detector.insert_prefix(&tokens[..150], session.clone());

        // Should find the prefix
        let result = detector.find_longest_prefix(&tokens);
        assert!(result.is_some());

        let (len, found_session) = result.unwrap();
        assert_eq!(len, 150);
        assert_eq!(found_session.token_count, 150);
    }

    #[test]
    fn test_cache_operations() {
        let cache = PrefixCache::new(10, 3600, 2, None).unwrap();

        let text = "This is a test prefix that is long enough to be cached";
        let tokens: Vec<i32> = (0..150).collect();

        // Initially no match
        assert!(cache.find_prefix_session(text, &tokens).is_none());

        // Register the prefix
        let session_data = vec![1, 2, 3, 4, 5];
        cache.register_prefix(text, &tokens, session_data).unwrap();

        // Now should find it
        let result = cache.find_prefix_session(text, &tokens);
        assert!(result.is_some());

        // Check stats
        let stats = cache.stats();
        assert_eq!(stats.session_count, 1);
        assert_eq!(stats.total_hits, 1);
    }
}
