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

//! Integration tests for prefix cache functionality

#[cfg(test)]
mod tests {
    use embellama::cache::prefix_cache::{PrefixCache, PrefixDetector};
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_prefix_cache_creation() {
        let cache = PrefixCache::new(100, 3600, 5, None).expect("Failed to create cache");
        let stats = cache.stats();
        assert_eq!(stats.session_count, 0);
        assert_eq!(stats.total_hits, 0);
        assert_eq!(stats.total_misses, 0);
    }

    #[test]
    fn test_prefix_registration_and_retrieval() {
        let cache = PrefixCache::new(10, 3600, 5, None).expect("Failed to create cache");

        let text = "import numpy as np\nimport pandas as pd\nclass DataProcessor:";
        let tokens: Vec<i32> = vec![1; 101]; // Mock tokens (101 to exceed MIN_PREFIX_LENGTH)
        let session_data = vec![1, 2, 3, 4, 5]; // Mock session data

        // Register prefix
        let result = cache.register_prefix(text, &tokens, session_data.clone());
        assert!(result.is_ok());

        // Retrieve prefix
        let retrieved = cache.find_prefix_session(text, &tokens);
        assert!(retrieved.is_some());

        // Check stats
        let stats = cache.stats();
        assert_eq!(stats.session_count, 1);
    }

    #[test]
    fn test_prefix_cache_eviction() {
        let cache = PrefixCache::new(2, 3600, 5, None).expect("Failed to create cache"); // Small cache for testing eviction

        // Register first prefix
        let text1 = "prefix1".repeat(20); // Make it long enough
        let tokens1: Vec<i32> = vec![1; 101]; // 101 tokens to exceed MIN_PREFIX_LENGTH
        let session1 = vec![1, 2, 3];
        cache.register_prefix(&text1, &tokens1, session1).unwrap();

        // Register second prefix
        let text2 = "prefix2".repeat(20);
        let tokens2: Vec<i32> = vec![2; 101];
        let session2 = vec![4, 5, 6];
        cache.register_prefix(&text2, &tokens2, session2).unwrap();

        // Register third prefix (should trigger eviction)
        let text3 = "prefix3".repeat(20);
        let tokens3: Vec<i32> = vec![3; 101];
        let session3 = vec![7, 8, 9];
        cache.register_prefix(&text3, &tokens3, session3).unwrap();

        // Cache should still have only 2 entries
        let stats = cache.stats();
        assert!(stats.session_count <= 2);
    }

    #[test]
    fn test_prefix_cache_expiration() {
        let cache = PrefixCache::new(10, 3600, 5, None).expect("Failed to create cache");

        // Register multiple prefixes
        for i in 0..5 {
            let text = format!("prefix_{}", i).repeat(20);
            let tokens: Vec<i32> = vec![i as i32; 101]; // 101 tokens to exceed MIN_PREFIX_LENGTH
            let session = vec![i as u8; 10];
            cache.register_prefix(&text, &tokens, session).unwrap();
        }

        // Verify all are registered
        let stats = cache.stats();
        assert_eq!(stats.session_count, 5);

        // Clear cache
        cache.clear();

        // Verify cache is empty
        let stats = cache.stats();
        assert_eq!(stats.session_count, 0);
    }

    #[test]
    fn test_prefix_detector() {
        let detector = PrefixDetector::new(5);

        // Analyze tokens multiple times to build frequency
        let base_tokens: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut tokens = Vec::new();
        for _ in 0..11 {
            tokens.extend_from_slice(&base_tokens);
        } // 110 tokens

        // First analysis - not enough frequency data yet, so no suggestion
        let suggestion = detector.analyze_tokens(&tokens);
        assert!(
            suggestion.is_none(),
            "First analysis should not suggest caching yet"
        );

        // Analyze same prefix multiple times to increase frequency
        for _ in 0..10 {
            detector.analyze_tokens(&tokens);
        }

        // Should now suggest caching this prefix
        let suggestion = detector.analyze_tokens(&tokens);
        assert!(suggestion.is_some());
    }

    #[test]
    fn test_cache_hit_miss_tracking() {
        let cache = PrefixCache::new(10, 3600, 5, None).expect("Failed to create cache");

        let text = "test_prefix".repeat(20);
        let tokens: Vec<i32> = vec![1; 101]; // 101 tokens to exceed MIN_PREFIX_LENGTH
        let session = vec![1, 2, 3, 4, 5];

        // Register prefix
        cache.register_prefix(&text, &tokens, session).unwrap();

        // Hit: retrieve existing prefix
        let _ = cache.find_prefix_session(&text, &tokens);

        // Miss: try to retrieve non-existent prefix
        let _ = cache.find_prefix_session("nonexistent_text", &[]);

        let stats = cache.stats();
        assert!(stats.total_hits > 0);
        assert!(stats.total_misses > 0);
    }

    #[test]
    fn test_concurrent_prefix_operations() {
        let cache = Arc::new(PrefixCache::new(100, 3600, 5, None).expect("Failed to create cache"));
        let mut handles = vec![];

        // Spawn multiple threads that register and retrieve prefixes
        for i in 0..10 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                let text = format!("thread_{}_prefix", i).repeat(20);
                let tokens: Vec<i32> = vec![i as i32; 101]; // 101 tokens to exceed MIN_PREFIX_LENGTH
                let session = vec![i as u8; 10];

                cache_clone
                    .register_prefix(&text, &tokens, session)
                    .unwrap();

                // Try to retrieve it
                assert!(cache_clone.find_prefix_session(&text, &tokens).is_some());
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Check stats
        let stats = cache.stats();
        assert!(stats.session_count > 0);
    }

    #[test]
    fn test_prefix_cache_with_empty_input() {
        let cache = PrefixCache::new(10, 3600, 5, None).expect("Failed to create cache");

        let text = "";
        let tokens: Vec<i32> = vec![];
        let session = vec![];

        // Empty tokens are below MIN_PREFIX_LENGTH, so registration should fail
        let result = cache.register_prefix(text, &tokens, session);
        assert!(
            result.is_err(),
            "Empty input should fail registration (below MIN_PREFIX_LENGTH)"
        );
    }
}
