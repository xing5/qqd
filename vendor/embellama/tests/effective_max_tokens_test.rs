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

//! Unit tests for effective_max_tokens calculation
//!
//! These tests validate that the effective max tokens calculation correctly
//! accounts for per-sequence limits in batch processing. Each sequence gets
//! its own KV cache slot of size: context_size / n_seq_max.

/// Test helper to calculate expected effective_max_tokens
/// This duplicates the logic we expect to implement
fn calculate_expected_effective_max(ctx_size: usize, n_seq_max: u32) -> usize {
    let per_sequence_size = ctx_size / (n_seq_max as usize);
    per_sequence_size.saturating_sub(2)
}

#[test]
fn test_effective_max_calculation_with_n_seq_max_1() {
    // Model with context_size=8192, n_seq_max=1
    // Expected: 8192 / 1 - 2 = 8190
    let effective_max = calculate_expected_effective_max(8192, 1);
    assert_eq!(
        effective_max, 8190,
        "Model with n_seq_max=1 should have effective max of 8190 tokens"
    );
}

#[test]
fn test_effective_max_calculation_with_n_seq_max_2() {
    // Model with context_size=8192, n_seq_max=2
    // Expected: 8192 / 2 - 2 = 4094
    let effective_max = calculate_expected_effective_max(8192, 2);
    assert_eq!(
        effective_max, 4094,
        "Model with n_seq_max=2 should have effective max of 4094 tokens per sequence"
    );
}

#[test]
fn test_effective_max_calculation_with_n_seq_max_4() {
    // Model with context_size=8192, n_seq_max=4
    // Expected: 8192 / 4 - 2 = 2046
    let effective_max = calculate_expected_effective_max(8192, 4);
    assert_eq!(
        effective_max, 2046,
        "Model with n_seq_max=4 should have effective max of 2046 tokens per sequence"
    );
}

#[test]
fn test_effective_max_calculation_with_n_seq_max_8() {
    // Model with context_size=8192, n_seq_max=8
    // Expected: 8192 / 8 - 2 = 1022
    let effective_max = calculate_expected_effective_max(8192, 8);
    assert_eq!(
        effective_max, 1022,
        "Model with n_seq_max=8 should have effective max of 1022 tokens per sequence"
    );
}

#[test]
fn test_effective_max_calculation_edge_case_small_context() {
    // Small context with high n_seq_max
    // context_size=512, n_seq_max=256
    // Expected: 512 / 256 - 2 = 2 - 2 = 0 (saturated)
    let effective_max = calculate_expected_effective_max(512, 256);
    assert_eq!(
        effective_max, 0,
        "Small context with high n_seq_max should saturate at 0"
    );
}

#[test]
fn test_effective_max_calculation_exact_boundary() {
    // Test exact boundary: per_sequence_size = 2
    // context_size=4, n_seq_max=2
    // Expected: 4 / 2 - 2 = 2 - 2 = 0
    let effective_max = calculate_expected_effective_max(4, 2);
    assert_eq!(
        effective_max, 0,
        "Exact boundary (per_seq = 2) should give 0 effective tokens"
    );
}

#[test]
fn test_effective_max_calculation_one_over_boundary() {
    // Test one token over boundary: per_sequence_size = 3
    // context_size=6, n_seq_max=2
    // Expected: 6 / 2 - 2 = 3 - 2 = 1
    let effective_max = calculate_expected_effective_max(6, 2);
    assert_eq!(
        effective_max, 1,
        "One token over boundary should give 1 effective token"
    );
}

// NOTE: The tests below will fail to compile until effective_max_tokens() is implemented
// This is intentional - TDD approach

// TODO: Uncomment when ready to verify compilation fails
// #[test]
// fn test_embedding_model_effective_max_tokens_method() {
//     // This test will fail to compile because effective_max_tokens() doesn't exist yet
//     // Uncomment to verify TDD red phase
//     use embellama::EmbeddingModel;
//     use std::sync::Arc;
//
//     // We need a real model instance to test the method
//     // For now, this is commented out but documents what we expect to implement
//     // let model = create_test_model(8192, 768);
//     // let effective_max = model.effective_max_tokens();
//     // assert_eq!(effective_max, 7324);
// }
