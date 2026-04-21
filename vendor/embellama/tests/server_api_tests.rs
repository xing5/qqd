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

//! Core API integration tests for the Embellama server

#![cfg(feature = "server")]

mod server_test_helpers;

use reqwest::StatusCode;
use serial_test::serial;
use server_test_helpers::*;

#[tokio::test]
#[serial]
async fn test_health_endpoint() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let response = client
        .health_check(&server.base_url)
        .await
        .expect("Failed to check health");

    assert_eq!(response.status(), StatusCode::OK);

    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["status"], "healthy");
    assert!(body["model"].is_string());
    assert!(body["version"].is_string());
}

#[tokio::test]
#[serial]
async fn test_models_endpoint() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let response = client
        .list_models(&server.base_url)
        .await
        .expect("Failed to list models");

    assert_eq!(response.status(), StatusCode::OK);

    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert!(body["data"].is_array());
    assert!(!body["data"].as_array().unwrap().is_empty());

    // Check first model
    let first_model = &body["data"][0];
    assert_eq!(first_model["object"], "model");
    assert_eq!(first_model["id"], "test-model");
    assert_eq!(first_model["owned_by"], "embellama");
    // Verify context_size field is present (can be null or a number)
    assert!(
        first_model.get("context_size").is_some(),
        "context_size field should be present"
    );
}

#[tokio::test]
#[serial]
async fn test_single_embedding_short_text() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Single("Hello world".to_string()),
            None,
        )
        .await
        .expect("Failed to create embedding");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();
    validate_embedding_response(&embedding_response, 1);

    // Verify it's float format by default
    match &embedding_response.data[0].embedding {
        EmbeddingValue::Float(vec) => {
            assert!(!vec.is_empty());
        }
        EmbeddingValue::Base64(_) => panic!("Expected float embedding format"),
    }
}

#[tokio::test]
#[serial]
async fn test_single_embedding_medium_text() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let medium_text = "This is a medium length text that contains multiple sentences. \
                       It's designed to test how the embedding API handles typical paragraph-sized inputs. \
                       The text should be processed correctly and return valid embeddings.";

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Single(medium_text.to_string()),
            None,
        )
        .await
        .expect("Failed to create embedding");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();
    validate_embedding_response(&embedding_response, 1);
}

#[tokio::test]
#[serial]
async fn test_single_embedding_long_text() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let long_text = "Lorem ipsum ".repeat(100); // Create a long text

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Single(long_text),
            None,
        )
        .await
        .expect("Failed to create embedding");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();
    validate_embedding_response(&embedding_response, 1);
}

#[tokio::test]
#[serial]
async fn test_single_embedding_special_chars() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let special_text = "Text with special characters: café, naïve, 日本語, emoji 🚀, symbols @#$%";

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Single(special_text.to_string()),
            None,
        )
        .await
        .expect("Failed to create embedding");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();
    validate_embedding_response(&embedding_response, 1);
}

#[tokio::test]
#[serial]
async fn test_batch_embeddings_small() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let texts = vec![
        "First text".to_string(),
        "Second text".to_string(),
        "Third text".to_string(),
    ];

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Batch(texts.clone()),
            None,
        )
        .await
        .expect("Failed to create embeddings");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();
    validate_embedding_response(&embedding_response, texts.len());
}

#[tokio::test]
#[serial]
async fn test_batch_embeddings_medium() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let texts = generate_test_texts(10);

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Batch(texts.clone()),
            None,
        )
        .await
        .expect("Failed to create embeddings");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();
    validate_embedding_response(&embedding_response, texts.len());
}

#[tokio::test]
#[serial]
async fn test_batch_embeddings_large() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 4)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let texts = generate_test_texts(50);

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Batch(texts.clone()),
            None,
        )
        .await
        .expect("Failed to create embeddings");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();
    validate_embedding_response(&embedding_response, texts.len());
}

#[tokio::test]
#[serial]
#[ignore = "This test is slow"]
async fn test_batch_embeddings_very_large() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 4)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let texts = generate_test_texts(100);

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Batch(texts.clone()),
            None,
        )
        .await
        .expect("Failed to create embeddings");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();
    validate_embedding_response(&embedding_response, texts.len());
}

#[tokio::test]
#[serial]
async fn test_batch_embeddings_mixed_lengths() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let texts = vec![
        "Short".to_string(),
        "This is a medium length text with more words".to_string(),
        "This is a much longer text that contains multiple sentences. It's designed to test how the embedding system handles various text lengths in a single batch. We want to ensure consistent processing.".to_string(),
        "Another short one".to_string(),
    ];

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Batch(texts.clone()),
            None,
        )
        .await
        .expect("Failed to create embeddings");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();
    validate_embedding_response(&embedding_response, texts.len());
}

#[tokio::test]
#[serial]
async fn test_batch_embeddings_duplicate_texts() {
    let model_path = get_test_model_path().expect("Test model not found");
    // Use n_seq_max=1 to ensure duplicate texts produce identical embeddings
    let server = TestServer::spawn_with_config(model_path, 2, Some(1))
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let texts = vec![
        "Duplicate text".to_string(),
        "Unique text".to_string(),
        "Duplicate text".to_string(), // Same as first
        "Another unique".to_string(),
        "Duplicate text".to_string(), // Same as first again
    ];

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Batch(texts.clone()),
            None,
        )
        .await
        .expect("Failed to create embeddings");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();
    validate_embedding_response(&embedding_response, texts.len());

    // Verify duplicate texts produce same embeddings
    if let EmbeddingValue::Float(emb1) = &embedding_response.data[0].embedding
        && let EmbeddingValue::Float(emb2) = &embedding_response.data[2].embedding
        && let EmbeddingValue::Float(emb3) = &embedding_response.data[4].embedding
    {
        // Check embeddings are identical for duplicate texts
        for i in 0..emb1.len() {
            assert!((emb1[i] - emb2[i]).abs() < 1e-6);
            assert!((emb1[i] - emb3[i]).abs() < 1e-6);
        }
    } else {
        panic!("Expected embeddings in Float format");
    }
}

#[tokio::test]
#[serial]
async fn test_encoding_format_float() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Single("Test text".to_string()),
            Some("float"),
        )
        .await
        .expect("Failed to create embedding");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();
    validate_embedding_response(&embedding_response, 1);

    // Verify it's float format
    match &embedding_response.data[0].embedding {
        EmbeddingValue::Float(_) => {}
        EmbeddingValue::Base64(_) => panic!("Expected float embedding format"),
    }
}

#[tokio::test]
#[serial]
async fn test_encoding_format_base64() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Single("Test text".to_string()),
            Some("base64"),
        )
        .await
        .expect("Failed to create embedding");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();
    validate_embedding_response(&embedding_response, 1);

    // Verify it's base64 format
    match &embedding_response.data[0].embedding {
        EmbeddingValue::Base64(_) => {}
        EmbeddingValue::Float(_) => panic!("Expected base64 embedding format"),
    }
}

#[tokio::test]
#[serial]
async fn test_error_empty_input() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Single(String::new()),
            None,
        )
        .await
        .expect("Failed to send request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let error: ErrorResponse = response.json().await.unwrap();
    assert!(error.error.message.contains("empty"));
}

#[tokio::test]
#[serial]
async fn test_error_empty_batch() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Batch(vec![]),
            None,
        )
        .await
        .expect("Failed to send request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let error: ErrorResponse = response.json().await.unwrap();
    assert!(error.error.message.contains("empty"));
}

#[tokio::test]
#[serial]
async fn test_error_invalid_encoding_format() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Single("Test".to_string()),
            Some("invalid_format"),
        )
        .await
        .expect("Failed to send request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let error: ErrorResponse = response.json().await.unwrap();
    assert!(error.error.message.contains("encoding_format"));
}

#[tokio::test]
#[serial]
async fn test_error_missing_model_field() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    // Send request without model field
    let response = client
        .client
        .post(format!("{}/v1/embeddings", server.base_url))
        .json(&serde_json::json!({
            "input": "Test text"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
#[serial]
async fn test_error_missing_input_field() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    // Send request without input field
    let response = client
        .client
        .post(format!("{}/v1/embeddings", server.base_url))
        .json(&serde_json::json!({
            "model": "test-model"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
#[serial]
async fn test_error_invalid_json() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let response = client
        .client
        .post(format!("{}/v1/embeddings", server.base_url))
        .header("Content-Type", "application/json")
        .body("{ invalid json }")
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
#[serial]
async fn test_usage_metrics_single() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Single("Test text for usage metrics".to_string()),
            None,
        )
        .await
        .expect("Failed to create embedding");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();

    // Verify usage metrics are present and reasonable
    assert!(embedding_response.usage.prompt_tokens > 0);
    assert!(embedding_response.usage.total_tokens > 0);
    assert_eq!(
        embedding_response.usage.prompt_tokens,
        embedding_response.usage.total_tokens
    );
}

#[tokio::test]
#[serial]
async fn test_usage_metrics_batch() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let texts = vec![
        "First text".to_string(),
        "Second text with more words".to_string(),
        "Third text that is even longer than the second one".to_string(),
    ];

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Batch(texts),
            None,
        )
        .await
        .expect("Failed to create embeddings");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();

    // Verify usage metrics increase with more text
    assert!(embedding_response.usage.prompt_tokens > 3); // More than number of texts
    assert!(embedding_response.usage.total_tokens > 3);
}
