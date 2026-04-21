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

//! `OpenAI` compatibility tests for the Embellama server
//!
//! These tests validate that the API responses exactly match the `OpenAI`
//! embeddings API specification.

#![cfg(feature = "server")]

mod server_test_helpers;

use reqwest::StatusCode;
use serde_json::{Value, json};
use serial_test::serial;
use server_test_helpers::*;

#[tokio::test]
#[serial]
async fn test_openai_response_structure() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Single("OpenAI compatibility test".to_string()),
            None,
        )
        .await
        .expect("Failed to create embedding");

    assert_eq!(response.status(), StatusCode::OK);

    let body: Value = response.json().await.unwrap();

    // Validate exact response structure per OpenAI spec
    assert_eq!(body["object"], "list", "Top-level object must be 'list'");
    assert!(body["data"].is_array(), "'data' field must be an array");
    assert!(body["model"].is_string(), "'model' field must be a string");
    assert!(body["usage"].is_object(), "'usage' field must be an object");

    // Validate data array structure
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);

    let first_item = &data[0];
    assert_eq!(
        first_item["object"], "embedding",
        "Data item object must be 'embedding'"
    );
    assert_eq!(first_item["index"], 0, "Index must be 0 for first item");
    assert!(
        first_item["embedding"].is_array(),
        "Embedding must be an array"
    );

    // Validate usage structure
    let usage = &body["usage"];
    assert!(
        usage["prompt_tokens"].is_number(),
        "prompt_tokens must be a number"
    );
    assert!(
        usage["total_tokens"].is_number(),
        "total_tokens must be a number"
    );

    // Ensure no extra fields are present at top level
    let expected_fields = ["object", "data", "model", "usage"];
    for (key, _) in body.as_object().unwrap() {
        assert!(
            expected_fields.contains(&key.as_str()),
            "Unexpected field '{key}' in response"
        );
    }
}

#[tokio::test]
#[serial]
async fn test_openai_batch_response_structure() {
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

    let body: Value = response.json().await.unwrap();

    // Validate data array has correct number of items
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), texts.len());

    // Validate each item has correct index
    for (i, item) in data.iter().enumerate() {
        assert_eq!(item["index"], i, "Index must match position in array");
        assert_eq!(item["object"], "embedding");
        assert!(item["embedding"].is_array());
    }
}

#[tokio::test]
#[serial]
async fn test_openai_base64_encoding_format() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Single("Base64 encoding test".to_string()),
            Some("base64"),
        )
        .await
        .expect("Failed to create embedding");

    assert_eq!(response.status(), StatusCode::OK);

    let body: Value = response.json().await.unwrap();

    // Validate base64 encoding
    let embedding = &body["data"][0]["embedding"];
    assert!(embedding.is_string(), "Base64 embedding must be a string");

    // Verify it's valid base64
    let base64_str = embedding.as_str().unwrap();
    assert!(!base64_str.is_empty());

    // Decode to verify it's valid base64
    let decoded = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, base64_str);
    assert!(decoded.is_ok(), "Embedding must be valid base64");

    // Verify decoded data is reasonable size (multiple of 4 bytes for f32)
    let decoded_bytes = decoded.unwrap();
    assert_eq!(
        decoded_bytes.len() % 4,
        0,
        "Decoded bytes should be multiple of 4 (f32 size)"
    );
}

#[tokio::test]
#[serial]
async fn test_openai_error_response_format() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    // Test with empty input to trigger error
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

    let body: Value = response.json().await.unwrap();

    // Validate OpenAI error response structure
    assert!(
        body["error"].is_object(),
        "Error response must contain 'error' object"
    );

    let error = &body["error"];
    assert!(
        error["message"].is_string(),
        "Error must have 'message' field"
    );
    assert!(error["type"].is_string(), "Error must have 'type' field");

    // Code field is optional but if present must be string
    if error.get("code").is_some() {
        assert!(
            error["code"].is_string(),
            "Error 'code' must be string if present"
        );
    }
}

#[tokio::test]
#[serial]
async fn test_openai_models_list_format() {
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

    let body: Value = response.json().await.unwrap();

    // Validate models list structure per OpenAI spec
    assert_eq!(
        body["object"], "list",
        "Models response object must be 'list'"
    );
    assert!(body["data"].is_array(), "Models data must be an array");

    let models = body["data"].as_array().unwrap();
    assert!(!models.is_empty(), "At least one model should be available");

    // Validate model object structure
    for model in models {
        assert_eq!(
            model["object"], "model",
            "Each item must have object='model'"
        );
        assert!(model["id"].is_string(), "Model must have 'id' field");
        assert!(
            model["created"].is_number(),
            "Model must have 'created' timestamp"
        );
        assert!(
            model["owned_by"].is_string(),
            "Model must have 'owned_by' field"
        );
    }
}

#[tokio::test]
#[serial]
async fn test_openai_request_with_optional_fields() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    // Test with OpenAI optional fields
    let request_body = json!({
        "model": "test-model",
        "input": "Test with optional fields",
        "encoding_format": "float",
        "user": "test-user-123"  // Optional field per OpenAI spec
    });

    let response = client
        .client
        .post(format!("{}/v1/embeddings", server.base_url))
        .json(&request_body)
        .send()
        .await
        .expect("Failed to send request");

    // Should accept optional fields without error
    assert_eq!(response.status(), StatusCode::OK);

    let body: Value = response.json().await.unwrap();
    assert_eq!(body["object"], "list");
}

#[tokio::test]
#[serial]
async fn test_openai_input_type_flexibility() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    // Test 1: String input (OpenAI supports both string and array)
    let response1 = client
        .client
        .post(format!("{}/v1/embeddings", server.base_url))
        .json(&json!({
            "model": "test-model",
            "input": "Single string input"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(response1.status(), StatusCode::OK);
    let body1: Value = response1.json().await.unwrap();
    assert_eq!(body1["data"].as_array().unwrap().len(), 1);

    // Test 2: Array with single string
    let response2 = client
        .client
        .post(format!("{}/v1/embeddings", server.base_url))
        .json(&json!({
            "model": "test-model",
            "input": ["Single string in array"]
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(response2.status(), StatusCode::OK);
    let body2: Value = response2.json().await.unwrap();
    assert_eq!(body2["data"].as_array().unwrap().len(), 1);

    // Test 3: Array with multiple strings
    let response3 = client
        .client
        .post(format!("{}/v1/embeddings", server.base_url))
        .json(&json!({
            "model": "test-model",
            "input": ["First", "Second", "Third"]
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(response3.status(), StatusCode::OK);
    let body3: Value = response3.json().await.unwrap();
    assert_eq!(body3["data"].as_array().unwrap().len(), 3);
}

#[tokio::test]
#[serial]
async fn test_openai_embedding_dimensions() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Single("Check embedding dimensions".to_string()),
            None,
        )
        .await
        .expect("Failed to create embedding");

    assert_eq!(response.status(), StatusCode::OK);

    let body: Value = response.json().await.unwrap();
    let embedding = body["data"][0]["embedding"].as_array().unwrap();

    // OpenAI embeddings are typically 1536 or 3072 dimensions
    // Our test model will have its own dimension count
    assert!(!embedding.is_empty(), "Embedding should not be empty");
    assert!(
        embedding.len() >= 64,
        "Embedding should have reasonable dimensions"
    );

    // All values should be valid floats
    for value in embedding {
        assert!(value.is_number(), "All embedding values must be numbers");
        let num = value.as_f64().unwrap();
        assert!(num.is_finite(), "All embedding values must be finite");
    }
}

#[tokio::test]
#[serial]
async fn test_openai_normalized_embeddings() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    let response = client
        .embedding_request(
            &server.base_url,
            "test-model",
            EmbeddingInput::Single("Test normalization".to_string()),
            None,
        )
        .await
        .expect("Failed to create embedding");

    assert_eq!(response.status(), StatusCode::OK);

    let embedding_response: EmbeddingResponse = response.json().await.unwrap();

    if let EmbeddingValue::Float(embedding) = &embedding_response.data[0].embedding {
        // Calculate L2 norm
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        // OpenAI embeddings are typically normalized to L2 norm = 1
        // Allow some tolerance for floating point precision
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embedding should be normalized. L2 norm: {norm}"
        );
    }
}

#[tokio::test]
#[serial]
async fn test_openai_consistent_model_name() {
    let model_path = get_test_model_path().expect("Test model not found");
    let server = TestServer::spawn(model_path, 2)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new();

    // Get model from models list
    let models_response = client
        .list_models(&server.base_url)
        .await
        .expect("Failed to list models");

    let models_body: Value = models_response.json().await.unwrap();
    let model_id = models_body["data"][0]["id"].as_str().unwrap();

    // Use same model in embeddings request
    let embedding_response = client
        .embedding_request(
            &server.base_url,
            model_id,
            EmbeddingInput::Single("Test model consistency".to_string()),
            None,
        )
        .await
        .expect("Failed to create embedding");

    let embedding_body: Value = embedding_response.json().await.unwrap();

    // Model in response should match requested model
    assert_eq!(
        embedding_body["model"].as_str().unwrap(),
        model_id,
        "Model in response should match requested model"
    );
}
