#!/bin/bash

# Test script for Embellama server OpenAI-compatible API
# Usage: ./test_api.sh [host:port]

SERVER="${1:-http://localhost:8080}"

echo "Testing Embellama API at $SERVER"
echo "================================"
echo

# Test health endpoint
echo "1. Testing /health endpoint:"
curl -s "$SERVER/health" | jq .
echo

# Test models endpoint
echo "2. Testing /v1/models endpoint:"
curl -s "$SERVER/v1/models" | jq .
echo

# Test embeddings with single text
echo "3. Testing /v1/embeddings with single text:"
curl -s -X POST "$SERVER/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": "Hello, world!"
  }' | jq '.object, .model, .usage, .data[0].index'
echo

# Test embeddings with batch
echo "4. Testing /v1/embeddings with batch:"
curl -s -X POST "$SERVER/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": ["Hello", "World", "Test"]
  }' | jq '.object, .model, (.data | length)'
echo

# Test base64 encoding
echo "5. Testing /v1/embeddings with base64 encoding:"
curl -s -X POST "$SERVER/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": "Test base64",
    "encoding_format": "base64"
  }' | jq '.data[0].embedding | type'
echo

# Test error handling - empty input
echo "6. Testing error handling - empty input:"
curl -s -X POST "$SERVER/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": ""
  }' | jq .
echo

# Test error handling - invalid encoding format
echo "7. Testing error handling - invalid encoding format:"
curl -s -X POST "$SERVER/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": "test",
    "encoding_format": "invalid"
  }' | jq .
echo

echo "================================"
echo "API tests complete!"
echo
echo "Note: To run the server with a model file:"
echo "  cargo run --features server --bin embellama-server -- --model-path /path/to/model.gguf"
