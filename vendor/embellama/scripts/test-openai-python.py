#!/usr/bin/env python3
"""
Test script for validating Embellama server compatibility with OpenAI Python SDK.

Prerequisites:
    pip install openai

Usage:
    python test-openai-python.py [server_url]
"""

import sys
import json
from typing import List
import base64

# Try to import openai
try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI Python SDK not installed.")
    print("Please install it with: pip install openai")
    sys.exit(1)


def test_single_embedding(client: OpenAI) -> None:
    """Test single text embedding."""
    print("Testing single embedding...")

    response = client.embeddings.create(
        model="test-model",
        input="Hello from OpenAI Python SDK!"
    )

    assert response.object == "list"
    assert len(response.data) == 1
    assert response.data[0].index == 0
    assert response.data[0].object == "embedding"
    assert len(response.data[0].embedding) > 0
    assert response.usage.prompt_tokens > 0
    assert response.usage.total_tokens > 0

    print(f"✓ Single embedding: {len(response.data[0].embedding)} dimensions")


def test_batch_embedding(client: OpenAI) -> None:
    """Test batch text embeddings."""
    print("Testing batch embeddings...")

    texts = [
        "First text from Python",
        "Second text from Python",
        "Third text from Python"
    ]

    response = client.embeddings.create(
        model="test-model",
        input=texts
    )

    assert response.object == "list"
    assert len(response.data) == len(texts)

    for i, embedding_data in enumerate(response.data):
        assert embedding_data.index == i
        assert embedding_data.object == "embedding"
        assert len(embedding_data.embedding) > 0

    print(f"✓ Batch embeddings: {len(texts)} texts processed")


def test_base64_encoding(client: OpenAI) -> None:
    """Test base64 encoding format."""
    print("Testing base64 encoding...")

    response = client.embeddings.create(
        model="test-model",
        input="Test base64 encoding",
        encoding_format="base64"
    )

    assert response.object == "list"
    assert len(response.data) == 1

    # In base64 format, embedding should be a string
    embedding = response.data[0].embedding
    assert isinstance(embedding, str)

    # Verify it's valid base64
    try:
        decoded = base64.b64decode(embedding)
        # Should decode to bytes with length multiple of 4 (f32)
        assert len(decoded) % 4 == 0
        print(f"✓ Base64 encoding: {len(decoded)} bytes decoded")
    except Exception as e:
        raise AssertionError(f"Invalid base64 encoding: {e}")


def test_list_models(client: OpenAI) -> None:
    """Test listing available models."""
    print("Testing list models...")

    models = client.models.list()

    model_list = list(models)
    assert len(model_list) > 0

    first_model = model_list[0]
    assert hasattr(first_model, 'id')
    assert hasattr(first_model, 'created')
    assert hasattr(first_model, 'owned_by')

    print(f"✓ Models listed: {len(model_list)} available")
    for model in model_list:
        print(f"  - {model.id} (owned by: {model.owned_by})")


def test_error_handling(client: OpenAI) -> None:
    """Test error handling."""
    print("Testing error handling...")

    try:
        # Empty input should cause an error
        response = client.embeddings.create(
            model="test-model",
            input=""
        )
        raise AssertionError("Expected error for empty input")
    except Exception as e:
        # Should get an API error
        assert "empty" in str(e).lower() or "invalid" in str(e).lower()
        print("✓ Error handling: Empty input rejected correctly")


def test_embedding_normalization(client: OpenAI) -> None:
    """Test that embeddings are normalized."""
    print("Testing embedding normalization...")

    response = client.embeddings.create(
        model="test-model",
        input="Test normalization"
    )

    embedding = response.data[0].embedding

    # Calculate L2 norm
    norm = sum(x * x for x in embedding) ** 0.5

    # Should be normalized to approximately 1.0
    assert abs(norm - 1.0) < 0.01, f"Embedding not normalized: L2 norm = {norm}"

    print(f"✓ Normalization: L2 norm = {norm:.4f}")


def main():
    """Main test function."""
    # Get server URL from command line or use default
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

    print(f"\n=== Testing Embellama Server with OpenAI Python SDK ===")
    print(f"Server URL: {server_url}")
    print()

    # Create OpenAI client pointing to Embellama server
    client = OpenAI(
        base_url=f"{server_url}/v1",
        api_key="dummy-key"  # Embellama doesn't require API key by default
    )

    try:
        # Run all tests
        test_single_embedding(client)
        test_batch_embedding(client)
        test_base64_encoding(client)
        test_list_models(client)
        test_error_handling(client)
        test_embedding_normalization(client)

        print("\n✅ All tests passed! Embellama is fully compatible with OpenAI Python SDK.")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Is the Embellama server running?")
        sys.exit(1)


if __name__ == "__main__":
    main()
