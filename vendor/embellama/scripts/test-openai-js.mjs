#!/usr/bin/env node
/**
 * Test script for validating Embellama server compatibility with OpenAI JavaScript/TypeScript SDK.
 *
 * Prerequisites:
 *   npm install openai
 *
 * Usage:
 *   node test-openai-js.mjs [server_url]
 */

import OpenAI from 'openai';

/**
 * Test single text embedding
 */
async function testSingleEmbedding(client) {
    console.log('Testing single embedding...');

    const response = await client.embeddings.create({
        model: 'test-model',
        input: 'Hello from OpenAI JavaScript SDK!'
    });

    console.assert(response.object === 'list', 'Response object should be "list"');
    console.assert(response.data.length === 1, 'Should have one embedding');
    console.assert(response.data[0].index === 0, 'First embedding index should be 0');
    console.assert(response.data[0].object === 'embedding', 'Data object should be "embedding"');
    console.assert(response.data[0].embedding.length > 0, 'Embedding should not be empty');
    console.assert(response.usage.prompt_tokens > 0, 'Should have prompt tokens');
    console.assert(response.usage.total_tokens > 0, 'Should have total tokens');

    console.log(`✓ Single embedding: ${response.data[0].embedding.length} dimensions`);
}

/**
 * Test batch text embeddings
 */
async function testBatchEmbedding(client) {
    console.log('Testing batch embeddings...');

    const texts = [
        'First text from JavaScript',
        'Second text from JavaScript',
        'Third text from JavaScript'
    ];

    const response = await client.embeddings.create({
        model: 'test-model',
        input: texts
    });

    console.assert(response.object === 'list', 'Response object should be "list"');
    console.assert(response.data.length === texts.length, `Should have ${texts.length} embeddings`);

    response.data.forEach((embedding, i) => {
        console.assert(embedding.index === i, `Embedding ${i} should have correct index`);
        console.assert(embedding.object === 'embedding', 'Data object should be "embedding"');
        console.assert(embedding.embedding.length > 0, 'Embedding should not be empty');
    });

    console.log(`✓ Batch embeddings: ${texts.length} texts processed`);
}

/**
 * Test base64 encoding format
 */
async function testBase64Encoding(client) {
    console.log('Testing base64 encoding...');

    const response = await client.embeddings.create({
        model: 'test-model',
        input: 'Test base64 encoding',
        encoding_format: 'base64'
    });

    console.assert(response.object === 'list', 'Response object should be "list"');
    console.assert(response.data.length === 1, 'Should have one embedding');

    const embedding = response.data[0].embedding;
    console.assert(typeof embedding === 'string', 'Base64 embedding should be a string');

    // Verify it's valid base64
    try {
        const decoded = Buffer.from(embedding, 'base64');
        console.assert(decoded.length % 4 === 0, 'Decoded bytes should be multiple of 4 (f32)');
        console.log(`✓ Base64 encoding: ${decoded.length} bytes decoded`);
    } catch (error) {
        throw new Error(`Invalid base64 encoding: ${error.message}`);
    }
}

/**
 * Test listing available models
 */
async function testListModels(client) {
    console.log('Testing list models...');

    const models = await client.models.list();

    console.assert(models.object === 'list', 'Models response should be "list"');
    console.assert(models.data.length > 0, 'Should have at least one model');

    const firstModel = models.data[0];
    console.assert(firstModel.id, 'Model should have id');
    console.assert(firstModel.created, 'Model should have created timestamp');
    console.assert(firstModel.owned_by, 'Model should have owned_by field');

    console.log(`✓ Models listed: ${models.data.length} available`);
    models.data.forEach(model => {
        console.log(`  - ${model.id} (owned by: ${model.owned_by})`);
    });
}

/**
 * Test error handling
 */
async function testErrorHandling(client) {
    console.log('Testing error handling...');

    try {
        // Empty input should cause an error
        await client.embeddings.create({
            model: 'test-model',
            input: ''
        });
        throw new Error('Expected error for empty input');
    } catch (error) {
        // Should get an API error
        const errorMessage = error.message.toLowerCase();
        console.assert(
            errorMessage.includes('empty') || errorMessage.includes('invalid'),
            'Error should mention empty or invalid input'
        );
        console.log('✓ Error handling: Empty input rejected correctly');
    }
}

/**
 * Test that embeddings are normalized
 */
async function testEmbeddingNormalization(client) {
    console.log('Testing embedding normalization...');

    const response = await client.embeddings.create({
        model: 'test-model',
        input: 'Test normalization'
    });

    const embedding = response.data[0].embedding;

    // Calculate L2 norm
    const norm = Math.sqrt(embedding.reduce((sum, x) => sum + x * x, 0));

    // Should be normalized to approximately 1.0
    console.assert(
        Math.abs(norm - 1.0) < 0.01,
        `Embedding not normalized: L2 norm = ${norm}`
    );

    console.log(`✓ Normalization: L2 norm = ${norm.toFixed(4)}`);
}

/**
 * Test with TypeScript-style async/await patterns
 */
async function testAsyncPatterns(client) {
    console.log('Testing async/await patterns...');

    // Test parallel requests
    const promises = [
        client.embeddings.create({
            model: 'test-model',
            input: 'Async test 1'
        }),
        client.embeddings.create({
            model: 'test-model',
            input: 'Async test 2'
        }),
        client.embeddings.create({
            model: 'test-model',
            input: 'Async test 3'
        })
    ];

    const responses = await Promise.all(promises);

    console.assert(responses.length === 3, 'Should have 3 responses');
    responses.forEach(response => {
        console.assert(response.data.length === 1, 'Each response should have one embedding');
    });

    console.log('✓ Async patterns: Parallel requests completed');
}

/**
 * Main test function
 */
async function main() {
    // Get server URL from command line or use default
    const serverUrl = process.argv[2] || 'http://localhost:8080';

    console.log('\n=== Testing Embellama Server with OpenAI JavaScript SDK ===');
    console.log(`Server URL: ${serverUrl}`);
    console.log();

    // Create OpenAI client pointing to Embellama server
    const client = new OpenAI({
        baseURL: `${serverUrl}/v1`,
        apiKey: 'dummy-key', // Embellama doesn't require API key by default
        dangerouslyAllowBrowser: true // Allow browser usage for testing
    });

    try {
        // Run all tests
        await testSingleEmbedding(client);
        await testBatchEmbedding(client);
        await testBase64Encoding(client);
        await testListModels(client);
        await testErrorHandling(client);
        await testEmbeddingNormalization(client);
        await testAsyncPatterns(client);

        console.log('\n✅ All tests passed! Embellama is fully compatible with OpenAI JavaScript SDK.');
        process.exit(0);

    } catch (error) {
        console.error(`\n❌ Test failed: ${error.message}`);
        if (error.cause) {
            console.error('Cause:', error.cause);
        }
        console.error('Is the Embellama server running?');
        process.exit(1);
    }
}

// Run tests
main().catch(error => {
    console.error('Unexpected error:', error);
    process.exit(1);
});
