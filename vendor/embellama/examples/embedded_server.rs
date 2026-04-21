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

//! Example of using Embellama server as an embedded library
//!
//! This example demonstrates how to:
//! - Configure the server using the builder pattern
//! - Create a custom router with additional routes
//! - Add custom middleware
//! - Integrate with an existing Axum application

#[cfg(feature = "server")]
use axum::{
    Router,
    response::{IntoResponse, Json},
    routing::get,
};
#[cfg(feature = "server")]
use embellama::server::{
    AppState, EngineConfig, ModelConfig, ServerConfig, create_router, inject_request_id,
    limit_request_size,
};
#[cfg(feature = "server")]
use serde_json::json;
#[cfg(feature = "server")]
use std::env;
#[cfg(feature = "server")]
use tower::ServiceBuilder;

#[cfg(feature = "server")]
async fn custom_info_handler() -> impl IntoResponse {
    Json(json!({
        "name": "Embedded Embellama Server",
        "description": "This is a custom endpoint added to the embedded server",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

#[cfg(feature = "server")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    embellama::init_with_env_filter("info,embellama=debug");

    // Get model path from environment or command line
    let model_path = env::args()
        .nth(1)
        .or_else(|| env::var("EMBELLAMA_MODEL_PATH").ok())
        .expect("Please provide model path as first argument or set EMBELLAMA_MODEL_PATH");

    println!("Starting embedded server example with model: {model_path}");

    // Example 1: Simple usage with run_server
    example_simple_server(&model_path).await?;

    // Example 2: Custom router with additional routes and middleware
    example_custom_router(&model_path).await?;

    // Example 3: Integration with existing application
    example_integrated_app(&model_path).await?;

    Ok(())
}

#[cfg(feature = "server")]
async fn example_simple_server(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Example 1: Simple Server ===");
    println!("This uses the convenient run_server function");

    // Build engine configuration first
    let engine_config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("embedded-model")
        .build()?;

    // Build server configuration
    let config = ServerConfig::builder()
        .engine_config(engine_config)
        .host("127.0.0.1")
        .port(8081)
        .worker_count(2)
        .queue_size(50)
        .build()?;

    // Start server (this would normally block)
    // For demo purposes, we'll just show the configuration
    println!("Server would run with config: {config:?}");

    // Uncomment to actually run:
    // run_server(config).await?;

    Ok(())
}

#[cfg(feature = "server")]
async fn example_custom_router(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Example 2: Custom Router ===");
    println!("This creates a custom router with additional routes");

    // Build engine configuration
    let engine_config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("custom-model")
        .build()?;

    // Create server configuration
    let config = ServerConfig::builder()
        .engine_config(engine_config)
        .port(8082)
        .build()?;

    // Create application state
    let state = AppState::new(config)?;

    // Create the base router
    let mut app = create_router(state.clone());

    // Add custom routes - note these need to match the state type
    app = app
        .route("/custom/info", get(custom_info_handler))
        .route(
            "/custom/echo",
            axum::routing::post(|body: String| async move {
                Json(json!({
                    "echo": body,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                }))
            }),
        )
        // Add a route group with shared middleware
        .nest(
            "/api",
            Router::new()
                .route("/status", get(|| async { "API is running" }))
                .route("/version", get(|| async { env!("CARGO_PKG_VERSION") })),
        );

    // Add custom middleware to the entire application
    app = app.layer(
        ServiceBuilder::new()
            // Add request ID injection
            .layer(axum::middleware::from_fn(inject_request_id))
            // Add request size limiting
            .layer(axum::middleware::from_fn(limit_request_size)),
    );

    println!("Custom router created with additional routes:");
    println!("  - /custom/info (GET)");
    println!("  - /custom/echo (POST)");
    println!("  - /api/status (GET)");
    println!("  - /api/version (GET)");
    println!("  Plus standard embedding routes at /v1/*");

    // Start the server (commented out for demo)
    // let listener = tokio::net::TcpListener::bind("127.0.0.1:8082").await?;
    // axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(feature = "server")]
async fn example_integrated_app(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Example 3: Integrated Application ===");
    println!("This shows how to integrate embedding server into an existing app");

    // Your existing application might have its own state
    #[derive(Clone)]
    struct MyAppState {
        embedding_state: AppState,
        custom_data: String,
    }

    // Create engine configuration
    let engine_config = EngineConfig::builder()
        .with_model_path(model_path)
        .with_model_name("integrated-model")
        .build()?;

    // Create embedding server configuration
    let config = ServerConfig::builder()
        .engine_config(engine_config)
        .build()?;

    // Create embedding state
    let embedding_state = AppState::new(config)?;

    // Create your application state
    let app_state = MyAppState {
        embedding_state: embedding_state.clone(),
        custom_data: "Some application data".to_string(),
    };

    // Create your main application router
    let _app = Router::new()
        // Your existing application routes
        .route("/", get(|| async { "Welcome to my application!" }))
        .route(
            "/app/data",
            get({
                let state = app_state.clone();
                move || async move { state.custom_data.clone() }
            }),
        )
        // Mount the embedding server at a subpath
        .nest("/embeddings", create_router(embedding_state));

    println!("Integrated application created with:");
    println!("  - Main app routes at /");
    println!("  - Embedding API at /embeddings/v1/*");

    // Server would start here
    // let listener = tokio::net::TcpListener::bind("127.0.0.1:8083").await?;
    // axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(not(feature = "server"))]
fn main() {
    eprintln!("This example requires the 'server' feature to be enabled.");
    eprintln!("Run with: cargo run --example embedded_server --features server");
    std::process::exit(1);
}
