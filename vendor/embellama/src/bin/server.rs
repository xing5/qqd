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

use clap::Parser;
use embellama::server::ServerConfig;
use embellama::{EngineConfig, ModelConfig, NormalizationMode, PoolingStrategy};
use std::path::PathBuf;
use tracing::info;

/// Embellama server - OpenAI-compatible embedding API
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(long, env = "EMBELLAMA_MODEL_PATH")]
    model_path: PathBuf,

    /// Model identifier for API responses
    #[arg(long, env = "EMBELLAMA_MODEL_NAME", default_value = "default")]
    model_name: String,

    /// Bind address
    #[arg(long, env = "EMBELLAMA_HOST", default_value = "127.0.0.1")]
    host: String,

    /// Server port
    #[arg(short, long, env = "EMBELLAMA_PORT", default_value_t = 8080)]
    port: u16,

    /// Number of worker threads
    #[arg(long, env = "EMBELLAMA_WORKERS", default_value_t = num_cpus::get())]
    workers: usize,

    /// Maximum pending requests per worker
    #[arg(long, env = "EMBELLAMA_QUEUE_SIZE", default_value_t = 100)]
    queue_size: usize,

    /// Request timeout in seconds
    #[arg(long, env = "EMBELLAMA_REQUEST_TIMEOUT", default_value_t = 60)]
    request_timeout: u64,

    /// Maximum number of sequences to process in parallel (`n_seq_max`)
    #[arg(long, env = "EMBELLAMA_N_SEQ_MAX", default_value_t = 8)]
    n_seq_max: u32,

    /// Pooling strategy for embeddings (mean, cls, max, mean-sqrt, last)
    #[arg(long, env = "EMBELLAMA_POOLING_STRATEGY")]
    pooling_strategy: Option<String>,

    /// Normalization mode for embeddings (none, l2, max-abs, p-norm)
    #[arg(long, env = "EMBELLAMA_NORMALIZATION")]
    normalization_mode: Option<String>,

    /// P-norm exponent (only used if normalization = p-norm)
    #[arg(long, env = "EMBELLAMA_PNORM_EXPONENT", default_value_t = 2)]
    pnorm_exponent: i32,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, env = "EMBELLAMA_LOG_LEVEL", default_value = "info")]
    log_level: String,
}

/// Parse pooling strategy from string
fn parse_pooling_strategy(s: &str) -> Result<PoolingStrategy, String> {
    match s.to_lowercase().as_str() {
        "mean" => Ok(PoolingStrategy::Mean),
        "cls" => Ok(PoolingStrategy::Cls),
        "max" => Ok(PoolingStrategy::Max),
        "mean-sqrt" | "meansqrt" => Ok(PoolingStrategy::MeanSqrt),
        "last" => Ok(PoolingStrategy::Last),
        "none" => Ok(PoolingStrategy::None),
        "rank" => Ok(PoolingStrategy::Rank),
        _ => Err(format!(
            "Invalid pooling strategy '{s}'. Valid options: mean, cls, max, mean-sqrt, last, none, rank"
        )),
    }
}

/// Parse normalization mode from string
fn parse_normalization_mode(s: &str, pnorm_exponent: i32) -> Result<NormalizationMode, String> {
    match s.to_lowercase().as_str() {
        "none" => Ok(NormalizationMode::None),
        "l2" => Ok(NormalizationMode::L2),
        "max-abs" | "maxabs" => Ok(NormalizationMode::MaxAbs),
        "p-norm" | "pnorm" => {
            if pnorm_exponent <= 0 {
                Err(format!(
                    "P-norm exponent must be positive, got {pnorm_exponent}"
                ))
            } else {
                Ok(NormalizationMode::PNorm(pnorm_exponent))
            }
        }
        _ => Err(format!(
            "Invalid normalization mode '{s}'. Valid options: none, l2, max-abs, p-norm"
        )),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize logging
    init_logging(&args.log_level);

    info!("Starting Embellama server v{}", env!("CARGO_PKG_VERSION"));
    info!("Model: {:?} ({})", args.model_path, args.model_name);
    info!("Workers: {}, Queue size: {}", args.workers, args.queue_size);
    info!(
        "Request timeout: {}s, n_seq_max: {}",
        args.request_timeout, args.n_seq_max
    );
    info!(
        "Pooling: {}, Normalization: {}",
        args.pooling_strategy.as_deref().unwrap_or("auto"),
        args.normalization_mode.as_deref().unwrap_or("auto")
    );

    // Parse pooling strategy if provided
    let pooling_strategy = if let Some(ref strategy_str) = args.pooling_strategy {
        Some(parse_pooling_strategy(strategy_str)?)
    } else {
        None
    };

    // Parse normalization mode if provided
    let normalization_mode = if let Some(ref norm_str) = args.normalization_mode {
        Some(parse_normalization_mode(norm_str, args.pnorm_exponent)?)
    } else {
        None
    };

    // Build model configuration first
    let mut model_config_builder = ModelConfig::builder()
        .with_model_path(args.model_path.to_string_lossy().to_string())
        .with_model_name(&args.model_name)
        .with_n_seq_max(args.n_seq_max);

    // Add pooling strategy if specified
    if let Some(strategy) = pooling_strategy {
        model_config_builder = model_config_builder.with_pooling_strategy(strategy);
    }

    // Add normalization mode if specified
    if let Some(mode) = normalization_mode {
        model_config_builder = model_config_builder.with_normalization_mode(mode);
    }

    let model_config = model_config_builder.build()?;

    // Build engine configuration
    let engine_config = EngineConfig::builder()
        .with_model_config(model_config)
        .build()?;

    // Create server configuration using the library API
    let config = ServerConfig::builder()
        .engine_config(engine_config)
        .worker_count(args.workers)
        .queue_size(args.queue_size)
        .host(args.host)
        .port(args.port)
        .request_timeout(std::time::Duration::from_secs(args.request_timeout))
        .build()?;

    // Use the library's run_server function
    embellama::server::run_server(config).await?;

    Ok(())
}

/// Initialize logging with the specified level
fn init_logging(level: &str) {
    use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_target(true)
                .with_thread_ids(true)
                .with_thread_names(true)
                .with_file(true)
                .with_line_number(true),
        )
        .with(env_filter)
        .init();
}
