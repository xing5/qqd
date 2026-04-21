use anyhow::{Context, Result, anyhow};
use embellama::{
    EmbeddingEngine, EngineConfig, NormalizationMode, PoolingStrategy, TruncateTokens,
};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

const EMBED_MODEL_NAME: &str = "qqd-embed";
const RERANK_MODEL_NAME: &str = "qqd-rerank";
const DETERMINISTIC_BACKEND_ID: &str = "qqd-deterministic-v1";
const DEFAULT_EMBED_MODEL_HINTS: &[&str] = &[
    "hf_ggml-org_embeddinggemma-300M-Q8_0.gguf",
    "embeddinggemma",
];
const DEFAULT_RERANK_MODEL_HINTS: &[&str] =
    &["hf_ggml-org_qwen3-reranker-0.6b-q8_0.gguf", "reranker"];

static MODEL_RUNTIME: OnceLock<Result<Option<Mutex<LocalModelRuntime>>, String>> = OnceLock::new();

pub struct LocalModelRuntime {
    engine: EmbeddingEngine,
    embed_model_name: Option<String>,
    rerank_model_name: Option<String>,
    embed_backend_id: String,
}

pub struct RerankHit {
    pub index: usize,
    pub relevance_score: f32,
}

impl LocalModelRuntime {
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let model = self
            .embed_model_name
            .as_deref()
            .ok_or_else(|| anyhow!("embed model is not configured"))?;
        self.engine
            .embed(Some(model), text)
            .context("failed to generate GGUF embedding")
    }

    pub fn rerank<'a>(&self, query: &str, documents: &'a [&'a str]) -> Result<Vec<RerankHit>> {
        let model = self
            .rerank_model_name
            .as_deref()
            .ok_or_else(|| anyhow!("rerank model is not configured"))?;
        let mut hits = Vec::with_capacity(documents.len());
        for (index, document) in documents.iter().enumerate() {
            let result = self
                .engine
                .rerank(Some(model), query, &[*document], Some(1), true)
                .context("failed to rerank candidate document")?;
            let score = result.first().map(|hit| hit.relevance_score).unwrap_or(0.0);
            hits.push(RerankHit {
                index,
                relevance_score: score,
            });
        }
        hits.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(hits)
    }

    pub fn has_embedder(&self) -> bool {
        self.embed_model_name.is_some()
    }

    pub fn has_reranker(&self) -> bool {
        self.rerank_model_name.is_some()
    }

    pub fn embed_backend_id(&self) -> &str {
        &self.embed_backend_id
    }
}

pub fn deterministic_backend_id() -> &'static str {
    DETERMINISTIC_BACKEND_ID
}

pub fn current_embed_backend_id() -> Result<String> {
    Ok(match runtime()? {
        Some(runtime) => embed_backend_id_for_runtime(Some(&*runtime)),
        _ => embed_backend_id_for_runtime(None),
    })
}

pub fn embed_backend_id_for_runtime(runtime: Option<&LocalModelRuntime>) -> String {
    match runtime {
        Some(runtime) if runtime.has_embedder() => runtime.embed_backend_id().to_string(),
        _ => DETERMINISTIC_BACKEND_ID.to_string(),
    }
}

pub fn runtime() -> Result<Option<MutexGuard<'static, LocalModelRuntime>>> {
    let state = MODEL_RUNTIME.get_or_init(|| {
        init_runtime()
            .map(|runtime| runtime.map(Mutex::new))
            .map_err(|error| format!("{error:#}"))
    });
    match state {
        Ok(Some(runtime)) => runtime
            .lock()
            .map(Some)
            .map_err(|_| anyhow!("model runtime mutex was poisoned")),
        Ok(None) => Ok(None),
        Err(message) => Err(anyhow!(message.clone())),
    }
}

fn init_runtime() -> Result<Option<LocalModelRuntime>> {
    let embed_model_path = resolve_model_path("QQD_EMBED_MODEL", DEFAULT_EMBED_MODEL_HINTS);
    let rerank_model_path = resolve_model_path("QQD_RERANK_MODEL", DEFAULT_RERANK_MODEL_HINTS);
    if embed_model_path.is_none() && rerank_model_path.is_none() {
        return Ok(None);
    }

    embellama::init();

    let mut engine = if let Some(path) = &embed_model_path {
        EmbeddingEngine::new(build_engine_config(path, EMBED_MODEL_NAME, false)?)
            .with_context(|| format!("failed to load QQD embed model at {}", path.display()))?
    } else if let Some(path) = &rerank_model_path {
        EmbeddingEngine::new(build_engine_config(path, RERANK_MODEL_NAME, true)?)
            .with_context(|| format!("failed to load QQD rerank model at {}", path.display()))?
    } else {
        return Ok(None);
    };

    if let Some(path) = &rerank_model_path {
        engine
            .load_model(build_engine_config(path, RERANK_MODEL_NAME, true)?)
            .with_context(|| format!("failed to load QQD rerank model at {}", path.display()))?;
    }

    let embed_backend_id = embed_model_path
        .as_ref()
        .map(|path| format!("gguf:{}", canonicalish(path)))
        .unwrap_or_else(|| DETERMINISTIC_BACKEND_ID.to_string());

    Ok(Some(LocalModelRuntime {
        engine,
        embed_model_name: embed_model_path.map(|_| EMBED_MODEL_NAME.to_string()),
        rerank_model_name: rerank_model_path.map(|_| RERANK_MODEL_NAME.to_string()),
        embed_backend_id,
    }))
}

pub fn discover_embed_model_path() -> Option<PathBuf> {
    resolve_model_path("QQD_EMBED_MODEL", DEFAULT_EMBED_MODEL_HINTS)
}

pub fn discover_rerank_model_path() -> Option<PathBuf> {
    resolve_model_path("QQD_RERANK_MODEL", DEFAULT_RERANK_MODEL_HINTS)
}

fn resolve_model_path(env_name: &str, hints: &[&str]) -> Option<PathBuf> {
    env::var(env_name).ok().map(PathBuf::from).or_else(|| {
        if autodiscovery_disabled() {
            None
        } else {
            discover_default_model(hints)
        }
    })
}

fn discover_default_model(hints: &[&str]) -> Option<PathBuf> {
    let models_dir = default_models_dir();
    discover_default_model_in(&models_dir, hints)
}

fn discover_default_model_in(models_dir: &Path, hints: &[&str]) -> Option<PathBuf> {
    if !models_dir.is_dir() {
        return None;
    }
    for hint in hints {
        let exact = models_dir.join(hint);
        if exact.exists() {
            return Some(exact);
        }
    }
    let entries = std::fs::read_dir(models_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name()?.to_string_lossy().to_ascii_lowercase();
        if path.extension().and_then(|value| value.to_str()) != Some("gguf") {
            continue;
        }
        if hints
            .iter()
            .any(|hint| name.contains(&hint.to_ascii_lowercase()))
        {
            return Some(path);
        }
    }
    None
}

fn default_models_dir() -> PathBuf {
    let cache_root = env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(
                env::var("HOME")
                    .map(|home| format!("{home}/.cache"))
                    .unwrap_or_else(|_| ".cache".to_string()),
            )
        });
    cache_root.join("qmd").join("models")
}

fn autodiscovery_disabled() -> bool {
    env::var("QQD_DISABLE_MODEL_AUTODISCOVERY")
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

fn build_engine_config(path: &PathBuf, model_name: &str, reranker: bool) -> Result<EngineConfig> {
    let mut builder = EngineConfig::builder()
        .with_model_path(path)
        .with_model_name(model_name)
        .with_n_threads(
            env::var("QQD_MODEL_THREADS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or_else(default_threads),
        )
        .with_context_size(if reranker { 4096 } else { 2048 })
        .with_truncate_tokens(TruncateTokens::Yes);
    if reranker {
        builder = builder
            .with_n_seq_max(16)
            .with_pooling_strategy(PoolingStrategy::Rank)
            .with_normalization_mode(NormalizationMode::None);
    } else {
        builder = builder
            .with_n_seq_max(1)
            .with_pooling_strategy(PoolingStrategy::Mean)
            .with_normalization_mode(NormalizationMode::L2);
    }
    builder
        .build()
        .with_context(|| format!("failed to build model config for {}", path.display()))
}

fn canonicalish(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_default_models_in_qmd_cache_dir() {
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join(".cache/qmd/models");
        std::fs::create_dir_all(&models).expect("models dir");
        let embed = models.join("hf_ggml-org_embeddinggemma-300M-Q8_0.gguf");
        let rerank = models.join("hf_ggml-org_qwen3-reranker-0.6b-q8_0.gguf");
        std::fs::write(&embed, b"stub").expect("embed stub");
        std::fs::write(&rerank, b"stub").expect("rerank stub");

        assert_eq!(
            discover_default_model_in(&models, DEFAULT_EMBED_MODEL_HINTS).as_deref(),
            Some(embed.as_path())
        );
        assert_eq!(
            discover_default_model_in(&models, DEFAULT_RERANK_MODEL_HINTS).as_deref(),
            Some(rerank.as_path())
        );
    }
}
