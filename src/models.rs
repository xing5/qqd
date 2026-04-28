use anyhow::{Context, Result, anyhow};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, VocabType};
use serde_json::json;
use std::env;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

const DETERMINISTIC_BACKEND_ID: &str = "qqd-deterministic-v1";
const DEFAULT_EMBED_MODEL_HINTS: &[&str] = &[
    "hf_ggml-org_embeddinggemma-300M-Q8_0.gguf",
    "embeddinggemma",
];
const DEFAULT_RERANK_MODEL_HINTS: &[&str] =
    &["hf_ggml-org_qwen3-reranker-0.6b-q8_0.gguf", "reranker"];
const DEFAULT_EMBED_CONTEXT_SIZE: u32 = 2048;
const QMD_LIKE_EMBED_BATCH_SIZE: usize = 512;

static MODEL_RUNTIME: OnceLock<Result<Option<Mutex<LocalModelRuntime>>, String>> = OnceLock::new();

pub struct LocalModelRuntime {
    backend: Arc<LlamaBackend>,
    embed_model: Option<DirectEmbedModel>,
    rerank_model: Option<DirectRerankModel>,
    embed_backend_id: String,
}

struct DirectEmbedModel {
    model: LlamaModel,
    context_size: u32,
}

struct DirectRerankModel {
    model: LlamaModel,
    context_size: u32,
    n_threads: i32,
}

pub struct RerankHit {
    pub index: usize,
    pub relevance_score: f32,
}

type IndexedEmbedding = (usize, Vec<f32>);
type EmbedWorkerOutput = (Vec<IndexedEmbedding>, EmbedRuntimeTrace);

impl DirectEmbedModel {
    fn new(backend: &LlamaBackend, path: &Path) -> Result<Self> {
        let model = LlamaModel::load_from_file(backend, path, &LlamaModelParams::default())
            .with_context(|| format!("failed to load QQD embed model at {}", path.display()))?;
        Ok(Self {
            model,
            context_size: DEFAULT_EMBED_CONTEXT_SIZE,
        })
    }

    fn tokenize_texts(&self, texts: &[&str]) -> Result<Vec<Vec<llama_cpp_2::token::LlamaToken>>> {
        texts
            .iter()
            .map(|text| self.prepare_embedding_tokens(text))
            .collect()
    }

    fn tokenize_text(
        &self,
        text: &str,
        add_bos: AddBos,
    ) -> Result<Vec<llama_cpp_2::token::LlamaToken>> {
        self.model
            .str_to_token(text, add_bos)
            .with_context(|| "failed to tokenize embedding text")
    }

    fn prepare_embedding_tokens(&self, text: &str) -> Result<Vec<llama_cpp_2::token::LlamaToken>> {
        let mut tokens = self.tokenize_text(text, AddBos::Never)?;
        if matches!(self.model.vocab_type(), VocabType::SPM) {
            let bos = self.model.token_bos();
            if tokens.first().copied() != Some(bos) {
                tokens.insert(0, bos);
            }
            let eos = self.model.token_eos();
            if tokens.last().copied() != Some(eos) {
                tokens.push(eos);
            }
        }
        Ok(tokens)
    }

    fn detokenize_tokens(&self, tokens: &[llama_cpp_2::token::LlamaToken]) -> Result<String> {
        #[allow(deprecated)]
        self.model
            .tokens_to_str(tokens, llama_cpp_2::model::Special::Plaintext)
            .with_context(|| "failed to detokenize embedding tokens")
    }
}

impl DirectRerankModel {
    fn new(backend: &LlamaBackend, path: &Path) -> Result<Self> {
        let model = LlamaModel::load_from_file(backend, path, &LlamaModelParams::default())
            .with_context(|| format!("failed to load QQD rerank model at {}", path.display()))?;
        Ok(Self {
            model,
            context_size: 4096,
            n_threads: default_threads(),
        })
    }

    fn score_document(&self, backend: &LlamaBackend, query: &str, document: &str) -> Result<f32> {
        if query.is_empty() {
            return Err(anyhow!("rerank query cannot be empty"));
        }
        if document.is_empty() {
            return Err(anyhow!("rerank document cannot be empty"));
        }

        let combined = format!("{query}\n\n{document}");
        let mut tokens = self
            .model
            .str_to_token(&combined, AddBos::Always)
            .with_context(|| "failed to tokenize rerank input")?;
        let max_tokens = self.context_size as usize;
        if tokens.len() > max_tokens {
            tokens.truncate(max_tokens);
        }

        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.context_size))
            .with_n_batch(tokens.len() as u32)
            .with_n_ubatch(tokens.len() as u32)
            .with_n_threads(self.n_threads)
            .with_n_threads_batch(self.n_threads)
            .with_pooling_type(LlamaPoolingType::Rank)
            .with_embeddings(true);
        let mut ctx = self
            .model
            .new_context(backend, params)
            .context("failed to create llama.cpp rerank context")?;
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        batch
            .add_sequence(&tokens, 0, false)
            .context("failed to add rerank sequence to llama batch")?;
        ctx.clear_kv_cache();
        ctx.decode(&mut batch)
            .context("llama rerank decode failed")?;
        let score_embedding = ctx
            .embeddings_seq_ith(0)
            .context("failed to read rerank score")?;
        let raw_score = *score_embedding
            .first()
            .ok_or_else(|| anyhow!("rerank model produced no score"))?;
        Ok(1.0 / (1.0 + (-raw_score).exp()))
    }
}

impl LocalModelRuntime {
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let vectors = self.embed_texts(&[text])?;
        vectors
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("embed batch returned no vectors"))
    }

    pub fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let embed_model = self
            .embed_model
            .as_ref()
            .ok_or_else(|| anyhow!("embed model is not configured"))?;
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let parallelism = embed_parallelism(texts.len());
        let threads = embed_threads_per_context(parallelism);
        let trace_enabled = env::var("QQD_EMBED_RUNTIME_TRACE")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            .unwrap_or(false);
        let wall_start = Instant::now();
        let mut chunks = Vec::with_capacity(parallelism);
        let chunk_size = texts.len().div_ceil(parallelism);
        for (chunk_index, chunk) in texts.chunks(chunk_size).enumerate() {
            chunks.push((chunk_index, chunk));
        }

        let (mut results, traces) = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(chunks.len());
            for (chunk_index, chunk) in chunks {
                handles.push(scope.spawn(move || -> Result<EmbedWorkerOutput> {
                    let mut trace = EmbedRuntimeTrace {
                        texts: chunk.len(),
                        ..Default::default()
                    };
                    let tokenize_start = Instant::now();
                    let token_lists = embed_model.tokenize_texts(chunk)?;
                    trace.tokenize_ms = elapsed_ms(tokenize_start);
                    let max_tokens = token_lists.iter().map(Vec::len).max().unwrap_or(1);
                    if max_tokens > embed_model.context_size as usize {
                        return Err(anyhow!(
                            "input requires {max_tokens} tokens but embed context is {}",
                            embed_model.context_size
                        ));
                    }
                    trace.max_tokens = max_tokens;
                    let eval_batch_size =
                        QMD_LIKE_EMBED_BATCH_SIZE.min(embed_model.context_size as usize);
                    let context_start = Instant::now();
                    let params = LlamaContextParams::default()
                        .with_n_ctx(NonZeroU32::new(embed_model.context_size))
                        .with_n_batch(eval_batch_size as u32)
                        .with_n_ubatch(eval_batch_size as u32)
                        .with_n_threads(threads)
                        .with_n_threads_batch(threads)
                        .with_embeddings(true);
                    let mut ctx = embed_model
                        .model
                        .new_context(&self.backend, params)
                        .context("failed to create llama.cpp embedding context")?;
                    trace.context_create_ms = elapsed_ms(context_start);

                    let mut out = Vec::with_capacity(chunk.len());
                    for (item_index, tokens) in token_lists.iter().enumerate() {
                        ctx.clear_kv_cache();
                        let mut token_offset = 0usize;
                        while token_offset < tokens.len() {
                            let token_end = (token_offset + eval_batch_size).min(tokens.len());
                            let mut batch = LlamaBatch::new(token_end - token_offset, 1);
                            for (position, token) in
                                tokens[token_offset..token_end].iter().enumerate()
                            {
                                let absolute_position = i32::try_from(token_offset + position)
                                    .map_err(|_| anyhow!("embedding position exceeded i32"))?;
                                batch
                                    .add(
                                        *token,
                                        absolute_position,
                                        &[0],
                                        token_end == tokens.len()
                                            && position + 1 == token_end - token_offset,
                                    )
                                    .context("failed to add token to embedding batch")?;
                            }
                            let encode_start = Instant::now();
                            ctx.encode(&mut batch).context("llama encode failed")?;
                            trace.encode_ms += elapsed_ms(encode_start);
                            token_offset = token_end;
                        }
                        let extract_start = Instant::now();
                        let embedding = ctx
                            .embeddings_seq_ith(0)
                            .context("failed to read llama embedding")?;
                        trace.extract_ms += elapsed_ms(extract_start);
                        out.push((
                            chunk_index * chunk_size + item_index,
                            normalize_embedding(embedding),
                        ));
                    }
                    Ok((out, trace))
                }));
            }

            let mut flattened = Vec::with_capacity(texts.len());
            let mut traces = Vec::new();
            for handle in handles {
                let (mut part, trace) = handle
                    .join()
                    .map_err(|_| anyhow!("embedding worker panicked"))??;
                flattened.append(&mut part);
                traces.push(trace);
            }
            Ok::<(Vec<IndexedEmbedding>, Vec<EmbedRuntimeTrace>), anyhow::Error>((
                flattened, traces,
            ))
        })?;

        results.sort_by_key(|(index, _)| *index);
        if trace_enabled {
            let totals = traces
                .iter()
                .fold(EmbedRuntimeTrace::default(), |mut acc, trace| {
                    acc.texts += trace.texts;
                    acc.max_tokens = acc.max_tokens.max(trace.max_tokens);
                    acc.tokenize_ms += trace.tokenize_ms;
                    acc.context_create_ms += trace.context_create_ms;
                    acc.encode_ms += trace.encode_ms;
                    acc.extract_ms += trace.extract_ms;
                    acc
                });
            eprintln!(
                "QQD_EMBED_RUNTIME_TRACE {}",
                serde_json::to_string(&json!({
                    "workers": traces.len(),
                    "texts": totals.texts,
                    "max_tokens": totals.max_tokens,
                    "phase_ms": {
                        "tokenize": totals.tokenize_ms,
                        "context_create": totals.context_create_ms,
                        "encode": totals.encode_ms,
                        "extract": totals.extract_ms,
                        "wall": elapsed_ms(wall_start),
                    },
                    "parallelism": parallelism,
                    "threads_per_context": threads,
                }))?
            );
        }
        Ok(results
            .into_iter()
            .map(|(_, embedding)| embedding)
            .collect())
    }

    pub fn rerank<'a>(&self, query: &str, documents: &'a [&'a str]) -> Result<Vec<RerankHit>> {
        let rerank_model = self
            .rerank_model
            .as_ref()
            .ok_or_else(|| anyhow!("rerank model is not configured"))?;
        let mut hits = Vec::with_capacity(documents.len());
        for (index, document) in documents.iter().enumerate() {
            let score = rerank_model
                .score_document(&self.backend, query, document)
                .context("failed to rerank candidate document")?;
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
        self.embed_model.is_some()
    }

    pub fn has_reranker(&self) -> bool {
        self.rerank_model.is_some()
    }

    pub fn embed_backend_id(&self) -> &str {
        &self.embed_backend_id
    }

    pub fn tokenize_embedding_text(
        &self,
        text: &str,
    ) -> Result<Vec<llama_cpp_2::token::LlamaToken>> {
        self.embed_model
            .as_ref()
            .ok_or_else(|| anyhow!("embed model is not configured"))?
            .prepare_embedding_tokens(text)
    }

    pub fn detokenize_embedding_tokens(
        &self,
        tokens: &[llama_cpp_2::token::LlamaToken],
    ) -> Result<String> {
        self.embed_model
            .as_ref()
            .ok_or_else(|| anyhow!("embed model is not configured"))?
            .detokenize_tokens(tokens)
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

    let mut backend = LlamaBackend::init().map_err(|error| match error {
        llama_cpp_2::LlamaCppError::BackendAlreadyInitialized => {
            anyhow!("llama backend already initialized unexpectedly")
        }
        other => anyhow!(other),
    })?;
    backend.void_logs();
    let backend = Arc::new(backend);

    let embed_model = if let Some(path) = &embed_model_path {
        Some(DirectEmbedModel::new(&backend, path)?)
    } else {
        None
    };
    let rerank_model = if let Some(path) = &rerank_model_path {
        Some(DirectRerankModel::new(&backend, path)?)
    } else {
        None
    };

    let embed_backend_id = embed_model_path
        .as_ref()
        .map(|path| format!("gguf:{}", canonicalish(path)))
        .unwrap_or_else(|| DETERMINISTIC_BACKEND_ID.to_string());

    Ok(Some(LocalModelRuntime {
        backend,
        embed_model,
        rerank_model,
        embed_backend_id,
    }))
}

pub fn discover_embed_model_path() -> Option<PathBuf> {
    resolve_model_path("QQD_EMBED_MODEL", DEFAULT_EMBED_MODEL_HINTS)
}

pub fn discover_rerank_model_path() -> Option<PathBuf> {
    resolve_model_path("QQD_RERANK_MODEL", DEFAULT_RERANK_MODEL_HINTS)
}

pub fn format_query_for_embedding(query: &str) -> String {
    format_query_for_embedding_with_model(query, active_embed_model_hint().as_deref())
}

pub fn format_document_for_embedding(text: &str, title: Option<&str>) -> String {
    format_document_for_embedding_with_model(text, title, active_embed_model_hint().as_deref())
}

fn format_query_for_embedding_with_model(query: &str, model_hint: Option<&str>) -> String {
    if is_qwen3_embedding_model(model_hint) {
        format!("Instruct: Retrieve relevant documents for the given query\nQuery: {query}")
    } else {
        format!("task: search result | query: {query}")
    }
}

fn format_document_for_embedding_with_model(
    text: &str,
    title: Option<&str>,
    model_hint: Option<&str>,
) -> String {
    if is_qwen3_embedding_model(model_hint) {
        title
            .map(|title| format!("{title}\n{text}"))
            .unwrap_or_else(|| text.to_string())
    } else {
        format!("title: {} | text: {text}", title.unwrap_or("none"))
    }
}

fn active_embed_model_hint() -> Option<String> {
    env::var("QQD_EMBED_MODEL").ok().or_else(|| {
        discover_embed_model_path().map(|path| {
            path.file_name()
                .and_then(|value| value.to_str())
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| path.display().to_string())
        })
    })
}

fn is_qwen3_embedding_model(model_hint: Option<&str>) -> bool {
    model_hint
        .map(|value| {
            let lower = value.to_ascii_lowercase();
            lower.contains("qwen") && lower.contains("embed")
        })
        .unwrap_or(false)
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

fn canonicalish(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

fn default_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get() as i32)
        .unwrap_or(1)
}

fn normalize_embedding(values: &[f32]) -> Vec<f32> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= 0.0 {
        return values.to_vec();
    }
    values.iter().map(|value| *value / norm).collect()
}

fn embed_parallelism(text_count: usize) -> usize {
    if let Some(value) = env_usize("QQD_EMBED_CONTEXTS") {
        return value.clamp(1, 4).min(text_count.max(1));
    }
    let cores = logical_cpu_cores();
    let max_contexts = (cores / 4).clamp(1, 4);
    max_contexts.min(text_count.max(1))
}

fn embed_threads_per_context(parallelism: usize) -> i32 {
    if let Some(value) = env_usize("QQD_EMBED_THREADS_PER_CONTEXT") {
        return i32::try_from(value).unwrap_or(i32::MAX).max(1);
    }
    let cores = logical_cpu_cores();
    let split = (cores / parallelism.max(1)).max(1);
    split as i32
}

fn env_usize(name: &str) -> Option<usize> {
    env::var(name)
        .ok()?
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
}

fn logical_cpu_cores() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(4)
}

#[derive(Default)]
struct EmbedRuntimeTrace {
    texts: usize,
    max_tokens: usize,
    tokenize_ms: f64,
    context_create_ms: f64,
    encode_ms: f64,
    extract_ms: f64,
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
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

    #[test]
    fn formats_embeddinggemma_queries_like_qmd() {
        assert_eq!(
            format_query_for_embedding_with_model("hybrid search", Some("embeddinggemma")),
            "task: search result | query: hybrid search"
        );
    }

    #[test]
    fn formats_embeddinggemma_documents_like_qmd() {
        assert_eq!(
            format_document_for_embedding_with_model(
                "Body text",
                Some("Document Title"),
                Some("embeddinggemma"),
            ),
            "title: Document Title | text: Body text"
        );
        assert_eq!(
            format_document_for_embedding_with_model("Body text", None, Some("embeddinggemma")),
            "title: none | text: Body text"
        );
    }

    #[test]
    fn formats_qwen_embedding_inputs_like_qmd() {
        assert_eq!(
            format_query_for_embedding_with_model(
                "semantic retrieval",
                Some("Qwen3-Embedding-0.6B-Q8_0.gguf"),
            ),
            "Instruct: Retrieve relevant documents for the given query\nQuery: semantic retrieval"
        );
        assert_eq!(
            format_document_for_embedding_with_model(
                "Body text",
                Some("Document Title"),
                Some("Qwen3-Embedding-0.6B-Q8_0.gguf"),
            ),
            "Document Title\nBody text"
        );
    }
}
