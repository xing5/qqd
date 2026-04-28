use crate::{models, sqlite_vec};
use anyhow::{Context, Result, anyhow};
use percent_encoding::percent_decode_str;
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_EMBED_MAX_DOCS_PER_BATCH: usize = 512;
const DEFAULT_EMBED_MAX_BATCH_BYTES: usize = 64 * 1024 * 1024;
const CHUNK_SIZE_TOKENS: usize = 900;
const EMBED_CHUNK_SIZE_CHARS: usize = 2700;
const EMBED_CHUNK_OVERLAP_CHARS: usize = 405;
const EMBED_CHUNK_WINDOW_CHARS: usize = 600;
const RERANK_CHUNK_SIZE_CHARS: usize = 3600;
const RERANK_CHUNK_OVERLAP_CHARS: usize = 540;
const RERANK_CHUNK_WINDOW_CHARS: usize = 800;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchConfig {
    pub index_name: Option<String>,
    pub limit: usize,
    pub collections: Vec<String>,
    pub query: String,
    pub json: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VSearchConfig {
    pub index_name: Option<String>,
    pub query: String,
    pub json: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryConfig {
    pub index_name: Option<String>,
    pub query: String,
    pub json: bool,
    pub limit: usize,
    pub collections: Vec<String>,
    pub min_score: f64,
    pub intent: Option<String>,
    pub rerank: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GetConfig {
    pub index_name: Option<String>,
    pub target: String,
    pub from_line: Option<usize>,
    pub max_lines: Option<usize>,
    pub line_numbers: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiGetConfig {
    pub index_name: Option<String>,
    pub pattern: String,
    pub max_lines: Option<usize>,
    pub max_bytes: usize,
    pub format: OutputFormat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LsConfig {
    pub index_name: Option<String>,
    pub path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusConfig {
    pub index_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionListConfig {
    pub index_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionShowConfig {
    pub index_name: Option<String>,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionAddConfig {
    pub index_name: Option<String>,
    pub path: String,
    pub name: Option<String>,
    pub pattern: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionRemoveConfig {
    pub index_name: Option<String>,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionRenameConfig {
    pub index_name: Option<String>,
    pub old_name: String,
    pub new_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextListConfig {
    pub index_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextAddConfig {
    pub index_name: Option<String>,
    pub path: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextRemoveConfig {
    pub index_name: Option<String>,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateConfig {
    pub index_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CleanupConfig {
    pub index_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedConfig {
    pub index_name: Option<String>,
    pub force: bool,
    pub max_docs_per_batch: Option<usize>,
    pub max_batch_bytes: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Cli,
    Json,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileConfig {
    #[serde(default)]
    pub collections: std::collections::BTreeMap<String, FileCollection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileCollection {
    pub path: String,
    pub pattern: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<std::collections::BTreeMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub update: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "includeByDefault")]
    pub include_by_default: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    docid: String,
    score: i64,
    file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<String>,
    line: usize,
    title: String,
    snippet: String,
    #[serde(skip_serializing)]
    rerank_text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct McpSearchResult {
    pub docid: String,
    pub file: String,
    pub title: String,
    pub score: f64,
    pub context: Option<String>,
    pub snippet: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct QueryResultItem {
    pub docid: String,
    pub score: f64,
    pub file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    pub line: usize,
    pub title: String,
    pub snippet: String,
}

struct QueryExecutionOptions<'a> {
    index_name: Option<&'a str>,
    limit: usize,
    collections: &'a [String],
    min_score: f64,
    intent_override: Option<&'a str>,
    rerank: bool,
}

struct CandidateScoringOptions<'a> {
    runtime: Option<&'a models::LocalModelRuntime>,
    primary_query: &'a str,
    primary_terms: &'a [String],
    intent_terms: &'a [String],
    intent: Option<&'a str>,
    rerank: bool,
    min_score: f64,
    limit: usize,
}

#[derive(Debug, Clone)]
struct EmbedChunk {
    hash: String,
    title: String,
    seq: usize,
    pos: usize,
    text: String,
}

#[derive(Debug, Clone, Copy)]
struct BreakPoint {
    pos: usize,
    score: f64,
}

#[derive(Debug, Clone, Copy)]
struct CodeFenceRegion {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedStructuredQuery {
    searches: Vec<ExpandedQuery>,
    intent: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpandedQuery {
    kind: QueryKind,
    query: String,
    line: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryKind {
    Lex,
    Vec,
    Hyde,
}

const VECTOR_DIMENSIONS: usize = 64;

impl SearchConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }

        let (index_name, rest) = split_index_args(args)?;

        if rest.first().map(String::as_str) != Some("search") {
            return Ok(None);
        }

        let mut limit = 20usize;
        let mut collections = Vec::new();
        let mut query_parts = Vec::new();
        let mut json = false;
        let mut idx = 1usize;

        while idx < rest.len() {
            match rest[idx].as_str() {
                "--json" => {
                    json = true;
                    idx += 1;
                }
                "-n" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("-n requires a numeric value"))?;
                    limit = value
                        .parse::<usize>()
                        .with_context(|| format!("invalid limit: {value}"))?;
                    idx += 2;
                }
                "-c" | "--collection" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("--collection requires a value"))?;
                    collections.push(value.clone());
                    idx += 2;
                }
                unsupported if unsupported.starts_with('-') => {
                    return Ok(None);
                }
                value => {
                    query_parts.push(value.to_string());
                    idx += 1;
                }
            }
        }

        if query_parts.is_empty() {
            return Ok(None);
        }

        let query = query_parts.join(" ");
        if !is_simple_fts_query(&query) {
            return Ok(None);
        }

        Ok(Some(Self {
            index_name,
            limit,
            collections,
            query,
            json,
        }))
    }
}

impl VSearchConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }

        let (index_name, rest) = split_index_args(args)?;
        match rest.first().map(String::as_str) {
            Some("vsearch") | Some("vector-search") => {}
            _ => return Ok(None),
        }

        let mut json = false;
        let mut query_parts = Vec::new();
        let mut idx = 1usize;
        while idx < rest.len() {
            match rest[idx].as_str() {
                "--json" => {
                    json = true;
                    idx += 1;
                }
                unsupported if unsupported.starts_with('-') => return Ok(None),
                value => {
                    query_parts.push(value.to_string());
                    idx += 1;
                }
            }
        }

        if query_parts.is_empty() {
            return Ok(None);
        }

        Ok(Some(Self {
            index_name,
            query: query_parts.join(" "),
            json,
        }))
    }
}

impl QueryConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }

        let (index_name, rest) = split_index_args(args)?;
        if rest.first().map(String::as_str) != Some("query") {
            return Ok(None);
        }

        let mut json = false;
        let mut limit = 10usize;
        let mut collections = Vec::new();
        let mut min_score = 0.0f64;
        let mut intent = None;
        let mut rerank = true;
        let mut query_parts = Vec::new();
        let mut idx = 1usize;
        while idx < rest.len() {
            match rest[idx].as_str() {
                "--json" => {
                    json = true;
                    idx += 1;
                }
                "-n" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("-n requires a numeric value"))?;
                    limit = value
                        .parse::<usize>()
                        .with_context(|| format!("invalid limit: {value}"))?;
                    idx += 2;
                }
                "-c" | "--collection" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("--collection requires a value"))?;
                    collections.push(value.clone());
                    idx += 2;
                }
                "--min-score" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("--min-score requires a value"))?;
                    min_score = value
                        .parse::<f64>()
                        .with_context(|| format!("invalid min score: {value}"))?;
                    idx += 2;
                }
                "--intent" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("--intent requires a value"))?;
                    intent = Some(value.clone());
                    idx += 2;
                }
                "--no-rerank" => {
                    rerank = false;
                    idx += 1;
                }
                unsupported if unsupported.starts_with('-') => return Ok(None),
                value => {
                    query_parts.push(value.to_string());
                    idx += 1;
                }
            }
        }

        if query_parts.is_empty() {
            return Ok(None);
        }

        Ok(Some(Self {
            index_name,
            query: query_parts.join(" "),
            json,
            limit,
            collections,
            min_score,
            intent,
            rerank,
        }))
    }
}

impl GetConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }

        let (index_name, rest) = split_index_args(args)?;
        if rest.first().map(String::as_str) != Some("get") {
            return Ok(None);
        }

        let mut from_line = None;
        let mut max_lines = None;
        let mut line_numbers = false;
        let mut target = None;
        let mut idx = 1usize;

        while idx < rest.len() {
            match rest[idx].as_str() {
                "--from" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("--from requires a value"))?;
                    from_line = Some(value.parse().with_context(|| "invalid --from value")?);
                    idx += 2;
                }
                "-l" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("-l requires a value"))?;
                    max_lines = Some(value.parse().with_context(|| "invalid -l value")?);
                    idx += 2;
                }
                "--line-numbers" => {
                    line_numbers = true;
                    idx += 1;
                }
                unsupported if unsupported.starts_with('-') => return Ok(None),
                value => {
                    target = Some(value.to_string());
                    idx += 1;
                }
            }
        }

        let mut target = match target {
            Some(value) => value,
            None => return Ok(None),
        };

        if from_line.is_none() {
            if let Some((path, parsed_line)) = split_line_suffix(&target) {
                target = path;
                from_line = Some(parsed_line);
            }
        }

        Ok(Some(Self {
            index_name,
            target,
            from_line,
            max_lines,
            line_numbers,
        }))
    }
}

impl MultiGetConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }

        let (index_name, rest) = split_index_args(args)?;
        if rest.first().map(String::as_str) != Some("multi-get") {
            return Ok(None);
        }

        let mut max_lines = None;
        let mut max_bytes = 10 * 1024usize;
        let mut format = OutputFormat::Cli;
        let mut pattern = None;
        let mut idx = 1usize;

        while idx < rest.len() {
            match rest[idx].as_str() {
                "-l" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("-l requires a value"))?;
                    max_lines = Some(value.parse().with_context(|| "invalid -l value")?);
                    idx += 2;
                }
                "--max-bytes" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("--max-bytes requires a value"))?;
                    max_bytes = value.parse().with_context(|| "invalid --max-bytes value")?;
                    idx += 2;
                }
                "--json" => {
                    format = OutputFormat::Json;
                    idx += 1;
                }
                unsupported if unsupported.starts_with('-') => return Ok(None),
                value => {
                    pattern = Some(value.to_string());
                    idx += 1;
                }
            }
        }

        Ok(pattern.map(|pattern| Self {
            index_name,
            pattern,
            max_lines,
            max_bytes,
            format,
        }))
    }
}

impl LsConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }

        let (index_name, rest) = split_index_args(args)?;
        if rest.first().map(String::as_str) != Some("ls") {
            return Ok(None);
        }

        if rest.len() != 2 {
            return Ok(None);
        }

        Ok(Some(Self {
            index_name,
            path: rest.get(1).cloned(),
        }))
    }
}

impl StatusConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }

        let (index_name, rest) = split_index_args(args)?;
        if rest.as_slice() != ["status"] {
            return Ok(None);
        }

        Ok(Some(Self { index_name }))
    }
}

impl CollectionListConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }

        let (index_name, rest) = split_index_args(args)?;
        if rest.as_slice() != ["collection", "list"] {
            return Ok(None);
        }

        Ok(Some(Self { index_name }))
    }
}

impl CollectionShowConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }

        let (index_name, rest) = split_index_args(args)?;
        match rest.as_slice() {
            [first, second, name]
                if first == "collection" && (second == "show" || second == "info") =>
            {
                Ok(Some(Self {
                    index_name,
                    name: name.clone(),
                }))
            }
            _ => Ok(None),
        }
    }
}

impl CollectionAddConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }
        let (index_name, rest) = split_index_args(args)?;
        if rest.len() < 2 || rest[0] != "collection" || rest[1] != "add" {
            return Ok(None);
        }

        let mut path: Option<String> = None;
        let mut name: Option<String> = None;
        let mut pattern = "**/*.md".to_string();
        let mut idx = 2usize;
        while idx < rest.len() {
            match rest[idx].as_str() {
                "--name" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("--name requires a value"))?;
                    name = Some(value.clone());
                    idx += 2;
                }
                "--mask" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("--mask requires a value"))?;
                    pattern = value.clone();
                    idx += 2;
                }
                unsupported if unsupported.starts_with('-') => return Ok(None),
                value => {
                    if path.is_some() {
                        return Ok(None);
                    }
                    path = Some(value.to_string());
                    idx += 1;
                }
            }
        }

        Ok(Some(Self {
            index_name,
            path: path.unwrap_or_else(|| ".".to_string()),
            name,
            pattern,
        }))
    }
}

impl CollectionRemoveConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }
        let (index_name, rest) = split_index_args(args)?;
        match rest.as_slice() {
            [first, second, name]
                if first == "collection" && matches!(second.as_str(), "remove" | "rm") =>
            {
                Ok(Some(Self {
                    index_name,
                    name: name.clone(),
                }))
            }
            _ => Ok(None),
        }
    }
}

impl CollectionRenameConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }
        let (index_name, rest) = split_index_args(args)?;
        match rest.as_slice() {
            [first, second, old_name, new_name]
                if first == "collection" && matches!(second.as_str(), "rename" | "mv") =>
            {
                Ok(Some(Self {
                    index_name,
                    old_name: old_name.clone(),
                    new_name: new_name.clone(),
                }))
            }
            _ => Ok(None),
        }
    }
}

impl ContextListConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }
        let (index_name, rest) = split_index_args(args)?;
        if rest.as_slice() == ["context", "list"] {
            Ok(Some(Self { index_name }))
        } else {
            Ok(None)
        }
    }
}

impl ContextAddConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }
        let (index_name, rest) = split_index_args(args)?;
        if rest.len() < 3 || rest[0] != "context" || rest[1] != "add" {
            return Ok(None);
        }
        let tail = &rest[2..];
        let (path, text) = if tail.len() == 1 {
            (None, tail[0].clone())
        } else {
            (Some(tail[0].clone()), tail[1..].join(" "))
        };
        if text.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            index_name,
            path,
            text,
        }))
    }
}

impl ContextRemoveConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }
        let (index_name, rest) = split_index_args(args)?;
        match rest.as_slice() {
            [first, second, path]
                if first == "context" && matches!(second.as_str(), "rm" | "remove") =>
            {
                Ok(Some(Self {
                    index_name,
                    path: path.clone(),
                }))
            }
            _ => Ok(None),
        }
    }
}

impl UpdateConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }
        let (index_name, rest) = split_index_args(args)?;
        if rest.as_slice() == ["update"] {
            Ok(Some(Self { index_name }))
        } else {
            Ok(None)
        }
    }
}

impl CleanupConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }
        let (index_name, rest) = split_index_args(args)?;
        if rest.as_slice() == ["cleanup"] {
            Ok(Some(Self { index_name }))
        } else {
            Ok(None)
        }
    }
}

impl EmbedConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }
        let (index_name, rest) = split_index_args(args)?;
        if rest.first().map(String::as_str) != Some("embed") {
            return Ok(None);
        }

        let mut force = false;
        let mut max_docs_per_batch = None;
        let mut max_batch_bytes = None;
        let mut idx = 1usize;
        while idx < rest.len() {
            match rest[idx].as_str() {
                "-f" | "--force" => force = true,
                "--max-docs-per-batch" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("--max-docs-per-batch requires a value"))?;
                    max_docs_per_batch =
                        Some(parse_positive_integer_option("max-docs-per-batch", value)?);
                    idx += 1;
                }
                "--max-batch-mb" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("--max-batch-mb requires a value"))?;
                    max_batch_bytes =
                        Some(parse_positive_integer_option("max-batch-mb", value)? * 1024 * 1024);
                    idx += 1;
                }
                unsupported if unsupported.starts_with('-') => return Ok(None),
                _ => return Ok(None),
            }
            idx += 1;
        }

        Ok(Some(Self {
            index_name,
            force,
            max_docs_per_batch,
            max_batch_bytes,
        }))
    }
}

pub fn run_search(config: &SearchConfig) -> Result<()> {
    let db_path = default_db_path(config.index_name.as_deref())?;
    let connection = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .context("failed to open qmd index for native search")?;
    let collections = if config.collections.is_empty() {
        default_collections(&connection)?
    } else {
        config.collections.clone()
    };
    let results = search_results(
        &connection,
        config.index_name.as_deref(),
        &config.query,
        config.limit,
        &collections,
    )?;
    if config.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else if results.is_empty() {
        println!("No results found.");
    } else {
        for result in results {
            println!("{} {}", result.file, result.title);
        }
    }
    Ok(())
}

pub fn run_vsearch(config: &VSearchConfig) -> Result<()> {
    let connection = open_readwrite(config.index_name.as_deref())?;
    ensure_schema(&connection)?;
    let vec0_ready = sqlite_vec::vectors_vec_is_usable(&connection)?;
    if !vec0_ready {
        ensure_qqd_vectors_if_qmd_vectors_exist(&connection)?;
    }
    if vec0_ready || has_native_vectors(&connection)? {
        let runtime = models::runtime()?;
        let results = vector_results(
            &connection,
            runtime.as_deref(),
            config.index_name.as_deref(),
            &config.query,
            20,
            &[],
        )?;
        if config.json {
            println!("{}", serde_json::to_string_pretty(&results)?);
        } else if results.is_empty() {
            println!("No results found.");
        } else {
            for result in results {
                println!("{} {}", result.file, result.title);
            }
        }
        return Ok(());
    }

    let needs_embedding = needs_embedding_count(&connection)?;
    let total_docs: i64 = connection.query_row(
        "SELECT COUNT(*) FROM documents WHERE active = 1",
        [],
        |row| row.get(0),
    )?;
    if total_docs > 0 && needs_embedding == total_docs {
        eprintln!(
            "Warning: {} documents (100%) need embeddings. Run 'qqd embed' for better results.",
            total_docs
        );
    }

    if config.json {
        println!("[]");
    } else {
        println!("No results found.");
    }
    Ok(())
}

pub fn run_query(config: &QueryConfig) -> Result<()> {
    let connection = open_readwrite(config.index_name.as_deref())?;
    ensure_schema(&connection)?;
    if !sqlite_vec::vectors_vec_is_usable(&connection)? {
        ensure_qqd_vectors_if_qmd_vectors_exist(&connection)?;
    }
    let results = query_results(
        &connection,
        &config.query,
        QueryExecutionOptions {
            index_name: config.index_name.as_deref(),
            limit: config.limit,
            collections: &config.collections,
            min_score: config.min_score,
            intent_override: config.intent.as_deref(),
            rerank: config.rerank,
        },
    )?;
    if config.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else if results.is_empty() {
        println!("No results found.");
    } else {
        for result in results {
            println!("{} {}", result.file, result.title);
        }
    }
    Ok(())
}

pub fn run_get(config: &GetConfig) -> Result<()> {
    let connection = open_readonly(config.index_name.as_deref())?;
    let doc = resolve_document(&connection, &config.target)?
        .ok_or_else(|| anyhow!("Document not found: {}", config.target))?;
    let mut output = doc.body;
    let start_line = config.from_line.unwrap_or(1);

    if config.from_line.is_some() || config.max_lines.is_some() {
        let lines = output.lines().collect::<Vec<_>>();
        let start = start_line.saturating_sub(1);
        let end = config
            .max_lines
            .map(|max_lines| start.saturating_add(max_lines))
            .unwrap_or(lines.len());
        output = lines
            .iter()
            .skip(start)
            .take(end.saturating_sub(start))
            .copied()
            .collect::<Vec<_>>()
            .join("\n");
    }

    if config.line_numbers {
        output = add_line_numbers(&output, start_line);
    }
    if let Some(context) = context_for_path(
        &connection,
        config.index_name.as_deref(),
        &doc.collection,
        &doc.path,
    )? {
        println!("Folder Context: {}\n---\n", context);
    }

    println!("{output}");
    Ok(())
}

pub fn run_multi_get(config: &MultiGetConfig) -> Result<()> {
    let connection = open_readonly(config.index_name.as_deref())?;
    let entries = resolve_multi_get_entries(&connection, &config.pattern)?;
    if entries.is_empty() {
        return Err(anyhow!("No files matched pattern: {}", config.pattern));
    }

    let mut results: Vec<MultiGetResult> = Vec::new();
    for entry in entries {
        let display_path = entry.display_path.clone();
        let title = entry.title.clone();
        let body_len = entry.body.len();
        let mut body = entry.body;
        if body_len > config.max_bytes {
            results.push(MultiGetResult {
                display_path: display_path.clone(),
                title,
                body: String::new(),
                skipped: true,
                reason: Some(format!(
                    "File too large ({}KB > {}KB). Use 'qqd get {}' to retrieve.",
                    body_len / 1024,
                    config.max_bytes / 1024,
                    display_path
                )),
            });
            continue;
        }

        if let Some(max_lines) = config.max_lines {
            let lines = body.lines().map(ToOwned::to_owned).collect::<Vec<_>>();
            body = lines
                .iter()
                .take(max_lines)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            if lines.len() > max_lines {
                body.push_str(&format!(
                    "\n\n[... truncated {} more lines]",
                    lines.len() - max_lines
                ));
            }
        }

        results.push(MultiGetResult {
            display_path,
            title,
            body,
            skipped: false,
            reason: None,
        });
    }

    match config.format {
        OutputFormat::Json => {
            let json = results
                .iter()
                .map(|result| {
                    if result.skipped {
                        serde_json::json!({
                            "file": result.display_path,
                            "title": result.title,
                            "skipped": true,
                            "reason": result.reason,
                        })
                    } else {
                        let (collection, path) = resolve_display_path_to_collection_path(
                            &connection,
                            &result.display_path,
                        )
                        .unwrap_or_else(|| (String::new(), result.display_path.clone()));
                        let mut value = serde_json::json!({
                            "file": result.display_path,
                            "title": result.title,
                            "body": result.body,
                        });
                        if let Ok(Some(context)) = context_for_path(
                            &connection,
                            config.index_name.as_deref(),
                            &collection,
                            &path,
                        ) {
                            if let Some(map) = value.as_object_mut() {
                                map.insert(
                                    "context".to_string(),
                                    serde_json::Value::String(context),
                                );
                            }
                        }
                        value
                    }
                })
                .collect::<Vec<_>>();
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        OutputFormat::Cli => {
            for result in results {
                println!(
                    "\n{}\nFile: {}\n{}\n",
                    "=".repeat(60),
                    result.display_path,
                    "=".repeat(60)
                );
                if result.skipped {
                    println!("[SKIPPED: {}]", result.reason.unwrap_or_default());
                } else {
                    if let Some((collection, path)) =
                        resolve_display_path_to_collection_path(&connection, &result.display_path)
                    {
                        if let Some(context) = context_for_path(
                            &connection,
                            config.index_name.as_deref(),
                            &collection,
                            &path,
                        )? {
                            println!("Folder Context: {}\n---\n", context);
                        }
                    }
                    println!("{}", result.body);
                }
            }
        }
    }

    Ok(())
}

pub fn run_ls(config: &LsConfig) -> Result<()> {
    let connection = open_readonly(config.index_name.as_deref())?;
    match &config.path {
        None => {
            println!("No collections found. Run 'qqd collection add .' to index files.");
        }
        Some(path) => {
            let (collection, prefix) = if let Some((collection, path)) = parse_virtual_path(path) {
                (collection, Some(path))
            } else if let Some((collection, path)) = path.split_once('/') {
                (collection.to_string(), Some(path.to_string()))
            } else {
                (path.clone(), None)
            };

            let exists = connection
                .prepare("SELECT 1 FROM store_collections WHERE name = ?1 LIMIT 1")?
                .exists([collection.as_str()])?;
            if !exists {
                return Err(anyhow!("Collection not found: {collection}"));
            }

            let rows = if let Some(ref prefix) = prefix {
                connection
                    .prepare(
                        "SELECT d.path, d.modified_at, LENGTH(content.doc) as size
                         FROM documents d
                         JOIN content ON content.hash = d.hash
                         WHERE d.collection = ?1 AND d.path LIKE ?2 AND d.active = 1
                         ORDER BY d.path",
                    )?
                    .query_map([collection.as_str(), &format!("{prefix}%")], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, i64>(2)?,
                        ))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?
            } else {
                connection
                    .prepare(
                        "SELECT d.path, d.modified_at, LENGTH(content.doc) as size
                         FROM documents d
                         JOIN content ON content.hash = d.hash
                         WHERE d.collection = ?1 AND d.active = 1
                         ORDER BY d.path",
                    )?
                    .query_map([collection.as_str()], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, i64>(2)?,
                        ))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?
            };

            if rows.is_empty() {
                if let Some(prefix) = prefix {
                    println!("No files found under qmd://{collection}/{prefix}");
                } else {
                    println!("No files found in collection: {collection}");
                }
                return Ok(());
            }

            let width = rows
                .iter()
                .map(|(_, _, size)| format_bytes(*size as usize).len())
                .max()
                .unwrap_or(0);
            for (path, modified_at, size) in rows {
                let size_str = format_bytes(size as usize);
                println!(
                    "{:>width$}  {}  qmd://{}/{}",
                    size_str,
                    format_ls_time(&connection, &modified_at)?,
                    collection,
                    path,
                    width = width
                );
            }
        }
    }

    Ok(())
}

pub fn run_status(config: &StatusConfig) -> Result<()> {
    let db_path = default_db_path(config.index_name.as_deref())?;
    let connection = open_readonly(config.index_name.as_deref())?;

    let index_size = std::fs::metadata(&db_path)
        .map(|meta| meta.len() as usize)
        .unwrap_or(0);
    let total_docs: i64 = connection.query_row(
        "SELECT COUNT(*) FROM documents WHERE active = 1",
        [],
        |row| row.get(0),
    )?;
    let vector_count = embedded_hash_count(&connection)?;
    let needs_embedding = needs_embedding_count(&connection)?;
    let latest_modified: Option<String> = connection.query_row(
        "SELECT MAX(modified_at) FROM documents WHERE active = 1",
        [],
        |row| row.get(0),
    )?;
    let collections = connection
        .prepare(
            "SELECT sc.name, sc.pattern, COUNT(d.id) AS file_count, MAX(d.modified_at) AS last_modified
             FROM store_collections sc
             LEFT JOIN documents d ON d.collection = sc.name AND d.active = 1
             GROUP BY sc.name, sc.pattern
             ORDER BY sc.name",
        )?
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    println!("QQD Status\n");
    println!("Index: {}", db_path.display());
    println!("Size:  {}", format_bytes(index_size));
    println!();
    println!("Documents");
    println!("  Total:    {} files indexed", total_docs);
    println!("  Vectors:  {} embedded", vector_count);
    if needs_embedding > 0 {
        println!(
            "  Pending:  {} need embedding (run 'qqd embed')",
            needs_embedding
        );
    }
    if let Some(latest_modified) = latest_modified {
        println!(
            "  Updated:  {}",
            format_time_ago(&connection, &latest_modified)?
        );
    }

    println!("\nAST Chunking");
    println!("  Status:   active");
    println!("  Languages: typescript, tsx, javascript, python, go, rust");

    if !collections.is_empty() {
        println!("\nCollections");
        for (name, pattern, file_count, last_modified) in &collections {
            let updated = last_modified
                .as_deref()
                .map(|value| format_time_ago(&connection, value))
                .transpose()?
                .unwrap_or_else(|| "never".to_string());
            println!("  {} (qmd://{}/)", name, name);
            println!("    Pattern:  {}", pattern);
            println!("    Files:    {} (updated {})", file_count, updated);
        }

        println!("\nExamples");
        println!("  # List files in a collection");
        println!("  qqd ls {}", collections[0].0);
        println!("  # Get a document");
        println!("  qqd get qmd://{}/path/to/file.md", collections[0].0);
        println!("  # Search within a collection");
        println!("  qqd search \"query\" -c {}", collections[0].0);
    } else {
        println!("\nNo collections. Run 'qqd collection add .' to index markdown files.");
    }

    println!("\nModels");
    println!("  Embedding:   https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF");
    println!("  Reranking:   https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF");
    println!("  Generation:  https://huggingface.co/tobil/qmd-query-expansion-1.7B-gguf");

    println!("\nDevice");
    println!("  GPU:      none (running on CPU — models will be slow)");
    println!("  Tip: Install CUDA, Vulkan, or Metal support for GPU acceleration.");
    let cpu_cores = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    println!("  CPU:      {} math cores", cpu_cores);

    if collections.iter().any(|(_, _, _, _)| true) {
        let names = collections
            .iter()
            .map(|(name, _, _, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        println!("\nTips");
        println!(
            "  Add context to collections for better search results: {}",
            names
        );
        println!("    qqd context add qmd://<name>/ \"What this collection contains\"");
        println!("    qqd context add qmd://<name>/meeting-notes \"Weekly team meeting notes\"");
    }

    Ok(())
}

pub fn run_collection_list(config: &CollectionListConfig) -> Result<()> {
    let connection = open_readonly(config.index_name.as_deref())?;
    let collections = connection
        .prepare(
            "SELECT sc.name, sc.pattern, COUNT(d.id) AS file_count, MAX(d.modified_at) AS last_modified
             FROM store_collections sc
             LEFT JOIN documents d ON d.collection = sc.name AND d.active = 1
             GROUP BY sc.name, sc.pattern
             ORDER BY sc.name",
        )?
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    if collections.is_empty() {
        println!("No collections found. Run 'qqd collection add .' to create one.");
        return Ok(());
    }

    println!("Collections ({}):\n", collections.len());
    for (name, pattern, file_count, last_modified) in collections {
        let updated = last_modified
            .as_deref()
            .map(|value| format_time_ago(&connection, value))
            .transpose()?
            .unwrap_or_else(|| "never".to_string());
        println!("{name} (qmd://{name}/)");
        println!("  Pattern:  {pattern}");
        println!("  Files:    {file_count}");
        println!("  Updated:  {updated}\n");
    }

    Ok(())
}

pub fn run_collection_show(config: &CollectionShowConfig) -> Result<()> {
    let connection = open_readonly(config.index_name.as_deref())?;
    let row = connection
        .prepare(
            "SELECT name, path, pattern, COALESCE(include_by_default, 1)
             FROM store_collections
             WHERE name = ?1
             LIMIT 1",
        )?
        .query_row([config.name.as_str()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })
        .optional()?;

    let Some((name, path, pattern, include_by_default)) = row else {
        return Err(anyhow!("Collection not found: {}", config.name));
    };

    println!("Collection: {name}");
    println!("  Path:     {path}");
    println!("  Pattern:  {pattern}");
    println!(
        "  Include:  {}",
        if include_by_default != 0 {
            "yes (default)"
        } else {
            "no"
        }
    );
    Ok(())
}

pub fn run_collection_add(config: &CollectionAddConfig) -> Result<()> {
    let mut connection = open_readwrite(config.index_name.as_deref())?;
    ensure_schema(&connection)?;
    let mut file_config = load_file_config(config.index_name.as_deref())?;
    let resolved_path = resolve_context_fs_path(&config.path)?;
    let name = config.name.clone().unwrap_or_else(|| {
        std::path::Path::new(&resolved_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("root")
            .to_string()
    });

    if file_config.collections.contains_key(&name) {
        return Err(anyhow!("Collection '{}' already exists.", name));
    }
    if file_config.collections.iter().any(|(_, collection)| {
        collection.path == resolved_path && collection.pattern == config.pattern
    }) {
        return Err(anyhow!(
            "A collection already exists for this path and pattern."
        ));
    }

    file_config.collections.insert(
        name.clone(),
        FileCollection {
            path: resolved_path.clone(),
            pattern: config.pattern.clone(),
            ..Default::default()
        },
    );
    save_file_config(config.index_name.as_deref(), &file_config)?;
    upsert_store_collection(&connection, &name, &resolved_path, &config.pattern)?;

    println!("Creating collection '{}'...", name);
    println!("Collection: {} ({})", resolved_path, config.pattern);
    let result =
        reindex_collection_native(&mut connection, &resolved_path, &config.pattern, &name)?;
    println!();
    println!(
        "Indexed: {} new, {} updated, {} unchanged, {} removed",
        result.indexed, result.updated, result.unchanged, result.removed
    );
    let needs_embedding = needs_embedding_count(&connection)?;
    if needs_embedding > 0 {
        println!();
        println!(
            "Run 'qqd embed' to update embeddings ({} unique hashes need vectors)",
            needs_embedding
        );
    }
    println!("✓ Collection '{}' created successfully", name);
    Ok(())
}

pub fn run_collection_remove(config: &CollectionRemoveConfig) -> Result<()> {
    let connection = open_readwrite(config.index_name.as_deref())?;
    let exists = connection
        .prepare("SELECT 1 FROM store_collections WHERE name = ?1 LIMIT 1")?
        .exists([config.name.as_str()])?;
    if !exists {
        return Err(anyhow!("Collection not found: {}", config.name));
    }

    let deleted_docs = connection
        .prepare("DELETE FROM documents WHERE collection = ?1")?
        .execute([config.name.as_str()])?;
    let cleaned_hashes = connection
        .prepare(
            "DELETE FROM content
             WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
        )?
        .execute([])?;
    connection
        .prepare("DELETE FROM store_collections WHERE name = ?1")?
        .execute([config.name.as_str()])?;

    println!("✓ Removed collection '{}'", config.name);
    println!("  Deleted {} documents", deleted_docs);
    if cleaned_hashes > 0 {
        println!("  Cleaned up {} orphaned content hashes", cleaned_hashes);
    }
    let mut file_config = load_file_config(config.index_name.as_deref())?;
    file_config.collections.remove(&config.name);
    save_file_config(config.index_name.as_deref(), &file_config)?;
    Ok(())
}

pub fn run_collection_rename(config: &CollectionRenameConfig) -> Result<()> {
    let connection = open_readwrite(config.index_name.as_deref())?;
    let old_exists = connection
        .prepare("SELECT 1 FROM store_collections WHERE name = ?1 LIMIT 1")?
        .exists([config.old_name.as_str()])?;
    if !old_exists {
        return Err(anyhow!("Collection not found: {}", config.old_name));
    }
    let new_exists = connection
        .prepare("SELECT 1 FROM store_collections WHERE name = ?1 LIMIT 1")?
        .exists([config.new_name.as_str()])?;
    if new_exists {
        return Err(anyhow!(
            "Collection name already exists: {}",
            config.new_name
        ));
    }

    connection
        .prepare("UPDATE documents SET collection = ?1 WHERE collection = ?2")?
        .execute([config.new_name.as_str(), config.old_name.as_str()])?;
    connection
        .prepare("UPDATE store_collections SET name = ?1 WHERE name = ?2")?
        .execute([config.new_name.as_str(), config.old_name.as_str()])?;
    let mut file_config = load_file_config(config.index_name.as_deref())?;
    if let Some(collection) = file_config.collections.remove(&config.old_name) {
        file_config
            .collections
            .insert(config.new_name.clone(), collection);
        save_file_config(config.index_name.as_deref(), &file_config)?;
    }

    println!(
        "✓ Renamed collection '{}' to '{}'",
        config.old_name, config.new_name
    );
    println!(
        "  Virtual paths updated: qmd://{}/ → qmd://{}/",
        config.old_name, config.new_name
    );
    Ok(())
}

pub fn run_context_list(config: &ContextListConfig) -> Result<()> {
    let connection = open_readonly(config.index_name.as_deref())?;
    let rows = connection
        .prepare("SELECT name, context FROM store_collections ORDER BY name")?
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let mut entries = Vec::new();
    for (collection, context) in rows {
        if let Some(context) = context {
            let map: std::collections::BTreeMap<String, String> = serde_json::from_str(&context)?;
            for (path, text) in map {
                entries.push((collection.clone(), path, text));
            }
        }
    }

    if entries.is_empty() {
        println!("No contexts configured. Use 'qqd context add' to add one.");
        return Ok(());
    }

    println!("\nConfigured Contexts\n");
    let mut last_collection = String::new();
    for (collection, path, text) in entries {
        if collection != last_collection {
            println!("{collection}");
            last_collection = collection.clone();
        }
        if path.is_empty() || path == "/" {
            println!("  / (root)");
        } else {
            println!("  {}", path);
        }
        println!("    {}", text);
    }
    Ok(())
}

pub fn run_context_add(config: &ContextAddConfig) -> Result<()> {
    let connection = open_readwrite(config.index_name.as_deref())?;
    let mut file_config = load_file_config(config.index_name.as_deref())?;
    if config.path.as_deref() == Some("/") {
        connection
            .prepare(
                "INSERT INTO store_config (key, value) VALUES ('global_context', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            )?
            .execute([config.text.as_str()])?;
        file_config.global_context = Some(config.text.clone());
        save_file_config(config.index_name.as_deref(), &file_config)?;
        println!("✓ Set global context");
        println!("Context: {}", config.text);
        return Ok(());
    }

    let (collection, path) = detect_context_target(&connection, config.path.as_deref())?;
    let mut ctx_map = load_context_map(&connection, &collection)?;
    ctx_map.insert(path.clone(), config.text.clone());
    save_context_map(&connection, &collection, &ctx_map)?;
    if let Some(file_collection) = file_config.collections.get_mut(&collection) {
        let map = file_collection
            .context
            .get_or_insert_with(std::collections::BTreeMap::new);
        map.insert(path.clone(), config.text.clone());
    }
    save_file_config(config.index_name.as_deref(), &file_config)?;
    let display_path = if path.is_empty() {
        format!("qmd://{collection}/")
    } else {
        format!("qmd://{collection}/{path}")
    };
    println!("✓ Added context for: {}", display_path);
    println!("Context: {}", config.text);
    Ok(())
}

pub fn run_context_remove(config: &ContextRemoveConfig) -> Result<()> {
    let connection = open_readwrite(config.index_name.as_deref())?;
    let mut file_config = load_file_config(config.index_name.as_deref())?;
    if config.path == "/" {
        connection
            .prepare("DELETE FROM store_config WHERE key = 'global_context'")?
            .execute([])?;
        file_config.global_context = None;
        save_file_config(config.index_name.as_deref(), &file_config)?;
        println!("✓ Removed global context");
        return Ok(());
    }

    let (collection, path) = detect_context_target(&connection, Some(config.path.as_str()))?;
    let mut ctx_map = load_context_map(&connection, &collection)?;
    if ctx_map.remove(&path).is_none() {
        return Err(anyhow!("No context found for: qmd://{collection}/{path}"));
    }
    save_context_map(&connection, &collection, &ctx_map)?;
    if let Some(file_collection) = file_config.collections.get_mut(&collection) {
        if let Some(map) = &mut file_collection.context {
            map.remove(&path);
            if map.is_empty() {
                file_collection.context = None;
            }
        }
    }
    save_file_config(config.index_name.as_deref(), &file_config)?;
    println!("✓ Removed context for: qmd://{}/{}", collection, path);
    Ok(())
}

pub fn run_update(config: &UpdateConfig) -> Result<()> {
    let mut connection = open_readwrite(config.index_name.as_deref())?;
    ensure_schema(&connection)?;
    let file_config = load_file_config(config.index_name.as_deref())?;
    if file_config.collections.is_empty() {
        println!("No collections found. Run 'qqd collection add .' to index markdown files.");
        return Ok(());
    }

    connection.execute("DELETE FROM llm_cache", [])?;
    println!(
        "Updating {} collection(s)...\n",
        file_config.collections.len()
    );
    let mut ordered = file_config.collections.into_iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.0.cmp(&b.0));

    for (index, (name, collection)) in ordered.iter().enumerate() {
        println!(
            "[{}/{}] {} ({})",
            index + 1,
            ordered.len(),
            name,
            collection.pattern
        );
        if let Some(update_cmd) = &collection.update {
            println!("    Running update command: {}", update_cmd);
            let output = std::process::Command::new("bash")
                .arg("-c")
                .arg(update_cmd)
                .current_dir(&collection.path)
                .output()
                .with_context(|| format!("failed to run update command for {}", name))?;
            if !output.stdout.is_empty() {
                print!("{}", String::from_utf8_lossy(&output.stdout));
            }
            if !output.stderr.is_empty() {
                print!("{}", String::from_utf8_lossy(&output.stderr));
            }
            if !output.status.success() {
                return Err(anyhow!("update command failed for {}", name));
            }
        }

        println!("Collection: {} ({})", collection.path, collection.pattern);
        let result = reindex_collection_native(
            &mut connection,
            &collection.path,
            &collection.pattern,
            name,
        )?;
        println!(
            "\nIndexed: {} new, {} updated, {} unchanged, {} removed\n",
            result.indexed, result.updated, result.unchanged, result.removed
        );
    }

    let needs_embedding = needs_embedding_count(&connection)?;
    println!("✓ All collections updated.");
    if needs_embedding > 0 {
        println!(
            "\nRun 'qqd embed' to update embeddings ({} unique hashes need vectors)",
            needs_embedding
        );
    }
    Ok(())
}

pub fn run_cleanup(config: &CleanupConfig) -> Result<()> {
    let connection = open_readwrite(config.index_name.as_deref())?;
    ensure_schema(&connection)?;

    let cache_count = connection.execute("DELETE FROM llm_cache", [])?;
    println!("✓ Cleared {} cached API responses", cache_count);

    let orphaned_vecs = if table_exists(&connection, "vectors_vec")? {
        if let Some(path) = sqlite_vec::discover_vec0_path() {
            if sqlite_vec::load_vec0_extension(&connection, &path).is_ok() {
                connection.execute(
                    "DELETE FROM vectors_vec WHERE hash_seq IN (
                       SELECT cv.hash || '_' || cv.seq
                       FROM content_vectors cv
                       WHERE NOT EXISTS (
                         SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1
                       )
                     )",
                    [],
                )?;
                connection.execute(
                    "DELETE FROM content_vectors
                     WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
                    [],
                )?
            } else {
                0
            }
        } else {
            0
        }
    } else {
        0
    };
    if orphaned_vecs > 0 {
        println!("✓ Removed {} orphaned embedding chunks", orphaned_vecs);
    } else {
        println!("No orphaned embeddings to remove");
    }

    let inactive_docs = connection.execute("DELETE FROM documents WHERE active = 0", [])?;
    if inactive_docs > 0 {
        println!("✓ Removed {} inactive document records", inactive_docs);
    }

    connection.execute_batch("VACUUM")?;
    println!("✓ Database vacuumed");
    Ok(())
}

pub fn run_embed(config: &EmbedConfig) -> Result<()> {
    let trace_enabled = trace_indexing_enabled();
    let total_start = Instant::now();
    let mut trace = EmbedTrace::default();
    let mut connection = open_readwrite(config.index_name.as_deref())?;
    ensure_schema(&connection)?;
    ensure_qqd_vectors_schema(&connection)?;
    let runtime_start = Instant::now();
    let runtime = models::runtime()?;
    trace.runtime_init_ms = elapsed_ms(runtime_start);
    let vec0_path = sqlite_vec::discover_vec0_path();
    let mut vec0_ready = false;

    if config.force {
        if table_exists(&connection, "qqd_vectors")? {
            connection.execute("DELETE FROM qqd_vectors", [])?;
        }
        connection.execute("DELETE FROM content_vectors", [])?;
        if let Some(path) = vec0_path.as_deref() {
            if sqlite_vec::load_vec0_extension(&connection, path).is_ok() {
                connection.execute_batch("DROP TABLE IF EXISTS vectors_vec")?;
            }
        }
    }

    let docs_query_start = Instant::now();
    let docs = connection
        .prepare(if config.force {
            "SELECT c.hash, MIN(d.title), c.doc
             FROM content c
             JOIN documents d ON d.hash = c.hash AND d.active = 1
             GROUP BY c.hash, c.doc
             ORDER BY c.hash"
        } else {
            "SELECT c.hash, MIN(d.title), c.doc
             FROM content c
             JOIN documents d ON d.hash = c.hash AND d.active = 1
             WHERE EXISTS (SELECT 1 FROM documents d2 WHERE d2.hash = c.hash AND d2.active = 1)
               AND NOT EXISTS (SELECT 1 FROM qqd_vectors qv WHERE qv.hash = c.hash)
             GROUP BY c.hash, c.doc
             ORDER BY c.hash"
        })?
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    trace.docs_query_ms = elapsed_ms(docs_query_start);
    trace.docs_seen = docs.len();

    if docs.is_empty() {
        println!("✓ No non-empty documents to embed.");
        return Ok(());
    }

    let now = current_timestamp();
    let mut embedded = 0usize;
    let backend_id = models::embed_backend_id_for_runtime(runtime.as_deref());
    let model_name = if runtime
        .as_deref()
        .map(|value| value.has_embedder())
        .unwrap_or(false)
    {
        "qqd-gguf"
    } else {
        "qqd-deterministic"
    };
    let max_docs_per_batch = config
        .max_docs_per_batch
        .unwrap_or(DEFAULT_EMBED_MAX_DOCS_PER_BATCH);
    let max_batch_bytes = config
        .max_batch_bytes
        .unwrap_or(DEFAULT_EMBED_MAX_BATCH_BYTES);
    let mut batch_start = 0usize;
    let mut batch_index = 0usize;
    while batch_start < docs.len() {
        let batch_total_start = Instant::now();
        let batch_end =
            next_embed_batch_end(&docs, batch_start, max_docs_per_batch, max_batch_bytes);
        let batch = &docs[batch_start..batch_end];
        trace.batches += 1;
        let mut doc_ranges = Vec::with_capacity(batch.len());
        let mut batch_chunks = Vec::new();
        for (hash, title, body) in batch {
            let start = batch_chunks.len();
            let chunks = chunk_document_for_embedding(runtime.as_deref(), body);
            for (seq, (pos, text)) in chunks.into_iter().enumerate() {
                batch_chunks.push(EmbedChunk {
                    hash: hash.clone(),
                    title: title.clone(),
                    seq,
                    pos,
                    text,
                });
            }
            let end = batch_chunks.len();
            doc_ranges.push((hash.as_str(), start, end));
        }
        let texts = batch_chunks
            .iter()
            .map(|chunk| models::format_document_for_embedding(&chunk.text, Some(&chunk.title)))
            .collect::<Vec<_>>();
        let text_refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
        let compute_start = Instant::now();
        let vectors = compute_embeddings(runtime.as_deref(), text_refs.as_slice())?;
        trace.compute_embeddings_ms += elapsed_ms(compute_start);
        if !vec0_ready && !vectors.is_empty() {
            if let Some(path) = vec0_path.as_deref() {
                sqlite_vec::load_vec0_extension(&connection, path)?;
                sqlite_vec::ensure_vec0_table(&connection, "vectors_vec", vectors[0].len())?;
                vec0_ready = true;
            }
        }
        if !vec0_ready {
            ensure_qqd_vectors_schema(&connection)?;
        }
        let tx = connection.transaction()?;
        let mut doc_vectors = std::collections::BTreeMap::<String, Vec<Vec<f32>>>::new();
        for (chunk, vector) in batch_chunks.iter().zip(vectors.into_iter()) {
            let serialize_start = Instant::now();
            let vector_json = serde_json::to_string(&vector)?;
            trace.serialize_vectors_ms += elapsed_ms(serialize_start);
            let content_vector_delete_start = Instant::now();
            tx.execute(
                "DELETE FROM content_vectors WHERE hash = ?1 AND seq = ?2",
                (&chunk.hash, chunk.seq as i64),
            )?;
            trace.content_vector_delete_ms += elapsed_ms(content_vector_delete_start);
            let content_vector_insert_start = Instant::now();
            tx.execute(
                "INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                (
                    &chunk.hash,
                    chunk.seq as i64,
                    chunk.pos as i64,
                    model_name,
                    now.as_str(),
                ),
            )?;
            trace.content_vector_insert_ms += elapsed_ms(content_vector_insert_start);
            if vec0_ready {
                let hash_seq = format!("{}_{}", chunk.hash, chunk.seq);
                tx.execute("DELETE FROM vectors_vec WHERE hash_seq = ?1", [&hash_seq])?;
                tx.execute(
                    "INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?1, ?2)",
                    (&hash_seq, &vector_json),
                )?;
            }
            doc_vectors
                .entry(chunk.hash.clone())
                .or_default()
                .push(vector);
            embedded += 1;
        }
        for (hash, start, end) in doc_ranges {
            if start == end {
                continue;
            }
            let Some(vectors) = doc_vectors.get(hash) else {
                continue;
            };
            if !vec0_ready {
                let averaged = average_embeddings(vectors);
                let qqd_vector_write_start = Instant::now();
                tx.execute(
                    "INSERT INTO qqd_vectors (hash, vector, updated_at)
                     VALUES (?1, ?2, ?3)
                     ON CONFLICT(hash) DO UPDATE SET vector = excluded.vector, updated_at = excluded.updated_at",
                    (hash, serde_json::to_string(&averaged)?, now.as_str()),
                )?;
                trace.qqd_vector_write_ms += elapsed_ms(qqd_vector_write_start);
            }
        }
        let tx_commit_start = Instant::now();
        tx.commit()?;
        trace.tx_commit_ms += elapsed_ms(tx_commit_start);
        let batch_total_ms = elapsed_ms(batch_total_start);
        if batch_total_ms > trace.slowest_batch_ms {
            trace.slowest_batch_ms = batch_total_ms;
            trace.slowest_batch_index = batch_index;
            trace.slowest_batch_docs = batch.len();
        }
        batch_start = batch_end;
        batch_index += 1;
    }
    write_store_config_value(&connection, "qqd_embed_backend", &backend_id)?;
    trace.total_ms = elapsed_ms(total_start);

    if trace_enabled {
        eprintln!(
            "QQD_EMBED_TRACE {}",
            serde_json::to_string(&json!({
                "docs_seen": trace.docs_seen,
                "batches": trace.batches,
                "embedded": embedded,
                "phase_ms": {
                    "runtime_init": trace.runtime_init_ms,
                    "docs_query": trace.docs_query_ms,
                    "compute_embeddings": trace.compute_embeddings_ms,
                    "serialize_vectors": trace.serialize_vectors_ms,
                    "qqd_vector_write": trace.qqd_vector_write_ms,
                    "content_vector_delete": trace.content_vector_delete_ms,
                    "content_vector_insert": trace.content_vector_insert_ms,
                    "tx_commit": trace.tx_commit_ms,
                    "total": trace.total_ms,
                },
                "slowest_batch": {
                    "index": trace.slowest_batch_index,
                    "docs": trace.slowest_batch_docs,
                    "ms": trace.slowest_batch_ms,
                },
            }))?
        );
    }

    if embedded == 0 {
        println!("✓ All content hashes already have embeddings.");
    } else {
        println!(
            "✓ Done! Embedded {} chunks from {} documents.",
            embedded, embedded
        );
    }
    Ok(())
}

pub fn default_db_path(index_name: Option<&str>) -> Result<PathBuf> {
    if let Ok(index_path) = env::var("INDEX_PATH") {
        return Ok(PathBuf::from(index_path));
    }

    let mut normalized = index_name.unwrap_or("index").to_string();
    if normalized.contains('/') {
        let cwd = env::current_dir()?;
        let absolute = cwd.join(&normalized);
        normalized = absolute.display().to_string().replace('/', "_");
        normalized = normalized.trim_start_matches('_').to_string();
    }

    let cache_root = env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(
                env::var("HOME")
                    .map(|home| format!("{home}/.cache"))
                    .unwrap_or_else(|_| ".cache".to_string()),
            )
        });
    Ok(cache_root.join("qmd").join(format!("{normalized}.sqlite")))
}

pub fn should_render_help(args: &[String]) -> bool {
    args.is_empty() || args.iter().any(|arg| arg == "--help" || arg == "-h")
}

pub fn run_help() -> Result<()> {
    let index = default_db_path(None)?;
    let embed_model = models::discover_embed_model_path()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "not found".to_string());
    let rerank_model = models::discover_rerank_model_path()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "not found".to_string());
    println!(
        "qqd — Quick Markdown Search\n\n\
Usage:\n  qqd <command> [options]\n\n\
Primary commands:\n  qqd query <query>             - Hybrid search with auto expansion + reranking (recommended)\n  qqd query 'lex:..\\nvec:...'   - Structured query document (you provide lex/vec/hyde lines)\n  qqd search <query>            - Full-text BM25 keywords (no LLM)\n  qqd vsearch <query>           - Vector similarity only\n  qqd get <file>[:line] [-l N]  - Show a single document, optional line slice\n  qqd multi-get <pattern>       - Batch fetch via glob or comma-separated list\n  qqd mcp                       - Start the MCP server (stdio transport for AI agents)\n  qqd bench-latency <query>     - Compare qqd vs qmd latency on a selected workload\n  qqd bench-quality <fixture>   - Run the committed query-quality gate fixture\n  qqd bench-metrics <fixture>   - Report precision/recall/F1/MRR/nDCG on a fixture\n\n\
Collections & context:\n  qqd collection add/list/remove/rename/show   - Manage indexed folders\n  qqd context add/list/rm                      - Attach human-written summaries\n  qqd ls [collection[/path]]                   - Inspect indexed files\n\n\
Maintenance:\n  qqd status                    - View index + collection health\n  qqd update                    - Re-index collections\n  qqd embed [-f]                - Generate/refresh vector embeddings\n    --max-docs-per-batch <n>    - Cap docs loaded into memory per embedding batch\n    --max-batch-mb <n>          - Cap UTF-8 MB loaded into memory per embedding batch\n  qqd cleanup                   - Clear caches, vacuum DB\n\n\
Global options:\n  --index <name>             - Use a named index (default: index)\n  QMD_EDITOR_URI             - Editor link template for clickable TTY search output\n\n\
Local model setup:\n  - qqd auto-discovers GGUF models from ~/.cache/qmd/models to match qmd-style setup.\n  - Optional overrides: QQD_EMBED_MODEL, QQD_RERANK_MODEL, QQD_MODEL_THREADS.\n  - Discovered embed model: {}\n  - Discovered rerank model: {}\n\n\
Index: {}\n",
        embed_model,
        rerank_model,
        index.display()
    );
    Ok(())
}

fn is_simple_fts_query(query: &str) -> bool {
    query.chars().all(|c| {
        c.is_ascii_alphanumeric() || c.is_ascii_whitespace() || matches!(c, '_' | '-' | '.')
    })
}

fn default_collections(connection: &Connection) -> Result<Vec<String>> {
    let mut statement = connection.prepare(
        "SELECT name FROM store_collections
         WHERE include_by_default IS NULL OR include_by_default != 0
         ORDER BY name",
    )?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

pub fn context_for_path(
    connection: &Connection,
    index_name: Option<&str>,
    collection: &str,
    path: &str,
) -> Result<Option<String>> {
    let file_config = load_file_config(index_name).unwrap_or_default();
    let global = if table_exists(connection, "store_config")? {
        connection
            .prepare("SELECT value FROM store_config WHERE key = 'global_context'")?
            .query_row([], |row| row.get::<_, String>(0))
            .optional()?
            .or_else(|| file_config.global_context.clone())
    } else {
        file_config.global_context.clone()
    };
    let local = if table_exists(connection, "store_collections")? {
        connection
            .prepare("SELECT context FROM store_collections WHERE name = ?1")?
            .query_row([collection], |row| row.get::<_, Option<String>>(0))
            .optional()?
            .flatten()
            .or_else(|| {
                file_config
                    .collections
                    .get(collection)
                    .and_then(|value| value.context.clone())
                    .map(|value| serde_json::to_string(&value))
                    .transpose()
                    .ok()
                    .flatten()
            })
    } else {
        file_config
            .collections
            .get(collection)
            .and_then(|value| value.context.clone())
            .map(|value| serde_json::to_string(&value))
            .transpose()?
    };

    let mut contexts = Vec::new();
    if let Some(global) = global {
        contexts.push(global);
    }
    if let Some(local) = local {
        let map: std::collections::BTreeMap<String, String> = serde_json::from_str(&local)?;
        let normalized = if path.starts_with('/') {
            path.to_string()
        } else {
            format!("/{path}")
        };
        let mut matches = map
            .into_iter()
            .filter_map(|(prefix, text)| {
                let prefix = if prefix.starts_with('/') {
                    prefix
                } else {
                    format!("/{prefix}")
                };
                normalized
                    .starts_with(&prefix)
                    .then_some((prefix.len(), text))
            })
            .collect::<Vec<_>>();
        matches.sort_by_key(|(len, _)| *len);
        for (_, text) in matches {
            contexts.push(text);
        }
    }
    Ok((!contexts.is_empty()).then(|| contexts.join("\n\n")))
}

fn has_native_vectors(connection: &Connection) -> Result<bool> {
    if !table_exists(connection, "qqd_vectors")? {
        return Ok(false);
    }
    Ok(connection
        .prepare("SELECT 1 FROM qqd_vectors LIMIT 1")?
        .exists([])?)
}

fn has_qmd_vector_state(connection: &Connection) -> Result<bool> {
    if !table_exists(connection, "content_vectors")? {
        return Ok(false);
    }
    Ok(connection
        .prepare("SELECT 1 FROM content_vectors LIMIT 1")?
        .exists([])?)
}

fn read_store_config_value(connection: &Connection, key: &str) -> Result<Option<String>> {
    if !table_exists(connection, "store_config")? {
        return Ok(None);
    }
    connection
        .prepare("SELECT value FROM store_config WHERE key = ?1 LIMIT 1")?
        .query_row([key], |row| row.get::<_, String>(0))
        .optional()
        .map_err(Into::into)
}

fn write_store_config_value(connection: &Connection, key: &str, value: &str) -> Result<()> {
    connection.execute(
        "INSERT INTO store_config (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )?;
    Ok(())
}

fn stored_embed_backend_id(connection: &Connection) -> Result<Option<String>> {
    read_store_config_value(connection, "qqd_embed_backend")
}

fn ensure_vector_backend_compatibility(connection: &Connection) -> Result<()> {
    if !has_native_vectors(connection)? {
        return Ok(());
    }
    let current = models::current_embed_backend_id()?;
    let stored = stored_embed_backend_id(connection)?
        .unwrap_or_else(|| models::deterministic_backend_id().to_string());
    if stored == current {
        return Ok(());
    }
    Err(anyhow!(
        "qqd vectors were built with backend '{stored}', but current backend is '{current}'. Run 'qqd embed --force' to rebuild embeddings."
    ))
}

fn embedded_hash_count(connection: &Connection) -> Result<i64> {
    if !table_exists(connection, "content_vectors")? && !table_exists(connection, "qqd_vectors")? {
        return Ok(0);
    }
    Ok(connection
        .query_row(
            "SELECT COUNT(*) FROM (
           SELECT hash FROM content_vectors
           UNION
           SELECT hash FROM qqd_vectors
         )",
            [],
            |row| row.get(0),
        )
        .or_else(|_| {
            connection.query_row("SELECT COUNT(*) FROM content_vectors", [], |row| row.get(0))
        })?)
}

fn needs_embedding_count(connection: &Connection) -> Result<i64> {
    Ok(connection
        .query_row(
            "SELECT COUNT(*)
         FROM content c
         WHERE EXISTS (
           SELECT 1 FROM documents d WHERE d.hash = c.hash AND d.active = 1
         ) AND NOT EXISTS (
           SELECT 1 FROM content_vectors v WHERE v.hash = c.hash
         ) AND NOT EXISTS (
           SELECT 1 FROM qqd_vectors qv WHERE qv.hash = c.hash
         )",
            [],
            |row| row.get(0),
        )
        .or_else(|_| {
            connection.query_row(
                "SELECT COUNT(*)
             FROM content c
             WHERE EXISTS (
               SELECT 1 FROM documents d WHERE d.hash = c.hash AND d.active = 1
             ) AND NOT EXISTS (
               SELECT 1 FROM content_vectors v WHERE v.hash = c.hash
             )",
                [],
                |row| row.get(0),
            )
        })?)
}

fn ensure_qqd_vectors_if_qmd_vectors_exist(connection: &Connection) -> Result<()> {
    if has_native_vectors(connection)? {
        ensure_vector_backend_compatibility(connection)?;
        return Ok(());
    }
    if !has_qmd_vector_state(connection)? {
        return Ok(());
    }
    ensure_qqd_vectors_schema(connection)?;
    let now = current_timestamp();
    let docs = connection
        .prepare(
            "SELECT c.hash, MIN(d.title), c.doc
             FROM content c
             JOIN documents d ON d.hash = c.hash AND d.active = 1
             GROUP BY c.hash, c.doc
             ORDER BY c.hash",
        )?
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let runtime = models::runtime()?;
    let backend_id = models::embed_backend_id_for_runtime(runtime.as_deref());
    let texts = docs
        .iter()
        .map(|(_, title, body)| models::format_document_for_embedding(body, Some(title)))
        .collect::<Vec<_>>();
    let vectors = compute_embeddings(
        runtime.as_deref(),
        texts
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
    )?;
    for ((hash, _title, _body), vector) in docs.into_iter().zip(vectors.into_iter()) {
        connection.execute(
            "INSERT INTO qqd_vectors (hash, vector, updated_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(hash) DO UPDATE SET vector = excluded.vector, updated_at = excluded.updated_at",
            (hash.as_str(), serde_json::to_string(&vector)?, now.as_str()),
        )?;
    }
    write_store_config_value(connection, "qqd_embed_backend", &backend_id)?;
    Ok(())
}

fn query_results(
    connection: &Connection,
    raw_query: &str,
    options: QueryExecutionOptions<'_>,
) -> Result<Vec<QueryResultItem>> {
    let runtime = models::runtime()?;
    let parsed = parse_structured_query(raw_query)?;
    let intent = options
        .intent_override
        .map(ToOwned::to_owned)
        .or_else(|| parsed.as_ref().and_then(|value| value.intent.clone()));
    let has_vectors = has_native_vectors(connection)? || has_qmd_vector_state(connection)?;
    let searches = match parsed {
        Some(value) => value.searches,
        None => build_plain_query_searches(raw_query, has_vectors),
    };
    if searches.is_empty() {
        return Ok(Vec::new());
    }

    let collections = if options.collections.is_empty() {
        default_collections(connection)?
    } else {
        options.collections.to_vec()
    };
    let candidate_limit = options.limit.max(40);
    let mut ranked_lists = Vec::new();
    for search in &searches {
        let results = match search.kind {
            QueryKind::Lex => search_results(
                connection,
                options.index_name,
                &search.query,
                candidate_limit,
                &collections,
            )?,
            QueryKind::Vec | QueryKind::Hyde if has_vectors => vector_results(
                connection,
                runtime.as_deref(),
                options.index_name,
                &search.query,
                candidate_limit,
                &collections,
            )?,
            QueryKind::Vec | QueryKind::Hyde => Vec::new(),
        };
        if !results.is_empty() {
            ranked_lists.push(results);
        }
    }
    if ranked_lists.is_empty() {
        return Ok(Vec::new());
    }

    let primary_query = searches
        .iter()
        .find(|search| search.kind == QueryKind::Lex)
        .or_else(|| searches.first())
        .map(|search| search.query.as_str())
        .unwrap_or(raw_query);
    let primary_terms = query_terms(primary_query);
    let intent_terms = intent.as_deref().map(query_terms).unwrap_or_default();

    #[derive(Default)]
    struct Candidate {
        item: Option<SearchResultItem>,
        score: f64,
        sources: usize,
    }

    let mut fused = std::collections::BTreeMap::<String, Candidate>::new();
    for (list_index, results) in ranked_lists.into_iter().enumerate() {
        let weight = if list_index == 0 { 2.0 } else { 1.0 };
        for (rank, item) in results.into_iter().enumerate() {
            let entry = fused.entry(item.file.clone()).or_default();
            entry.score += weight / (rank as f64 + 1.0);
            entry.sources += 1;
            if entry.item.is_none() {
                entry.item = Some(item);
            }
        }
    }

    let mut fused = fused
        .into_values()
        .filter_map(|candidate| Some((candidate.score, candidate.sources, candidate.item?)))
        .collect::<Vec<_>>();
    fused.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.file.cmp(&b.2.file))
    });

    let candidates = fused.into_iter().collect::<Vec<_>>();
    rerank_or_score_candidates(
        CandidateScoringOptions {
            runtime: runtime.as_deref(),
            primary_query,
            primary_terms: &primary_terms,
            intent_terms: &intent_terms,
            intent: intent.as_deref(),
            rerank: options.rerank,
            min_score: options.min_score,
            limit: options.limit,
        },
        candidates,
    )
}

pub fn query_results_for_mcp(
    index_name: Option<&str>,
    searches: &[serde_json::Value],
    limit: Option<usize>,
    min_score: Option<f64>,
    collections: &[String],
    intent: Option<&str>,
    rerank: bool,
) -> Result<Vec<McpSearchResult>> {
    let connection = open_readwrite(index_name)?;
    ensure_schema(&connection)?;
    if !sqlite_vec::vectors_vec_is_usable(&connection)? {
        ensure_qqd_vectors_if_qmd_vectors_exist(&connection)?;
    }
    let query_doc = searches
        .iter()
        .filter_map(|search| {
            Some(format!(
                "{}: {}",
                search.get("type")?.as_str()?,
                search.get("query")?.as_str()?
            ))
        })
        .collect::<Vec<_>>()
        .join("\n");
    let results = query_results(
        &connection,
        &query_doc,
        QueryExecutionOptions {
            index_name,
            limit: limit.unwrap_or(10),
            collections,
            min_score: min_score.unwrap_or(0.0),
            intent_override: intent,
            rerank,
        },
    )?;
    Ok(results
        .into_iter()
        .map(|result| McpSearchResult {
            docid: result.docid,
            file: result
                .file
                .strip_prefix("qmd://")
                .unwrap_or(&result.file)
                .to_string(),
            title: result.title,
            score: result.score,
            context: result.context,
            snippet: result.snippet,
        })
        .collect())
}

fn rerank_or_score_candidates(
    options: CandidateScoringOptions<'_>,
    candidates: Vec<(f64, usize, SearchResultItem)>,
) -> Result<Vec<QueryResultItem>> {
    if options.rerank {
        if let Some(runtime) = options.runtime {
            if runtime.has_reranker() {
                let rerank_query = options
                    .intent
                    .map(|value| format!("{value}\n\n{}", options.primary_query))
                    .unwrap_or_else(|| options.primary_query.to_string());
                let documents = candidates
                    .iter()
                    .map(|(_, _, item)| {
                        let (_pos, best_chunk) = best_chunk_for_query(
                            &item.rerank_text,
                            options.primary_terms,
                            options.intent_terms,
                        );
                        format!("# {}\n\n{}", item.title, best_chunk)
                    })
                    .collect::<Vec<_>>();
                let document_refs = documents.iter().map(String::as_str).collect::<Vec<_>>();
                let reranked = runtime.rerank(&rerank_query, &document_refs)?;
                let max_base = candidates
                    .iter()
                    .map(|(base, _, _)| *base)
                    .fold(0.0f64, f64::max)
                    .max(1.0);
                let candidate_by_index = candidates
                    .into_iter()
                    .enumerate()
                    .map(|(rank, (base, sources, item))| {
                        (rank, (base / max_base, sources, rank, item))
                    })
                    .collect::<std::collections::BTreeMap<_, _>>();
                return Ok(reranked
                    .into_iter()
                    .filter_map(|hit| {
                        let (base_score, sources, rank, item) =
                            candidate_by_index.get(&hit.index)?;
                        let heuristic_score = estimate_query_score(
                            item,
                            options.primary_terms,
                            options.intent_terms,
                            *rank,
                            *sources,
                        );
                        let score = 0.4 * f64::from(hit.relevance_score)
                            + 0.2 * *base_score
                            + 0.4 * heuristic_score;
                        (score >= options.min_score).then_some(QueryResultItem {
                            docid: item.docid.clone(),
                            score: round_score(score),
                            file: item.file.clone(),
                            context: item.context.clone(),
                            line: item.line,
                            title: item.title.clone(),
                            snippet: item.snippet.clone(),
                        })
                    })
                    .take(options.limit)
                    .collect());
            }
        }
    }

    Ok(candidates
        .into_iter()
        .enumerate()
        .filter_map(|(rank, (_rrf_score, sources, item))| {
            let score = if options.rerank {
                estimate_query_score(
                    &item,
                    options.primary_terms,
                    options.intent_terms,
                    rank,
                    sources,
                )
            } else {
                round_score((1.0 / (rank as f64 + 1.0)).max(0.0))
            };
            (score >= options.min_score).then_some(QueryResultItem {
                docid: item.docid,
                score,
                file: item.file,
                context: item.context,
                line: item.line,
                title: item.title,
                snippet: item.snippet,
            })
        })
        .take(options.limit)
        .collect())
}

fn compute_embeddings(
    runtime: Option<&models::LocalModelRuntime>,
    texts: &[&str],
) -> Result<Vec<Vec<f32>>> {
    match runtime {
        Some(runtime) if runtime.has_embedder() => compute_live_embeddings(runtime, texts),
        _ => texts
            .iter()
            .map(|text| compute_embedding(runtime, text))
            .collect(),
    }
}

fn compute_embedding(runtime: Option<&models::LocalModelRuntime>, text: &str) -> Result<Vec<f32>> {
    match runtime {
        Some(runtime) if runtime.has_embedder() => compute_live_embedding(runtime, text),
        _ => Ok(embed_text(text)),
    }
}

fn compute_live_embeddings(
    runtime: &models::LocalModelRuntime,
    texts: &[&str],
) -> Result<Vec<Vec<f32>>> {
    match runtime.embed_texts(texts) {
        Ok(vectors) => Ok(vectors),
        Err(_) => {
            let mut all_vectors = Vec::with_capacity(texts.len());
            for text in texts {
                all_vectors.push(compute_live_embedding(runtime, text)?);
            }
            Ok(all_vectors)
        }
    }
}

fn compute_live_embedding(runtime: &models::LocalModelRuntime, text: &str) -> Result<Vec<f32>> {
    if let Some(embedding) = chunked_live_embedding(runtime, text, 2000)? {
        return Ok(embedding);
    }
    match runtime.embed_text(text) {
        Ok(vector) => Ok(vector),
        Err(error) => {
            if let Some(embedding) =
                chunked_live_embedding(runtime, text, (text.len() / 2).max(512))?
            {
                Ok(embedding)
            } else {
                Err(error)
            }
        }
    }
}

fn chunked_live_embedding(
    runtime: &models::LocalModelRuntime,
    text: &str,
    target_chars: usize,
) -> Result<Option<Vec<f32>>> {
    if text.len() <= target_chars {
        return Ok(None);
    }
    let chunks = split_text_for_embedding(text, target_chars);
    if chunks.len() <= 1 {
        return Ok(None);
    }
    let chunk_vectors = chunks
        .iter()
        .map(|chunk| compute_live_embedding(runtime, chunk))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(average_embeddings(&chunk_vectors)))
}

fn split_text_for_embedding(text: &str, target_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        if line.len() > target_chars {
            if !current.is_empty() {
                chunks.push(current);
                current = String::new();
            }
            chunks.extend(split_line_for_embedding(line, target_chars));
            continue;
        }
        let additional = if current.is_empty() {
            line.len()
        } else {
            line.len() + 1
        };
        if !current.is_empty() && current.len() + additional > target_chars {
            chunks.push(current);
            current = String::new();
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    if chunks.is_empty() {
        vec![text.to_string()]
    } else {
        chunks
    }
}

fn split_line_for_embedding(line: &str, target_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for word in line.split_whitespace() {
        let additional = if current.is_empty() {
            word.len()
        } else {
            word.len() + 1
        };
        if !current.is_empty() && current.len() + additional > target_chars {
            chunks.push(current);
            current = String::new();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    if chunks.is_empty() {
        vec![line.chars().take(target_chars).collect()]
    } else {
        chunks
    }
}

fn chunk_document_for_embedding(
    runtime: Option<&models::LocalModelRuntime>,
    content: &str,
) -> Vec<(usize, String)> {
    if let Some(runtime) = runtime.filter(|runtime| runtime.has_embedder()) {
        return token_limited_chunks(runtime, content);
    }
    chunk_document_charwise(
        content,
        EMBED_CHUNK_SIZE_CHARS,
        EMBED_CHUNK_OVERLAP_CHARS,
        EMBED_CHUNK_WINDOW_CHARS,
    )
}

fn chunk_document_for_rerank(content: &str) -> Vec<(usize, String)> {
    chunk_document_charwise(
        content,
        RERANK_CHUNK_SIZE_CHARS,
        RERANK_CHUNK_OVERLAP_CHARS,
        RERANK_CHUNK_WINDOW_CHARS,
    )
}

fn best_chunk_for_query(
    content: &str,
    query_terms: &[String],
    intent_terms: &[String],
) -> (usize, String) {
    let chunks = chunk_document_for_rerank(content);
    if chunks.is_empty() {
        return (0, content.to_string());
    }
    let mut best = 0usize;
    let mut best_score = -1.0f64;
    for (idx, (_pos, text)) in chunks.iter().enumerate() {
        let lower = text.to_ascii_lowercase();
        let mut score = 0.0f64;
        for term in query_terms {
            if lower.contains(term) {
                score += 1.0;
            }
        }
        for term in intent_terms {
            if lower.contains(term) {
                score += 0.5;
            }
        }
        if score > best_score {
            best_score = score;
            best = idx;
        }
    }
    chunks[best].clone()
}

fn token_limited_chunks(
    runtime: &models::LocalModelRuntime,
    content: &str,
) -> Vec<(usize, String)> {
    let char_chunks = chunk_document_charwise(
        content,
        EMBED_CHUNK_SIZE_CHARS,
        EMBED_CHUNK_OVERLAP_CHARS,
        EMBED_CHUNK_WINDOW_CHARS,
    );
    let mut results = Vec::new();
    for (chunk_pos, chunk_text) in char_chunks {
        push_chunk_within_token_limit(runtime, &chunk_text, chunk_pos, &mut results);
    }
    if results.is_empty() {
        vec![(0, content.to_string())]
    } else {
        results
    }
}

fn push_chunk_within_token_limit(
    runtime: &models::LocalModelRuntime,
    text: &str,
    pos: usize,
    results: &mut Vec<(usize, String)>,
) {
    let Ok(tokens) = runtime.tokenize_embedding_text(text) else {
        results.push((pos, text.to_string()));
        return;
    };
    if tokens.len() <= CHUNK_SIZE_TOKENS || text.len() <= 1 {
        results.push((pos, text.to_string()));
        return;
    }

    let actual_chars_per_token = text.len() as f64 / tokens.len() as f64;
    let mut safe_max_chars = (CHUNK_SIZE_TOKENS as f64 * actual_chars_per_token * 0.95).floor();
    if !safe_max_chars.is_finite() || safe_max_chars < 1.0 {
        safe_max_chars = (text.len() / 2) as f64;
    }
    let safe_max_chars = safe_max_chars
        .max(1.0)
        .min((text.len().saturating_sub(1)) as f64) as usize;

    let next_overlap_chars = clamp_overlap_chars(
        (EMBED_CHUNK_OVERLAP_CHARS as f64 * actual_chars_per_token / 2.0).floor() as usize,
        safe_max_chars,
    );
    let next_window_chars =
        ((EMBED_CHUNK_WINDOW_CHARS as f64 * actual_chars_per_token / 2.0).floor() as usize).max(0);
    let mut sub_chunks =
        chunk_document_charwise(text, safe_max_chars, next_overlap_chars, next_window_chars);
    if sub_chunks.len() <= 1 || sub_chunks[0].1.len() == text.len() {
        let half = (text.len() / 2).max(1);
        sub_chunks = chunk_document_charwise(text, half, 0, 0);
    }

    if sub_chunks.len() <= 1 || sub_chunks[0].1.len() == text.len() {
        let fallback_text = runtime
            .detokenize_embedding_tokens(&tokens[..CHUNK_SIZE_TOKENS.min(tokens.len())])
            .unwrap_or_else(|_| text.to_string());
        results.push((pos, fallback_text));
        return;
    }

    for (sub_pos, sub_text) in sub_chunks {
        push_chunk_within_token_limit(runtime, &sub_text, pos + sub_pos, results);
    }
}

fn chunk_document_charwise(
    content: &str,
    max_chars: usize,
    overlap_chars: usize,
    window_chars: usize,
) -> Vec<(usize, String)> {
    if content.len() <= max_chars {
        return vec![(0, content.to_string())];
    }
    let break_points = scan_break_points(content);
    let code_fences = find_code_fences(content);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < content.len() {
        let ideal_end = (start + max_chars).min(content.len());
        let end = if ideal_end == content.len() {
            content.len()
        } else {
            best_chunk_end(
                content,
                &break_points,
                &code_fences,
                start,
                ideal_end,
                window_chars,
            )
        };
        if end <= start {
            break;
        }
        chunks.push((start, content[start..end].to_string()));
        if end == content.len() {
            break;
        }
        let mut next_start = end.saturating_sub(overlap_chars);
        next_start = ceil_char_boundary(content, next_start.min(content.len()));
        if next_start <= start {
            next_start = end;
        }
        start = next_start;
    }
    if chunks.is_empty() {
        vec![(0, content.to_string())]
    } else {
        chunks
    }
}

fn scan_break_points(text: &str) -> Vec<BreakPoint> {
    let mut points = std::collections::BTreeMap::<usize, BreakPoint>::new();
    let mut offset = 0usize;
    let mut prev_blank = false;
    for line in text.split_inclusive('\n') {
        let line_start = offset;
        let breakpoint_pos = line_start.saturating_sub(1);
        let trimmed = line.trim_end_matches('\n').trim();
        if line_start > 0 {
            points.entry(breakpoint_pos).or_insert(BreakPoint {
                pos: breakpoint_pos,
                score: 1.0,
            });

            if prev_blank {
                set_break_point(&mut points, breakpoint_pos, 20.0);
            }

            let heading_score = heading_score(trimmed);
            if heading_score > 0.0 {
                set_break_point(&mut points, breakpoint_pos, heading_score);
            } else if trimmed.starts_with("```") {
                set_break_point(&mut points, breakpoint_pos, 80.0);
            } else if matches!(trimmed, "---" | "***" | "___") {
                set_break_point(&mut points, breakpoint_pos, 60.0);
            } else if trimmed.starts_with("- ")
                || trimmed.starts_with("* ")
                || ordered_list_prefix(trimmed)
            {
                set_break_point(&mut points, breakpoint_pos, 5.0);
            }
        }

        prev_blank = trimmed.is_empty();
        offset += line.len();
    }
    points.into_values().collect()
}

fn set_break_point(
    points: &mut std::collections::BTreeMap<usize, BreakPoint>,
    pos: usize,
    score: f64,
) {
    match points.get(&pos) {
        Some(existing) if existing.score >= score => {}
        _ => {
            points.insert(pos, BreakPoint { pos, score });
        }
    }
}

fn heading_score(trimmed: &str) -> f64 {
    let hashes = trimmed.chars().take_while(|ch| *ch == '#').count();
    match hashes {
        1 => 100.0,
        2 => 90.0,
        3 => 80.0,
        4 => 70.0,
        5 => 60.0,
        6 => 50.0,
        _ => 0.0,
    }
}

fn ordered_list_prefix(trimmed: &str) -> bool {
    let digits = trimmed.chars().take_while(|ch| ch.is_ascii_digit()).count();
    digits > 0 && trimmed[digits..].starts_with(". ")
}

fn find_code_fences(text: &str) -> Vec<CodeFenceRegion> {
    let mut fences = Vec::new();
    let mut in_fence = false;
    let mut fence_start = 0usize;
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        if line.trim_start().starts_with("```") {
            if !in_fence {
                fence_start = offset;
                in_fence = true;
            } else {
                fences.push(CodeFenceRegion {
                    start: fence_start,
                    end: offset + line.len(),
                });
                in_fence = false;
            }
        }
        offset += line.len();
    }
    if in_fence {
        fences.push(CodeFenceRegion {
            start: fence_start,
            end: text.len(),
        });
    }
    fences
}

fn is_inside_code_fence(pos: usize, fences: &[CodeFenceRegion]) -> bool {
    fences
        .iter()
        .any(|fence| pos > fence.start && pos < fence.end)
}

fn best_chunk_end(
    content: &str,
    break_points: &[BreakPoint],
    code_fences: &[CodeFenceRegion],
    start: usize,
    ideal_end: usize,
    window_chars: usize,
) -> usize {
    let window_start = ideal_end.saturating_sub(window_chars);
    let mut best_score = -1.0;
    let mut best_pos = ideal_end;
    for break_point in break_points {
        if break_point.pos < window_start {
            continue;
        }
        if break_point.pos > ideal_end {
            break;
        }
        if break_point.pos <= start || is_inside_code_fence(break_point.pos, code_fences) {
            continue;
        }
        let distance = (ideal_end - break_point.pos) as f64;
        let normalized = distance / window_chars.max(1) as f64;
        let multiplier = 1.0 - (normalized * normalized) * 0.7;
        let final_score = break_point.score * multiplier;
        if final_score > best_score {
            best_score = final_score;
            best_pos = break_point.pos;
        }
    }
    floor_char_boundary(content, best_pos.min(content.len()).max(start + 1))
}

fn clamp_overlap_chars(value: usize, max_chars: usize) -> usize {
    if max_chars <= 1 {
        return 0;
    }
    value.min(max_chars - 1)
}

fn floor_char_boundary(content: &str, mut idx: usize) -> usize {
    idx = idx.min(content.len());
    while idx > 0 && !content.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn ceil_char_boundary(content: &str, mut idx: usize) -> usize {
    idx = idx.min(content.len());
    while idx < content.len() && !content.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

fn parse_positive_integer_option(name: &str, value: &str) -> Result<usize> {
    let parsed = value
        .parse::<usize>()
        .with_context(|| format!("invalid --{name} value"))?;
    if parsed == 0 {
        return Err(anyhow!("--{name} must be a positive integer"));
    }
    Ok(parsed)
}

fn next_embed_batch_end(
    docs: &[(String, String, String)],
    start: usize,
    max_docs_per_batch: usize,
    max_batch_bytes: usize,
) -> usize {
    let mut end = start;
    let mut batch_bytes = 0usize;
    while end < docs.len() && (end - start) < max_docs_per_batch {
        let doc_bytes = docs[end].1.len();
        if end > start && batch_bytes.saturating_add(doc_bytes) > max_batch_bytes {
            break;
        }
        batch_bytes = batch_bytes.saturating_add(doc_bytes);
        end += 1;
    }
    end.max(start + 1)
}

fn average_embeddings(vectors: &[Vec<f32>]) -> Vec<f32> {
    let dim = vectors.first().map(Vec::len).unwrap_or(0);
    if dim == 0 {
        return Vec::new();
    }
    let mut output = vec![0.0f32; dim];
    for vector in vectors {
        for (index, value) in vector.iter().enumerate() {
            output[index] += value;
        }
    }
    let scale = 1.0f32 / vectors.len() as f32;
    for value in &mut output {
        *value *= scale;
    }
    let norm = output.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut output {
            *value /= norm;
        }
    }
    output
}

fn build_plain_query_searches(query: &str, has_vectors: bool) -> Vec<ExpandedQuery> {
    let mut searches = vec![ExpandedQuery {
        kind: QueryKind::Lex,
        query: query.trim().to_string(),
        line: 1,
    }];
    if has_vectors {
        searches.push(ExpandedQuery {
            kind: QueryKind::Vec,
            query: query.trim().to_string(),
            line: 1,
        });
    }
    searches
}

fn parse_structured_query(query: &str) -> Result<Option<ParsedStructuredQuery>> {
    let raw_lines = query
        .split('\n')
        .enumerate()
        .map(|(idx, line)| (idx + 1, line.trim().to_string()))
        .filter(|(_, line)| !line.is_empty())
        .collect::<Vec<_>>();
    if raw_lines.is_empty() {
        return Ok(None);
    }

    let mut searches = Vec::new();
    let mut intent = None;
    for (line_number, line) in &raw_lines {
        let lower = line.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("expand:") {
            if raw_lines.len() > 1 {
                return Err(anyhow!(
                    "Line {} starts with expand:, but query documents cannot mix expand with typed lines. Submit a single expand query instead.",
                    line_number
                ));
            }
            if rest.trim().is_empty() {
                return Err(anyhow!("expand: query must include text."));
            }
            return Ok(None);
        }
        if lower.starts_with("intent:") {
            if intent.is_some() {
                return Err(anyhow!(
                    "Line {}: only one intent: line is allowed per query document.",
                    line_number
                ));
            }
            let text = line["intent:".len()..].trim();
            if text.is_empty() {
                return Err(anyhow!("Line {}: intent: must include text.", line_number));
            }
            intent = Some(text.to_string());
            continue;
        }

        let kind = if lower.starts_with("lex:") {
            QueryKind::Lex
        } else if lower.starts_with("vec:") {
            QueryKind::Vec
        } else if lower.starts_with("hyde:") {
            QueryKind::Hyde
        } else {
            if raw_lines.len() == 1 {
                return Ok(None);
            }
            return Err(anyhow!(
                "Line {} is missing a lex:/vec:/hyde:/intent: prefix. Each line in a query document must start with one.",
                line_number
            ));
        };
        let text = line
            .split_once(':')
            .map(|(_, value)| value.trim())
            .unwrap_or_default();
        if text.is_empty() {
            let label = match kind {
                QueryKind::Lex => "lex",
                QueryKind::Vec => "vec",
                QueryKind::Hyde => "hyde",
            };
            return Err(anyhow!(
                "Line {} ({}:) must include text.",
                line_number,
                label
            ));
        }
        searches.push(ExpandedQuery {
            kind,
            query: text.to_string(),
            line: *line_number,
        });
    }

    if intent.is_some() && searches.is_empty() {
        return Err(anyhow!(
            "intent: cannot appear alone. Add at least one lex:, vec:, or hyde: line."
        ));
    }
    Ok((!searches.is_empty()).then_some(ParsedStructuredQuery { searches, intent }))
}

fn query_terms(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(|term| term.to_ascii_lowercase())
        .collect()
}

fn fallback_fts_query(query: &str) -> Option<String> {
    let tokens = query_terms(query);
    (!tokens.is_empty()).then(|| tokens.join(" "))
}

fn should_retry_fts_query(error: &rusqlite::Error) -> bool {
    let message = error.to_string();
    message.contains("fts5:")
        || message.contains("no such column:")
        || message.contains("unterminated string")
}

fn estimate_query_score(
    item: &SearchResultItem,
    query_terms: &[String],
    intent_terms: &[String],
    rank: usize,
    sources: usize,
) -> f64 {
    let title = item.title.to_ascii_lowercase();
    let snippet = item.snippet.to_ascii_lowercase();
    let title_hit = query_terms.iter().any(|term| title.contains(term));
    let intent_hit = intent_terms
        .iter()
        .any(|term| snippet.contains(term) || title.contains(term));
    let base = 0.85
        + if item.line <= 3 { 0.03 } else { 0.0 }
        + if title_hit { 0.01 } else { 0.0 }
        + if intent_hit { 0.01 } else { 0.0 }
        + if sources > 1 { 0.02 } else { 0.0 }
        - (rank as f64 * 0.02);
    round_score(base)
}

fn round_score(value: f64) -> f64 {
    ((value.clamp(0.0, 0.99) * 100.0).round()) / 100.0
}

fn vector_results(
    connection: &Connection,
    runtime: Option<&models::LocalModelRuntime>,
    index_name: Option<&str>,
    query: &str,
    limit: usize,
    collections: &[String],
) -> Result<Vec<SearchResultItem>> {
    if sqlite_vec::vectors_vec_is_usable(connection)? {
        return vector_results_via_vec0(connection, runtime, index_name, query, limit, collections);
    }
    let formatted_query = models::format_query_for_embedding(query);
    let query_vector = compute_embedding(runtime, &formatted_query)?;
    let rows = connection
        .prepare(
            "SELECT substr(d.hash, 1, 6) AS docid, d.collection, d.path, d.title, c.doc, qv.vector
             FROM documents d
             JOIN content c ON c.hash = d.hash
             JOIN qqd_vectors qv ON qv.hash = d.hash
             WHERE d.active = 1",
        )?
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let mut scored = rows
        .into_iter()
        .filter_map(|(docid, collection, path, title, body, raw_vector)| {
            if !collections.is_empty() && !collections.iter().any(|name| name == &collection) {
                return None;
            }
            let vector: Vec<f32> = serde_json::from_str(&raw_vector).ok()?;
            let similarity = cosine_similarity(&query_vector, &vector);
            if similarity <= 0.0 {
                return None;
            }
            let (line, snippet) = extract_snippet(&body, query);
            Some((
                similarity,
                SearchResultItem {
                    docid: format!("#{docid}"),
                    score: (similarity * 100.0).round() as i64,
                    file: format!("qmd://{collection}/{path}"),
                    context: context_for_path(connection, index_name, &collection, &path)
                        .ok()
                        .flatten(),
                    line,
                    title,
                    snippet,
                    rerank_text: body.chars().take(3000).collect(),
                },
            ))
        })
        .collect::<Vec<_>>();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored
        .into_iter()
        .take(limit)
        .map(|(_, item)| item)
        .collect())
}

fn vector_results_via_vec0(
    connection: &Connection,
    runtime: Option<&models::LocalModelRuntime>,
    index_name: Option<&str>,
    query: &str,
    limit: usize,
    collections: &[String],
) -> Result<Vec<SearchResultItem>> {
    let formatted_query = models::format_query_for_embedding(query);
    let query_vector = compute_embedding(runtime, &formatted_query)?;
    let query_json = serde_json::to_string(&query_vector)?;
    let vec_results = connection
        .prepare(
            "SELECT hash_seq, distance
             FROM vectors_vec
             WHERE embedding MATCH ?1 AND k = ?2",
        )?
        .query_map((query_json.as_str(), (limit * 3) as i64), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if vec_results.is_empty() {
        return Ok(Vec::new());
    }

    let hash_seqs = vec_results
        .iter()
        .map(|(hash_seq, _)| hash_seq)
        .collect::<Vec<_>>();
    let distance_by_hash_seq = vec_results
        .iter()
        .map(|(hash_seq, distance)| (hash_seq.clone(), *distance))
        .collect::<std::collections::BTreeMap<_, _>>();

    let placeholders = hash_seqs
        .iter()
        .enumerate()
        .map(|(index, _)| format!("?{}", index + 1))
        .collect::<Vec<_>>()
        .join(", ");
    let mut sql = format!(
        "SELECT cv.hash || '_' || cv.seq AS hash_seq,
                cv.hash,
                cv.pos,
                d.collection,
                d.path,
                d.title,
                c.doc
         FROM content_vectors cv
         JOIN documents d ON d.hash = cv.hash AND d.active = 1
         JOIN content c ON c.hash = d.hash
         WHERE cv.hash || '_' || cv.seq IN ({placeholders})"
    );
    let mut params = hash_seqs
        .iter()
        .map(|value| value.as_str())
        .collect::<Vec<_>>();
    if !collections.is_empty() {
        sql.push_str(" AND d.collection IN (");
        for idx in 0..collections.len() {
            if idx > 0 {
                sql.push_str(", ");
            }
            sql.push('?');
            sql.push_str(&(params.len() + idx + 1).to_string());
        }
        sql.push(')');
        params.extend(collections.iter().map(String::as_str));
    }
    let rows = connection
        .prepare(&sql)?
        .query_map(rusqlite::params_from_iter(params), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, usize>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let mut best_by_file = std::collections::BTreeMap::<String, (f64, SearchResultItem)>::new();
    for (hash_seq, docid, _pos, collection, path, title, body) in rows {
        let Some(distance) = distance_by_hash_seq.get(&hash_seq).copied() else {
            continue;
        };
        let file = format!("qmd://{collection}/{path}");
        let similarity = 1.0 - distance;
        let (line, snippet) = extract_snippet(&body, query);
        let item = SearchResultItem {
            docid: format!("#{}", &docid[..6.min(docid.len())]),
            score: (similarity * 100.0).round() as i64,
            file: file.clone(),
            context: context_for_path(connection, index_name, &collection, &path)
                .ok()
                .flatten(),
            line,
            title,
            snippet,
            rerank_text: body.chars().take(3000).collect(),
        };
        match best_by_file.get(&file) {
            Some((best_distance, _)) if distance >= *best_distance => {}
            _ => {
                best_by_file.insert(file, (distance, item));
            }
        }
    }

    let mut values = best_by_file.into_values().collect::<Vec<_>>();
    values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(values
        .into_iter()
        .take(limit)
        .map(|(_, item)| item)
        .collect())
}

fn search_results(
    connection: &Connection,
    index_name: Option<&str>,
    query: &str,
    limit: usize,
    collections: &[String],
) -> Result<Vec<SearchResultItem>> {
    fn run_search_query(
        connection: &Connection,
        index_name: Option<&str>,
        match_query: &str,
        original_query: &str,
        limit: usize,
        collections: &[String],
    ) -> rusqlite::Result<Vec<SearchResultItem>> {
        let mut sql = String::from(
            "SELECT substr(documents.hash, 1, 6) AS docid,
                    documents.collection,
                    documents.path,
                    documents.title,
                    content.doc
             FROM documents_fts
             JOIN documents ON documents_fts.rowid = documents.id
             JOIN content ON content.hash = documents.hash
             WHERE documents.active = 1
               AND documents_fts MATCH ?1",
        );

        if !collections.is_empty() {
            sql.push_str(" AND documents.collection IN (");
            for idx in 0..collections.len() {
                if idx > 0 {
                    sql.push_str(", ");
                }
                sql.push('?');
                sql.push_str(&(idx + 2).to_string());
            }
            sql.push(')');
        }
        sql.push_str(" ORDER BY bm25(documents_fts) LIMIT ");
        sql.push_str(&limit.to_string());

        let mut statement = connection.prepare(&sql)?;
        let mut values = vec![match_query];
        values.extend(collections.iter().map(String::as_str));
        let rows = statement.query_map(rusqlite::params_from_iter(values), |row| {
            let docid: String = row.get(0)?;
            let collection: String = row.get(1)?;
            let path: String = row.get(2)?;
            let title: String = row.get(3)?;
            let body: String = row.get(4)?;
            let (line, snippet) = extract_snippet(&body, original_query);
            Ok(SearchResultItem {
                docid: format!("#{docid}"),
                score: 0,
                file: format!("qmd://{collection}/{path}"),
                context: context_for_path(connection, index_name, &collection, &path)
                    .ok()
                    .flatten(),
                line,
                title,
                snippet,
                rerank_text: body.chars().take(3000).collect(),
            })
        })?;

        rows.collect::<rusqlite::Result<Vec<_>>>()
    }

    match run_search_query(connection, index_name, query, query, limit, collections) {
        Ok(results) => Ok(results),
        Err(error) if should_retry_fts_query(&error) => {
            let Some(fallback_query) = fallback_fts_query(query) else {
                return Err(error.into());
            };
            Ok(run_search_query(
                connection,
                index_name,
                &fallback_query,
                query,
                limit,
                collections,
            )?)
        }
        Err(error) => Err(error.into()),
    }
}

fn embed_text(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0f32; VECTOR_DIMENSIONS];
    for token in text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
    {
        let token = token.to_ascii_lowercase();
        let mut hash = 1469598103934665603u64;
        for byte in token.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(1099511628211);
        }
        let idx = (hash as usize) % VECTOR_DIMENSIONS;
        vector[idx] += 1.0;
    }

    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn extract_snippet(body: &str, query: &str) -> (usize, String) {
    let search_terms = query
        .split_whitespace()
        .map(|term| term.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let lines = body.split('\n').collect::<Vec<_>>();
    let total_lines = lines.len();
    for (index, line) in lines.iter().enumerate() {
        let lower = line.to_ascii_lowercase();
        if search_terms.iter().any(|term| lower.contains(term)) {
            let start = index.saturating_sub(1);
            let end = (index + 3).min(lines.len());
            let snippet_lines = &lines[start..end];
            let snippet_body = snippet_lines.join("\n");
            return (
                index + 1,
                format!(
                    "@@ -{},{} @@ ({} before, {} after)\n{}",
                    start + 1,
                    snippet_lines.len(),
                    start,
                    total_lines.saturating_sub(start + snippet_lines.len()),
                    snippet_body
                ),
            );
        }
    }

    (
        1,
        "@@ -1,1 @@ (0 before, 0 after)\n\n".to_string()
            + lines.first().copied().unwrap_or("")
            + "\n",
    )
}

pub fn open_readonly(index_name: Option<&str>) -> Result<Connection> {
    let db_path = default_db_path(index_name)?;
    Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .context("failed to open qmd index")
}

fn open_readwrite(index_name: Option<&str>) -> Result<Connection> {
    let db_path = default_db_path(index_name)?;
    Connection::open(db_path).context("failed to open qmd index")
}

#[derive(Debug, Default)]
struct NativeReindexResult {
    indexed: usize,
    updated: usize,
    unchanged: usize,
    removed: usize,
}

#[derive(Debug, Default)]
struct ReindexTrace {
    files_seen: usize,
    bytes_read: usize,
    walk_ms: f64,
    read_ms: f64,
    hash_ms: f64,
    title_ms: f64,
    metadata_ms: f64,
    content_insert_ms: f64,
    existing_lookup_ms: f64,
    update_doc_ms: f64,
    insert_doc_ms: f64,
    existing_paths_ms: f64,
    deactivate_ms: f64,
    tx_commit_ms: f64,
    slowest_file_path: String,
    slowest_file_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Default)]
struct EmbedTrace {
    docs_seen: usize,
    batches: usize,
    runtime_init_ms: f64,
    docs_query_ms: f64,
    compute_embeddings_ms: f64,
    serialize_vectors_ms: f64,
    qqd_vector_write_ms: f64,
    content_vector_delete_ms: f64,
    content_vector_insert_ms: f64,
    tx_commit_ms: f64,
    slowest_batch_index: usize,
    slowest_batch_docs: usize,
    slowest_batch_ms: f64,
    total_ms: f64,
}

fn trace_indexing_enabled() -> bool {
    env::var("QQD_TRACE_INDEXING")
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn ensure_schema(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        r#"
        PRAGMA journal_mode = WAL;
        CREATE TABLE IF NOT EXISTS content (
          hash TEXT PRIMARY KEY,
          doc TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          collection TEXT NOT NULL,
          path TEXT NOT NULL,
          title TEXT NOT NULL,
          hash TEXT NOT NULL,
          created_at TEXT NOT NULL,
          modified_at TEXT NOT NULL,
          active INTEGER NOT NULL DEFAULT 1,
          UNIQUE(collection, path)
        );
        CREATE TABLE IF NOT EXISTS llm_cache (
          hash TEXT PRIMARY KEY,
          result TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS content_vectors (
          hash TEXT NOT NULL,
          seq INTEGER NOT NULL DEFAULT 0,
          pos INTEGER NOT NULL DEFAULT 0,
          model TEXT NOT NULL,
          embedded_at TEXT NOT NULL,
          PRIMARY KEY (hash, seq)
        );
        CREATE TABLE IF NOT EXISTS store_collections (
          name TEXT PRIMARY KEY,
          path TEXT NOT NULL,
          pattern TEXT NOT NULL DEFAULT '**/*.md',
          ignore_patterns TEXT,
          include_by_default INTEGER DEFAULT 1,
          update_command TEXT,
          context TEXT
        );
        CREATE TABLE IF NOT EXISTS store_config (
          key TEXT PRIMARY KEY,
          value TEXT
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(filepath, title, body, tokenize='porter unicode61');
        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents
        WHEN new.active = 1
        BEGIN
          INSERT INTO documents_fts(rowid, filepath, title, body)
          SELECT
            new.id,
            new.collection || '/' || new.path,
            new.title,
            (SELECT doc FROM content WHERE hash = new.hash)
          WHERE new.active = 1;
        END;
        CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
          DELETE FROM documents_fts WHERE rowid = old.id;
        END;
        CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents
        BEGIN
          DELETE FROM documents_fts WHERE rowid = old.id;
          INSERT INTO documents_fts(rowid, filepath, title, body)
          SELECT
            new.id,
            new.collection || '/' || new.path,
            new.title,
            (SELECT doc FROM content WHERE hash = new.hash)
          WHERE new.active = 1;
        END;
        "#,
    )?;
    Ok(())
}

fn ensure_qqd_vectors_schema(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS qqd_vectors (
          hash TEXT PRIMARY KEY,
          vector TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        "#,
    )?;
    Ok(())
}

fn get_config_dir() -> PathBuf {
    if let Ok(dir) = env::var("QMD_CONFIG_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = env::var("XDG_CONFIG_HOME") {
        return PathBuf::from(dir).join("qmd");
    }
    PathBuf::from(env::var("HOME").unwrap_or_else(|_| ".".to_string()))
        .join(".config")
        .join("qmd")
}

fn get_config_path(index_name: Option<&str>) -> PathBuf {
    let name = index_name.unwrap_or("index");
    get_config_dir().join(format!("{name}.yml"))
}

pub fn load_file_config(index_name: Option<&str>) -> Result<FileConfig> {
    let path = get_config_path(index_name);
    if !path.exists() {
        return Ok(FileConfig::default());
    }
    Ok(serde_yaml::from_str(&fs::read_to_string(path)?)?)
}

fn save_file_config(index_name: Option<&str>, config: &FileConfig) -> Result<()> {
    let path = get_config_path(index_name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_yaml::to_string(config)?)?;
    Ok(())
}

fn upsert_store_collection(
    connection: &Connection,
    name: &str,
    path: &str,
    pattern: &str,
) -> Result<()> {
    connection.execute(
        "INSERT INTO store_collections (name, path, pattern)
         VALUES (?1, ?2, ?3)
         ON CONFLICT(name) DO UPDATE SET path = excluded.path, pattern = excluded.pattern",
        (name, path, pattern),
    )?;
    Ok(())
}

fn reindex_collection_native(
    connection: &mut Connection,
    collection_path: &str,
    pattern: &str,
    collection_name: &str,
) -> Result<NativeReindexResult> {
    let trace_enabled = trace_indexing_enabled();
    let total_start = Instant::now();
    let mut trace = ReindexTrace::default();
    let mut result = NativeReindexResult::default();
    let mut seen_paths = std::collections::HashSet::new();
    let now = current_timestamp();
    let walk_start = Instant::now();
    let files = walk_markdown_files(collection_path, pattern)?;
    trace.walk_ms = elapsed_ms(walk_start);
    let tx = connection.transaction()?;

    for (relative_path, absolute_path) in files {
        let file_start = Instant::now();
        trace.files_seen += 1;
        seen_paths.insert(relative_path.clone());
        let read_start = Instant::now();
        let body = fs::read_to_string(&absolute_path)?;
        trace.read_ms += elapsed_ms(read_start);
        trace.bytes_read += body.len();
        if body.trim().is_empty() {
            continue;
        }
        let hash_start = Instant::now();
        let hash = hash_content_sync(&body);
        trace.hash_ms += elapsed_ms(hash_start);
        let title_start = Instant::now();
        let title = extract_title(&body, &relative_path);
        trace.title_ms += elapsed_ms(title_start);
        let metadata_start = Instant::now();
        let metadata = fs::metadata(&absolute_path)?;
        let modified_at =
            current_timestamp_from(metadata.modified().ok()).unwrap_or_else(|| now.clone());
        let created_at =
            current_timestamp_from(metadata.created().ok()).unwrap_or_else(|| modified_at.clone());
        trace.metadata_ms += elapsed_ms(metadata_start);

        let content_insert_start = Instant::now();
        tx.execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?1, ?2, ?3)",
            (&hash, &body, &now),
        )?;
        trace.content_insert_ms += elapsed_ms(content_insert_start);

        let existing_lookup_start = Instant::now();
        let existing = tx
            .prepare(
                "SELECT id, hash, title FROM documents
                 WHERE collection = ?1 AND path = ?2 AND active = 1",
            )?
            .query_row((collection_name, relative_path.as_str()), |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .optional()?;
        trace.existing_lookup_ms += elapsed_ms(existing_lookup_start);

        match existing {
            Some((_id, existing_hash, existing_title))
                if existing_hash == hash && existing_title == title =>
            {
                result.unchanged += 1;
            }
            Some((id, _, _)) => {
                let update_doc_start = Instant::now();
                tx.execute(
                    "UPDATE documents SET title = ?1, hash = ?2, modified_at = ?3 WHERE id = ?4",
                    (&title, &hash, &modified_at, id),
                )?;
                trace.update_doc_ms += elapsed_ms(update_doc_start);
                result.updated += 1;
            }
            None => {
                let insert_doc_start = Instant::now();
                tx.execute(
                    "INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1)
                     ON CONFLICT(collection, path) DO UPDATE SET
                       title = excluded.title,
                       hash = excluded.hash,
                       modified_at = excluded.modified_at,
                       active = 1",
                    (collection_name, relative_path.as_str(), &title, &hash, &created_at, &modified_at),
                )?;
                trace.insert_doc_ms += elapsed_ms(insert_doc_start);
                result.indexed += 1;
            }
        }

        let file_ms = elapsed_ms(file_start);
        if file_ms > trace.slowest_file_ms {
            trace.slowest_file_ms = file_ms;
            trace.slowest_file_path = relative_path;
        }
    }

    let existing_paths_start = Instant::now();
    let existing_paths = tx
        .prepare("SELECT path FROM documents WHERE collection = ?1 AND active = 1")?
        .query_map([collection_name], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    trace.existing_paths_ms += elapsed_ms(existing_paths_start);
    let deactivate_start = Instant::now();
    for path in existing_paths {
        if !seen_paths.contains(&path) {
            tx.execute(
                "UPDATE documents SET active = 0 WHERE collection = ?1 AND path = ?2",
                (collection_name, path.as_str()),
            )?;
            result.removed += 1;
        }
    }
    trace.deactivate_ms += elapsed_ms(deactivate_start);
    let tx_commit_start = Instant::now();
    tx.commit()?;
    trace.tx_commit_ms = elapsed_ms(tx_commit_start);
    trace.total_ms = elapsed_ms(total_start);

    if trace_enabled {
        let per_file_divisor = trace.files_seen.max(1) as f64;
        eprintln!(
            "QQD_INDEX_TRACE {}",
            serde_json::to_string(&json!({
                "collection": collection_name,
                "files_seen": trace.files_seen,
                "bytes_read": trace.bytes_read,
                "result": {
                    "indexed": result.indexed,
                    "updated": result.updated,
                    "unchanged": result.unchanged,
                    "removed": result.removed,
                },
                "phase_ms": {
                    "walk": trace.walk_ms,
                    "read": trace.read_ms,
                    "hash": trace.hash_ms,
                    "title": trace.title_ms,
                    "metadata": trace.metadata_ms,
                    "content_insert": trace.content_insert_ms,
                    "existing_lookup": trace.existing_lookup_ms,
                    "update_doc": trace.update_doc_ms,
                    "insert_doc": trace.insert_doc_ms,
                    "existing_paths": trace.existing_paths_ms,
                    "deactivate": trace.deactivate_ms,
                    "tx_commit": trace.tx_commit_ms,
                    "total": trace.total_ms,
                },
                "per_file_ms": {
                    "read": trace.read_ms / per_file_divisor,
                    "hash": trace.hash_ms / per_file_divisor,
                    "title": trace.title_ms / per_file_divisor,
                    "metadata": trace.metadata_ms / per_file_divisor,
                    "content_insert": trace.content_insert_ms / per_file_divisor,
                    "existing_lookup": trace.existing_lookup_ms / per_file_divisor,
                },
                "fts": "maintained by documents_ai/documents_ad/documents_au triggers",
                "slowest_file": {
                    "path": trace.slowest_file_path,
                    "ms": trace.slowest_file_ms,
                },
            }))?
        );
    }

    Ok(result)
}

fn hash_content_sync(content: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn extract_title(content: &str, relative_path: &str) -> String {
    content
        .lines()
        .find_map(|line| line.strip_prefix("# ").map(str::trim))
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| {
            std::path::Path::new(relative_path)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or(relative_path)
                .to_string()
        })
}

fn walk_markdown_files(root: &str, pattern: &str) -> Result<Vec<(String, PathBuf)>> {
    let root_path = PathBuf::from(root);
    let mut files = Vec::new();
    collect_files(&root_path, &root_path, pattern, &mut files)?;
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

fn collect_files(
    root: &PathBuf,
    current: &PathBuf,
    pattern: &str,
    output: &mut Vec<(String, PathBuf)>,
) -> Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect_files(root, &path, pattern, output)?;
        } else {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            if glob_match(pattern, &relative) {
                output.push((relative, path));
            }
        }
    }
    Ok(())
}

fn current_timestamp() -> String {
    current_timestamp_from(Some(std::time::SystemTime::now()))
        .unwrap_or_else(|| "1970-01-01T00:00:00.000Z".to_string())
}

fn current_timestamp_from(time: Option<std::time::SystemTime>) -> Option<String> {
    let time = time?;
    let datetime: chrono::DateTime<chrono::Utc> = time.into();
    Some(datetime.to_rfc3339())
}

fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

fn format_ls_time(connection: &Connection, modified_at: &str) -> Result<String> {
    let (month_num, day, time, year, old) = connection.query_row(
        "SELECT
            CAST(strftime('%m', ?1) AS INTEGER),
            strftime('%d', ?1),
            strftime('%H:%M', ?1),
            strftime('%Y', ?1),
            CASE
              WHEN julianday('now') - julianday(?1) > 180 THEN 1
              ELSE 0
            END",
        [modified_at],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
            ))
        },
    )?;
    let month = match month_num {
        1 => "Jan",
        2 => "Feb",
        3 => "Mar",
        4 => "Apr",
        5 => "May",
        6 => "Jun",
        7 => "Jul",
        8 => "Aug",
        9 => "Sep",
        10 => "Oct",
        11 => "Nov",
        12 => "Dec",
        _ => "???",
    };
    if old != 0 {
        Ok(format!("{month} {}  {year}", day.trim_start()))
    } else {
        Ok(format!("{month} {} {time}", day.trim_start()))
    }
}

fn format_time_ago(connection: &Connection, modified_at: &str) -> Result<String> {
    let seconds: i64 = connection.query_row(
        "SELECT CAST((julianday('now') - julianday(?1)) * 86400 AS INTEGER)",
        [modified_at],
        |row| row.get(0),
    )?;
    if seconds < 60 {
        return Ok(format!("{}s ago", seconds.max(0)));
    }
    let minutes = seconds / 60;
    if minutes < 60 {
        return Ok(format!("{minutes}m ago"));
    }
    let hours = minutes / 60;
    if hours < 24 {
        return Ok(format!("{hours}h ago"));
    }
    Ok(format!("{}d ago", hours / 24))
}

pub fn split_index_args(args: &[String]) -> Result<(Option<String>, Vec<String>)> {
    let mut cursor = 0usize;
    let mut index_name = None;
    let mut rest = Vec::new();

    while cursor < args.len() {
        match args[cursor].as_str() {
            "--index" => {
                let value = args
                    .get(cursor + 1)
                    .ok_or_else(|| anyhow!("--index requires a value"))?;
                index_name = Some(value.clone());
                cursor += 2;
            }
            _ => {
                rest.extend_from_slice(&args[cursor..]);
                break;
            }
        }
    }

    Ok((index_name, rest))
}

#[derive(Debug, Serialize)]
pub struct McpDocument {
    pub file: String,
    pub title: String,
    pub body: String,
}

#[derive(Debug, Serialize)]
pub struct McpStatus {
    #[serde(rename = "totalDocuments")]
    pub total_documents: i64,
    #[serde(rename = "needsEmbedding")]
    pub needs_embedding: i64,
    #[serde(rename = "hasVectorIndex")]
    pub has_vector_index: bool,
    pub collections: Vec<McpStatusCollection>,
}

#[derive(Debug, Serialize)]
pub struct McpStatusCollection {
    pub name: String,
    pub path: String,
    pub pattern: String,
    pub documents: i64,
    #[serde(rename = "lastUpdated")]
    pub last_updated: String,
}

pub fn get_document_for_mcp(
    index_name: Option<&str>,
    target: &str,
    from_line: Option<usize>,
    max_lines: Option<usize>,
    line_numbers: bool,
) -> Result<McpDocument> {
    let connection = open_readonly(index_name)?;
    let doc = resolve_document(&connection, target)?
        .ok_or_else(|| anyhow!("Document not found: {}", target))?;
    let mut body = doc.body;
    let start_line = from_line.unwrap_or(1);
    if from_line.is_some() || max_lines.is_some() {
        let lines = body.lines().collect::<Vec<_>>();
        let start = start_line.saturating_sub(1);
        let end = max_lines
            .map(|max_lines| start.saturating_add(max_lines))
            .unwrap_or(lines.len());
        body = lines
            .iter()
            .skip(start)
            .take(end.saturating_sub(start))
            .copied()
            .collect::<Vec<_>>()
            .join("\n");
    }
    if line_numbers {
        body = add_line_numbers(&body, start_line);
    }
    if let Some(context) = context_for_path(&connection, index_name, &doc.collection, &doc.path)? {
        body = format!("<!-- Context: {} -->\n\n{}", context, body);
    }
    Ok(McpDocument {
        file: format!("{}/{}", doc.collection, doc.path),
        title: doc.title,
        body,
    })
}

pub fn multi_get_for_mcp(
    index_name: Option<&str>,
    pattern: &str,
    max_lines: Option<usize>,
    max_bytes: Option<usize>,
    line_numbers: bool,
) -> Result<Vec<serde_json::Value>> {
    let connection = open_readonly(index_name)?;
    let entries = resolve_multi_get_entries(&connection, pattern)?;
    let max_bytes = max_bytes.unwrap_or(10 * 1024);
    Ok(entries
        .into_iter()
        .map(|entry| {
            let display_path = entry.display_path.clone();
            let (collection, path) =
                resolve_display_path_to_collection_path(&connection, &display_path)
                    .unwrap_or_else(|| ("".to_string(), display_path.clone()));
            if entry.body.len() > max_bytes {
                return json!({
                    "type":"text",
                    "text": format!("[SKIPPED: {} - File too large]", display_path)
                });
            }
            let mut body = entry.body;
            if let Some(max_lines) = max_lines {
                let lines = body.lines().map(ToOwned::to_owned).collect::<Vec<_>>();
                body = lines
                    .iter()
                    .take(max_lines)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n");
            }
            if line_numbers {
                body = add_line_numbers(&body, 1);
            }
            if let Ok(Some(context)) = context_for_path(&connection, index_name, &collection, &path)
            {
                body = format!("<!-- Context: {} -->\n\n{}", context, body);
            }
            json!({
                    "type": "resource",
                    "resource": {
                    "uri": format!("qmd://{}/{}", collection, encode_qmd_path(&path)),
                    "name": format!("{}/{}", collection, path),
                    "title": entry.title,
                    "mimeType": "text/markdown",
                    "text": body
                }
            })
        })
        .collect())
}

pub fn status_for_mcp(index_name: Option<&str>) -> Result<McpStatus> {
    let connection = open_readonly(index_name)?;
    let total_docs: i64 = connection.query_row(
        "SELECT COUNT(*) FROM documents WHERE active = 1",
        [],
        |row| row.get(0),
    )?;
    let needs_embedding = needs_embedding_count(&connection)?;
    let collections = connection
        .prepare(
            "SELECT sc.name, sc.path, sc.pattern, COUNT(d.id) AS documents, COALESCE(MAX(d.modified_at), 'never')
             FROM store_collections sc
             LEFT JOIN documents d ON d.collection = sc.name AND d.active = 1
             GROUP BY sc.name, sc.path, sc.pattern
             ORDER BY sc.name",
        )?
        .query_map([], |row| {
            Ok(McpStatusCollection {
                name: row.get(0)?,
                path: row.get(1)?,
                pattern: row.get(2)?,
                documents: row.get(3)?,
                last_updated: row.get(4)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(McpStatus {
        total_documents: total_docs,
        needs_embedding,
        has_vector_index: has_native_vectors(&connection)? || has_qmd_vector_state(&connection)?,
        collections,
    })
}

fn detect_context_target(
    connection: &Connection,
    maybe_path: Option<&str>,
) -> Result<(String, String)> {
    if let Some(path) = maybe_path {
        if let Some((collection, path)) = parse_virtual_path(path) {
            ensure_collection_exists(connection, &collection)?;
            return Ok((collection, path));
        }

        let fs_path = resolve_context_fs_path(path)?;
        if let Some(detected) = detect_collection_from_path(connection, &fs_path)? {
            return Ok(detected);
        }
        return Err(anyhow!(
            "Path is not in any indexed collection: {}",
            fs_path
        ));
    }

    let fs_path = resolve_context_fs_path(".")?;
    detect_collection_from_path(connection, &fs_path)?
        .ok_or_else(|| anyhow!("Path is not in any indexed collection: {}", fs_path))
}

fn resolve_context_fs_path(path: &str) -> Result<String> {
    let cwd = env::current_dir()
        .unwrap_or_else(|_| PathBuf::from(env::var("PWD").unwrap_or_else(|_| ".".to_string())));
    let path = if path == "." || path == "./" {
        cwd.display().to_string()
    } else if let Some(stripped) = path.strip_prefix("~/") {
        format!("{}/{}", env::var("HOME").unwrap_or_default(), stripped)
    } else if path.starts_with('/') {
        path.to_string()
    } else {
        cwd.join(path).display().to_string()
    };
    Ok(std::fs::canonicalize(&path)
        .unwrap_or_else(|_| std::path::PathBuf::from(path))
        .display()
        .to_string())
}

fn detect_collection_from_path(
    connection: &Connection,
    fs_path: &str,
) -> Result<Option<(String, String)>> {
    let rows = connection
        .prepare("SELECT name, path FROM store_collections ORDER BY LENGTH(path) DESC")?
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for (collection, root) in rows {
        let root = std::fs::canonicalize(&root)
            .unwrap_or_else(|_| std::path::PathBuf::from(&root))
            .display()
            .to_string();
        if fs_path == root {
            return Ok(Some((collection, String::new())));
        }
        let prefix = format!("{}/", root.trim_end_matches('/'));
        if let Some(relative) = fs_path.strip_prefix(&prefix) {
            return Ok(Some((collection, relative.replace('\\', "/"))));
        }
    }
    Ok(None)
}

fn ensure_collection_exists(connection: &Connection, collection: &str) -> Result<()> {
    let exists = connection
        .prepare("SELECT 1 FROM store_collections WHERE name = ?1 LIMIT 1")?
        .exists([collection])?;
    if exists {
        Ok(())
    } else {
        Err(anyhow!("Collection not found: {}", collection))
    }
}

fn load_context_map(
    connection: &Connection,
    collection: &str,
) -> Result<std::collections::BTreeMap<String, String>> {
    let raw = connection
        .prepare("SELECT context FROM store_collections WHERE name = ?1")?
        .query_row([collection], |row| row.get::<_, Option<String>>(0))
        .optional()?
        .flatten();
    Ok(match raw {
        Some(raw) => serde_json::from_str(&raw)?,
        None => std::collections::BTreeMap::new(),
    })
}

fn save_context_map(
    connection: &Connection,
    collection: &str,
    context: &std::collections::BTreeMap<String, String>,
) -> Result<()> {
    let raw = if context.is_empty() {
        None
    } else {
        Some(serde_json::to_string(context)?)
    };
    connection
        .prepare("UPDATE store_collections SET context = ?1 WHERE name = ?2")?
        .execute((raw, collection))?;
    Ok(())
}

pub fn table_exists(connection: &Connection, table: &str) -> Result<bool> {
    Ok(connection
        .prepare(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?1 LIMIT 1",
        )?
        .exists([table])?)
}

fn split_line_suffix(target: &str) -> Option<(String, usize)> {
    let (path, line) = target.rsplit_once(':')?;
    if path.starts_with('#') {
        return None;
    }
    let line = line.parse().ok()?;
    Some((path.to_string(), line))
}

#[derive(Debug)]
struct ResolvedDocument {
    collection: String,
    path: String,
    title: String,
    body: String,
}

fn resolve_document(connection: &Connection, target: &str) -> Result<Option<ResolvedDocument>> {
    if is_docid(target) {
        return query_document(
            connection,
            "WHERE lower(substr(d.hash, 1, 6)) = lower(?1) AND d.active = 1 LIMIT 1",
            &[target.trim_start_matches('#')],
        );
    }

    if let Some((collection, path)) = parse_virtual_path(target) {
        return query_document(
            connection,
            "WHERE d.collection = ?1 AND d.path = ?2 AND d.active = 1 LIMIT 1",
            &[&collection, &path],
        );
    }

    if let Some((collection, path)) = split_collection_path(connection, target)? {
        if let Some(doc) = query_document(
            connection,
            "WHERE d.collection = ?1 AND d.path = ?2 AND d.active = 1 LIMIT 1",
            &[&collection, &path],
        )? {
            return Ok(Some(doc));
        }
    }

    if let Some(doc) = query_document(
        connection,
        "WHERE d.path = ?1 AND d.active = 1 LIMIT 1",
        &[target],
    )? {
        return Ok(Some(doc));
    }

    query_document(
        connection,
        "WHERE d.path LIKE ?1 AND d.active = 1 LIMIT 1",
        &[&format!("%{target}")],
    )
}

fn query_document(
    connection: &Connection,
    where_clause: &str,
    params: &[&str],
) -> Result<Option<ResolvedDocument>> {
    let sql = format!(
        "SELECT d.collection, d.path, d.title, content.doc
         FROM documents d
         JOIN content ON content.hash = d.hash
         {where_clause}"
    );
    let mut statement = connection.prepare(&sql)?;
    let doc = statement
        .query_row(rusqlite::params_from_iter(params.iter().copied()), |row| {
            Ok(ResolvedDocument {
                collection: row.get(0)?,
                path: row.get(1)?,
                title: row.get(2)?,
                body: row.get(3)?,
            })
        })
        .optional()?;
    Ok(doc)
}

fn parse_virtual_path(path: &str) -> Option<(String, String)> {
    let rest = path.strip_prefix("qmd://")?;
    let (collection, path) = rest.split_once('/')?;
    Some((collection.to_string(), decode_qmd_path(path)))
}

fn split_collection_path(
    connection: &Connection,
    target: &str,
) -> Result<Option<(String, String)>> {
    let Some((collection, path)) = target.split_once('/') else {
        return Ok(None);
    };
    let exists = connection
        .prepare("SELECT 1 FROM store_collections WHERE name = ?1 LIMIT 1")?
        .exists([collection])?;
    Ok(exists.then(|| (collection.to_string(), path.to_string())))
}

fn is_docid(target: &str) -> bool {
    let value = target.trim_start_matches('#');
    value.len() == 6 && value.chars().all(|char| char.is_ascii_hexdigit())
}

pub fn add_line_numbers(body: &str, start_line: usize) -> String {
    body.lines()
        .enumerate()
        .map(|(index, line)| format!("{}: {}", start_line + index, line))
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug)]
struct MultiGetEntry {
    display_path: String,
    title: String,
    body: String,
}

#[derive(Debug)]
struct MultiGetResult {
    display_path: String,
    title: String,
    body: String,
    skipped: bool,
    reason: Option<String>,
}

fn resolve_multi_get_entries(connection: &Connection, pattern: &str) -> Result<Vec<MultiGetEntry>> {
    let rows = connection
        .prepare(
            "SELECT d.collection, d.path, d.title, content.doc
             FROM documents d
             JOIN content ON content.hash = d.hash
             WHERE d.active = 1
             ORDER BY d.path",
        )?
        .query_map([], |row| {
            Ok(ResolvedDocument {
                collection: row.get(0)?,
                path: row.get(1)?,
                title: row.get(2)?,
                body: row.get(3)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    if is_comma_separated(pattern) {
        let names = pattern
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>();
        let mut results = Vec::new();
        for name in names {
            if let Some(doc) = rows.iter().find(|doc| multi_get_name_matches(doc, name)) {
                results.push(MultiGetEntry {
                    display_path: doc.path.clone(),
                    title: doc.title.clone(),
                    body: doc.body.clone(),
                });
            }
        }
        return Ok(results);
    }

    Ok(rows
        .into_iter()
        .filter(|doc| glob_match(pattern, &format!("{}/{}", doc.collection, doc.path)))
        .map(|doc| MultiGetEntry {
            display_path: doc.path,
            title: doc.title,
            body: doc.body,
        })
        .collect())
}

fn is_comma_separated(pattern: &str) -> bool {
    pattern.contains(',')
        && !pattern.contains('*')
        && !pattern.contains('?')
        && !pattern.contains('{')
}

fn multi_get_name_matches(doc: &ResolvedDocument, name: &str) -> bool {
    if is_docid(name) {
        return false;
    }
    if let Some((collection, path)) = parse_virtual_path(name) {
        return doc.collection == collection && doc.path == path;
    }
    doc.path == name || doc.path.ends_with(name)
}

fn resolve_display_path_to_collection_path(
    connection: &Connection,
    display_path: &str,
) -> Option<(String, String)> {
    connection
        .prepare("SELECT collection, path FROM documents WHERE path = ?1 AND active = 1 LIMIT 1")
        .ok()?
        .query_row([display_path], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .optional()
        .ok()
        .flatten()
}

pub fn encode_qmd_path(path: &str) -> String {
    path.split('/')
        .map(encode_qmd_segment)
        .collect::<Vec<_>>()
        .join("/")
}

pub fn decode_qmd_path(path: &str) -> String {
    path.split('/')
        .map(|segment| percent_decode_str(segment).decode_utf8_lossy().to_string())
        .collect::<Vec<_>>()
        .join("/")
}

fn encode_qmd_segment(segment: &str) -> String {
    let mut output = String::new();
    for &byte in segment.as_bytes() {
        let ch = byte as char;
        if ch.is_ascii_alphanumeric()
            || matches!(ch, '-' | '_' | '.' | '!' | '~' | '*' | '\'' | '(' | ')')
        {
            output.push(ch);
        } else {
            output.push_str(&format!("%{:02X}", byte));
        }
    }
    output
}

fn glob_match(pattern: &str, value: &str) -> bool {
    let pattern_segments = pattern.split('/').collect::<Vec<_>>();
    let value_segments = value.split('/').collect::<Vec<_>>();
    match_segments(&pattern_segments, &value_segments)
}

fn match_segments(pattern: &[&str], value: &[&str]) -> bool {
    if pattern.is_empty() {
        return value.is_empty();
    }
    if pattern[0] == "**" {
        return (0..=value.len()).any(|index| match_segments(&pattern[1..], &value[index..]));
    }
    if value.is_empty() {
        return false;
    }
    segment_match(pattern[0], value[0]) && match_segments(&pattern[1..], &value[1..])
}

fn segment_match(pattern: &str, value: &str) -> bool {
    let pattern = pattern.as_bytes();
    let value = value.as_bytes();
    let mut p = 0usize;
    let mut v = 0usize;
    let mut star = None;
    let mut match_from = 0usize;

    while v < value.len() {
        if p < pattern.len() && (pattern[p] == value[v] || pattern[p] == b'?') {
            p += 1;
            v += 1;
        } else if p < pattern.len() && pattern[p] == b'*' {
            star = Some(p);
            match_from = v;
            p += 1;
        } else if let Some(star_index) = star {
            p = star_index + 1;
            match_from += 1;
            v = match_from;
        } else {
            return false;
        }
    }

    while p < pattern.len() && pattern[p] == b'*' {
        p += 1;
    }
    p == pattern.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use std::path::PathBuf;

    #[test]
    fn rejects_complex_queries_for_native_path() {
        let parsed =
            SearchConfig::parse(&["search".into(), "\"exact phrase\"".into(), "--json".into()])
                .unwrap();
        assert!(parsed.is_none());
    }

    #[test]
    fn parses_supported_native_search() {
        let parsed = SearchConfig::parse(&[
            "--index".into(),
            "bench".into(),
            "search".into(),
            "--json".into(),
            "-n".into(),
            "3".into(),
            "-c".into(),
            "docs".into(),
            "rust".into(),
        ])
        .unwrap()
        .expect("native config");

        assert_eq!(parsed.index_name.as_deref(), Some("bench"));
        assert_eq!(parsed.limit, 3);
        assert_eq!(parsed.collections, vec!["docs"]);
        assert_eq!(parsed.query, "rust");
        assert!(parsed.json);
    }

    #[test]
    fn parses_get_command_with_line_suffix() {
        let parsed = GetConfig::parse(&["get".into(), "docs/a.md:5".into()])
            .unwrap()
            .expect("get config");
        assert_eq!(parsed.target, "docs/a.md");
        assert_eq!(parsed.from_line, Some(5));
    }

    #[test]
    fn matches_segment_globs_without_crossing_directories() {
        assert!(glob_match("docs/*.md", "docs/a.md"));
        assert!(!glob_match("docs/*.md", "docs/sub/b.md"));
        assert!(glob_match("docs/**", "docs/sub/b.md"));
    }

    #[test]
    fn parses_ls_command() {
        let parsed = LsConfig::parse(&["ls".into(), "docs".into()])
            .unwrap()
            .expect("ls config");
        assert_eq!(parsed.path.as_deref(), Some("docs"));
    }

    #[test]
    fn parses_status_command() {
        let parsed = StatusConfig::parse(&["status".into()])
            .unwrap()
            .expect("status config");
        assert_eq!(parsed.index_name, None);
    }

    #[test]
    fn parses_collection_list_command() {
        let parsed = CollectionListConfig::parse(&["collection".into(), "list".into()])
            .unwrap()
            .expect("collection list config");
        assert_eq!(parsed.index_name, None);
    }

    #[test]
    fn parses_collection_show_command() {
        let parsed =
            CollectionShowConfig::parse(&["collection".into(), "show".into(), "docs".into()])
                .unwrap()
                .expect("collection show config");
        assert_eq!(parsed.name, "docs");
    }

    #[test]
    fn parses_vsearch_command() {
        let parsed = VSearchConfig::parse(&["vsearch".into(), "--json".into(), "rust".into()])
            .unwrap()
            .expect("vsearch config");
        assert!(parsed.json);
        assert_eq!(parsed.query, "rust");
    }

    #[test]
    fn parses_query_lex_json_command() {
        let parsed = QueryConfig::parse(&["query".into(), "--json".into(), "lex: rust".into()])
            .unwrap()
            .expect("query config");
        assert_eq!(parsed.query, "lex: rust");
        assert!(parsed.json);
        assert_eq!(parsed.limit, 10);
    }

    #[test]
    fn parses_plain_query_command() {
        let parsed = QueryConfig::parse(&[
            "query".into(),
            "-n".into(),
            "5".into(),
            "-c".into(),
            "docs".into(),
            "--no-rerank".into(),
            "rust".into(),
        ])
        .unwrap()
        .expect("query config");
        assert_eq!(parsed.query, "rust");
        assert_eq!(parsed.limit, 5);
        assert_eq!(parsed.collections, vec!["docs"]);
        assert!(!parsed.rerank);
    }

    #[test]
    fn parses_structured_query_document() {
        let parsed = parse_structured_query("intent: docs\nlex: alpha\nvec: rust").unwrap();
        let parsed = parsed.expect("structured query");
        assert_eq!(parsed.intent.as_deref(), Some("docs"));
        assert_eq!(parsed.searches.len(), 2);
        assert_eq!(parsed.searches[0].kind, QueryKind::Lex);
        assert_eq!(parsed.searches[1].kind, QueryKind::Vec);
    }

    #[test]
    fn parses_collection_remove_command() {
        let parsed =
            CollectionRemoveConfig::parse(&["collection".into(), "remove".into(), "docs".into()])
                .unwrap()
                .expect("collection remove config");
        assert_eq!(parsed.name, "docs");
    }

    #[test]
    fn parses_context_add_command() {
        let parsed = ContextAddConfig::parse(&[
            "context".into(),
            "add".into(),
            "qmd://docs/".into(),
            "hello".into(),
        ])
        .unwrap()
        .expect("context add config");
        assert_eq!(parsed.path.as_deref(), Some("qmd://docs/"));
        assert_eq!(parsed.text, "hello");
    }

    #[test]
    fn parses_update_command() {
        assert!(UpdateConfig::parse(&["update".into()]).unwrap().is_some());
    }

    #[test]
    fn parses_cleanup_command() {
        assert!(CleanupConfig::parse(&["cleanup".into()]).unwrap().is_some());
    }

    #[test]
    fn chunker_matches_qmd_fixture_count_for_doc01() {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let doc_path = manifest
            .join("tests")
            .join("fixtures")
            .join("qmd-corpus")
            .join("doc-01.md");
        let db_path = manifest
            .join("tests")
            .join("fixtures")
            .join("qmd-index.sqlite");
        if !doc_path.exists() || !db_path.exists() {
            return;
        }

        let content = std::fs::read_to_string(&doc_path).expect("fixture doc");
        let runtime = crate::models::runtime().expect("runtime");
        let chunks = chunk_document_for_embedding(runtime.as_deref(), &content);

        let db = Connection::open(db_path).expect("fixture db");
        let expected: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM content_vectors cv
                 JOIN documents d ON d.hash = cv.hash AND d.active = 1
                 WHERE d.path = 'doc-01.md'",
                [],
                |row| row.get(0),
            )
            .expect("fixture count");
        assert_eq!(chunks.len() as i64, expected);
    }

    #[test]
    fn chunker_matches_qmd_fixture_counts_for_pinned_corpus() {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let corpus_dir = manifest.join("tests").join("fixtures").join("qmd-corpus");
        let db_path = manifest
            .join("tests")
            .join("fixtures")
            .join("qmd-index.sqlite");
        if !corpus_dir.is_dir() || !db_path.exists() {
            return;
        }

        let runtime = crate::models::runtime().expect("runtime");
        let db = Connection::open(db_path).expect("fixture db");
        for entry in std::fs::read_dir(&corpus_dir)
            .expect("read corpus")
            .flatten()
        {
            let path = entry.path();
            if path.extension().and_then(|v| v.to_str()) != Some("md") {
                continue;
            }
            let rel = path
                .file_name()
                .and_then(|v| v.to_str())
                .expect("fixture file name")
                .to_string();
            let content = std::fs::read_to_string(&path).expect("fixture doc");
            let chunks = chunk_document_for_embedding(runtime.as_deref(), &content);
            let expected: i64 = db
                .query_row(
                    "SELECT COUNT(*) FROM content_vectors cv
                     JOIN documents d ON d.hash = cv.hash AND d.active = 1
                     WHERE d.path = ?1",
                    [rel.as_str()],
                    |row| row.get(0),
                )
                .expect("fixture count");
            assert_eq!(
                chunks.len() as i64,
                expected,
                "chunk count mismatch for {rel}"
            );
        }
    }
}
