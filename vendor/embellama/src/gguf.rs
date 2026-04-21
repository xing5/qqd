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

//! GGUF file metadata extraction utilities.
//!
//! This module provides functions to read and parse GGUF file metadata
//! without loading the entire model, which is useful for determining
//! model characteristics before initialization.
//!
//! This is a minimal GGUF reader that only extracts KV metadata pairs,
//! based on the GGUF v3 specification from llama.cpp.

use crate::error::{Error, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use dashmap::DashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use tracing::{debug, warn};

/// Global cache for GGUF metadata indexed by canonicalized file path
static METADATA_CACHE: LazyLock<DashMap<PathBuf, GGUFMetadata>> = LazyLock::new(DashMap::new);

/// GGUF model metadata extracted from the file header
#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    /// Model architecture (e.g., "qwen2", "llama", "bert", "jina-bert-v2")
    pub architecture: Option<String>,
    /// Embedding dimensions
    pub embedding_dimensions: usize,
    /// Context size / max sequence length
    pub context_size: usize,
    /// Pooling type from GGUF metadata (e.g., 4 = Rank for reranker models)
    pub pooling_type: Option<u32>,
}

impl GGUFMetadata {
    /// Checks if the model is a decoder-based architecture.
    ///
    /// Decoder architectures include: qwen, qwen2, llama, mistral, phi, etc.
    /// Encoder architectures include: bert, jina-bert-v2, nomic-bert, etc.
    ///
    /// # Returns
    ///
    /// Returns true if the architecture is a known decoder model, false for encoders,
    /// and true (default) for unknown architectures.
    pub fn is_decoder(&self) -> bool {
        if let Some(ref arch) = self.architecture {
            let arch_lower = arch.to_lowercase();

            // Known decoder architectures
            let decoder_archs = [
                "llama", "qwen", "qwen2", "mistral", "mixtral", "phi", "falcon", "gpt2", "gptj",
                "gptneox", "stablelm",
            ];

            // Known encoder architectures
            let encoder_archs = [
                "bert",
                "jina-bert-v2",
                "nomic-bert",
                "roberta",
                "distilbert",
                "electra",
                "gemma-embedding",
            ];

            // Check decoder patterns first
            for pattern in &decoder_archs {
                if arch_lower.contains(pattern) {
                    return true;
                }
            }

            // Check encoder patterns
            for pattern in &encoder_archs {
                if arch_lower.contains(pattern) {
                    return false;
                }
            }

            // Unknown architecture - default to decoder for backward compatibility
            warn!(
                "Unknown GGUF architecture '{}', defaulting to decoder",
                arch
            );
            true
        } else {
            // No architecture found - default to decoder
            warn!("No architecture found in GGUF metadata, defaulting to decoder");
            true
        }
    }

    /// Checks if the model is a reranker based on GGUF `pooling_type` metadata.
    ///
    /// Pooling type 4 corresponds to Rank pooling in llama.cpp, which is used
    /// by cross-encoder reranking models like bge-reranker-v2-m3.
    pub fn is_reranker(&self) -> bool {
        self.pooling_type == Some(4)
    }
}

// ---------- GGUF Constants and Types ----------

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// GGUF value types as defined in the specification
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufType {
    fn from_raw(v: i32) -> Result<Self> {
        Ok(match v {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => {
                return Err(Error::ModelLoadError {
                    path: std::path::PathBuf::new(),
                    source: anyhow::anyhow!("invalid gguf_type {v}"),
                });
            }
        })
    }

    fn scalar_size_bytes(self) -> Option<usize> {
        Some(match self {
            Self::Uint8 | Self::Int8 | Self::Bool => 1,
            Self::Uint16 | Self::Int16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Uint64 | Self::Int64 | Self::Float64 => 8,
            Self::String | Self::Array => return None,
        })
    }
}

/// Parsed KV value
#[derive(Clone)]
enum KvValue {
    /// Raw bytes for numeric/bool types
    Raw {
        elem_ty: GgufType,
        ne: usize,
        bytes: Vec<u8>,
    },
    /// Single string
    Str(String),
    /// Array of strings
    StrArr(Vec<String>),
}

impl std::fmt::Debug for KvValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvValue::Raw { elem_ty, ne, bytes } => {
                const MAX_BYTES: usize = 64;
                if bytes.len() <= MAX_BYTES {
                    f.debug_struct("Raw")
                        .field("elem_ty", elem_ty)
                        .field("ne", ne)
                        .field("bytes", bytes)
                        .finish()
                } else {
                    f.debug_struct("Raw")
                        .field("elem_ty", elem_ty)
                        .field("ne", ne)
                        .field(
                            "bytes",
                            &format!(
                                "[{} bytes, showing first 64: {:?}...]",
                                bytes.len(),
                                &bytes[..MAX_BYTES]
                            ),
                        )
                        .finish()
                }
            }
            KvValue::Str(s) => {
                const MAX_LEN: usize = 100;
                if s.len() <= MAX_LEN {
                    f.debug_tuple("Str").field(s).finish()
                } else {
                    f.debug_tuple("Str")
                        .field(&format!("{}... ({} chars total)", &s[..MAX_LEN], s.len()))
                        .finish()
                }
            }
            KvValue::StrArr(arr) => {
                const MAX_ITEMS: usize = 5;
                if arr.len() <= MAX_ITEMS {
                    f.debug_tuple("StrArr").field(arr).finish()
                } else {
                    f.debug_tuple("StrArr")
                        .field(&format!(
                            "[showing first {} of {} items: {:?}...]",
                            MAX_ITEMS,
                            arr.len(),
                            &arr[..MAX_ITEMS]
                        ))
                        .finish()
                }
            }
        }
    }
}

/// Helper struct for reading binary data
struct Reader<'a> {
    file: &'a mut File,
    path: std::path::PathBuf,
}

impl<'a> Reader<'a> {
    fn new(file: &'a mut File, path: std::path::PathBuf) -> Self {
        Self { file, path }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        self.file
            .read_exact(buf)
            .map_err(|e| Error::ModelLoadError {
                path: self.path.clone(),
                source: anyhow::anyhow!("failed to read {} bytes: {}", buf.len(), e),
            })
    }

    #[allow(dead_code)]
    fn read_u8(&mut self) -> Result<u8> {
        self.file.read_u8().map_err(|e| Error::ModelLoadError {
            path: self.path.clone(),
            source: anyhow::anyhow!("failed to read u8: {e}"),
        })
    }

    fn read_i8(&mut self) -> Result<i8> {
        self.file.read_i8().map_err(|e| Error::ModelLoadError {
            path: self.path.clone(),
            source: anyhow::anyhow!("failed to read i8: {e}"),
        })
    }

    fn read_u32(&mut self) -> Result<u32> {
        self.file
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::ModelLoadError {
                path: self.path.clone(),
                source: anyhow::anyhow!("failed to read u32: {e}"),
            })
    }

    fn read_i32(&mut self) -> Result<i32> {
        self.file
            .read_i32::<LittleEndian>()
            .map_err(|e| Error::ModelLoadError {
                path: self.path.clone(),
                source: anyhow::anyhow!("failed to read i32: {e}"),
            })
    }

    fn read_u64(&mut self) -> Result<u64> {
        self.file
            .read_u64::<LittleEndian>()
            .map_err(|e| Error::ModelLoadError {
                path: self.path.clone(),
                source: anyhow::anyhow!("failed to read u64: {e}"),
            })
    }

    fn read_i64(&mut self) -> Result<i64> {
        self.file
            .read_i64::<LittleEndian>()
            .map_err(|e| Error::ModelLoadError {
                path: self.path.clone(),
                source: anyhow::anyhow!("failed to read i64: {e}"),
            })
    }

    fn read_bool(&mut self) -> Result<bool> {
        let v = self.read_i8()?;
        Ok(v != 0)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = usize::try_from(self.read_u64()?).map_err(|_| Error::ModelLoadError {
            path: self.path.clone(),
            source: anyhow::anyhow!("String length too large for platform"),
        })?;
        let mut buf = vec![0u8; len];
        if len > 0 {
            self.read_exact(&mut buf)?;
        }
        String::from_utf8(buf).map_err(|e| Error::ModelLoadError {
            path: self.path.clone(),
            source: anyhow::anyhow!("invalid utf-8 in GGUF string: {e}"),
        })
    }

    #[allow(dead_code)]
    fn pos(&mut self) -> Result<u64> {
        self.file
            .stream_position()
            .map_err(|e| Error::ModelLoadError {
                path: self.path.clone(),
                source: anyhow::anyhow!("failed to get position: {e}"),
            })
    }
}

/// Extract comprehensive metadata from a GGUF file
///
/// This function reads only the header and KV metadata from the GGUF file
/// without loading tensor data, making it efficient for model inspection.
///
/// Results are cached in memory based on the canonicalized file path, so
/// subsequent calls for the same file will return immediately without re-parsing.
///
/// # Arguments
///
/// * `path` - Path to the GGUF model file
///
/// # Returns
///
/// Returns a `GGUFMetadata` struct containing the extracted information
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed
#[allow(clippy::too_many_lines)]
pub fn extract_metadata(path: &Path) -> Result<GGUFMetadata> {
    // Canonicalize path for cache key (resolves symlinks, relative paths, etc.)
    let canonical_path = path.canonicalize().map_err(|e| Error::ModelLoadError {
        path: path.to_path_buf(),
        source: anyhow::anyhow!("Failed to canonicalize path: {e}"),
    })?;

    // Check cache first
    if let Some(cached) = METADATA_CACHE.get(&canonical_path) {
        debug!(
            "Using cached GGUF metadata for: {} (cache size: {})",
            path.display(),
            METADATA_CACHE.len()
        );
        return Ok(cached.clone());
    }

    debug!("Parsing GGUF file (not in cache): {}", path.display());
    if let Ok(metadata) = std::fs::metadata(path) {
        debug!("File size: {} bytes", metadata.len());
    }

    let mut file = File::open(path).map_err(|e| Error::ModelLoadError {
        path: path.to_path_buf(),
        source: anyhow::anyhow!("Failed to open GGUF file: {e}"),
    })?;

    let mut reader = Reader::new(&mut file, path.to_path_buf());

    // Read and validate magic bytes
    debug!("Reading GGUF magic bytes");
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        let shown: Vec<char> = magic
            .iter()
            .map(|&c| if c.is_ascii_graphic() { c as char } else { '?' })
            .collect();
        return Err(Error::ModelLoadError {
            path: path.to_path_buf(),
            source: anyhow::anyhow!("invalid GGUF magic: got '{shown:?}', expected 'GGUF'"),
        });
    }
    debug!("Valid GGUF magic bytes found");

    // Read version
    let version = reader.read_u32()?;
    debug!("GGUF version: {}", version);
    if version == 0 {
        return Err(Error::ModelLoadError {
            path: path.to_path_buf(),
            source: anyhow::anyhow!("bad GGUF version: 0"),
        });
    }
    if version.trailing_zeros() >= 16 {
        return Err(Error::ModelLoadError {
            path: path.to_path_buf(),
            source: anyhow::anyhow!(
                "unreasonable GGUF version {version:08x} — likely endianness mismatch"
            ),
        });
    }

    // Read counts (n_tensors, n_kv)
    let n_tensors_i64 = reader.read_i64()?;
    if n_tensors_i64 < 0 {
        return Err(Error::ModelLoadError {
            path: path.to_path_buf(),
            source: anyhow::anyhow!("negative n_tensors: {n_tensors_i64}"),
        });
    }
    let n_tensors = usize::try_from(n_tensors_i64).map_err(|_| Error::ModelLoadError {
        path: path.to_path_buf(),
        source: anyhow::anyhow!("n_tensors too large for platform: {n_tensors_i64}"),
    })?;

    let n_kv_i64 = reader.read_i64()?;
    if n_kv_i64 < 0 {
        return Err(Error::ModelLoadError {
            path: path.to_path_buf(),
            source: anyhow::anyhow!("negative n_kv: {n_kv_i64}"),
        });
    }
    let n_kv = usize::try_from(n_kv_i64).map_err(|_| Error::ModelLoadError {
        path: path.to_path_buf(),
        source: anyhow::anyhow!("n_kv too large for platform: {n_kv_i64}"),
    })?;

    debug!("GGUF file has {} KV pairs and {} tensors", n_kv, n_tensors);

    // Parse KV pairs
    let mut seen_keys = HashSet::with_capacity(n_kv);
    let mut architecture: Option<String> = None;
    let mut dimensions = 0usize;
    let mut context_size = 512usize; // Default fallback
    let mut pooling_type: Option<u32> = None;

    debug!("Reading {} KV pairs", n_kv);
    for i in 0..n_kv {
        // Read key
        let key = reader.read_string().map_err(|e| Error::ModelLoadError {
            path: path.to_path_buf(),
            source: anyhow::anyhow!("reading key {i} failed: {e}"),
        })?;

        if !seen_keys.insert(key.clone()) {
            return Err(Error::ModelLoadError {
                path: path.to_path_buf(),
                source: anyhow::anyhow!("duplicate key '{key}' at kv index {i}"),
            });
        }

        // Read type
        let ty_tag = GgufType::from_raw(reader.read_i32()?)?;
        let (is_array, elem_ty, ne) = if ty_tag == GgufType::Array {
            let elem = GgufType::from_raw(reader.read_i32()?)?;
            let n = usize::try_from(reader.read_u64()?).map_err(|_| Error::ModelLoadError {
                path: path.to_path_buf(),
                source: anyhow::anyhow!("Array length too large for platform"),
            })?;
            (true, elem, n)
        } else {
            (false, ty_tag, 1)
        };

        // Parse value based on element type
        let value = match elem_ty {
            GgufType::String => {
                if is_array {
                    let mut strings = Vec::with_capacity(ne);
                    for _ in 0..ne {
                        strings.push(reader.read_string()?);
                    }
                    KvValue::StrArr(strings)
                } else {
                    KvValue::Str(reader.read_string()?)
                }
            }
            GgufType::Bool => {
                let mut bytes = Vec::with_capacity(ne);
                for _ in 0..ne {
                    bytes.push(u8::from(reader.read_bool()?));
                }
                KvValue::Raw { elem_ty, ne, bytes }
            }
            GgufType::Uint8
            | GgufType::Int8
            | GgufType::Uint16
            | GgufType::Int16
            | GgufType::Uint32
            | GgufType::Int32
            | GgufType::Float32
            | GgufType::Uint64
            | GgufType::Int64
            | GgufType::Float64 => {
                let el_sz = elem_ty
                    .scalar_size_bytes()
                    .ok_or_else(|| Error::ModelLoadError {
                        path: path.to_path_buf(),
                        source: anyhow::anyhow!("unexpected non-scalar gguf type {elem_ty:?}"),
                    })?;
                let mut bytes = vec![0u8; el_sz * ne];
                reader.read_exact(&mut bytes)?;
                KvValue::Raw { elem_ty, ne, bytes }
            }
            GgufType::Array => {
                return Err(Error::ModelLoadError {
                    path: path.to_path_buf(),
                    source: anyhow::anyhow!("unexpected nested ARRAY (not allowed)"),
                });
            }
        };

        debug!("  KV[{}]: '{}' = {:?}", i, key, value);

        // Extract the values we need
        if key == "general.architecture" {
            if let KvValue::Str(ref arch) = value {
                architecture = Some(arch.clone());
                debug!("Found architecture: {}", arch);
            }
        }

        // Check for embedding dimensions
        if dimensions == 0
            && (key == "llama.embedding_length"
                || key == "embedding_length"
                || key == "n_embd"
                || key == "bert.embedding_length"
                || key.ends_with(".embedding_length"))
        {
            if let Some(dim) = extract_usize_from_value(&value) {
                dimensions = dim;
                debug!(
                    "Found embedding dimensions: {} from key: {}",
                    dimensions, key
                );
            }
        }

        // Check for context length
        if context_size == 512
            && (key == "llama.context_length"
                || key == "context_length"
                || key == "n_ctx"
                || key == "max_position_embeddings"
                || key == "bert.context_length"
                || key.ends_with(".context_length"))
        {
            if let Some(ctx) = extract_usize_from_value(&value) {
                context_size = ctx;
                debug!("Found context size: {} from key: {}", context_size, key);
            }
        }

        // Check for pooling type
        if pooling_type.is_none() && (key.ends_with(".pooling_type") || key == "pooling_type") {
            if let Some(pt) = extract_usize_from_value(&value) {
                pooling_type = u32::try_from(pt).ok();
                debug!("Found pooling type: {} from key: {}", pt, key);
            }
        }
    }

    debug!("Finished parsing KV pairs");
    debug!("  Architecture: {:?}", architecture);
    debug!("  Embedding dimensions: {}", dimensions);
    debug!("  Context size: {}", context_size);

    if dimensions == 0 {
        warn!("Could not determine embedding dimensions from GGUF metadata");
    }

    let metadata = GGUFMetadata {
        architecture,
        embedding_dimensions: dimensions,
        context_size,
        pooling_type,
    };

    // Cache the result for future calls
    METADATA_CACHE.insert(canonical_path, metadata.clone());
    debug!(
        "Cached GGUF metadata (cache size now: {})",
        METADATA_CACHE.len()
    );

    Ok(metadata)
}

/// Clear the GGUF metadata cache
///
/// This is useful for testing or when model files have been updated.
/// Normally you don't need to call this as the cache is indexed by
/// canonicalized path and will automatically handle file updates.
pub fn clear_metadata_cache() {
    let count = METADATA_CACHE.len();
    METADATA_CACHE.clear();
    debug!("Cleared GGUF metadata cache ({} entries removed)", count);
}

/// Get the current size of the GGUF metadata cache
///
/// This is primarily useful for monitoring and debugging.
pub fn metadata_cache_size() -> usize {
    METADATA_CACHE.len()
}

/// Helper function to extract usize value from KV value
fn extract_usize_from_value(value: &KvValue) -> Option<usize> {
    match value {
        KvValue::Raw { elem_ty, ne, bytes } if *ne == 1 => {
            use std::io::Cursor;
            let mut cursor = Cursor::new(bytes);
            match elem_ty {
                GgufType::Uint8 => cursor.read_u8().ok().map(|v| v as usize),
                #[allow(clippy::cast_sign_loss)]
                GgufType::Int8 => cursor
                    .read_i8()
                    .ok()
                    .filter(|&v| v >= 0)
                    .map(|v| v as usize),
                GgufType::Uint16 => cursor.read_u16::<LittleEndian>().ok().map(|v| v as usize),
                #[allow(clippy::cast_sign_loss)]
                GgufType::Int16 => cursor
                    .read_i16::<LittleEndian>()
                    .ok()
                    .filter(|&v| v >= 0)
                    .map(|v| v as usize),
                GgufType::Uint32 => cursor
                    .read_u32::<LittleEndian>()
                    .ok()
                    .and_then(|v| v.try_into().ok()),
                GgufType::Int32 => cursor
                    .read_i32::<LittleEndian>()
                    .ok()
                    .filter(|&v| v >= 0)
                    .and_then(|v| v.try_into().ok()),
                GgufType::Uint64 => cursor
                    .read_u64::<LittleEndian>()
                    .ok()
                    .and_then(|v| v.try_into().ok()),
                GgufType::Int64 => cursor
                    .read_i64::<LittleEndian>()
                    .ok()
                    .filter(|&v| v >= 0)
                    .and_then(|v| v.try_into().ok()),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_decoder_qwen() {
        let metadata = GGUFMetadata {
            architecture: Some("qwen2".to_string()),
            embedding_dimensions: 896,
            context_size: 32768,
            pooling_type: None,
        };
        assert!(metadata.is_decoder(), "qwen2 should be detected as decoder");
    }

    #[test]
    fn test_is_decoder_llama() {
        let metadata = GGUFMetadata {
            architecture: Some("llama".to_string()),
            embedding_dimensions: 4096,
            context_size: 2048,
            pooling_type: None,
        };
        assert!(metadata.is_decoder(), "llama should be detected as decoder");
    }

    #[test]
    fn test_is_encoder_bert() {
        let metadata = GGUFMetadata {
            architecture: Some("bert".to_string()),
            embedding_dimensions: 768,
            context_size: 512,
            pooling_type: None,
        };
        assert!(!metadata.is_decoder(), "bert should be detected as encoder");
    }

    #[test]
    fn test_is_encoder_jina_bert() {
        let metadata = GGUFMetadata {
            architecture: Some("jina-bert-v2".to_string()),
            embedding_dimensions: 768,
            context_size: 8192,
            pooling_type: None,
        };
        assert!(
            !metadata.is_decoder(),
            "jina-bert-v2 should be detected as encoder"
        );
    }

    #[test]
    fn test_is_decoder_unknown() {
        let metadata = GGUFMetadata {
            architecture: Some("unknown-arch".to_string()),
            embedding_dimensions: 1024,
            context_size: 2048,
            pooling_type: None,
        };
        assert!(
            metadata.is_decoder(),
            "unknown architecture should default to decoder"
        );
    }

    #[test]
    fn test_is_decoder_none() {
        let metadata = GGUFMetadata {
            architecture: None,
            embedding_dimensions: 1024,
            context_size: 2048,
            pooling_type: None,
        };
        assert!(
            metadata.is_decoder(),
            "no architecture should default to decoder"
        );
    }

    #[test]
    fn test_is_reranker_with_rank_pooling() {
        let metadata = GGUFMetadata {
            architecture: Some("bert".to_string()),
            embedding_dimensions: 768,
            context_size: 512,
            pooling_type: Some(4),
        };
        assert!(
            metadata.is_reranker(),
            "pooling_type 4 should be detected as reranker"
        );
    }

    #[test]
    fn test_is_reranker_with_mean_pooling() {
        let metadata = GGUFMetadata {
            architecture: Some("bert".to_string()),
            embedding_dimensions: 768,
            context_size: 512,
            pooling_type: Some(1),
        };
        assert!(
            !metadata.is_reranker(),
            "pooling_type 1 (Mean) should not be reranker"
        );
    }

    #[test]
    fn test_is_reranker_with_no_pooling_type() {
        let metadata = GGUFMetadata {
            architecture: Some("bert".to_string()),
            embedding_dimensions: 768,
            context_size: 512,
            pooling_type: None,
        };
        assert!(
            !metadata.is_reranker(),
            "None pooling_type should not be reranker"
        );
    }

    #[test]
    fn test_metadata_cache() {
        use tempfile::NamedTempFile;

        // Clear cache first
        clear_metadata_cache();
        assert_eq!(metadata_cache_size(), 0);

        // Create a minimal valid GGUF file for testing
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Write minimal GGUF structure
        use std::io::Write;
        let mut file = std::fs::File::create(path).unwrap();

        // Magic
        file.write_all(b"GGUF").unwrap();
        // Version (3)
        file.write_all(&3u32.to_le_bytes()).unwrap();
        // n_tensors (0)
        file.write_all(&0i64.to_le_bytes()).unwrap();
        // n_kv (1)
        file.write_all(&1i64.to_le_bytes()).unwrap();

        // KV: general.architecture = "test"
        // Key length (20 bytes for "general.architecture")
        file.write_all(&20u64.to_le_bytes()).unwrap();
        // Key
        file.write_all(b"general.architecture").unwrap();
        // Type: String (8)
        file.write_all(&8i32.to_le_bytes()).unwrap();
        // Value length
        file.write_all(&4u64.to_le_bytes()).unwrap();
        // Value
        file.write_all(b"test").unwrap();
        drop(file);

        // First call should parse the file
        let result1 = extract_metadata(path);
        assert!(result1.is_ok());
        assert_eq!(metadata_cache_size(), 1);

        // Second call should use cache
        let result2 = extract_metadata(path);
        assert!(result2.is_ok());
        assert_eq!(metadata_cache_size(), 1);

        // Results should be identical
        let meta1 = result1.unwrap();
        let meta2 = result2.unwrap();
        assert_eq!(meta1.architecture, meta2.architecture);
        assert_eq!(meta1.architecture, Some("test".to_string()));

        // Clear cache
        clear_metadata_cache();
        assert_eq!(metadata_cache_size(), 0);
    }
}
