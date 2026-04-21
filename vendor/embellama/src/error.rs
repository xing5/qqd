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

use std::path::PathBuf;
use thiserror::Error;

/// Custom error type for the embellama library
#[derive(Error, Debug)]
pub enum Error {
    /// Error when model loading fails
    #[error("Failed to load model from path: {path}")]
    ModelLoadError {
        /// Path to the model that failed to load
        path: PathBuf,
        #[source]
        /// Underlying error from llama-cpp-2
        source: anyhow::Error,
    },

    /// Error when a requested model is not found
    #[error("Model not found: {name}")]
    ModelNotFound {
        /// Name of the model that was not found
        name: String,
    },

    /// Error during embedding generation
    #[error("Failed to generate embedding: {message}")]
    EmbeddingGenerationError {
        /// Description of what went wrong
        message: String,
        #[source]
        /// Optional underlying error
        source: Option<anyhow::Error>,
    },

    /// Error in configuration
    #[error("Configuration error: {message}")]
    ConfigurationError {
        /// Description of the configuration error
        message: String,
    },

    /// Error for invalid input
    #[error("Invalid input: {message}")]
    InvalidInput {
        /// Description of what makes the input invalid
        message: String,
    },

    /// Error during model initialization
    #[error("Model initialization failed: {message}")]
    ModelInitError {
        /// Description of initialization failure
        message: String,
        #[source]
        /// Optional underlying error for init
        source: Option<anyhow::Error>,
    },

    /// Error when creating llama context
    #[error("Context creation failed")]
    ContextError {
        #[source]
        /// Underlying error from context creation
        source: anyhow::Error,
    },

    /// Error during text tokenization
    #[error("Tokenization failed: {message}")]
    TokenizationError {
        /// Description of tokenization failure
        message: String,
    },

    /// Error during batch processing
    #[error("Batch processing error: {message}")]
    BatchError {
        /// Description of batch processing error
        message: String,
        /// Indices of texts that failed processing
        failed_indices: Vec<usize>,
    },

    /// Error from thread pool operations
    #[error("Thread pool error")]
    ThreadPoolError {
        #[source]
        /// Underlying thread pool error
        source: anyhow::Error,
    },

    /// Error when a lock is poisoned
    #[error("Model registry lock poisoned")]
    LockPoisoned,

    /// I/O operation error
    #[error("IO error: {message}")]
    IoError {
        /// Description of I/O error
        message: String,
        #[source]
        /// The underlying I/O error
        source: std::io::Error,
    },

    /// Error during model warmup
    #[error("Model warmup failed")]
    WarmupError {
        #[source]
        /// Underlying warmup error
        source: anyhow::Error,
    },

    /// Error when embedding dimensions don't match expectations
    #[error("Invalid model dimensions: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected number of dimensions
        expected: usize,
        /// Actual number of dimensions
        actual: usize,
    },

    /// Error when resource limits are exceeded
    #[error("Resource limit exceeded: {message}")]
    ResourceLimitExceeded {
        /// Description of which limit was exceeded
        message: String,
    },

    /// Error when an operation times out
    #[error("Operation timeout: {message}")]
    Timeout {
        /// Description of what timed out
        message: String,
    },

    /// Error when an operation is invalid or not supported
    #[error("Invalid operation: {message}")]
    InvalidOperation {
        /// Description of the invalid operation
        message: String,
    },

    /// Catch-all for other errors
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl Error {
    /// Create a configuration error with a custom message
    pub fn config(message: impl Into<String>) -> Self {
        Self::ConfigurationError {
            message: message.into(),
        }
    }

    /// Create an invalid input error with a custom message
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// Create a model load error
    pub fn model_load(path: PathBuf, source: anyhow::Error) -> Self {
        Self::ModelLoadError { path, source }
    }

    /// Create an embedding generation error
    pub fn embedding_failed(message: impl Into<String>) -> Self {
        Self::EmbeddingGenerationError {
            message: message.into(),
            source: None,
        }
    }

    /// Create an embedding generation error with source
    pub fn embedding_failed_with_source(message: impl Into<String>, source: anyhow::Error) -> Self {
        Self::EmbeddingGenerationError {
            message: message.into(),
            source: Some(source),
        }
    }

    /// Check if this is a retryable error
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Timeout { .. } | Self::ThreadPoolError { .. } | Self::LockPoisoned
        )
    }

    /// Check if this is a configuration-related error
    pub fn is_configuration_error(&self) -> bool {
        matches!(
            self,
            Self::ConfigurationError { .. }
                | Self::InvalidInput { .. }
                | Self::ModelNotFound { .. }
        )
    }
}

/// Type alias for Results in this crate
pub type Result<T> = std::result::Result<T, Error>;

/// Convert `std::io::Error` to our Error type
impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::IoError {
            message: err.to_string(),
            source: err,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_error_display() {
        // Test all error variants display correctly
        let err = Error::ModelNotFound {
            name: "test-model".to_string(),
        };
        assert_eq!(err.to_string(), "Model not found: test-model");

        let err = Error::ConfigurationError {
            message: "Invalid config".to_string(),
        };
        assert_eq!(err.to_string(), "Configuration error: Invalid config");

        let err = Error::InvalidInput {
            message: "Empty text".to_string(),
        };
        assert_eq!(err.to_string(), "Invalid input: Empty text");

        let err = Error::TokenizationError {
            message: "Token overflow".to_string(),
        };
        assert_eq!(err.to_string(), "Tokenization failed: Token overflow");

        let err = Error::DimensionMismatch {
            expected: 384,
            actual: 512,
        };
        assert_eq!(
            err.to_string(),
            "Invalid model dimensions: expected 384, got 512"
        );

        let err = Error::ResourceLimitExceeded {
            message: "Memory limit reached".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Resource limit exceeded: Memory limit reached"
        );

        let err = Error::Timeout {
            message: "Request timed out".to_string(),
        };
        assert_eq!(err.to_string(), "Operation timeout: Request timed out");
    }

    #[test]
    fn test_error_helpers() {
        let err = Error::config("Invalid setting");
        assert!(err.is_configuration_error());
        assert!(!err.is_retryable());

        let err = Error::Timeout {
            message: "Operation timed out".to_string(),
        };
        assert!(err.is_retryable());
        assert!(!err.is_configuration_error());

        let err = Error::ThreadPoolError {
            source: anyhow::anyhow!("Thread pool exhausted"),
        };
        assert!(err.is_retryable());
        assert!(!err.is_configuration_error());

        let err = Error::LockPoisoned;
        assert!(err.is_retryable());
        assert!(!err.is_configuration_error());
    }

    #[test]
    fn test_invalid_input() {
        let err = Error::invalid_input("Empty text provided");
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.is_configuration_error());
        assert_eq!(err.to_string(), "Invalid input: Empty text provided");
    }

    #[test]
    fn test_model_load_error() {
        let path = PathBuf::from("/path/to/model.gguf");
        let source = anyhow::anyhow!("File not found");
        let err = Error::model_load(path.clone(), source);

        assert!(matches!(err, Error::ModelLoadError { .. }));
        assert!(!err.is_configuration_error());
        assert!(!err.is_retryable());
        assert!(err.to_string().contains("/path/to/model.gguf"));
    }

    #[test]
    fn test_embedding_errors() {
        let err = Error::embedding_failed("Model not ready");
        assert!(matches!(
            err,
            Error::EmbeddingGenerationError { source: None, .. }
        ));
        assert_eq!(
            err.to_string(),
            "Failed to generate embedding: Model not ready"
        );

        let source = anyhow::anyhow!("CUDA out of memory");
        let err = Error::embedding_failed_with_source("GPU error", source);
        assert!(matches!(
            err,
            Error::EmbeddingGenerationError {
                source: Some(_),
                ..
            }
        ));
        assert_eq!(err.to_string(), "Failed to generate embedding: GPU error");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "File not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::IoError { .. }));
        assert!(!err.is_configuration_error());
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_batch_error() {
        let err = Error::BatchError {
            message: "Processing failed".to_string(),
            failed_indices: vec![1, 3, 5],
        };
        assert_eq!(err.to_string(), "Batch processing error: Processing failed");

        if let Error::BatchError { failed_indices, .. } = err {
            assert_eq!(failed_indices, vec![1, 3, 5]);
        } else {
            panic!("Expected BatchError");
        }
    }

    #[test]
    fn test_model_init_error() {
        let err = Error::ModelInitError {
            message: "Backend initialization failed".to_string(),
            source: None,
        };
        assert_eq!(
            err.to_string(),
            "Model initialization failed: Backend initialization failed"
        );
        assert!(!err.is_configuration_error());
        assert!(!err.is_retryable());

        let source = anyhow::anyhow!("CUDA not available");
        let err = Error::ModelInitError {
            message: "GPU init failed".to_string(),
            source: Some(source),
        };
        assert!(matches!(
            err,
            Error::ModelInitError {
                source: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn test_context_error() {
        let source = anyhow::anyhow!("Context size too large");
        let err = Error::ContextError { source };
        assert_eq!(err.to_string(), "Context creation failed");
        assert!(!err.is_configuration_error());
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_warmup_error() {
        let source = anyhow::anyhow!("Warmup inference failed");
        let err = Error::WarmupError { source };
        assert_eq!(err.to_string(), "Model warmup failed");
        assert!(!err.is_configuration_error());
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_configuration_error_helpers() {
        let err = Error::ModelNotFound {
            name: "missing-model".to_string(),
        };
        assert!(err.is_configuration_error());

        let err = Error::InvalidInput {
            message: "Text too long".to_string(),
        };
        assert!(err.is_configuration_error());

        let err = Error::ConfigurationError {
            message: "Invalid batch size".to_string(),
        };
        assert!(err.is_configuration_error());
    }

    #[test]
    fn test_retryable_errors() {
        // Test all retryable error types
        let retryable_errors = vec![
            Error::Timeout {
                message: "timeout".to_string(),
            },
            Error::ThreadPoolError {
                source: anyhow::anyhow!("pool error"),
            },
            Error::LockPoisoned,
        ];

        for err in retryable_errors {
            assert!(err.is_retryable(), "{err:?} should be retryable");
        }

        // Test non-retryable errors
        let non_retryable = vec![
            Error::ModelNotFound {
                name: "model".to_string(),
            },
            Error::ConfigurationError {
                message: "config".to_string(),
            },
            Error::InvalidInput {
                message: "input".to_string(),
            },
        ];

        for err in non_retryable {
            assert!(!err.is_retryable(), "{err:?} should not be retryable");
        }
    }

    #[test]
    fn test_error_chaining() {
        // Test that error chaining works correctly with source errors
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "Access denied");
        let err = Error::IoError {
            message: "Cannot read model file".to_string(),
            source: io_err,
        };

        // Check that we can access the source error
        if let Error::IoError { source, .. } = &err {
            assert_eq!(source.kind(), io::ErrorKind::PermissionDenied);
        } else {
            panic!("Expected IoError");
        }
    }

    #[test]
    fn test_anyhow_conversion() {
        let anyhow_err = anyhow::anyhow!("Generic error");
        let err: Error = Error::Other(anyhow_err);
        assert!(matches!(err, Error::Other(_)));
        assert_eq!(err.to_string(), "Generic error");
    }

    #[test]
    fn test_result_type_alias() {
        // Verify the Result type alias works correctly
        fn returns_result() -> Result<()> {
            Err(Error::config("test"))
        }

        let result = returns_result();
        assert!(result.is_err());

        if let Err(e) = result {
            assert!(matches!(e, Error::ConfigurationError { .. }));
        }
    }
}
