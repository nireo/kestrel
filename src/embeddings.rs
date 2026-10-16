use std::path::PathBuf;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

/// The model and preprocessing that define this index's embedding space.
/// Keep this configuration with stored vectors when persistence is added.
/// This identifies the model, not an immutable revision of downloaded weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingConfig {
    pub model_id: &'static str,
    pub artifact_repo: &'static str,
    pub artifact_file: &'static str,
    pub dimensions: usize,
    /// Model-token limit including special tokens and the query instruction.
    pub max_tokens: usize,
    pub query_instruction: &'static str,
}

pub(crate) const CONFIG: EmbeddingConfig = EmbeddingConfig {
    model_id: "BAAI/bge-small-en-v1.5",
    artifact_repo: "Xenova/bge-small-en-v1.5",
    artifact_file: "onnx/model.onnx",
    dimensions: 384,
    max_tokens: 512,
    query_instruction: "Represent this sentence for searching relevant passages: ",
};

impl Default for EmbeddingConfig {
    /// The English BGE configuration used by `TextEmbedder` and `Collection`.
    /// Reading these settings does not load or download a model.
    fn default() -> Self {
        CONFIG
    }
}

const DOCUMENT_BATCH_SIZE: usize = 32;

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("embedding input {index} is empty or whitespace-only")]
    EmptyText { index: usize },
    #[error("embedding backend failed: {0}")]
    Backend(#[from] fastembed::Error),
    #[error("expected {expected} embedding vectors, got {actual}")]
    UnexpectedCount { expected: usize, actual: usize },
    #[error("embedding {index} has {actual} dimensions; expected {expected}")]
    UnexpectedDimensions {
        index: usize,
        expected: usize,
        actual: usize,
    },
}

/// Local text embeddings, loaded once and reused for documents and queries.
///
/// First use downloads model files; subsequent loads reuse the cache.
/// Long inputs are truncated by the model
/// tokenizer at 512 tokens. Start with short passages; chunk longer documents.
/// The embedding tokenizer is independent of BM25's word tokenizer.
pub struct TextEmbedder {
    model: TextEmbedding,
}

impl TextEmbedder {
    /// Load the fixed English BGE model, using FastEmbed's default cache policy.
    pub fn new() -> Result<Self, EmbeddingError> {
        Self::load(None)
    }

    /// Load the same model with a custom cache directory. FastEmbed's HF_HOME
    /// environment override still takes precedence over this directory.
    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Result<Self, EmbeddingError> {
        Self::load(Some(cache_dir.into()))
    }

    fn load(cache_dir: Option<PathBuf>) -> Result<Self, EmbeddingError> {
        let mut options = TextInitOptions::new(EmbeddingModel::BGESmallENV15)
            .with_max_length(CONFIG.max_tokens)
            .with_show_download_progress(true);
        if let Some(cache_dir) = cache_dir {
            options = options.with_cache_dir(cache_dir);
        }
        Ok(Self {
            model: TextEmbedding::try_new(options)?,
        })
    }

    pub fn config(&self) -> &'static EmbeddingConfig {
        &CONFIG
    }

    pub fn dimensions(&self) -> usize {
        CONFIG.dimensions
    }

    pub(crate) fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer
    }

    /// Embed documents in batches, preserving input order. No instruction is
    /// added to passages. Empty batches return no vectors; blank texts error.
    pub fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        validate_texts(texts)?;
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Pool one batch before running the next: FastEmbed retains the raw
        // token-level outputs of each batch in a single embed() call.
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(DOCUMENT_BATCH_SIZE) {
            let batch_vectors = self.model.embed(batch, None)?;
            validate_output(&batch_vectors, batch.len())?;
            vectors.extend(batch_vectors);
        }
        Ok(vectors)
    }

    /// Add BGE's retrieval instruction and embed a single query. Pass the
    /// original query text, without the instruction, to BM25 and hybrid_search.
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        validate_texts(&[text])?;
        let input = format!("{}{text}", CONFIG.query_instruction);
        let mut vectors = self.model.embed([input], None)?;
        validate_output(&vectors, 1)?;
        Ok(vectors.remove(0))
    }
}

fn validate_texts(texts: &[&str]) -> Result<(), EmbeddingError> {
    for (index, text) in texts.iter().enumerate() {
        if text.trim().is_empty() {
            return Err(EmbeddingError::EmptyText { index });
        }
    }
    Ok(())
}

fn validate_output(vectors: &[Vec<f32>], expected: usize) -> Result<(), EmbeddingError> {
    if vectors.len() != expected {
        return Err(EmbeddingError::UnexpectedCount {
            expected,
            actual: vectors.len(),
        });
    }
    for (index, vector) in vectors.iter().enumerate() {
        if vector.len() != CONFIG.dimensions {
            return Err(EmbeddingError::UnexpectedDimensions {
                index,
                expected: CONFIG.dimensions,
                actual: vector.len(),
            });
        }
    }
    Ok(())
}
