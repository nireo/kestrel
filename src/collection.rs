use std::path::Path;

use crate::chunking::Chunker;
use crate::{
    Chunk, ChunkHit, ChunkId, ChunkingConfig, ChunkingError, DocId, EmbeddingConfig,
    EmbeddingError, HybridSearchOptions, IndexError, IvfError, PreparedChunk, PreparedDocument,
    RetrievalCollection, SnapshotError, TextEmbedder, VectorError, VectorSearchMode,
};

#[derive(Debug, thiserror::Error)]
pub enum CollectionError {
    #[error(transparent)]
    Embedding(#[from] EmbeddingError),
    #[error(transparent)]
    Chunking(#[from] ChunkingError),
    #[error(transparent)]
    Index(#[from] IndexError),
    #[error(transparent)]
    Vector(#[from] VectorError),
    #[error(transparent)]
    Ivf(#[from] IvfError),
    #[error(transparent)]
    Snapshot(#[from] SnapshotError),
    #[error("invalid prepared document: {0}")]
    InvalidDocument(&'static str),
    #[error("retrieval embedding configuration does not match the supplied embedder")]
    IncompatibleEmbeddingConfig,
}

/// Local chunking and embedding convenience wrapper around `RetrievalCollection`.
/// BM25 and vector search both rank chunks; len() and document() refer to source
/// documents. Use `into_retrieval` to serve prepared queries without the model.
pub struct Collection {
    embedder: TextEmbedder,
    chunker: Chunker,
    retrieval: RetrievalCollection,
}

impl Collection {
    pub fn new() -> Result<Self, CollectionError> {
        Self::with_chunking(ChunkingConfig::default())
    }

    pub fn with_chunking(config: ChunkingConfig) -> Result<Self, CollectionError> {
        config.validate(crate::embeddings::CONFIG.max_tokens)?;
        Self::from_embedder_with_chunking(TextEmbedder::new()?, config)
    }

    pub fn from_embedder(embedder: TextEmbedder) -> Result<Self, CollectionError> {
        Self::from_embedder_with_chunking(embedder, ChunkingConfig::default())
    }

    pub fn from_embedder_with_chunking(
        embedder: TextEmbedder,
        config: ChunkingConfig,
    ) -> Result<Self, CollectionError> {
        let retrieval = RetrievalCollection::new(*embedder.config(), config)?;
        Self::from_retrieval(retrieval, embedder)
    }

    /// Attach local chunking and embedding to an existing retrieval collection.
    /// The embedding space must match exactly; existing IDs and retrieval mode
    /// are preserved.
    pub fn from_retrieval(
        retrieval: RetrievalCollection,
        embedder: TextEmbedder,
    ) -> Result<Self, CollectionError> {
        if retrieval.embedding_config() != embedder.config() {
            return Err(CollectionError::IncompatibleEmbeddingConfig);
        }
        let chunker = Chunker::new(
            embedder.tokenizer().clone(),
            retrieval.chunking_config(),
            embedder.config().max_tokens,
        )?;
        Ok(Self {
            embedder,
            chunker,
            retrieval,
        })
    }

    /// Borrow the model-free retrieval core for precomputed-query searches.
    pub fn retrieval(&self) -> &RetrievalCollection {
        &self.retrieval
    }

    /// Drop the local model and chunker, retaining all indexed data and settings.
    pub fn into_retrieval(self) -> RetrievalCollection {
        self.retrieval
    }

    /// Save all chunk vectors, including buffered IVF writes. Retrieval mode is
    /// runtime-only; loading starts in exact mode.
    /// Atomic replacement after writing and syncing the temporary file. The
    /// parent directory must exist. SyncAfterReplace means the new file exists
    /// but syncing its parent directory failed (Unix only).
    pub fn save_snapshot(&self, path: impl AsRef<Path>) -> Result<(), CollectionError> {
        self.retrieval.save_snapshot(path)
    }

    /// Restore stored boundaries and vectors without chunking or embedding the
    /// documents again. Version 1 snapshots become one chunk per document.
    /// Retrieval starts in exact mode; select IVF to train a fresh index.
    /// This wrapper loads the local model; use `RetrievalCollection::load_snapshot`
    /// to restore solely for precomputed-query retrieval.
    pub fn load_snapshot(path: impl AsRef<Path>) -> Result<Self, CollectionError> {
        let retrieval = RetrievalCollection::load_snapshot(path, EmbeddingConfig::default())?;
        Self::from_retrieval(retrieval, TextEmbedder::new()?)
    }

    pub fn load_snapshot_with_embedder(
        path: impl AsRef<Path>,
        embedder: TextEmbedder,
    ) -> Result<Self, CollectionError> {
        let retrieval = RetrievalCollection::load_snapshot(path, *embedder.config())?;
        Self::from_retrieval(retrieval, embedder)
    }

    pub fn len(&self) -> usize {
        self.retrieval.len()
    }
    pub fn is_empty(&self) -> bool {
        self.retrieval.is_empty()
    }
    pub fn chunk_count(&self) -> usize {
        self.retrieval.chunk_count()
    }

    pub fn document(&self, doc_id: DocId) -> Option<&str> {
        self.retrieval.document(doc_id)
    }

    pub fn chunk(&self, chunk_id: ChunkId) -> Option<&Chunk> {
        self.retrieval.chunk(chunk_id)
    }

    pub fn chunk_text(&self, chunk_id: ChunkId) -> Option<&str> {
        self.retrieval.chunk_text(chunk_id)
    }

    pub fn embedding_config(&self) -> &'static EmbeddingConfig {
        self.embedder.config()
    }
    pub fn chunking_config(&self) -> ChunkingConfig {
        self.chunker.config()
    }

    /// Select exact or IVF retrieval. Changing build settings retrains all chunks;
    /// changing only probes keeps the existing index and insertion buffer.
    /// Build failures leave the current backend and buffer unchanged.
    pub fn set_vector_search_mode(
        &mut self,
        mode: VectorSearchMode,
    ) -> Result<(), CollectionError> {
        self.retrieval.set_vector_search_mode(mode)
    }

    pub fn vector_search_mode(&self) -> VectorSearchMode {
        self.retrieval.vector_search_mode()
    }

    /// Number of newly inserted chunk vectors awaiting an explicit IVF rebuild.
    /// They are already searchable through the exact insertion buffer.
    pub fn buffered_vector_count(&self) -> usize {
        self.retrieval.buffered_vector_count()
    }

    /// Retrain IVF over all chunks and empty its insertion buffer after success.
    /// This synchronous operation needs exclusive access; exact mode is a no-op.
    pub fn rebuild_vector_index(&mut self) -> Result<(), CollectionError> {
        self.retrieval.rebuild_vector_index()
    }

    /// Prepare a query with the collection's embedding model and instruction.
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>, CollectionError> {
        Ok(self.embedder.embed_query(query)?)
    }

    /// Split and embed every passage before publishing a document. Returned
    /// errors leave documents, chunks, indexes, and the next IDs unchanged.
    pub fn add_document(&mut self, text: impl Into<String>) -> Result<DocId, CollectionError> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(EmbeddingError::EmptyText { index: 0 }.into());
        }
        let ranges = self.chunker.ranges(&text)?;
        let passages: Vec<_> = ranges.iter().map(|range| &text[range.clone()]).collect();
        let embeddings = self.embedder.embed_documents(&passages)?;
        let chunks = ranges
            .into_iter()
            .zip(embeddings)
            .map(|(byte_range, embedding)| PreparedChunk {
                byte_range,
                embedding,
            })
            .collect();
        self.retrieval
            .add_document(PreparedDocument { text, chunks })
    }

    /// Rank passages, including their parent document IDs. The limit counts
    /// chunks, not distinct documents. Blank queries error even at limit zero.
    pub fn search(
        &mut self,
        query: &str,
        options: HybridSearchOptions,
    ) -> Result<Vec<ChunkHit>, CollectionError> {
        let query_vector = self.embed_query(query)?;
        self.search_with_vector(query, &query_vector, options)
    }

    /// Read-only retrieval with a precomputed query embedding. The vector must
    /// use this collection's model and query instruction; dimensions are validated.
    /// Blank query text errors even when limit is zero.
    pub fn search_with_vector(
        &self,
        query: &str,
        query_vector: &[f32],
        options: HybridSearchOptions,
    ) -> Result<Vec<ChunkHit>, CollectionError> {
        self.retrieval
            .search_with_vector(query, query_vector, options)
    }
}
