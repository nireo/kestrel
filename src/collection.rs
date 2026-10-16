use crate::chunking::Chunker;
use crate::{
    Chunk, ChunkHit, ChunkId, ChunkingConfig, ChunkingError, DocId, EmbeddingConfig,
    EmbeddingError, HybridSearchOptions, IndexError, IvfError, PreparedChunk, PreparedDocument,
    RetrievalCollection, StorageError, TextEmbedder, VectorError, VectorSearchMode,
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
    Storage(#[from] StorageError),
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

    /// Chunk and embed a document without inserting it. Prepared output can be
    /// passed to either `RetrievalCollection` or `PersistentCollection`.
    pub fn prepare_document(
        &mut self,
        text: impl Into<String>,
    ) -> Result<PreparedDocument, CollectionError> {
        Ok(self.prepare_documents([text])?.remove(0))
    }

    /// Prepare a batch with embedding inference shared across documents. No
    /// collection data is changed, including when preparation fails.
    pub fn prepare_documents<I, S>(
        &mut self,
        texts: I,
    ) -> Result<Vec<PreparedDocument>, CollectionError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let texts: Vec<String> = texts.into_iter().map(Into::into).collect();
        let mut ranges = Vec::with_capacity(texts.len());
        for (index, text) in texts.iter().enumerate() {
            if text.trim().is_empty() {
                return Err(EmbeddingError::EmptyText { index }.into());
            }
            ranges.push(self.chunker.ranges(text)?);
        }
        let passages: Vec<_> = texts
            .iter()
            .zip(&ranges)
            .flat_map(|(text, ranges)| ranges.iter().map(|range| &text[range.clone()]))
            .collect();
        let mut embeddings = self.embedder.embed_documents(&passages)?.into_iter();
        Ok(texts
            .into_iter()
            .zip(ranges)
            .map(|(text, ranges)| PreparedDocument {
                text,
                chunks: ranges
                    .into_iter()
                    .map(|byte_range| PreparedChunk {
                        byte_range,
                        embedding: embeddings.next().expect("embedder validated passage count"),
                    })
                    .collect(),
            })
            .collect())
    }

    /// Split and embed all passages before publishing the document.
    pub fn add_document(&mut self, text: impl Into<String>) -> Result<DocId, CollectionError> {
        let document = self.prepare_document(text)?;
        self.retrieval.add_document(document)
    }

    /// Batch local chunking and embedding, then atomically insert every document.
    pub fn add_documents<I, S>(&mut self, texts: I) -> Result<Vec<DocId>, CollectionError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let documents = self.prepare_documents(texts)?;
        self.retrieval.add_documents(documents)
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
