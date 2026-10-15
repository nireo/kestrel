use std::ops::Range;
use std::path::Path;

use crate::chunking::Chunker;
use crate::{
    ChunkingConfig, ChunkingError, DocId, EmbeddingConfig, EmbeddingError, HybridSearchOptions,
    IndexError, IvfError, LexicalIndex, SnapshotError, TextEmbedder, VectorError,
    reciprocal_rank_fusion, snapshot,
};

mod vectors;
pub(crate) use vectors::CollectionVectors;
pub use vectors::VectorSearchMode;

/// A passage ID, distinct from the ID of its source document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub doc_id: DocId,
    pub byte_range: Range<usize>,
}

/// A ranked passage. Multiple hits can belong to the same source document.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkHit {
    pub chunk_id: ChunkId,
    pub doc_id: DocId,
    pub score: f64,
    pub bm25_rank: Option<usize>,
    pub vector_rank: Option<usize>,
}

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
}

/// Original text is owned once; chunks reference byte ranges in that text.
#[derive(Debug)]
pub(crate) struct CollectionData {
    pub(crate) documents: Vec<String>,
    pub(crate) chunks: Vec<Chunk>,
    pub(crate) bm25: LexicalIndex,
    pub(crate) vectors: CollectionVectors,
}

impl CollectionData {
    pub(crate) fn new(dimensions: usize) -> Result<Self, VectorError> {
        Ok(Self {
            documents: Vec::new(),
            chunks: Vec::new(),
            bm25: LexicalIndex::default(),
            vectors: CollectionVectors::new(dimensions)?,
        })
    }

    /// Caller validates ranges and normalized vectors. All fallible lexical
    /// preparation happens before changing any of the document/chunk stores.
    pub(crate) fn append_document(
        &mut self,
        text: String,
        ranges: Vec<Range<usize>>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<DocId, IndexError> {
        debug_assert!(!ranges.is_empty());
        debug_assert_eq!(ranges.len(), embeddings.len());
        let doc_id =
            DocId::try_from(self.documents.len()).map_err(|_| IndexError::TooManyDocuments)?;
        self.bm25.check_capacity(ranges.len())?;
        let normalized: Vec<_> = ranges
            .iter()
            .map(|range| text[range.clone()].to_lowercase())
            .collect();
        let prepared: Vec<_> = normalized
            .iter()
            .map(|text| LexicalIndex::prepare(text))
            .collect::<Result<_, _>>()?;
        for ((prepared, range), embedding) in prepared.into_iter().zip(ranges).zip(embeddings) {
            let chunk_id = self.bm25.append(prepared);
            debug_assert_eq!(chunk_id as usize, self.chunks.len());
            self.vectors.append_normalized(chunk_id, &embedding);
            self.chunks.push(Chunk {
                doc_id,
                byte_range: range,
            });
        }
        self.documents.push(text);
        Ok(doc_id)
    }

    pub(crate) fn search(
        &self,
        query: &str,
        query_vector: &[f32],
        options: HybridSearchOptions,
    ) -> Result<Vec<ChunkHit>, VectorError> {
        let depth = if options.limit == 0 {
            0
        } else {
            options.candidate_depth
        };
        let vector_hits = self.vectors.search(query_vector, depth)?;
        let bm25_hits = self.bm25.search(query, depth);
        Ok(
            reciprocal_rank_fusion(&bm25_hits, &vector_hits, options.limit)
                .into_iter()
                .map(|hit| ChunkHit {
                    chunk_id: ChunkId(hit.doc_id),
                    doc_id: self.chunks[hit.doc_id as usize].doc_id,
                    score: hit.score,
                    bm25_rank: hit.bm25_rank,
                    vector_rank: hit.vector_rank,
                })
                .collect(),
        )
    }
}

/// Append-only documents with token-sized searchable passages. BM25 and vector
/// search both rank chunks; len() and document() refer to source documents.
pub struct Collection {
    embedder: TextEmbedder,
    chunker: Chunker,
    data: CollectionData,
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
        let data = CollectionData::new(embedder.dimensions())?;
        Self::from_data(embedder, data, config)
    }

    fn from_data(
        embedder: TextEmbedder,
        data: CollectionData,
        config: ChunkingConfig,
    ) -> Result<Self, CollectionError> {
        let chunker = Chunker::new(
            embedder.tokenizer().clone(),
            config,
            embedder.config().max_tokens,
        )?;
        Ok(Self {
            embedder,
            chunker,
            data,
        })
    }

    /// Save all chunk vectors, including buffered IVF writes. Retrieval mode is
    /// runtime-only; loading starts in exact mode.
    /// Atomic replacement after writing and syncing the temporary file. The
    /// parent directory must exist. SyncAfterReplace means the new file exists
    /// but syncing its parent directory failed (Unix only).
    pub fn save_snapshot(&self, path: impl AsRef<Path>) -> Result<(), CollectionError> {
        Ok(snapshot::save(
            path.as_ref(),
            &self.data,
            self.embedding_config(),
            self.chunking_config(),
        )?)
    }

    /// Restore stored boundaries and vectors without chunking or embedding the
    /// documents again. Version 1 snapshots become one chunk per document.
    /// Retrieval starts in exact mode; select IVF to train a fresh index.
    pub fn load_snapshot(path: impl AsRef<Path>) -> Result<Self, CollectionError> {
        let (data, config) = snapshot::load(path.as_ref(), &crate::embeddings::CONFIG)?;
        Self::from_data(TextEmbedder::new()?, data, config)
    }

    pub fn load_snapshot_with_embedder(
        path: impl AsRef<Path>,
        embedder: TextEmbedder,
    ) -> Result<Self, CollectionError> {
        let (data, config) = snapshot::load(path.as_ref(), embedder.config())?;
        Self::from_data(embedder, data, config)
    }

    pub fn len(&self) -> usize {
        self.data.documents.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.documents.is_empty()
    }
    pub fn chunk_count(&self) -> usize {
        self.data.chunks.len()
    }

    pub fn document(&self, doc_id: DocId) -> Option<&str> {
        self.data.documents.get(doc_id as usize).map(String::as_str)
    }

    pub fn chunk(&self, chunk_id: ChunkId) -> Option<&Chunk> {
        self.data.chunks.get(chunk_id.0 as usize)
    }

    pub fn chunk_text(&self, chunk_id: ChunkId) -> Option<&str> {
        let chunk = self.chunk(chunk_id)?;
        self.document(chunk.doc_id)?.get(chunk.byte_range.clone())
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
        Ok(self.data.vectors.set_mode(mode)?)
    }

    pub fn vector_search_mode(&self) -> VectorSearchMode {
        self.data.vectors.mode()
    }

    /// Number of newly inserted chunk vectors awaiting an explicit IVF rebuild.
    /// They are already searchable through the exact insertion buffer.
    pub fn buffered_vector_count(&self) -> usize {
        self.data.vectors.buffered_vector_count()
    }

    /// Retrain IVF over all chunks and empty its insertion buffer after success.
    /// This synchronous operation needs exclusive access; exact mode is a no-op.
    pub fn rebuild_vector_index(&mut self) -> Result<(), CollectionError> {
        Ok(self.data.vectors.rebuild()?)
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
        index_document(&mut self.data, text, ranges, embeddings)
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
        if query.trim().is_empty() {
            return Err(EmbeddingError::EmptyText { index: 0 }.into());
        }
        Ok(self.data.search(query, query_vector, options)?)
    }
}

fn index_document(
    data: &mut CollectionData,
    text: String,
    ranges: Vec<Range<usize>>,
    embeddings: Vec<Vec<f32>>,
) -> Result<DocId, CollectionError> {
    let normalized: Vec<_> = embeddings
        .iter()
        .map(|embedding| data.vectors.normalize(embedding))
        .collect::<Result<_, _>>()?;
    Ok(data.append_document(text, ranges, normalized)?)
}

#[cfg(test)]
mod tests;
