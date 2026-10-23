use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::{
    ChunkingConfig, CollectionError, DocId, EmbeddingConfig, EmbeddingError, Filter, HybridHit,
    IndexError, LexicalIndex, Metadata, SearchScratch, VectorError,
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

/// Collection search settings. Filtering restricts candidates before ranking.
#[derive(Debug, Clone, Copy)]
pub struct SearchOptions<'a> {
    /// Maximum eligible candidates retrieved from EACH index before fusion.
    pub candidate_depth: usize,
    /// Maximum fused results returned, independent of candidate_depth.
    pub limit: usize,
    /// Document-level predicate. Filtered vector searches use exact scoring,
    /// including when the collection has an IVF index and insertion buffer.
    pub filter: Option<&'a Filter<'a>>,
}

impl Default for SearchOptions<'_> {
    fn default() -> Self {
        Self {
            candidate_depth: 50,
            limit: 10,
            filter: None,
        }
    }
}

impl SearchOptions<'_> {
    fn retrieval_depth(self) -> usize {
        if self.limit == 0 {
            0
        } else {
            self.candidate_depth
        }
    }
}

/// Source text and metadata, shared by all of a document's passages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub text: String,
    pub metadata: Metadata,
}

impl Document {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            metadata: Metadata::default(),
        }
    }

    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }
}

impl From<String> for Document {
    fn from(text: String) -> Self {
        Self::new(text)
    }
}

impl From<&str> for Document {
    fn from(text: &str) -> Self {
        Self::new(text)
    }
}

#[derive(Debug)]
pub(crate) struct StoredDocument {
    source: Document,
    chunk_range: Range<usize>,
}

/// Original text and metadata are owned once; chunks reference their source.
#[derive(Debug)]
pub(crate) struct CollectionData {
    pub(crate) documents: Vec<StoredDocument>,
    pub(crate) chunks: Vec<Chunk>,
    pub(crate) bm25: LexicalIndex,
    pub(crate) vectors: CollectionVectors,
}

impl CollectionData {
    fn reserve(&mut self, documents: usize, chunks: usize) {
        self.documents.reserve(documents);
        self.chunks.reserve(chunks);
        self.bm25.reserve(chunks);
        self.vectors.reserve(chunks);
    }

    pub(crate) fn new(dimensions: usize) -> Result<Self, VectorError> {
        Ok(Self {
            documents: Vec::new(),
            chunks: Vec::new(),
            bm25: LexicalIndex::default(),
            vectors: CollectionVectors::new(dimensions)?,
        })
    }

    /// Prepare every document before persistence or publication. Recovery passes
    /// normalized vectors through unchanged to preserve their exact float bits.
    /// Arrays and vectors share this path; returning a row range avoids an ID
    /// vector allocation for single-document inserts.
    pub(crate) fn insert_batch(
        &mut self,
        mut documents: impl AsMut<[PreparedDocument]> + IntoIterator<Item = PreparedDocument>,
        normalize_vectors: bool,
        persist: impl FnOnce(&[PreparedDocument]) -> Result<(), CollectionError>,
    ) -> Result<Range<usize>, CollectionError> {
        let batch = documents.as_mut();
        let start = self.documents.len();
        let end = start
            .checked_add(batch.len())
            .ok_or(IndexError::TooManyDocuments)?;
        if batch.is_empty() {
            return Ok(start..end);
        }
        DocId::try_from(end - 1).map_err(|_| IndexError::TooManyDocuments)?;
        let chunk_count = batch.iter().try_fold(0_usize, |count, doc| {
            count
                .checked_add(doc.chunks.len())
                .ok_or(IndexError::TooManyDocuments)
        })?;
        self.bm25.check_capacity(chunk_count)?;
        for document in &mut *batch {
            validate_document_ranges(
                &document.text,
                document.chunks.iter().map(|chunk| &chunk.byte_range),
            )
            .map_err(CollectionError::InvalidDocument)?;
            for chunk in &mut document.chunks {
                if normalize_vectors {
                    self.vectors.normalize_in_place(&mut chunk.embedding)?;
                } else {
                    self.vectors.validate_normalized(&chunk.embedding)?;
                }
            }
        }
        let lowercase: Vec<_> = batch
            .iter()
            .flat_map(|document| {
                document
                    .chunks
                    .iter()
                    .map(|chunk| document.text[chunk.byte_range.clone()].to_lowercase())
            })
            .collect();
        let mut frequencies = std::collections::HashMap::new();
        let lexical = lowercase
            .iter()
            .map(|text| LexicalIndex::prepare_reusing(text, &mut frequencies))
            .collect::<Result<Vec<_>, _>>()?;
        self.reserve(batch.len(), chunk_count);
        // No fallible preparation may occur after this callback. For persistent
        // collections it appends and syncs the transaction before publishing it.
        persist(batch)?;
        let mut lexical = lexical.into_iter();
        for document in documents {
            let doc_id = self.documents.len() as DocId;
            let chunk_start = self.chunks.len();
            for chunk in document.chunks {
                let chunk_id = self
                    .bm25
                    .append(lexical.next().expect("prepared every passage"));
                self.vectors.append_normalized(chunk_id, &chunk.embedding);
                self.chunks.push(Chunk {
                    doc_id,
                    byte_range: chunk.byte_range,
                });
            }
            self.documents.push(StoredDocument {
                source: Document {
                    text: document.text,
                    metadata: document.metadata,
                },
                chunk_range: chunk_start..self.chunks.len(),
            });
        }
        Ok(start..end)
    }

    pub(crate) fn search(
        &self,
        query: &str,
        query_vector: &[f32],
        options: SearchOptions<'_>,
    ) -> Result<Vec<ChunkHit>, VectorError> {
        self.search_with_workspace(query, query_vector, options, &mut SearchScratch::default())
    }

    fn search_with_workspace(
        &self,
        query: &str,
        vector: &[f32],
        options: SearchOptions<'_>,
        scratch: &mut SearchScratch,
    ) -> Result<Vec<ChunkHit>, VectorError> {
        let mut output = Vec::new();
        self.search_into(query, vector, options, scratch, &mut output)?;
        Ok(output)
    }

    fn search_into(
        &self,
        query: &str,
        vector: &[f32],
        options: SearchOptions<'_>,
        scratch: &mut SearchScratch,
        output: &mut Vec<ChunkHit>,
    ) -> Result<(), VectorError> {
        output.clear();
        let depth = options.retrieval_depth();
        if let Some(filter) = options.filter {
            // Validate even when no documents match or no results are requested.
            crate::vector::normalize_into(vector, self.vectors.dimensions(), &mut scratch.query)?;
            if depth == 0 {
                return Ok(());
            }
            scratch.allowed_documents.clear();
            let mut any_allowed = false;
            for document in &self.documents {
                let allowed = filter.matches(&document.source.metadata);
                scratch.allowed_documents.push(allowed);
                any_allowed |= allowed;
            }
            if !any_allowed {
                return Ok(());
            }
            // Visit only eligible vectors. Each document's chunks are contiguous;
            // no chunk mask, candidate-ID allocation, or rejected dot products.
            let chunks = self
                .documents
                .iter()
                .zip(&scratch.allowed_documents)
                .filter(|(_, allowed)| **allowed)
                .flat_map(|(document, _)| document.chunk_range.clone());
            crate::top_k_into(
                chunks.map(|row| {
                    let doc_id = row as DocId;
                    let vector = self.vectors.vector(doc_id).expect("chunk has a vector");
                    crate::SearchHit {
                        doc_id,
                        score: crate::vector::dot(vector, &scratch.query).clamp(-1.0, 1.0),
                    }
                }),
                depth,
                &mut scratch.heap,
                &mut scratch.vector_hits,
            );
            self.bm25.search_into(
                query,
                depth,
                &mut scratch.bm25,
                &mut scratch.heap,
                &mut scratch.bm25_hits,
                |chunk_id| {
                    scratch.allowed_documents[self.chunks[chunk_id as usize].doc_id as usize]
                },
            );
        } else {
            self.vectors.search_into(vector, depth, scratch)?;
            self.bm25.search_into(
                query,
                depth,
                &mut scratch.bm25,
                &mut scratch.heap,
                &mut scratch.bm25_hits,
                |_| true,
            );
        }
        crate::hybrid::collect_candidates(
            &scratch.bm25_hits,
            &scratch.vector_hits,
            &mut scratch.candidates,
        );
        crate::top_k_into(
            scratch
                .candidates
                .iter()
                .enumerate()
                .map(|(row, hit)| crate::SearchHit {
                    doc_id: row as DocId,
                    score: hit.score,
                }),
            options.limit,
            &mut scratch.heap,
            &mut scratch.selected,
        );
        output.extend(
            scratch
                .selected
                .iter()
                .map(|selected| self.chunk_hit(scratch.candidates[selected.doc_id as usize])),
        );
        Ok(())
    }

    fn chunk_hit(&self, hit: HybridHit) -> ChunkHit {
        ChunkHit {
            chunk_id: ChunkId(hit.doc_id),
            doc_id: self.chunks[hit.doc_id as usize].doc_id,
            score: hit.score,
            bm25_rank: hit.bm25_rank,
            vector_rank: hit.vector_rank,
        }
    }
}

/// One externally prepared passage and its embedding. The byte range refers
/// to the original UTF-8 text in its `PreparedDocument`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedChunk {
    pub byte_range: Range<usize>,
    /// Finite, nonzero vector in the collection's embedding space. Insertion
    /// normalizes it, so callers need not provide a unit vector.
    pub embedding: Vec<f32>,
}

/// A source document with all passages and embeddings prepared by the caller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedDocument {
    pub text: String,
    pub metadata: Metadata,
    /// Nonblank passages in strictly advancing byte-range order. Adjacent
    /// passages may overlap but must cover the entire document without gaps.
    pub chunks: Vec<PreparedChunk>,
}

impl PreparedDocument {
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Model-free, append-only passage retrieval. Owns source text, BM25, vectors,
/// and collection settings; never loads a model or tokenizes with a model tokenizer.
/// Search accepts query vectors generated by the caller and supports shared readers.
#[derive(Debug)]
pub struct RetrievalCollection {
    pub(crate) data: CollectionData,
    embedding: EmbeddingConfig,
    chunking: ChunkingConfig,
}

impl RetrievalCollection {
    /// Reserve storage for an expected number of additional documents and
    /// passages. Does not publish data or change either ID namespace.
    pub fn reserve(&mut self, documents: usize, chunks: usize) {
        self.data.reserve(documents, chunks);
    }

    /// Record the embedding space and chunking provenance for persistence
    /// compatibility. Token counts and model identity cannot be verified from
    /// supplied vectors; the caller must prepare passages using these settings.
    pub fn new(
        embedding: EmbeddingConfig,
        chunking: ChunkingConfig,
    ) -> Result<Self, CollectionError> {
        chunking.validate(embedding.max_tokens)?;
        Ok(Self {
            data: CollectionData::new(embedding.dimensions)?,
            embedding,
            chunking,
        })
    }

    /// Validate all ranges and embeddings before publishing any part of a
    /// document. Returned errors leave indexes and the next IDs unchanged.
    pub fn add_document(&mut self, document: PreparedDocument) -> Result<DocId, CollectionError> {
        let ids = self.data.insert_batch([document], true, |_| Ok(()))?;
        Ok(ids.start as DocId)
    }

    /// Insert an entire batch atomically. Invalid documents leave all indexes
    /// and both ID namespaces unchanged. Empty batches are a no-op.
    pub fn add_documents(
        &mut self,
        documents: Vec<PreparedDocument>,
    ) -> Result<Vec<DocId>, CollectionError> {
        let ids = self.data.insert_batch(documents, true, |_| Ok(()))?;
        Ok(ids.map(|id| id as DocId).collect())
    }

    /// Rank passages using query text for BM25 and a precomputed vector for
    /// semantic retrieval. The vector must use this collection's embedding
    /// model and query instruction; only dimensions and numeric validity can
    /// be checked. Blank queries and invalid vectors error even at limit zero.
    pub fn search_with_vector(
        &self,
        query: &str,
        query_vector: &[f32],
        options: SearchOptions<'_>,
    ) -> Result<Vec<ChunkHit>, CollectionError> {
        if query.trim().is_empty() {
            return Err(EmbeddingError::EmptyText { index: 0 }.into());
        }
        Ok(self.data.search(query, query_vector, options)?)
    }

    /// Read-only retrieval reusing normalization, candidate, and selection
    /// buffers as well as BM25 scores. The returned hits own their storage.
    pub fn search_with_scratch(
        &self,
        query: &str,
        query_vector: &[f32],
        options: SearchOptions<'_>,
        scratch: &mut SearchScratch,
    ) -> Result<Vec<ChunkHit>, CollectionError> {
        if query.trim().is_empty() {
            return Err(EmbeddingError::EmptyText { index: 0 }.into());
        }
        Ok(self
            .data
            .search_with_workspace(query, query_vector, options, scratch)?)
    }

    /// Retrieve into caller-owned result storage, retaining its capacity.
    /// Clears `output` before validation, including when the query errors.
    pub fn search_into(
        &self,
        query: &str,
        query_vector: &[f32],
        options: SearchOptions<'_>,
        scratch: &mut SearchScratch,
        output: &mut Vec<ChunkHit>,
    ) -> Result<(), CollectionError> {
        output.clear();
        if query.trim().is_empty() {
            return Err(EmbeddingError::EmptyText { index: 0 }.into());
        }
        Ok(self
            .data
            .search_into(query, query_vector, options, scratch, output)?)
    }

    /// Retrieve a batch in input order with one workspace per worker.
    /// Worker count is capped by batch size and available CPU parallelism.
    /// One worker runs inline; larger batches use scoped threads per call.
    pub fn search_many_with_vectors(
        &self,
        queries: &[(&str, &[f32])],
        options: SearchOptions<'_>,
        workers: std::num::NonZeroUsize,
    ) -> Result<Vec<Vec<ChunkHit>>, CollectionError> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let workers = workers
            .get()
            .min(queries.len())
            .min(std::thread::available_parallelism().map_or(1, |n| n.get()));
        let run = |batch: &[(&str, &[f32])]| {
            let mut scratch = SearchScratch::default();
            batch
                .iter()
                .map(|(query, vector)| {
                    self.search_with_scratch(query, vector, options, &mut scratch)
                })
                .collect::<Result<Vec<_>, _>>()
        };
        if workers == 1 {
            return run(queries);
        }
        std::thread::scope(|scope| {
            let handles: Vec<_> = queries
                .chunks(queries.len().div_ceil(workers))
                .map(|batch| scope.spawn(move || run(batch)))
                .collect();
            // Join every worker before returning; collecting in input order also
            // makes the first reported validation error deterministic.
            let batches: Vec<_> = handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                })
                .collect();
            let mut output = Vec::with_capacity(queries.len());
            for batch in batches {
                output.extend(batch?);
            }
            Ok(output)
        })
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
        self.data
            .documents
            .get(doc_id as usize)
            .map(|document| document.source.text.as_str())
    }

    /// Borrow the source metadata shared by all passages in this document.
    pub fn metadata(&self, doc_id: DocId) -> Option<&Metadata> {
        self.data
            .documents
            .get(doc_id as usize)
            .map(|document| &document.source.metadata)
    }

    pub fn chunk(&self, chunk_id: ChunkId) -> Option<&Chunk> {
        self.data.chunks.get(chunk_id.0 as usize)
    }

    pub fn chunk_text(&self, chunk_id: ChunkId) -> Option<&str> {
        let chunk = self.chunk(chunk_id)?;
        self.document(chunk.doc_id)?.get(chunk.byte_range.clone())
    }

    pub fn embedding_config(&self) -> &EmbeddingConfig {
        &self.embedding
    }

    pub fn chunking_config(&self) -> ChunkingConfig {
        self.chunking
    }

    /// Changing build settings retrains all chunks; changing only probes keeps
    /// the existing index and insertion buffer. Errors leave the mode unchanged.
    pub fn set_vector_search_mode(
        &mut self,
        mode: VectorSearchMode,
    ) -> Result<(), CollectionError> {
        Ok(self.data.vectors.set_mode(mode)?)
    }

    pub fn vector_search_mode(&self) -> VectorSearchMode {
        self.data.vectors.mode()
    }

    /// Newly inserted chunk vectors awaiting an explicit IVF rebuild. They
    /// are already searchable through the exact insertion buffer.
    pub fn buffered_vector_count(&self) -> usize {
        self.data.vectors.buffered_vector_count()
    }

    /// Synchronously retrain IVF and fold in buffered writes. Requires exclusive
    /// access; a no-op in exact mode. Errors preserve the existing index.
    pub fn rebuild_vector_index(&mut self) -> Result<(), CollectionError> {
        Ok(self.data.vectors.rebuild()?)
    }
}

/// Validate before slicing source text or mutating an index.
pub(crate) fn validate_document_ranges<'a>(
    text: &str,
    ranges: impl IntoIterator<Item = &'a Range<usize>>,
) -> Result<(), &'static str> {
    if text.trim().is_empty() {
        return Err("blank document");
    }
    let mut previous: Option<&Range<usize>> = None;
    for range in ranges {
        if range.start >= range.end
            || text
                .get(range.clone())
                .is_none_or(|passage| passage.trim().is_empty())
        {
            return Err("invalid chunk byte range");
        }
        if let Some(previous) = previous {
            if range.start <= previous.start
                || range.end <= previous.end
                || range.start > previous.end
            {
                return Err("chunks do not cover document in order");
            }
        } else if range.start != 0 {
            return Err("first chunk does not start at zero");
        }
        previous = Some(range);
    }
    if previous.is_none_or(|range| range.end != text.len()) {
        return Err("chunks do not cover entire document");
    }
    Ok(())
}

#[cfg(test)]
mod tests;
