use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::{
    Bm25Scratch, ChunkingConfig, CollectionError, DocId, EmbeddingConfig, EmbeddingError,
    HybridSearchOptions, IndexError, LexicalIndex, VectorError, reciprocal_rank_fusion,
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
            let ranges: Vec<_> = document
                .chunks
                .iter()
                .map(|chunk| chunk.byte_range.clone())
                .collect();
            validate_document_ranges(&document.text, &ranges)
                .map_err(CollectionError::InvalidDocument)?;
            for chunk in &mut document.chunks {
                if normalize_vectors {
                    chunk.embedding = self.vectors.normalize(&chunk.embedding)?;
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
        let lexical = lowercase
            .iter()
            .map(|text| LexicalIndex::prepare(text))
            .collect::<Result<Vec<_>, _>>()?;
        // No fallible preparation may occur after this callback. For persistent
        // collections it appends and syncs the transaction before publishing it.
        persist(batch)?;
        let mut lexical = lexical.into_iter();
        for document in documents {
            let doc_id = self.documents.len() as DocId;
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
            self.documents.push(document.text);
        }
        Ok(start..end)
    }

    pub(crate) fn search(
        &self,
        query: &str,
        query_vector: &[f32],
        options: HybridSearchOptions,
    ) -> Result<Vec<ChunkHit>, VectorError> {
        self.search_with_scratch(query, query_vector, options, None)
    }

    fn search_with_scratch(
        &self,
        query: &str,
        query_vector: &[f32],
        options: HybridSearchOptions,
        scratch: Option<&mut Bm25Scratch>,
    ) -> Result<Vec<ChunkHit>, VectorError> {
        let depth = if options.limit == 0 {
            0
        } else {
            options.candidate_depth
        };
        let vector_hits = self.vectors.search(query_vector, depth)?;
        let bm25_hits = match scratch {
            Some(scratch) => self.bm25.search_with_scratch(query, depth, scratch),
            None => self.bm25.search(query, depth),
        };
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
    /// Nonblank passages in strictly advancing byte-range order. Adjacent
    /// passages may overlap but must cover the entire document without gaps.
    pub chunks: Vec<PreparedChunk>,
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
        options: HybridSearchOptions,
    ) -> Result<Vec<ChunkHit>, CollectionError> {
        if query.trim().is_empty() {
            return Err(EmbeddingError::EmptyText { index: 0 }.into());
        }
        Ok(self.data.search(query, query_vector, options)?)
    }

    /// Read-only retrieval with reusable BM25 score storage. Keep one workspace
    /// per reader to avoid per-match hashing without serializing shared reads.
    /// Validation and ranking match `search_with_vector`.
    pub fn search_with_vector_and_scratch(
        &self,
        query: &str,
        query_vector: &[f32],
        options: HybridSearchOptions,
        scratch: &mut Bm25Scratch,
    ) -> Result<Vec<ChunkHit>, CollectionError> {
        if query.trim().is_empty() {
            return Err(EmbeddingError::EmptyText { index: 0 }.into());
        }
        Ok(self
            .data
            .search_with_scratch(query, query_vector, options, Some(scratch))?)
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
pub(crate) fn validate_document_ranges(
    text: &str,
    ranges: &[Range<usize>],
) -> Result<(), &'static str> {
    if text.trim().is_empty() {
        return Err("blank document");
    }
    for (index, range) in ranges.iter().enumerate() {
        if range.start >= range.end
            || text
                .get(range.clone())
                .is_none_or(|passage| passage.trim().is_empty())
        {
            return Err("invalid chunk byte range");
        }
        if index == 0 {
            if range.start != 0 {
                return Err("first chunk does not start at zero");
            }
        } else {
            let previous = &ranges[index - 1];
            if range.start <= previous.start
                || range.end <= previous.end
                || range.start > previous.end
            {
                return Err("chunks do not cover document in order");
            }
        }
    }
    if ranges.last().is_none_or(|range| range.end != text.len()) {
        return Err("chunks do not cover entire document");
    }
    Ok(())
}

#[cfg(test)]
mod tests;
