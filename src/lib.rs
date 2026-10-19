use std::borrow::Cow;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::ops::Range;

mod chunking;
mod collection;
mod embeddings;
mod hybrid;
mod ivf;
mod retrieval;
mod storage;
mod vector;

pub use chunking::{ChunkingConfig, ChunkingError};
pub use collection::{Collection, CollectionError};
pub use embeddings::{EmbeddingConfig, EmbeddingError, TextEmbedder};
pub use hybrid::{
    HybridHit, HybridSearchOptions, hybrid_search, hybrid_search_ivf, reciprocal_rank_fusion,
};
pub use ivf::{IvfBuildOptions, IvfError, IvfIndex};
pub use retrieval::{
    Chunk, ChunkHit, ChunkId, PreparedChunk, PreparedDocument, RetrievalCollection,
    VectorSearchMode,
};
pub use storage::{PersistentCollection, StorageError};
pub use vector::{VectorError, VectorIndex};

pub type DocId = u32;

/// One term's occurrences in one document. Exactly 8 bytes, without pointers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Posting {
    pub doc_id: DocId,
    pub term_frequency: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchHit {
    pub doc_id: DocId,
    pub score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum IndexError {
    #[error("document IDs no longer fit in u32")]
    TooManyDocuments,
    #[error("document token count exceeds u32::MAX")]
    DocumentTooLong,
}

/// An in-memory index with stable IDs and no updates or deletions.
///
/// Text is lowercased, then split on non-alphanumeric characters. Indexing and
/// querying use the same tokenizer. Empty documents are stored and count toward
/// the corpus size and average document length.
#[derive(Debug, Default)]
pub struct InvertedIndex {
    core: LexicalIndex,
    documents: Vec<String>,
}

/// Reusable BM25 workspace for one reader. Share the index across threads and
/// keep a separate workspace per reader. Retains O(N) score capacity for the
/// largest index searched; only matched rows are cleared after scoring.
#[derive(Debug, Default)]
pub struct Bm25Scratch {
    normalized: String,
    seen: Vec<Range<usize>>,
    scores: Vec<f64>,
    touched: Vec<DocId>,
}

/// Scoring data shared by standalone document search and collection chunks.
/// It owns no source text. IDs are dense row numbers chosen by insertion order.
#[derive(Debug, Default)]
pub(crate) struct LexicalIndex {
    terms: HashMap<Box<str>, Vec<Posting>>,
    document_lengths: Vec<u32>,
    total_tokens: u64,
}

pub(crate) struct PreparedText<'a> {
    frequencies: HashMap<&'a str, u32>,
    length: u32,
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a document, assigning its position in the document store as its ID.
    /// Errors leave the index unchanged.
    pub fn add_document(&mut self, text: impl Into<String>) -> Result<DocId, IndexError> {
        self.core.check_capacity(1)?;
        let text = text.into();
        let normalized = text.to_lowercase();
        let prepared = LexicalIndex::prepare(&normalized)?;
        let doc_id = self.core.append(prepared);
        self.documents.push(text);
        Ok(doc_id)
    }

    pub fn len(&self) -> usize {
        self.core.len()
    }

    pub fn is_empty(&self) -> bool {
        self.core.is_empty()
    }

    pub fn vocabulary_len(&self) -> usize {
        self.core.terms.len()
    }

    pub fn document(&self, doc_id: DocId) -> Option<&str> {
        self.documents.get(doc_id as usize).map(String::as_str)
    }

    pub fn document_length(&self, doc_id: DocId) -> Option<u32> {
        self.core.document_lengths.get(doc_id as usize).copied()
    }

    pub fn average_document_length(&self) -> f64 {
        self.core.average_document_length()
    }

    /// Inspect a posting list using an already normalized, single term.
    pub fn postings(&self, term: &str) -> &[Posting] {
        self.core.postings(term)
    }

    /// Rank documents matching ANY query term using BM25 (k1 = 1.2, b = 0.75).
    /// Repeated terms count once. Ties prefer lower IDs; scores are not probabilities.
    pub fn search(&self, query: &str, limit: usize) -> Vec<SearchHit> {
        self.core.search(query, limit)
    }

    /// Search with reusable dense score storage, avoiding per-match hashing.
    /// Results and validation match `search`; the workspace may be reused with
    /// other indexes and after appending documents.
    pub fn search_with_scratch(
        &self,
        query: &str,
        limit: usize,
        scratch: &mut Bm25Scratch,
    ) -> Vec<SearchHit> {
        self.core.search_with_scratch(query, limit, scratch)
    }
}

impl LexicalIndex {
    pub(crate) fn check_capacity(&self, additional: usize) -> Result<(), IndexError> {
        let count = self
            .len()
            .checked_add(additional)
            .ok_or(IndexError::TooManyDocuments)?;
        if count != 0 {
            DocId::try_from(count - 1).map_err(|_| IndexError::TooManyDocuments)?;
        }
        Ok(())
    }

    /// Borrow counts from a caller-owned lowercase buffer. Prepare all chunks
    /// before committing any, without allocating dictionary keys per document.
    pub(crate) fn prepare(normalized: &str) -> Result<PreparedText<'_>, IndexError> {
        let mut frequencies = HashMap::<&str, u32>::new();
        let mut length = 0_u32;

        // Borrow terms from one normalized buffer instead of allocating a
        // separate String for every token. Repeated tokens share one count.
        for term in tokens(normalized) {
            length = length.checked_add(1).ok_or(IndexError::DocumentTooLong)?;
            *frequencies.entry(term).or_default() += 1;
        }

        Ok(PreparedText {
            frequencies,
            length,
        })
    }

    pub(crate) fn append(&mut self, prepared: PreparedText<'_>) -> DocId {
        debug_assert!(self.check_capacity(1).is_ok());
        let doc_id = self.len() as DocId;

        for (term, term_frequency) in prepared.frequencies {
            let posting = Posting {
                doc_id,
                term_frequency,
            };

            // Allocate a dictionary key only when this term is new to the index.
            if let Some(postings) = self.terms.get_mut(term) {
                postings.push(posting);
            } else {
                self.terms.insert(term.into(), vec![posting]);
            }
        }

        // IDs only increase, so every posting list stays sorted without sorting.
        self.document_lengths.push(prepared.length);
        self.total_tokens += u64::from(prepared.length);
        doc_id
    }

    pub fn len(&self) -> usize {
        self.document_lengths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.document_lengths.is_empty()
    }

    pub fn average_document_length(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.total_tokens as f64 / self.len() as f64
        }
    }

    /// Inspect a posting list using an already normalized, single term.
    pub fn postings(&self, term: &str) -> &[Posting] {
        self.terms.get(term).map_or(&[], Vec::as_slice)
    }

    /// Rank documents matching ANY query term using BM25 (k1 = 1.2, b = 0.75).
    ///
    /// Repeated query terms count once. Results have descending scores, with
    /// lower document IDs first on ties. Unknown terms contribute nothing.
    /// Scores are relevance weights, not probabilities.
    pub fn search(&self, query: &str, limit: usize) -> Vec<SearchHit> {
        if limit == 0 || self.total_tokens == 0 {
            return Vec::new();
        }

        let normalized = if query.is_ascii() && !query.bytes().any(|b| b.is_ascii_uppercase()) {
            Cow::Borrowed(query)
        } else {
            Cow::Owned(query.to_lowercase())
        };
        let mut seen = Vec::new();
        let mut scores = HashMap::<DocId, f64>::new();
        self.accumulate(&normalized, &mut seen, |doc_id, contribution| {
            *scores.entry(doc_id).or_default() += contribution;
        });

        top_k(
            scores
                .into_iter()
                .map(|(doc_id, score)| SearchHit { doc_id, score }),
            limit,
        )
    }

    pub(crate) fn search_with_scratch(
        &self,
        query: &str,
        limit: usize,
        scratch: &mut Bm25Scratch,
    ) -> Vec<SearchHit> {
        scratch.seen.clear();
        if limit == 0 || self.total_tokens == 0 {
            return Vec::new();
        }
        scratch.normalized.clear();
        if query.is_ascii() {
            scratch.normalized.push_str(query);
            scratch.normalized.make_ascii_lowercase();
        } else {
            // Use str's contextual Unicode lowercasing, matching indexing.
            scratch.normalized.push_str(&query.to_lowercase());
        }
        if scratch.scores.len() < self.len() {
            scratch.scores.resize(self.len(), 0.0);
        }
        self.accumulate(
            &scratch.normalized,
            &mut scratch.seen,
            |id, contribution| {
                let score = &mut scratch.scores[id as usize];
                if *score == 0.0 {
                    scratch.touched.push(id);
                }
                *score += contribution;
            },
        );
        top_k(
            scratch.touched.drain(..).map(|doc_id| {
                let score = std::mem::replace(&mut scratch.scores[doc_id as usize], 0.0);
                SearchHit { doc_id, score }
            }),
            limit,
        )
    }

    fn accumulate(
        &self,
        normalized: &str,
        seen: &mut Vec<Range<usize>>,
        mut add: impl FnMut(DocId, f64),
    ) {
        const K1: f64 = 1.2;
        const B: f64 = 0.75;
        let document_count = self.len() as f64;
        let length_scale = K1 * B / self.average_document_length();
        let length_base = K1 * (1.0 - B);
        let mut many_terms = None;

        for term in tokens(normalized) {
            // A short query needs no hash table. Switch to hashing for long
            // queries so duplicate detection does not become quadratic.
            if seen.len() == 16 {
                let terms = many_terms.get_or_insert_with(|| {
                    seen.iter()
                        .map(|range| &normalized[range.clone()])
                        .collect::<HashSet<_>>()
                });
                if !terms.insert(term) {
                    continue;
                }
            } else {
                if seen.iter().any(|range| &normalized[range.clone()] == term) {
                    continue;
                }
                let start = term.as_ptr() as usize - normalized.as_ptr() as usize;
                seen.push(start..start + term.len());
            }
            let postings = self.postings(term);
            if postings.is_empty() {
                continue;
            }

            let document_frequency = postings.len() as f64;
            // Positive IDF variant: common terms still have nonnegative weight.
            let idf =
                ((document_count - document_frequency + 0.5) / (document_frequency + 0.5)).ln_1p();
            let weight = idf * (K1 + 1.0);

            for posting in postings {
                let tf = f64::from(posting.term_frequency);
                let length = f64::from(self.document_lengths[posting.doc_id as usize]);
                let contribution = weight * tf / (tf + length_base + length_scale * length);
                add(posting.doc_id, contribution);
            }
        }
    }
}

fn tokens(normalized: &str) -> impl Iterator<Item = &str> {
    normalized
        .split(|c: char| !c.is_alphanumeric())
        .filter(|term| !term.is_empty())
}

// Bump this when tokenization changes so stored collections cannot silently rebuild a
// different lexical index. Independent of the storage format version.
const TOKENIZER_VERSION: u32 = 1;

/// Shared by lexical and vector search. Reverse puts the worst retained hit at
/// the root, so a better candidate can replace it without sorting every match.
fn top_k(hits: impl IntoIterator<Item = SearchHit>, limit: usize) -> Vec<SearchHit> {
    if limit == 0 {
        return Vec::new();
    }
    let hits = hits.into_iter();
    let mut best = BinaryHeap::with_capacity(limit.min(hits.size_hint().0));
    for hit in hits {
        let candidate = Reverse(RankedHit(hit));
        if best.len() < limit {
            best.push(candidate);
        } else if candidate < *best.peek().expect("nonempty heap: limit is positive") {
            *best.peek_mut().expect("nonempty heap: limit is positive") = candidate;
        }
    }
    best.into_sorted_vec()
        .into_iter()
        .map(|hit| hit.0.0)
        .collect()
}

/// Larger means better: higher score, then lower document ID.
#[derive(Debug, Clone, Copy)]
struct RankedHit(SearchHit);

impl Ord for RankedHit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .score
            .total_cmp(&other.0.score)
            .then_with(|| other.0.doc_id.cmp(&self.0.doc_id))
    }
}

impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for RankedHit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for RankedHit {}
