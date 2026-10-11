use std::collections::HashMap;

use crate::{DocId, SearchHit, top_k};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VectorError {
    #[error("vector dimensions must be greater than zero")]
    ZeroDimensions,
    #[error("expected {expected} dimensions, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("vector component {index} is not finite")]
    NonFiniteComponent { index: usize },
    #[error("a zero vector has no direction and cannot be normalized")]
    ZeroVector,
    #[error("document {doc_id} already has a vector")]
    DuplicateDocument { doc_id: DocId },
    #[error("stored vector must have unit length")]
    NotNormalized,
}

/// Exact cosine-similarity search over fixed-size, unit-length vectors.
///
/// Document IDs come from the caller (typically an `InvertedIndex`). IDs may be
/// sparse and inserted in any order. Each ID can be inserted once; this index
/// does not own document text or verify that IDs exist in another index.
#[derive(Debug)]
pub struct VectorIndex {
    dimensions: usize,
    // Row i occupies vectors[i * dimensions .. (i + 1) * dimensions].
    vectors: Vec<f32>,
    doc_ids: Vec<DocId>,
    // Map an external document ID to its row, without requiring dense IDs.
    positions: HashMap<DocId, usize>,
}

impl VectorIndex {
    pub fn new(dimensions: usize) -> Result<Self, VectorError> {
        if dimensions == 0 {
            return Err(VectorError::ZeroDimensions);
        }

        Ok(Self {
            dimensions,
            vectors: Vec::new(),
            doc_ids: Vec::new(),
            positions: HashMap::new(),
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Normalize and append a vector for an existing document ID.
    /// Validation errors leave the index unchanged; the input is not modified.
    pub fn insert(&mut self, doc_id: DocId, vector: &[f32]) -> Result<(), VectorError> {
        if self.positions.contains_key(&doc_id) {
            return Err(VectorError::DuplicateDocument { doc_id });
        }
        let normalized = self.normalize(vector)?;
        self.append_normalized(doc_id, &normalized);
        Ok(())
    }

    /// Internal commit step. The caller must validate the vector and ensure
    /// the document ID is unused before reaching this point.
    pub(crate) fn append_normalized(&mut self, doc_id: DocId, normalized: &[f32]) {
        debug_assert_eq!(normalized.len(), self.dimensions);
        debug_assert!(!self.positions.contains_key(&doc_id));
        let row = self.len();

        self.vectors.extend_from_slice(normalized);
        self.doc_ids.push(doc_id);
        self.positions.insert(doc_id, row);
    }

    /// Borrow the stored, normalized vector for a document.
    pub fn vector(&self, doc_id: DocId) -> Option<&[f32]> {
        let row = *self.positions.get(&doc_id)?;
        let start = row * self.dimensions;

        Some(&self.vectors[start..start + self.dimensions])
    }

    /// Compare the query against every stored vector and return the best hits.
    ///
    /// Scores approximate cosine similarity in [-1, 1]; higher is better.
    /// Negative and zero scores are eligible. Ties prefer lower document IDs.
    /// The query is validated even for an empty index or a zero result limit.
    /// For N vectors of D dimensions, scoring takes O(N * D) work; the bounded
    /// top-k heap uses O(k) space, in addition to the O(D) normalized query.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchHit>, VectorError> {
        let query = self.normalize(query)?;
        let hits = self
            .vectors
            .chunks_exact(self.dimensions)
            .zip(&self.doc_ids)
            .map(|(vector, &doc_id)| {
                // Cosine similarity is just a dot product for unit vectors.
                let score: f64 = vector
                    .iter()
                    .zip(&query)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                SearchHit {
                    doc_id,
                    // Stored f32 unit vectors can round slightly outside [-1, 1].
                    score: score.clamp(-1.0, 1.0),
                }
            });
        Ok(top_k(hits, limit))
    }

    pub(crate) fn normalize(&self, vector: &[f32]) -> Result<Vec<f32>, VectorError> {
        let length = self.vector_length(vector)?;
        Ok(vector
            .iter()
            .map(|&component| (f64::from(component) / length) as f32)
            .collect())
    }

    /// Check a stored vector without changing its float values during restore.
    pub(crate) fn validate_normalized(&self, vector: &[f32]) -> Result<(), VectorError> {
        let length = self.vector_length(vector)?;
        // Allow rounding from f64 normalization back to stored f32 components.
        if (length - 1.0).abs() > 1e-5 {
            return Err(VectorError::NotNormalized);
        }
        Ok(())
    }

    fn vector_length(&self, vector: &[f32]) -> Result<f64, VectorError> {
        if vector.len() != self.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: vector.len(),
            });
        }
        let mut squared_length = 0.0_f64;
        for (index, &component) in vector.iter().enumerate() {
            if !component.is_finite() {
                return Err(VectorError::NonFiniteComponent { index });
            }
            // Promote BEFORE squaring: finite f32 values can overflow or
            // underflow when squared in f32, but their squares fit in f64.
            let component = f64::from(component);
            squared_length += component * component;
        }
        if squared_length == 0.0 {
            return Err(VectorError::ZeroVector);
        }
        Ok(squared_length.sqrt())
    }
}
