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
    // Consecutive IDs (including IVF insertion buffers) need no lookup table.
    // Lazily fall back to a map when callers supply sparse or unordered IDs.
    positions: Option<HashMap<DocId, usize>>,
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
            positions: None,
        })
    }

    /// Allocate storage for a known number of vectors up front. As with Vec,
    /// an overflowing or unallocatable capacity panics.
    pub fn with_capacity(dimensions: usize, capacity: usize) -> Result<Self, VectorError> {
        let mut index = Self::new(dimensions)?;
        index.reserve(capacity);
        Ok(index)
    }

    /// Reserve room for additional rows without changing IDs or indexed data.
    pub fn reserve(&mut self, additional: usize) {
        let components = additional
            .checked_mul(self.dimensions)
            .expect("vector capacity overflow");
        self.vectors.reserve(components);
        self.doc_ids.reserve(additional);
        if let Some(positions) = &mut self.positions {
            positions.reserve(additional);
        }
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
        if self.position(doc_id).is_some() {
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
        debug_assert!(self.position(doc_id).is_none());
        let row = self.len();

        if self.positions.is_none()
            && self
                .doc_ids
                .first()
                .is_some_and(|&first| u64::from(doc_id) != u64::from(first) + row as u64)
        {
            self.positions = Some(
                self.doc_ids
                    .iter()
                    .enumerate()
                    .map(|(row, &id)| (id, row))
                    .collect(),
            );
        }

        self.vectors.extend_from_slice(normalized);
        self.doc_ids.push(doc_id);
        if let Some(positions) = &mut self.positions {
            positions.insert(doc_id, row);
        }
    }

    /// Borrow the stored, normalized vector for a document.
    pub fn vector(&self, doc_id: DocId) -> Option<&[f32]> {
        let row = self.position(doc_id)?;
        let start = row * self.dimensions;

        Some(&self.vectors[start..start + self.dimensions])
    }

    fn position(&self, doc_id: DocId) -> Option<usize> {
        if let Some(positions) = &self.positions {
            return positions.get(&doc_id).copied();
        }
        let row = doc_id.checked_sub(*self.doc_ids.first()?)? as usize;
        (row < self.len()).then_some(row)
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
        Ok(self.search_normalized(&query, limit))
    }

    /// Internal retrieval after the caller has validated and normalized the query.
    pub(crate) fn search_normalized(&self, query: &[f32], limit: usize) -> Vec<SearchHit> {
        let hits = self
            .vectors
            .chunks_exact(self.dimensions)
            .zip(&self.doc_ids)
            .map(|(vector, &doc_id)| {
                // Cosine similarity is just a dot product for unit vectors.
                let score = dot(vector, query);
                SearchHit {
                    doc_id,
                    // Stored f32 unit vectors can round slightly outside [-1, 1].
                    score: score.clamp(-1.0, 1.0),
                }
            });
        top_k(hits, limit)
    }

    pub(crate) fn normalize(&self, vector: &[f32]) -> Result<Vec<f32>, VectorError> {
        normalize(vector, self.dimensions)
    }

    pub(crate) fn row(&self, row: usize) -> (DocId, &[f32]) {
        let start = row * self.dimensions;
        (
            self.doc_ids[row],
            &self.vectors[start..start + self.dimensions],
        )
    }
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f64 {
    // Independent lanes break the serial accumulation dependency and allow
    // auto-vectorization. Promote before multiplying to retain f64 precision.
    debug_assert_eq!(a.len(), b.len());
    let mut sums = [0.0_f64; 4];
    let mut a_chunks = a.chunks_exact(4);
    let mut b_chunks = b.chunks_exact(4);
    for (a, b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for lane in 0..4 {
            sums[lane] += f64::from(a[lane]) * f64::from(b[lane]);
        }
    }
    let tail: f64 = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder())
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    sums.into_iter().sum::<f64>() + tail
}

pub(crate) fn normalize(vector: &[f32], dimensions: usize) -> Result<Vec<f32>, VectorError> {
    let length = vector_length(vector, dimensions)?;
    Ok(vector
        .iter()
        .map(|&component| (f64::from(component) / length) as f32)
        .collect())
}

pub(crate) fn validate_normalized(vector: &[f32], dimensions: usize) -> Result<(), VectorError> {
    let length = vector_length(vector, dimensions)?;
    // Allow rounding from f64 normalization back to stored f32 components.
    if (length - 1.0).abs() > 1e-5 {
        return Err(VectorError::NotNormalized);
    }
    Ok(())
}

fn vector_length(vector: &[f32], dimensions: usize) -> Result<f64, VectorError> {
    if vector.len() != dimensions {
        return Err(VectorError::DimensionMismatch {
            expected: dimensions,
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
