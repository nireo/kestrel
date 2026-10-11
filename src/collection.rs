use std::path::Path;

use crate::{
    DocId, EmbeddingConfig, EmbeddingError, HybridHit, HybridSearchOptions, IndexError,
    InvertedIndex, SnapshotError, TextEmbedder, VectorError, VectorIndex, hybrid_search, snapshot,
};

#[derive(Debug, thiserror::Error)]
pub enum CollectionError {
    #[error(transparent)]
    Embedding(#[from] EmbeddingError),
    #[error(transparent)]
    Index(#[from] IndexError),
    #[error(transparent)]
    Vector(#[from] VectorError),
    #[error(transparent)]
    Snapshot(#[from] SnapshotError),
}

/// An append-only, in-memory collection that coordinates embedding and search.
/// Each document has one vector under the same ID in both indexes. Private
/// indexes keep callers from changing one without updating the other.
pub struct Collection {
    embedder: TextEmbedder,
    bm25: InvertedIndex,
    vectors: VectorIndex,
}

impl Collection {
    /// Create an empty collection and load its reusable embedding model.
    pub fn new() -> Result<Self, CollectionError> {
        Self::from_embedder(TextEmbedder::new()?)
    }

    /// Use an already loaded embedder, for example one with a custom cache.
    pub fn from_embedder(embedder: TextEmbedder) -> Result<Self, CollectionError> {
        let vectors = VectorIndex::new(embedder.dimensions())?;
        Ok(Self {
            embedder,
            bm25: InvertedIndex::new(),
            vectors,
        })
    }

    /// Save documents and their existing embeddings, replacing any previous
    /// snapshot only after a temporary file has been fully written and synced.
    /// The parent directory must already exist. On Unix it is synced after
    /// replacement; a SyncAfterReplace error means replacement already occurred.
    pub fn save_snapshot(&self, path: impl AsRef<Path>) -> Result<(), CollectionError> {
        Ok(snapshot::save(
            path.as_ref(),
            &self.bm25,
            &self.vectors,
            self.embedding_config(),
        )?)
    }

    /// Restore stable IDs and stored embeddings, rebuilding BM25 from text.
    /// Validate the snapshot before loading the model; document embeddings are
    /// never recomputed. Model files still need to be cached or downloaded.
    pub fn load_snapshot(path: impl AsRef<Path>) -> Result<Self, CollectionError> {
        let (bm25, vectors) = snapshot::load(path.as_ref(), &crate::embeddings::CONFIG)?;
        Ok(Self {
            embedder: TextEmbedder::new()?,
            bm25,
            vectors,
        })
    }

    /// Restore using a model that is already loaded, including a custom cache.
    pub fn load_snapshot_with_embedder(
        path: impl AsRef<Path>,
        embedder: TextEmbedder,
    ) -> Result<Self, CollectionError> {
        let (bm25, vectors) = snapshot::load(path.as_ref(), embedder.config())?;
        Ok(Self {
            embedder,
            bm25,
            vectors,
        })
    }

    pub fn len(&self) -> usize {
        self.bm25.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bm25.is_empty()
    }

    pub fn document(&self, doc_id: DocId) -> Option<&str> {
        self.bm25.document(doc_id)
    }

    pub fn embedding_config(&self) -> &'static EmbeddingConfig {
        self.embedder.config()
    }

    /// Embed and index one document, assigning its stable ID automatically.
    /// Returned errors leave document storage and both indexes unchanged.
    /// Text is subject to the embedder's blank-text and token-limit policies.
    pub fn add_document(&mut self, text: impl Into<String>) -> Result<DocId, CollectionError> {
        let text = text.into();
        let mut embeddings = self.embedder.embed_documents(&[text.as_str()])?;

        // embed_documents checks that exactly one vector was returned.
        let embedding = embeddings.remove(0);
        index_document(&mut self.bm25, &mut self.vectors, text, &embedding)
    }

    /// Embed the query once, retrieve from both indexes, and fuse their ranks.
    /// Needs mutable access because the inference runtime uses mutable state.
    /// Blank queries return an error, including when depth or limit is zero.
    pub fn search(
        &mut self,
        query: &str,
        options: HybridSearchOptions,
    ) -> Result<Vec<HybridHit>, CollectionError> {
        let query_vector = self.embedder.embed_query(query)?;
        Ok(hybrid_search(
            &self.bm25,
            &self.vectors,
            query,
            &query_vector,
            options,
        )?)
    }
}

fn index_document(
    bm25: &mut InvertedIndex,
    vectors: &mut VectorIndex,
    text: String,
    embedding: &[f32],
) -> Result<DocId, CollectionError> {
    debug_assert_eq!(bm25.len(), vectors.len());

    // Validate before changing either index. BM25 also checks its limits before
    // mutation; once it succeeds, the validated vector append cannot error.
    let normalized = vectors.normalize(embedding)?;
    let doc_id = bm25.add_document(text)?;
    vectors.append_normalized(doc_id, &normalized);

    Ok(doc_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinated_insertion_assigns_matching_ids_and_normalizes_vectors() {
        let mut bm25 = InvertedIndex::new();
        let mut vectors = VectorIndex::new(2).unwrap();
        let first = index_document(&mut bm25, &mut vectors, "rust".into(), &[3.0, 4.0]).unwrap();
        let second = index_document(&mut bm25, &mut vectors, "engine".into(), &[0.0, 1.0]).unwrap();
        assert_eq!((first, second), (0, 1));
        assert_eq!(bm25.len(), vectors.len());
        assert_eq!(bm25.document(first), Some("rust"));
        assert_eq!(vectors.vector(first), Some(&[0.6, 0.8][..]));
        assert_eq!(bm25.search("rust", 1)[0].doc_id, first);
        assert_eq!(vectors.search(&[1.0, 0.0], 1).unwrap()[0].doc_id, first);
    }

    #[test]
    fn invalid_embeddings_leave_existing_text_postings_vectors_and_ids_unchanged() {
        let mut bm25 = InvertedIndex::new();
        let mut vectors = VectorIndex::new(2).unwrap();
        index_document(&mut bm25, &mut vectors, "rust".into(), &[1.0, 0.0]).unwrap();
        let lexical_before = bm25.search("rust", 10);
        let vector_before = vectors.search(&[1.0, 0.0], 10).unwrap();
        for (embedding, expected) in [
            (
                vec![1.0],
                VectorError::DimensionMismatch {
                    expected: 2,
                    actual: 1,
                },
            ),
            (vec![0.0, 0.0], VectorError::ZeroVector),
            (
                vec![f32::NAN, 1.0],
                VectorError::NonFiniteComponent { index: 0 },
            ),
            (
                vec![1.0, f32::INFINITY],
                VectorError::NonFiniteComponent { index: 1 },
            ),
        ] {
            let result = index_document(
                &mut bm25,
                &mut vectors,
                "engine rust rust".into(),
                &embedding,
            );
            match result {
                Err(CollectionError::Vector(error)) => assert_eq!(error, expected),
                other => panic!("expected vector error, got {other:?}"),
            }
            assert_eq!(bm25.len(), 1);
            assert_eq!(vectors.len(), 1);
            assert_eq!(bm25.document(0), Some("rust"));
            assert!(bm25.document(1).is_none());
            assert!(bm25.postings("engine").is_empty());
            assert!(vectors.vector(1).is_none());
            assert_eq!(bm25.search("rust", 10), lexical_before);
            assert_eq!(vectors.search(&[1.0, 0.0], 10).unwrap(), vector_before);
        }
        assert_eq!(
            index_document(&mut bm25, &mut vectors, "engine".into(), &[0.0, 1.0]).unwrap(),
            1
        );
    }
}
