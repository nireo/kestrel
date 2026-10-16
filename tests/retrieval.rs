#![allow(
    clippy::single_range_in_vec_init,
    reason = "each element is one chunk's byte range"
)]

use std::ops::Range;

use kestrel::{
    ChunkId, ChunkingConfig, ChunkingError, CollectionError, EmbeddingConfig, EmbeddingError,
    HybridSearchOptions, IvfBuildOptions, PreparedChunk, PreparedDocument, RetrievalCollection,
    SnapshotError, VectorError, VectorSearchMode,
};

const CONFIG: EmbeddingConfig = EmbeddingConfig {
    model_id: "external-test-model",
    artifact_repo: "no-download",
    artifact_file: "unused.onnx",
    dimensions: 2,
    max_tokens: 512,
    query_instruction: "query: ",
};

fn collection() -> RetrievalCollection {
    RetrievalCollection::new(CONFIG, ChunkingConfig::default()).unwrap()
}

fn prepared(text: &str, ranges: Vec<Range<usize>>, vectors: Vec<Vec<f32>>) -> PreparedDocument {
    assert_eq!(ranges.len(), vectors.len());
    PreparedDocument {
        text: text.into(),
        chunks: ranges
            .into_iter()
            .zip(vectors)
            .map(|(byte_range, embedding)| PreparedChunk {
                byte_range,
                embedding,
            })
            .collect(),
    }
}

fn single(text: &str, vector: [f32; 2]) -> PreparedDocument {
    prepared(text, vec![0..text.len()], vec![vector.to_vec()])
}

fn ivf_mode() -> VectorSearchMode {
    VectorSearchMode::Ivf {
        build: IvfBuildOptions {
            clusters: 2,
            ..IvfBuildOptions::default()
        },
        probes: usize::MAX,
    }
}

#[test]
fn external_embeddings_rank_passages_and_preserve_source_text() {
    let mut core = collection();
    assert!(core.is_empty());
    // Overlapping passages reference the original Unicode text, with no model
    // or tokenizer involved. Unnormalized input vectors are accepted.
    let text = "café rust engine";
    assert_eq!(
        core.add_document(prepared(
            text,
            vec![0..10, 6..17],
            vec![vec![3.0, 4.0], vec![0.0, 5.0]],
        ))
        .unwrap(),
        0
    );
    assert_eq!(core.add_document(single("pasta", [-1.0, 0.0])).unwrap(), 1);
    assert_eq!(core.len(), 2);
    assert_eq!(core.chunk_count(), 3);
    assert_eq!(core.document(0), Some(text));
    assert_eq!(core.chunk_text(ChunkId(0)), Some("café rust"));
    assert_eq!(core.chunk_text(ChunkId(1)), Some("rust engine"));
    assert_eq!(core.chunk(ChunkId(1)).unwrap().doc_id, 0);
    assert!(core.document(2).is_none());
    assert!(core.chunk(ChunkId(3)).is_none());
    assert!(core.chunk_text(ChunkId(3)).is_none());
    let hits = core
        .search_with_vector("engine", &[0.0, 2.0], HybridSearchOptions::default())
        .unwrap();
    assert_eq!(hits[0].chunk_id, ChunkId(1));
    assert_eq!(hits[0].doc_id, 0);
    assert_eq!(hits[0].bm25_rank, Some(1));
    assert_eq!(hits[0].vector_rank, Some(1));
    assert_eq!(hits.iter().filter(|hit| hit.doc_id == 0).count(), 2);
}

#[test]
fn malformed_prepared_documents_leave_exact_and_ivf_indexes_unchanged() {
    for mode in [VectorSearchMode::Exact, ivf_mode()] {
        let mut core = collection();
        core.add_document(single("base", [1.0, 0.0])).unwrap();
        core.set_vector_search_mode(mode).unwrap();
        core.add_document(single("fresh", [0.0, 1.0])).unwrap();
        let before = core
            .search_with_vector("base", &[1.0, 0.0], HybridSearchOptions::default())
            .unwrap();
        let buffered = core.buffered_vector_count();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("collection.kestrel");
        core.save_snapshot(&path).unwrap();
        let saved = std::fs::read(&path).unwrap();
        for (text, ranges) in [
            (" ", vec![0..1]),
            ("rust", vec![]),
            ("rust", vec![0..0]),
            ("rust", vec![0..5]),
            ("rust", vec![1..4]),
            ("rust", vec![0..3]),
            ("rust", vec![0..1, 2..4]),
            ("rust", vec![0..3, 0..4]),
            ("rust", vec![0..4, 1..3]),
            ("rust", vec![0..4, 0..4]),
            ("a b", vec![0..1, 1..2, 2..3]),
            ("café", vec![0..4, 4..5]), // splits a UTF-8 code point
        ] {
            let vectors = vec![vec![1.0, 0.0]; ranges.len()];
            assert!(matches!(
                core.add_document(prepared(text, ranges, vectors)),
                Err(CollectionError::InvalidDocument(_))
            ));
        }
        for invalid in [
            vec![1.0],
            vec![0.0, 0.0],
            vec![f32::NAN, 1.0],
            vec![1.0, f32::INFINITY],
        ] {
            assert!(matches!(
                core.add_document(prepared(
                    "rust engine",
                    vec![0..5, 5..11],
                    vec![vec![1.0, 0.0], invalid],
                )),
                Err(CollectionError::Vector(_))
            ));
        }
        assert_eq!(core.len(), 2);
        assert_eq!(core.chunk_count(), 2);
        assert_eq!(core.buffered_vector_count(), buffered);
        assert_eq!(
            core.search_with_vector("base", &[1.0, 0.0], HybridSearchOptions::default())
                .unwrap(),
            before
        );
        core.save_snapshot(&path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), saved);
        assert_eq!(core.add_document(single("next", [1.0, 0.0])).unwrap(), 2);
    }
}

#[test]
fn snapshots_restore_external_embeddings_and_buffered_writes_without_a_model() {
    let mut core = collection();
    core.add_document(single("base", [3.0, 4.0])).unwrap();
    core.set_vector_search_mode(ivf_mode()).unwrap();
    core.add_document(prepared(
        "rust engine",
        vec![0..5, 5..11],
        vec![vec![0.0, 1.0], vec![1.0, 0.0]],
    ))
    .unwrap();
    assert_eq!(core.buffered_vector_count(), 2);
    let options = HybridSearchOptions::default();
    let expected = core
        .search_with_vector("engine", &[1.0, 0.0], options)
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    core.save_snapshot(&path).unwrap();
    let saved = std::fs::read(&path).unwrap();
    drop(core);

    let mut restored = RetrievalCollection::load_snapshot(&path, CONFIG).unwrap();
    assert_eq!(*restored.embedding_config(), CONFIG);
    assert_eq!(restored.chunking_config(), ChunkingConfig::default());
    assert_eq!(restored.vector_search_mode(), VectorSearchMode::Exact);
    assert_eq!(restored.len(), 2);
    assert_eq!(restored.chunk_count(), 3);
    assert_eq!(restored.chunk_text(ChunkId(2)), Some("engine"));
    for mode in [ivf_mode(), VectorSearchMode::Exact] {
        restored.set_vector_search_mode(mode).unwrap();
        restored.rebuild_vector_index().unwrap();
        assert_eq!(
            restored
                .search_with_vector("engine", &[1.0, 0.0], options)
                .unwrap(),
            expected
        );
        restored.save_snapshot(&path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), saved);
    }
    assert_eq!(
        restored.add_document(single("next", [1.0, 0.0])).unwrap(),
        2
    );
    assert!(matches!(
        RetrievalCollection::load_snapshot(
            &path,
            EmbeddingConfig {
                model_id: "another-model",
                ..CONFIG
            },
        ),
        Err(CollectionError::Snapshot(
            SnapshotError::IncompatibleEmbeddingConfig
        ))
    ));
}

#[test]
fn empty_snapshots_round_trip_without_a_model() {
    let core = collection();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty.kestrel");
    core.save_snapshot(&path).unwrap();
    let restored = RetrievalCollection::load_snapshot(&path, CONFIG).unwrap();
    assert!(restored.is_empty());
    assert_eq!(restored.chunk_count(), 0);
    assert!(
        restored
            .search_with_vector("rust", &[1.0, 0.0], HybridSearchOptions::default())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn validates_configuration_and_queries_even_for_empty_indexes_or_zero_limits() {
    assert!(matches!(
        RetrievalCollection::new(
            EmbeddingConfig {
                dimensions: 0,
                ..CONFIG
            },
            ChunkingConfig::default(),
        ),
        Err(CollectionError::Vector(VectorError::ZeroDimensions))
    ));
    assert!(matches!(
        RetrievalCollection::new(
            CONFIG,
            ChunkingConfig {
                max_tokens: 513,
                overlap_tokens: 0
            },
        ),
        Err(CollectionError::Chunking(
            ChunkingError::InvalidMaxTokens { .. }
        ))
    ));
    let core = collection();
    for limit in [0, 10] {
        let options = HybridSearchOptions {
            limit,
            ..HybridSearchOptions::default()
        };
        assert!(matches!(
            core.search_with_vector(" \n", &[1.0, 0.0], options),
            Err(CollectionError::Embedding(EmbeddingError::EmptyText { .. }))
        ));
        for vector in [vec![1.0], vec![0.0, 0.0], vec![f32::NAN, 1.0]] {
            assert!(matches!(
                core.search_with_vector("rust", &vector, options),
                Err(CollectionError::Vector(_))
            ));
        }
    }
}

#[test]
fn retrieval_supports_concurrent_readers() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RetrievalCollection>();
    let mut core = collection();
    core.add_document(single("rust", [1.0, 0.0])).unwrap();
    core.set_vector_search_mode(ivf_mode()).unwrap();
    core.add_document(single("engine", [0.0, 1.0])).unwrap();
    let options = HybridSearchOptions::default();
    let expected = core
        .search_with_vector("rust", &[1.0, 0.0], options)
        .unwrap();
    std::thread::scope(|scope| {
        let readers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    core.search_with_vector("rust", &[1.0, 0.0], options)
                        .unwrap()
                })
            })
            .collect();
        for reader in readers {
            assert_eq!(reader.join().unwrap(), expected);
        }
    });
}
