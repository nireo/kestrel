use kestrel::{
    ChunkId, ChunkingConfig, CollectionError, EmbeddingConfig, HybridSearchOptions,
    IvfBuildOptions, PersistentCollection, PreparedChunk, PreparedDocument, StorageError,
    VectorSearchMode,
};

const CONFIG: EmbeddingConfig = EmbeddingConfig {
    model_id: "external-test-model",
    artifact_repo: "no-download",
    artifact_file: "unused.onnx",
    dimensions: 2,
    max_tokens: 512,
    query_instruction: "query: ",
};

fn single(text: &str, embedding: [f32; 2]) -> PreparedDocument {
    PreparedDocument {
        text: text.into(),
        chunks: vec![PreparedChunk {
            byte_range: 0..text.len(),
            embedding: embedding.to_vec(),
        }],
    }
}

#[test]
fn durable_batches_preserve_ids_passages_rankings_and_ivf_buffered_writes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    let mut collection =
        PersistentCollection::create(&path, CONFIG, ChunkingConfig::default()).unwrap();
    assert!(collection.retrieval().is_empty());
    assert_eq!(
        collection.add_document(single("base", [3.0, 4.0])).unwrap(),
        0
    );
    let prefix = std::fs::read(&path).unwrap();
    let mode = VectorSearchMode::Ivf {
        build: IvfBuildOptions {
            clusters: 2,
            ..IvfBuildOptions::default()
        },
        probes: usize::MAX,
    };
    collection.set_vector_search_mode(mode).unwrap();
    assert_eq!(
        collection
            .add_documents(vec![
                PreparedDocument {
                    text: "café rust engine".into(),
                    chunks: vec![
                        PreparedChunk {
                            byte_range: 0..10,
                            embedding: vec![0.0, 5.0]
                        },
                        PreparedChunk {
                            byte_range: 6..17,
                            embedding: vec![7.0, 0.0]
                        },
                    ],
                },
                single("pasta", [-1.0, 0.0]),
            ])
            .unwrap(),
        [1, 2]
    );
    assert_eq!(collection.retrieval().buffered_vector_count(), 3);
    assert!(std::fs::read(&path).unwrap().starts_with(&prefix));
    let expected = collection
        .retrieval()
        .search_with_vector("engine", &[1.0, 0.0], HybridSearchOptions::default())
        .unwrap();
    collection.rebuild_vector_index().unwrap();
    assert_eq!(collection.retrieval().buffered_vector_count(), 0);
    let saved = std::fs::read(&path).unwrap();
    drop(collection);

    let mut restored = PersistentCollection::open(&path, CONFIG).unwrap();
    let core = restored.retrieval();
    assert_eq!(core.len(), 3);
    assert_eq!(core.chunk_count(), 4);
    assert_eq!(core.vector_search_mode(), VectorSearchMode::Exact);
    assert_eq!(*core.embedding_config(), CONFIG);
    assert_eq!(core.chunk_text(ChunkId(2)), Some("rust engine"));
    assert_eq!(core.chunk(ChunkId(2)).unwrap().doc_id, 1);
    assert_eq!(
        core.search_with_vector("engine", &[1.0, 0.0], HybridSearchOptions::default())
            .unwrap(),
        expected
    );
    assert_eq!(std::fs::read(&path).unwrap(), saved);
    assert_eq!(
        restored
            .add_documents(vec![single("next", [1.0, 0.0]), single("last", [0.0, 1.0])])
            .unwrap(),
        [3, 4]
    );
    drop(restored);
    assert_eq!(
        PersistentCollection::open(&path, CONFIG)
            .unwrap()
            .retrieval()
            .len(),
        5
    );
}

#[test]
fn invalid_batches_and_empty_batches_change_neither_file_nor_indexes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    let mut collection =
        PersistentCollection::create(&path, CONFIG, ChunkingConfig::default()).unwrap();
    collection.add_document(single("base", [1.0, 0.0])).unwrap();
    let before = std::fs::read(&path).unwrap();
    assert!(
        collection
            .add_documents(vec![single("valid", [0.0, 1.0]), single("bad", [0.0, 0.0])])
            .is_err()
    );
    let mut malformed = single("café", [1.0, 0.0]);
    malformed.chunks[0].byte_range.end = 4;
    assert!(
        collection
            .add_documents(vec![single("valid", [0.0, 1.0]), malformed])
            .is_err()
    );
    assert!(collection.add_documents(vec![]).unwrap().is_empty());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(collection.retrieval().len(), 1);
    assert_eq!(collection.retrieval().chunk_count(), 1);
    assert_eq!(
        collection.add_document(single("next", [0.0, 1.0])).unwrap(),
        1
    );
}

#[test]
fn creating_never_overwrites_and_an_open_collection_has_one_writer() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    let collection =
        PersistentCollection::create(&path, CONFIG, ChunkingConfig::default()).unwrap();
    let before = std::fs::read(&path).unwrap();
    assert!(PersistentCollection::create(&path, CONFIG, ChunkingConfig::default()).is_err());
    assert!(matches!(
        PersistentCollection::open(&path, CONFIG),
        Err(CollectionError::Storage(StorageError::Locked))
    ));
    assert_eq!(std::fs::read(&path).unwrap(), before);
    drop(collection);
    assert!(
        PersistentCollection::open(&path, CONFIG)
            .unwrap()
            .retrieval()
            .is_empty()
    );
}

#[test]
fn opening_checks_the_complete_embedding_configuration_without_changing_the_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    assert!(PersistentCollection::open(&path, CONFIG).is_err());
    drop(PersistentCollection::create(&path, CONFIG, ChunkingConfig::default()).unwrap());
    let before = std::fs::read(&path).unwrap();
    for config in [
        EmbeddingConfig {
            model_id: "different",
            ..CONFIG
        },
        EmbeddingConfig {
            artifact_repo: "different",
            ..CONFIG
        },
        EmbeddingConfig {
            artifact_file: "different",
            ..CONFIG
        },
        EmbeddingConfig {
            dimensions: 3,
            ..CONFIG
        },
        EmbeddingConfig {
            max_tokens: 256,
            ..CONFIG
        },
        EmbeddingConfig {
            query_instruction: "different",
            ..CONFIG
        },
    ] {
        assert!(matches!(
            PersistentCollection::open(&path, config),
            Err(CollectionError::Storage(
                StorageError::IncompatibleEmbeddingConfig
            ))
        ));
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }
}

#[test]
#[ignore = "loads the real model and downloads its files if not cached"]
fn local_preparation_batches_feed_durable_storage_without_mutating_the_preparer()
-> Result<(), Box<dyn std::error::Error>> {
    let mut local = kestrel::Collection::new()?;
    let texts = [
        "Reset your password using the recovery link.",
        "Cook pasta in salted water.",
    ];
    let prepared = local.prepare_documents(texts)?;
    assert!(local.is_empty());
    assert_eq!(prepared.len(), 2);
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("collection.kestrel");
    let config = *local.embedding_config();
    let mut stored = PersistentCollection::create(&path, config, local.chunking_config())?;
    assert_eq!(stored.add_documents(prepared)?, [0, 1]);
    let query = "I forgot my login credentials";
    let vector = local.embed_query(query)?;
    assert_eq!(local.add_documents(texts)?, [0, 1]);
    let expected = local.search_with_vector(query, &vector, HybridSearchOptions::default())?;
    assert!(local.add_documents(["valid", " "]).is_err());
    assert_eq!(local.len(), 2);
    drop(local);
    drop(stored);
    let restored = PersistentCollection::open(&path, config)?;
    assert_eq!(
        restored
            .retrieval()
            .search_with_vector(query, &vector, HybridSearchOptions::default())?,
        expected
    );
    Ok(())
}
