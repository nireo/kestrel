use kestrel::{
    ChunkingConfig, Collection, CollectionError, EmbeddingConfig, EmbeddingError,
    HybridSearchOptions, IvfBuildOptions, RetrievalCollection, TextEmbedder, VectorError,
    VectorSearchMode,
};

#[test]
#[ignore = "loads the real model and downloads its files if not cached"]
fn collection_coordinates_real_embeddings_ids_search_and_failed_insertions()
-> Result<(), Box<dyn std::error::Error>> {
    let embedder = TextEmbedder::new()?;
    let mut collection = Collection::from_embedder(embedder)?;
    assert!(collection.is_empty());
    assert_eq!(collection.embedding_config().dimensions, 384);
    assert!(collection.document(0).is_none());
    assert!(
        collection
            .search("password recovery", HybridSearchOptions::default())?
            .is_empty()
    );
    assert!(matches!(
        collection.add_document(" \n"),
        Err(CollectionError::Embedding(EmbeddingError::EmptyText {
            index: 0
        }))
    ));
    assert!(collection.is_empty());

    let first = collection
        .add_document("Reset your password using the recovery link sent to your email address.")?;
    let second = collection.add_document(String::from(
        "Boil pasta in salted water, then serve it with tomato sauce.",
    ))?;
    assert_eq!((first, second), (0, 1));
    assert_eq!(collection.len(), 2);
    assert_eq!(
        collection.document(second),
        Some("Boil pasta in salted water, then serve it with tomato sauce.")
    );
    assert!(collection.document(2).is_none());

    let query = "I forgot my login credentials";
    let options = HybridSearchOptions::default();
    let before = collection.search(query, options)?;
    assert_eq!(before.len(), 2);
    assert_eq!(before[0].doc_id, first);
    assert_eq!(before[0].vector_rank, Some(1));
    assert!(collection.search_many(&[], options)?.is_empty());
    assert!(matches!(
        collection.search_many(&[query, " \t"], options),
        Err(CollectionError::Embedding(EmbeddingError::EmptyText {
            index: 1
        }))
    ));
    let queries = [query, "How do I cook pasta?"];
    let batched = collection.search_many(&queries, options)?;
    for (query, hits) in queries.iter().zip(batched) {
        let single = collection.search(query, options)?;
        assert_eq!(hits, single);
    }
    assert!(matches!(
        collection.add_document(""),
        Err(CollectionError::Embedding(EmbeddingError::EmptyText {
            index: 0
        }))
    ));
    assert_eq!(collection.len(), 2);
    assert_eq!(collection.search(query, options)?, before);
    assert!(matches!(
        collection.search(" \t", options),
        Err(CollectionError::Embedding(EmbeddingError::EmptyText {
            index: 0
        }))
    ));
    for options in [
        HybridSearchOptions {
            candidate_depth: 0,
            limit: 3,
        },
        HybridSearchOptions {
            candidate_depth: 3,
            limit: 0,
        },
    ] {
        assert!(collection.search(query, options)?.is_empty());
    }
    assert_eq!(
        collection.add_document("An inverted index supports keyword search.")?,
        2
    );
    assert_eq!(collection.search("keyword search", options)?[0].doc_id, 2);
    Ok(())
}

#[test]
#[ignore = "loads the real model and downloads its files if not cached"]
fn ivf_collection_supports_buffered_writes_shared_readers_and_model_detachment()
-> Result<(), Box<dyn std::error::Error>> {
    let mut collection = Collection::new()?;
    collection.add_document("Reset your password using the recovery link in your email.")?;
    collection.add_document("Cook pasta in salted water and serve with tomato sauce.")?;
    assert_eq!(collection.vector_search_mode(), VectorSearchMode::Exact);
    let mode = VectorSearchMode::Ivf {
        build: IvfBuildOptions {
            clusters: 2,
            ..IvfBuildOptions::default()
        },
        probes: usize::MAX,
    };
    collection.set_vector_search_mode(mode)?;
    assert_eq!(collection.buffered_vector_count(), 0);
    let new_id = collection.add_document("Use a security key for passwordless account login.")?;
    assert_eq!(collection.buffered_vector_count(), 1);
    let query = "How do I log into my account?";
    let vector = collection.embed_query(query)?;
    let options = HybridSearchOptions::default();
    let expected = collection.search_with_vector(query, &vector, options)?;
    assert!(
        expected
            .iter()
            .any(|h| h.doc_id == new_id && h.vector_rank.is_some())
    );
    assert_eq!(collection.search(query, options)?, expected);
    std::thread::scope(|scope| {
        let shared = &collection;
        let vector = &vector;
        let readers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(move || shared.search_with_vector(query, vector, options).unwrap())
            })
            .collect();
        for reader in readers {
            assert_eq!(reader.join().unwrap(), expected);
        }
    });
    assert!(matches!(
        collection.search_with_vector(" ", &vector, options),
        Err(CollectionError::Embedding(EmbeddingError::EmptyText { .. }))
    ));
    assert!(matches!(
        collection.search_with_vector(query, &[1.0], options),
        Err(CollectionError::Vector(
            VectorError::DimensionMismatch { .. }
        ))
    ));
    let zero = vec![0.0; vector.len()];
    assert!(matches!(
        collection.search_with_vector(
            query,
            &zero,
            HybridSearchOptions {
                limit: 0,
                ..options
            }
        ),
        Err(CollectionError::Vector(VectorError::ZeroVector))
    ));
    assert_eq!(
        collection
            .retrieval()
            .search_with_vector(query, &vector, options)?,
        expected
    );
    let retrieval = collection.into_retrieval();
    assert_eq!(retrieval.vector_search_mode(), mode);
    assert_eq!(retrieval.buffered_vector_count(), 1);
    assert_eq!(
        retrieval.search_with_vector(query, &vector, options)?,
        expected
    );
    let mut collection = Collection::from_retrieval(retrieval, TextEmbedder::new()?)?;
    assert_eq!(collection.buffered_vector_count(), 1);
    collection.rebuild_vector_index()?;
    assert_eq!(collection.buffered_vector_count(), 0);
    assert_eq!(
        collection.search_with_vector(query, &vector, options)?,
        expected
    );
    collection.set_vector_search_mode(VectorSearchMode::Exact)?;
    assert_eq!(
        collection.search_with_vector(query, &vector, options)?,
        expected
    );
    assert_eq!(collection.add_document("A fourth document.")?, 3);
    Ok(())
}

#[test]
#[ignore = "loads the real model and downloads its files if not cached"]
fn attaching_an_embedder_rejects_a_different_query_instruction()
-> Result<(), Box<dyn std::error::Error>> {
    let retrieval = RetrievalCollection::new(
        EmbeddingConfig {
            query_instruction: "incompatible instruction: ",
            ..EmbeddingConfig::default()
        },
        ChunkingConfig::default(),
    )?;
    assert!(matches!(
        Collection::from_retrieval(retrieval, TextEmbedder::new()?),
        Err(CollectionError::IncompatibleEmbeddingConfig)
    ));
    Ok(())
}
