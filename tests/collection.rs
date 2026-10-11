use kestrel::{Collection, CollectionError, EmbeddingError, HybridSearchOptions, TextEmbedder};

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
