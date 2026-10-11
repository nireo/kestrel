use kestrel::{Collection, CollectionError, HybridSearchOptions, SnapshotError, TextEmbedder};

#[test]
fn invalid_file_and_missing_path_fail_before_model_initialization() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    assert!(matches!(
        Collection::load_snapshot(&path),
        Err(CollectionError::Snapshot(SnapshotError::Io(_)))
    ));
    std::fs::write(&path, b"not a snapshot").unwrap();
    assert!(matches!(
        Collection::load_snapshot(&path),
        Err(CollectionError::Snapshot(SnapshotError::InvalidMagic))
    ));
}

#[test]
#[ignore = "loads the real model and downloads its files if not cached"]
fn collection_snapshot_round_trip_with_real_model() -> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("collection.kestrel");
    let mut collection = Collection::new()?;
    collection.save_snapshot(&path)?;
    let mut restored = Collection::load_snapshot(&path)?;
    assert!(restored.is_empty());
    assert!(
        restored
            .search("password recovery", HybridSearchOptions::default())?
            .is_empty()
    );
    drop(restored);

    let texts = [
        "Reset your password using the recovery link sent to your email address.",
        "Boil pasta in salted water, then serve it with tomato sauce.",
    ];
    for (id, text) in texts.iter().enumerate() {
        assert_eq!(collection.add_document(*text)?, id as u32);
    }
    let query = "I forgot my login credentials";
    let options = HybridSearchOptions::default();
    let before = collection.search(query, options)?;
    collection.save_snapshot(&path)?;
    let saved = std::fs::read(&path)?;
    let config = *collection.embedding_config();
    drop(collection);

    let embedder = TextEmbedder::new()?;
    let mut restored = Collection::load_snapshot_with_embedder(&path, embedder)?;
    assert_eq!(restored.len(), 2);
    assert_eq!(*restored.embedding_config(), config);
    for (id, text) in texts.iter().enumerate() {
        assert_eq!(restored.document(id as u32), Some(*text));
    }
    assert_eq!(restored.search(query, options)?, before);
    restored.save_snapshot(&path)?;
    assert_eq!(std::fs::read(&path)?, saved);
    assert_eq!(
        restored.add_document("An inverted index supports keyword search.")?,
        2
    );
    // New documents only become durable after an explicit save.
    assert_eq!(std::fs::read(&path)?, saved);
    restored.save_snapshot(&path)?;
    drop(restored);

    let mut again = Collection::load_snapshot(&path)?;
    assert_eq!(again.len(), 3);
    assert_eq!(again.search("keyword search", options)?[0].doc_id, 2);
    assert_eq!(again.add_document("A fourth document.")?, 3);
    Ok(())
}
