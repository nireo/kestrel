use kestrel::{
    ChunkId, ChunkingConfig, Collection, CollectionError, EmbeddingError, PersistentCollection,
    SearchOptions, TextEmbedder,
};

#[test]
#[ignore = "loads the real model and downloads its files if not cached"]
fn long_document_chunks_retrieve_the_tail_and_survive_durable_batch_recovery()
-> Result<(), Box<dyn std::error::Error>> {
    let config = ChunkingConfig {
        max_tokens: 64,
        overlap_tokens: 8,
    };
    let embedder = TextEmbedder::new()?;
    let mut collection = Collection::from_embedder_with_chunking(embedder, config)?;
    let text = format!(
        "{}\nThe telescope recovery code is NEBULA-7429. Use it to unlock the observatory console.",
        "pasta ".repeat(700)
    );
    let doc_id = collection.add_document(text.clone())?;
    assert_eq!(doc_id, 0);
    assert_eq!(collection.len(), 1);
    assert_eq!(collection.document(doc_id), Some(text.as_str()));
    let guide_chunks = collection.chunk_count();
    assert!(guide_chunks > 10);
    let mut previous_end = 0;
    for id in 0..guide_chunks as u32 {
        let chunk = collection.chunk(ChunkId(id)).unwrap();
        assert_eq!(chunk.doc_id, doc_id);
        assert!(chunk.byte_range.start <= previous_end);
        assert!(chunk.byte_range.end > previous_end);
        assert_eq!(
            collection.chunk_text(ChunkId(id)),
            text.get(chunk.byte_range.clone())
        );
        previous_end = chunk.byte_range.end;
    }
    assert_eq!(previous_end, text.len());
    assert!(collection.chunk(ChunkId(guide_chunks as u32)).is_none());
    assert!(
        collection
            .chunk_text(ChunkId(guide_chunks as u32))
            .is_none()
    );

    // A tokenless input and a blank input must consume neither ID namespace.
    assert!(matches!(
        collection.add_document("\0"),
        Err(CollectionError::Chunking(_))
    ));
    assert!(matches!(
        collection.add_document(" \n"),
        Err(CollectionError::Embedding(EmbeddingError::EmptyText {
            index: 0
        }))
    ));
    assert_eq!(collection.len(), 1);
    assert_eq!(collection.chunk_count(), guide_chunks);
    assert_eq!(
        collection.add_document("Invite a coworker to join your workspace.")?,
        1
    );
    assert_eq!(
        collection
            .chunk(ChunkId(guide_chunks as u32))
            .unwrap()
            .doc_id,
        1
    );

    let query = "NEBULA-7429 telescope recovery code";
    let options = SearchOptions {
        candidate_depth: collection.chunk_count(),
        limit: collection.chunk_count(),
        filter: None,
    };
    let before = collection.search(query, options)?;
    assert_eq!(before[0].doc_id, doc_id);
    assert!(
        collection
            .chunk_text(before[0].chunk_id)
            .unwrap()
            .contains("NEBULA-7429")
    );
    assert!(
        collection
            .chunk(before[0].chunk_id)
            .unwrap()
            .byte_range
            .start
            > 512
    );
    assert_eq!(
        before.iter().filter(|hit| hit.doc_id == doc_id).count(),
        guide_chunks
    );

    let directory = tempfile::tempdir()?;
    let path = directory.path().join("chunks.kestrel");
    let mut stored = PersistentCollection::create(&path, *collection.embedding_config(), config)?;
    let prepared = collection.prepare_documents([
        text.clone(),
        "Invite a coworker to join your workspace.".into(),
    ])?;
    assert_eq!(stored.add_documents(prepared)?, [0, 1]);
    let bytes = std::fs::read(&path)?;
    let boundaries: Vec<_> = (0..collection.chunk_count() as u32)
        .map(|id| collection.chunk(ChunkId(id)).unwrap().clone())
        .collect();
    let query_vector = collection.embed_query(query)?;
    let third = collection.prepare_document("A third document.")?;
    let embedding = *collection.embedding_config();
    drop(collection);
    drop(stored);
    let mut stored = PersistentCollection::open(&path, embedding)?;
    let restored = stored.retrieval();
    assert_eq!(restored.len(), 2);
    assert_eq!(restored.chunking_config(), config);
    assert_eq!(restored.document(doc_id), Some(text.as_str()));
    for (id, chunk) in boundaries.iter().enumerate() {
        assert_eq!(restored.chunk(ChunkId(id as u32)), Some(chunk));
    }
    assert_eq!(
        restored.search_with_vector(query, &query_vector, options)?,
        before
    );
    assert_eq!(std::fs::read(&path)?, bytes);
    assert_eq!(stored.add_document(third)?, 2);
    assert_eq!(
        stored
            .retrieval()
            .chunk(ChunkId(boundaries.len() as u32))
            .unwrap()
            .doc_id,
        2
    );
    Ok(())
}
