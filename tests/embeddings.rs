use kestrel::{
    EmbeddingError, HybridSearchOptions, InvertedIndex, TextEmbedder, VectorIndex, hybrid_search,
};

#[test]
#[ignore = "loads the real model and downloads its files if not cached"]
fn real_embeddings_support_batching_validation_and_semantic_search()
-> Result<(), Box<dyn std::error::Error>> {
    let mut embedder = TextEmbedder::new()?;
    assert_eq!(embedder.config().model_id, "BAAI/bge-small-en-v1.5");
    assert_eq!(embedder.config().artifact_repo, "Xenova/bge-small-en-v1.5");
    assert_eq!(embedder.dimensions(), 384);
    assert_eq!(embedder.config().max_tokens, 512);
    assert!(embedder.embed_documents(&[])?.is_empty());
    assert!(matches!(
        embedder.embed_documents(&["valid", " \n\t"]),
        Err(EmbeddingError::EmptyText { index: 1 })
    ));
    assert!(matches!(
        embedder.embed_query("  "),
        Err(EmbeddingError::EmptyText { index: 0 })
    ));

    let texts = [
        "Reset your password using the recovery link sent to your email address.",
        "Boil pasta in salted water, then serve it with tomato sauce.",
        "An inverted index maps words to documents for keyword search.",
    ];
    let embeddings = embedder.embed_documents(&texts)?;
    assert_eq!(embeddings.len(), texts.len());
    for vector in &embeddings {
        assert_eq!(vector.len(), embedder.dimensions());
        assert!(vector.iter().all(|component| component.is_finite()));
        assert!(vector.iter().any(|&component| component != 0.0));
    }
    // Compare a batched document with the same text embedded on its own.
    let single = embedder.embed_documents(&texts[1..2])?;
    let max_error = embeddings[1]
        .iter()
        .zip(&single[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_error < 1e-4,
        "batching changed the embedding: {max_error}"
    );

    // Cross the wrapper's 32-document batch boundary and verify alignment.
    let repeated: Vec<_> = (0..35).map(|index| texts[index % texts.len()]).collect();
    let batched = embedder.embed_documents(&repeated)?;
    assert_eq!(batched.len(), repeated.len());
    for index in [0, 31, 32, 33, 34] {
        let max_error = batched[index]
            .iter()
            .zip(&embeddings[index % texts.len()])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error < 1e-4, "embedding order changed at row {index}");
    }

    let mut bm25 = InvertedIndex::new();
    let mut vectors = VectorIndex::new(embedder.dimensions())?;
    for (text, vector) in texts.into_iter().zip(embeddings) {
        let id = bm25.add_document(text)?;
        vectors.insert(id, &vector)?;
    }
    for (query, expected_id) in [
        ("I cannot sign in because I forgot my credentials", 0),
        ("How do I cook an Italian noodle dish?", 1),
    ] {
        let query_vector = embedder.embed_query(query)?;
        assert_eq!(query_vector.len(), embedder.dimensions());
        let hits = vectors.search(&query_vector, 3)?;
        assert_eq!(hits[0].doc_id, expected_id, "{query}");
        let hybrid = hybrid_search(
            &bm25,
            &vectors,
            query,
            &query_vector,
            HybridSearchOptions::default(),
        )?;
        assert_eq!(hybrid.len(), 3);
        assert_eq!(
            hybrid
                .iter()
                .find(|hit| hit.doc_id == expected_id)
                .unwrap()
                .vector_rank,
            Some(1)
        );
    }
    Ok(())
}
