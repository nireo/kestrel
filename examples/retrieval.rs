//! Model-free ingestion, snapshot restoration, and passage retrieval.
//! The two-dimensional vectors are hand-written for demonstration only.

use kestrel::{
    ChunkingConfig, EmbeddingConfig, HybridSearchOptions, PreparedChunk, PreparedDocument,
    RetrievalCollection,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EmbeddingConfig {
        model_id: "example-vectors",
        artifact_repo: "example",
        artifact_file: "unused",
        dimensions: 2,
        max_tokens: 512,
        query_instruction: "",
    };
    let mut collection = RetrievalCollection::new(config, ChunkingConfig::default())?;
    for (text, embedding) in [
        ("An inverted index supports keyword search.", vec![1.0, 0.0]),
        (
            "Cosine similarity compares vector directions.",
            vec![0.0, 1.0],
        ),
    ] {
        collection.add_document(PreparedDocument {
            text: text.into(),
            chunks: vec![PreparedChunk {
                byte_range: 0..text.len(),
                embedding,
            }],
        })?;
    }

    let directory = tempfile::tempdir()?;
    let path = directory.path().join("collection.kestrel");
    collection.save_snapshot(&path)?;
    drop(collection);

    // No embedding model is initialized or downloaded, including on restore.
    let restored = RetrievalCollection::load_snapshot(&path, config)?;
    let hits = restored.search_with_vector(
        "keyword search",
        &[1.0, 0.0],
        HybridSearchOptions::default(),
    )?;
    for hit in hits {
        println!(
            "document {}: {}",
            hit.doc_id,
            restored.chunk_text(hit.chunk_id).unwrap(),
        );
    }
    Ok(())
}
