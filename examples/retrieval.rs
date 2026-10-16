//! Model-free ingestion, durable batch recovery, and passage retrieval.
//! The two-dimensional vectors are hand-written for demonstration only.

use kestrel::{
    ChunkingConfig, EmbeddingConfig, HybridSearchOptions, PersistentCollection, PreparedChunk,
    PreparedDocument,
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
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("collection.kestrel");
    let mut collection = PersistentCollection::create(&path, config, ChunkingConfig::default())?;
    let mut documents = Vec::new();
    for (text, embedding) in [
        ("An inverted index supports keyword search.", vec![1.0, 0.0]),
        (
            "Cosine similarity compares vector directions.",
            vec![0.0, 1.0],
        ),
    ] {
        documents.push(PreparedDocument {
            text: text.into(),
            chunks: vec![PreparedChunk {
                byte_range: 0..text.len(),
                embedding,
            }],
        });
    }

    collection.add_documents(documents)?;
    drop(collection);

    // No embedding model is initialized or downloaded, including on restore.
    let restored = PersistentCollection::open(&path, config)?;
    let hits = restored.retrieval().search_with_vector(
        "keyword search",
        &[1.0, 0.0],
        HybridSearchOptions::default(),
    )?;
    for hit in hits {
        println!(
            "document {}: {}",
            hit.doc_id,
            restored.retrieval().chunk_text(hit.chunk_id).unwrap(),
        );
    }
    Ok(())
}
