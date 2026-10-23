use std::path::PathBuf;

use kestrel::{Collection, EmbeddingConfig, PersistentCollection, SearchOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("usage: cargo run --example persistence -- <collection-path>")?,
    );
    let mut local = Collection::new()?;
    let stored = if path.exists() {
        PersistentCollection::open(&path, EmbeddingConfig::default())?
    } else {
        let prepared = local.prepare_documents([
            "Reset your password using the recovery link sent to your email address.",
            "Boil pasta in salted water, then serve it with tomato sauce.",
            "An inverted index maps words to documents for efficient keyword search.",
        ])?;
        let mut stored = PersistentCollection::create(
            &path,
            *local.embedding_config(),
            local.chunking_config(),
        )?;
        stored.add_documents(prepared)?;
        stored
    };
    let query = "I forgot my login credentials";
    let vector = local.embed_query(query)?;
    let retrieval = stored.retrieval();
    for hit in retrieval.search_with_vector(query, &vector, SearchOptions::default())? {
        println!(
            "{:.6}  doc {}  chunk {}  {}",
            hit.score,
            hit.doc_id,
            hit.chunk_id.0,
            retrieval.chunk_text(hit.chunk_id).unwrap()
        );
    }
    Ok(())
}
