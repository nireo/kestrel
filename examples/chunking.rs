use kestrel::{ChunkingConfig, Collection, SearchOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut collection = Collection::with_chunking(ChunkingConfig {
        max_tokens: 64,
        overlap_tokens: 8,
    })?;
    let guide = format!(
        "{}\nThe telescope recovery code is NEBULA-7429. Use it to unlock the observatory console.",
        "Boil pasta in salted water and serve it with tomato sauce. ".repeat(60)
    );
    let guide_id = collection.add_document(guide)?;
    collection.add_document("Invite a coworker from Team settings to join your workspace.")?;
    println!(
        "{} documents, {} chunks; guide document ID: {guide_id}\n",
        collection.len(),
        collection.chunk_count()
    );
    for hit in collection.search(
        "NEBULA-7429 telescope recovery code",
        SearchOptions {
            candidate_depth: 10,
            limit: 3,
            filter: None,
        },
    )? {
        let chunk = collection.chunk(hit.chunk_id).unwrap();
        println!(
            "{:.6}  doc {}  chunk {}  bytes {:?}\n{}\n",
            hit.score,
            hit.doc_id,
            hit.chunk_id.0,
            chunk.byte_range,
            collection.chunk_text(hit.chunk_id).unwrap()
        );
    }
    Ok(())
}
