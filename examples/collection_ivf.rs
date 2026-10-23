use kestrel::{Collection, IvfBuildOptions, SearchOptions, VectorSearchMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut collection = Collection::new()?;
    collection.add_document("An inverted index maps terms to documents for keyword search.")?;
    collection.add_document("Vector search finds passages with similar meaning.")?;
    collection.set_vector_search_mode(VectorSearchMode::Ivf {
        build: IvfBuildOptions {
            clusters: 2,
            ..IvfBuildOptions::default()
        },
        probes: 1,
    })?;

    // New chunks are searched exactly alongside the immutable IVF index.
    collection
        .add_document("A vector insertion buffer makes new passages immediately searchable.")?;
    println!("Buffered vectors: {}", collection.buffered_vector_count());

    let query = "How are new passages searchable before rebuilding?";
    let query_vector = collection.embed_query(query)?;
    // Retrieval only borrows &self; prepared vectors can be searched concurrently.
    for hit in collection.search_with_vector(query, &query_vector, SearchOptions::default())? {
        println!(
            "doc {} chunk {}: {}",
            hit.doc_id,
            hit.chunk_id.0,
            collection.chunk_text(hit.chunk_id).unwrap()
        );
    }
    collection.rebuild_vector_index()?;
    println!(
        "Buffered vectors after rebuild: {}",
        collection.buffered_vector_count()
    );
    Ok(())
}
