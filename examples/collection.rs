use kestrel::{Collection, SearchOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut collection = Collection::new()?;
    for text in [
        "Reset your password using the recovery link sent to your email address.",
        "Boil pasta in salted water, then serve it with tomato sauce.",
        "An inverted index maps words to documents for efficient keyword search.",
    ] {
        collection.add_document(text)?;
    }
    let query = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let query = if query.is_empty() {
        "I forgot my login credentials"
    } else {
        &query
    };
    println!("Query: {query}\n");
    for hit in collection.search(query, SearchOptions::default())? {
        println!(
            "{:.6}  doc {}  chunk {}  BM25 rank {:?}  vector rank {:?}  {}",
            hit.score,
            hit.doc_id,
            hit.chunk_id.0,
            hit.bm25_rank,
            hit.vector_rank,
            collection.chunk_text(hit.chunk_id).unwrap()
        );
    }
    Ok(())
}
