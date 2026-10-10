use kestrel::InvertedIndex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut index = InvertedIndex::new();
    for text in [
        "Rust makes it possible to build a fast search engine.",
        "An inverted index maps terms to documents for keyword search.",
        "BM25 ranks search results using term frequency and document length.",
        "Vector search finds documents with similar meaning.",
        "A hybrid search engine combines keyword and vector search.",
    ] {
        index.add_document(text)?;
    }

    let query = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let query = if query.is_empty() {
        "hybrid search"
    } else {
        &query
    };
    println!(
        "{} documents, {} unique terms",
        index.len(),
        index.vocabulary_len()
    );
    println!("Postings for 'search': {:?}", index.postings("search"));
    println!("\nQuery: {query}\n");
    for hit in index.search(query, 3) {
        println!(
            "{:.4}  doc {}  {}",
            hit.score,
            hit.doc_id,
            index.document(hit.doc_id).unwrap()
        );
    }
    Ok(())
}
