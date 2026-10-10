use kestrel::{HybridSearchOptions, InvertedIndex, TextEmbedder, VectorIndex, hybrid_search};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut embedder = TextEmbedder::new()?;
    println!("Embedding configuration: {:?}\n", embedder.config());
    let texts = [
        "Reset your password using the recovery link sent to your email address.",
        "Protect your account by enabling two-factor authentication.",
        "Change the background color and font size in the appearance settings.",
        "Boil pasta in salted water, then serve it with tomato sauce.",
        "An inverted index maps words to documents for efficient keyword search.",
    ];
    let embeddings = embedder.embed_documents(&texts)?;
    let mut bm25 = InvertedIndex::new();
    let mut vectors = VectorIndex::new(embedder.dimensions())?;
    for (text, vector) in texts.into_iter().zip(embeddings) {
        let doc_id = bm25.add_document(text)?;
        vectors.insert(doc_id, &vector)?;
    }

    let query = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let query = if query.is_empty() {
        "How can I recover my account?"
    } else {
        &query
    };
    let query_vector = embedder.embed_query(query)?;
    let options = HybridSearchOptions {
        candidate_depth: 5,
        limit: 3,
    };
    println!("Query: {query}");
    println!("\nBM25:");
    for hit in bm25.search(query, options.limit) {
        println!(
            "{:.4}  doc {}  {}",
            hit.score,
            hit.doc_id,
            bm25.document(hit.doc_id).unwrap()
        );
    }
    println!("\nVector:");
    for hit in vectors.search(&query_vector, options.limit)? {
        println!(
            "{:.4}  doc {}  {}",
            hit.score,
            hit.doc_id,
            bm25.document(hit.doc_id).unwrap()
        );
    }
    println!("\nHybrid:");
    for hit in hybrid_search(&bm25, &vectors, query, &query_vector, options)? {
        println!(
            "{:.6}  doc {}  BM25 rank {:?}  vector rank {:?}  {}",
            hit.score,
            hit.doc_id,
            hit.bm25_rank,
            hit.vector_rank,
            bm25.document(hit.doc_id).unwrap()
        );
    }
    Ok(())
}
