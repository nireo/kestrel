use kestrel::{HybridSearchOptions, InvertedIndex, VectorIndex, hybrid_search};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut bm25 = InvertedIndex::new();
    let mut vectors = VectorIndex::new(2)?;
    // Manual vectors let us inspect fusion without an embedding model.
    let lexical_only = bm25.add_document("rust")?;
    let both = bm25.add_document("rust engine")?;
    let vector_only = bm25.add_document("semantic only")?;
    vectors.insert(both, &[1.0, 0.0])?;
    vectors.insert(vector_only, &[0.0, 1.0])?;

    println!("Query text: rust; query vector: [1, 0]");
    println!("Document {lexical_only} has no vector.\n");
    for hit in hybrid_search(
        &bm25,
        &vectors,
        "rust",
        &[1.0, 0.0],
        HybridSearchOptions {
            candidate_depth: 2,
            limit: 3,
        },
    )? {
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
