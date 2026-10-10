use kestrel::{InvertedIndex, VectorIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut documents = InvertedIndex::new();
    let mut vectors = VectorIndex::new(2)?;
    // Artificial vectors for learning the math; these are not text embeddings.
    for (text, vector) in [
        ("same direction", [1.0, 0.0]),
        ("partially aligned", [3.0, 4.0]),
        ("perpendicular", [0.0, 1.0]),
        ("opposite direction", [-1.0, 0.0]),
    ] {
        let doc_id = documents.add_document(text)?;
        vectors.insert(doc_id, &vector)?;
    }

    let query = [1.0, 0.0];
    println!("Query: {query:?}");
    for hit in vectors.search(&query, 4)? {
        println!(
            "{:+.3}  doc {}  {}",
            hit.score,
            hit.doc_id,
            documents.document(hit.doc_id).unwrap()
        );
    }
    Ok(())
}
