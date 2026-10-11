use std::path::PathBuf;

use kestrel::{Collection, HybridSearchOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("usage: cargo run --example snapshots -- <snapshot-path>")?,
    );
    let mut collection = if path.exists() {
        println!("Loading {}", path.display());
        Collection::load_snapshot(&path)?
    } else {
        let mut collection = Collection::new()?;
        for text in [
            "Reset your password using the recovery link sent to your email address.",
            "Boil pasta in salted water, then serve it with tomato sauce.",
            "An inverted index maps words to documents for efficient keyword search.",
        ] {
            collection.add_document(text)?;
        }
        collection.save_snapshot(&path)?;
        println!("Saved {} documents to {}", collection.len(), path.display());
        collection
    };
    for hit in collection.search(
        "I forgot my login credentials",
        HybridSearchOptions::default(),
    )? {
        println!(
            "{:.6}  doc {}  {}",
            hit.score,
            hit.doc_id,
            collection.document(hit.doc_id).unwrap()
        );
    }
    Ok(())
}
