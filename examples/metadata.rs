//! Typed document metadata, durable recovery, and filtered passage retrieval.
//! Hand-written vectors keep this example model-free; no model is downloaded.

use kestrel::{
    ChunkingConfig, EmbeddingConfig, Field, Filter, FiniteF64, Metadata, PersistentCollection,
    PreparedChunk, PreparedDocument, SearchOptions,
};

const TEAM: Field<'static, String> = Field::new("team");
const PUBLISHED: Field<'static, bool> = Field::new("published");
const YEAR: Field<'static, i64> = Field::new("year");
const RATING: Field<'static, FiniteF64> = Field::new("rating");
const TAGS: Field<'static, Vec<String>> = Field::new("tags");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EmbeddingConfig {
        model_id: "example-vectors",
        artifact_repo: "no-download",
        artifact_file: "unused",
        dimensions: 2,
        max_tokens: 512,
        query_instruction: "",
    };
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("metadata.kestrel");
    let mut stored = PersistentCollection::create(&path, config, ChunkingConfig::default())?;
    for (text, team, published, embedding) in [
        (
            "An inverted index supports keyword search.",
            "engineering",
            true,
            [1.0, 0.0],
        ),
        (
            "Unreleased plans for a new search engine.",
            "engineering",
            false,
            [1.0, 0.0],
        ),
        (
            "Search the recipe index for pasta dishes.",
            "cooking",
            true,
            [0.0, 1.0],
        ),
    ] {
        let mut metadata = Metadata::default();
        metadata.insert(TEAM, team);
        metadata.insert(PUBLISHED, published);
        metadata.insert(YEAR, 2025_i64);
        metadata.insert(RATING, FiniteF64::new(4.5).unwrap());
        metadata.insert(TAGS, vec!["search".to_owned(), "guide".to_owned()]);
        stored.add_document(PreparedDocument {
            text: text.into(),
            metadata,
            chunks: vec![PreparedChunk {
                byte_range: 0..text.len(),
                embedding: embedding.to_vec(),
            }],
        })?;
    }
    drop(stored);

    // Metadata is restored with its original types. Every passage inherits its
    // source document's attributes; metadata is not embedded or indexed as text.
    let restored = PersistentCollection::open(&path, config)?;
    let core = restored.retrieval();
    let clauses = [
        TEAM.eq("engineering"),
        PUBLISHED.eq(true),
        YEAR.range(2024..=2026),
        RATING.range(FiniteF64::new(4.0).unwrap()..),
        TAGS.contains("search"),
    ];
    let filter = Filter::and(&clauses);
    let hits = core.search_with_vector(
        "keyword search",
        &[1.0, 0.0],
        SearchOptions {
            filter: Some(&filter),
            ..Default::default()
        },
    )?;
    for hit in hits {
        let metadata = core.metadata(hit.doc_id).unwrap();
        println!(
            "{} (team {}, year {}): {}",
            hit.doc_id,
            metadata.get(TEAM).unwrap(),
            metadata.get(YEAR).unwrap(),
            core.chunk_text(hit.chunk_id).unwrap(),
        );
    }
    Ok(())
}
