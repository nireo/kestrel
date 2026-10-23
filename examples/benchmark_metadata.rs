//! Model-free filtered retrieval latency and warmed allocation checks.
//! cargo run --release --offline --example benchmark_metadata
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use kestrel::{
    ChunkingConfig, EmbeddingConfig, Field, Filter, IvfBuildOptions, Metadata, PreparedChunk,
    PreparedDocument, RetrievalCollection, SearchOptions, SearchScratch, VectorSearchMode,
};

struct Allocator;
static COUNT: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && COUNT.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, size) };
        if !pointer.is_null() && COUNT.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

const SHARD: Field<'static, i64> = Field::new("shard");
const DIMENSIONS: usize = 384;
const ROWS: usize = 10_000;
const CONFIG: EmbeddingConfig = EmbeddingConfig {
    model_id: "metadata-benchmark",
    artifact_repo: "no-download",
    artifact_file: "unused",
    dimensions: DIMENSIONS,
    max_tokens: 512,
    query_instruction: "",
};

fn vector(row: usize) -> Vec<f32> {
    (0..DIMENSIONS)
        .map(|i| ((row * 17 + i * 13) % 97) as f32 - 48.0)
        .collect()
}

fn measure(
    collection: &RetrievalCollection,
    queries: &[(String, Vec<f32>)],
    name: &str,
    filter: Option<&Filter<'_>>,
) {
    let options = SearchOptions {
        filter,
        ..SearchOptions::default()
    };
    let mut scratch = SearchScratch::default();
    let mut output = Vec::new();
    let mut run = || {
        for (query, vector) in queries {
            collection
                .search_into(query, vector, options, &mut scratch, &mut output)
                .unwrap();
            black_box(&output);
        }
    };
    // Warm all queries, including result capacity and the dense BM25 workspace.
    run();
    let mut samples = [0.0; 5];
    for sample in &mut samples {
        let start = Instant::now();
        run();
        *sample = start.elapsed().as_secs_f64() * 1000.0;
    }
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNT.store(true, Ordering::Relaxed);
    run();
    COUNT.store(false, Ordering::Relaxed);
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    assert_eq!(allocations, 0, "warmed search_into must not allocate");
    let mut sorted = samples;
    sorted.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({
            "case": name, "queries": queries.len(), "median_ms": sorted[2],
            "samples_ms": samples, "allocations": allocations,
        })
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("run this benchmark with --release".into());
    }
    let mut collection = RetrievalCollection::new(CONFIG, ChunkingConfig::default())?;
    collection.reserve(ROWS, ROWS);
    for start in (0..ROWS).step_by(256) {
        let batch = (start..(start + 256).min(ROWS))
            .map(|row| {
                let text = format!("search topic{} record{row}", row % 64);
                let mut metadata = Metadata::default();
                metadata.insert(SHARD, (row % 100) as i64);
                PreparedDocument {
                    chunks: vec![PreparedChunk {
                        byte_range: 0..text.len(),
                        embedding: vector(row),
                    }],
                    text,
                    metadata,
                }
            })
            .collect();
        collection.add_documents(batch)?;
    }
    let queries: Vec<_> = (0..64)
        .map(|row| (format!("search topic{row}"), vector(row)))
        .collect();
    let cases = [
        ("all", SHARD.range(0..100)),
        ("ten_percent", SHARD.range(0..10)),
        ("one_percent", SHARD.eq(0)),
        ("none", SHARD.eq(-1)),
    ];
    // Preserve exact results as an oracle for filtered queries in IVF mode.
    let expected: Vec<Vec<_>> =
        cases
            .iter()
            .map(|(_, filter)| {
                queries.iter().map(|(query, vector)| {
        let hits = collection.search_with_vector(query, vector, SearchOptions {
            filter: Some(filter), ..SearchOptions::default()
        }).unwrap();
        assert!(hits.iter().all(|hit| filter.matches(collection.metadata(hit.doc_id).unwrap())));
        hits
    }).collect()
            })
            .collect();
    measure(&collection, &queries, "exact_unfiltered", None);
    for (name, filter) in &cases {
        measure(
            &collection,
            &queries,
            &format!("exact_{name}"),
            Some(filter),
        );
    }
    collection.set_vector_search_mode(VectorSearchMode::Ivf {
        build: IvfBuildOptions {
            clusters: 32,
            training_samples: 128,
            iterations: 3,
            ..IvfBuildOptions::default()
        },
        probes: 4,
    })?;
    measure(&collection, &queries, "ivf_unfiltered", None);
    for ((name, filter), expected) in cases.iter().zip(&expected) {
        for ((query, vector), expected) in queries.iter().zip(expected) {
            assert_eq!(
                &collection.search_with_vector(
                    query,
                    vector,
                    SearchOptions {
                        filter: Some(filter),
                        ..SearchOptions::default()
                    }
                )?,
                expected
            );
        }
        measure(&collection, &queries, &format!("ivf_{name}"), Some(filter));
    }
    Ok(())
}
