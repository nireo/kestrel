//! Allocation and latency measurements for ingestion, retrieval, and persistence.
//! Use --only vector|ingest|query|ivf|storage to isolate a stage. No model is loaded.
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};
use std::time::Instant;

use kestrel::{
    Bm25Scratch, ChunkingConfig, EmbeddingConfig, HybridSearchOptions, IvfBuildOptions, IvfIndex,
    PersistentCollection, PreparedChunk, PreparedDocument, RetrievalCollection, VectorIndex,
    VectorSearchMode,
};
use serde_json::json;

type Error = Box<dyn std::error::Error>;
const DIMENSIONS: usize = 384;
const CONFIG: EmbeddingConfig = EmbeddingConfig {
    model_id: "pipeline-benchmark",
    artifact_repo: "no-download",
    artifact_file: "unused",
    dimensions: DIMENSIONS,
    max_tokens: 512,
    query_instruction: "",
};
struct Allocator;
static COUNT: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_DELTA: AtomicIsize = AtomicIsize::new(0);
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && COUNT.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            LIVE_DELTA.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COUNT.load(Ordering::Relaxed) {
            LIVE_DELTA.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(ptr, layout, size) };
        if !result.is_null() && COUNT.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(size, Ordering::Relaxed);
            LIVE_DELTA.fetch_add(size as isize - layout.size() as isize, Ordering::Relaxed);
        }
        result
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

// Setup and destruction of returned indexes are outside timing. Allocation
// accounting runs separately, so atomic counter updates do not distort timing.
fn measure<I, T>(
    name: &str,
    rounds: usize,
    units: usize,
    mut setup: impl FnMut() -> I,
    mut run: impl FnMut(I) -> T,
) {
    let mut samples = Vec::new();
    for round in 0..=rounds {
        let input = setup();
        let start = Instant::now();
        let result = run(input);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        black_box(&result);
        if round > 0 {
            samples.push(elapsed);
        }
    }
    let input = setup();
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    LIVE_DELTA.store(0, Ordering::Relaxed);
    COUNT.store(true, Ordering::Relaxed);
    let result = run(input);
    COUNT.store(false, Ordering::Relaxed);
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let allocated_bytes = BYTES.load(Ordering::Relaxed);
    let net_heap_bytes = LIVE_DELTA.load(Ordering::Relaxed);
    black_box(&result);
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{}",
        json!({"case":name,"median_ms":sorted[sorted.len()/2],"samples_ms":samples,
        "units":units,"allocations":allocations,"allocated_bytes":allocated_bytes,"net_heap_bytes":net_heap_bytes})
    );
}
fn vector(row: usize) -> Vec<f32> {
    (0..DIMENSIONS)
        .map(|i| ((row * 17 + i * 13) % 97) as f32 - 48.0)
        .collect()
}
fn batches(rows: usize, batch_size: usize) -> Vec<Vec<PreparedDocument>> {
    (0..rows)
        .step_by(batch_size)
        .map(|start| {
            (start..(start + batch_size).min(rows))
                .map(|row| {
                    let text = format!("search topic{} record{row}", row % 64);
                    let len = text.len();
                    PreparedDocument {
                        text,
                        chunks: vec![PreparedChunk {
                            byte_range: 0..len,
                            embedding: vector(row),
                        }],
                    }
                })
                .collect()
        })
        .collect()
}
fn collection(rows: usize, reserve: bool) -> RetrievalCollection {
    let mut index = RetrievalCollection::new(CONFIG, ChunkingConfig::default()).unwrap();
    if reserve {
        index.reserve(rows, rows);
    }
    for batch in batches(rows, 256) {
        index.add_documents(batch).unwrap();
    }
    index
}
fn build_options() -> IvfBuildOptions {
    IvfBuildOptions {
        clusters: 32,
        training_samples: 128,
        iterations: 3,
        ..IvfBuildOptions::default()
    }
}
fn main() -> Result<(), Error> {
    if cfg!(debug_assertions) {
        return Err("run this benchmark with --release".into());
    }
    let mut rounds = 5;
    let mut only = String::from("all");
    let mut reserve = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--rounds" => rounds = args.next().ok_or("missing rounds")?.parse()?,
            "--only" => only = args.next().ok_or("missing stage")?,
            "--reserve" => reserve = true,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if rounds == 0
        || !["all", "vector", "ingest", "query", "ivf", "storage"].contains(&only.as_str())
    {
        return Err("invalid benchmark options".into());
    }
    let enabled = |stage: &str| only == "all" || only == stage;
    if enabled("vector") {
        let inputs: Vec<_> = (0..100_000).map(vector).collect();
        measure(
            "vector_build_100k",
            rounds,
            inputs.len(),
            || &inputs,
            |inputs| {
                let mut index = if reserve {
                    VectorIndex::with_capacity(DIMENSIONS, inputs.len()).unwrap()
                } else {
                    VectorIndex::new(DIMENSIONS).unwrap()
                };
                for (row, v) in inputs.iter().enumerate() {
                    index.insert(row as u32, v).unwrap();
                }
                index
            },
        );
    }
    if enabled("ingest") {
        measure(
            "collection_ingest_10k",
            rounds,
            10_000,
            || batches(10_000, 256),
            |batches| {
                let mut index =
                    RetrievalCollection::new(CONFIG, ChunkingConfig::default()).unwrap();
                if reserve {
                    index.reserve(10_000, 10_000);
                }
                for batch in batches {
                    index.add_documents(batch).unwrap();
                }
                index
            },
        );
    }
    if enabled("query") {
        let mut index = collection(10_000, reserve);
        let query_vectors: Vec<_> = (0..64).map(vector).collect();
        let queries: Vec<_> = (0..64)
            .map(|i| match i % 4 {
                0 => format!("topic{i}"),
                1 => "search".into(),
                2 => format!("record{i}"),
                _ => "missing".into(),
            })
            .collect();
        let options = HybridSearchOptions::default();
        for (name, mode) in [
            ("query_exact_64", VectorSearchMode::Exact),
            (
                "query_ivf_64",
                VectorSearchMode::Ivf {
                    build: build_options(),
                    probes: 4,
                },
            ),
        ] {
            index.set_vector_search_mode(mode)?;
            let expected: Vec<_> = queries
                .iter()
                .zip(&query_vectors)
                .map(|(q, v)| index.search_with_vector(q, v, options).unwrap())
                .collect();
            measure(
                name,
                rounds,
                64,
                || {
                    let mut scratch = Bm25Scratch::default();
                    for ((q, v), expected) in queries.iter().zip(&query_vectors).zip(&expected) {
                        assert_eq!(
                            &index
                                .search_with_vector_and_scratch(q, v, options, &mut scratch)
                                .unwrap(),
                            expected
                        );
                    }
                    scratch
                },
                |mut scratch| {
                    for (q, v) in queries.iter().zip(&query_vectors) {
                        black_box(
                            index
                                .search_with_vector_and_scratch(
                                    black_box(q),
                                    black_box(v),
                                    options,
                                    &mut scratch,
                                )
                                .unwrap(),
                        );
                    }
                    scratch
                },
            );
        }
    }
    if enabled("ivf") {
        let mut source = VectorIndex::new(DIMENSIONS)?;
        for row in 0..10_000 {
            source.insert(row, &vector(row as usize))?;
        }
        measure(
            "ivf_build_10k",
            rounds,
            10_000,
            || (),
            |()| IvfIndex::build(&source, build_options()).unwrap(),
        );
        let index = IvfIndex::build(&source, build_options())?;
        measure(
            "ivf_lookup_1m",
            rounds,
            1_000_000,
            || (),
            |()| {
                let mut sum = 0.0_f64;
                for i in 0..1_000_000_usize {
                    sum +=
                        f64::from(index.vector(black_box((i * 7919 % 10_000) as u32)).unwrap()[0]);
                }
                black_box(sum)
            },
        );
    }
    if enabled("storage") {
        measure(
            "persistent_append_4096",
            rounds,
            4096,
            || {
                let dir = tempfile::tempdir().unwrap();
                let stored = PersistentCollection::create(
                    dir.path().join("data"),
                    CONFIG,
                    ChunkingConfig::default(),
                )
                .unwrap();
                (dir, stored, batches(4096, 64))
            },
            |(dir, mut stored, batches)| {
                for batch in batches {
                    stored.add_documents(batch).unwrap();
                }
                (dir, stored)
            },
        );
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("data");
        let mut stored = PersistentCollection::create(&path, CONFIG, ChunkingConfig::default())?;
        for batch in batches(4096, 64) {
            stored.add_documents(batch)?;
        }
        drop(stored);
        measure(
            "persistent_reopen_4096",
            rounds,
            4096,
            || (),
            |()| PersistentCollection::open(&path, CONFIG).unwrap(),
        );
    }
    Ok(())
}
