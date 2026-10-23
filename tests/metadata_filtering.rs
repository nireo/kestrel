use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use kestrel::{
    ChunkHit, ChunkId, ChunkingConfig, Collection, CollectionError, Document, EmbeddingConfig,
    EmbeddingError, Field, Filter, FiniteF64, IvfBuildOptions, Metadata, PersistentCollection,
    PreparedChunk, PreparedDocument, RetrievalCollection, SearchOptions, SearchScratch,
    VectorError, VectorSearchMode,
};

const CONFIG: EmbeddingConfig = EmbeddingConfig {
    model_id: "metadata-test-vectors",
    artifact_repo: "no-download",
    artifact_file: "unused.onnx",
    dimensions: 2,
    max_tokens: 512,
    query_instruction: "",
};
const TEAM: Field<'static, String> = Field::new("team");
const ACTIVE: Field<'static, bool> = Field::new("active");
const YEAR: Field<'static, i64> = Field::new("year");
const RATING: Field<'static, FiniteF64> = Field::new("rating");
const TAGS: Field<'static, Vec<String>> = Field::new("tags");

fn finite(value: f64) -> FiniteF64 {
    FiniteF64::new(value).unwrap()
}

fn metadata(team: &str, active: bool, year: i64, rating: f64, tags: &[&str]) -> Metadata {
    let mut metadata = Metadata::default();
    metadata.insert(TEAM, team);
    metadata.insert(ACTIVE, active);
    metadata.insert(YEAR, year);
    metadata.insert(RATING, finite(rating));
    metadata.insert(
        TAGS,
        tags.iter().map(|tag| (*tag).to_owned()).collect::<Vec<_>>(),
    );
    metadata
}

fn single(text: &str, embedding: [f32; 2]) -> PreparedDocument {
    PreparedDocument {
        text: text.into(),
        metadata: Default::default(),
        chunks: vec![PreparedChunk {
            byte_range: 0..text.len(),
            embedding: embedding.to_vec(),
        }],
    }
}

fn documents() -> Vec<PreparedDocument> {
    let mut blue = metadata("blue", true, 2025, 4.5, &["rust", "search"]);
    blue.insert(Field::<String>::new("note"), "metadataonlytoken");
    let mut wrong = Metadata::default();
    wrong.insert(Field::<bool>::new("team"), true);
    wrong.insert(Field::<String>::new("active"), "true");
    wrong.insert(Field::<FiniteF64>::new("year"), finite(2025.0));
    wrong.insert(Field::<i64>::new("rating"), 4_i64);
    wrong.insert(Field::<String>::new("tags"), "rust");
    vec![
        single("rust rust", [1.0, 0.0]).with_metadata(metadata(
            "red",
            false,
            2019,
            1.0,
            &["legacy"],
        )),
        PreparedDocument {
            text: "café rust engine".into(),
            metadata: blue,
            chunks: vec![
                PreparedChunk {
                    byte_range: 0..10,
                    embedding: vec![0.0, 1.0],
                },
                PreparedChunk {
                    byte_range: 6..17,
                    embedding: vec![1.0, 0.0],
                },
            ],
        },
        single("guide engine", [-1.0, 0.0]).with_metadata(metadata(
            "blue",
            false,
            2024,
            3.0,
            &["docs"],
        )),
        single("rust metadata", [0.0, -1.0]),
        single("engine rust", [1.0, 1.0]).with_metadata(wrong),
    ]
}

fn ivf_mode(probes: usize) -> VectorSearchMode {
    VectorSearchMode::Ivf {
        build: IvfBuildOptions {
            clusters: 3,
            training_samples: 32,
            iterations: 8,
            seed: 42,
        },
        probes,
    }
}

fn collection(documents: &[PreparedDocument], mode: VectorSearchMode) -> RetrievalCollection {
    let mut core = RetrievalCollection::new(CONFIG, ChunkingConfig::default()).unwrap();
    core.add_documents(documents.to_vec()).unwrap();
    core.set_vector_search_mode(mode).unwrap();
    core
}

fn terms(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(str::to_owned)
        .collect()
}

// Brute-force oracle: no collection searches, Filter::matches, top-k helpers,
// vector index, or RRF implementation. Eligibility is supplied independently.
// BM25 statistics include every passage, even those excluded by the filter.
fn expected(
    documents: &[PreparedDocument],
    eligible: &[usize],
    query: &str,
    vector: &[f32],
    depth: usize,
    limit: usize,
) -> Vec<ChunkHit> {
    if depth == 0 || limit == 0 || documents.is_empty() {
        return Vec::new();
    }
    let rows: Vec<_> = documents
        .iter()
        .enumerate()
        .flat_map(|(doc_id, document)| {
            document.chunks.iter().map(move |chunk| {
                (
                    doc_id,
                    chunk,
                    terms(&document.text[chunk.byte_range.clone()]),
                )
            })
        })
        .collect();
    let query_terms: BTreeSet<_> = terms(query).into_iter().collect();
    let average_length =
        rows.iter().map(|(_, _, terms)| terms.len()).sum::<usize>() as f64 / rows.len() as f64;
    let mut lexical = Vec::new();
    let mut semantic = Vec::new();
    for (row, (doc_id, chunk, terms)) in rows.iter().enumerate() {
        if !eligible.contains(doc_id) {
            continue;
        }
        let mut score = 0.0;
        for term in &query_terms {
            let tf = terms.iter().filter(|word| *word == term).count() as f64;
            if tf == 0.0 {
                continue;
            }
            let df = rows
                .iter()
                .filter(|(_, _, words)| words.contains(term))
                .count() as f64;
            let idf = ((rows.len() as f64 - df + 0.5) / (df + 0.5)).ln_1p();
            score +=
                idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * terms.len() as f64 / average_length));
        }
        if score > 0.0 {
            lexical.push((row, score));
        }
        let dot: f64 = chunk
            .embedding
            .iter()
            .zip(vector)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let length = |values: &[f32]| {
            values
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        semantic.push((row, dot / (length(&chunk.embedding) * length(vector))));
    }
    let rank = |hits: &mut Vec<(usize, f64)>| {
        hits.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        hits.truncate(depth);
    };
    rank(&mut lexical);
    rank(&mut semantic);
    let mut hits = Vec::new();
    for (row, (doc_id, _, _)) in rows.iter().enumerate() {
        let bm25_rank = lexical
            .iter()
            .position(|hit| hit.0 == row)
            .map(|rank| rank + 1);
        let vector_rank = semantic
            .iter()
            .position(|hit| hit.0 == row)
            .map(|rank| rank + 1);
        if bm25_rank.is_some() || vector_rank.is_some() {
            hits.push(ChunkHit {
                chunk_id: ChunkId(row as u32),
                doc_id: *doc_id as u32,
                score: bm25_rank.map_or(0.0, |rank| 1.0 / (60.0 + rank as f64))
                    + vector_rank.map_or(0.0, |rank| 1.0 / (60.0 + rank as f64)),
                bm25_rank,
                vector_rank,
            });
        }
    }
    hits.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });
    hits.truncate(limit);
    hits
}

#[test]
fn public_document_builders_and_typed_metadata_need_no_model() {
    let options = SearchOptions::default();
    assert_eq!((options.candidate_depth, options.limit), (50, 10));
    assert!(options.filter.is_none());
    for document in [
        Document::new("rust"),
        Document::from("rust"),
        Document::from(String::from("rust")),
    ] {
        assert_eq!(document.text, "rust");
        assert_eq!(document.metadata, Metadata::default());
    }
    let mut attributes = metadata("blue", true, 2025, 4.5, &["rust"]);
    let document = Document::new("rust").with_metadata(attributes.clone());
    assert_eq!(
        document.metadata.get(TEAM).map(String::as_str),
        Some("blue")
    );
    assert_eq!(document.metadata.get(ACTIVE), Some(&true));
    assert_eq!(document.metadata.get(YEAR), Some(&2025));
    assert_eq!(document.metadata.get(RATING), Some(&finite(4.5)));
    assert_eq!(document.metadata.get(TAGS).unwrap(), &["rust"]);
    attributes.insert(Field::<bool>::new("team"), false);
    assert_eq!(attributes.get(TEAM), None);
    assert_eq!(attributes.get(Field::<bool>::new("team")), Some(&false));
    assert!(!TEAM.eq("blue").matches(&attributes));
    assert!(Filter::negate(&TEAM.eq("blue")).matches(&attributes));
    assert!(TEAM.exists().matches(&attributes));
    // Metadata owns borrowed names and values, rather than keeping their source alive.
    {
        let name = String::from("temporary");
        let value = String::from("owned after insertion");
        attributes.insert(Field::<String>::new(&name), value.as_str());
    }
    assert_eq!(
        attributes
            .get(Field::<String>::new("temporary"))
            .map(String::as_str),
        Some("owned after insertion")
    );
    let prepared = single("rust", [1.0, 0.0]).with_metadata(document.metadata.clone());
    assert_eq!(prepared.metadata, document.metadata);
}

#[test]
fn typed_filters_select_whole_documents_including_overlapping_unicode_passages() {
    let documents = documents();
    let core = collection(&documents, VectorSearchMode::Exact);
    for (id, document) in documents.iter().enumerate() {
        assert_eq!(core.metadata(id as u32), Some(&document.metadata));
    }
    assert_eq!(core.metadata(documents.len() as u32), None);
    let blue = TEAM.eq("blue");
    let alternatives = [ACTIVE.eq(true), YEAR.eq(2019)];
    let clauses = [
        TEAM.eq("blue"),
        YEAR.range(2024..=2025),
        TAGS.contains("rust"),
    ];
    let cases: Vec<(Filter<'_>, &[usize])> = vec![
        (TEAM.eq("blue"), &[1, 2]),
        (TEAM.eq("Blue"), &[]),
        (TEAM.in_set(&["red", "blue"]), &[0, 1, 2]),
        (TEAM.in_set(&[]), &[]),
        (ACTIVE.eq(true), &[1]),
        (ACTIVE.eq(false), &[0, 2]),
        (YEAR.eq(2025), &[1]),
        (YEAR.in_set(&[2019, 2025]), &[0, 1]),
        (YEAR.range(2024..=2025), &[1, 2]),
        (YEAR.range(..2020), &[0]),
        (RATING.eq(finite(4.5)), &[1]),
        (RATING.range(finite(3.0)..finite(4.5)), &[2]),
        (TAGS.contains("rust"), &[1]),
        (TAGS.contains("Rust"), &[]),
        (TEAM.exists(), &[0, 1, 2, 4]),
        (ACTIVE.exists(), &[0, 1, 2, 4]),
        (YEAR.exists(), &[0, 1, 2, 4]),
        (RATING.exists(), &[0, 1, 2, 4]),
        (TAGS.exists(), &[0, 1, 2, 4]),
        (Field::<String>::new("missing").exists(), &[]),
        (Filter::negate(&blue), &[0, 3, 4]),
        (Filter::or(&alternatives), &[0, 1]),
        (Filter::and(&clauses), &[1]),
        (Filter::and(&[]), &[0, 1, 2, 3, 4]),
        (Filter::or(&[]), &[]),
    ];
    for (filter, eligible) in cases {
        let options = SearchOptions {
            candidate_depth: usize::MAX,
            limit: usize::MAX,
            filter: Some(&filter),
        };
        let hits = core
            .search_with_vector("RUST rust engine café", &[1.0, 0.0], options)
            .unwrap();
        assert_eq!(
            hits,
            expected(
                &documents,
                eligible,
                "RUST rust engine café",
                &[1.0, 0.0],
                usize::MAX,
                usize::MAX
            ),
            "{filter:?}"
        );
        let blue_chunks: Vec<_> = hits
            .iter()
            .filter(|hit| hit.doc_id == 1)
            .map(|hit| hit.chunk_id)
            .collect();
        assert_eq!(blue_chunks.len(), if eligible.contains(&1) { 2 } else { 0 });
    }
    assert_eq!(core.chunk_text(ChunkId(1)), Some("café rust"));
    assert_eq!(core.chunk_text(ChunkId(2)), Some("rust engine"));
    let hits = core
        .search_with_vector("metadataonlytoken", &[1.0, 0.0], SearchOptions::default())
        .unwrap();
    assert!(hits.iter().all(|hit| hit.bm25_rank.is_none()));
}

#[test]
fn eligibility_precedes_both_modalities_candidate_depth_and_fusion() {
    let allowed = metadata("blue", true, 2025, 4.5, &[]);
    let documents = vec![
        single("needle", [1.0, 0.0]),
        single("needle padding", [-1.0, 0.0]).with_metadata(allowed.clone()),
        single("other", [0.0, 1.0]).with_metadata(allowed),
    ];
    let filter = ACTIVE.eq(true);
    let options = SearchOptions {
        candidate_depth: 1,
        limit: 10,
        filter: Some(&filter),
    };
    let oracle = vec![
        ChunkHit {
            chunk_id: ChunkId(1),
            doc_id: 1,
            score: 1.0 / 61.0,
            bm25_rank: Some(1),
            vector_rank: None,
        },
        ChunkHit {
            chunk_id: ChunkId(2),
            doc_id: 2,
            score: 1.0 / 61.0,
            bm25_rank: None,
            vector_rank: Some(1),
        },
    ];
    for mode in [VectorSearchMode::Exact, ivf_mode(1)] {
        let core = collection(&documents, mode);
        let unfiltered = core
            .search_with_vector(
                "needle",
                &[1.0, 0.0],
                SearchOptions {
                    filter: None,
                    ..options
                },
            )
            .unwrap();
        assert_eq!(unfiltered.len(), 1);
        assert_eq!(unfiltered[0].doc_id, 0);
        assert_eq!(unfiltered[0].bm25_rank, Some(1));
        assert_eq!(unfiltered[0].vector_rank, Some(1));
        assert_eq!(
            core.search_with_vector("needle", &[1.0, 0.0], options)
                .unwrap(),
            oracle
        );
    }
}

#[test]
fn filtering_keeps_bm25_global_document_frequencies_and_average_length() {
    let allowed = metadata("blue", true, 2025, 1.0, &[]);
    for (documents, query) in [
        // Rejected text makes "rare" rarer than "common" globally. Rebuilding
        // statistics on accepted passages would instead tie and choose ID 1.
        (
            vec![
                single("common common common", [1.0, 0.0]),
                single("common padding", [0.0, 1.0]).with_metadata(allowed.clone()),
                single("rare padding", [0.0, 1.0]).with_metadata(allowed.clone()),
            ],
            "common rare",
        ),
        // This rejected passage raises global average length enough to favor
        // the longer, repeated-term passage. Accepted-only statistics favor ID 1.
        (
            vec![
                single(&"padding ".repeat(50), [1.0, 0.0]),
                single("needle", [0.0, 1.0]).with_metadata(allowed.clone()),
                single("needle needle padding padding", [0.0, 1.0]).with_metadata(allowed.clone()),
            ],
            "needle",
        ),
    ] {
        let core = collection(&documents, VectorSearchMode::Exact);
        let filter = TEAM.eq("blue");
        let options = SearchOptions {
            candidate_depth: 1,
            limit: 10,
            filter: Some(&filter),
        };
        let hits = core
            .search_with_vector(query, &[0.0, 1.0], options)
            .unwrap();
        assert_eq!(
            hits,
            expected(&documents, &[1, 2], query, &[0.0, 1.0], 1, 10)
        );
        assert_eq!(hits.len(), 2);
        assert_eq!(
            (hits[0].doc_id, hits[0].bm25_rank, hits[0].vector_rank),
            (1, None, Some(1))
        );
        assert_eq!(
            (hits[1].doc_id, hits[1].bm25_rank, hits[1].vector_rank),
            (2, Some(1), None)
        );
    }
}

#[test]
fn filtered_ivf_is_exact_before_and_after_buffered_writes_and_rebuilds() {
    let mut documents = documents();
    let mut core = collection(&documents, ivf_mode(1));
    let all = Filter::and(&[]);
    let blue = TEAM.eq("blue");
    let none = Filter::or(&[]);
    for phase in 0..3 {
        if phase == 1 {
            let fresh = single("fresh rust", [-1.0, -1.0]).with_metadata(metadata(
                "blue",
                true,
                2026,
                5.0,
                &["new"],
            ));
            core.add_document(fresh.clone()).unwrap();
            documents.push(fresh);
            assert_eq!(core.buffered_vector_count(), 1);
        } else if phase == 2 {
            core.rebuild_vector_index().unwrap();
            assert_eq!(core.buffered_vector_count(), 0);
        }
        let all_ids: Vec<_> = (0..documents.len()).collect();
        let blue_ids = if phase == 0 {
            vec![1, 2]
        } else {
            vec![1, 2, 5]
        };
        for (filter, eligible) in [
            (&all, all_ids.as_slice()),
            (&blue, blue_ids.as_slice()),
            (&none, &[][..]),
        ] {
            for depth in [1, 2, usize::MAX] {
                let options = SearchOptions {
                    candidate_depth: depth,
                    limit: usize::MAX,
                    filter: Some(filter),
                };
                for (query, vector) in [
                    ("rust", [1.0, 0.0]),
                    ("engine", [0.0, 1.0]),
                    ("unknown", [-1.0, 0.0]),
                    ("CAFÉ café", [-1.0, -1.0]),
                ] {
                    assert_eq!(
                        core.search_with_vector(query, &vector, options).unwrap(),
                        expected(&documents, eligible, query, &vector, depth, usize::MAX),
                        "phase {phase}, {filter:?}"
                    );
                }
            }
        }
        let options = SearchOptions {
            candidate_depth: usize::MAX,
            limit: usize::MAX,
            filter: None,
        };
        let approximate = core
            .search_with_vector("unknown", &[1.0, 0.0], options)
            .unwrap();
        let exact = core
            .search_with_vector(
                "unknown",
                &[1.0, 0.0],
                SearchOptions {
                    filter: Some(&all),
                    ..options
                },
            )
            .unwrap();
        assert_eq!(exact.len(), core.chunk_count());
        assert!(
            approximate.len() < exact.len(),
            "unfiltered IVF must still probe only one cluster"
        );
        assert_eq!(core.vector_search_mode(), ivf_mode(1));
    }
}

#[test]
fn scratch_and_output_reuse_switch_filters_indexes_dimensions_and_errors() {
    let documents = documents();
    let blue = TEAM.eq("blue");
    let not_blue = Filter::negate(&blue);
    let all = Filter::and(&[]);
    let none = Filter::or(&[]);
    let mut scratch = SearchScratch::default();
    let mut output = Vec::new();
    let all_ids: Vec<_> = (0..documents.len()).collect();
    for mode in [VectorSearchMode::Exact, ivf_mode(1), ivf_mode(usize::MAX)] {
        let core = collection(&documents, mode);
        let held = core
            .search_with_scratch("rust", &[1.0, 0.0], SearchOptions::default(), &mut scratch)
            .unwrap();
        for (filter, eligible) in [
            (Some(&blue), &[1, 2][..]),
            (Some(&none), &[][..]),
            (None, all_ids.as_slice()),
            (Some(&not_blue), &[0, 3, 4][..]),
            (Some(&all), all_ids.as_slice()),
        ] {
            for (candidate_depth, limit) in
                [(1, 1), (2, 10), (0, 10), (50, 0), (usize::MAX, usize::MAX)]
            {
                let options = SearchOptions {
                    candidate_depth,
                    limit,
                    filter,
                };
                for (query, vector) in [
                    ("RUST rust", [1.0, 0.0]),
                    ("engine café", [0.0, 1.0]),
                    ("missing", [-1.0, 0.0]),
                ] {
                    let hits = core.search_with_vector(query, &vector, options).unwrap();
                    if filter.is_some() || mode == VectorSearchMode::Exact {
                        assert_eq!(
                            hits,
                            expected(&documents, eligible, query, &vector, candidate_depth, limit)
                        );
                    }
                    assert_eq!(
                        core.search_with_scratch(query, &vector, options, &mut scratch)
                            .unwrap(),
                        hits
                    );
                    core.search_into(query, &vector, options, &mut scratch, &mut output)
                        .unwrap();
                    assert_eq!(output, hits);
                    for invalid in [
                        &[0.0, 0.0][..],
                        &[1.0][..],
                        &[f32::NAN, 1.0][..],
                        &[1.0, f32::INFINITY][..],
                    ] {
                        assert!(matches!(
                            core.search_with_vector(query, invalid, options),
                            Err(CollectionError::Vector(_))
                        ));
                        assert!(matches!(
                            core.search_with_scratch(query, invalid, options, &mut scratch),
                            Err(CollectionError::Vector(_))
                        ));
                        output.clone_from(&held);
                        assert!(matches!(
                            core.search_into(query, invalid, options, &mut scratch, &mut output),
                            Err(CollectionError::Vector(_))
                        ));
                        assert!(output.is_empty());
                    }
                    output.clone_from(&held);
                    assert!(matches!(
                        core.search_into(" \n", &vector, options, &mut scratch, &mut output),
                        Err(CollectionError::Embedding(EmbeddingError::EmptyText {
                            index: 0
                        }))
                    ));
                    assert!(output.is_empty());
                    core.search_into(query, &vector, options, &mut scratch, &mut output)
                        .unwrap();
                    assert_eq!(output, hits);
                }
            }
        }
        assert_eq!(
            held,
            core.search_with_vector("rust", &[1.0, 0.0], SearchOptions::default())
                .unwrap()
        );
    }
    let mut other = RetrievalCollection::new(
        EmbeddingConfig {
            dimensions: 3,
            ..CONFIG
        },
        ChunkingConfig::default(),
    )
    .unwrap();
    let mut document = single("new index", [1.0, 0.0]);
    document.chunks[0].embedding = vec![0.0, 0.0, 1.0];
    other.add_document(document).unwrap();
    for (filter, expected_len) in [
        (Some(&blue), 0),
        (None, 1),
        (Some(&all), 1),
        (Some(&none), 0),
        (None, 1),
    ] {
        let options = SearchOptions {
            filter,
            ..Default::default()
        };
        let hits = other
            .search_with_vector("new", &[0.0, 0.0, 1.0], options)
            .unwrap();
        assert_eq!(hits.len(), expected_len);
        assert_eq!(
            other
                .search_with_scratch("new", &[0.0, 0.0, 1.0], options, &mut scratch)
                .unwrap(),
            hits
        );
        other
            .search_into("new", &[0.0, 0.0, 1.0], options, &mut scratch, &mut output)
            .unwrap();
        assert_eq!(output, hits);
    }
}

#[test]
fn empty_or_fully_excluded_searches_validate_even_with_zero_depth_or_limit() {
    let none = Filter::or(&[]);
    let all = Filter::and(&[]);
    let mut scratch = SearchScratch::default();
    let mut output = Vec::new();
    for documents in [vec![], documents()] {
        let core = collection(&documents, ivf_mode(1));
        for (filter, excludes_all) in [(Some(&none), true), (Some(&all), false), (None, false)] {
            for (candidate_depth, limit) in [(0, 10), (10, 0), (10, 10)] {
                let options = SearchOptions {
                    candidate_depth,
                    limit,
                    filter,
                };
                assert!(matches!(
                    core.search_with_vector(" \t", &[1.0, 0.0], options),
                    Err(CollectionError::Embedding(EmbeddingError::EmptyText {
                        index: 0
                    }))
                ));
                assert!(matches!(
                    core.search_with_scratch(" \t", &[1.0, 0.0], options, &mut scratch),
                    Err(CollectionError::Embedding(EmbeddingError::EmptyText {
                        index: 0
                    }))
                ));
                assert!(matches!(
                    core.search_into(" \t", &[1.0, 0.0], options, &mut scratch, &mut output),
                    Err(CollectionError::Embedding(EmbeddingError::EmptyText {
                        index: 0
                    }))
                ));
                assert!(output.is_empty());
                for invalid in [&[1.0][..], &[0.0, 0.0][..], &[f32::NAN, 0.0][..]] {
                    assert!(matches!(
                        core.search_with_vector("rust", invalid, options),
                        Err(CollectionError::Vector(_))
                    ));
                    assert!(matches!(
                        core.search_with_scratch("rust", invalid, options, &mut scratch),
                        Err(CollectionError::Vector(_))
                    ));
                    assert!(matches!(
                        core.search_into("rust", invalid, options, &mut scratch, &mut output),
                        Err(CollectionError::Vector(_))
                    ));
                    assert!(output.is_empty());
                }
                core.search_into("rust", &[1.0, 0.0], options, &mut scratch, &mut output)
                    .unwrap();
                if documents.is_empty() || candidate_depth == 0 || limit == 0 || excludes_all {
                    assert!(output.is_empty());
                }
            }
        }
    }
}

#[test]
fn filtered_batches_preserve_worker_order_errors_and_concurrent_reader_rankings() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RetrievalCollection>();
    assert_send_sync::<Metadata>();
    assert_send_sync::<Filter<'static>>();
    assert_send_sync::<SearchOptions<'static>>();
    let mut documents = documents();
    let mut core = collection(&documents, ivf_mode(1));
    let fresh = single("fresh rust", [-1.0, -1.0]).with_metadata(metadata(
        "blue",
        true,
        2026,
        5.0,
        &["new"],
    ));
    core.add_document(fresh.clone()).unwrap();
    documents.push(fresh);
    let inputs = [
        ("rust", &[1.0, 0.0][..]),
        ("engine", &[0.0, 1.0][..]),
        ("unknown", &[-1.0, 0.0][..]),
        ("CAFÉ café", &[-1.0, -1.0][..]),
        ("fresh", &[0.0, -1.0][..]),
    ];
    let blue = TEAM.eq("blue");
    let none = Filter::or(&[]);
    let all = Filter::and(&[]);
    for (filter, eligible) in [
        (&blue, &[1, 2, 5][..]),
        (&none, &[][..]),
        (&all, &[0, 1, 2, 3, 4, 5][..]),
    ] {
        for (candidate_depth, limit) in [(1, 10), (3, 2), (50, 10), (0, 10), (50, 0)] {
            let options = SearchOptions {
                candidate_depth,
                limit,
                filter: Some(filter),
            };
            let rankings: Vec<_> = inputs
                .iter()
                .map(|(query, vector)| {
                    expected(&documents, eligible, query, vector, candidate_depth, limit)
                })
                .collect();
            for workers in [1, 2, 4, usize::MAX] {
                let workers = NonZeroUsize::new(workers).unwrap();
                assert_eq!(
                    core.search_many_with_vectors(&inputs, options, workers)
                        .unwrap(),
                    rankings
                );
                assert!(
                    core.search_many_with_vectors(&[], options, workers)
                        .unwrap()
                        .is_empty()
                );
                let invalid = [
                    ("rust", &[1.0, 0.0][..]),
                    ("bad vector", &[1.0][..]),
                    (" ", &[1.0, 0.0][..]),
                ];
                assert!(matches!(
                    core.search_many_with_vectors(&invalid, options, workers),
                    Err(CollectionError::Vector(
                        VectorError::DimensionMismatch { .. }
                    ))
                ));
                let blank = [(" ", &[1.0, 0.0][..]), ("bad vector", &[0.0, 0.0][..])];
                assert!(matches!(
                    core.search_many_with_vectors(&blank, options, workers),
                    Err(CollectionError::Embedding(EmbeddingError::EmptyText {
                        index: 0
                    }))
                ));
            }
            std::thread::scope(|scope| {
                for _ in 0..4 {
                    scope.spawn(|| {
                        let mut scratch = SearchScratch::default();
                        let mut output = Vec::new();
                        for ((query, vector), ranking) in inputs.iter().zip(&rankings) {
                            assert_eq!(
                                &core
                                    .search_with_scratch(query, vector, options, &mut scratch)
                                    .unwrap(),
                                ranking
                            );
                            core.search_into(query, vector, options, &mut scratch, &mut output)
                                .unwrap();
                            assert_eq!(&output, ranking);
                        }
                    });
                }
            });
        }
    }
}

#[test]
fn durable_metadata_roundtrips_and_rejected_batches_publish_nothing_or_consume_ids() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("metadata.kestrel");
    let mut documents = documents();
    let mut stored =
        PersistentCollection::create(&path, CONFIG, ChunkingConfig::default()).unwrap();
    assert_eq!(
        stored.add_documents(documents.clone()).unwrap(),
        [0, 1, 2, 3, 4]
    );
    stored.set_vector_search_mode(ivf_mode(1)).unwrap();
    let mut attributes = metadata("blue", true, i64::MAX, f64::MAX, &["", "rust", "rust"]);
    attributes.insert(Field::<i64>::new("minimum"), i64::MIN);
    attributes.insert(Field::<FiniteF64>::new("negative_zero"), finite(-0.0));
    let fresh = single("fresh engine", [-1.0, -1.0]).with_metadata(attributes);
    assert_eq!(stored.add_document(fresh.clone()).unwrap(), 5);
    documents.push(fresh);
    assert_eq!(stored.retrieval().buffered_vector_count(), 1);
    let blue = TEAM.eq("blue");
    let options = SearchOptions {
        filter: Some(&blue),
        ..Default::default()
    };
    let ranking = expected(&documents, &[1, 2, 5], "rust engine", &[1.0, 0.0], 50, 10);
    assert_eq!(
        stored
            .retrieval()
            .search_with_vector("rust engine", &[1.0, 0.0], options)
            .unwrap(),
        ranking
    );
    let bytes = std::fs::read(&path).unwrap();
    for malformed_ranges in [false, true] {
        let valid = single("unpublished rust", [1.0, 0.0]).with_metadata(metadata(
            "rejected",
            true,
            2027,
            2.0,
            &["secret"],
        ));
        let mut invalid =
            single("café", [0.0, 0.0]).with_metadata(metadata("rejected", false, 2027, 2.0, &[]));
        if malformed_ranges {
            invalid.chunks[0].embedding = vec![1.0, 0.0];
            invalid.chunks[0].byte_range.end = 4;
        }
        assert!(stored.add_documents(vec![valid, invalid]).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(stored.retrieval().len(), 6);
        assert_eq!(stored.retrieval().chunk_count(), 7);
        assert_eq!(stored.retrieval().buffered_vector_count(), 1);
        assert_eq!(stored.retrieval().metadata(6), None);
        assert_eq!(stored.retrieval().document(6), None);
        let rejected = TEAM.eq("rejected");
        assert!(
            stored
                .retrieval()
                .search_with_vector(
                    "rust",
                    &[1.0, 0.0],
                    SearchOptions {
                        filter: Some(&rejected),
                        ..Default::default()
                    }
                )
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            stored
                .retrieval()
                .search_with_vector("rust engine", &[1.0, 0.0], options)
                .unwrap(),
            ranking
        );
    }
    assert!(stored.add_documents(vec![]).unwrap().is_empty());
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    drop(stored);
    let mut restored = PersistentCollection::open(&path, CONFIG).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    assert_eq!(
        restored.retrieval().vector_search_mode(),
        VectorSearchMode::Exact
    );
    for (id, document) in documents.iter().enumerate() {
        assert_eq!(
            restored.retrieval().document(id as u32),
            Some(document.text.as_str())
        );
        assert_eq!(
            restored.retrieval().metadata(id as u32),
            Some(&document.metadata)
        );
    }
    let zero = restored
        .retrieval()
        .metadata(5)
        .unwrap()
        .get(Field::<FiniteF64>::new("negative_zero"))
        .unwrap();
    assert_eq!(zero.get().to_bits(), (-0.0_f64).to_bits());
    assert_eq!(
        restored
            .retrieval()
            .search_with_vector("rust engine", &[1.0, 0.0], options)
            .unwrap(),
        ranking
    );
    restored.set_vector_search_mode(ivf_mode(1)).unwrap();
    restored.rebuild_vector_index().unwrap();
    assert_eq!(
        restored
            .retrieval()
            .search_with_vector("rust engine", &[1.0, 0.0], options)
            .unwrap(),
        ranking
    );
    let next =
        single("next rust", [0.0, -1.0]).with_metadata(metadata("blue", true, 2028, 4.0, &["new"]));
    assert_eq!(restored.add_document(next.clone()).unwrap(), 6);
    assert_eq!(restored.retrieval().chunk(ChunkId(7)).unwrap().doc_id, 6);
    documents.push(next);
    drop(restored);
    let restored = PersistentCollection::open(&path, CONFIG).unwrap();
    assert_eq!(restored.retrieval().len(), 7);
    assert_eq!(
        restored.retrieval().metadata(6),
        Some(&documents[6].metadata)
    );
    assert_eq!(
        restored
            .retrieval()
            .search_with_vector("rust engine", &[1.0, 0.0], options)
            .unwrap(),
        expected(
            &documents,
            &[1, 2, 5, 6],
            "rust engine",
            &[1.0, 0.0],
            50,
            10
        )
    );
}

#[test]
#[ignore = "loads the real model and downloads its files if not cached"]
fn local_ingestion_preparation_and_embedded_batch_search_preserve_metadata()
-> Result<(), Box<dyn std::error::Error>> {
    let mut local = Collection::new()?;
    let attributes = metadata("blue", true, 2025, 4.5, &["rust"]);
    let document = Document::new("An inverted index supports keyword search.")
        .with_metadata(attributes.clone());
    let prepared = local.prepare_document(document.clone())?;
    assert_eq!(prepared.metadata, attributes);
    assert!(local.prepare_document("plain text")?.metadata == Metadata::default());
    let batch = local.prepare_documents([
        document.clone(),
        Document::new("Pasta cooks in salted water."),
    ])?;
    assert_eq!(batch[0].metadata, attributes);
    assert_eq!(batch[1].metadata, Metadata::default());
    assert!(local.is_empty());
    assert_eq!(local.add_document(document)?, 0);
    assert_eq!(
        local.add_documents([Document::new("Pasta cooks in salted water.")])?,
        [1]
    );
    assert_eq!(local.metadata(0), Some(&attributes));
    assert_eq!(local.metadata(1), Some(&Metadata::default()));
    assert_eq!(local.metadata(2), None);
    let filter = TEAM.eq("blue");
    let options = SearchOptions {
        filter: Some(&filter),
        ..Default::default()
    };
    let queries = ["keyword search", "pasta"];
    let vectors = local.embed_queries(&queries)?;
    let expected: Vec<_> = queries
        .iter()
        .zip(&vectors)
        .map(|(query, vector)| local.search_with_vector(query, vector, options).unwrap())
        .collect();
    assert!(
        expected
            .iter()
            .all(|hits| hits.len() == 1 && hits[0].doc_id == 0)
    );
    for (query, hits) in queries.iter().zip(&expected) {
        assert_eq!(&local.search(query, options)?, hits);
    }
    assert_eq!(local.search_many(&queries, options)?, expected);
    for workers in [1, 2] {
        assert_eq!(
            local.search_many_with_workers(
                &queries,
                options,
                NonZeroUsize::new(workers).unwrap()
            )?,
            expected
        );
    }
    Ok(())
}
