use std::collections::HashSet;

use kestrel::{InvertedIndex, Posting, SearchHit};

fn index_documents(documents: &[&str]) -> InvertedIndex {
    let mut index = InvertedIndex::new();
    for document in documents {
        index.add_document(*document).unwrap();
    }
    index
}

#[test]
fn postings_are_compact_sorted_and_count_each_document_once() {
    assert_eq!(std::mem::size_of::<Posting>(), 8);
    let index = index_documents(&["rust search search", "rust engine", "search"]);
    assert_eq!(
        index.postings("search"),
        &[
            Posting {
                doc_id: 0,
                term_frequency: 2
            },
            Posting {
                doc_id: 2,
                term_frequency: 1
            },
        ]
    );
    assert_eq!(index.len(), 3);
    assert_eq!(index.vocabulary_len(), 3);
    assert_eq!(index.document(1), Some("rust engine"));
    assert_eq!(index.document_length(0), Some(3));
    assert_eq!(index.average_document_length(), 2.0);
    assert!(index.document(3).is_none());
    assert!(index.document_length(3).is_none());
    assert!(index.postings("missing").is_empty());
}

#[test]
fn indexing_and_search_share_case_and_punctuation_rules() {
    let index = index_documents(&["RUST, Rust! café ÄITI 123", "unrelated"]);
    assert_eq!(index.document_length(0), Some(5));
    assert_eq!(index.postings("rust")[0].term_frequency, 2);
    for query in ["rust", "RuSt!", "CAFÉ", "äiti", "123"] {
        assert_eq!(index.search(query, 1)[0].doc_id, 0, "{query}");
    }
    // Matching means whole normalized tokens, not substrings.
    assert!(index.search("rus", 1).is_empty());
}

#[test]
fn empty_corpora_documents_queries_and_zero_limits_are_safe() {
    let mut index = InvertedIndex::new();
    assert!(index.is_empty());
    assert_eq!(index.average_document_length(), 0.0);
    assert!(index.search("rust", 10).is_empty());
    index.add_document("").unwrap();
    index.add_document("--- !!!").unwrap();
    assert!(index.search("rust", 10).is_empty());
    assert_eq!(index.document_length(1), Some(0));
    index.add_document("rust").unwrap();
    assert!(index.search("rust", 0).is_empty());
    assert!(index.search("", 10).is_empty());
    assert!(index.search("!!!", 10).is_empty());
    assert!(index.search("missing", 10).is_empty());
    assert_eq!(index.search("rust", usize::MAX).len(), 1);
    assert!((index.average_document_length() - 1.0 / 3.0).abs() < 1e-12);
    assert!(index.search("rust", 1)[0].score.is_finite());
}

#[test]
fn score_matches_a_hand_calculated_bm25_example() {
    let index = index_documents(&["rust rust search", "search engine"]);
    // N=2, df(rust)=1, tf=2, length=3, average length=2.5.
    let expected = 2.0_f64.ln() * (2.0 * 2.2) / (2.0 + 1.2 * (0.25 + 0.75 * 3.0 / 2.5));
    let hits = index.search("rust", 10);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, 0);
    assert!((hits[0].score - expected).abs() < 1e-12);
}

#[test]
fn repeated_and_unknown_query_terms_do_not_change_scores() {
    let index = index_documents(&["rust search", "rust", "engine"]);
    assert_eq!(
        index.search("rust", 10),
        index.search("RUST rust rust missing", 10)
    );
    // OR semantics: either term qualifies a document.
    assert_eq!(index.search("rust engine", 10).len(), 3);
}

#[test]
fn term_frequency_saturates_and_document_length_is_normalized() {
    let index = index_documents(&[
        "rust filler filler filler",
        "rust rust filler filler",
        "rust rust rust filler",
    ]);
    let hits = index.search("rust", 3);
    assert_eq!(
        hits.iter().map(|hit| hit.doc_id).collect::<Vec<_>>(),
        [2, 1, 0]
    );
    assert!(hits[0].score - hits[1].score < hits[1].score - hits[2].score);

    let index = index_documents(&["rust", "rust filler filler filler"]);
    assert_eq!(index.search("rust", 2)[0].doc_id, 0);
}

#[test]
fn rare_terms_contribute_more_than_common_terms() {
    let index = index_documents(&["rare common", "common common", "common common"]);
    assert!(index.search("rare", 1)[0].score > index.search("common", 3)[2].score);
}

#[test]
fn ties_keep_lowest_ids_and_results_remain_sorted() {
    let index = index_documents(&["rust", "rust", "rust", "rust"]);
    assert_eq!(
        index
            .search("rust", 2)
            .iter()
            .map(|hit| hit.doc_id)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert!(index.search("rust", 4).iter().all(|hit| hit.score > 0.0));
}

#[test]
fn appending_documents_updates_corpus_statistics_and_scores() {
    let mut index = index_documents(&["rust"]);
    let before = index.search("rust", 1)[0].score;
    assert_eq!(index.add_document("engine").unwrap(), 1);
    let after = index.search("rust", 1)[0].score;
    assert!(after > before);
    assert_eq!(index.add_document("rust").unwrap(), 2);
    assert_eq!(index.postings("rust")[1].doc_id, 2);
    assert_eq!(index.search("rust", 10).len(), 2);
}

/// Independent, deliberately slow scorer: inspect raw documents for every query
/// term and sort every match. Fixtures here use lowercase, whitespace-only text.
fn full_scan(documents: &[&str], query: &str, limit: usize) -> Vec<SearchHit> {
    let documents: Vec<Vec<&str>> = documents
        .iter()
        .map(|document| document.split_whitespace().collect())
        .collect();
    let mut seen = HashSet::new();
    let terms: Vec<_> = query
        .split_whitespace()
        .filter(|term| seen.insert(*term))
        .collect();
    let average_length =
        documents.iter().map(Vec::len).sum::<usize>() as f64 / documents.len() as f64;
    let mut hits = Vec::new();

    for (doc_id, document) in documents.iter().enumerate() {
        let mut score = 0.0;
        for term in &terms {
            let frequency = document.iter().filter(|token| *token == term).count() as f64;
            if frequency == 0.0 {
                continue;
            }
            let df = documents
                .iter()
                .filter(|document| document.contains(term))
                .count() as f64;
            let idf = (1.0 + (documents.len() as f64 - df + 0.5) / (df + 0.5)).ln();
            score += idf * frequency * 2.2
                / (frequency + 1.2 * (0.25 + 0.75 * document.len() as f64 / average_length));
        }
        if score > 0.0 {
            hits.push(SearchHit {
                doc_id: doc_id as u32,
                score,
            });
        }
    }
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));
    hits.truncate(limit);
    hits
}

#[test]
fn inverted_search_and_top_k_agree_with_an_independent_full_scan() {
    let vocabulary = [
        "rust", "search", "engine", "vector", "hybrid", "index", "bm25",
    ];
    let documents: Vec<String> = (0..100)
        .map(|i| {
            (0..(i * 7 % 19))
                .map(|j| vocabulary[(i * 3 + j * j) % vocabulary.len()])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    let documents: Vec<&str> = documents.iter().map(String::as_str).collect();
    let index = index_documents(&documents);
    let mut scratch = kestrel::Bm25Scratch::default();

    for query in [
        "rust",
        "search engine",
        "hybrid vector rust",
        "bm25 bm25 missing",
        "missing",
    ] {
        for limit in [0, 1, 3, 10, 100, usize::MAX] {
            let actual = index.search(query, limit);
            assert_eq!(
                actual,
                index.search_with_scratch(query, limit, &mut scratch)
            );
            let expected = full_scan(&documents, query, limit);
            assert_eq!(actual.len(), expected.len(), "{query}, limit {limit}");
            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(actual.doc_id, expected.doc_id, "{query}, limit {limit}");
                assert!((actual.score - expected.score).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn scratch_resets_across_indexes_growth_zero_limits_and_unicode_queries() {
    let mut index = index_documents(&["Café ΟΣ rust", "rust search", ""]);
    let mut scratch = kestrel::Bm25Scratch::default();
    let many_terms = (0..80)
        .map(|i| format!("term{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    index.add_document(&many_terms).unwrap();
    assert_eq!(
        index.search(&many_terms, 10),
        index.search(&format!("{many_terms} {many_terms}"), 10)
    );
    for query in ["RUST rust", "CAFÉ ΟΣ", "missing", "", &many_terms, "rust"] {
        for limit in [0, 1, usize::MAX] {
            assert_eq!(
                index.search(query, limit),
                index.search_with_scratch(query, limit, &mut scratch)
            );
        }
    }
    let smaller = index_documents(&["search", "Café"]);
    assert_eq!(
        smaller.search("search", 10),
        smaller.search_with_scratch("search", 10, &mut scratch)
    );
    index.add_document("search rust").unwrap();
    assert_eq!(
        index.search("search", 10),
        index.search_with_scratch("search", 10, &mut scratch)
    );
}
