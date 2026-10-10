use kestrel::{
    HybridSearchOptions, InvertedIndex, SearchHit, VectorError, VectorIndex, hybrid_search,
    reciprocal_rank_fusion,
};

fn ranked(ids: &[u32]) -> Vec<SearchHit> {
    ids.iter()
        .enumerate()
        .map(|(position, &doc_id)| SearchHit {
            doc_id,
            score: (ids.len() - position) as f64,
        })
        .collect()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn overlapping_lists_sum_one_based_rank_contributions() {
    let hits = reciprocal_rank_fusion(&ranked(&[10, 20]), &ranked(&[20, 30]), 10);
    assert_eq!(
        hits.iter().map(|hit| hit.doc_id).collect::<Vec<_>>(),
        [20, 10, 30]
    );
    assert_close(hits[0].score, 1.0 / 62.0 + 1.0 / 61.0);
    assert_eq!((hits[0].bm25_rank, hits[0].vector_rank), (Some(2), Some(1)));
    assert_close(hits[1].score, 1.0 / 61.0);
    assert_eq!((hits[1].bm25_rank, hits[1].vector_rank), (Some(1), None));
    assert_close(hits[2].score, 1.0 / 62.0);
    assert_eq!((hits[2].bm25_rank, hits[2].vector_rank), (None, Some(2)));
}

#[test]
fn absent_sources_contribute_zero_and_empty_inputs_are_safe() {
    assert!(reciprocal_rank_fusion(&[], &[], 10).is_empty());
    for (bm25, vectors) in [(ranked(&[9, 2]), vec![]), (vec![], ranked(&[9, 2]))] {
        let hits = reciprocal_rank_fusion(&bm25, &vectors, usize::MAX);
        assert_eq!(
            hits.iter().map(|hit| hit.doc_id).collect::<Vec<_>>(),
            [9, 2]
        );
        assert_close(hits[0].score, 1.0 / 61.0);
        assert_close(hits[1].score, 1.0 / 62.0);
        assert_eq!(hits[0].bm25_rank.is_some(), !bm25.is_empty());
        assert_eq!(hits[0].vector_rank.is_some(), !vectors.is_empty());
    }
}

#[test]
fn fusion_uses_input_order_and_ignores_raw_score_scales() {
    let bm25 = ranked(&[7, 8, 9]);
    let vectors = ranked(&[9, 7, 10]);
    let expected = reciprocal_rank_fusion(&bm25, &vectors, 10);
    let bm25: Vec<_> = bm25
        .into_iter()
        .map(|hit| SearchHit {
            score: -1000.0,
            ..hit
        })
        .collect();
    let vectors: Vec<_> = vectors
        .into_iter()
        .map(|hit| SearchHit {
            score: f64::NAN,
            ..hit
        })
        .collect();
    assert_eq!(reciprocal_rank_fusion(&bm25, &vectors, 10), expected);
}

#[test]
fn duplicate_ids_count_once_per_source_at_their_first_position() {
    let hits = reciprocal_rank_fusion(&ranked(&[10, 10, 20]), &ranked(&[20, 20, 10]), 10);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, 10);
    assert_eq!(hits[1].doc_id, 20);
    assert_eq!((hits[0].bm25_rank, hits[0].vector_rank), (Some(1), Some(3)));
    assert_eq!((hits[1].bm25_rank, hits[1].vector_rank), (Some(3), Some(1)));
    for hit in hits {
        assert_close(hit.score, 1.0 / 61.0 + 1.0 / 63.0);
    }
}

#[test]
fn tied_fused_scores_prefer_lower_document_ids() {
    let hits = reciprocal_rank_fusion(&ranked(&[9, 2]), &ranked(&[2, 9]), 1);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, 2);
    let disjoint = reciprocal_rank_fusion(&ranked(&[9]), &ranked(&[2]), 2);
    assert_eq!(
        disjoint.iter().map(|hit| hit.doc_id).collect::<Vec<_>>(),
        [2, 9]
    );
}

#[test]
fn top_k_matches_a_full_fusion_and_sort() {
    let bm25_ids: Vec<_> = (0..100).map(|i| (i * 37) % 100).collect();
    let vector_ids: Vec<_> = (0..100).map(|i| 50 + (i * 13) % 100).collect();
    // Independent reference: look up each document's position in each list.
    let mut expected: Vec<_> = (0..150)
        .map(|doc_id| {
            let score: f64 = [&bm25_ids, &vector_ids]
                .iter()
                .filter_map(|ids| ids.iter().position(|&id| id == doc_id))
                .map(|position| 1.0 / (61.0 + position as f64))
                .sum();
            SearchHit { doc_id, score }
        })
        .collect();
    expected.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));
    for limit in [0, 1, 3, 10, 150, usize::MAX] {
        let actual = reciprocal_rank_fusion(&ranked(&bm25_ids), &ranked(&vector_ids), limit);
        assert_eq!(actual.len(), limit.min(expected.len()));
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.doc_id, expected.doc_id);
            assert_close(actual.score, expected.score);
        }
    }
}

fn indexes() -> (InvertedIndex, VectorIndex) {
    let mut bm25 = InvertedIndex::new();
    let mut vectors = VectorIndex::new(2).unwrap();
    // Lexical order for "rust": 0, 1. Vector order for [1, 0]: 1, 2.
    bm25.add_document("rust").unwrap();
    bm25.add_document("rust engine").unwrap();
    bm25.add_document("semantic only").unwrap();
    // Document 0 intentionally has no vector.
    vectors.insert(1, &[1.0, 0.0]).unwrap();
    vectors.insert(2, &[0.0, 1.0]).unwrap();
    (bm25, vectors)
}

#[test]
fn hybrid_search_retrieves_then_fuses_and_preserves_source_ranks() {
    let (bm25, vectors) = indexes();
    let options = HybridSearchOptions {
        candidate_depth: 2,
        limit: 10,
    };
    let hits = hybrid_search(&bm25, &vectors, "rust", &[1.0, 0.0], options).unwrap();
    assert_eq!(
        hits,
        reciprocal_rank_fusion(
            &bm25.search("rust", 2),
            &vectors.search(&[1.0, 0.0], 2).unwrap(),
            10
        )
    );
    assert_eq!(
        hits.iter().map(|hit| hit.doc_id).collect::<Vec<_>>(),
        [1, 0, 2]
    );
    assert_eq!((hits[0].bm25_rank, hits[0].vector_rank), (Some(2), Some(1)));
    assert_eq!((hits[1].bm25_rank, hits[1].vector_rank), (Some(1), None));
    assert_eq!((hits[2].bm25_rank, hits[2].vector_rank), (None, Some(2)));
}

#[test]
fn candidate_depth_is_independent_of_the_final_result_limit() {
    let (bm25, vectors) = indexes();
    // Depth 1 produces two tied single-source hits, so document 0 wins the tie.
    let shallow = hybrid_search(
        &bm25,
        &vectors,
        "rust",
        &[1.0, 0.0],
        HybridSearchOptions {
            candidate_depth: 1,
            limit: 1,
        },
    )
    .unwrap();
    assert_eq!(shallow[0].doc_id, 0);
    // Depth 2 discovers agreement on document 1, changing the top fused result.
    let deeper = hybrid_search(
        &bm25,
        &vectors,
        "rust",
        &[1.0, 0.0],
        HybridSearchOptions {
            candidate_depth: 2,
            limit: 1,
        },
    )
    .unwrap();
    assert_eq!(deeper[0].doc_id, 1);
    let union = hybrid_search(
        &bm25,
        &vectors,
        "rust",
        &[1.0, 0.0],
        HybridSearchOptions {
            candidate_depth: 1,
            limit: 10,
        },
    )
    .unwrap();
    assert_eq!(union.len(), 2);
}

#[test]
fn hybrid_search_handles_empty_sources_and_zero_limits() {
    let (bm25, vectors) = indexes();
    let empty_bm25 = InvertedIndex::new();
    let empty_vectors = VectorIndex::new(2).unwrap();
    let options = HybridSearchOptions::default();
    assert_eq!(options.candidate_depth, 50);
    assert_eq!(options.limit, 10);
    assert!(
        hybrid_search(&empty_bm25, &empty_vectors, "rust", &[1.0, 0.0], options)
            .unwrap()
            .is_empty()
    );
    let lexical_only = hybrid_search(&bm25, &empty_vectors, "rust", &[1.0, 0.0], options).unwrap();
    assert!(lexical_only.iter().all(|hit| hit.vector_rank.is_none()));
    let vector_only = hybrid_search(&bm25, &vectors, "missing", &[1.0, 0.0], options).unwrap();
    assert!(vector_only.iter().all(|hit| hit.bm25_rank.is_none()));
    for options in [
        HybridSearchOptions {
            candidate_depth: 0,
            limit: 10,
        },
        HybridSearchOptions {
            candidate_depth: 10,
            limit: 0,
        },
    ] {
        assert!(
            hybrid_search(&bm25, &vectors, "rust", &[1.0, 0.0], options)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            hybrid_search(&bm25, &vectors, "rust", &[0.0, 0.0], options),
            Err(VectorError::ZeroVector)
        );
    }
}

#[test]
fn hybrid_search_propagates_invalid_query_errors() {
    let (bm25, vectors) = indexes();
    assert_eq!(
        hybrid_search(
            &bm25,
            &vectors,
            "rust",
            &[1.0],
            HybridSearchOptions::default()
        ),
        Err(VectorError::DimensionMismatch {
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        hybrid_search(
            &bm25,
            &vectors,
            "rust",
            &[f32::NAN, 0.0],
            HybridSearchOptions::default()
        ),
        Err(VectorError::NonFiniteComponent { index: 0 })
    );
}
