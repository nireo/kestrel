use kestrel::{InvertedIndex, SearchHit, VectorError, VectorIndex};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 2e-7,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn hand_calculated_directions_have_expected_cosine_scores() {
    let mut index = VectorIndex::new(2).unwrap();
    for (doc_id, vector) in [
        (10, [1.0, 0.0]),  // Same direction as query: 1.
        (20, [3.0, 4.0]),  // Unit vector is [0.6, 0.8]: 0.6.
        (30, [0.0, 1.0]),  // Perpendicular: 0.
        (40, [-1.0, 0.0]), // Opposite direction: -1.
    ] {
        index.insert(doc_id, &vector).unwrap();
    }
    let hits = index.search(&[1.0, 0.0], 4).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.doc_id).collect::<Vec<_>>(),
        [10, 20, 30, 40]
    );
    for (hit, expected) in hits.iter().zip([1.0, 0.6, 0.0, -1.0]) {
        assert_close(hit.score, expected);
    }
}

#[test]
fn insertion_stores_a_unit_vector_without_changing_the_input() {
    let mut index = VectorIndex::new(2).unwrap();
    let input = [3.0, 4.0];
    index.insert(7, &input).unwrap();
    assert_eq!(input, [3.0, 4.0]);
    let stored = index.vector(7).unwrap();
    assert_close(f64::from(stored[0]), 0.6);
    assert_close(f64::from(stored[1]), 0.8);
    assert_eq!(index.dimensions(), 2);
    assert_eq!(index.len(), 1);
    assert!(!index.is_empty());
    assert!(index.vector(8).is_none());
}

#[test]
fn positive_scaling_of_documents_and_queries_does_not_change_ranking() {
    let mut index = VectorIndex::new(2).unwrap();
    index.insert(9, &[3.0, 4.0]).unwrap();
    index.insert(2, &[30.0, 40.0]).unwrap();
    index.insert(5, &[0.0, 1.0]).unwrap();
    let original = index.search(&[1.0, 0.0], 3).unwrap();
    assert_eq!(original, index.search(&[100.0, 0.0], 3).unwrap());
    assert_eq!(
        original.iter().map(|hit| hit.doc_id).collect::<Vec<_>>(),
        [2, 9, 5]
    );
    assert_eq!(original[0].score, original[1].score);
}

#[test]
fn top_k_keeps_highest_scores_and_lowest_ids_on_ties() {
    let mut index = VectorIndex::new(2).unwrap();
    // Insert the best candidates late, in an order different from their IDs.
    index.insert(50, &[-1.0, 0.0]).unwrap();
    index.insert(40, &[0.0, 1.0]).unwrap();
    for doc_id in [30, 10, 20] {
        index.insert(doc_id, &[1.0, 0.0]).unwrap();
    }
    assert_eq!(
        index.search(&[1.0, 0.0], 2).unwrap(),
        [
            SearchHit {
                doc_id: 10,
                score: 1.0
            },
            SearchHit {
                doc_id: 20,
                score: 1.0
            },
        ]
    );
    assert_eq!(index.search(&[1.0, 0.0], usize::MAX).unwrap().len(), 5);
    assert!(index.search(&[1.0, 0.0], 0).unwrap().is_empty());
}

#[test]
fn empty_index_has_no_matches() {
    let index = VectorIndex::new(2).unwrap();
    assert!(index.is_empty());
    assert!(index.search(&[1.0, 0.0], 10).unwrap().is_empty());
}

#[test]
fn zero_dimensions_and_wrong_vector_dimensions_are_rejected() {
    assert_eq!(
        VectorIndex::new(0).unwrap_err(),
        VectorError::ZeroDimensions
    );
    let mut index = VectorIndex::new(2).unwrap();
    for input in [&[][..], &[1.0][..], &[1.0, 2.0, 3.0][..]] {
        let expected = VectorError::DimensionMismatch {
            expected: 2,
            actual: input.len(),
        };
        assert_eq!(index.insert(0, input), Err(expected));
        assert_eq!(index.search(input, 10), Err(expected));
    }
    assert!(index.is_empty());
}

#[test]
fn zero_and_non_finite_vectors_are_rejected_for_insert_and_search() {
    let mut index = VectorIndex::new(2).unwrap();
    for input in [[0.0, 0.0], [-0.0, 0.0]] {
        assert_eq!(index.insert(0, &input), Err(VectorError::ZeroVector));
        // Query validation also applies to empty indexes and zero limits.
        assert_eq!(index.search(&input, 0), Err(VectorError::ZeroVector));
        assert_eq!(index.search(&input, 10), Err(VectorError::ZeroVector));
    }
    for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for (input, position) in [([non_finite, 1.0], 0), ([1.0, non_finite], 1)] {
            let expected = VectorError::NonFiniteComponent { index: position };
            assert_eq!(index.insert(0, &input), Err(expected));
            assert_eq!(index.search(&input, 10), Err(expected));
        }
    }
    assert!(index.is_empty());
    assert!(index.vector(0).is_none());
    // Failed insertions did not reserve the ID or leave a partial vector.
    index.insert(0, &[1.0, 0.0]).unwrap();
    assert_eq!(index.search(&[1.0, 0.0], 1).unwrap()[0].score, 1.0);
}

#[test]
fn duplicate_ids_leave_existing_vectors_and_results_unchanged() {
    let mut index = VectorIndex::new(2).unwrap();
    index.insert(7, &[1.0, 0.0]).unwrap();
    assert_eq!(
        index.insert(7, &[0.0, 1.0]),
        Err(VectorError::DuplicateDocument { doc_id: 7 })
    );
    assert_eq!(index.len(), 1);
    assert_eq!(index.vector(7), Some(&[1.0, 0.0][..]));
    assert_eq!(index.search(&[1.0, 0.0], 1).unwrap()[0].score, 1.0);
}

#[test]
fn normalization_handles_large_and_subnormal_finite_components() {
    let tiny = f32::from_bits(1);
    let mut index = VectorIndex::new(2).unwrap();
    index.insert(0, &[f32::MAX, f32::MAX]).unwrap();
    index.insert(1, &[tiny, tiny]).unwrap();
    for query in [[1.0, 1.0], [f32::MAX, f32::MAX], [tiny, tiny]] {
        let hits = index.search(&query, 2).unwrap();
        for hit in hits {
            assert_close(hit.score, 1.0);
            assert!((-1.0..=1.0).contains(&hit.score));
        }
    }
}

#[test]
fn vector_hits_resolve_to_the_same_documents_as_bm25_hits() {
    let mut documents = InvertedIndex::new();
    let rust = documents.add_document("rust search").unwrap();
    let engine = documents.add_document("engine search").unwrap();
    let mut vectors = VectorIndex::new(2).unwrap();
    vectors.insert(engine, &[0.0, 1.0]).unwrap();
    vectors.insert(rust, &[1.0, 0.0]).unwrap();
    let vector_hit = vectors.search(&[1.0, 0.0], 1).unwrap()[0];
    let lexical_hit = documents.search("rust", 1)[0];
    assert_eq!(vector_hit.doc_id, lexical_hit.doc_id);
    assert_eq!(documents.document(vector_hit.doc_id), Some("rust search"));
}

// Reference computes cosine directly from the original unnormalized inputs,
// then sorts every score rather than using the production normalization/heap.
fn direct_cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    let norm_a = a.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt();
    dot / (norm_a * norm_b)
}

#[test]
fn exact_top_k_agrees_with_direct_cosine_and_a_full_sort() {
    let documents: Vec<(u32, Vec<f32>)> = (0..100)
        .map(|i| {
            let vector = (0..5)
                .map(|j| ((i * 17 + j * j * 13 + i * j) % 97) as f32 - 48.0)
                .collect();
            (1000 - i * 7, vector)
        })
        .collect();
    let mut index = VectorIndex::new(5).unwrap();
    for (doc_id, vector) in &documents {
        index.insert(*doc_id, vector).unwrap();
    }
    for query in [[1.0, 0.0, -1.0, 2.0, 3.0], [-4.0, 3.0, 2.0, 1.0, 0.0]] {
        let mut expected: Vec<_> = documents
            .iter()
            .map(|(doc_id, vector)| SearchHit {
                doc_id: *doc_id,
                score: direct_cosine(vector, &query),
            })
            .collect();
        expected.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));
        for limit in [0, 1, 3, 10, 100, usize::MAX] {
            let actual = index.search(&query, limit).unwrap();
            assert_eq!(actual.len(), limit.min(expected.len()));
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.doc_id, expected.doc_id);
                assert_close(actual.score, expected.score);
            }
        }
    }
}
