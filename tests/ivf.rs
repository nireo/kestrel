use kestrel::{
    HybridSearchOptions, InvertedIndex, IvfBuildOptions, IvfError, IvfIndex, VectorError,
    VectorIndex, hybrid_search, hybrid_search_ivf,
};

fn options(clusters: usize) -> IvfBuildOptions {
    IvfBuildOptions {
        clusters,
        training_samples: 128,
        iterations: 8,
        seed: 42,
    }
}

fn corpus() -> VectorIndex {
    let mut source = VectorIndex::new(5).unwrap();
    for i in 0..100 {
        let vector: Vec<_> = (0..5)
            .map(|j| ((i * 17 + j * j * 13 + i * j) % 97) as f32 - 48.0)
            .collect();
        source.insert(1000 - i * 7, &vector).unwrap();
    }
    source
}

#[test]
fn probing_every_cluster_matches_exact_scores_ids_and_ties() {
    let source = corpus();
    for clusters in [1, 8, 128] {
        let ivf = IvfIndex::build(&source, options(clusters)).unwrap();
        assert_eq!(ivf.len(), source.len());
        assert_eq!(ivf.dimensions(), source.dimensions());
        assert_eq!(ivf.cluster_count(), clusters.min(source.len()));
        assert_eq!(ivf.cluster_sizes().sum::<usize>(), source.len());
        for id in (0..100).map(|i| 1000 - i * 7) {
            assert_eq!(ivf.vector(id), source.vector(id));
        }
        assert!(ivf.vector(0).is_none());
        for query in [[1.0, 0.0, -1.0, 2.0, 3.0], [-4.0, 3.0, 2.0, 1.0, 0.0]] {
            for limit in [0, 1, 10, 100, usize::MAX] {
                let expected = source.search(&query, limit).unwrap();
                assert_eq!(
                    ivf.search(&query, limit, ivf.cluster_count()).unwrap(),
                    expected
                );
                assert_eq!(ivf.search(&query, limit, usize::MAX).unwrap(), expected);
            }
        }
    }
    let mut ties = VectorIndex::new(2).unwrap();
    for id in [9, 3, 7, 1] {
        ties.insert(id, &[1.0, 0.0]).unwrap();
    }
    let ivf = IvfIndex::build(&ties, options(4)).unwrap();
    assert_eq!(
        ivf.search(&[1.0, 0.0], 2, 1).unwrap(),
        ties.search(&[1.0, 0.0], 2).unwrap()
    );
}

#[test]
fn one_probe_finds_a_separated_cluster_and_more_probes_improve_recall() {
    let mut source = VectorIndex::new(2).unwrap();
    for i in 0..40 {
        source
            .insert(i * 3, &[if i < 20 { 1.0 } else { -1.0 }, i as f32 * 0.001])
            .unwrap();
    }
    let ivf = IvfIndex::build(&source, options(2)).unwrap();
    assert_eq!(ivf.cluster_sizes().collect::<Vec<_>>(), [20, 20]);
    for query in [[1.0, 0.0], [-1.0, 0.0]] {
        assert_eq!(
            ivf.search(&query, 10, 1).unwrap(),
            source.search(&query, 10).unwrap()
        );
        assert_eq!(ivf.search(&query, 40, 1).unwrap().len(), 20);
    }

    let source = corpus();
    let sampled = IvfBuildOptions {
        training_samples: 32,
        ..options(8)
    };
    let a = IvfIndex::build(&source, sampled).unwrap();
    let b = IvfIndex::build(&source, sampled).unwrap();
    for query in [[1.0, 0.0, -1.0, 2.0, 3.0], [-4.0, 3.0, 2.0, 1.0, 0.0]] {
        let exact = source.search(&query, 10).unwrap();
        let mut previous = 0;
        for probes in 1..=8 {
            let hits = a.search(&query, 10, probes).unwrap();
            assert_eq!(hits, b.search(&query, 10, probes).unwrap());
            let overlap = hits
                .iter()
                .filter(|hit| exact.iter().any(|e| e.doc_id == hit.doc_id))
                .count();
            assert!(overlap >= previous);
            assert!(
                hits.iter()
                    .all(|hit| source.search(&query, usize::MAX).unwrap().contains(hit))
            );
            previous = overlap;
        }
        assert_eq!(previous, 10);
    }
}

#[test]
fn empty_singleton_duplicate_and_antipodal_data_are_searchable() {
    let empty = VectorIndex::new(2).unwrap();
    let ivf = IvfIndex::build(&empty, options(8)).unwrap();
    assert!(ivf.is_empty());
    assert_eq!(ivf.cluster_count(), 0);
    assert!(ivf.search(&[1.0, 0.0], 10, 1).unwrap().is_empty());
    for vectors in [
        vec![[1.0, 0.0]],
        vec![[1.0, 0.0]; 8],
        vec![[1.0, 0.0], [-1.0, 0.0]],
    ] {
        let mut source = VectorIndex::new(2).unwrap();
        for (id, vector) in vectors.iter().enumerate() {
            source.insert(id as u32, vector).unwrap();
        }
        for clusters in [1, 8] {
            let ivf = IvfIndex::build(&source, options(clusters)).unwrap();
            assert_eq!(
                ivf.search(&[-1.0, 0.0], 100, 8).unwrap(),
                source.search(&[-1.0, 0.0], 100).unwrap()
            );
        }
    }
}

#[test]
fn validation_applies_to_empty_indexes_zero_limits_and_invalid_build_settings() {
    let source = VectorIndex::new(2).unwrap();
    for invalid in [
        IvfBuildOptions {
            clusters: 0,
            ..options(1)
        },
        IvfBuildOptions {
            iterations: 0,
            ..options(1)
        },
        IvfBuildOptions {
            training_samples: 0,
            ..options(1)
        },
        IvfBuildOptions {
            training_samples: 1,
            ..options(2)
        },
    ] {
        assert_eq!(
            IvfIndex::build(&source, invalid).unwrap_err(),
            IvfError::InvalidBuildOptions
        );
    }
    let ivf = IvfIndex::build(&source, options(8)).unwrap();
    assert_eq!(ivf.search(&[1.0, 0.0], 0, 0), Err(IvfError::ZeroProbes));
    for limit in [0, 10] {
        assert_eq!(
            ivf.search(&[0.0, 0.0], limit, 1),
            Err(IvfError::Vector(VectorError::ZeroVector))
        );
        assert_eq!(
            ivf.search(&[1.0], limit, 1),
            Err(IvfError::Vector(VectorError::DimensionMismatch {
                expected: 2,
                actual: 1
            }))
        );
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                ivf.search(&[bad, 1.0], limit, 1),
                Err(IvfError::Vector(VectorError::NonFiniteComponent {
                    index: 0
                }))
            );
        }
    }
}

#[test]
fn index_owns_its_data_and_handles_extreme_input_magnitudes() {
    let mut source = VectorIndex::new(2).unwrap();
    source.insert(7, &[f32::MAX, f32::MAX]).unwrap();
    source
        .insert(2, &[f32::from_bits(1), f32::from_bits(1)])
        .unwrap();
    let ivf = IvfIndex::build(&source, options(2)).unwrap();
    let expected = source.search(&[f32::MAX, f32::MAX], 10).unwrap();
    source.insert(99, &[1.0, 0.0]).unwrap();
    drop(source);
    assert_eq!(ivf.len(), 2);
    assert_eq!(ivf.search(&[f32::MAX, f32::MAX], 10, 2).unwrap(), expected);
}

#[test]
fn hybrid_with_all_probes_matches_exact_hybrid_including_zero_limits() {
    let mut bm25 = InvertedIndex::new();
    let mut vectors = VectorIndex::new(2).unwrap();
    for (text, vector) in [
        ("keyword search", [1.0, 0.0]),
        ("semantic search", [0.0, 1.0]),
        ("storage", [-1.0, 0.0]),
    ] {
        let id = bm25.add_document(text).unwrap();
        vectors.insert(id, &vector).unwrap();
    }
    let ivf = IvfIndex::build(&vectors, options(2)).unwrap();
    for limit in [0, 1, 3] {
        for candidate_depth in [0, 1, 3] {
            let options = HybridSearchOptions {
                limit,
                candidate_depth,
            };
            assert_eq!(
                hybrid_search_ivf(&bm25, &ivf, "search", &[1.0, 0.0], options, 2).unwrap(),
                hybrid_search(&bm25, &vectors, "search", &[1.0, 0.0], options).unwrap()
            );
        }
    }
}
