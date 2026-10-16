#![allow(
    clippy::single_range_in_vec_init,
    reason = "each element is one chunk's byte range"
)]

use super::*;
use crate::{Collection, IvfBuildOptions, IvfError};

fn index_document(
    data: &mut CollectionData,
    text: String,
    ranges: Vec<std::ops::Range<usize>>,
    embeddings: Vec<Vec<f32>>,
) -> Result<DocId, CollectionError> {
    assert_eq!(ranges.len(), embeddings.len());
    let document = PreparedDocument {
        text,
        chunks: ranges
            .into_iter()
            .zip(embeddings)
            .map(|(byte_range, embedding)| PreparedChunk {
                byte_range,
                embedding,
            })
            .collect(),
    };
    Ok(data.insert_batch([document], true, |_| Ok(()))?.start as DocId)
}

fn ivf_mode(probes: usize) -> VectorSearchMode {
    VectorSearchMode::Ivf {
        build: IvfBuildOptions {
            clusters: 2,
            training_samples: 16,
            iterations: 8,
            seed: 42,
        },
        probes,
    }
}

fn append(data: &mut CollectionData, text: &str, vector: [f32; 2]) {
    index_document(
        data,
        text.into(),
        vec![0..text.len()],
        vec![vector.to_vec()],
    )
    .unwrap();
}

#[test]
fn chunks_share_parent_ids_but_have_distinct_index_rows() {
    let mut data = CollectionData::new(2).unwrap();
    let first = index_document(
        &mut data,
        "rust engine".into(),
        vec![0..5, 5..11],
        vec![vec![3.0, 4.0], vec![0.0, 1.0]],
    )
    .unwrap();
    let second =
        index_document(&mut data, "pasta".into(), vec![0..5], vec![vec![-1.0, 0.0]]).unwrap();
    assert_eq!((first, second), (0, 1));
    assert_eq!(data.documents.len(), 2);
    assert_eq!(data.chunks.len(), 3);
    assert_eq!(data.vectors.vector(0), Some(&[0.6, 0.8][..]));
    assert_eq!(data.bm25.search("engine", 1)[0].doc_id, 1);
    let hits = data
        .search("engine", &[0.0, 1.0], HybridSearchOptions::default())
        .unwrap();
    assert_eq!(hits[0].chunk_id, ChunkId(1));
    assert_eq!(hits[0].doc_id, first);
    assert_eq!(
        &data.documents[0][data.chunks[1].byte_range.clone()],
        "engine"
    );
    assert_eq!(hits.iter().filter(|hit| hit.doc_id == first).count(), 2);
}

#[test]
fn invalid_later_chunk_does_not_publish_earlier_chunks_or_consume_ids() {
    let mut data = CollectionData::new(2).unwrap();
    index_document(&mut data, "rust".into(), vec![0..4], vec![vec![1.0, 0.0]]).unwrap();
    let before = data
        .search("rust", &[1.0, 0.0], HybridSearchOptions::default())
        .unwrap();
    for invalid in [
        vec![1.0],
        vec![0.0, 0.0],
        vec![f32::NAN, 1.0],
        vec![1.0, f32::INFINITY],
    ] {
        let result = index_document(
            &mut data,
            "engine search".into(),
            vec![0..7, 7..13],
            vec![vec![0.0, 1.0], invalid],
        );
        assert!(matches!(result, Err(CollectionError::Vector(_))));
        assert_eq!(data.documents, ["rust"]);
        assert_eq!(data.chunks.len(), 1);
        assert_eq!(data.bm25.len(), 1);
        assert_eq!(data.vectors.len(), 1);
        assert!(data.bm25.postings("engine").is_empty());
        assert_eq!(
            data.search("rust", &[1.0, 0.0], HybridSearchOptions::default())
                .unwrap(),
            before
        );
    }
    assert_eq!(
        index_document(&mut data, "engine".into(), vec![0..6], vec![vec![0.0, 1.0]]).unwrap(),
        1
    );
    assert_eq!(data.chunks[1].doc_id, 1);
}

#[test]
fn ivf_buffer_is_immediately_searchable_and_vector_ranks_are_merged_before_fusion() {
    let mut data = CollectionData::new(2).unwrap();
    let mut exact = CollectionData::new(2).unwrap();
    for (text, vector) in [
        ("base shared", [0.8, 0.6]),
        ("older shared", [0.6, 0.8]),
        ("other shared", [0.0, 1.0]),
    ] {
        append(&mut data, text, vector);
        append(&mut exact, text, vector);
    }
    data.vectors.set_mode(ivf_mode(usize::MAX)).unwrap();
    assert_eq!(data.vectors.buffered_vector_count(), 0);
    for (text, vector) in [("fresh shared", [1.0, 0.0]), ("tied shared", [0.8, 0.6])] {
        append(&mut data, text, vector);
        append(&mut exact, text, vector);
    }
    assert_eq!(data.vectors.buffered_vector_count(), 2);
    assert!(
        matches!(&data.vectors, CollectionVectors::Ivf { index, buffer, .. } if index.len() == 3 && buffer.len() == 2)
    );
    data.vectors.set_mode(ivf_mode(1)).unwrap();
    let fresh = data
        .search(
            "absent",
            &[1.0, 0.0],
            HybridSearchOptions {
                candidate_depth: 1,
                limit: 1,
            },
        )
        .unwrap();
    assert_eq!(fresh[0].chunk_id, ChunkId(3));
    assert_eq!(fresh[0].vector_rank, Some(1));
    data.vectors.set_mode(ivf_mode(usize::MAX)).unwrap();
    let hits = data
        .search(
            "absent",
            &[1.0, 0.0],
            HybridSearchOptions {
                candidate_depth: 3,
                limit: 3,
            },
        )
        .unwrap();
    assert_eq!(
        hits.iter().map(|h| h.chunk_id.0).collect::<Vec<_>>(),
        [3, 0, 4]
    );
    assert_eq!(
        hits.iter()
            .map(|h| h.vector_rank.unwrap())
            .collect::<Vec<_>>(),
        [1, 2, 3]
    );
    for query in ["shared", "base", "fresh", "absent", ""] {
        for candidate_depth in [0, 1, 2, 5, usize::MAX] {
            for limit in [0, 1, 3, 20] {
                let options = HybridSearchOptions {
                    candidate_depth,
                    limit,
                };
                assert_eq!(
                    data.search(query, &[1.0, 0.0], options).unwrap(),
                    exact.search(query, &[1.0, 0.0], options).unwrap()
                );
            }
        }
    }
}

#[test]
fn rebuilds_and_mode_switches_preserve_ids_vectors_and_all_probe_results() {
    let mut data = CollectionData::new(2).unwrap();
    append(&mut data, "base", [0.8, 0.6]);
    data.vectors.set_mode(ivf_mode(usize::MAX)).unwrap();
    index_document(
        &mut data,
        "fresh chunks".into(),
        vec![0..6, 6..12],
        vec![vec![1.0, 0.0], vec![-1.0, 0.0]],
    )
    .unwrap();
    let options = HybridSearchOptions::default();
    let expected = data.search("fresh", &[1.0, 0.0], options).unwrap();
    let vectors: Vec<_> = (0..3)
        .map(|id| data.vectors.vector(id).unwrap().to_vec())
        .collect();
    data.vectors.rebuild().unwrap();
    assert_eq!(data.vectors.buffered_vector_count(), 0);
    assert!(
        matches!(&data.vectors, CollectionVectors::Ivf { index, buffer, .. } if index.len() == 3 && buffer.is_empty())
    );
    for _ in 0..2 {
        assert_eq!(
            data.search("fresh", &[1.0, 0.0], options).unwrap(),
            expected
        );
        for (id, vector) in vectors.iter().enumerate() {
            assert_eq!(data.vectors.vector(id as u32).unwrap(), vector);
        }
        data.vectors.rebuild().unwrap();
    }
    append(&mut data, "another fresh", [0.0, 1.0]);
    let expected = data.search("fresh", &[1.0, 0.0], options).unwrap();
    data.vectors.set_mode(VectorSearchMode::Exact).unwrap();
    assert_eq!(data.vectors.mode(), VectorSearchMode::Exact);
    assert_eq!(data.vectors.buffered_vector_count(), 0);
    assert_eq!(
        data.search("fresh", &[1.0, 0.0], options).unwrap(),
        expected
    );
    data.vectors.rebuild().unwrap(); // Exact mode is a no-op.
    data.vectors.set_mode(ivf_mode(usize::MAX)).unwrap();
    assert_eq!(
        data.search("fresh", &[1.0, 0.0], options).unwrap(),
        expected
    );
    assert_eq!(data.chunks[1].doc_id, 1);
    assert_eq!(data.chunks[2].doc_id, 1);
    assert_eq!(data.documents.len(), 3);
}

#[test]
fn probe_changes_preserve_buffer_and_invalid_changes_leave_search_usable() {
    let mut data = CollectionData::new(2).unwrap();
    append(&mut data, "base", [0.0, 1.0]);
    data.vectors.set_mode(ivf_mode(1)).unwrap();
    append(&mut data, "fresh", [1.0, 0.0]);
    let base_pointer = data.vectors.vector(0).unwrap().as_ptr();
    data.vectors.set_mode(ivf_mode(2)).unwrap();
    assert_eq!(data.vectors.vector(0).unwrap().as_ptr(), base_pointer);
    assert_eq!(data.vectors.buffered_vector_count(), 1);
    let expected = data
        .search("fresh", &[1.0, 0.0], HybridSearchOptions::default())
        .unwrap();
    assert_eq!(
        data.vectors.set_mode(ivf_mode(0)),
        Err(IvfError::ZeroProbes)
    );
    let invalid = VectorSearchMode::Ivf {
        build: IvfBuildOptions {
            clusters: 0,
            ..IvfBuildOptions::default()
        },
        probes: 1,
    };
    assert_eq!(
        data.vectors.set_mode(invalid),
        Err(IvfError::InvalidBuildOptions)
    );
    assert_eq!(data.vectors.mode(), ivf_mode(2));
    assert_eq!(data.vectors.buffered_vector_count(), 1);
    assert_eq!(
        data.search("fresh", &[1.0, 0.0], HybridSearchOptions::default())
            .unwrap(),
        expected
    );
    let result = index_document(
        &mut data,
        "broken chunks".into(),
        vec![0..7, 7..13],
        vec![vec![1.0, 0.0], vec![0.0, 0.0]],
    );
    assert!(matches!(
        result,
        Err(CollectionError::Vector(VectorError::ZeroVector))
    ));
    assert_eq!(data.documents.len(), 2);
    assert_eq!(data.chunks.len(), 2);
    assert_eq!(data.vectors.len(), 2);
    assert_eq!(data.vectors.buffered_vector_count(), 1);
    assert!(data.bm25.postings("broken").is_empty());
    // Changing training settings builds from both stores and folds in the buffer.
    let mode = VectorSearchMode::Ivf {
        build: IvfBuildOptions {
            clusters: 1,
            ..IvfBuildOptions::default()
        },
        probes: 1,
    };
    data.vectors.set_mode(mode).unwrap();
    assert_eq!(data.vectors.buffered_vector_count(), 0);
    assert_eq!(
        data.search("fresh", &[1.0, 0.0], HybridSearchOptions::default())
            .unwrap(),
        expected
    );
}

#[test]
fn empty_ivf_collects_new_chunks_until_rebuild_and_validates_queries_once() {
    let mut data = CollectionData::new(2).unwrap();
    data.vectors.set_mode(ivf_mode(1)).unwrap();
    for limit in [0, 10] {
        assert!(data.vectors.search(&[1.0, 0.0], limit).unwrap().is_empty());
        assert_eq!(
            data.vectors.search(&[0.0, 0.0], limit),
            Err(VectorError::ZeroVector)
        );
        assert!(matches!(
            data.vectors.search(&[1.0], limit),
            Err(VectorError::DimensionMismatch { .. })
        ));
    }
    append(&mut data, "fresh", [1.0, 0.0]);
    assert_eq!(data.vectors.buffered_vector_count(), 1);
    assert_eq!(data.vectors.search(&[1.0, 0.0], 10).unwrap()[0].doc_id, 0);
    data.vectors.rebuild().unwrap();
    assert_eq!(data.vectors.buffered_vector_count(), 0);
    append(&mut data, "second", [-1.0, 0.0]);
    assert_eq!(data.vectors.search(&[-1.0, 0.0], 10).unwrap()[0].doc_id, 1);
}

#[test]
fn collection_supports_shared_readers() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Collection>();
}
