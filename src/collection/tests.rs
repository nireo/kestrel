#![allow(
    clippy::single_range_in_vec_init,
    reason = "each element is one chunk's byte range"
)]

use super::*;

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
