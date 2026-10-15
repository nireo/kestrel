#![allow(
    clippy::single_range_in_vec_init,
    reason = "each element is one chunk's byte range"
)]

use super::*;
use crate::{HybridSearchOptions, IvfBuildOptions, VectorIndex, VectorSearchMode};

const CONFIG: EmbeddingConfig = EmbeddingConfig {
    model_id: "test-model",
    artifact_repo: "test-repo",
    artifact_file: "model.onnx",
    dimensions: 3,
    max_tokens: 512,
    query_instruction: "query: ",
};
const CHUNKING: ChunkingConfig = ChunkingConfig {
    max_tokens: 16,
    overlap_tokens: 2,
};

fn data() -> CollectionData {
    let mut data = CollectionData::new(CONFIG.dimensions).unwrap();
    for (text, ranges, vectors) in [
        (
            "Rust rust search engine",
            vec![0..10, 10..23],
            vec![[1.0, 2.0, 7.0], [0.1, 3.0, -2.0]],
        ),
        (
            "Pasta with tomato sauce",
            vec![0..23],
            vec![[-3.0, 4.0, 0.1]],
        ),
        (
            "Helsinki: yö, café, 世界!",
            vec![0.."Helsinki: yö, café, 世界!".len()],
            vec![[0.0, -2.0, 3.0]],
        ),
    ] {
        let embeddings = vectors
            .iter()
            .map(|v| data.vectors.normalize(v).unwrap())
            .collect();
        data.append_document(text.into(), ranges, embeddings)
            .unwrap();
    }
    data
}

fn snapshot() -> Snapshot<String, StoredChunk> {
    let data = data();
    Snapshot {
        tokenizer_version: TOKENIZER_VERSION,
        embedding: StoredEmbeddingConfig::from(&CONFIG),
        chunking: CHUNKING,
        documents: data.documents,
        chunks: data
            .chunks
            .iter()
            .enumerate()
            .map(|(id, chunk)| StoredChunk {
                doc_id: chunk.doc_id,
                start: chunk.byte_range.start,
                end: chunk.byte_range.end,
                embedding: data.vectors.vector(id as u32).unwrap().to_vec(),
            })
            .collect(),
    }
}

fn encoded(snapshot: &Snapshot<String, StoredChunk>) -> Vec<u8> {
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    postcard::to_io(snapshot, &mut bytes).unwrap();
    bytes
}

#[test]
fn round_trip_preserves_documents_chunks_float_bits_searches_and_next_ids() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    let original = data();
    save(&path, &original, &CONFIG, CHUNKING).unwrap();
    let before = fs::read(&path).unwrap();
    let (mut restored, chunking) = load(&path, &CONFIG).unwrap();
    assert_eq!(chunking, CHUNKING);
    assert_eq!(restored.documents, original.documents);
    assert_eq!(restored.chunks, original.chunks);
    for id in 0..original.chunks.len() as u32 {
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(restored.vectors.vector(id).unwrap()),
            bits(original.vectors.vector(id).unwrap())
        );
    }
    for query in [
        "rust engine",
        "RUST rust",
        "pasta sauce",
        "YÖ 世界",
        "unknown",
        "",
    ] {
        assert_eq!(
            restored.bm25.search(query, 10),
            original.bm25.search(query, 10)
        );
        for vector in [[1.0, 0.0, 0.0], [-2.0, 3.0, 1.0]] {
            assert_eq!(
                restored.vectors.search(&vector, 10).unwrap(),
                original.vectors.search(&vector, 10).unwrap()
            );
            assert_eq!(
                restored
                    .search(query, &vector, HybridSearchOptions::default())
                    .unwrap(),
                original
                    .search(query, &vector, HybridSearchOptions::default())
                    .unwrap()
            );
        }
    }
    save(&path, &restored, &CONFIG, chunking).unwrap();
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(
        restored
            .append_document(
                "new document".into(),
                vec![0..12],
                vec![vec![1.0, 0.0, 0.0]]
            )
            .unwrap(),
        3
    );
    assert_eq!(restored.chunks[4].doc_id, 3);
    save(&path, &restored, &CONFIG, chunking).unwrap();
    let (again, _) = load(&path, &CONFIG).unwrap();
    assert_eq!(again.documents.len(), 4);
    assert_eq!(again.chunks.len(), 5);
}

#[test]
fn empty_collection_round_trips() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty.kestrel");
    save(
        &path,
        &CollectionData::new(CONFIG.dimensions).unwrap(),
        &CONFIG,
        CHUNKING,
    )
    .unwrap();
    let (data, chunking) = load(&path, &CONFIG).unwrap();
    assert!(data.documents.is_empty());
    assert!(data.chunks.is_empty());
    assert!(data.bm25.is_empty());
    assert!(data.vectors.is_empty());
    assert_eq!(chunking, CHUNKING);
}

#[test]
fn ivf_snapshot_includes_buffer_and_preserves_float_bits_across_rebuilds() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("buffered.kestrel");
    let mut original = data();
    let mode = VectorSearchMode::Ivf {
        build: IvfBuildOptions {
            clusters: 2,
            ..IvfBuildOptions::default()
        },
        probes: usize::MAX,
    };
    original.vectors.set_mode(mode).unwrap();
    original
        .append_document(
            "new document".into(),
            vec![0..12],
            vec![vec![1.0, 0.0, 0.0]],
        )
        .unwrap();
    assert_eq!(original.vectors.buffered_vector_count(), 1);
    save(&path, &original, &CONFIG, CHUNKING).unwrap();
    let saved = fs::read(&path).unwrap();
    let (restored, _) = load(&path, &CONFIG).unwrap();
    assert_eq!(restored.vectors.mode(), VectorSearchMode::Exact);
    assert_eq!(restored.documents, original.documents);
    assert_eq!(restored.chunks, original.chunks);
    for id in 0..original.chunks.len() as u32 {
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(restored.vectors.vector(id).unwrap()),
            bits(original.vectors.vector(id).unwrap())
        );
    }
    for query in ["rust", "new document", "absent"] {
        for vector in [[1.0, 0.0, 0.0], [-2.0, 3.0, 1.0]] {
            let options = HybridSearchOptions::default();
            assert_eq!(
                restored.search(query, &vector, options).unwrap(),
                original.search(query, &vector, options).unwrap()
            );
        }
    }
    // Runtime layout changes must not change the version-2 snapshot contents.
    original.vectors.rebuild().unwrap();
    assert_eq!(original.vectors.buffered_vector_count(), 0);
    save(&path, &original, &CONFIG, CHUNKING).unwrap();
    assert_eq!(fs::read(&path).unwrap(), saved);
    original.vectors.set_mode(VectorSearchMode::Exact).unwrap();
    save(&path, &original, &CONFIG, CHUNKING).unwrap();
    assert_eq!(fs::read(&path).unwrap(), saved);
}

#[test]
fn fixed_v1_fixture_loads_as_one_chunk_per_document() {
    let config = EmbeddingConfig {
        model_id: "m",
        artifact_repo: "r",
        artifact_file: "f",
        dimensions: 1,
        max_tokens: 8,
        query_instruction: "",
    };
    let bytes = b"KESTREL\0\x01\0\0\0\x01\x01m\x01r\x01f\x01\x08\0\x01\x01x\x01\0\0\x80\x3f";
    let (data, _) = decode(bytes, &config).unwrap();
    assert_eq!(data.documents, ["x"]);
    assert_eq!(
        data.chunks,
        [crate::Chunk {
            doc_id: 0,
            byte_range: 0..1
        }]
    );
    assert_eq!(data.vectors.vector(0), Some(&[1.0][..]));
}

#[test]
fn fixed_v2_fixture_remains_readable() {
    let config = EmbeddingConfig {
        model_id: "m",
        artifact_repo: "r",
        artifact_file: "f",
        dimensions: 1,
        max_tokens: 8,
        query_instruction: "",
    };
    // Header; tokenizer; embedding config; chunking 8/0; documents ["x"];
    // chunks [{parent: 0, start: 0, end: 1, embedding: [1.0]}].
    let bytes = b"KESTREL\0\x02\0\0\0\x01\x01m\x01r\x01f\x01\x08\0\x08\0\x01\x01x\x01\0\0\x01\x01\0\0\x80\x3f";
    let (data, chunking) = decode(bytes, &config).unwrap();
    assert_eq!(data.documents, ["x"]);
    assert_eq!(
        chunking,
        ChunkingConfig {
            max_tokens: 8,
            overlap_tokens: 0
        }
    );
    assert_eq!(data.vectors.vector(0), Some(&[1.0][..]));
}

#[test]
fn rejects_truncation_wrong_magic_unknown_format_and_trailing_bytes() {
    let bytes = encoded(&snapshot());
    for end in 0..bytes.len() {
        assert!(decode(&bytes[..end], &CONFIG).is_err(), "prefix {end}");
    }
    let mut invalid = bytes.clone();
    invalid[0] ^= 1;
    assert!(matches!(
        decode(&invalid, &CONFIG),
        Err(SnapshotError::InvalidMagic)
    ));
    let mut invalid = bytes.clone();
    invalid[MAGIC.len()..HEADER_SIZE].copy_from_slice(&99_u32.to_le_bytes());
    assert!(matches!(
        decode(&invalid, &CONFIG),
        Err(SnapshotError::UnsupportedVersion { found: 99 })
    ));
    let mut invalid = bytes;
    invalid.push(0);
    assert!(matches!(
        decode(&invalid, &CONFIG),
        Err(SnapshotError::InvalidData("trailing bytes"))
    ));
}

#[test]
fn rejects_incompatible_tokenizer_embedding_and_chunking_settings() {
    let mut invalid = snapshot();
    invalid.tokenizer_version = 99;
    assert!(matches!(
        decode(&encoded(&invalid), &CONFIG),
        Err(SnapshotError::UnsupportedTokenizer { found: 99 })
    ));
    for config in [
        EmbeddingConfig {
            model_id: "other",
            ..CONFIG
        },
        EmbeddingConfig {
            artifact_repo: "other",
            ..CONFIG
        },
        EmbeddingConfig {
            artifact_file: "other",
            ..CONFIG
        },
        EmbeddingConfig {
            dimensions: 2,
            ..CONFIG
        },
        EmbeddingConfig {
            max_tokens: 256,
            ..CONFIG
        },
        EmbeddingConfig {
            query_instruction: "other",
            ..CONFIG
        },
    ] {
        let mut invalid = snapshot();
        invalid.embedding = StoredEmbeddingConfig::from(&config);
        assert!(matches!(
            decode(&encoded(&invalid), &CONFIG),
            Err(SnapshotError::IncompatibleEmbeddingConfig)
        ));
    }
    for chunking in [
        ChunkingConfig {
            max_tokens: 0,
            overlap_tokens: 0,
        },
        ChunkingConfig {
            max_tokens: 513,
            overlap_tokens: 0,
        },
        ChunkingConfig {
            max_tokens: 8,
            overlap_tokens: 8,
        },
    ] {
        let mut invalid = snapshot();
        invalid.chunking = chunking;
        assert!(matches!(
            decode(&encoded(&invalid), &CONFIG),
            Err(SnapshotError::Chunking(_))
        ));
    }
}

#[test]
fn rejects_invalid_vectors_and_blank_documents() {
    for (vector, expected) in [
        (
            vec![1.0],
            VectorError::DimensionMismatch {
                expected: 3,
                actual: 1,
            },
        ),
        (vec![0.0, 0.0, 0.0], VectorError::ZeroVector),
        (
            vec![f32::NAN, 0.0, 1.0],
            VectorError::NonFiniteComponent { index: 0 },
        ),
        (
            vec![0.0, f32::INFINITY, 1.0],
            VectorError::NonFiniteComponent { index: 1 },
        ),
        (vec![2.0, 0.0, 0.0], VectorError::NotNormalized),
    ] {
        let mut invalid = snapshot();
        invalid.chunks[1].embedding = vector;
        match decode(&encoded(&invalid), &CONFIG) {
            Err(SnapshotError::InvalidVector { chunk_id, source }) => {
                assert_eq!(chunk_id, ChunkId(1));
                assert_eq!(source, expected);
            }
            other => panic!("expected invalid vector, got {other:?}"),
        }
    }
    let mut invalid = snapshot();
    invalid.documents[0] = " \n".into();
    assert!(matches!(
        decode(&encoded(&invalid), &CONFIG),
        Err(SnapshotError::InvalidData("blank document"))
    ));
}

#[test]
fn rejects_invalid_utf8_ranges_gaps_missing_parents_and_chunk_order() {
    for case in 0..9 {
        let mut invalid = snapshot();
        match case {
            0 => invalid.chunks[0].start = 1,
            1 => invalid.chunks[0].end = usize::MAX,
            2 => invalid.chunks[0].end = 0,
            3 => invalid.chunks[1].start = invalid.chunks[0].end + 1,
            4 => invalid.chunks[1].start = 0,
            5 => invalid.chunks[1].end -= 1,
            6 => invalid.chunks[0].doc_id = 99,
            7 => invalid.chunks[3].end = invalid.documents[2].find('ö').unwrap() + 1,
            8 => {
                invalid.chunks.swap(0, 2);
            }
            _ => unreachable!(),
        }
        assert!(
            matches!(
                decode(&encoded(&invalid), &CONFIG),
                Err(SnapshotError::InvalidData(_))
            ),
            "case {case}"
        );
    }
    let mut invalid = snapshot();
    invalid.chunks.clear();
    assert!(decode(&encoded(&invalid), &CONFIG).is_err());
}

#[test]
fn failed_write_preserves_previous_snapshot_and_removes_temporary_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    save(&path, &data(), &CONFIG, CHUNKING).unwrap();
    let before = fs::read(&path).unwrap();
    let result = write_atomic(&path, |writer| {
        writer.write_all(b"partial replacement")?;
        Err(io::Error::other("injected write failure").into())
    });
    assert!(matches!(result, Err(SnapshotError::Io(_))));
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    assert_eq!(load(&path, &CONFIG).unwrap().0.documents.len(), 3);
}

#[test]
fn failed_rename_cleans_up_temporary_file() {
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("existing-directory");
    fs::create_dir(&target).unwrap();
    let result = write_atomic(&target, |writer| {
        writer.write_all(b"replacement")?;
        Ok(())
    });
    assert!(matches!(result, Err(SnapshotError::Io(_))));
    assert!(target.is_dir());
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn save_rejects_missing_vectors_and_mismatched_dimensions_before_replacement() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    let mut data = data();
    save(&path, &data, &CONFIG, CHUNKING).unwrap();
    let before = fs::read(&path).unwrap();
    let mut wrong_ids = VectorIndex::new(CONFIG.dimensions).unwrap();
    for id in 1..=data.chunks.len() as u32 {
        wrong_ids.insert(id, &[1.0, 0.0, 0.0]).unwrap();
    }
    data.vectors = crate::collection::CollectionVectors::Exact(wrong_ids);
    assert!(matches!(
        save(&path, &data, &CONFIG, CHUNKING),
        Err(SnapshotError::InvalidData("chunk has no vector"))
    ));
    data.vectors = crate::collection::CollectionVectors::Exact(VectorIndex::new(2).unwrap());
    assert!(save(&path, &data, &CONFIG, CHUNKING).is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}
