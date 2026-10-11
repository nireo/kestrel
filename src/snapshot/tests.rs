use super::*;
use crate::{HybridSearchOptions, hybrid_search};

const CONFIG: EmbeddingConfig = EmbeddingConfig {
    model_id: "test-model",
    artifact_repo: "test-repo",
    artifact_file: "model.onnx",
    dimensions: 3,
    max_tokens: 512,
    query_instruction: "query: ",
};

fn indexes() -> (InvertedIndex, VectorIndex) {
    let mut bm25 = InvertedIndex::new();
    let mut vectors = VectorIndex::new(CONFIG.dimensions).unwrap();
    for (text, vector) in [
        ("Rust rust search engine", [1.0, 2.0, 7.0]),
        ("Pasta with tomato sauce", [-3.0, 4.0, 0.1]),
        ("Helsinki: yö, café, 世界!", [0.0, -2.0, 3.0]),
    ] {
        let doc_id = bm25.add_document(text).unwrap();
        vectors.insert(doc_id, &vector).unwrap();
    }
    (bm25, vectors)
}

fn encoded(tokenizer: u32, config: &EmbeddingConfig, documents: &[(&str, &[f32])]) -> Vec<u8> {
    let snapshot = Snapshot {
        tokenizer_version: tokenizer,
        embedding: StoredEmbeddingConfig::from(config),
        documents: documents
            .iter()
            .map(|&(text, embedding)| DocumentRef { text, embedding })
            .collect(),
    };
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    postcard::to_io(&snapshot, &mut bytes).unwrap();
    bytes
}

#[test]
fn round_trip_preserves_ids_float_bits_all_searches_and_next_id() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    let (bm25, vectors) = indexes();
    save(&path, &bm25, &vectors, &CONFIG).unwrap();
    let first_bytes = fs::read(&path).unwrap();
    let (mut restored_bm25, mut restored_vectors) = load(&path, &CONFIG).unwrap();
    assert_eq!(restored_bm25.len(), bm25.len());
    assert_eq!(restored_vectors.len(), vectors.len());
    assert_eq!(restored_vectors.dimensions(), vectors.dimensions());
    for doc_id in 0..bm25.len() as DocId {
        assert_eq!(restored_bm25.document(doc_id), bm25.document(doc_id));
        assert_eq!(
            restored_bm25.document_length(doc_id),
            bm25.document_length(doc_id)
        );
        let bits = |vector: &[f32]| vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(restored_vectors.vector(doc_id).unwrap()),
            bits(vectors.vector(doc_id).unwrap())
        );
    }
    assert_eq!(
        restored_bm25.average_document_length(),
        bm25.average_document_length()
    );
    for query in [
        "rust engine",
        "RUST rust",
        "pasta sauce",
        "YÖ 世界",
        "unknown",
        "",
    ] {
        assert_eq!(restored_bm25.search(query, 10), bm25.search(query, 10));
        for vector in [[1.0, 0.0, 0.0], [-2.0, 3.0, 1.0]] {
            assert_eq!(
                restored_vectors.search(&vector, 10).unwrap(),
                vectors.search(&vector, 10).unwrap()
            );
            let options = HybridSearchOptions::default();
            assert_eq!(
                hybrid_search(&restored_bm25, &restored_vectors, query, &vector, options).unwrap(),
                hybrid_search(&bm25, &vectors, query, &vector, options).unwrap()
            );
        }
    }
    save(&path, &restored_bm25, &restored_vectors, &CONFIG).unwrap();
    assert_eq!(fs::read(&path).unwrap(), first_bytes);
    let next = restored_bm25.add_document("new document").unwrap();
    assert_eq!(next, 3);
    restored_vectors.insert(next, &[1.0, 0.0, 0.0]).unwrap();
    save(&path, &restored_bm25, &restored_vectors, &CONFIG).unwrap();
    let (again, _) = load(&path, &CONFIG).unwrap();
    assert_eq!(again.len(), 4);
    assert_eq!(again.document(next), Some("new document"));
}

#[test]
fn empty_collection_round_trips() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty.kestrel");
    save(
        &path,
        &InvertedIndex::new(),
        &VectorIndex::new(CONFIG.dimensions).unwrap(),
        &CONFIG,
    )
    .unwrap();
    let (bm25, vectors) = load(&path, &CONFIG).unwrap();
    assert!(bm25.is_empty());
    assert!(vectors.is_empty());
    assert_eq!(vectors.dimensions(), CONFIG.dimensions);
}

#[test]
fn fixed_v1_fixture_remains_readable() {
    let config = EmbeddingConfig {
        model_id: "m",
        artifact_repo: "r",
        artifact_file: "f",
        dimensions: 1,
        max_tokens: 8,
        query_instruction: "",
    };
    // Header; tokenizer; model/repo/file; dimensions/max_tokens/instruction;
    // document count; text; vector length; little-endian f32 value 1.0.
    let bytes = b"KESTREL\0\x01\0\0\0\x01\x01m\x01r\x01f\x01\x08\0\x01\x01x\x01\0\0\x80\x3f";
    let (bm25, vectors) = decode(bytes, &config).unwrap();
    assert_eq!(bm25.document(0), Some("x"));
    assert_eq!(vectors.vector(0), Some(&[1.0][..]));
}

#[test]
fn rejects_truncation_wrong_magic_unknown_format_and_trailing_bytes() {
    let bytes = encoded(TOKENIZER_VERSION, &CONFIG, &[("rust", &[1.0, 0.0, 0.0])]);
    for end in 0..bytes.len() {
        assert!(decode(&bytes[..end], &CONFIG).is_err(), "prefix {end}");
    }
    let mut wrong_magic = bytes.clone();
    wrong_magic[0] ^= 1;
    assert!(matches!(
        decode(&wrong_magic, &CONFIG),
        Err(SnapshotError::InvalidMagic)
    ));
    let mut wrong_version = bytes.clone();
    wrong_version[MAGIC.len()..HEADER_SIZE].copy_from_slice(&99_u32.to_le_bytes());
    assert!(matches!(
        decode(&wrong_version, &CONFIG),
        Err(SnapshotError::UnsupportedVersion { found: 99 })
    ));
    let mut trailing = bytes;
    trailing.push(0);
    assert!(matches!(
        decode(&trailing, &CONFIG),
        Err(SnapshotError::InvalidData("trailing bytes"))
    ));
}

#[test]
fn rejects_incompatible_tokenizer_and_every_embedding_setting() {
    assert!(matches!(
        decode(&encoded(99, &CONFIG, &[]), &CONFIG),
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
        assert!(matches!(
            decode(&encoded(TOKENIZER_VERSION, &config, &[]), &CONFIG),
            Err(SnapshotError::IncompatibleEmbeddingConfig)
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
        // Include a valid first record: failure must not return a partial index.
        let bytes = encoded(
            TOKENIZER_VERSION,
            &CONFIG,
            &[("first", &[1.0, 0.0, 0.0]), ("second", &vector)],
        );
        match decode(&bytes, &CONFIG) {
            Err(SnapshotError::InvalidVector { doc_id, source }) => {
                assert_eq!(doc_id, 1);
                assert_eq!(source, expected);
            }
            other => panic!("expected invalid vector, got {other:?}"),
        }
    }
    assert!(matches!(
        decode(
            &encoded(TOKENIZER_VERSION, &CONFIG, &[(" \n", &[1.0, 0.0, 0.0])]),
            &CONFIG
        ),
        Err(SnapshotError::InvalidData("blank document"))
    ));
}

#[test]
fn failed_write_preserves_previous_snapshot_and_removes_temporary_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    let (bm25, vectors) = indexes();
    save(&path, &bm25, &vectors, &CONFIG).unwrap();
    let before = fs::read(&path).unwrap();
    let result = write_atomic(&path, |writer| {
        writer.write_all(b"partially written replacement")?;
        Err(io::Error::other("injected write failure").into())
    });
    assert!(matches!(result, Err(SnapshotError::Io(_))));
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    assert_eq!(load(&path, &CONFIG).unwrap().0.len(), 3);
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
    let (bm25, vectors) = indexes();
    save(&path, &bm25, &vectors, &CONFIG).unwrap();
    let before = fs::read(&path).unwrap();
    let mut wrong_ids = VectorIndex::new(CONFIG.dimensions).unwrap();
    for doc_id in 1..=3 {
        wrong_ids.insert(doc_id, &[1.0, 0.0, 0.0]).unwrap();
    }
    assert!(matches!(
        save(&path, &bm25, &wrong_ids, &CONFIG),
        Err(SnapshotError::InvalidData("document has no vector"))
    ));
    let wrong_dimensions = VectorIndex::new(2).unwrap();
    assert!(save(&path, &bm25, &wrong_dimensions, &CONFIG).is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}
