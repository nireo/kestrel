use super::*;
use crate::PreparedChunk;

const CONFIG: EmbeddingConfig = EmbeddingConfig {
    model_id: "test-model",
    artifact_repo: "unused",
    artifact_file: "unused",
    dimensions: 2,
    max_tokens: 512,
    query_instruction: "query: ",
};

fn document(text: &str, embedding: [f32; 2]) -> PreparedDocument {
    PreparedDocument {
        text: text.into(),
        chunks: vec![PreparedChunk {
            byte_range: 0..text.len(),
            embedding: embedding.to_vec(),
        }],
    }
}

fn record(documents: &[PreparedDocument]) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_frame(&mut bytes, &encode(documents, MAX_BATCH_BYTES).unwrap()).unwrap();
    bytes
}

fn empty_file(path: &Path) -> Vec<u8> {
    drop(PersistentCollection::create(path, CONFIG, ChunkingConfig::default()).unwrap());
    std::fs::read(path).unwrap()
}

#[test]
fn every_incomplete_final_batch_is_discarded_as_a_whole_before_new_appends() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    let mut collection =
        PersistentCollection::create(&path, CONFIG, ChunkingConfig::default()).unwrap();
    collection
        .add_document(document("base", [1.0, 0.0]))
        .unwrap();
    let prefix = std::fs::read(&path).unwrap();
    drop(collection);
    let batch = record(&[
        document("first", [0.0, 1.0]),
        document("second", [-1.0, 0.0]),
    ]);
    for cut in 0..batch.len() {
        let mut torn = prefix.clone();
        torn.extend_from_slice(&batch[..cut]);
        std::fs::write(&path, torn).unwrap();
        let mut recovered = PersistentCollection::open(&path, CONFIG).unwrap();
        assert_eq!(recovered.retrieval().len(), 1, "cut {cut}");
        assert_eq!(std::fs::read(&path).unwrap(), prefix, "cut {cut}");
        assert_eq!(
            recovered
                .add_document(document("replacement", [0.0, 1.0]))
                .unwrap(),
            1
        );
        drop(recovered);
        let reopened = PersistentCollection::open(&path, CONFIG).unwrap();
        assert_eq!(reopened.retrieval().len(), 2, "cut {cut}");
        assert_eq!(reopened.retrieval().document(1), Some("replacement"));
    }
    let mut complete = prefix;
    complete.extend_from_slice(&batch);
    std::fs::write(&path, &complete).unwrap();
    let recovered = PersistentCollection::open(&path, CONFIG).unwrap();
    assert_eq!(recovered.retrieval().len(), 3);
    assert_eq!(std::fs::read(&path).unwrap(), complete);
}

#[test]
fn complete_corrupt_records_are_errors_and_never_trigger_truncation() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    let header = empty_file(&path);
    let batch = record(&[document("first", [1.0, 0.0])]);
    for index in [0, 8, FRAME_PREFIX_BYTES, batch.len() - 1] {
        let mut corrupt = header.clone();
        corrupt.extend_from_slice(&batch);
        corrupt[header.len() + index] ^= 1;
        // A later incomplete record must not hide earlier corruption.
        corrupt.extend_from_slice(&batch[..3]);
        std::fs::write(&path, &corrupt).unwrap();
        assert!(matches!(
            PersistentCollection::open(&path, CONFIG),
            Err(CollectionError::Storage(StorageError::InvalidData(_)))
        ));
        assert_eq!(std::fs::read(&path).unwrap(), corrupt);
    }
}

#[test]
fn incomplete_or_corrupt_headers_are_rejected_without_modifying_the_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    let header = empty_file(&path);
    for cut in 0..header.len() {
        std::fs::write(&path, &header[..cut]).unwrap();
        assert!(PersistentCollection::open(&path, CONFIG).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), &header[..cut]);
    }
    for index in [0, 12, header.len() - 1] {
        let mut corrupt = header.clone();
        corrupt[index] ^= 1;
        std::fs::write(&path, &corrupt).unwrap();
        assert!(PersistentCollection::open(&path, CONFIG).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), corrupt);
    }
    let mut unsupported = header;
    unsupported[8..12].copy_from_slice(&99_u32.to_le_bytes());
    std::fs::write(&path, &unsupported).unwrap();
    assert!(matches!(
        PersistentCollection::open(&path, CONFIG),
        Err(CollectionError::Storage(StorageError::UnsupportedVersion(
            99
        )))
    ));
}

#[test]
fn checksummed_but_invalid_documents_empty_batches_and_trailing_bytes_are_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    let header = empty_file(&path);
    let mut invalid_range = document("café", [1.0, 0.0]);
    invalid_range.chunks[0].byte_range.end = 4;
    let mut missing_chunks = document("text", [1.0, 0.0]);
    missing_chunks.chunks.clear();
    for documents in [
        vec![],
        vec![document("blank", [0.0, 0.0])],
        vec![document("wrong norm", [3.0, 4.0])],
        vec![document("nan", [f32::NAN, 0.0])],
        vec![document("  ", [1.0, 0.0])],
        vec![invalid_range],
        vec![missing_chunks],
    ] {
        let mut invalid = header.clone();
        invalid.extend_from_slice(&record(&documents));
        std::fs::write(&path, &invalid).unwrap();
        assert!(PersistentCollection::open(&path, CONFIG).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), invalid);
    }
    let mut payload = encode(&vec![document("valid", [1.0, 0.0])], MAX_BATCH_BYTES).unwrap();
    payload.push(0);
    let mut invalid = header;
    write_frame(&mut invalid, &payload).unwrap();
    std::fs::write(&path, &invalid).unwrap();
    assert!(matches!(
        PersistentCollection::open(&path, CONFIG),
        Err(CollectionError::Storage(StorageError::InvalidData(
            "trailing bytes in record"
        )))
    ));
}

#[test]
fn restored_vectors_keep_their_exact_normalized_float_bits() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    let mut original =
        PersistentCollection::create(&path, CONFIG, ChunkingConfig::default()).unwrap();
    original
        .add_documents(vec![
            document("first", [0.1, 3.7]),
            document("second", [-0.0001, 8.9]),
        ])
        .unwrap();
    let bits = |collection: &PersistentCollection| {
        (0..2)
            .map(|id| {
                collection
                    .retrieval
                    .data
                    .vectors
                    .vector(id)
                    .unwrap()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let expected = bits(&original);
    drop(original);
    for _ in 0..3 {
        assert_eq!(
            bits(&PersistentCollection::open(&path, CONFIG).unwrap()),
            expected
        );
    }
}

#[test]
fn an_io_failure_poisons_the_writer_without_publishing_the_failed_batch() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    let mut collection =
        PersistentCollection::create(&path, CONFIG, ChunkingConfig::default()).unwrap();
    collection
        .add_document(document("base", [1.0, 0.0]))
        .unwrap();
    let before = std::fs::read(&path).unwrap();
    // A small frame fits in the buffer, so the read-only descriptor induces a
    // flush failure without relying on disk exhaustion or permission changes.
    collection.writer = Some(BufWriter::new(File::open(&path).unwrap()));
    assert!(matches!(
        collection.add_document(document("failed", [0.0, 1.0])),
        Err(CollectionError::Storage(StorageError::Io(_)))
    ));
    assert_eq!(collection.retrieval().len(), 1);
    assert!(collection.writer.is_none());
    assert!(matches!(
        collection.add_document(document("retry", [0.0, 1.0])),
        Err(CollectionError::Storage(StorageError::Poisoned))
    ));
    assert_eq!(std::fs::read(&path).unwrap(), before);
    drop(collection);
    let mut reopened = PersistentCollection::open(&path, CONFIG).unwrap();
    assert_eq!(
        reopened
            .add_document(document("retry", [0.0, 1.0]))
            .unwrap(),
        1
    );
}

#[test]
fn record_size_limits_are_checked_before_allocating_payloads_during_recovery() {
    assert!(matches!(
        encode(&"a large record", 2),
        Err(StorageError::RecordTooLarge { max_bytes: 2 })
    ));
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    let mut bytes = empty_file(&path);
    let length = (MAX_BATCH_BYTES as u64 + 1).to_le_bytes();
    bytes.extend_from_slice(&length);
    bytes.extend_from_slice(&crc32fast::hash(&length).to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    assert!(matches!(
        PersistentCollection::open(&path, CONFIG),
        Err(CollectionError::Storage(
            StorageError::RecordTooLarge { .. }
        ))
    ));
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

#[test]
fn invalid_configuration_is_rejected_before_creating_a_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    assert!(
        PersistentCollection::create(
            &path,
            EmbeddingConfig {
                dimensions: 0,
                ..CONFIG
            },
            ChunkingConfig::default()
        )
        .is_err()
    );
    assert!(!path.exists());
    assert!(
        PersistentCollection::create(
            &path,
            CONFIG,
            ChunkingConfig {
                max_tokens: 0,
                overlap_tokens: 0
            }
        )
        .is_err()
    );
    assert!(!path.exists());
}

#[test]
fn child_process() {
    let Some(path) = std::env::var_os("KESTREL_STORAGE_CHILD_PATH") else {
        return;
    };
    let mut collection = PersistentCollection::open(path, CONFIG).unwrap();
    collection
        .add_documents(vec![
            document("first", [1.0, 0.0]),
            document("second", [0.0, 1.0]),
        ])
        .unwrap();
    // Exit without dropping the collection: durability cannot depend on Drop.
    std::process::exit(23);
}

#[test]
fn acknowledged_batches_survive_process_exit_without_cleanup() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collection.kestrel");
    empty_file(&path);
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "storage::tests::child_process", "--nocapture"])
        .env("KESTREL_STORAGE_CHILD_PATH", &path)
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(23),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reopened = PersistentCollection::open(&path, CONFIG).unwrap();
    assert_eq!(reopened.retrieval().len(), 2);
    assert_eq!(reopened.retrieval().document(1), Some("second"));
}
