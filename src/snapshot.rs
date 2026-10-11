//! Version 1: eight magic bytes, a little-endian u32 format version, then a
//! Postcard payload. Documents appear in ID order; each stores text and one
//! normalized f32 vector. Posting lists and runtime/model objects are omitted.
//! Loading currently holds the encoded file and decoded records in memory.

use std::fs;
#[cfg(unix)]
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    DocId, EmbeddingConfig, IndexError, InvertedIndex, TOKENIZER_VERSION, VectorError, VectorIndex,
};

const MAGIC: &[u8; 8] = b"KESTREL\0";
const FORMAT_VERSION: u32 = 1;
const HEADER_SIZE: usize = MAGIC.len() + 4;

#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("snapshot I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("snapshot was replaced, but syncing its parent directory failed: {0}")]
    SyncAfterReplace(#[source] io::Error),
    #[error("snapshot encoding or decoding failed: {0}")]
    Codec(#[from] postcard::Error),
    #[error("not a Kestrel snapshot")]
    InvalidMagic,
    #[error("unsupported snapshot format version {found}")]
    UnsupportedVersion { found: u32 },
    #[error("unsupported tokenizer version {found}")]
    UnsupportedTokenizer { found: u32 },
    #[error("snapshot embedding configuration does not match the loaded model settings")]
    IncompatibleEmbeddingConfig,
    #[error("invalid snapshot: {0}")]
    InvalidData(&'static str),
    #[error("invalid embedding for document {doc_id}: {source}")]
    InvalidVector {
        doc_id: DocId,
        #[source]
        source: VectorError,
    },
    #[error("cannot rebuild snapshot index: {0}")]
    Index(#[from] IndexError),
}

// These are file-format types, deliberately separate from the runtime indexes.
// Changing their field order or types requires a new FORMAT_VERSION.
#[derive(Serialize, Deserialize)]
struct Snapshot<D> {
    tokenizer_version: u32,
    embedding: StoredEmbeddingConfig,
    documents: Vec<D>,
}

#[derive(Serialize, Deserialize, PartialEq, Eq)]
struct StoredEmbeddingConfig {
    model_id: String,
    artifact_repo: String,
    artifact_file: String,
    dimensions: usize,
    max_tokens: usize,
    query_instruction: String,
}

impl From<&EmbeddingConfig> for StoredEmbeddingConfig {
    fn from(config: &EmbeddingConfig) -> Self {
        Self {
            model_id: config.model_id.into(),
            artifact_repo: config.artifact_repo.into(),
            artifact_file: config.artifact_file.into(),
            dimensions: config.dimensions,
            max_tokens: config.max_tokens,
            query_instruction: config.query_instruction.into(),
        }
    }
}

#[derive(Serialize)]
struct DocumentRef<'a> {
    text: &'a str,
    embedding: &'a [f32],
}

#[derive(Deserialize)]
struct Document {
    text: String,
    embedding: Vec<f32>,
}

pub(crate) fn save(
    path: &Path,
    bm25: &InvertedIndex,
    vectors: &VectorIndex,
    config: &EmbeddingConfig,
) -> Result<(), SnapshotError> {
    if bm25.len() != vectors.len() || vectors.dimensions() != config.dimensions {
        return Err(SnapshotError::InvalidData("indexes are inconsistent"));
    }
    // Borrow the text and float buffers instead of copying the collection.
    let mut documents = Vec::with_capacity(bm25.len());
    for position in 0..bm25.len() {
        let doc_id = DocId::try_from(position).map_err(|_| IndexError::TooManyDocuments)?;
        documents.push(DocumentRef {
            text: bm25.document(doc_id).expect("dense document IDs"),
            embedding: vectors
                .vector(doc_id)
                .ok_or(SnapshotError::InvalidData("document has no vector"))?,
        });
    }
    let snapshot = Snapshot {
        tokenizer_version: TOKENIZER_VERSION,
        embedding: StoredEmbeddingConfig::from(config),
        documents,
    };
    write_atomic(path, |writer| {
        writer.write_all(MAGIC)?;
        writer.write_all(&FORMAT_VERSION.to_le_bytes())?;
        postcard::to_io(&snapshot, writer)?;
        Ok(())
    })
}

pub(crate) fn load(
    path: &Path,
    config: &EmbeddingConfig,
) -> Result<(InvertedIndex, VectorIndex), SnapshotError> {
    decode(&fs::read(path)?, config)
}

fn decode(
    bytes: &[u8],
    config: &EmbeddingConfig,
) -> Result<(InvertedIndex, VectorIndex), SnapshotError> {
    if bytes.len() < HEADER_SIZE {
        return Err(SnapshotError::InvalidData("truncated header"));
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        return Err(SnapshotError::InvalidMagic);
    }
    let version = u32::from_le_bytes(bytes[MAGIC.len()..HEADER_SIZE].try_into().unwrap());
    if version != FORMAT_VERSION {
        return Err(SnapshotError::UnsupportedVersion { found: version });
    }
    let (snapshot, remaining): (Snapshot<Document>, _) =
        postcard::take_from_bytes(&bytes[HEADER_SIZE..])?;
    if !remaining.is_empty() {
        return Err(SnapshotError::InvalidData("trailing bytes"));
    }
    if snapshot.tokenizer_version != TOKENIZER_VERSION {
        return Err(SnapshotError::UnsupportedTokenizer {
            found: snapshot.tokenizer_version,
        });
    }
    if snapshot.embedding != StoredEmbeddingConfig::from(config) {
        return Err(SnapshotError::IncompatibleEmbeddingConfig);
    }
    if !snapshot.documents.is_empty() && DocId::try_from(snapshot.documents.len() - 1).is_err() {
        return Err(IndexError::TooManyDocuments.into());
    }
    let mut bm25 = InvertedIndex::new();
    let mut vectors = VectorIndex::new(config.dimensions)
        .map_err(|_| SnapshotError::InvalidData("zero embedding dimensions"))?;
    for document in snapshot.documents {
        let doc_id = bm25.len() as DocId;
        if document.text.trim().is_empty() {
            return Err(SnapshotError::InvalidData("blank document"));
        }
        vectors
            .validate_normalized(&document.embedding)
            .map_err(|source| SnapshotError::InvalidVector { doc_id, source })?;
        bm25.add_document(document.text)?;
        // Preserve the stored f32 values exactly; do not normalize them again.
        vectors.append_normalized(doc_id, &document.embedding);
    }
    Ok((bm25, vectors))
}

fn write_atomic(
    path: &Path,
    write: impl FnOnce(&mut dyn Write) -> Result<(), SnapshotError>,
) -> Result<(), SnapshotError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    // Open before replacement so an open failure cannot leave an ambiguous save.
    #[cfg(unix)]
    let directory = File::open(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        write(&mut writer)?;
        writer.flush()?;
    }
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| SnapshotError::Io(error.error))?;
    // On other platforms only file syncing and atomic replacement are provided.
    #[cfg(unix)]
    directory
        .sync_all()
        .map_err(SnapshotError::SyncAfterReplace)?;
    Ok(())
}

#[cfg(test)]
mod tests;
