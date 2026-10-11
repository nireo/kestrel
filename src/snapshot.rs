//! Version 2 stores original documents once, chunk boundaries, normalized
//! vectors, and chunking settings. Version 1 loads as one chunk per document.
//! Loading holds the encoded file and decoded records in memory.

use std::fs;
#[cfg(unix)]
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::collection::CollectionData;
use crate::{
    ChunkId, ChunkingConfig, ChunkingError, DocId, EmbeddingConfig, IndexError, TOKENIZER_VERSION,
    VectorError,
};

const MAGIC: &[u8; 8] = b"KESTREL\0";
const FORMAT_VERSION: u32 = 2;
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
    #[error("invalid embedding for chunk {chunk_id:?}: {source}")]
    InvalidVector {
        chunk_id: ChunkId,
        #[source]
        source: VectorError,
    },
    #[error("invalid snapshot chunking settings: {0}")]
    Chunking(#[from] ChunkingError),
    #[error("cannot rebuild snapshot index: {0}")]
    Index(#[from] IndexError),
}

// File-format types, separate from runtime indexes. Field order/types are part
// of the binary format and must not change without bumping FORMAT_VERSION.
#[derive(Serialize, Deserialize)]
struct Snapshot<D, C> {
    tokenizer_version: u32,
    embedding: StoredEmbeddingConfig,
    chunking: ChunkingConfig,
    documents: Vec<D>,
    chunks: Vec<C>,
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
struct ChunkRef<'a> {
    doc_id: DocId,
    start: usize,
    end: usize,
    embedding: &'a [f32],
}

#[derive(Serialize, Deserialize)]
struct StoredChunk {
    doc_id: DocId,
    start: usize,
    end: usize,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct LegacySnapshot {
    tokenizer_version: u32,
    embedding: StoredEmbeddingConfig,
    documents: Vec<LegacyDocument>,
}

#[derive(Deserialize)]
struct LegacyDocument {
    text: String,
    embedding: Vec<f32>,
}

pub(crate) fn save(
    path: &Path,
    data: &CollectionData,
    config: &EmbeddingConfig,
    chunking: ChunkingConfig,
) -> Result<(), SnapshotError> {
    if data.bm25.len() != data.chunks.len()
        || data.vectors.len() != data.chunks.len()
        || data.vectors.dimensions() != config.dimensions
    {
        return Err(SnapshotError::InvalidData("indexes are inconsistent"));
    }
    chunking.validate(config.max_tokens)?;
    // Only references/metadata are allocated; source text and floats are borrowed.
    let chunks = data
        .chunks
        .iter()
        .enumerate()
        .map(|(id, chunk)| {
            Ok(ChunkRef {
                doc_id: chunk.doc_id,
                start: chunk.byte_range.start,
                end: chunk.byte_range.end,
                embedding: data
                    .vectors
                    .vector(id as u32)
                    .ok_or(SnapshotError::InvalidData("chunk has no vector"))?,
            })
        })
        .collect::<Result<Vec<_>, SnapshotError>>()?;
    let snapshot = Snapshot {
        tokenizer_version: TOKENIZER_VERSION,
        embedding: StoredEmbeddingConfig::from(config),
        chunking,
        documents: data
            .documents
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        chunks,
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
) -> Result<(CollectionData, ChunkingConfig), SnapshotError> {
    decode(&fs::read(path)?, config)
}

fn decode(
    bytes: &[u8],
    config: &EmbeddingConfig,
) -> Result<(CollectionData, ChunkingConfig), SnapshotError> {
    if bytes.len() < HEADER_SIZE {
        return Err(SnapshotError::InvalidData("truncated header"));
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        return Err(SnapshotError::InvalidMagic);
    }
    let version = u32::from_le_bytes(bytes[MAGIC.len()..HEADER_SIZE].try_into().unwrap());
    let (snapshot, remaining): (Snapshot<String, StoredChunk>, _) = match version {
        FORMAT_VERSION => postcard::take_from_bytes(&bytes[HEADER_SIZE..])?,
        1 => {
            let (legacy, remaining): (LegacySnapshot, _) =
                postcard::take_from_bytes(&bytes[HEADER_SIZE..])?;
            let mut documents = Vec::with_capacity(legacy.documents.len());
            let mut chunks = Vec::with_capacity(legacy.documents.len());
            for document in legacy.documents {
                let doc_id =
                    DocId::try_from(documents.len()).map_err(|_| IndexError::TooManyDocuments)?;
                chunks.push(StoredChunk {
                    doc_id,
                    start: 0,
                    end: document.text.len(),
                    embedding: document.embedding,
                });
                documents.push(document.text);
            }
            // Existing v1 vectors may represent truncated long texts. Preserve
            // them as-is; these settings apply to newly added documents only.
            let defaults = ChunkingConfig::default();
            let chunking = if config.max_tokens >= defaults.max_tokens {
                defaults
            } else {
                ChunkingConfig {
                    max_tokens: config.max_tokens,
                    overlap_tokens: 0,
                }
            };
            (
                Snapshot {
                    tokenizer_version: legacy.tokenizer_version,
                    embedding: legacy.embedding,
                    chunking,
                    documents,
                    chunks,
                },
                remaining,
            )
        }
        found => return Err(SnapshotError::UnsupportedVersion { found }),
    };
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
    snapshot.chunking.validate(config.max_tokens)?;
    if !snapshot.documents.is_empty() && DocId::try_from(snapshot.documents.len() - 1).is_err() {
        return Err(IndexError::TooManyDocuments.into());
    }
    let mut data = CollectionData::new(config.dimensions)
        .map_err(|_| SnapshotError::InvalidData("zero embedding dimensions"))?;
    data.bm25.check_capacity(snapshot.chunks.len())?;
    let mut chunks = snapshot.chunks.into_iter().peekable();
    for (doc_id, text) in snapshot.documents.into_iter().enumerate() {
        if text.trim().is_empty() {
            return Err(SnapshotError::InvalidData("blank document"));
        }
        let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
        let mut embeddings = Vec::new();
        while chunks
            .peek()
            .is_some_and(|chunk| chunk.doc_id as usize == doc_id)
        {
            let chunk = chunks.next().unwrap();
            let range = chunk.start..chunk.end;
            if chunk.start >= chunk.end
                || text
                    .get(range.clone())
                    .is_none_or(|passage| passage.trim().is_empty())
            {
                return Err(SnapshotError::InvalidData("invalid chunk byte range"));
            }
            if let Some(previous) = ranges.last() {
                if range.start <= previous.start
                    || range.end <= previous.end
                    || range.start > previous.end
                {
                    return Err(SnapshotError::InvalidData(
                        "chunks do not cover document in order",
                    ));
                }
            } else if range.start != 0 {
                return Err(SnapshotError::InvalidData(
                    "first chunk does not start at zero",
                ));
            }
            let chunk_id = ChunkId((data.chunks.len() + ranges.len()) as u32);
            data.vectors
                .validate_normalized(&chunk.embedding)
                .map_err(|source| SnapshotError::InvalidVector { chunk_id, source })?;
            ranges.push(range);
            embeddings.push(chunk.embedding);
        }
        if ranges.last().is_none_or(|range| range.end != text.len()) {
            return Err(SnapshotError::InvalidData(
                "chunks do not cover entire document",
            ));
        }
        data.append_document(text, ranges, embeddings)?;
    }
    if chunks.next().is_some() {
        return Err(SnapshotError::InvalidData(
            "unknown or out-of-order chunk parent",
        ));
    }
    Ok((data, snapshot.chunking))
}

fn write_atomic(
    path: &Path,
    write: impl FnOnce(&mut dyn Write) -> Result<(), SnapshotError>,
) -> Result<(), SnapshotError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
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
    #[cfg(unix)]
    directory
        .sync_all()
        .map_err(SnapshotError::SyncAfterReplace)?;
    Ok(())
}

#[cfg(test)]
mod tests;
