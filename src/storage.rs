//! One append-only file: a checked configuration header followed by atomic
//! document batches. IDs are inferred from insertion order. No index structures
//! or model artifacts are persisted, and only the current format is supported.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    ChunkingConfig, CollectionError, DocId, EmbeddingConfig, PreparedDocument, RetrievalCollection,
    TOKENIZER_VERSION, VectorSearchMode,
};

const MAGIC: &[u8; 8] = b"KSTRLOG\0";
const FORMAT_VERSION: u32 = 1;
const MAX_HEADER_BYTES: usize = 64 * 1024;
/// Split larger transactions into smaller batches. Recovery buffers one encoded
/// transaction at a time, rather than holding the complete file in memory.
const MAX_BATCH_BYTES: usize = 64 * 1024 * 1024;
// Keep ordinary batch buffers, but do not pin a rare 64 MiB record per writer.
const MAX_RETAINED_BYTES: usize = 1024 * 1024;
const FRAME_PREFIX_BYTES: usize = 12; // u64 payload length + CRC32 of length

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("collection storage I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("collection encoding or decoding failed: {0}")]
    Codec(#[from] postcard::Error),
    #[error("collection is already open by another writer")]
    Locked,
    #[error("a previous write failed; reopen the collection before writing again")]
    Poisoned,
    #[error("invalid collection file: {0}")]
    InvalidData(&'static str),
    #[error("unsupported collection format version {0}")]
    UnsupportedVersion(u32),
    #[error("stored embedding configuration does not match the supplied settings")]
    IncompatibleEmbeddingConfig,
    #[error("encoded record exceeds {max_bytes} bytes; use a smaller batch")]
    RecordTooLarge { max_bytes: usize },
}

// Field order/types form the current format. Bump FORMAT_VERSION when changing
// these or the serialized PreparedDocument/PreparedChunk representation.
#[derive(Serialize, Deserialize)]
struct Header {
    tokenizer_version: u32,
    embedding: StoredEmbeddingConfig,
    chunking: ChunkingConfig,
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

impl From<EmbeddingConfig> for StoredEmbeddingConfig {
    fn from(config: EmbeddingConfig) -> Self {
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

/// Model-free retrieval backed by a single append-only file with an exclusive
/// writer lock. A successful insertion is synced before it becomes searchable.
///
/// Each batch is one transaction (at most 64 MiB encoded) and one sync. Opening
/// rebuilds in-memory indexes, preserves IDs and vector bits, and starts in exact
/// mode. An incomplete final transaction is removed; complete corrupt records
/// are errors. An I/O failure can have an uncertain outcome, so it disables
/// further writes on this handle. Reopen to recover before retrying.
pub struct PersistentCollection {
    retrieval: RetrievalCollection,
    // None after a write/flush/sync failure. Taking the writer lets us discard
    // its pending bytes without BufWriter retrying a failed flush during Drop.
    writer: Option<BufWriter<File>>,
    buffer: Vec<u8>,
}

impl PersistentCollection {
    /// Create a new collection file. Never overwrites an existing path. The
    /// parent directory must exist. A failed creation may leave a partial file.
    pub fn create(
        path: impl AsRef<Path>,
        embedding: EmbeddingConfig,
        chunking: ChunkingConfig,
    ) -> Result<Self, CollectionError> {
        let retrieval = RetrievalCollection::new(embedding, chunking)?;
        let header = encode(
            &Header {
                tokenizer_version: TOKENIZER_VERSION,
                embedding: embedding.into(),
                chunking,
            },
            MAX_HEADER_BYTES,
        )?;

        let path = path.as_ref();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(StorageError::Io)?;

        lock(&file)?;
        let mut writer = BufWriter::new(file);
        let result = writer
            .write_all(MAGIC)
            .and_then(|()| writer.write_all(&FORMAT_VERSION.to_le_bytes()))
            .map_err(StorageError::Io)
            .and_then(|()| write_synced_frame(&mut writer, &header));
        if let Err(error) = result {
            let _ = writer.into_parts();
            return Err(error.into());
        }

        // Persist the newly created directory entry before acknowledging any
        // writes. Existing append operations only need to sync the file itself.
        #[cfg(unix)]
        {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            File::open(parent)
                .and_then(|directory| directory.sync_all())
                .map_err(StorageError::Io)?;
        }

        Ok(Self {
            retrieval,
            writer: Some(writer),
            buffer: header,
        })
    }

    /// Open an existing file without loading an embedding model. Records are
    /// decoded one batch at a time. Recovery truncates only an incomplete final
    /// frame, after validating all preceding records and configuration settings.
    pub fn open(
        path: impl AsRef<Path>,
        embedding: EmbeddingConfig,
    ) -> Result<Self, CollectionError> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(StorageError::Io)?;
        lock(&file)?;

        let file_len = file.metadata().map_err(StorageError::Io)?.len();
        let mut reader = BufReader::new(&mut file);
        let mut magic = [0; 8];
        reader.read_exact(&mut magic).map_err(StorageError::Io)?;
        if &magic != MAGIC {
            return Err(StorageError::InvalidData("wrong magic").into());
        }

        let mut version = [0; 4];
        reader.read_exact(&mut version).map_err(StorageError::Io)?;
        let version = u32::from_le_bytes(version);
        if version != FORMAT_VERSION {
            return Err(StorageError::UnsupportedVersion(version).into());
        }

        let mut remaining = file_len - 12;
        let mut buffer = Vec::new();
        let Frame::Complete =
            read_frame(&mut reader, &mut remaining, MAX_HEADER_BYTES, &mut buffer)?
        else {
            return Err(StorageError::InvalidData("incomplete configuration header").into());
        };

        let header: Header = decode(&buffer)?;
        if header.tokenizer_version != TOKENIZER_VERSION {
            return Err(StorageError::InvalidData("unsupported tokenizer version").into());
        }

        if header.embedding != StoredEmbeddingConfig::from(embedding) {
            return Err(StorageError::IncompatibleEmbeddingConfig.into());
        }

        let mut retrieval = RetrievalCollection::new(embedding, header.chunking)?;
        let mut valid_end = file_len - remaining;
        let torn_tail = loop {
            match read_frame(&mut reader, &mut remaining, MAX_BATCH_BYTES, &mut buffer)? {
                Frame::Complete => {
                    let documents: Vec<PreparedDocument> = decode(&buffer)?;
                    recycle_buffer(&mut buffer);
                    if documents.is_empty() {
                        return Err(StorageError::InvalidData("empty transaction").into());
                    }
                    retrieval.data.insert_batch(documents, false, |_| Ok(()))?;
                    valid_end = file_len - remaining;
                }
                Frame::End => break false,
                Frame::Incomplete => break true,
            }
        };
        recycle_buffer(&mut buffer);

        drop(reader);
        if torn_tail {
            file.set_len(valid_end).map_err(StorageError::Io)?;
        }

        // A previous process may have exited after completing an append but
        // before syncing or acknowledging it. Make recovered records durable
        // before allowing new acknowledgements, including on a clean EOF.
        file.sync_all().map_err(StorageError::Io)?;
        file.seek(SeekFrom::End(0)).map_err(StorageError::Io)?;

        Ok(Self {
            retrieval,
            writer: Some(BufWriter::new(file)),
            buffer,
        })
    }

    pub fn add_document(&mut self, document: PreparedDocument) -> Result<DocId, CollectionError> {
        let ids = self.insert_documents([document])?;
        Ok(ids.start as DocId)
    }

    /// Validate and normalize every document, then persist and publish the
    /// whole batch atomically. Validation/encoding errors change neither disk
    /// nor indexes. Empty batches do not write or sync anything.
    pub fn add_documents(
        &mut self,
        documents: Vec<PreparedDocument>,
    ) -> Result<Vec<DocId>, CollectionError> {
        let ids = self.insert_documents(documents)?;
        Ok(ids.map(|id| id as DocId).collect())
    }

    fn insert_documents(
        &mut self,
        documents: impl AsMut<[PreparedDocument]> + IntoIterator<Item = PreparedDocument>,
    ) -> Result<Range<usize>, CollectionError> {
        let writer = self.writer.as_mut().ok_or(StorageError::Poisoned)?;
        let buffer = &mut self.buffer;
        let mut failed = false;
        let result = self
            .retrieval
            .data
            .insert_batch(documents, true, |documents| {
                encode_into(documents, MAX_BATCH_BYTES, buffer)?;
                let result = write_synced_frame(writer, buffer);
                failed = result.is_err();
                // Release oversized payloads before publishing the decoded
                // documents into the indexes, matching the original peak usage.
                recycle_buffer(buffer);
                result.map_err(CollectionError::from)
            });
        recycle_buffer(buffer);
        if failed {
            let writer = self
                .writer
                .take()
                .expect("writer exists until an I/O failure");
            let _ = writer.into_parts();
        }
        result
    }

    /// Shared read access; mutable access is withheld so inserts cannot bypass
    /// persistence. Searches can run concurrently through this reference.
    pub fn retrieval(&self) -> &RetrievalCollection {
        &self.retrieval
    }

    /// Retrieval settings are runtime-only and do not alter the durable data.
    pub fn set_vector_search_mode(
        &mut self,
        mode: VectorSearchMode,
    ) -> Result<(), CollectionError> {
        self.retrieval.set_vector_search_mode(mode)
    }

    pub fn rebuild_vector_index(&mut self) -> Result<(), CollectionError> {
        self.retrieval.rebuild_vector_index()
    }
}

fn lock(file: &File) -> Result<(), StorageError> {
    file.try_lock().map_err(|error| match error {
        TryLockError::WouldBlock => StorageError::Locked,
        TryLockError::Error(error) => StorageError::Io(error),
    })
}

fn encode(value: &(impl Serialize + ?Sized), limit: usize) -> Result<Vec<u8>, StorageError> {
    let mut bytes = Vec::new();
    encode_into(value, limit, &mut bytes)?;
    Ok(bytes)
}

fn encode_into(
    value: &(impl Serialize + ?Sized),
    limit: usize,
    bytes: &mut Vec<u8>,
) -> Result<(), StorageError> {
    bytes.clear();
    let size = postcard::experimental::serialized_size(value)?;
    if size > limit {
        return Err(StorageError::RecordTooLarge { max_bytes: limit });
    }
    // These private callers serialize plain owned headers and documents. Their
    // derived serializers are deterministic across the size and output passes.
    if size > bytes.capacity() {
        let capacity = size.max(bytes.capacity().saturating_mul(2)).min(limit);
        bytes.reserve_exact(capacity);
    }
    postcard::to_io(value, &mut *bytes)?;
    debug_assert_eq!(bytes.len(), size);
    Ok(())
}

fn recycle_buffer(bytes: &mut Vec<u8>) {
    bytes.clear();
    if bytes.capacity() > MAX_RETAINED_BYTES {
        *bytes = Vec::new();
    }
}

fn decode<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T, StorageError> {
    let (value, remaining) = postcard::take_from_bytes(bytes)?;
    if !remaining.is_empty() {
        return Err(StorageError::InvalidData("trailing bytes in record"));
    }
    Ok(value)
}

fn write_frame(writer: &mut impl Write, bytes: &[u8]) -> Result<(), StorageError> {
    let length = (bytes.len() as u64).to_le_bytes();
    writer.write_all(&length)?;
    writer.write_all(&crc32fast::hash(&length).to_le_bytes())?;
    writer.write_all(bytes)?;
    writer.write_all(&crc32fast::hash(bytes).to_le_bytes())?;
    Ok(())
}

fn write_synced_frame(writer: &mut BufWriter<File>, bytes: &[u8]) -> Result<(), StorageError> {
    write_frame(writer, bytes)?;
    // Flush the userspace buffer before asking the OS to persist the file.
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

enum Frame {
    Complete,
    End,
    Incomplete,
}

fn read_frame(
    reader: &mut impl Read,
    remaining: &mut u64,
    limit: usize,
    bytes: &mut Vec<u8>,
) -> Result<Frame, StorageError> {
    if *remaining == 0 {
        return Ok(Frame::End);
    }
    if *remaining < FRAME_PREFIX_BYTES as u64 {
        return Ok(Frame::Incomplete);
    }
    let mut prefix = [0; FRAME_PREFIX_BYTES];
    reader.read_exact(&mut prefix)?;
    let length_bytes: [u8; 8] = prefix[..8].try_into().unwrap();
    let checksum = u32::from_le_bytes(prefix[8..].try_into().unwrap());
    if crc32fast::hash(&length_bytes) != checksum {
        return Err(StorageError::InvalidData("record length checksum mismatch"));
    }
    let length = u64::from_le_bytes(length_bytes);
    if length > limit as u64 {
        return Err(StorageError::RecordTooLarge { max_bytes: limit });
    }
    let frame_size = FRAME_PREFIX_BYTES as u64 + length + 4;
    if *remaining < frame_size {
        return Ok(Frame::Incomplete);
    }
    bytes.resize(length as usize, 0);
    reader.read_exact(bytes)?;
    let mut checksum = [0; 4];
    reader.read_exact(&mut checksum)?;
    if crc32fast::hash(bytes) != u32::from_le_bytes(checksum) {
        return Err(StorageError::InvalidData(
            "record payload checksum mismatch",
        ));
    }
    *remaining -= frame_size;
    Ok(Frame::Complete)
}

#[cfg(test)]
mod tests;
