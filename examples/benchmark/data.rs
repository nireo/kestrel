use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::{Deserialize, Serialize};

pub type Error = Box<dyn std::error::Error>;
pub const FORMAT_VERSION: u32 = 1;
pub const BATCH_SIZE: usize = 256;

/// First JSONL line. Remaining lines are documents in stable row-ID order.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub version: u32,
    pub name: String,
    pub dimensions: usize,
    pub document_count: usize,
    pub embedding_model: Option<String>,
    pub queries: Vec<Query>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Query {
    pub id: String,
    pub text: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Document {
    pub text: String,
    pub vector: Vec<f32>,
}

impl Header {
    pub fn validate(&self) -> Result<(), Error> {
        if self.version != FORMAT_VERSION {
            return Err(format!("unsupported dataset version: {}", self.version).into());
        }
        if self.name.trim().is_empty() || self.dimensions == 0 || self.document_count == 0 {
            return Err("dataset needs a name, positive dimensions and documents".into());
        }
        if self.document_count > u32::MAX as usize {
            return Err("too many documents".into());
        }
        if self.queries.is_empty() {
            return Err("dataset needs queries".into());
        }
        let mut ids = std::collections::HashSet::new();
        for query in &self.queries {
            if query.id.trim().is_empty() || !ids.insert(&query.id) {
                return Err("query IDs must be nonblank and unique".into());
            }
            validate_row(&query.text, &query.vector, self.dimensions)?;
        }
        Ok(())
    }
}

pub fn validate_row(text: &str, vector: &[f32], dimensions: usize) -> Result<(), Error> {
    if text.trim().is_empty() {
        return Err("dataset text must not be blank".into());
    }
    if vector.len() != dimensions
        || vector.iter().any(|v| !v.is_finite())
        || vector.iter().all(|&v| v == 0.0)
    {
        return Err("dataset vector has wrong dimensions, nonfinite values, or zero length".into());
    }
    Ok(())
}

pub fn open(path: &Path) -> Result<(Header, BufReader<File>), Error> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let header: Header = serde_json::from_str(&line)?;
    header.validate()?;
    Ok((header, reader))
}

/// Streaming keeps input vectors from doubling the index's memory footprint.
pub fn read_batches(
    mut reader: impl BufRead,
    header: &Header,
    mut consume: impl FnMut(Vec<Document>) -> Result<(), Error>,
) -> Result<(), Error> {
    let mut batch = Vec::with_capacity(BATCH_SIZE);
    let mut count = 0;
    let mut line = String::new();
    while reader.read_line(&mut line)? != 0 {
        count += 1;
        if count > header.document_count {
            return Err("dataset contains more documents than its header declares".into());
        }
        let document: Document =
            serde_json::from_str(&line).map_err(|e| format!("document row {}: {e}", count - 1))?;
        validate_row(&document.text, &document.vector, header.dimensions)
            .map_err(|e| format!("document row {}: {e}", count - 1))?;
        batch.push(document);
        if batch.len() == BATCH_SIZE {
            consume(std::mem::replace(
                &mut batch,
                Vec::with_capacity(BATCH_SIZE),
            ))?;
        }
        line.clear();
    }
    if count != header.document_count {
        return Err(format!(
            "expected {} documents, found {count}",
            header.document_count
        )
        .into());
    }
    if !batch.is_empty() {
        consume(batch)?;
    }
    Ok(())
}

// SplitMix64: deterministic across platforms; independent streams per row.
fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn signed(value: u64) -> f32 {
    ((mix(value) >> 40) as f32 / 8_388_608.0) - 1.0
}

fn clustered_vector(seed: u64, topic: usize, row: u64, dimensions: usize) -> Vec<f32> {
    (0..dimensions)
        .map(|dimension| {
            let centroid = signed(seed ^ (topic as u64).wrapping_mul(7919) ^ dimension as u64);
            let noise = signed(seed ^ row.wrapping_mul(104729) ^ dimension as u64);
            // A tiny positive offset also makes the one-dimensional case nonzero.
            centroid + 0.3 * noise + 0.0001
        })
        .collect()
}

pub fn synthetic_document(row: usize, dimensions: usize, seed: u64) -> Document {
    let mut text = format!("search document topic{} record{row}", row % 64);
    let words = 32 + (mix(seed ^ row as u64) % 97) as usize;
    for position in 0..words {
        let term = mix(seed ^ (row as u64).wrapping_mul(104729) ^ position as u64) % 1024;
        text.push_str(&format!(" term{term}"));
    }
    Document {
        text,
        vector: clustered_vector(seed, row % 64, row as u64, dimensions),
    }
}

pub fn synthetic_header(count: usize, dimensions: usize, queries: usize, seed: u64) -> Header {
    Header {
        version: FORMAT_VERSION,
        name: "synthetic-clustered-v1".into(),
        dimensions,
        document_count: count,
        embedding_model: None,
        queries: (0..queries)
            .map(|i| {
                let row = (mix(seed ^ i as u64) % count as u64) as usize;
                let (kind, text) = match i % 4 {
                    0 => ("topic", format!("topic{}", row % 64)),
                    1 => ("common", "search document".into()),
                    2 => ("rare", format!("record{row}")),
                    _ => ("missing", "absentkeyword".into()),
                };
                Query {
                    id: format!("{kind}-{i}"),
                    text,
                    vector: clustered_vector(seed, row % 64, !(i as u64), dimensions),
                }
            })
            .collect(),
    }
}

/// Stable workload identifier, not a cryptographic integrity check.
pub struct Fingerprint(u64);

impl Fingerprint {
    pub fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    pub fn update(&mut self, bytes: &[u8]) {
        for byte in (bytes.len() as u64).to_le_bytes().iter().chain(bytes) {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    pub fn row(&mut self, text: &str, vector: &[f32]) {
        self.update(text.as_bytes());
        for component in vector {
            self.update(&component.to_bits().to_le_bytes());
        }
    }

    pub fn finish(&self) -> String {
        format!("{:016x}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_rows_are_reproducible_and_workload_has_four_selectivities() {
        let a = synthetic_document(123, 384, 42);
        let b = synthetic_document(123, 384, 42);
        assert_eq!(a.text, b.text);
        assert_eq!(a.vector, b.vector);
        assert_ne!(a.vector, synthetic_document(124, 384, 42).vector);
        let header = synthetic_header(1000, 384, 8, 42);
        header.validate().unwrap();
        assert!(header.queries[0].text.starts_with("topic"));
        assert_eq!(header.queries[1].text, "search document");
        assert!(header.queries[2].text.starts_with("record"));
        assert_eq!(header.queries[3].text, "absentkeyword");
    }

    #[test]
    fn streamed_input_rejects_truncation_extra_rows_and_invalid_vectors() {
        let header = synthetic_header(1, 2, 1, 42);
        let row = serde_json::to_string(&synthetic_document(0, 2, 42)).unwrap();
        let mut received = 0;
        read_batches(row.as_bytes(), &header, |batch| {
            received += batch.len();
            Ok(())
        })
        .unwrap();
        assert_eq!(received, 1);
        for input in [
            String::new(),
            format!("{row}\n{row}"),
            r#"{"text":"x","vector":[0,0]}"#.into(),
        ] {
            assert!(read_batches(input.as_bytes(), &header, |_| Ok(())).is_err());
        }
    }

    #[test]
    fn streaming_preserves_rows_across_batch_boundaries() {
        let header = synthetic_header(BATCH_SIZE + 1, 2, 1, 42);
        let rows: Vec<_> = (0..header.document_count)
            .map(|row| synthetic_document(row, 2, 42))
            .collect();
        let input = rows
            .iter()
            .map(|row| serde_json::to_string(row).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let mut batch_sizes = Vec::new();
        let mut received = Vec::new();
        read_batches(input.as_bytes(), &header, |batch| {
            batch_sizes.push(batch.len());
            received.extend(batch);
            Ok(())
        })
        .unwrap();
        assert_eq!(batch_sizes, [BATCH_SIZE, 1]);
        for (expected, actual) in rows.iter().zip(received) {
            assert_eq!(expected.text, actual.text);
            assert_eq!(expected.vector, actual.vector);
        }
    }
}
