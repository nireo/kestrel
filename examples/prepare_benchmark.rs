//! Embed a text corpus once, then benchmark retrieval independently of inference.

#[allow(dead_code)]
#[path = "benchmark/data.rs"]
mod data;

use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use kestrel::TextEmbedder;
use serde::Deserialize;

use data::{Document, Error, Header, Query};

// Extra fields such as relevance labels in eval/dataset.json are intentionally ignored.
#[derive(Deserialize)]
struct Input {
    name: String,
    documents: Vec<Text>,
    queries: Vec<TextQuery>,
}

#[derive(Deserialize)]
struct Text {
    text: String,
}

#[derive(Deserialize)]
struct TextQuery {
    id: String,
    text: String,
}

impl Input {
    fn validate(&self) -> Result<(), Error> {
        if self.name.trim().is_empty() || self.documents.is_empty() || self.queries.is_empty() {
            return Err("input needs a name, documents and queries".into());
        }
        if self.documents.len() > u32::MAX as usize
            || self.documents.iter().any(|d| d.text.trim().is_empty())
            || self.queries.iter().any(|q| q.text.trim().is_empty())
        {
            return Err("input contains blank text or too many documents".into());
        }
        let mut ids = std::collections::HashSet::new();
        if self
            .queries
            .iter()
            .any(|q| q.id.trim().is_empty() || !ids.insert(&q.id))
        {
            return Err("query IDs must be nonblank and unique".into());
        }
        Ok(())
    }
}

fn main() -> Result<(), Error> {
    let mut input_path = None;
    let mut output_path = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!(
                "cargo run --release --example prepare_benchmark -- --input TEXT.json --output VECTORS.jsonl\n\
                Text JSON: name, documents [{{text}}], queries [{{id, text}}].\n\
                Use short passages: the embedding model truncates inputs at 512 tokens.\n\
                Output must be a new file. Embeddings may download on first use."
            );
            return Ok(());
        }
        let value = args.next().ok_or_else(|| format!("{arg} needs a value"))?;
        match arg.as_str() {
            "--input" => input_path = Some(PathBuf::from(value)),
            "--output" => output_path = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    let input_path = input_path.ok_or("--input is required")?;
    let output_path = output_path.ok_or("--output is required")?;
    if output_path.exists() {
        return Err("output already exists; choose a new file".into());
    }
    let mut input: Input = serde_json::from_reader(fs::File::open(input_path)?)?;
    input.validate()?;
    let parent = output_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    let start = Instant::now();
    let mut embedder = TextEmbedder::new()?;
    let model_load = start.elapsed();
    let mut queries = Vec::with_capacity(input.queries.len());
    let mut query_times = Vec::with_capacity(input.queries.len());
    for query in input.queries {
        let start = Instant::now();
        let vector = embedder.embed_query(&query.text)?;
        query_times.push(start.elapsed().as_secs_f64() * 1000.0);
        queries.push(Query {
            id: query.id,
            text: query.text,
            vector,
        });
    }
    let header = Header {
        version: data::FORMAT_VERSION,
        name: input.name,
        dimensions: embedder.dimensions(),
        document_count: input.documents.len(),
        embedding_model: Some(embedder.config().model_id.into()),
        queries,
    };
    header.validate()?;
    let mut writer = BufWriter::new(temporary.as_file_mut());
    serde_json::to_writer(&mut writer, &header)?;
    writeln!(writer)?;
    let mut document_time = Duration::ZERO;
    for (batch_index, batch) in input.documents.chunks_mut(data::BATCH_SIZE).enumerate() {
        let texts: Vec<_> = batch.iter().map(|d| d.text.as_str()).collect();
        let start = Instant::now();
        let vectors = embedder.embed_documents(&texts)?;
        document_time += start.elapsed();
        for (document, vector) in batch.iter_mut().zip(vectors) {
            serde_json::to_writer(
                &mut writer,
                &Document {
                    text: std::mem::take(&mut document.text),
                    vector,
                },
            )?;
            writeln!(writer)?;
        }
        println!(
            "Embedded {}/{} passages",
            ((batch_index + 1) * data::BATCH_SIZE).min(header.document_count),
            header.document_count
        );
    }
    writer.flush()?;
    drop(writer);
    temporary.persist_noclobber(&output_path)?;
    println!(
        "Model load: {:.1} ms (may include download)",
        model_load.as_secs_f64() * 1000.0
    );
    println!(
        "Document embedding: {:.1} ms, {:.1} passages/s",
        document_time.as_secs_f64() * 1000.0,
        header.document_count as f64 / document_time.as_secs_f64()
    );
    println!(
        "Query embedding: {:.1} ms total, {:.1} ms mean (one pass; includes first inference)",
        query_times.iter().sum::<f64>(),
        query_times.iter().sum::<f64>() / query_times.len() as f64
    );
    println!("Prepared dataset: {}", output_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_accepts_existing_eval_schema_and_rejects_duplicate_queries() {
        let mut input: Input = serde_json::from_str(include_str!("../eval/dataset.json")).unwrap();
        input.validate().unwrap();
        input.queries[1].id = input.queries[0].id.clone();
        assert!(input.validate().is_err());
    }
}
