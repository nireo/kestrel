//! Compare length-grouped batches with the original input-order batches.
//! Loads the real model. Run in release mode after caching model artifacts.
use kestrel::{
    ChunkingConfig, EmbeddingConfig, HybridSearchOptions, PreparedChunk, PreparedDocument,
    RetrievalCollection, TextEmbedder,
};
use serde_json::json;
use std::{hint::black_box, time::Instant};

fn embed(
    embedder: &mut TextEmbedder,
    texts: &[&str],
    queries: bool,
    grouped: bool,
) -> Vec<Vec<f32>> {
    if grouped {
        if queries {
            embedder.embed_queries(texts).unwrap()
        } else {
            embedder.embed_documents(texts).unwrap()
        }
    } else {
        texts
            .chunks(32)
            .flat_map(|batch| {
                if queries {
                    embedder.embed_queries(batch).unwrap()
                } else {
                    embedder.embed_documents(batch).unwrap()
                }
            })
            .collect()
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut embedder = TextEmbedder::new()?;
    let rounds: usize = std::env::args().nth(1).unwrap_or("3".into()).parse()?;
    assert!(rounds > 0);
    let mut index =
        RetrievalCollection::new(EmbeddingConfig::default(), ChunkingConfig::default())?;
    let documents = (0..10_000)
        .map(|row| {
            let text = format!("search record{row}");
            PreparedDocument {
                chunks: vec![PreparedChunk {
                    byte_range: 0..text.len(),
                    embedding: (0..384)
                        .map(|i| ((row * 17 + i * 13) % 97) as f32 - 48.0)
                        .collect(),
                }],
                text,
            }
        })
        .collect::<Vec<_>>();
    index.add_documents(documents)?;
    for (name, lengths) in [("mixed", vec![4, 16, 64, 192]), ("uniform", vec![32])] {
        let owned: Vec<_> = (0..64)
            .map(|i| format!("record{i} {}", "search ".repeat(lengths[i % lengths.len()])))
            .collect();
        let texts: Vec<_> = owned.iter().map(String::as_str).collect();
        for queries in [false, true] {
            let expected = embed(&mut embedder, &texts, queries, false);
            let actual = embed(&mut embedder, &texts, queries, true);
            let error = actual
                .iter()
                .flatten()
                .zip(expected.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(error < 1e-4, "batch alignment/numerical error: {error}");
            let mut samples = [Vec::new(), Vec::new()];
            for round in 0..rounds {
                for grouped in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let start = Instant::now();
                    let vectors = embed(&mut embedder, &texts, queries, grouped);
                    samples[usize::from(grouped)].push(start.elapsed().as_secs_f64() * 1000.0);
                    black_box(vectors);
                }
            }
            for (grouped, mut values) in samples.into_iter().enumerate() {
                values.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    json!({"workload":name,"queries":queries,"grouped":grouped==1,"median_ms":values[values.len()/2],"samples_ms":values,"max_error":error})
                );
            }
        }
        let variants = [(false, 1), (true, 1), (false, 8), (true, 8)];
        let mut samples = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for round in 0..=rounds {
            let order: Vec<_> = if round % 2 == 0 {
                (0..4).collect()
            } else {
                (0..4).rev().collect()
            };
            for variant in order {
                let (grouped, workers) = variants[variant];
                let start = Instant::now();
                let vectors = embed(&mut embedder, &texts, true, grouped);
                let inputs: Vec<_> = texts
                    .iter()
                    .zip(&vectors)
                    .map(|(&q, v)| (q, v.as_slice()))
                    .collect();
                let results = index.search_many_with_vectors(
                    &inputs,
                    HybridSearchOptions::default(),
                    std::num::NonZeroUsize::new(workers).unwrap(),
                )?;
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                black_box(results);
                if round > 0 {
                    samples[variant].push(elapsed);
                }
            }
        }
        for ((grouped, workers), mut values) in variants.into_iter().zip(samples) {
            values.sort_by(f64::total_cmp);
            println!(
                "{}",
                json!({"workload":name,"end_to_end":true,"grouped":grouped,"workers":workers,"median_ms":values[values.len()/2],"samples_ms":values})
            );
        }
    }
    Ok(())
}
