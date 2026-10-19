//! Compare retrieval quality on a small, editable dataset. This measures
//! relevance, not throughput; document/query embeddings are shared by all modes.

#[path = "evaluate/dataset.rs"]
mod dataset;
#[path = "evaluate/metrics.rs"]
mod metrics;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use kestrel::{HybridSearchOptions, InvertedIndex, TextEmbedder, VectorIndex, hybrid_search};

use dataset::Dataset;
use metrics::{Metrics, score};

const RECALL_CUTOFF: usize = 5;
const METHODS: [&str; 3] = ["BM25", "Vector", "Hybrid"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("eval/dataset.json");
    let mut candidate_depth = HybridSearchOptions::default().candidate_depth;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => path = args.next().ok_or("--dataset needs a path")?.into(),
            "--candidate-depth" => {
                candidate_depth = args
                    .next()
                    .ok_or("--candidate-depth needs a number")?
                    .parse()?;
            }
            "--help" | "-h" => {
                println!("cargo run --example evaluate -- [--dataset PATH] [--candidate-depth N]");
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    // Reject invalid labels before loading the embedding model.
    let dataset = Dataset::parse(&std::fs::read_to_string(&path)?)?;
    println!("Dataset: {}", dataset.name);
    println!(
        "{} documents, {} queries",
        dataset.documents.len(),
        dataset.queries.len()
    );
    println!("Hybrid candidates per source: {candidate_depth}");
    println!("Recall@{RECALL_CUTOFF}; MRR over all returned ranks\n");

    let mut embedder = TextEmbedder::new()?;
    println!("Embedding settings: {:?}\n", embedder.config());
    let texts: Vec<_> = dataset
        .documents
        .iter()
        .map(|doc| doc.text.as_str())
        .collect();
    let embeddings = embedder.embed_documents(&texts)?;
    let mut bm25 = InvertedIndex::new();
    let mut vectors = VectorIndex::new(embedder.dimensions())?;
    let mut document_ids = HashMap::new();
    for (document, embedding) in dataset.documents.iter().zip(embeddings) {
        let doc_id = bm25.add_document(document.text.as_str())?;
        vectors.insert(doc_id, &embedding)?;
        document_ids.insert(document.id.as_str(), doc_id);
    }

    let mut measurements: [Vec<Metrics>; 3] = std::array::from_fn(|_| Vec::new());
    let queries: Vec<_> = dataset.queries.iter().map(|q| q.text.as_str()).collect();
    let query_vectors = embedder.embed_queries(&queries)?;
    for (query, query_vector) in dataset.queries.iter().zip(query_vectors) {
        let relevant: HashSet<_> = query
            .relevant_document_ids
            .iter()
            .map(|id| document_ids[id.as_str()])
            .collect();
        // Retrieve full rankings for reciprocal rank; only recall uses top 5.
        let options = HybridSearchOptions {
            candidate_depth,
            limit: bm25.len(),
        };
        let rankings = [
            bm25.search(&query.text, bm25.len())
                .into_iter()
                .map(|hit| hit.doc_id)
                .collect::<Vec<_>>(),
            vectors
                .search(&query_vector, bm25.len())?
                .into_iter()
                .map(|hit| hit.doc_id)
                .collect(),
            hybrid_search(&bm25, &vectors, &query.text, &query_vector, options)?
                .into_iter()
                .map(|hit| hit.doc_id)
                .collect(),
        ];
        println!("{}: {}", query.id, query.text);
        println!("  Relevant: {}", query.relevant_document_ids.join(", "));
        for (method, ranking) in rankings.iter().enumerate() {
            let result = score(ranking, &relevant, RECALL_CUTOFF);
            measurements[method].push(result);
            let top_ids: Vec<_> = ranking
                .iter()
                .take(RECALL_CUTOFF)
                .map(|&id| dataset.documents[id as usize].id.as_str())
                .collect();
            println!(
                "  {:<6} Recall@5={:.3} RR={:.3}  [{}]",
                METHODS[method],
                result.recall_at_k,
                result.reciprocal_rank,
                top_ids.join(", ")
            );
        }
        println!();
    }
    println!(
        "Mean across {} queries (each query has equal weight):",
        dataset.queries.len()
    );
    println!("{:<8} {:>10} {:>10}", "Method", "Recall@5", "MRR");
    for (method, results) in METHODS.iter().zip(&measurements) {
        let mean = Metrics::mean(results);
        println!(
            "{method:<8} {:>10.3} {:>10.3}",
            mean.recall_at_k, mean.reciprocal_rank
        );
    }
    Ok(())
}
