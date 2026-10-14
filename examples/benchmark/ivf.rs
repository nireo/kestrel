use std::time::Instant;

use kestrel::{
    HybridSearchOptions, InvertedIndex, IvfBuildOptions, IvfIndex, VectorIndex, hybrid_search_ivf,
};
use serde::Serialize;

use super::data::{Error, Header};
use super::measure::{self, Timing};
use super::{Config, Neighbor, peak_rss_bytes, print_timing};

#[derive(Serialize)]
pub struct IvfReport {
    build_ms: f64,
    clusters: usize,
    training_samples: usize,
    cluster_sizes: Vec<usize>,
    peak_rss_after_build_bytes: Option<u64>,
    peak_rss_after_queries_bytes: Option<u64>,
    pub probes: Vec<ProbeReport>,
}

#[derive(Serialize)]
pub struct ProbeReport {
    pub probes: usize,
    pub mean_recall_at_k: f64,
    recall_per_query: Vec<f64>,
    vector: Timing,
    hybrid: Timing,
}

pub fn run(
    config: &Config,
    bm25: &InvertedIndex,
    source: &VectorIndex,
    header: &Header,
    exact: &[Vec<Neighbor>],
) -> Result<Option<IvfReport>, Error> {
    let Some(clusters) = config.ivf_clusters else {
        return Ok(None);
    };
    let start = Instant::now();
    let index = IvfIndex::build(
        source,
        IvfBuildOptions {
            clusters,
            training_samples: config.ivf_samples,
            iterations: config.ivf_iterations,
            ..IvfBuildOptions::default()
        },
    )?;
    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    let peak_rss_after_build_bytes = peak_rss_bytes();
    println!(
        "\nIVF build: {build_ms:.1} ms, {} clusters, {} training vectors",
        index.cluster_count(),
        config.ivf_samples.min(source.len())
    );
    let mut probes = Vec::new();
    for &count in &config.ivf_probes {
        let mut recall_per_query = vec![0.0; header.queries.len()];
        let vector = measure::measure(
            &header.queries,
            config.warmup,
            config.rounds,
            |q| Ok(index.search(&q.vector, config.limit, count)?),
            |query, hits| {
                let reference = &exact[query];
                let overlap = hits
                    .iter()
                    .filter(|h| reference.iter().any(|e| e.doc_id == h.doc_id))
                    .count();
                recall_per_query[query] = overlap as f64 / reference.len() as f64;
            },
        )?;
        print_timing(&format!("IVF/{count}"), &vector);
        let mean_recall_at_k = recall_per_query.iter().sum::<f64>() / recall_per_query.len() as f64;
        println!(
            "  Vector recall@{}: {:.2}% ({} clusters probed)",
            config.limit,
            mean_recall_at_k * 100.0,
            count.min(index.cluster_count())
        );
        let options = HybridSearchOptions {
            limit: config.limit,
            candidate_depth: config.candidate_depth,
        };
        let hybrid = measure::measure(
            &header.queries,
            config.warmup,
            config.rounds,
            |q| {
                Ok(hybrid_search_ivf(
                    bm25, &index, &q.text, &q.vector, options, count,
                )?)
            },
            |_, _| {},
        )?;
        print_timing(&format!("Hybrid/{count}"), &hybrid);
        probes.push(ProbeReport {
            probes: count,
            mean_recall_at_k,
            recall_per_query,
            vector,
            hybrid,
        });
    }
    Ok(Some(IvfReport {
        build_ms,
        clusters: index.cluster_count(),
        training_samples: config.ivf_samples.min(source.len()),
        cluster_sizes: index.cluster_sizes().collect(),
        peak_rss_after_build_bytes,
        peak_rss_after_queries_bytes: peak_rss_bytes(),
        probes,
    }))
}
