//! Reproducible single-threaded retrieval benchmark; no embedding model is loaded.

#[path = "benchmark/data.rs"]
mod data;
#[path = "benchmark/ivf.rs"]
mod ivf;
#[path = "benchmark/measure.rs"]
mod measure;

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use kestrel::{
    Bm25Scratch, HybridSearchOptions, InvertedIndex, VectorIndex, hybrid_search,
    reciprocal_rank_fusion,
};
use serde::Serialize;

use data::{Document, Error, Fingerprint, Header};
use measure::Timing;

#[derive(Debug, Serialize)]
struct Config {
    sizes: Vec<usize>,
    dimensions: usize,
    queries: usize,
    seed: u64,
    rounds: usize,
    warmup: usize,
    limit: usize,
    candidate_depth: usize,
    ivf_clusters: Option<usize>,
    ivf_probes: Vec<usize>,
    ivf_samples: usize,
    ivf_iterations: usize,
    dataset: Option<PathBuf>,
    output: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sizes: vec![1_000, 10_000, 100_000],
            dimensions: 384,
            queries: 100,
            seed: 42,
            rounds: 3,
            warmup: 1,
            limit: 10,
            candidate_depth: 50,
            ivf_clusters: None,
            ivf_probes: vec![1, 4, 16],
            ivf_samples: 4096,
            ivf_iterations: 12,
            dataset: None,
            output: None,
        }
    }
}

impl Config {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Self>, Error> {
        let mut config = Self::default();
        let mut args = args.into_iter();
        let mut synthetic_options = false;
        let mut ivf_options = false;
        while let Some(arg) = args.next() {
            if arg == "--help" || arg == "-h" {
                println!(
                    "cargo run --release --example benchmark -- [OPTIONS]\n\
                    --sizes 1000,10000,100000  Synthetic corpus sizes (1M is opt-in)\n\
                    --dimensions 384          Synthetic vector dimensions\n\
                    --queries 100             Synthetic query count\n\
                    --seed 42                 Synthetic data seed\n\
                    --dataset PATH            Prepared JSONL; replaces synthetic input\n\
                    --rounds 3                Measured passes over every query\n\
                    --warmup 1                Untimed passes per method (may be zero)\n\
                    --limit 10                Returned results per query\n\
                    --candidate-depth 50      Candidates per source in hybrid search\n\
                    --ivf-clusters N          Enable IVF-flat comparison with N clusters\n\
                    --ivf-probes 1,4,16        Cluster probe counts to compare\n\
                    --ivf-samples 4096         Maximum training vectors (seed 42)\n\
                    --ivf-iterations 12        Maximum spherical k-means iterations\n\
                    --output PATH             Save JSON report and exact neighbors"
                );
                return Ok(None);
            }
            let value = args.next().ok_or_else(|| format!("{arg} needs a value"))?;
            match arg.as_str() {
                "--sizes" => {
                    config.sizes = value.split(',').map(str::parse).collect::<Result<_, _>>()?
                }
                "--dimensions" => config.dimensions = value.parse()?,
                "--queries" => config.queries = value.parse()?,
                "--seed" => config.seed = value.parse()?,
                "--rounds" => config.rounds = value.parse()?,
                "--warmup" => config.warmup = value.parse()?,
                "--limit" => config.limit = value.parse()?,
                "--candidate-depth" => config.candidate_depth = value.parse()?,
                "--ivf-clusters" => config.ivf_clusters = Some(value.parse()?),
                "--ivf-probes" => {
                    config.ivf_probes =
                        value.split(',').map(str::parse).collect::<Result<_, _>>()?
                }
                "--ivf-samples" => config.ivf_samples = value.parse()?,
                "--ivf-iterations" => config.ivf_iterations = value.parse()?,
                "--dataset" => config.dataset = Some(value.into()),
                "--output" => config.output = Some(value.into()),
                _ => return Err(format!("unknown argument: {arg}").into()),
            }
            synthetic_options |= matches!(
                arg.as_str(),
                "--sizes" | "--dimensions" | "--queries" | "--seed"
            );
            ivf_options |= matches!(
                arg.as_str(),
                "--ivf-probes" | "--ivf-samples" | "--ivf-iterations"
            );
        }
        if config.dataset.is_some() && synthetic_options {
            return Err("--dataset cannot be combined with synthetic input options".into());
        }
        if ivf_options && config.ivf_clusters.is_none() {
            return Err("IVF options require --ivf-clusters".into());
        }
        if let Some(clusters) = config.ivf_clusters
            && (clusters == 0
                || config.ivf_samples < clusters
                || config.ivf_iterations == 0
                || config.ivf_probes.is_empty()
                || config.ivf_probes.contains(&0))
        {
            return Err(
                "IVF needs positive clusters, probes and iterations, with samples >= clusters"
                    .into(),
            );
        }
        if config.sizes.is_empty()
            || config
                .sizes
                .iter()
                .any(|&n| n == 0 || n > u32::MAX as usize)
            || config.dimensions == 0
            || config.queries == 0
            || config.rounds == 0
            || config.limit == 0
            || config.candidate_depth < config.limit
        {
            return Err("sizes, dimensions, queries, rounds and limit must be positive; candidate-depth must be >= limit".into());
        }
        // Prevent accidentally replacing the dataset with the report.
        if let (Some(dataset), Some(output)) = (&config.dataset, &config.output)
            && output.exists()
            && dataset.canonicalize()? == output.canonicalize()?
        {
            return Err("report output must differ from dataset input".into());
        }
        Ok(Some(config))
    }
}

#[derive(Serialize)]
struct Neighbor {
    doc_id: u32,
    score: f64,
}

#[derive(Serialize)]
struct Run {
    dataset: Header,
    workload_fingerprint: String,
    source_text_bytes: usize,
    vector_payload_bytes: usize,
    vocabulary_terms: usize,
    input_and_build_wall_ms: f64,
    bm25_build_ms: f64,
    vector_build_ms: f64,
    peak_rss_before_build_bytes: Option<u64>,
    peak_rss_after_build_bytes: Option<u64>,
    peak_rss_after_queries_bytes: Option<u64>,
    bm25: Timing,
    bm25_scratch: Timing,
    vector: Timing,
    hybrid: Timing,
    hybrid_scratch: Timing,
    /// Entry i is the exact vector top-k for dataset.queries[i]. IDs are row numbers.
    exact_neighbors: Vec<Vec<Neighbor>>,
    ivf: Option<ivf::IvfReport>,
}

#[derive(Serialize)]
struct Report {
    report_version: u32,
    unix_timestamp_seconds: u64,
    os: &'static str,
    arch: &'static str,
    cpu: Option<String>,
    available_parallelism: Option<usize>,
    rustc: Option<String>,
    git_revision_at_run: Option<String>,
    git_dirty_at_run: Option<bool>,
    config: Config,
    runs: Vec<Run>,
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

/// Lifetime peak RSS includes runtime, input buffers, both indexes and queries.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes the supplied rusage on success. RUSAGE_SELF
    // queries this process; no pointer is retained by the call.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: the successful call above initialized the whole structure.
    let bytes = u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).ok()?;
    bytes.checked_mul(if cfg!(target_os = "macos") { 1 } else { 1024 })
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn peak_rss_bytes() -> Option<u64> {
    None
}

fn cpu_name() -> Option<String> {
    if cfg!(target_os = "macos") {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
    } else {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()?
            .lines()
            .find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|s| s.split_once(':'))
                    .map(|(_, value)| value.trim().to_owned())
            })
    }
}

fn run(config: &Config, count: usize) -> Result<Run, Error> {
    let (header, reader) = if let Some(path) = &config.dataset {
        let (header, reader) = data::open(path)?;
        (header, Some(reader))
    } else {
        (
            data::synthetic_header(count, config.dimensions, config.queries, config.seed),
            None,
        )
    };
    header.validate()?;
    header
        .queries
        .len()
        .checked_mul(config.rounds)
        .ok_or("sample count overflow")?;
    let vector_payload_bytes = header
        .document_count
        .checked_mul(header.dimensions)
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .ok_or("vector size overflow")?;
    println!(
        "\n{}: {} rows, {} dimensions, {} queries",
        header.name,
        header.document_count,
        header.dimensions,
        header.queries.len()
    );
    let peak_rss_before_build_bytes = peak_rss_bytes();
    let mut bm25 = InvertedIndex::new();
    let mut vectors = VectorIndex::new(header.dimensions)?;
    let mut bm25_build = Duration::ZERO;
    let mut vector_build = Duration::ZERO;
    let mut text_bytes = 0;
    let mut fingerprint = Fingerprint::new();
    fingerprint.update(&serde_json::to_vec(&header)?);
    let wall = Instant::now();
    let mut consume = |mut batch: Vec<Document>| -> Result<(), Error> {
        for document in &batch {
            text_bytes += document.text.len();
            fingerprint.row(&document.text, &document.vector);
        }
        let first_id = bm25.len();
        let start = Instant::now();
        for document in &mut batch {
            bm25.add_document(std::mem::take(&mut document.text))?;
        }
        bm25_build += start.elapsed();
        let start = Instant::now();
        for (offset, document) in batch.iter().enumerate() {
            vectors.insert((first_id + offset) as u32, &document.vector)?;
        }
        vector_build += start.elapsed();
        Ok(())
    };
    if let Some(reader) = reader {
        data::read_batches(reader, &header, &mut consume)?;
    } else {
        for start in (0..count).step_by(data::BATCH_SIZE) {
            consume(
                (start..(start + data::BATCH_SIZE).min(count))
                    .map(|row| data::synthetic_document(row, header.dimensions, config.seed))
                    .collect(),
            )?;
        }
    }
    let input_and_build_wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
    let peak_rss_after_build_bytes = peak_rss_bytes();
    println!(
        "Build: BM25 {:.1} ms, vector {:.1} ms; process peak RSS {} MiB",
        bm25_build.as_secs_f64() * 1000.0,
        vector_build.as_secs_f64() * 1000.0,
        peak_rss_after_build_bytes
            .map(|n| format!("{:.1}", n as f64 / 1_048_576.0))
            .unwrap_or("unavailable".into())
    );
    println!(
        "{:<12} {:>10} {:>10} {:>10} {:>12}",
        "Method", "p50 ms", "p95 ms", "p99 ms", "seq QPS"
    );
    let bm25_timing = measure::measure(
        &header.queries,
        config.warmup,
        config.rounds,
        |q| Ok(bm25.search(&q.text, config.limit)),
        |_, _| {},
    )?;
    print_timing("BM25", &bm25_timing);
    let mut scratch = Bm25Scratch::default();
    let bm25_scratch = measure::measure(
        &header.queries,
        config.warmup,
        config.rounds,
        |q| Ok(bm25.search_with_scratch(&q.text, config.limit, &mut scratch)),
        |_, _| {},
    )?;
    print_timing("BM25/reuse", &bm25_scratch);
    let mut exact_neighbors: Vec<Vec<Neighbor>> =
        (0..header.queries.len()).map(|_| Vec::new()).collect();
    let vector_timing = measure::measure(
        &header.queries,
        config.warmup,
        config.rounds,
        |q| Ok(vectors.search(&q.vector, config.limit)?),
        |index, hits| {
            exact_neighbors[index] = hits
                .iter()
                .map(|h| Neighbor {
                    doc_id: h.doc_id,
                    score: h.score,
                })
                .collect()
        },
    )?;
    print_timing("Vector", &vector_timing);
    let options = HybridSearchOptions {
        limit: config.limit,
        candidate_depth: config.candidate_depth,
    };
    let hybrid_timing = measure::measure(
        &header.queries,
        config.warmup,
        config.rounds,
        |q| Ok(hybrid_search(&bm25, &vectors, &q.text, &q.vector, options)?),
        |_, _| {},
    )?;
    print_timing("Hybrid", &hybrid_timing);
    let hybrid_scratch = measure::measure(
        &header.queries,
        config.warmup,
        config.rounds,
        |q| {
            let vector_hits = vectors.search(&q.vector, options.candidate_depth)?;
            let bm25_hits =
                bm25.search_with_scratch(&q.text, options.candidate_depth, &mut scratch);
            Ok(reciprocal_rank_fusion(
                &bm25_hits,
                &vector_hits,
                options.limit,
            ))
        },
        |_, _| {},
    )?;
    print_timing("Hybrid/reuse", &hybrid_scratch);
    let peak_rss_after_queries_bytes = peak_rss_bytes();
    let ivf = ivf::run(config, &bm25, &vectors, &header, &exact_neighbors)?;
    Ok(Run {
        dataset: header,
        workload_fingerprint: fingerprint.finish(),
        source_text_bytes: text_bytes,
        vector_payload_bytes,
        vocabulary_terms: bm25.vocabulary_len(),
        input_and_build_wall_ms,
        bm25_build_ms: bm25_build.as_secs_f64() * 1000.0,
        vector_build_ms: vector_build.as_secs_f64() * 1000.0,
        peak_rss_before_build_bytes,
        peak_rss_after_build_bytes,
        peak_rss_after_queries_bytes,
        bm25: bm25_timing,
        bm25_scratch,
        vector: vector_timing,
        hybrid: hybrid_timing,
        hybrid_scratch,
        exact_neighbors,
        ivf,
    })
}

fn print_timing(name: &str, timing: &Timing) {
    println!(
        "{name:<12} {:>10.3} {:>10.3} {:>10.3} {:>12.1}",
        timing.p50_us / 1000.0,
        timing.p95_us / 1000.0,
        timing.p99_us / 1000.0,
        timing.sequential_qps
    );
}

fn main() -> Result<(), Error> {
    let Some(config) = Config::parse(std::env::args().skip(1))? else {
        return Ok(());
    };
    if cfg!(debug_assertions) {
        return Err(
            "benchmark requires an optimized build: cargo run --release --example benchmark".into(),
        );
    }
    println!(
        "Warm retrieval, single thread; {} warm-up and {} measured passes per method.",
        config.warmup, config.rounds
    );
    println!(
        "Synthetic data measures scaling, not relevance. Embedding and chunking are excluded."
    );
    let sizes = if config.dataset.is_some() {
        vec![0]
    } else {
        config.sizes.clone()
    };
    let mut runs = Vec::new();
    for size in sizes {
        runs.push(run(&config, size)?);
    }
    let report = Report {
        report_version: 2,
        unix_timestamp_seconds: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        cpu: cpu_name(),
        available_parallelism: std::thread::available_parallelism().ok().map(usize::from),
        rustc: command_output("rustc", &["--version"]),
        git_revision_at_run: command_output("git", &["rev-parse", "HEAD"]),
        git_dirty_at_run: command_output("git", &["status", "--porcelain"]).map(|s| !s.is_empty()),
        config,
        runs,
    };
    if let Some(path) = &report.config.output {
        let mut writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(&mut writer, &report)?;
        std::io::Write::flush(&mut writer)?;
        println!("\nReport: {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_workloads_and_conflicting_input_modes() {
        for args in [
            vec!["--sizes", "0"],
            vec!["--queries", "0"],
            vec!["--rounds", "0"],
            vec!["--limit", "51"],
            vec!["--dataset", "data.jsonl", "--seed", "42"],
            vec!["--ivf-probes", "1,4"],
            vec!["--ivf-clusters", "0"],
            vec!["--ivf-clusters", "8", "--ivf-probes", "0"],
            vec!["--ivf-clusters", "8", "--ivf-samples", "4"],
        ] {
            assert!(Config::parse(args.into_iter().map(String::from)).is_err());
        }
        assert!(
            Config::parse(["--warmup", "0"].map(String::from))
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn report_neighbors_follow_query_order_and_samples_exclude_warmup() {
        let config = Config {
            dimensions: 8,
            queries: 4,
            rounds: 2,
            warmup: 1,
            limit: 3,
            candidate_depth: 8,
            ..Config::default()
        };
        let result = run(&config, 70).unwrap();
        let mut reference = VectorIndex::new(8).unwrap();
        for row in 0..70 {
            reference
                .insert(
                    row,
                    &data::synthetic_document(row as usize, 8, config.seed).vector,
                )
                .unwrap();
        }
        for (query, neighbors) in result.dataset.queries.iter().zip(&result.exact_neighbors) {
            let expected = reference.search(&query.vector, 3).unwrap();
            assert_eq!(
                neighbors.iter().map(|h| h.doc_id).collect::<Vec<_>>(),
                expected.iter().map(|h| h.doc_id).collect::<Vec<_>>()
            );
        }
        for timing in [
            &result.bm25,
            &result.bm25_scratch,
            &result.vector,
            &result.hybrid,
            &result.hybrid_scratch,
        ] {
            assert_eq!(timing.samples, 8);
            assert_eq!(timing.query_indices.len(), 8);
            for index in 0..4 {
                assert_eq!(
                    timing.query_indices.iter().filter(|&&i| i == index).count(),
                    2
                );
            }
        }
        assert_eq!(result.vector_payload_bytes, 70 * 8 * 4);
    }

    #[test]
    fn ivf_report_measures_recall_against_exact_query_neighbors() {
        // Also check the recall denominator when k exceeds the corpus size.
        for limit in [3, 100] {
            let config = Config {
                dimensions: 8,
                queries: 4,
                rounds: 1,
                warmup: 0,
                limit,
                candidate_depth: limit,
                ivf_clusters: Some(8),
                ivf_samples: 128,
                ivf_probes: vec![1, 8],
                ..Config::default()
            };
            let result = run(&config, 70).unwrap();
            let probes = result.ivf.unwrap().probes;
            assert_eq!(probes[1].probes, 8);
            assert_eq!(probes[1].mean_recall_at_k, 1.0);
            assert!(probes[0].mean_recall_at_k >= 0.0 && probes[0].mean_recall_at_k <= 1.0);
        }
    }
}
