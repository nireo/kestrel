# profiling and small performance experiments

Measured on the same Apple Silicon machine as the previous follow-ups, starting
at `65ed784`. Release builds, offline dependencies, 384-dimensional vectors.
Each table value is the median of three process medians (five measured rounds
per process, one warm-up). Variants were interleaved, with their order reversed
in the second trial. Fixture construction is outside timing; allocation counts
are collected separately from timing. These are local synthetic workloads, not
production latency guarantees. Commands and raw JSON are under
`target/profile-pass/`; the stage harness is `examples/benchmark_pipeline.rs`.

## profiling

Three-second, one-millisecond macOS `sample` captures of the baseline showed:

- Exact retrieval: 2,519 of 2,544 main-thread samples were under collection
  vector search. Optimized/inlined symbols prevent a reliable split between
  the dot product and the surrounding scan. Final-result allocations are not
  the main single-query bottleneck.
- Persistent append: 1,607 of 1,869 main-thread samples were in
  `File::sync_all` / `fcntl`. These are wall-stack samples including waits,
  not CPU percentages. Durability is unchanged by this pass.
- Ingestion: 1,864 of 2,471 samples were in the insertion path; fixture creation
  and destruction account for much of the remainder. Lexical preparation and
  hash-table allocation were visible, alongside inlined vector preparation.

A separate embedding capture showed ONNX Runtime `MatMul` / GEMM kernels and
thread-pool coordination on the inference stacks. It includes startup and model
loading, so it is qualitative evidence; parked worker threads are not counted
as CPU cost. The embedding timing table below isolates inference more directly.

Captures: `target/profile-pass/baseline-{query,ingest,storage}.sample.txt` and
`target/profile-pass/embedding.sample.txt`.
Sampling needs permission to inspect the benchmark process on macOS; the
profiler launched and terminated its own benchmark child.

## ingestion

| workload | baseline ms | borrowed ranges ms | ranges + reused counter ms | baseline → combined allocations |
| --- | ---: | ---: | ---: | ---: |
| 10k short passages | 5.012 | 4.919 | 4.852 | 51,201 → 41,241 |
| 2k passages, 256 distinct terms | 27.193 | 27.119 | 15.774 | 34,772 → 18,836 |
| 2k passages, 16 repeated terms | 9.894 | 9.884 | 9.265 | 22,676 → 14,708 |

Range validation now borrows an iterator of ranges rather than collecting and
cloning a temporary vector for every document. This removes exactly one
allocation per document; the small timing difference is less conclusive.

Batch lexical preparation keeps one `HashMap<&str, u32>` across passages.
`drain().collect()` retains the table's capacity and stores each passage's
counts in a compact vector until the atomic batch can be committed. The
single-document lexical API keeps its existing owned-map path. Combined gains
are about 3%, 42%, and 6% for these workloads. The counter alone is about 42%
and 6% faster on the two long-passage cases relative to borrowed ranges.
Tests compare batch scores against independently prepared passages, including
repeated terms, Unicode, and changing vocabulary sizes.

Rejected experiment: sorting all tokens before counting reduced distinct-term
ingestion from 27.525 to 20.081 ms, but increased repetitive ingestion from
9.988 to 16.704 ms (67% slower). That implementation is not retained.

Reproduction: `cargo run --release --offline --example benchmark_pipeline --
--only ingest --rounds 5 --reserve`. Baseline and variants must be built from
their respective source revisions; the same extended harness was used for all.

## batch embedding and retrieval

`embed_documents` and `embed_queries` now share a bounded batching helper. For
more than 32 inputs it sorts **indices** by UTF-8 byte length, a cheap proxy for
token count, so the model pads fewer short inputs to the longest sequence.
Vectors are moved back to their original slots; texts and vectors are not
cloned. All inputs are validated before inference. This is a heuristic:
byte length and token count can diverge, especially across languages, so gains
depend on the input distribution.

```rust
order.sort_unstable_by_key(|&index| texts[index].len());
// After embedding each group of at most 32:
for (&index, vector) in batch.iter().zip(batch_vectors) {
    vectors[index] = vector;
}
```

The real cached BGE model was measured with 64 interleaved inputs of 4, 16, 64,
and 192 repeated words, plus a uniform 32-word control. A reference path calls
the API on original-order slices of 32, which bypasses sorting. Output components
were checked in input order against that reference; maximum error was below
`1.5e-7`, well within the existing `1e-4` tolerance. These are numerical
comparisons, not a new corpus-level relevance evaluation.

| embedding workload | input-order ms | grouped ms | change |
| --- | ---: | ---: | ---: |
| mixed documents | 711.815 | 390.251 | 45% faster |
| mixed queries | 734.010 | 417.160 | 43% faster |
| uniform documents | 110.416 | 110.467 | effectively unchanged |
| uniform queries | 133.125 | 131.578 | effectively unchanged |

Three processes, three interleaved timed rounds per variant in each process;
model loading, fixture creation, and numerical checks are outside timing.

`RetrievalCollection::search_many_with_vectors` adds opt-in scoped workers,
with one `SearchScratch` per worker and results collected in input order.
`Collection::search_many_with_workers` embeds the batch first, then uses those
workers. The existing `search_many` remains sequential for retrieval. Worker
count is capped at batch size and available CPU parallelism. This includes
thread creation/join costs and avoids a new runtime dependency or persistent
thread pool. Concurrent callers should choose their worker limits accordingly.

| 64 queries over 10k vectors | 1 worker ms | 2 workers ms | 4 workers ms | 8 workers ms |
| --- | ---: | ---: | ---: | ---: |
| exact | 34.403 | 18.236 | 10.072 | 7.099 |
| IVF, 4 probes | 5.637 | 2.968 | 1.552 | 1.180 |

Eight workers give about 4.8× the throughput here. Per-batch allocation requests
increase from about 254 KB to 1.80 MB because each worker owns its scratch;
thread stacks and model memory are not included in these allocator counts.
Tests cover exact/IVF rankings, empty batches, zero limits, oversized worker
requests, validation errors, and deterministic first-error selection.

`search_into` also lets a caller retain the final hit buffer:

```rust
let mut scratch = SearchScratch::default();
let mut hits = Vec::new();
index.search_into(query, vector, options, &mut scratch, &mut hits)?;
```

On these warmed ASCII queries it removes all remaining hot-path allocation:
64 allocations / 30,720 requested bytes become zero. Exact timings were
34.677 → 34.531 ms and IVF 5.689 → 5.661 ms, too small to claim a latency win.
Longer queries, Unicode normalization, or changing limits may grow other
workspace buffers. `hits` is cleared on validation errors, so stale results are
not returned. Existing APIs still return owned result vectors.

## combined embedding and retrieval

The same embedding fixture was followed by 64 exact hybrid searches over 10k
synthetic vectors. Model loading and index construction were outside timing;
embedding, input pairing, worker startup, and retrieval were inside. All four
combinations were measured, with iteration order reversed on alternating rounds.

| workload | original batches, 1 worker ms | grouped, 1 worker ms | original batches, 8 workers ms | grouped, 8 workers ms |
| --- | ---: | ---: | ---: | ---: |
| mixed lengths | 800.437 | 467.481 | 760.195 | 427.447 |
| uniform lengths | 168.332 | 168.932 | 139.648 | 140.382 |

Combined improvement: about **47%** for mixed lengths and **17%** for uniform
lengths. Inference dominates this workload, so retrieval's 4.8× speedup does not
translate into a 4.8× end-to-end speedup. Timings vary with CPU scheduling and
thermal state; these measurements run variants in the same process to reduce
that noise.

```sh
cargo run --release --offline --example benchmark_pipeline -- --only batch --rounds 5 --reserve
cargo run --release --offline --example benchmark_embedding_batches -- 3
```

Raw results: `target/profile-pass/final-batch-*.jsonl` and
`target/profile-pass/embedding-combined-*.jsonl`.

## persistence size preflight

A bounded `std::io::Write` adapter prevented oversized allocations but slowed
ordinary vector-heavy encoding substantially: roughly 0.009–0.027 ms became
0.085–0.103 ms for 64 passages. A direct Postcard output adapter and a variant
checking limits only on buffer growth were also slower. Those implementations
are rejected; raw probes remain in `target/profile-pass/encoding-*.log`.

The retained implementation computes the serialized size before writing and
reserves the necessary output capacity once. Its private callers are owned
headers and prepared documents with deterministic derived serializers; no
custom serialization API is exposed. It keeps the existing Postcard byte
format, size limits, sync behavior, and failure/publication rules.

```rust
let size = postcard::experimental::serialized_size(value)?;
if size > limit {
    return Err(StorageError::RecordTooLarge { max_bytes: limit });
}
if size > bytes.capacity() {
    let capacity = size.max(bytes.capacity().saturating_mul(2)).min(limit);
    bytes.reserve_exact(capacity); // bytes was cleared before the size pass
}
postcard::to_io(value, &mut *bytes)?;
```

| encoding probe | original | size preflight |
| --- | ---: | ---: |
| 64 passages × 384 floats, warmed encoding | 8.353 µs | 8.510 µs |
| same record, output buffer capacity | 147,456 bytes | 99,713 bytes |
| rejected 72 MiB string, warmed encoding | 1.42 ms | <0.001 ms |
| rejected string, output buffer capacity | 75,497,476 bytes | 0 bytes |

Normal encoding time is effectively unchanged; the main benefit is avoiding
allocation and copying of an oversized encoded record. The input documents and
lexical preparation still have their own storage. Tiny-string timing is below
one microsecond and too variable for a useful speedup claim. Rejection is
measured directly with a smaller generic fixture rather than running expensive
lexical preparation on a 72 MiB document. Tests cover exact limits, zero limits,
reusing a buffer after rejection, serialization errors, and wire-format
identity; the existing truncation, checksum, and durability tests also pass.

Normal document encoding uses three processes with five measured rounds and
1,000 operations per round. The oversized probe uses five baseline copies per
round and one million preflight rejections per round to resolve the much
shorter operation. Inputs/results pass through `black_box`; construction and
initial buffer growth are outside the reported warmed medians. The two paths
are in the same test binary. The source includes this opt-in release probe:

```sh
cargo test --release --offline --lib benchmark_record_encoding_limits -- --ignored --nocapture
```

Final interleaved storage check, separating ingestion changes from preflight:

| workload | baseline ms | ingestion only ms | ingestion + preflight ms |
| --- | ---: | ---: | ---: |
| durable append, 4,096 passages / 64 transactions | 261.769 | 260.571 | 261.459 |
| warm reopen, 4,096 passages | 9.472 | 8.730 | 8.935 |

Append timings are indistinguishable at this scale. Reopen varied enough
between trials and earlier runs that no speedup is claimed. Per-append
allocation requests fall from 21,703 to 17,662 across the complete pass.
The earlier bounded-writer run was about 5% slower on append and was not kept.
Raw final results: `target/profile-pass/{base,reuse,preflight}-storage-*.jsonl`.

## remaining bottlenecks and validation

The evidence points to vector scans, model inference, and sync waits as the
remaining major costs. Further single-query work should start with a dedicated
vector-kernel profile and recall/latency measurements for IVF settings. More
scratch-buffer tuning is unlikely to change exact-query latency much. Larger
write batches can amortize sync cost through the existing batch API; this pass
does not weaken durability or change write acknowledgement.

The complete pass also reran the original query mix: exact 33.531 → 33.484 ms
and IVF 4.764 → 4.795 ms for 64 queries, effectively unchanged. A later combined
ingestion check gave 5.281 → 5.089 ms (short), 28.838 → 16.389 ms (distinct),
and 10.315 → 9.660 ms (repeated), consistent with the isolated experiments.
These rows were measured independently of the embedding and storage runs.

Validation: 102 ordinary tests, seven cached real-model tests, the opt-in
serialization probe, `cargo fmt --check`, and Clippy with `-D warnings` pass.
Batch benchmark setup independently checks exact and IVF results against
single-query retrieval; real-model probes check embedding alignment and float
error across batch regrouping. CPU timing, allocator probes, and sampling were
run separately. Existing README edits and the earlier untracked performance
report were left untouched.
