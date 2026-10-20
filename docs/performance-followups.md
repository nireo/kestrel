# allocation and memory follow-ups

Baseline: `930a238`. Measurements use Rust 1.95.0 on macOS arm64, release
defaults, and deterministic 384-dimensional vectors. No embedding model is
loaded. Each case has one warm-up and five measured iterations; inputs are
prepared before timing, and returned indexes are dropped afterward. Allocation
counts are collected in a separate invocation with timing disabled.

`examples/benchmark_pipeline.rs` covers vector construction (100,000 rows),
collection ingestion (10,000 rows in batches of 256), 64 exact and IVF hybrid
queries, IVF construction and 1,000,000 ID lookups, durable insertion (4,096 rows
in batches of 64), and warm reopening of that durable file. Durable writes keep
their existing sync per batch. Query setup checks the workspace results against
the original retrieval API before measuring. `net_heap_bytes` is allocation
minus deallocation during the operation; it includes consumed input buffers for
ingestion, so only vector construction directly measures retained index storage.

```sh
cargo run --release --offline --example benchmark_pipeline -- --reserve
# isolate a stage: --only vector|ingest|query|ivf|storage
```

Before/after binaries are built from isolated source snapshots with distinct
package names. For the original baseline harness, reservation calls are no-ops
and full-workspace query calls use the existing BM25-only scratch API. Reported
times are medians of the per-process medians from three interleaved runs. Raw
samples, allocator measurements, binaries, and runner scripts are saved under
`target/performance-followups/` (gitignored).

## 1. preallocate index storage

`VectorIndex::with_capacity` and `reserve` allocate for a known number of rows.
`RetrievalCollection::reserve` reserves document, chunk, lexical-length, and
vector storage. Batches also reserve their required capacity after validation
and before persistence/publication. Returning from IVF to exact mode reserves
the complete known row count. Small incremental inserts still use amortized
growth rather than repeatedly reallocating exact-sized buffers.

```rust
let components = additional.checked_mul(self.dimensions)
    .expect("vector capacity overflow");
self.vectors.reserve(components);
self.doc_ids.reserve(additional);
```

Measured alone: 100k-vector construction fell from 36.231 to 29.895 ms (1.21×),
with live index allocations falling from 201,850,880 to 154,000,000 bytes: a
47,850,880-byte (45.63 MiB) saving. Collection ingestion fell from 5.808 to
5.117 ms (1.14×). Query latency was effectively unchanged. Durable append time
remained dominated by syncs; no write-latency improvement is claimed.
