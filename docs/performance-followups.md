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

## 2. normalize owned embeddings in place

Prepared chunks validate the full vector, then divide their existing components
by the f64 norm. Direct VectorIndex insertion similarly extends the destination
from normalized components, avoiding a temporary Vec. Invalid inputs are checked
before modifying either the input buffer or the index. Arithmetic and stored
float bits match the original normalization.

```rust
let length = vector_length(vector, dimensions)?;
for component in vector {
    *component = (f64::from(*component) / length) as f32;
}
```

Measured alone against the original baseline: vector construction fell from
36.158 to 33.941 ms (1.07×), and allocations fell from 100,034 to 34. Collection
insertion fell from 5.890 to 5.761 ms; allocations fell from 61,263 to 51,263,
removing 15,360,000 bytes of cumulative allocation traffic. Together with
preallocation, vector construction took 28.053 ms with two allocations and
154,000,000 live bytes; collection insertion took 5.066 ms. These standalone and
combined comparisons use separate source snapshots, not just successive timings
of the cumulative branch.

## 3. reuse the complete query workspace

`SearchScratch` retains the normalized query, BM25/vector candidate arrays, IVF
routing candidates, fusion candidates, and the bounded selection heap. The new
`RetrievalCollection::search_with_scratch` returns an independent result Vec;
`Collection::search_many` also uses this workspace. Existing search APIs retain
their behavior. Invalid queries preserve reusable buffers without exposing stale
results. Tests cover every retrieval mode, insertion buffers, cross-index reuse,
limits, errors, held results, and concurrent independent readers.

```rust
normalize_into(query, self.dimensions(), &mut normalized)?;
// select into retained storage instead of creating an owned temporary Vec
index.search_normalized_into(&normalized, limit,
    &mut scratch.heap, &mut scratch.vector_hits);
```

Measured alone: allocations across 64 exact queries fell from 368 to 64 and
requested bytes from 447,488 to 30,720. IVF allocations fell from 688 to 64 and
bytes from 527,360 to 30,720. Each query now allocates only its returned result.
Exact query latency was unchanged (33.288 vs 33.295 ms per 64 queries). IVF took
4.640 vs 4.520 ms; this is a modest timing difference, not a large speedup.
Combined with the first two changes, exact queries took 32.410 ms and IVF took
4.566 ms, with the same allocation reductions.

## 4. use dense IVF row positions

An IVF build with unique IDs in 0..count stores destination rows in a Vec<usize>.
This accepts any insertion permutation and preserves cluster reordering. Sparse
IDs keep their existing HashMap and never allocate up to an arbitrary maximum ID.

```rust
match self {
    Self::Dense(rows) => rows.get(id as usize).copied(),
    Self::Sparse(rows) => rows.get(&id).copied(),
}
```

Measured alone: one million lookups fell from 6.792 to 1.451 ms (4.68×). IVF
construction was effectively unchanged (16.737 vs 16.674 ms). A 10,000-row index
retained 198,536 fewer bytes (about 194 KiB). With the first four changes combined,
lookups took 1.477 ms and build time was 16.954 ms. This optimizes access by ID;
it does not reduce the number of vectors scored during an IVF query.

## 5. reuse bounded persistence payload buffers

A PersistentCollection keeps one serialization buffer. Recovery reads each frame
into that same buffer, decodes owned documents, and clears it before publishing.
Postcard writes into the existing Vec through its std::io::Write serializer;
tests compare the resulting bytes with the original serializer. Frame bytes,
checksums, size limits, atomicity, crash recovery, and sync frequency are unchanged.
Buffers with capacity above 1 MiB are released after use, including on errors.
This avoids retaining the maximum 64 MiB payload on every writer.

```rust
bytes.clear();
postcard::to_io(value, &mut *bytes)?;
// after reading/encoding a record
bytes.clear();
if bytes.capacity() > MAX_RETAINED_BYTES {
    *bytes = Vec::new();
}
```

Measured alone over 4,096 rows / 64 batches: append allocations fell from 26,708
to 25,821 and cumulative requested bytes from 35,325,159 to 21,792,948 (38% less).
Recovery allocations fell from 34,075 to 34,013 and requested bytes from
28,343,389 to 22,227,066 (22% less). Append time was 261.951 vs 263.039 ms;
reopening was 8.470 vs 8.502 ms. Neither is a meaningful latency improvement.
Ordinary reusable buffers do retain additional memory: about 160–200 KiB in
these cases, bounded by the 1 MiB retention limit.

## all five combined

A fresh three-run comparison against the original baseline used the same harness,
inputs, release configuration, and sync contract. Times below are milliseconds.

| case | baseline | combined | interpretation |
| --- | ---: | ---: | --- |
| construct 100k vectors | 36.483 | 28.408 | 1.28× faster |
| ingest 10k documents | 5.826 | 5.036 | 1.16× faster |
| 64 exact hybrid queries | 33.129 | 33.029 | effectively unchanged |
| 64 IVF hybrid queries | 4.673 | 4.605 | small difference |
| build IVF over 10k vectors | 16.687 | 16.688 | effectively unchanged |
| 1m IVF ID lookups | 6.787 | 1.444 | 4.70× faster |
| durably append 4,096 documents | 262.543 | 260.294 | effectively unchanged |
| reopen 4,096 documents | 8.139 | 8.212 | effectively unchanged |

The combined 100k-vector construction uses two allocations and retains
154,000,000 bytes versus 100,034 allocations and 201,850,880 bytes originally.
Collection ingestion allocations fall from 61,263 to 51,201. Exact and IVF query
allocations are both 64 for 64 queries, versus 368 and 688. Durable append
allocation traffic falls to 15,401,364 bytes; recovery falls to 22,126,938 bytes.
No unmeasured large latency gain is claimed for buffer reuse or scratch storage.

Validation: formatting and warning-free Clippy pass, along with 99 regular tests
and seven opt-in real-model tests. Existing storage corruption, truncation,
poisoning, process-exit durability, vector-bit preservation, and exact/IVF
consistency tests pass. New tests exercise reservation, normalization failures,
full-workspace reuse and concurrent readers, dense/sparse IVF permutations, wire
compatibility, and bounded persistence buffer retention.

Per-case medians and allocation results are in
`target/performance-followups/summary.json`. The folder also contains raw JSONL,
source snapshot archives for the isolated and combined builds, patches, the
snapshot runner, build logs, and validation logs. The initial preallocation
comparison above predates the fresh combined comparison, so its baseline timings
differ slightly from the final table.
