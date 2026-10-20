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

Captures: `target/profile-pass/baseline-{query,ingest,storage}.sample.txt`.
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
