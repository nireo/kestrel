# kestrel

A fast hybrid search engine to understand different vector search and other search engine techniques.
Loosely inspired by turbopuffer.

## Exact vector search

`VectorIndex` accepts fixed-size `f32` vectors associated with the same `DocId`
values used by `InvertedIndex`. Each ID can be inserted once, in any order.
All normalized vectors live in one contiguous array; a separate ID array maps
rows to documents, and a hash map supports lookup by document ID.

For two dimensions, the storage looks like this:

```text
doc_ids: [10,       20,        30]
vectors: [1.0, 0.0, 0.6, 0.8, 0.0, 1.0]
          row 0     row 1      row 2
```

Insertion converts each vector to unit length:

```text
length = sqrt(sum(component * component))
normalized_component = component / length

[3, 4] -> length 5 -> [0.6, 0.8]
```

Search normalizes the query once, visits every vector, and computes a dot product:

```text
score = sum(document_component * query_component)
```

For unit vectors this approximates cosine similarity: 1 means the same direction,
0 means perpendicular, and -1 means opposite directions. Scores are clamped to
[-1, 1] to handle floating-point rounding. Negative scores remain eligible.
The best `k` results are selected with the same bounded heap as BM25; ties prefer
lower document IDs. Scores have different meanings from BM25 relevance scores.

Vector storage costs `4 * dimensions` bytes per document, plus ID/lookup overhead
and spare capacity. For `N` vectors of `D` dimensions, scoring takes `O(N * D)`
work, heap selection takes `O(N * log(k + 1))`, and sorting the retained hits takes
`O(k * log(k + 1))`, where `k` is capped at `N`. Query scratch space is `O(D + k)`.
This exhaustive scan provides a baseline for later approximate search.

Zero dimensions, dimension mismatches, zero vectors, NaN/infinity, and duplicate
IDs return typed `VectorError` errors. Norms and dot products are calculated in
`f64` while vectors are stored as `f32`; this also handles very large and tiny
finite input components. Validation errors leave the index unchanged.
Queries are validated even with an empty index or a zero result limit.

```sh
cargo run --example vector_search
cargo test --all-targets
```

The example uses manually chosen vectors with scores 1, 0.6, 0, and -1 for query
`[1, 0]`. No embedding model is involved yet. Tests also compare top-k results
against an independent cosine calculation followed by sorting all results.
