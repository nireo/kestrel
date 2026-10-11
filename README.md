# kestrel

A hybrid search engine written in Rust, loosely inspired by turbopuffer.

Kestrel combines keyword matching and semantic search to find relevant passages
in document collections. Documents are automatically split into overlapping
chunks and embedded locally, with search results linked back to their original
source text. Collections can be saved as snapshots and restored without
generating embeddings again.

## Features

- BM25 keyword search with an inverted index.
- Exact vector search using cosine similarity.
- Hybrid ranking with reciprocal rank fusion (RRF).
- Local text embeddings with BGE-small-en-v1.5 through FastEmbed.
- Document collections with configurable token-based chunking and overlap.
- Passage-level results linked to their source documents.
- Versioned snapshots for saving and restoring collections.
- Relevance evaluation with Recall@5 and MRR.
