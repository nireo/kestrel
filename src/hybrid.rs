use std::collections::HashMap;

use crate::{DocId, InvertedIndex, IvfError, IvfIndex, SearchHit, VectorError, VectorIndex, top_k};

const RRF_K: f64 = 60.0;

/// A fused result, including its one-based rank in each candidate list.
/// A missing rank means that source did not retrieve the document at the
/// requested candidate depth. The score is an RRF weight, not a probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridHit {
    pub doc_id: DocId,
    pub score: f64,
    pub bm25_rank: Option<usize>,
    pub vector_rank: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridSearchOptions {
    /// Maximum candidates retrieved from EACH index before fusion.
    pub candidate_depth: usize,
    /// Maximum fused results returned. Independent of candidate_depth.
    pub limit: usize,
}

impl Default for HybridSearchOptions {
    fn default() -> Self {
        Self {
            candidate_depth: 50,
            limit: 10,
        }
    }
}

/// Combine best-first BM25 and vector result lists using reciprocal rank fusion.
///
/// Each source contributes `1 / (60 + rank)`, with ranks starting at one.
/// Input order determines rank; input scores are ignored. Documents absent from
/// a source receive no contribution from it. Duplicate IDs within a source count
/// once, at their first position; subsequent positions are not renumbered.
/// Results are unique, sorted by descending fused score and then ascending ID.
///
/// If C candidates are supplied and U IDs are unique, expected work is
/// O(C + U log(k + 1) + k log(k + 1)), with O(U + k) scratch space and k capped at U.
pub fn reciprocal_rank_fusion(
    bm25_hits: &[SearchHit],
    vector_hits: &[SearchHit],
    limit: usize,
) -> Vec<HybridHit> {
    if limit == 0 {
        return Vec::new();
    }

    let mut candidates = HashMap::<DocId, HybridHit>::new();
    for (hits, is_bm25) in [(bm25_hits, true), (vector_hits, false)] {
        for (position, hit) in hits.iter().enumerate() {
            let candidate = candidates.entry(hit.doc_id).or_insert(HybridHit {
                doc_id: hit.doc_id,
                score: 0.0,
                bm25_rank: None,
                vector_rank: None,
            });

            let source_rank = if is_bm25 {
                &mut candidate.bm25_rank
            } else {
                &mut candidate.vector_rank
            };

            if source_rank.is_none() {
                let rank = position + 1;
                *source_rank = Some(rank);
                candidate.score += 1.0 / (RRF_K + rank as f64);
            }
        }
    }

    // Reuse the same bounded heap and tie ordering as the individual searches.
    let best = top_k(
        candidates.values().map(|hit| SearchHit {
            doc_id: hit.doc_id,
            score: hit.score,
        }),
        limit,
    );

    best.into_iter()
        .map(|hit| candidates[&hit.doc_id])
        .collect()
}

/// Retrieve candidates independently, then fuse them into one ranked list.
///
/// Both indexes must use the same document ID namespace. Vectors can cover a
/// subset of documents. The caller supplies both query text and its vector;
/// this function does not generate embeddings or verify the query's meaning.
/// Invalid query vectors return VectorError, even when depth or limit is zero.
/// The cosine search's zero and negative scores remain eligible for fusion.
pub fn hybrid_search(
    bm25: &InvertedIndex,
    vectors: &VectorIndex,
    query: &str,
    query_vector: &[f32],
    options: HybridSearchOptions,
) -> Result<Vec<HybridHit>, VectorError> {
    let depth = if options.limit == 0 {
        0
    } else {
        options.candidate_depth
    };

    // Validate the vector query before spending work on lexical retrieval.
    let vector_hits = vectors.search(query_vector, depth)?;
    let bm25_hits = bm25.search(query, depth);
    Ok(reciprocal_rank_fusion(
        &bm25_hits,
        &vector_hits,
        options.limit,
    ))
}

/// Hybrid search using an immutable IVF index and an explicit cluster probe count.
/// ID alignment, candidate depth, fusion and tie ordering match hybrid_search().
/// Vector approximation can change the candidate list and fused results.
pub fn hybrid_search_ivf(
    bm25: &InvertedIndex,
    vectors: &IvfIndex,
    query: &str,
    query_vector: &[f32],
    options: HybridSearchOptions,
    probes: usize,
) -> Result<Vec<HybridHit>, IvfError> {
    let depth = if options.limit == 0 {
        0
    } else {
        options.candidate_depth
    };
    let vector_hits = vectors.search(query_vector, depth, probes)?;
    let bm25_hits = bm25.search(query, depth);
    Ok(reciprocal_rank_fusion(
        &bm25_hits,
        &vector_hits,
        options.limit,
    ))
}
