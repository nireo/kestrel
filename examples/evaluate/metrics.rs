use std::collections::HashSet;

use kestrel::DocId;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Metrics {
    pub recall_at_k: f64,
    pub reciprocal_rank: f64,
}

impl Metrics {
    /// Macro average: each query contributes equally, regardless of label count.
    pub fn mean(scores: &[Self]) -> Self {
        assert!(!scores.is_empty(), "need at least one query to average");
        let count = scores.len() as f64;
        Self {
            recall_at_k: scores.iter().map(|score| score.recall_at_k).sum::<f64>() / count,
            reciprocal_rank: scores
                .iter()
                .map(|score| score.reciprocal_rank)
                .sum::<f64>()
                / count,
        }
    }
}

/// Recall counts unique relevant IDs in the first k slots. Reciprocal rank
/// uses the first relevant hit anywhere in the returned ranking; no hit is zero.
pub fn score(ranking: &[DocId], relevant: &HashSet<DocId>, k: usize) -> Metrics {
    assert!(!relevant.is_empty(), "a query needs relevance labels");
    let retrieved_relevant: HashSet<_> = ranking
        .iter()
        .take(k)
        .filter(|id| relevant.contains(id))
        .copied()
        .collect();
    Metrics {
        recall_at_k: retrieved_relevant.len() as f64 / relevant.len() as f64,
        reciprocal_rank: ranking
            .iter()
            .position(|id| relevant.contains(id))
            .map_or(0.0, |position| 1.0 / (position + 1) as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_calculated_multiple_relevant_documents() {
        let relevant = HashSet::from([1, 4, 9]);
        let result = score(&[2, 4, 1, 7, 8, 9], &relevant, 5);
        assert_eq!(result.recall_at_k, 2.0 / 3.0);
        assert_eq!(result.reciprocal_rank, 0.5);
    }

    #[test]
    fn reciprocal_rank_includes_hits_beyond_recall_cutoff() {
        assert_eq!(
            score(&[0, 1, 2, 3, 4, 5], &HashSet::from([5]), 5),
            Metrics {
                recall_at_k: 0.0,
                reciprocal_rank: 1.0 / 6.0,
            }
        );
    }

    #[test]
    fn missing_hits_empty_results_zero_cutoff_and_duplicates() {
        let relevant = HashSet::from([1, 9]);
        assert_eq!(score(&[], &relevant, 5), Metrics::default());
        assert_eq!(score(&[2, 3], &relevant, 5), Metrics::default());
        assert_eq!(
            score(&[1], &relevant, 0),
            Metrics {
                recall_at_k: 0.0,
                reciprocal_rank: 1.0
            }
        );
        assert_eq!(score(&[1, 1, 1], &relevant, 5).recall_at_k, 0.5);
        assert_eq!(score(&[1, 9], &relevant, 5).recall_at_k, 1.0);
    }

    #[test]
    fn average_weights_queries_equally_instead_of_pooling_labels() {
        let first = score(&[1], &HashSet::from([1]), 5);
        let second = score(&[2, 1], &HashSet::from([1, 9]), 5);
        assert_eq!(
            Metrics::mean(&[first, second]),
            Metrics {
                recall_at_k: 0.75,
                reciprocal_rank: 0.75,
            }
        );
    }
}
