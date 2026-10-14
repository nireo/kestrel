use std::hint::black_box;
use std::time::Instant;

use serde::Serialize;

use super::data::{Error, Query};

#[derive(Debug, Serialize)]
pub struct Timing {
    pub samples: usize,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    /// Reciprocal of mean service time; single-threaded, without client overhead.
    pub sequential_qps: f64,
    pub query_indices: Vec<usize>,
    pub samples_us: Vec<f64>,
}

fn percentile(sorted: &[f64], percent: usize) -> f64 {
    sorted[(sorted.len() * percent).div_ceil(100).saturating_sub(1)]
}

impl Timing {
    fn new(samples_us: Vec<f64>, query_indices: Vec<usize>) -> Self {
        let mut sorted = samples_us.clone();
        sorted.sort_by(f64::total_cmp);
        let mean_us = sorted.iter().sum::<f64>() / sorted.len() as f64;
        Self {
            samples: sorted.len(),
            mean_us,
            p50_us: percentile(&sorted, 50),
            p95_us: percentile(&sorted, 95),
            p99_us: percentile(&sorted, 99),
            sequential_qps: 1_000_000.0 / mean_us,
            query_indices,
            samples_us,
        }
    }
}

/// Same deterministic shuffled order for every method, different per pass.
fn order(count: usize, pass: usize) -> Vec<usize> {
    let mut indices: Vec<_> = (0..count).collect();
    let mut state = (pass as u64).wrapping_add(42);
    for end in (1..count).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        indices.swap(end, (state % (end as u64 + 1)) as usize);
    }
    indices
}

pub fn measure<T>(
    queries: &[Query],
    warmup: usize,
    rounds: usize,
    mut search: impl FnMut(&Query) -> Result<T, Error>,
    mut observe: impl FnMut(usize, &T),
) -> Result<Timing, Error> {
    for pass in 0..warmup {
        for index in order(queries.len(), pass) {
            black_box(search(black_box(&queries[index]))?);
        }
    }
    let mut samples = Vec::with_capacity(queries.len() * rounds);
    let mut query_indices = Vec::with_capacity(samples.capacity());
    for pass in 0..rounds {
        for index in order(queries.len(), pass + warmup) {
            let start = Instant::now();
            let result = search(black_box(&queries[index]));
            let elapsed = start.elapsed();
            let hits = result?;
            black_box(&hits);
            samples.push(elapsed.as_secs_f64() * 1_000_000.0);
            query_indices.push(index);
            // Recording ground truth and destroying hits are outside timing.
            if pass == 0 {
                observe(index, &hits);
            }
        }
    }
    Ok(Timing::new(samples, query_indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentiles_and_throughput_match_hand_calculation() {
        let timing = Timing::new((1..=100).rev().map(f64::from).collect(), vec![]);
        assert_eq!(timing.p50_us, 50.0);
        assert_eq!(timing.p95_us, 95.0);
        assert_eq!(timing.p99_us, 99.0);
        assert_eq!(timing.mean_us, 50.5);
        assert_eq!(timing.sequential_qps, 1_000_000.0 / 50.5);
        assert_eq!(Timing::new(vec![7.0], vec![]).p99_us, 7.0);
    }

    #[test]
    fn query_schedule_is_a_reproducible_permutation() {
        let mut sorted = order(100, 2);
        assert_eq!(sorted, order(100, 2));
        assert_ne!(sorted, order(100, 3));
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
    }
}
