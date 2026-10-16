use std::collections::HashMap;

use crate::vector::{dot, normalize};
use crate::{DocId, SearchHit, VectorError, VectorIndex, top_k};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfBuildOptions {
    /// Maximum number of clusters; capped at the number of indexed vectors.
    pub clusters: usize,

    /// Maximum training vectors, sampled deterministically without replacement.
    pub training_samples: usize,

    /// Maximum spherical k-means iterations; stops early on stable assignments.
    pub iterations: usize,
    pub seed: u64,
}

impl Default for IvfBuildOptions {
    fn default() -> Self {
        Self {
            clusters: 128,
            training_samples: 4096,
            iterations: 12,
            seed: 42,
        }
    }
}

impl IvfBuildOptions {
    pub(crate) fn validate(self) -> Result<(), IvfError> {
        if self.clusters == 0 || self.iterations == 0 || self.training_samples < self.clusters {
            return Err(IvfError::InvalidBuildOptions);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum IvfError {
    #[error("clusters and iterations must be positive, and training_samples must be >= clusters")]
    InvalidBuildOptions,
    #[error("the number of probed clusters must be positive")]
    ZeroProbes,
    #[error(transparent)]
    Vector(#[from] VectorError),
}

/// Immutable IVF-flat cosine index: spherical k-means routes queries to clusters,
/// then full-precision vectors in those clusters are scored exactly.
///
/// Build copies normalized vectors into cluster order, preserving external IDs
/// and float values. It owns its data; the source may be dropped after building.
/// Updates require a rebuild. The trained index is rebuilt after reopening persistent collections.
#[derive(Debug)]
pub struct IvfIndex {
    dimensions: usize,
    centroids: Vec<f32>,
    // Cluster c owns rows offsets[c]..offsets[c + 1] in these contiguous arrays.
    offsets: Vec<usize>,
    vectors: Vec<f32>,
    doc_ids: Vec<DocId>,
    positions: HashMap<DocId, usize>,
}

impl IvfIndex {
    pub fn build(source: &VectorIndex, options: IvfBuildOptions) -> Result<Self, IvfError> {
        Self::build_from_rows(
            source.dimensions(),
            source.len(),
            |row| source.row(row),
            options,
        )
    }

    /// Borrow rows from existing storage so rebuilding base + buffer needs no
    /// temporary full-sized exact index. Rows must have unique IDs and unit vectors.
    pub(crate) fn build_from_rows<'a>(
        dimensions: usize,
        count: usize,
        row_at: impl Fn(usize) -> (DocId, &'a [f32]),
        options: IvfBuildOptions,
    ) -> Result<Self, IvfError> {
        options.validate()?;
        let clusters = options.clusters.min(count);
        if count == 0 {
            return Ok(Self {
                dimensions,
                centroids: Vec::new(),
                offsets: vec![0],
                vectors: Vec::new(),
                doc_ids: Vec::new(),
                positions: HashMap::new(),
            });
        }

        let sample = sample_rows(count, options.training_samples, options.seed);
        let centroids = train(&row_at, dimensions, &sample, clusters, options.iterations);

        let assignments: Vec<_> = (0..count)
            .map(|row| nearest(row_at(row).1, &centroids, dimensions))
            .collect();
        let mut offsets = vec![0; clusters + 1];
        for &cluster in &assignments {
            offsets[cluster + 1] += 1;
        }

        for cluster in 0..clusters {
            offsets[cluster + 1] += offsets[cluster];
        }

        let mut cursors = offsets[..clusters].to_vec();
        let mut vectors = vec![0.0; count * dimensions];
        let mut doc_ids = vec![0; count];
        let mut positions = HashMap::with_capacity(count);
        for (row, cluster) in assignments.into_iter().enumerate() {
            let (id, vector) = row_at(row);
            let destination = cursors[cluster];
            cursors[cluster] += 1;
            doc_ids[destination] = id;
            positions.insert(id, destination);
            vectors[destination * dimensions..(destination + 1) * dimensions]
                .copy_from_slice(vector);
        }

        Ok(Self {
            dimensions,
            centroids,
            offsets,
            vectors,
            doc_ids,
            positions,
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }
    pub fn cluster_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Borrow an original normalized vector by its stable document ID.
    pub fn vector(&self, doc_id: DocId) -> Option<&[f32]> {
        let row = *self.positions.get(&doc_id)?;
        let start = row * self.dimensions;
        Some(&self.vectors[start..start + self.dimensions])
    }

    /// Includes empty clusters, which can occur for duplicate/degenerate data.
    pub fn cluster_sizes(&self) -> impl Iterator<Item = usize> + '_ {
        self.offsets.windows(2).map(|pair| pair[1] - pair[0])
    }

    /// Search the nearest `probes` centroids; larger values trade speed for recall.
    /// Probes are capped at cluster_count(). Probing every cluster is exact,
    /// including tie ordering and scores. Lower probe counts may return fewer
    /// than `limit` results. Query validation applies even at limit zero.
    pub fn search(
        &self,
        query: &[f32],
        limit: usize,
        probes: usize,
    ) -> Result<Vec<SearchHit>, IvfError> {
        if probes == 0 {
            return Err(IvfError::ZeroProbes);
        }

        let query = normalize(query, self.dimensions)?;
        Ok(self.search_normalized(&query, limit, probes))
    }

    /// The caller validates/normalizes once when searching multiple vector stores.
    pub(crate) fn search_normalized(
        &self,
        query: &[f32],
        limit: usize,
        probes: usize,
    ) -> Vec<SearchHit> {
        debug_assert!(probes > 0);
        if limit == 0 || self.is_empty() {
            return Vec::new();
        }

        // All clusters share one contiguous array, so scan it directly.
        if probes >= self.cluster_count() {
            let hits = self
                .vectors
                .chunks_exact(self.dimensions)
                .zip(&self.doc_ids)
                .map(|(vector, &doc_id)| SearchHit {
                    doc_id,
                    score: dot(vector, query).clamp(-1.0, 1.0),
                });
            return top_k(hits, limit);
        }

        let clusters = top_k(
            self.centroids
                .chunks_exact(self.dimensions)
                .enumerate()
                .map(|(cluster, centroid)| SearchHit {
                    doc_id: cluster as DocId,
                    score: dot(centroid, query),
                }),
            probes,
        );

        let hits = clusters.into_iter().flat_map(|hit| {
            let cluster = hit.doc_id as usize;
            let start = self.offsets[cluster];
            let end = self.offsets[cluster + 1];
            self.vectors[start * self.dimensions..end * self.dimensions]
                .chunks_exact(self.dimensions)
                .zip(&self.doc_ids[start..end])
                .map(move |(vector, &doc_id)| SearchHit {
                    doc_id,
                    score: dot(vector, query).clamp(-1.0, 1.0),
                })
        });

        top_k(hits, limit)
    }
}

fn nearest(vector: &[f32], centroids: &[f32], dimensions: usize) -> usize {
    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (cluster, centroid) in centroids.chunks_exact(dimensions).enumerate() {
        let score = dot(vector, centroid);
        if score > best_score {
            best = cluster;
            best_score = score;
        }
    }

    best
}

fn sample_rows(count: usize, maximum: usize, mut state: u64) -> Vec<usize> {
    let mut rows: Vec<_> = (0..count.min(maximum)).collect();

    // Reservoir sampling avoids allocating an N-element shuffle permutation.
    for row in rows.len()..count {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^= value >> 31;
        let slot = (value % (row as u64 + 1)) as usize;
        if slot < rows.len() {
            rows[slot] = row;
        }
    }
    rows.sort_unstable();

    rows
}

fn train<'a>(
    row_at: &impl Fn(usize) -> (DocId, &'a [f32]),
    dimensions: usize,
    sample: &[usize],
    clusters: usize,
    iterations: usize,
) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(clusters * dimensions);
    let mut closest = vec![f64::NEG_INFINITY; sample.len()];
    let mut next = 0;

    // Farthest-first seeds cover distinct directions, including small clusters.
    for _ in 0..clusters {
        let centroid = row_at(sample[next]).1;
        centroids.extend_from_slice(centroid);
        for (&row, similarity) in sample.iter().zip(&mut closest) {
            *similarity = similarity.max(dot(row_at(row).1, centroid));
        }
        next = closest
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
    }

    let mut assignments = vec![usize::MAX; sample.len()];
    let mut sums = vec![0.0_f64; clusters * dimensions];
    let mut counts = vec![0_usize; clusters];

    for _ in 0..iterations {
        sums.fill(0.0);
        counts.fill(0);
        let mut changed = false;
        for (&row, assignment) in sample.iter().zip(&mut assignments) {
            let vector = row_at(row).1;
            let cluster = nearest(vector, &centroids, dimensions);
            changed |= *assignment != cluster;
            *assignment = cluster;
            counts[cluster] += 1;
            for (sum, &component) in sums[cluster * dimensions..(cluster + 1) * dimensions]
                .iter_mut()
                .zip(vector)
            {
                *sum += f64::from(component);
            }
        }

        if !changed {
            break;
        }

        for (cluster, count) in counts.iter().enumerate() {
            let sum = &sums[cluster * dimensions..(cluster + 1) * dimensions];
            let length = sum.iter().map(|v| v * v).sum::<f64>().sqrt();

            // Keep the previous direction for empty clusters or cancelling sums.
            if *count == 0 || length == 0.0 {
                continue;
            }
            for (centroid, &value) in centroids[cluster * dimensions..(cluster + 1) * dimensions]
                .iter_mut()
                .zip(sum)
            {
                *centroid = (value / length) as f32;
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_is_reproducible_without_duplicates_and_not_stride_based() {
        let rows = sample_rows(65536, 1024, 42);
        assert_eq!(rows, sample_rows(65536, 1024, 42));
        assert_eq!(rows.len(), 1024);
        assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(rows.iter().all(|&row| row < 65536));
        assert!(rows.iter().any(|row| row % 64 != 0));
        assert_ne!(rows, sample_rows(65536, 1024, 43));
    }
}
