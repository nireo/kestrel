use crate::vector::{normalize, validate_normalized};
use crate::{
    DocId, IvfBuildOptions, IvfError, IvfIndex, SearchHit, VectorError, VectorIndex, top_k,
};

/// Collection vector retrieval. Changing only probes does not retrain IVF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorSearchMode {
    #[default]
    Exact,
    Ivf {
        build: IvfBuildOptions,
        probes: usize,
    },
}

/// Each chunk's vector lives in exactly one store. IVF writes append to buffer.
#[derive(Debug)]
pub(crate) enum CollectionVectors {
    Exact(VectorIndex),
    Ivf {
        index: IvfIndex,
        buffer: VectorIndex,
        build: IvfBuildOptions,
        probes: usize,
    },
}

impl CollectionVectors {
    pub(crate) fn new(dimensions: usize) -> Result<Self, VectorError> {
        Ok(Self::Exact(VectorIndex::new(dimensions)?))
    }

    pub(crate) fn dimensions(&self) -> usize {
        match self {
            Self::Exact(index) => index.dimensions(),
            Self::Ivf { index, .. } => index.dimensions(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Exact(index) => index.len(),
            Self::Ivf { index, buffer, .. } => index.len() + buffer.len(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn mode(&self) -> VectorSearchMode {
        match self {
            Self::Exact(_) => VectorSearchMode::Exact,
            Self::Ivf { build, probes, .. } => VectorSearchMode::Ivf {
                build: *build,
                probes: *probes,
            },
        }
    }

    pub(crate) fn buffered_vector_count(&self) -> usize {
        match self {
            Self::Exact(_) => 0,
            Self::Ivf { buffer, .. } => buffer.len(),
        }
    }

    pub(crate) fn vector(&self, id: DocId) -> Option<&[f32]> {
        match self {
            Self::Exact(index) => index.vector(id),
            Self::Ivf { index, buffer, .. } => index.vector(id).or_else(|| buffer.vector(id)),
        }
    }

    pub(crate) fn normalize(&self, vector: &[f32]) -> Result<Vec<f32>, VectorError> {
        normalize(vector, self.dimensions())
    }

    pub(crate) fn validate_normalized(&self, vector: &[f32]) -> Result<(), VectorError> {
        validate_normalized(vector, self.dimensions())
    }

    pub(crate) fn append_normalized(&mut self, id: DocId, vector: &[f32]) {
        debug_assert!(self.vector(id).is_none());
        match self {
            Self::Exact(index) => index.append_normalized(id, vector),
            Self::Ivf { buffer, .. } => buffer.append_normalized(id, vector),
        }
    }

    pub(crate) fn search(
        &self,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchHit>, VectorError> {
        // Normalize once, then use exactly the same query for base and buffer.
        let query = self.normalize(query)?;
        if limit == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        Ok(match self {
            Self::Exact(index) => index.search_normalized(&query, limit),
            Self::Ivf {
                index,
                buffer,
                probes,
                ..
            } => {
                if index.is_empty() {
                    return Ok(buffer.search_normalized(&query, limit));
                }
                let base_hits = index.search_normalized(&query, limit, *probes);
                if buffer.is_empty() {
                    return Ok(base_hits);
                }
                let buffer_hits = buffer.search_normalized(&query, limit);
                // Merge by cosine score into ONE vector ranking before RRF.
                // Top-k from each disjoint store is sufficient for global top-k.
                top_k(base_hits.into_iter().chain(buffer_hits), limit)
            }
        })
    }

    pub(crate) fn set_mode(&mut self, mode: VectorSearchMode) -> Result<(), IvfError> {
        match mode {
            VectorSearchMode::Exact => {
                if matches!(self, Self::Exact(_)) {
                    return Ok(());
                }

                let mut exact = VectorIndex::new(self.dimensions())?;
                for row in 0..self.len() {
                    let id = row as DocId;
                    exact.append_normalized(
                        id,
                        self.vector(id).expect("collection chunk IDs are dense"),
                    );
                }
                *self = Self::Exact(exact);
            }
            VectorSearchMode::Ivf { build, probes } => {
                build.validate()?;
                if probes == 0 {
                    return Err(IvfError::ZeroProbes);
                }

                if let Self::Ivf {
                    build: current,
                    probes: current_probes,
                    ..
                } = self
                    && *current == build
                {
                    *current_probes = probes;
                    return Ok(());
                }
                self.build_ivf(build, probes)?;
            }
        }
        Ok(())
    }

    pub(crate) fn rebuild(&mut self) -> Result<(), IvfError> {
        if let VectorSearchMode::Ivf { build, probes } = self.mode() {
            self.build_ivf(build, probes)?;
        }
        Ok(())
    }

    fn build_ivf(&mut self, build: IvfBuildOptions, probes: usize) -> Result<(), IvfError> {
        // Stable chunk-ID order keeps rebuilds deterministic despite IVF's layout.
        let index = IvfIndex::build_from_rows(
            self.dimensions(),
            self.len(),
            |row| {
                let id = row as DocId;
                (id, self.vector(id).expect("collection chunk IDs are dense"))
            },
            build,
        )?;
        let buffer = VectorIndex::new(self.dimensions())?;
        // Replace only after success; the old index/buffer remain intact on error.
        *self = Self::Ivf {
            index,
            buffer,
            build,
            probes,
        };
        Ok(())
    }
}
