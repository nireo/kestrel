use std::ops::Range;

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

/// Limits count model tokens, not words. max_tokens includes special tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Maximum encoded passage length, including the model's special tokens.
    pub max_tokens: usize,
    /// Target overlap in the original tokenization. May be reduced when
    /// retokenizing a subword boundary requires a smaller window to advance.
    pub overlap_tokens: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            overlap_tokens: 32,
        }
    }
}

impl ChunkingConfig {
    pub(crate) fn validate(&self, model_limit: usize) -> Result<(), ChunkingError> {
        if self.max_tokens == 0 || self.max_tokens > model_limit {
            return Err(ChunkingError::InvalidMaxTokens {
                max_tokens: self.max_tokens,
                model_limit,
            });
        }
        if self.overlap_tokens >= self.max_tokens {
            return Err(ChunkingError::InvalidOverlap {
                overlap_tokens: self.overlap_tokens,
                content_limit: self.max_tokens,
            });
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ChunkingError {
    #[error(
        "chunk token limit {max_tokens} must be positive and no greater than model limit {model_limit}"
    )]
    InvalidMaxTokens {
        max_tokens: usize,
        model_limit: usize,
    },
    #[error("overlap {overlap_tokens} must be smaller than content token budget {content_limit}")]
    InvalidOverlap {
        overlap_tokens: usize,
        content_limit: usize,
    },
    #[error("chunk token limit leaves no room after {special_tokens} special tokens")]
    NoContentBudget { special_tokens: usize },
    #[error("tokenization failed: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("text has no model tokens")]
    NoTokens,
    #[error("text cannot be split into nonempty UTF-8 passages within the token budget")]
    UnchunkableText,
}

pub(crate) struct Chunker {
    tokenizer: Tokenizer,
    config: ChunkingConfig,
    content_limit: usize,
}

impl Chunker {
    pub(crate) fn new(
        mut tokenizer: Tokenizer,
        config: ChunkingConfig,
        model_limit: usize,
    ) -> Result<Self, ChunkingError> {
        config.validate(model_limit)?;
        tokenizer.with_truncation(None)?;
        tokenizer.with_padding(None);
        let special_tokens = tokenizer.encode("", true)?.len();
        let content_limit = config
            .max_tokens
            .checked_sub(special_tokens)
            .filter(|&limit| limit > 0)
            .ok_or(ChunkingError::NoContentBudget { special_tokens })?;

        if config.overlap_tokens >= content_limit {
            return Err(ChunkingError::InvalidOverlap {
                overlap_tokens: config.overlap_tokens,
                content_limit,
            });
        }

        Ok(Self {
            tokenizer,
            config,
            content_limit,
        })
    }

    pub(crate) fn config(&self) -> ChunkingConfig {
        self.config
    }

    /// Token offsets refer to original UTF-8 bytes, including normalization's
    /// effects. Gaps such as whitespace stay in the slices; text is never decoded
    /// back from token IDs. Each candidate is re-tokenized to catch subword
    /// boundary changes before inference could silently truncate it.
    pub(crate) fn ranges(&self, text: &str) -> Result<Vec<Range<usize>>, ChunkingError> {
        let encoding = self.tokenizer.encode(text, false)?;
        let offsets = encoding.get_offsets();
        if offsets.is_empty() {
            return Err(ChunkingError::NoTokens);
        }

        let mut ranges: Vec<Range<usize>> = Vec::new();
        let mut start = 0;
        while start < offsets.len() {
            let byte_start = if start == 0 { 0 } else { offsets[start].0 };
            let mut end = start.saturating_add(self.content_limit).min(offsets.len());
            let range = loop {
                if end <= start {
                    return Err(ChunkingError::UnchunkableText);
                }
                let byte_end = if end == offsets.len() {
                    text.len()
                } else {
                    offsets[end].0
                };

                let range = byte_start..byte_end;
                if let Some(passage) = text.get(range.clone())
                    && !passage.trim().is_empty()
                    && self.tokenizer.encode(passage, true)?.len() <= self.config.max_tokens
                {
                    break range;
                }
                end -= 1;
            };

            if ranges
                .last()
                .is_none_or(|previous| range.start > previous.start && range.end > previous.end)
            {
                ranges.push(range);
            }

            if end == offsets.len() {
                break;
            }
            // Shrinking a candidate can reduce its overlap, but must always
            // advance at least one token to avoid repeating a window forever.
            start = end - self.config.overlap_tokens.min(end - start - 1);
        }
        Ok(ranges)
    }
}

#[cfg(test)]
mod tests;
