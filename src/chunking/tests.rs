use super::*;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;

pub(crate) fn tokenizer() -> Tokenizer {
    let model = WordPiece::builder()
        .vocab([
            ("[UNK]".to_owned(), 0),
            ("[CLS]".to_owned(), 1),
            ("[SEP]".to_owned(), 2),
            ("a".to_owned(), 3),
            ("##bcd".to_owned(), 4),
            ("##ef".to_owned(), 5),
            ("b".to_owned(), 6),
            ("##c".to_owned(), 7),
            ("##d".to_owned(), 8),
            ("e".to_owned(), 9),
            ("##f".to_owned(), 10),
        ])
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer
        .with_normalizer(Some(BertNormalizer::new(true, true, Some(true), true)))
        .unwrap();
    tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
    tokenizer.with_post_processor(Some(BertProcessing::new(
        ("[SEP]".into(), 2),
        ("[CLS]".into(), 1),
    )));
    tokenizer
}

fn chunker(max_tokens: usize, overlap_tokens: usize) -> Chunker {
    Chunker::new(
        tokenizer(),
        ChunkingConfig {
            max_tokens,
            overlap_tokens,
        },
        512,
    )
    .unwrap()
}

fn assert_coverage(text: &str, ranges: &[Range<usize>], chunker: &Chunker) {
    assert_eq!(ranges.first().unwrap().start, 0);
    assert_eq!(ranges.last().unwrap().end, text.len());
    for range in ranges {
        let passage = text.get(range.clone()).unwrap();
        assert!(!passage.trim().is_empty());
        assert!(
            chunker.tokenizer.encode(passage, true).unwrap().len() <= chunker.config.max_tokens
        );
    }
    for adjacent in ranges.windows(2) {
        assert!(adjacent[0].start < adjacent[1].start);
        assert!(adjacent[0].end < adjacent[1].end);
        assert!(adjacent[1].start <= adjacent[0].end);
    }
}

#[test]
fn short_text_keeps_original_whitespace_and_unicode() {
    let text = "  Café yö 世界!\n";
    assert_eq!(chunker(32, 4).ranges(text).unwrap(), vec![0..text.len()]);
}

#[test]
fn model_token_windows_overlap_and_reserve_special_tokens() {
    let text = "one two three four five six seven eight nine ten";
    let chunker = chunker(6, 1);
    let ranges = chunker.ranges(text).unwrap();
    let passages: Vec<_> = ranges
        .iter()
        .map(|range| text[range.clone()].trim())
        .collect();
    assert_eq!(
        passages,
        [
            "one two three four",
            "four five six seven",
            "seven eight nine ten"
        ]
    );
    assert_coverage(text, &ranges, &chunker);
}

#[test]
fn no_overlap_reconstructs_original_bytes_without_gaps() {
    let text = "\nCafé\t世界!  🙂 punctuation...\nYÖ cafe\u{301}  tail\n";
    let chunker = chunker(5, 0);
    let ranges = chunker.ranges(text).unwrap();
    assert_coverage(text, &ranges, &chunker);
    let reconstructed: String = ranges.iter().map(|range| &text[range.clone()]).collect();
    assert_eq!(reconstructed, text);
}

#[test]
fn subword_boundary_retokenization_cannot_exceed_budget() {
    let text = "one abcdef two three";
    let chunker = chunker(5, 1);
    let ranges = chunker.ranges(text).unwrap();
    assert!(ranges.len() > 1);
    assert_coverage(text, &ranges, &chunker);
}

#[test]
fn long_input_is_not_truncated_by_embedding_tokenizer_settings() {
    let mut tokenizer = tokenizer();
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: 8,
            ..Default::default()
        }))
        .unwrap();
    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(8),
        ..Default::default()
    }));
    let chunker = Chunker::new(
        tokenizer,
        ChunkingConfig {
            max_tokens: 8,
            overlap_tokens: 2,
        },
        512,
    )
    .unwrap();
    let text = format!("{}tail", "word ".repeat(1000));
    let ranges = chunker.ranges(&text).unwrap();
    assert!(ranges.len() > 100);
    assert!(text[ranges.last().unwrap().clone()].ends_with("tail"));
    assert_coverage(&text, &ranges, &chunker);
}

#[test]
fn invalid_budgets_and_tokenless_input_fail_cleanly() {
    for (max_tokens, overlap_tokens) in [(0, 0), (513, 0), (2, 0), (6, 4), (6, 6)] {
        assert!(
            Chunker::new(
                tokenizer(),
                ChunkingConfig {
                    max_tokens,
                    overlap_tokens
                },
                512
            )
            .is_err()
        );
    }
    assert!(matches!(
        chunker(8, 1).ranges("\0"),
        Err(ChunkingError::NoTokens)
    ));
}

#[test]
#[ignore = "loads the real model and downloads its files if not cached"]
fn real_model_tokenizer_keeps_every_unicode_passage_within_budget() {
    let embedder = crate::TextEmbedder::new().unwrap();
    for (max_tokens, overlap_tokens) in [(8, 2), (64, 8), (256, 32), (512, 32)] {
        let chunker = Chunker::new(
            embedder.tokenizer().clone(),
            ChunkingConfig {
                max_tokens,
                overlap_tokens,
            },
            embedder.config().max_tokens,
        )
        .unwrap();
        let text = format!(
            "{}\nFINAL-MARKER",
            "Café cafe\u{301}, yö, 世界! 🙂 Misunderstanding tokenization.\n".repeat(60)
        );
        let ranges = chunker.ranges(&text).unwrap();
        assert_coverage(&text, &ranges, &chunker);
        assert!(text[ranges.last().unwrap().clone()].contains("FINAL"));
    }
}
