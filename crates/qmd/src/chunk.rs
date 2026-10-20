//! Smart document chunking, ported from TS `store.ts` (break-point scan,
//! code-fence detection, scored cutoff with squared-distance decay, and
//! token-accurate re-splitting).
//!
//! Positions are byte offsets into the UTF-8 document (the TS implementation
//! uses UTF-16 code units; identical for ASCII, and all cuts here are clamped
//! to `char` boundaries so multi-byte text is never split mid-character).

use std::sync::LazyLock;

use regex::Regex;

use crate::error::Result;

/// Tokens per chunk.
pub const CHUNK_SIZE_TOKENS: usize = 900;
/// Overlap between consecutive chunks (15% of the chunk size).
pub const CHUNK_OVERLAP_TOKENS: usize = CHUNK_SIZE_TOKENS * 15 / 100;
/// Break-point search window before the target cut position.
pub const CHUNK_WINDOW_TOKENS: usize = 200;

/// Char approximations (~4 chars/token) used by the sync char-based chunker.
pub const CHUNK_SIZE_CHARS: usize = CHUNK_SIZE_TOKENS * 4;
pub const CHUNK_OVERLAP_CHARS: usize = CHUNK_OVERLAP_TOKENS * 4;
pub const CHUNK_WINDOW_CHARS: usize = CHUNK_WINDOW_TOKENS * 4;

/// A potential break point with a base score (higher = better place to split).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BreakPoint {
    /// Byte position of the `\n` that starts the construct.
    pub pos: usize,
    /// Base quality score.
    pub score: u32,
    /// Pattern name for debugging (`h1`, `blank`, ...).
    pub kind: &'static str,
}

/// A region delimited by ``` fences; chunks must never split inside one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodeFenceRegion {
    /// Position of the `\n` before the opening fence.
    pub start: usize,
    /// Position just past the closing fence (document end if unclosed).
    pub end: usize,
}

/// One chunk of a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub text: String,
    /// Byte offset of the chunk within the original document.
    pub pos: usize,
}

/// One token-limited chunk with its measured token count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenChunk {
    pub text: String,
    pub pos: usize,
    pub tokens: usize,
}

/// Model tokenizer surface needed for token-accurate chunking.
pub trait Tokenizer {
    /// Number of model tokens in `text`.
    ///
    /// # Errors
    /// Propagates backend tokenization failures.
    fn count_tokens(&self, text: &str) -> Result<usize>;

    /// Truncate `text` to at most `max_tokens` tokens (tokenize, slice,
    /// detokenize).
    ///
    /// # Errors
    /// Propagates backend tokenization failures.
    fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> Result<String>;
}

static FENCE: LazyLock<Regex> = LazyLock::new(|| Regex::new("\n```").expect("valid regex"));
static HR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n(?:---|\*\*\*|___)\s*\n").expect("valid regex"));
static BLANK: LazyLock<Regex> = LazyLock::new(|| Regex::new("\n\n+").expect("valid regex"));
static LIST: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n[-*]\s").expect("valid regex"));
static NUMLIST: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n\d+\.\s").expect("valid regex"));
static NEWLINE: LazyLock<Regex> = LazyLock::new(|| Regex::new("\n").expect("valid regex"));

/// Scan text for all potential break points, keeping the highest-scoring
/// pattern at each position. Result is sorted by position.
pub fn scan_break_points(text: &str) -> Vec<BreakPoint> {
    let mut best: std::collections::HashMap<usize, BreakPoint> = std::collections::HashMap::new();
    let mut consider = |pos: usize, score: u32, kind: &'static str| {
        let entry = best.entry(pos).or_insert(BreakPoint { pos, score, kind });
        if score > entry.score {
            *entry = BreakPoint { pos, score, kind };
        }
    };

    let bytes = text.as_bytes();
    for (pos, _) in text.match_indices('\n') {
        let hashes = bytes[pos + 1..].iter().take_while(|b| **b == b'#').count();
        if (1..=6).contains(&hashes) {
            let kinds = ["h1", "h2", "h3", "h4", "h5", "h6"];
            #[allow(clippy::cast_possible_truncation)]
            consider(pos, (100 - 10 * (hashes - 1)) as u32, kinds[hashes - 1]);
        }
    }

    for (regex, score, kind) in [
        (&*FENCE, 80, "codeblock"),
        (&*HR, 60, "hr"),
        (&*BLANK, 20, "blank"),
        (&*LIST, 5, "list"),
        (&*NUMLIST, 5, "numlist"),
        (&*NEWLINE, 1, "newline"),
    ] {
        for m in regex.find_iter(text) {
            consider(m.start(), score, kind);
        }
    }

    let mut points: Vec<BreakPoint> = best.into_values().collect();
    points.sort_by_key(|bp| bp.pos);
    points
}

/// Find all code fence regions (``` pairs; an unclosed fence extends to the
/// end of the document).
pub fn find_code_fences(text: &str) -> Vec<CodeFenceRegion> {
    let mut regions = Vec::new();
    let mut fence_start: Option<usize> = None;
    for m in FENCE.find_iter(text) {
        match fence_start.take() {
            None => fence_start = Some(m.start()),
            Some(start) => regions.push(CodeFenceRegion {
                start,
                end: m.end(),
            }),
        }
    }
    if let Some(start) = fence_start {
        regions.push(CodeFenceRegion {
            start,
            end: text.len(),
        });
    }
    regions
}

/// Whether `pos` lies strictly inside any code fence region.
pub fn is_inside_code_fence(pos: usize, fences: &[CodeFenceRegion]) -> bool {
    fences.iter().any(|f| pos > f.start && pos < f.end)
}

/// Find the best cut position at or before `target_pos` using scored break
/// points with squared-distance decay (1.0 at target, `1 - decay` at the
/// window edge — a distant heading still beats a nearby newline).
pub fn find_best_cutoff(
    break_points: &[BreakPoint],
    target_pos: usize,
    window_chars: usize,
    decay_factor: f64,
    code_fences: &[CodeFenceRegion],
) -> usize {
    let window_start = target_pos.saturating_sub(window_chars);
    let mut best_score = -1.0f64;
    let mut best_pos = target_pos;

    for bp in break_points {
        if bp.pos < window_start {
            continue;
        }
        if bp.pos > target_pos {
            break;
        }
        if is_inside_code_fence(bp.pos, code_fences) {
            continue;
        }

        #[allow(clippy::cast_precision_loss)]
        let normalized_dist = if window_chars == 0 {
            0.0
        } else {
            (target_pos - bp.pos) as f64 / window_chars as f64
        };
        let multiplier = 1.0 - normalized_dist * normalized_dist * decay_factor;
        let final_score = f64::from(bp.score) * multiplier;
        if final_score > best_score {
            best_score = final_score;
            best_pos = bp.pos;
        }
    }

    best_pos
}

/// Largest char boundary `<= pos`.
fn floor_char_boundary(s: &str, mut pos: usize) -> usize {
    pos = pos.min(s.len());
    while !s.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

/// Smallest char boundary `> pos` (clamped to `s.len()`).
fn next_char_boundary(s: &str, pos: usize) -> usize {
    let mut p = (pos + 1).min(s.len());
    while !s.is_char_boundary(p) {
        p += 1;
    }
    p
}

/// Core chunk algorithm over precomputed break points and code fences.
pub fn chunk_document_with_break_points(
    content: &str,
    break_points: &[BreakPoint],
    code_fences: &[CodeFenceRegion],
    max_chars: usize,
    overlap_chars: usize,
    window_chars: usize,
) -> Vec<Chunk> {
    if content.len() <= max_chars {
        return vec![Chunk {
            text: content.to_owned(),
            pos: 0,
        }];
    }

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut char_pos = 0usize;

    while char_pos < content.len() {
        let target_end = floor_char_boundary(content, (char_pos + max_chars).min(content.len()));
        let mut end_pos = target_end;

        if end_pos < content.len() {
            let best = find_best_cutoff(break_points, target_end, window_chars, 0.7, code_fences);
            if best > char_pos && best <= target_end {
                end_pos = best;
            }
        }

        if end_pos <= char_pos {
            end_pos = next_char_boundary(content, char_pos);
        }

        chunks.push(Chunk {
            text: content[char_pos..end_pos].to_owned(),
            pos: char_pos,
        });

        if end_pos >= content.len() {
            break;
        }
        char_pos = floor_char_boundary(content, end_pos.saturating_sub(overlap_chars));
        let last_chunk_pos = chunks.last().expect("just pushed").pos;
        if char_pos <= last_chunk_pos {
            char_pos = end_pos;
        }
    }

    chunks
}

/// Chunk a document in character space (regex break-point strategy).
pub fn chunk_document(
    content: &str,
    max_chars: usize,
    overlap_chars: usize,
    window_chars: usize,
) -> Vec<Chunk> {
    let break_points = scan_break_points(content);
    let code_fences = find_code_fences(content);
    chunk_document_with_break_points(
        content,
        &break_points,
        &code_fences,
        max_chars,
        overlap_chars,
        window_chars,
    )
}

/// [`chunk_document`] with the default 900-token char approximations.
pub fn chunk_document_default(content: &str) -> Vec<Chunk> {
    chunk_document(
        content,
        CHUNK_SIZE_CHARS,
        CHUNK_OVERLAP_CHARS,
        CHUNK_WINDOW_CHARS,
    )
}

/// Chunk a document by actual token count using the model tokenizer:
/// a conservative char-space pass (~3 chars/token), then any oversize chunk
/// is re-split recursively with the measured chars-per-token ratio.
///
/// # Errors
/// Propagates tokenizer failures.
pub fn chunk_document_by_tokens(
    content: &str,
    tokenizer: &dyn Tokenizer,
    max_tokens: usize,
    overlap_tokens: usize,
    window_tokens: usize,
) -> Result<Vec<TokenChunk>> {
    let avg_chars_per_token = 3;
    let max_chars = max_tokens * avg_chars_per_token;
    let overlap_chars = overlap_tokens * avg_chars_per_token;
    let window_chars = window_tokens * avg_chars_per_token;

    let char_chunks = chunk_document(content, max_chars, overlap_chars, window_chars);

    let mut results: Vec<TokenChunk> = Vec::new();
    for chunk in char_chunks {
        push_chunk_within_token_limit(
            &mut results,
            tokenizer,
            &chunk.text,
            chunk.pos,
            max_tokens,
            overlap_chars,
            window_chars,
        )?;
    }
    Ok(results)
}

/// [`chunk_document_by_tokens`] with the default 900/135/200 constants.
///
/// # Errors
/// Propagates tokenizer failures.
pub fn chunk_document_by_tokens_default(
    content: &str,
    tokenizer: &dyn Tokenizer,
) -> Result<Vec<TokenChunk>> {
    chunk_document_by_tokens(
        content,
        tokenizer,
        CHUNK_SIZE_TOKENS,
        CHUNK_OVERLAP_TOKENS,
        CHUNK_WINDOW_TOKENS,
    )
}

fn clamp_overlap_chars(value: f64, max_chars: usize) -> usize {
    if max_chars <= 1 {
        return 0;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let floored = value.max(0.0).floor() as usize;
    floored.min(max_chars - 1)
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn push_chunk_within_token_limit(
    results: &mut Vec<TokenChunk>,
    tokenizer: &dyn Tokenizer,
    text: &str,
    pos: usize,
    max_tokens: usize,
    overlap_chars: usize,
    window_chars: usize,
) -> Result<()> {
    let tokens = tokenizer.count_tokens(text)?;
    if tokens <= max_tokens || text.chars().count() <= 1 {
        results.push(TokenChunk {
            text: text.to_owned(),
            pos,
            tokens,
        });
        return Ok(());
    }

    let actual_chars_per_token = text.len() as f64 / tokens as f64;
    let mut safe_max_chars = (max_tokens as f64 * actual_chars_per_token * 0.95).floor() as usize;
    if safe_max_chars < 1 {
        safe_max_chars = text.len() / 2;
    }
    safe_max_chars = safe_max_chars.clamp(1, text.len() - 1);

    let mut next_overlap = clamp_overlap_chars(
        overlap_chars as f64 * actual_chars_per_token / 2.0,
        safe_max_chars,
    );
    let mut next_window = (window_chars as f64 * actual_chars_per_token / 2.0).floor() as usize;
    let mut sub_chunks = chunk_document(text, safe_max_chars, next_overlap, next_window);

    if sub_chunks.len() <= 1
        || sub_chunks
            .first()
            .is_some_and(|c| c.text.len() == text.len())
    {
        safe_max_chars = (text.len() / 2).max(1);
        next_overlap = 0;
        next_window = 0;
        sub_chunks = chunk_document(text, safe_max_chars, next_overlap, next_window);
    }

    if sub_chunks.len() <= 1
        || sub_chunks
            .first()
            .is_some_and(|c| c.text.len() == text.len())
    {
        let truncated = tokenizer.truncate_to_tokens(text, max_tokens.max(1))?;
        let truncated_tokens = tokenizer.count_tokens(&truncated)?;
        results.push(TokenChunk {
            text: truncated,
            pos,
            tokens: truncated_tokens,
        });
        return Ok(());
    }

    for sub in sub_chunks {
        push_chunk_within_token_limit(
            results,
            tokenizer,
            &text[sub.pos..sub.pos + sub.text.len()],
            pos + sub.pos,
            max_tokens,
            overlap_chars,
            window_chars,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ~4 chars/token approximation, mirroring the TS char fallback.
    struct ApproxTokenizer;

    impl Tokenizer for ApproxTokenizer {
        fn count_tokens(&self, text: &str) -> Result<usize> {
            Ok(text.chars().count().div_ceil(4))
        }

        fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> Result<String> {
            Ok(text.chars().take(max_tokens * 4).collect())
        }
    }

    #[test]
    fn detects_h1_headings() {
        let breaks = scan_break_points("Intro\n# Heading 1\nMore text");
        let h1 = breaks.iter().find(|b| b.kind == "h1").unwrap();
        assert_eq!(h1.score, 100);
        assert_eq!(h1.pos, 5);
    }

    #[test]
    fn detects_multiple_heading_levels() {
        let breaks = scan_break_points("Text\n# H1\n## H2\n### H3\nMore");
        let score = |kind: &str| breaks.iter().find(|b| b.kind == kind).unwrap().score;
        assert_eq!(score("h1"), 100);
        assert_eq!(score("h2"), 90);
        assert_eq!(score("h3"), 80);
    }

    #[test]
    fn seven_hashes_is_not_a_heading() {
        let breaks = scan_break_points("Text\n####### too deep");
        assert!(!breaks.iter().any(|b| b.kind.starts_with('h')));
        assert_eq!(breaks.iter().find(|b| b.pos == 4).unwrap().kind, "newline");
    }

    #[test]
    fn detects_code_blocks() {
        let breaks = scan_break_points("Before\n```js\ncode\n```\nAfter");
        let fences: Vec<_> = breaks.iter().filter(|b| b.kind == "codeblock").collect();
        assert_eq!(fences.len(), 2);
        assert_eq!(fences[0].score, 80);
    }

    #[test]
    fn detects_horizontal_rules_and_blank_lines_and_lists() {
        let hr = scan_break_points("Text\n---\nMore text");
        assert_eq!(hr.iter().find(|b| b.kind == "hr").unwrap().score, 60);

        let blank = scan_break_points("First paragraph.\n\nSecond paragraph.");
        assert_eq!(blank.iter().find(|b| b.kind == "blank").unwrap().score, 20);

        let breaks = scan_break_points("Intro\n- Item 1\n- Item 2\n1. Numbered");
        assert_eq!(breaks.iter().filter(|b| b.kind == "list").count(), 2);
        assert_eq!(breaks.iter().filter(|b| b.kind == "numlist").count(), 1);
    }

    #[test]
    fn breaks_sorted_and_highest_score_wins_per_position() {
        let breaks = scan_break_points("Text\n# Heading");
        let at4: Vec<_> = breaks.iter().filter(|b| b.pos == 4).collect();
        assert_eq!(at4.len(), 1);
        assert_eq!(at4[0].kind, "h1");

        let breaks = scan_break_points("A\n# B\n\nC\n## D");
        assert!(breaks.windows(2).all(|w| w[0].pos < w[1].pos));
    }

    #[test]
    fn finds_code_fence_regions() {
        let fences = find_code_fences("Before\n```js\ncode here\n```\nAfter");
        assert_eq!(fences, vec![CodeFenceRegion { start: 6, end: 26 }]);

        let two = find_code_fences("Intro\n```\nblock1\n```\nMiddle\n```\nblock2\n```\nEnd");
        assert_eq!(two.len(), 2);

        let unclosed = find_code_fences("Before\n```\nunclosed code block");
        assert_eq!(unclosed.len(), 1);
        assert_eq!(unclosed[0].end, "Before\n```\nunclosed code block".len());

        assert!(find_code_fences("No code fences here").is_empty());
    }

    #[test]
    fn inside_code_fence_is_exclusive_of_boundaries() {
        let fences = [CodeFenceRegion { start: 10, end: 30 }];
        assert!(is_inside_code_fence(15, &fences));
        assert!(!is_inside_code_fence(10, &fences));
        assert!(!is_inside_code_fence(30, &fences));
        assert!(!is_inside_code_fence(5, &fences));
    }

    fn bp(pos: usize, score: u32, kind: &'static str) -> BreakPoint {
        BreakPoint { pos, score, kind }
    }

    #[test]
    fn cutoff_prefers_higher_scoring_break_points() {
        let points = [
            bp(100, 1, "newline"),
            bp(150, 100, "h1"),
            bp(180, 20, "blank"),
        ];
        assert_eq!(find_best_cutoff(&points, 200, 100, 0.7, &[]), 150);
    }

    #[test]
    fn h2_at_window_edge_beats_blank_at_target() {
        let points = [bp(100, 90, "h2"), bp(195, 20, "blank")];
        assert_eq!(find_best_cutoff(&points, 200, 100, 0.7, &[]), 100);
    }

    #[test]
    fn cutoff_returns_target_when_no_breaks_in_window() {
        let points = [bp(10, 100, "h1")];
        assert_eq!(find_best_cutoff(&points, 200, 100, 0.7, &[]), 200);
        assert_eq!(find_best_cutoff(&[], 200, 100, 0.7, &[]), 200);
    }

    #[test]
    fn cutoff_skips_break_points_inside_code_fences() {
        let points = [bp(150, 100, "h1"), bp(180, 20, "blank")];
        let fences = [CodeFenceRegion {
            start: 140,
            end: 160,
        }];
        assert_eq!(find_best_cutoff(&points, 200, 100, 0.7, &fences), 180);
    }

    #[test]
    fn small_documents_pass_through_whole() {
        let chunks = chunk_document("Small document content", 1000, 0, 800);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Small document content");
        assert_eq!(chunks[0].pos, 0);
    }

    #[test]
    fn large_documents_split_with_monotonic_positions() {
        let content = "A".repeat(10_000);
        let chunks = chunk_document(&content, 1000, 0, 800);
        assert!(chunks.len() > 1);
        assert!(chunks.windows(2).all(|w| w[0].pos < w[1].pos));
    }

    #[test]
    fn overlap_creates_overlapping_chunks() {
        let content = "A".repeat(3000);
        let chunks = chunk_document(&content, 1000, 150, 800);
        assert!(chunks.len() > 1);
        for w in chunks.windows(2) {
            let prev_end = w[0].pos + w[0].text.len();
            assert!(w[1].pos < prev_end, "chunks must overlap");
            assert!(w[1].pos > w[0].pos, "chunks must make progress");
        }
    }

    #[test]
    fn utf8_content_never_splits_mid_character() {
        let content = "こんにちは世界".repeat(500);
        let chunks = chunk_document(&content, 1000, 0, 800);
        assert!(chunks.len() > 1);
        let rejoined: usize = chunks.iter().map(|c| c.text.len()).sum();
        assert!(rejoined >= content.len());
    }

    #[test]
    fn default_params_produce_3600_char_chunks() {
        let content = "Word ".repeat(2500);
        let chunks = chunk_document_default(&content);
        assert!(chunks.len() > 1);
        assert!(chunks[0].text.len() > 2800);
        assert!(chunks[0].text.len() <= 3600);
    }

    #[test]
    fn chunking_prefers_headings_in_window() {
        let section1 = "Introduction text here. ".repeat(70);
        let section2 = "Main content text here. ".repeat(50);
        let content = format!("{section1}\n# Main Section\n{section2}");
        let chunks = chunk_document(&content, 2000, 0, 800);
        let heading_pos = content.find("\n# Main Section").unwrap();
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].text.len(), heading_pos);
    }

    #[test]
    fn chunking_does_not_split_inside_code_blocks() {
        let before = "Some intro text. ".repeat(30);
        let code = format!("```typescript\n{}```\n", "const x = 1;\n".repeat(100));
        let after = "More text after code. ".repeat(30);
        let content = format!("{before}{code}{after}");
        let fences = find_code_fences(&content);
        let chunks = chunk_document(&content, 1000, 0, 400);
        for c in &chunks {
            assert!(
                !is_inside_code_fence(c.pos, &fences),
                "chunk starts inside a code fence at {}",
                c.pos
            );
        }
    }

    #[test]
    fn token_chunking_small_document_is_single_chunk() {
        let chunks =
            chunk_document_by_tokens("This is a small document.", &ApproxTokenizer, 900, 135, 200)
                .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "This is a small document.");
        assert_eq!(chunks[0].pos, 0);
        assert!(chunks[0].tokens > 0 && chunks[0].tokens < 900);
    }

    #[test]
    fn token_chunking_splits_and_respects_limit() {
        let content = "The quick brown fox jumps over the lazy dog. ".repeat(250);
        let chunks = chunk_document_by_tokens(&content, &ApproxTokenizer, 900, 135, 200).unwrap();
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(
                c.tokens <= 950,
                "chunk of {} tokens exceeds limit",
                c.tokens
            );
            assert!(c.tokens > 0);
        }
        assert!(chunks.windows(2).all(|w| w[0].pos < w[1].pos));
    }

    #[test]
    fn token_chunking_overlaps() {
        let content = "Word ".repeat(500);
        let chunks = chunk_document_by_tokens(&content, &ApproxTokenizer, 200, 30, 200).unwrap();
        assert!(chunks.len() > 1);
        for w in chunks.windows(2) {
            assert!(w[1].pos < w[0].pos + w[0].text.len(), "chunks must overlap");
        }
    }

    #[test]
    fn token_chunking_handles_pathological_single_line_blob() {
        let content = "x".repeat(20_000);
        let chunks = chunk_document_by_tokens(&content, &ApproxTokenizer, 100, 15, 200).unwrap();
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.tokens <= 100);
        }
    }
}
