//! Smart document chunking, ported from TS `store.ts` (break-point scan,
//! code-fence detection, scored cutoff with squared-distance decay, and
//! token-accurate re-splitting).
//!
//! Positions are byte offsets into the UTF-8 document (the TS implementation
//! uses UTF-16 code units; identical for ASCII, and all cuts here are clamped
//! to `char` boundaries so multi-byte text is never split mid-character).

use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Tokens per chunk.
pub const CHUNK_SIZE_TOKENS: usize = 900;
/// Overlap between consecutive chunks (15% of the chunk size).
pub const CHUNK_OVERLAP_TOKENS: usize = CHUNK_SIZE_TOKENS * 15 / 100;
/// Break-point search window before the target cut position.
pub const CHUNK_WINDOW_TOKENS: usize = 200;
/// Under [`ChunkBoundary::Heading`] a heading may be the cut anywhere after
/// the first `1 / HEADING_SPAN_DIVISOR` of the chunk. Against the second
/// half, a quarter held more sections whole; an eighth or any heading added
/// almost nothing more for smaller chunks.
pub const HEADING_SPAN_DIVISOR: usize = 4;

/// Char approximations (~4 chars/token) used by the sync char-based chunker.
pub const CHUNK_SIZE_CHARS: usize = CHUNK_SIZE_TOKENS * 4;
pub const CHUNK_OVERLAP_CHARS: usize = CHUNK_OVERLAP_TOKENS * 4;
pub const CHUNK_WINDOW_CHARS: usize = CHUNK_WINDOW_TOKENS * 4;

/// Smallest and largest chunk size a collection may set. The ceiling keeps a
/// chunk plus its embedding prompt inside the 2048-token embed context.
pub const MIN_CHUNK_TOKENS: usize = 64;
pub const MAX_CHUNK_TOKENS: usize = 1800;

/// Shallowest and deepest heading level a collection's sections may be cut at.
pub const MIN_SECTION_LEVEL: u8 = 1;
pub const MAX_SECTION_LEVEL: u8 = 6;

/// Chunk geometry in tokens. Everything follows from the size: overlap is
/// 15% of it, and the break-point window is 200 tokens or half the size,
/// whichever is smaller, so a cut can never fall in a chunk's first half.
/// The default reproduces the TS constants (900/135/200, window boundary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkParams {
    pub size_tokens: usize,
    pub overlap_tokens: usize,
    pub window_tokens: usize,
    pub boundary: ChunkBoundary,
    /// Set by [`ChunkParams::sections`]: every chunk is one section at this
    /// heading level, or a paragraph-cut part of one longer than
    /// `size_tokens`; the window fields are unused.
    pub section_level: Option<u8>,
}

/// Where a chunk may end and the next begin.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChunkBoundary {
    /// The TS rule: the best-scoring break point in the window before the
    /// target cut, and every next chunk starts `overlap` before it.
    #[default]
    Window,
    /// A heading anywhere after the chunk's first quarter is the cut, and the next
    /// chunk starts at that heading with no overlap; otherwise as `Window`.
    Heading,
}

impl ChunkBoundary {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Window => "window",
            Self::Heading => "heading",
        }
    }
}

impl std::str::FromStr for ChunkBoundary {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "window" => Ok(Self::Window),
            "heading" => Ok(Self::Heading),
            other => Err(format!(
                "unknown chunk boundary '{other}': use window or heading"
            )),
        }
    }
}

impl Default for ChunkParams {
    fn default() -> Self {
        Self::for_size(CHUNK_SIZE_TOKENS)
    }
}

impl ChunkParams {
    /// Geometry for a chunk size, clamped to
    /// [`MIN_CHUNK_TOKENS`]..=[`MAX_CHUNK_TOKENS`].
    pub fn for_size(size_tokens: usize) -> Self {
        let size_tokens = size_tokens.clamp(MIN_CHUNK_TOKENS, MAX_CHUNK_TOKENS);
        Self {
            size_tokens,
            overlap_tokens: size_tokens * 15 / 100,
            window_tokens: CHUNK_WINDOW_TOKENS.min(size_tokens / 2),
            boundary: ChunkBoundary::Window,
            section_level: None,
        }
    }

    /// Section chunks at heading `level` (clamped to
    /// [`MIN_SECTION_LEVEL`]..=[`MAX_SECTION_LEVEL`]). A section is split
    /// only when it is longer than the embedder takes in one input,
    /// [`MAX_CHUNK_TOKENS`].
    pub fn sections(level: u8) -> Self {
        Self {
            size_tokens: MAX_CHUNK_TOKENS,
            overlap_tokens: 0,
            window_tokens: 0,
            boundary: ChunkBoundary::Window,
            section_level: Some(level.clamp(MIN_SECTION_LEVEL, MAX_SECTION_LEVEL)),
        }
    }

    /// The same geometry with `boundary`.
    #[must_use]
    pub fn with_boundary(self, boundary: ChunkBoundary) -> Self {
        Self { boundary, ..self }
    }

    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

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

/// The best heading cut in `(span_start, target_pos]`, scored as
/// [`find_best_cutoff`] scores, with that span as the window; `None` when
/// the span holds no heading outside a code fence.
fn find_heading_cutoff(
    break_points: &[BreakPoint],
    span_start: usize,
    target_pos: usize,
    code_fences: &[CodeFenceRegion],
) -> Option<usize> {
    let span = target_pos.saturating_sub(span_start).max(1);
    let mut best: Option<(f64, usize)> = None;
    for bp in break_points {
        if bp.pos <= span_start || !is_heading(bp) {
            continue;
        }
        if bp.pos > target_pos {
            break;
        }
        if is_inside_code_fence(bp.pos, code_fences) {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        let normalized_dist = (target_pos - bp.pos) as f64 / span as f64;
        let score = f64::from(bp.score) * (1.0 - normalized_dist * normalized_dist * 0.7);
        if best.is_none_or(|(s, _)| score > s) {
            best = Some((score, bp.pos));
        }
    }
    best.map(|(_, pos)| pos)
}

fn is_heading(bp: &BreakPoint) -> bool {
    matches!(bp.kind, "h1" | "h2" | "h3" | "h4" | "h5" | "h6")
}

/// Whether the break point at `pos` is a heading.
fn heading_at(break_points: &[BreakPoint], pos: usize) -> bool {
    break_points
        .binary_search_by_key(&pos, |bp| bp.pos)
        .is_ok_and(|i| is_heading(&break_points[i]))
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
    boundary: ChunkBoundary,
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
            let heading = match boundary {
                ChunkBoundary::Window => None,
                ChunkBoundary::Heading => find_heading_cutoff(
                    break_points,
                    char_pos + (target_end - char_pos) / HEADING_SPAN_DIVISOR,
                    target_end,
                    code_fences,
                ),
            };
            let best = heading.unwrap_or_else(|| {
                find_best_cutoff(break_points, target_end, window_chars, 0.7, code_fences)
            });
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
        if boundary == ChunkBoundary::Heading
            && heading_at(break_points, end_pos)
            && !is_inside_code_fence(end_pos, code_fences)
        {
            char_pos = end_pos + 1;
            continue;
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
    chunk_document_bounded(
        content,
        max_chars,
        overlap_chars,
        window_chars,
        ChunkBoundary::Window,
    )
}

/// [`chunk_document`] under a [`ChunkBoundary`] rule.
pub fn chunk_document_bounded(
    content: &str,
    max_chars: usize,
    overlap_chars: usize,
    window_chars: usize,
    boundary: ChunkBoundary,
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
        boundary,
    )
}

/// [`chunk_document`] with the default 900-token char approximations.
pub fn chunk_document_default(content: &str) -> Vec<Chunk> {
    chunk_document_for(content, &ChunkParams::default())
}

/// [`chunk_document`] with `params` in the ~4 chars/token approximation.
pub fn chunk_document_for(content: &str, params: &ChunkParams) -> Vec<Chunk> {
    if let Some(level) = params.section_level {
        return section_chunks(content, &CharTokens, level, params.size_tokens)
            .expect("the char approximation cannot fail")
            .into_iter()
            .map(|c| Chunk {
                text: c.text,
                pos: c.pos,
            })
            .collect();
    }
    chunk_document_bounded(
        content,
        params.size_tokens * 4,
        params.overlap_tokens * 4,
        params.window_tokens * 4,
        params.boundary,
    )
}

/// Chars per token of the token chunker's char-space first pass: a moderate
/// estimate (prose ~4, code ~2, mixed ~3); oversize chunks are re-split with
/// the measured ratio.
const TOKEN_PASS_CHARS_PER_TOKEN: usize = 3;

/// The chunks [`chunk_document_by_tokens_for`] stored at byte offsets
/// `positions`, rebuilt without the tokenizer from its char-space first pass,
/// which every chunk the re-split leaves alone comes from unchanged. A
/// re-split piece, absent from that pass, is approximated by a first-pass
/// length of text from its offset.
pub fn token_chunks_at(content: &str, params: &ChunkParams, positions: &[usize]) -> Vec<Chunk> {
    let max_chars = params.size_tokens * TOKEN_PASS_CHARS_PER_TOKEN;
    if let Some(level) = params.section_level {
        let sections = split_sections(content, level);
        return positions
            .iter()
            .map(|&pos| {
                let start = floor_char_boundary(content, pos.min(content.len()));
                let section_end = sections
                    .iter()
                    .find(|s| s.start <= start && start < s.end)
                    .map_or(content.len(), |s| s.end);
                let end = floor_char_boundary(content, section_end.min(start + max_chars));
                Chunk {
                    text: content[start..end].to_owned(),
                    pos: start,
                }
            })
            .collect();
    }
    let first_pass = chunk_document_bounded(
        content,
        max_chars,
        params.overlap_tokens * TOKEN_PASS_CHARS_PER_TOKEN,
        params.window_tokens * TOKEN_PASS_CHARS_PER_TOKEN,
        params.boundary,
    );
    positions
        .iter()
        .map(|&pos| {
            first_pass
                .iter()
                .find(|c| c.pos == pos)
                .cloned()
                .unwrap_or_else(|| {
                    let start = floor_char_boundary(content, pos.min(content.len()));
                    let end = floor_char_boundary(content, (start + max_chars).min(content.len()));
                    Chunk {
                        text: content[start..end].to_owned(),
                        pos: start,
                    }
                })
        })
        .collect()
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
    chunk_by_tokens(
        content,
        tokenizer,
        &ChunkParams {
            size_tokens: max_tokens,
            overlap_tokens,
            window_tokens,
            boundary: ChunkBoundary::Window,
            section_level: None,
        },
    )
}

fn chunk_by_tokens(
    content: &str,
    tokenizer: &dyn Tokenizer,
    params: &ChunkParams,
) -> Result<Vec<TokenChunk>> {
    let ChunkParams {
        size_tokens: max_tokens,
        overlap_tokens,
        window_tokens,
        boundary,
        section_level,
    } = *params;
    if let Some(level) = section_level {
        return section_chunks(content, tokenizer, level, max_tokens);
    }
    let avg_chars_per_token = TOKEN_PASS_CHARS_PER_TOKEN;
    let max_chars = max_tokens * avg_chars_per_token;
    let overlap_chars = overlap_tokens * avg_chars_per_token;
    let window_chars = window_tokens * avg_chars_per_token;

    let break_points = scan_break_points(content);
    let code_fences = find_code_fences(content);
    let char_chunks = chunk_document_with_break_points(
        content,
        &break_points,
        &code_fences,
        max_chars,
        overlap_chars,
        window_chars,
        boundary,
    );

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
            boundary,
            &code_fences,
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
    chunk_document_by_tokens_for(content, tokenizer, &ChunkParams::default())
}

/// [`chunk_document_by_tokens`] with `params`.
///
/// # Errors
/// Propagates tokenizer failures.
pub fn chunk_document_by_tokens_for(
    content: &str,
    tokenizer: &dyn Tokenizer,
    params: &ChunkParams,
) -> Result<Vec<TokenChunk>> {
    chunk_by_tokens(content, tokenizer, params)
}

fn clamp_overlap_chars(value: f64, max_chars: usize) -> usize {
    if max_chars <= 1 {
        return 0;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let floored = value.max(0.0).floor() as usize;
    floored.min(max_chars - 1)
}

/// The document's `fences` that overlap `start..start + len`, in that
/// piece's coordinates.
fn fences_within(fences: &[CodeFenceRegion], start: usize, len: usize) -> Vec<CodeFenceRegion> {
    fences
        .iter()
        .filter(|f| f.end > start && f.start < start + len)
        .map(|f| CodeFenceRegion {
            start: f.start.saturating_sub(start),
            end: (f.end - start).min(len),
        })
        .collect()
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)]
fn push_chunk_within_token_limit(
    results: &mut Vec<TokenChunk>,
    tokenizer: &dyn Tokenizer,
    text: &str,
    pos: usize,
    max_tokens: usize,
    overlap_chars: usize,
    window_chars: usize,
    boundary: ChunkBoundary,
    doc_fences: &[CodeFenceRegion],
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
    let sub_fences = match boundary {
        ChunkBoundary::Window => find_code_fences(text),
        ChunkBoundary::Heading => fences_within(doc_fences, pos, text.len()),
    };
    let mut sub_chunks = chunk_document_with_break_points(
        text,
        &scan_break_points(text),
        &sub_fences,
        safe_max_chars,
        next_overlap,
        next_window,
        boundary,
    );

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
            boundary,
            doc_fences,
        )?;
    }
    Ok(())
}

/// One section of a document at a heading level: from a heading at that
/// level or shallower to the next such heading, deeper headings inside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// Byte span in the document; the sections of a document partition it.
    pub start: usize,
    pub end: usize,
    /// Level of the section's heading; 0 for text before the first heading
    /// at the level or shallower, and for a document with none.
    pub level: u8,
    /// The heading's text, without its `#` marks; empty at level 0.
    pub heading: String,
    /// The headings enclosing it, outermost first, then its own.
    pub trail: Vec<String>,
    /// First and last line (1-based) holding text; trailing blank lines are
    /// not counted.
    pub start_line: usize,
    pub end_line: usize,
}

/// An ATX heading line (`#` to `######`, then a space or the line's end, at
/// most three spaces in): its level and text, closing `#`s stripped.
fn atx_heading(line: &str) -> Option<(u8, &str)> {
    let rest = line.trim_start_matches(' ');
    if line.len() - rest.len() > 3 {
        return None;
    }
    let hashes = rest.bytes().take_while(|b| *b == b'#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let after = &rest[hashes..];
    if !after.is_empty() && !after.starts_with([' ', '\t']) {
        return None;
    }
    let text = after.trim();
    let closed = text.trim_end_matches('#');
    let text = if closed.is_empty() || closed.ends_with([' ', '\t']) {
        closed.trim_end()
    } else {
        text
    };
    #[allow(clippy::cast_possible_truncation)]
    Some((hashes as u8, text))
}

/// A code fence opening line (three or more backticks or tildes, at most
/// three spaces in): its character and length.
fn fence_open(line: &str) -> Option<(u8, usize)> {
    let rest = line.trim_start_matches(' ');
    if line.len() - rest.len() > 3 {
        return None;
    }
    let ch = *rest.as_bytes().first()?;
    if ch != b'`' && ch != b'~' {
        return None;
    }
    let n = rest.bytes().take_while(|b| *b == ch).count();
    (n >= 3).then_some((ch, n))
}

/// Whether `line` closes a fence opened with `n` of `ch`.
fn fence_closes(line: &str, ch: u8, n: usize) -> bool {
    fence_open(line).is_some_and(|(c, m)| {
        c == ch && m >= n && line.trim_start_matches(' ')[m..].trim().is_empty()
    })
}

/// Each line of `content` as (byte offset, text without its line ending,
/// inside a code fence). A fence's closing line counts as inside, its
/// opening line does not.
fn scan_lines(content: &str) -> Vec<(usize, &str, bool)> {
    let mut lines = Vec::new();
    let mut fence: Option<(u8, usize)> = None;
    let mut offset = 0;
    for raw in content.split_inclusive('\n') {
        let start = offset;
        offset += raw.len();
        let text = raw.trim_end_matches('\n').trim_end_matches('\r');
        let fenced = if let Some((ch, n)) = fence {
            if fence_closes(text, ch, n) {
                fence = None;
            }
            true
        } else if let Some(open) = fence_open(text) {
            fence = Some(open);
            false
        } else {
            false
        };
        lines.push((start, text, fenced));
    }
    lines
}

/// Split `content` into its sections at heading `level`: a section starts at
/// each heading of that level or shallower outside a code fence and runs to
/// the next. Text before the first such heading is a level-0 section of its
/// own unless it is only whitespace, which joins the first section; a
/// document with no such heading is one level-0 section. A section holding
/// nothing but its heading joins the section after it, which keeps that
/// section's heading.
pub fn split_sections(content: &str, level: u8) -> Vec<Section> {
    struct Cut {
        start: usize,
        level: u8,
        heading: String,
        trail: Vec<String>,
    }
    let mut cuts: Vec<Cut> = Vec::new();
    let mut stack: Vec<(u8, String)> = Vec::new();
    for (start, text, fenced) in scan_lines(content) {
        if fenced {
            continue;
        }
        let Some((h, heading)) = atx_heading(text) else {
            continue;
        };
        while stack.last().is_some_and(|(l, _)| *l >= h) {
            stack.pop();
        }
        stack.push((h, heading.to_owned()));
        if h <= level {
            cuts.push(Cut {
                start,
                level: h,
                heading: heading.to_owned(),
                trail: stack.iter().map(|(_, t)| t.clone()).collect(),
            });
        }
    }

    let mut spans: Vec<(usize, usize, Option<usize>)> = Vec::new();
    match cuts.first() {
        None => spans.push((0, content.len(), None)),
        Some(first) if first.start > 0 && !content[..first.start].trim().is_empty() => {
            spans.push((0, first.start, None));
        }
        Some(_) => {}
    }
    for (i, cut) in cuts.iter().enumerate() {
        let start = if i == 0 && spans.is_empty() {
            0
        } else {
            cut.start
        };
        let end = cuts.get(i + 1).map_or(content.len(), |c| c.start);
        spans.push((start, end, Some(i)));
    }

    let mut joined: Vec<(usize, usize, Option<usize>)> = Vec::with_capacity(spans.len());
    let mut carry: Option<usize> = None;
    for (i, &(start, end, cut)) in spans.iter().enumerate() {
        let start = carry.take().unwrap_or(start);
        if cut.is_some() && i + 1 < spans.len() && heading_lines_only(&content[start..end]) {
            carry = Some(start);
        } else {
            joined.push((start, end, cut));
        }
    }

    let mut sections = Vec::with_capacity(joined.len());
    let mut line = 1usize;
    let mut counted = 0usize;
    for (start, end, cut) in joined {
        line += content[counted..start].matches('\n').count();
        counted = start;
        let body = content[start..end].trim_end();
        let (level, heading, trail) = match cut {
            Some(i) => (
                cuts[i].level,
                cuts[i].heading.clone(),
                cuts[i].trail.clone(),
            ),
            None => (0, String::new(), Vec::new()),
        };
        sections.push(Section {
            start,
            end,
            level,
            heading,
            trail,
            start_line: line,
            end_line: line + body.matches('\n').count(),
        });
    }
    sections
}

/// Whether every non-blank line of `text` is an ATX heading.
fn heading_lines_only(text: &str) -> bool {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .all(|l| atx_heading(l.trim_end_matches('\r')).is_some())
}

/// ~4 chars a token, the char-space approximation.
struct CharTokens;

impl Tokenizer for CharTokens {
    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(text.chars().count().div_ceil(4))
    }

    fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> Result<String> {
        Ok(text.chars().take(max_tokens * 4).collect())
    }
}

/// The chunks of `content` under [`ChunkParams::sections`]: each section at
/// `level` whole, or cut by [`section_parts`] when it is longer than
/// `max_tokens`.
///
/// # Errors
/// Propagates tokenizer failures.
pub fn section_chunks(
    content: &str,
    tokenizer: &dyn Tokenizer,
    level: u8,
    max_tokens: usize,
) -> Result<Vec<TokenChunk>> {
    let mut chunks = Vec::new();
    for section in split_sections(content, level) {
        chunks.extend(section_parts(
            &content[section.start..section.end],
            section.start,
            tokenizer,
            max_tokens,
        )?);
    }
    Ok(chunks)
}

/// How finely [`section_parts`] cuts a piece that does not fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cut {
    Paragraph,
    Line,
    Char,
}

/// One section (`text`, at byte offset `pos` in its document) as parts of
/// at most `max_tokens`: whole when it fits, else cut at paragraph starts
/// (a line after a blank line, outside a code fence), packed greedily. A
/// paragraph that alone does not fit is cut at line starts, and a line that
/// does not fit at characters. The parts partition the section: nothing is
/// truncated or overlapped.
///
/// # Errors
/// Propagates tokenizer failures.
pub fn section_parts(
    text: &str,
    pos: usize,
    tokenizer: &dyn Tokenizer,
    max_tokens: usize,
) -> Result<Vec<TokenChunk>> {
    let mut parts = Vec::new();
    fit_parts(
        text,
        pos,
        tokenizer,
        max_tokens.max(1),
        Cut::Paragraph,
        &mut parts,
    )?;
    Ok(parts)
}

fn fit_parts(
    text: &str,
    pos: usize,
    tokenizer: &dyn Tokenizer,
    max_tokens: usize,
    cut: Cut,
    parts: &mut Vec<TokenChunk>,
) -> Result<()> {
    let tokens = tokenizer.count_tokens(text)?;
    if tokens <= max_tokens || text.chars().nth(1).is_none() {
        parts.push(TokenChunk {
            text: text.to_owned(),
            pos,
            tokens,
        });
        return Ok(());
    }
    let starts: Vec<usize> = match cut {
        Cut::Paragraph => {
            let lines = scan_lines(text);
            lines
                .windows(2)
                .filter(|w| w[0].1.trim().is_empty() && !w[1].1.trim().is_empty() && !w[1].2)
                .map(|w| w[1].0)
                .collect()
        }
        Cut::Line => text.match_indices('\n').map(|(i, _)| i + 1).collect(),
        Cut::Char => {
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let step =
                ((text.len() as f64 * max_tokens as f64 / tokens as f64) * 0.9).max(1.0) as usize;
            let mut starts = Vec::new();
            let mut at = 0;
            loop {
                let mut next = floor_char_boundary(text, at + step);
                if next <= at {
                    next = next_char_boundary(text, at);
                }
                if next >= text.len() {
                    break;
                }
                starts.push(next);
                at = next;
            }
            starts
        }
    };
    let starts: Vec<usize> = starts
        .into_iter()
        .filter(|&s| s > 0 && s < text.len())
        .collect();
    let finer = match cut {
        Cut::Paragraph => Cut::Line,
        Cut::Line | Cut::Char => Cut::Char,
    };
    if starts.is_empty() {
        return fit_parts(text, pos, tokenizer, max_tokens, finer, parts);
    }

    let bounds: Vec<usize> = std::iter::once(0)
        .chain(starts)
        .chain(std::iter::once(text.len()))
        .collect();
    let flush = |start: usize, end: usize, parts: &mut Vec<TokenChunk>| {
        fit_parts(
            &text[start..end],
            pos + start,
            tokenizer,
            max_tokens,
            finer,
            parts,
        )
    };
    let mut run_start = 0usize;
    let mut run_end = bounds[1];
    for w in bounds[1..].windows(2) {
        let end = w[1];
        if tokenizer.count_tokens(&text[run_start..end])? <= max_tokens {
            run_end = end;
        } else {
            flush(run_start, run_end, parts)?;
            run_start = w[0];
            run_end = end;
        }
    }
    flush(run_start, run_end, parts)
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
    fn chunk_params_follow_the_size() {
        let default = ChunkParams::default();
        assert_eq!(
            (
                default.size_tokens,
                default.overlap_tokens,
                default.window_tokens
            ),
            (900, 135, 200)
        );
        assert!(default.is_default());
        let small = ChunkParams::for_size(300);
        assert_eq!((small.overlap_tokens, small.window_tokens), (45, 150));
        assert_eq!(ChunkParams::for_size(1).size_tokens, MIN_CHUNK_TOKENS);
        assert_eq!(ChunkParams::for_size(10_000).size_tokens, MAX_CHUNK_TOKENS);

        let content = "Word ".repeat(2500);
        let chunks = chunk_document_for(&content, &small);
        assert!(chunks.len() > 8);
        assert!(chunks.iter().all(|c| c.text.len() <= 1200));
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

    /// A heading at 60% of the chunk, a blank line near the target cut.
    fn heading_then_blank() -> (String, usize) {
        let intro = "Introduction text here. ".repeat(50);
        let body = "Body text goes on here. ".repeat(30);
        let tail = "More words after it all. ".repeat(80);
        let content = format!("{intro}\n## Section\n{body}\n\n{tail}");
        let heading = content.find("\n## Section").unwrap();
        (content, heading)
    }

    #[test]
    fn window_boundary_takes_the_nearer_blank_and_overlaps() {
        let (content, heading) = heading_then_blank();
        let chunks = chunk_document(&content, 2000, 300, 400);
        assert_eq!(chunks[0].text.len(), content.find("\n\n").unwrap());
        assert!(chunks[1].pos < chunks[0].text.len(), "overlaps");
        assert_ne!(chunks[1].pos, heading + 1);
    }

    #[test]
    fn heading_boundary_cuts_at_a_late_heading_without_overlap() {
        let (content, heading) = heading_then_blank();
        let chunks = chunk_document_bounded(&content, 2000, 300, 400, ChunkBoundary::Heading);
        assert_eq!(chunks[0].text.len(), heading);
        assert_eq!(chunks[1].pos, heading + 1);
        assert!(chunks[1].text.starts_with("## Section\n"));
    }

    #[test]
    fn heading_boundary_takes_a_heading_just_past_the_first_quarter() {
        let content = format!(
            "{}\n# Third\n{}",
            "a ".repeat(300),
            "Body text. ".repeat(300)
        );
        let chunks = chunk_document_bounded(&content, 2000, 300, 400, ChunkBoundary::Heading);
        assert_eq!(chunks[0].text.len(), 600);
        assert!(chunks[1].text.starts_with("# Third\n"));
    }

    #[test]
    fn heading_boundary_ignores_first_quarter_headings_and_fenced_ones() {
        let early = format!(
            "{}\n# Early\n{}",
            "a ".repeat(200),
            "Body text. ".repeat(300)
        );
        let chunks = chunk_document_bounded(&early, 2000, 300, 400, ChunkBoundary::Heading);
        let window = chunk_document(&early, 2000, 300, 400);
        assert_eq!(chunks[0], window[0], "no heading after the first quarter");
        assert!(
            chunks[1].pos < chunks[0].text.len(),
            "a non-heading cut overlaps"
        );

        let fenced = format!(
            "{}\n```\n# not a heading\n```\n{}",
            "Body text. ".repeat(130),
            "More text. ".repeat(200)
        );
        let chunks = chunk_document_bounded(&fenced, 2000, 300, 400, ChunkBoundary::Heading);
        assert!(!chunks.iter().any(|c| c.text.starts_with("# not")));
    }

    #[test]
    fn heading_boundary_does_not_cut_at_a_thematic_break() {
        let content = format!(
            "{}\nTitle\n---\n{}",
            "Lead words go here. ".repeat(60),
            "Body words. ".repeat(200)
        );
        let chunks = chunk_document_bounded(&content, 2000, 300, 400, ChunkBoundary::Heading);
        let window = chunk_document(&content, 2000, 300, 400);
        assert_eq!(chunks, window);
    }

    #[test]
    fn heading_boundary_overlaps_after_a_cut_on_a_fenced_comment() {
        let lines: String = (0..400).map(|i| format!("# comment {i:03}\n")).collect();
        let content = (0..16)
            .map(|pad| format!("{}\n```sh\n{lines}```\nTail.", "a".repeat(300 + pad)))
            .find(|c| c[2000..].starts_with("\n#"))
            .unwrap();
        let chunks = chunk_document_bounded(&content, 2000, 300, 400, ChunkBoundary::Heading);
        assert_eq!(chunks[0].text.len(), 2000);
        assert_eq!(chunks[1].pos, 1700, "overlaps as a window cut does");
    }

    #[test]
    fn heading_resplit_reads_the_documents_fences() {
        struct Dense;
        impl Tokenizer for Dense {
            fn count_tokens(&self, text: &str) -> Result<usize> {
                Ok(text.chars().count().div_ceil(2))
            }
            fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> Result<String> {
                Ok(text.chars().take(max_tokens * 2).collect())
            }
        }
        let steps: String = (0..30)
            .map(|i| format!("# step {i}\nrun thing {i}\n"))
            .collect();
        let content = format!(
            "{}\n```sh\n{steps}```\n{}",
            "Lead sentence words. ".repeat(25),
            "Tail sentence words. ".repeat(60)
        );
        let params = ChunkParams::for_size(200).with_boundary(ChunkBoundary::Heading);
        let chunks = chunk_document_by_tokens_for(&content, &Dense, &params).unwrap();
        assert!(chunks.iter().all(|c| !c.text.starts_with("# step")));
    }

    #[test]
    fn heading_boundary_holds_through_the_token_resplit() {
        struct Dense;
        impl Tokenizer for Dense {
            fn count_tokens(&self, text: &str) -> Result<usize> {
                Ok(text.chars().count().div_ceil(2))
            }
            fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> Result<String> {
                Ok(text.chars().take(max_tokens * 2).collect())
            }
        }
        let section = |n: usize| format!("## Section {n}\n\n{}\n\n", "Some words here. ".repeat(8));
        let content: String = (10..40).map(section).collect();
        let params = ChunkParams::for_size(200).with_boundary(ChunkBoundary::Heading);
        for tokenizer in [&Dense as &dyn Tokenizer, &ApproxTokenizer] {
            let chunks = chunk_document_by_tokens_for(&content, tokenizer, &params).unwrap();
            assert!(chunks.len() > 5);
            for c in &chunks {
                assert!(
                    c.text.starts_with("## Section"),
                    "chunk at {} mid-section",
                    c.pos
                );
                assert!(c.tokens <= 200);
            }
        }
        let chunks = chunk_document_by_tokens_for(&content, &ApproxTokenizer, &params).unwrap();
        let positions: Vec<usize> = chunks.iter().map(|c| c.pos).collect();
        let rebuilt = token_chunks_at(&content, &params, &positions);
        for (s, r) in chunks.iter().zip(&rebuilt) {
            assert_eq!((s.pos, s.text.as_str()), (r.pos, r.text.as_str()));
        }
    }

    #[test]
    fn chunk_boundary_parses() {
        assert_eq!("Heading".parse(), Ok(ChunkBoundary::Heading));
        assert_eq!("window".parse(), Ok(ChunkBoundary::Window));
        assert!("section".parse::<ChunkBoundary>().is_err());
        assert_eq!(ChunkParams::default().boundary, ChunkBoundary::Window);
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
    fn token_chunks_rebuild_from_their_offsets() {
        let content = "## Section\n\nThe quick brown fox jumps over the lazy dog.\n\n".repeat(200);
        let params = ChunkParams::for_size(200);
        let stored = chunk_document_by_tokens_for(&content, &ApproxTokenizer, &params).unwrap();
        assert!(stored.len() > 2);
        let positions: Vec<usize> = stored.iter().map(|c| c.pos).collect();
        let rebuilt = token_chunks_at(&content, &params, &positions);
        for (s, r) in stored.iter().zip(&rebuilt) {
            assert_eq!((s.pos, s.text.as_str()), (r.pos, r.text.as_str()));
        }

        struct Dense;
        impl Tokenizer for Dense {
            fn count_tokens(&self, text: &str) -> Result<usize> {
                Ok(text.chars().count())
            }
            fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> Result<String> {
                Ok(text.chars().take(max_tokens).collect())
            }
        }
        let pieces = chunk_document_by_tokens_for(&content, &Dense, &params).unwrap();
        let piece = pieces.iter().find(|p| !positions.contains(&p.pos)).unwrap();
        let rebuilt = token_chunks_at(&content, &params, &[piece.pos]);
        assert_eq!(rebuilt[0].pos, piece.pos);
        assert!(rebuilt[0].text.starts_with(&piece.text));
        assert_eq!(rebuilt[0].text.len(), 200 * TOKEN_PASS_CHARS_PER_TOKEN);
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

    /// Each section's (level, heading, text).
    fn sections_of(content: &str, level: u8) -> Vec<(u8, String, &str)> {
        let sections = split_sections(content, level);
        assert_eq!(sections.first().map(|s| s.start), Some(0));
        assert_eq!(sections.last().map(|s| s.end), Some(content.len()));
        assert!(sections.windows(2).all(|w| w[0].end == w[1].start));
        sections
            .into_iter()
            .map(|s| (s.level, s.heading, &content[s.start..s.end]))
            .collect()
    }

    #[test]
    fn sections_keep_deeper_headings_and_cut_at_shallower_ones() {
        let doc = "# Guide\n\nIntro.\n\n## Install\n\nSteps.\n\n### Linux\n\napt.\n\n\
                   #### Notes\n\nmore.\n\n## Use\n\nRun it.\n\n# Appendix\n\nExtra.\n";
        let at2 = sections_of(doc, 2);
        let headings: Vec<&str> = at2.iter().map(|s| s.1.as_str()).collect();
        assert_eq!(headings, ["Guide", "Install", "Use", "Appendix"]);
        assert!(at2[1].2.contains("### Linux") && at2[1].2.contains("#### Notes"));
        assert_eq!(at2[3].0, 1, "a shallower heading starts a section too");

        let at3 = sections_of(doc, 3);
        let headings: Vec<&str> = at3.iter().map(|s| s.1.as_str()).collect();
        assert_eq!(headings, ["Guide", "Install", "Linux", "Use", "Appendix"]);
        let linux = &split_sections(doc, 3)[2];
        assert_eq!(linux.trail, ["Guide", "Install", "Linux"]);
        assert_eq!((linux.start_line, linux.end_line), (9, 15));
    }

    #[test]
    fn sections_follow_skipped_levels() {
        let doc = "# Top\n\ntext\n\n### Deep\n\ndeep text\n\n## Mid\n\nmid text\n";
        let at2 = sections_of(doc, 2);
        assert_eq!(at2.len(), 2);
        assert!(at2[0].2.contains("### Deep"));
        let at3 = split_sections(doc, 3);
        assert_eq!(at3[1].heading, "Deep");
        assert_eq!(at3[1].trail, ["Top", "Deep"]);
        assert_eq!(at3[2].trail, ["Top", "Mid"]);
    }

    #[test]
    fn text_before_the_first_heading_is_its_own_section() {
        let doc = "---\ntitle: x\n---\nLead paragraph.\n\n## One\n\nBody.\n";
        let s = sections_of(doc, 2);
        assert_eq!(s.len(), 2);
        assert_eq!((s[0].0, s[0].1.as_str()), (0, ""));
        assert!(s[0].2.starts_with("---\ntitle"));
        assert_eq!(split_sections(doc, 2)[1].start_line, 6);

        let s = sections_of("\n\n## One\n\nBody.\n", 2);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].1, "One");
    }

    #[test]
    fn a_document_with_no_heading_at_the_level_is_one_section() {
        let doc = "Plain notes.\n\n### Too deep\n\nStill one section.\n";
        let s = sections_of(doc, 2);
        assert_eq!(s.len(), 1);
        assert_eq!((s[0].0, s[0].1.as_str()), (0, ""));
        let s = split_sections(doc, 2);
        assert_eq!((s[0].start_line, s[0].end_line), (1, 5));
    }

    #[test]
    fn headings_in_code_fences_are_not_headings() {
        let doc = "## Real\n\n```sh\n# comment\n## also a comment\n```\n\n~~~\n## tilde\n~~~\n\n\
                   ## Next\n\nx\n";
        let s = sections_of(doc, 2);
        let headings: Vec<&str> = s.iter().map(|s| s.1.as_str()).collect();
        assert_eq!(headings, ["Real", "Next"]);

        let s = sections_of("## A\n#hashtag\n    ## code\n####### seven\n", 2);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn a_heading_only_section_joins_the_next() {
        let doc = "# Title\n\n## Intro\n\nWords.\n\n## Empty\n\n## Last\n\nEnd.\n## Tail\n";
        let s = sections_of(doc, 2);
        let headings: Vec<&str> = s.iter().map(|s| s.1.as_str()).collect();
        assert_eq!(headings, ["Intro", "Last", "Tail"]);
        assert!(s[0].2.starts_with("# Title"));
        assert!(s[1].2.starts_with("## Empty"));
        assert_eq!(split_sections(doc, 2)[0].trail, ["Title", "Intro"]);
        assert_eq!(atx_heading("## Closed ##"), Some((2, "Closed")));
        assert_eq!(atx_heading("## C#"), Some((2, "C#")));
    }

    #[test]
    fn a_section_too_long_for_the_embedder_splits_at_paragraphs() {
        let para = |i: usize| format!("Paragraph {i} {}\n", "word ".repeat(40));
        let body: String = (0..12).map(|i| format!("{}\n", para(i))).collect();
        let doc = format!(
            "## Short\n\nsmall\n\n## Long\n\n{body}```\n{}```\n\n## After\n\nz\n",
            "line\n\nx\n".repeat(30)
        );
        let chunks = section_chunks(&doc, &ApproxTokenizer, 2, 120).unwrap();
        let sections = split_sections(&doc, 2);
        let long = &sections[1];
        let parts: Vec<&TokenChunk> = chunks
            .iter()
            .filter(|c| c.pos >= long.start && c.pos < long.end)
            .collect();
        assert!(parts.len() > 2, "{} parts", parts.len());
        assert_eq!(parts[0].pos, long.start);
        for w in parts.windows(2) {
            assert_eq!(w[0].pos + w[0].text.len(), w[1].pos);
        }
        let last = parts.last().unwrap();
        assert_eq!(last.pos + last.text.len(), long.end);
        for p in &parts {
            assert!(p.tokens <= 120, "{} tokens", p.tokens);
        }
        for p in &parts[1..] {
            assert!(doc[..p.pos].ends_with("\n\n") || doc[..p.pos].ends_with('\n'));
        }
        assert!(
            parts
                .iter()
                .filter(|p| p.text.starts_with("Paragraph"))
                .count()
                >= 2,
            "cut at paragraph starts"
        );
        assert_eq!(chunks[0].text, &doc[sections[0].start..sections[0].end]);
        assert_eq!(chunks.last().unwrap().text, "## After\n\nz\n");
        let rejoined: String = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(rejoined, doc);
    }

    #[test]
    fn an_unbroken_section_still_splits_without_losing_text() {
        let doc = format!("## Blob\n{}", "x".repeat(5_000));
        let chunks = section_chunks(&doc, &ApproxTokenizer, 2, 100).unwrap();
        assert!(chunks.len() > 10);
        assert!(chunks.iter().all(|c| c.tokens <= 100));
        let rejoined: String = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(rejoined, doc);
    }

    #[test]
    fn section_params_route_both_chunkers() {
        let params = ChunkParams::sections(2);
        assert_eq!(params.size_tokens, MAX_CHUNK_TOKENS);
        assert_eq!(
            ChunkParams::sections(9).section_level,
            Some(MAX_SECTION_LEVEL)
        );
        let doc = "## A\n\nalpha\n\n## B\n\nbeta\n";
        let chars = chunk_document_for(doc, &params);
        let tokens = chunk_document_by_tokens_for(doc, &ApproxTokenizer, &params).unwrap();
        assert_eq!(chars.len(), 2);
        assert_eq!(tokens.len(), 2);
        assert_eq!(chars[1].pos, tokens[1].pos);
        let rebuilt = token_chunks_at(doc, &params, &[tokens[1].pos]);
        assert_eq!(rebuilt[0].text, "## B\n\nbeta\n");
    }
}
