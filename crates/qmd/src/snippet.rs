//! Snippet extraction for search output, ported from TS `extractSnippet`,
//! `extractIntentTerms`, and `addLineNumbers`.

use std::collections::HashSet;
use std::sync::LazyLock;

use crate::chunk::CHUNK_SIZE_CHARS;

/// Weight of intent terms relative to query terms in snippet line scoring.
pub const INTENT_WEIGHT_SNIPPET: f64 = 0.3;

/// Common stop words filtered from intent strings before tokenization
/// (TS `INTENT_STOP_WORDS`).
#[rustfmt::skip]
static INTENT_STOP_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "am", "an", "as", "at", "be", "by", "do", "he", "if", "in", "is", "it", "me", "my", "no",
        "of", "on", "or", "so", "to", "up", "us", "we",
        "all", "and", "any", "are", "but", "can", "did", "for", "get", "has", "her", "him", "his",
        "how", "its", "let", "may", "not", "our", "out", "the", "too", "was", "who", "why", "you",
        "also", "does", "find", "from", "have", "into", "more", "need", "show", "some", "tell",
        "that", "them", "this", "want", "what", "when", "will", "with", "your",
        "about", "looking", "notes", "search", "where", "which",
    ]
    .into_iter()
    .collect()
});

/// Extract meaningful lowercase terms from an intent string, trimming
/// punctuation and filtering stop words (short domain terms like "API"
/// survive the >1 length threshold).
pub fn extract_intent_terms(intent: &str) -> Vec<String> {
    intent
        .to_lowercase()
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|t| t.chars().count() > 1 && !INTENT_STOP_WORDS.contains(t))
        .map(str::to_owned)
        .collect()
}

/// A snippet with its location metadata.
#[derive(Debug, Clone)]
pub struct Snippet {
    /// 1-indexed line of the best match.
    pub line: usize,
    /// 1-indexed line of the first body line under the header.
    pub start_line: usize,
    /// Snippet text including the diff-style `@@` header line.
    pub text: String,
}

/// Pick the best-matching line window for `query` in `body` and format it
/// with a diff-style header.
pub fn extract_snippet(body: &str, query: &str, max_len: usize) -> Snippet {
    extract_snippet_at(body, query, max_len, None, None, None)
}

/// Clamp a byte offset down to the nearest char boundary.
fn floor_char_boundary(s: &str, index: usize) -> usize {
    let mut i = index.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// [`extract_snippet`] with optional focus hints: a chunk region (byte
/// offset + length, ±100 chars of padding) restricts the line search, and
/// intent terms contribute at [`INTENT_WEIGHT_SNIPPET`]. When the chunk
/// region yields no match or an empty snippet, falls back per the TS rules:
/// pos 0 retries the full body; pos > 0 anchors on the chunk start (the
/// pipeline actively picked that chunk).
pub fn extract_snippet_at(
    body: &str,
    query: &str,
    max_len: usize,
    chunk_pos: Option<usize>,
    chunk_len: Option<usize>,
    intent: Option<&str>,
) -> Snippet {
    let total_lines = body.split('\n').count();
    let mut search_body = body;
    let mut line_offset = 0usize;
    let mut context_start = 0usize;

    if let Some(pos) = chunk_pos {
        let search_len = chunk_len.unwrap_or(CHUNK_SIZE_CHARS);
        context_start = floor_char_boundary(body, pos.saturating_sub(100));
        let context_end = floor_char_boundary(body, pos.saturating_add(search_len + 100));
        search_body = &body[context_start..context_end];
        if context_start > 0 {
            line_offset = body[..context_start].matches('\n').count();
        }
    }

    let lines: Vec<&str> = search_body.split('\n').collect();
    let query_terms: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let intent_terms: Vec<String> = intent.map(extract_intent_terms).unwrap_or_default();

    let mut best_line = 0usize;
    let mut best_score = -1.0f64;
    for (i, line) in lines.iter().enumerate() {
        let line_lower = line.to_lowercase();
        let mut score = query_terms
            .iter()
            .filter(|t| line_lower.contains(t.as_str()))
            .count() as f64;
        for term in &intent_terms {
            if line_lower.contains(term.as_str()) {
                score += INTENT_WEIGHT_SNIPPET;
            }
        }
        if score > best_score {
            best_score = score;
            best_line = i;
        }
    }

    if let Some(pos) = chunk_pos
        && best_score <= 0.0
    {
        if pos == 0 {
            return extract_snippet_at(body, query, max_len, None, None, intent);
        }
        let pos = floor_char_boundary(search_body, pos - context_start);
        best_line = search_body[..pos].matches('\n').count();
    }

    let start = best_line.saturating_sub(1);
    let end = (best_line + 3).min(lines.len());
    let mut text = lines[start..end].join("\n");
    if chunk_pos.is_some_and(|p| p > 0) && text.trim().is_empty() {
        return extract_snippet_at(body, query, max_len, None, None, intent);
    }
    if text.chars().count() > max_len {
        let truncated: String = text.chars().take(max_len.saturating_sub(3)).collect();
        text = format!("{truncated}...");
    }

    let absolute_start = line_offset + start + 1;
    let snippet_lines = end - start;
    let lines_before = absolute_start - 1;
    let lines_after = total_lines.saturating_sub(absolute_start + snippet_lines - 1);
    let header = format!(
        "@@ -{absolute_start},{snippet_lines} @@ ({lines_before} before, {lines_after} after)"
    );

    Snippet {
        line: line_offset + best_line + 1,
        start_line: absolute_start,
        text: format!("{header}\n{text}"),
    }
}

impl Snippet {
    /// The text with the `@@` header unnumbered and each body line prefixed
    /// with its real line number.
    pub fn numbered(&self) -> String {
        match self.text.split_once('\n') {
            Some((header, body)) => {
                format!("{header}\n{}", add_line_numbers(body, self.start_line))
            }
            None => self.text.clone(),
        }
    }
}

/// Add `N: ` line-number prefixes (TS `addLineNumbers`).
pub fn add_line_numbers(text: &str, start_line: usize) -> String {
    text.split('\n')
        .enumerate()
        .map(|(i, line)| format!("{}: {line}", start_line + i))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_finds_matching_line() {
        let body = "line one\nline two\nthe needle is here\nline four\nline five";
        let s = extract_snippet(body, "needle", 500);
        assert_eq!(s.line, 3);
        assert!(s.text.contains("the needle is here"));
        assert!(s.text.starts_with("@@ -2,4 @@"));
        assert!(s.numbered().contains("\n3: the needle is here\n"));
    }

    #[test]
    fn line_numbers_start_at_given_line() {
        assert_eq!(add_line_numbers("a\nb", 5), "5: a\n6: b");
    }

    #[test]
    fn intent_terms_filter_stop_words_and_punctuation() {
        assert_eq!(
            extract_intent_terms("looking for the API, latency & SQL!"),
            vec!["api", "latency", "sql"]
        );
        assert!(extract_intent_terms("the of and").is_empty());
    }

    #[test]
    fn chunk_hint_focuses_the_search() {
        let filler = "filler text\n".repeat(50);
        let body = format!("needle early\n{filler}needle late\nafter");
        let pos = body.find("needle late").unwrap();
        let s = extract_snippet_at(&body, "needle", 500, Some(pos), Some(20), None);
        assert!(s.text.contains("needle late"));
        assert_eq!(s.line, 52);
    }

    #[test]
    fn chunk_pos_zero_with_no_match_retries_full_body() {
        let body = "no match here\nstill nothing\nthe target line\nmore";
        let s = extract_snippet_at(body, "zzz", 500, Some(0), Some(8), None);
        assert_eq!(s.line, 1);
    }

    #[test]
    fn chunk_pos_nonzero_with_no_match_anchors_on_chunk() {
        let filler = "filler\n".repeat(40);
        let body = format!("{filler}chunk starts here\ntail");
        let pos = body.find("chunk starts").unwrap();
        let s = extract_snippet_at(&body, "zzz", 500, Some(pos), Some(10), None);
        assert!(s.text.contains("chunk starts here"));
    }

    #[test]
    fn intent_terms_boost_snippet_lines() {
        let body = "alpha beta\ngamma latency delta\nepsilon";
        let s = extract_snippet_at(body, "zzz", 500, None, None, Some("latency issues"));
        assert_eq!(s.line, 2);
    }
}
