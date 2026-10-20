//! FTS5 query building and BM25 search, ported from TS `store.ts`
//! (`sanitizeFTS5Term`, `buildFTS5Query`, `searchFTS`).

use std::sync::LazyLock;

use regex::Regex;
use rusqlite::params;

use crate::error::Result;
use crate::store::Store;

/// One BM25 search hit.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Virtual path `qmd://collection/path`.
    pub filepath: String,
    /// `collection/path` display form.
    pub display_path: String,
    pub title: String,
    pub hash: String,
    pub docid: String,
    pub collection_name: String,
    pub body: String,
    pub context: Option<String>,
    /// Normalized score in `[0, 1)`, higher is better.
    pub score: f64,
}

/// Keep letters, numbers, apostrophes, and underscores; lowercase. A term
/// with no letter or number is empty: the tokenizer drops `'` and `_`, so it
/// would match nothing and its AND would empty the whole query.
pub fn sanitize_fts5_term(term: &str) -> String {
    if !term.chars().any(char::is_alphanumeric) {
        return String::new();
    }
    term.chars()
        .filter(|c| c.is_alphanumeric() || *c == '\'' || *c == '_')
        .collect::<String>()
        .to_lowercase()
}

fn sanitize_fts5_phrase(phrase: &str) -> String {
    phrase
        .split_whitespace()
        .map(sanitize_fts5_term)
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

static HYPHENATED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[\p{L}\p{N}][\p{L}\p{N}'-]*-[\p{L}\p{N}][\p{L}\p{N}'-]*$").expect("valid regex")
});

/// Hyphenated compound word (e.g. `multi-agent`, `DEC-0054`, `gpt-4`).
fn is_hyphenated_token(token: &str) -> bool {
    HYPHENATED.is_match(token)
}

fn sanitize_hyphenated_term(term: &str) -> String {
    term.split('-')
        .map(sanitize_fts5_term)
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

static DOTTED_PART: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[\p{L}\p{N}_]+$").expect("valid regex"));

/// Dotted version-like string (e.g. `2026.4.10`, `3.14.0`).
fn is_dotted_token(token: &str) -> bool {
    let parts: Vec<&str> = token.split('.').collect();
    parts.len() >= 2
        && parts
            .iter()
            .all(|p| !p.is_empty() && DOTTED_PART.is_match(p))
}

fn sanitize_dotted_parts(term: &str) -> Vec<String> {
    term.split('.')
        .map(sanitize_fts5_term)
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{t}\"*"))
        .collect()
}

/// Parse lex query syntax into an FTS5 MATCH expression.
///
/// Supports quoted phrases (exact), `-term` negation, hyphenated compounds
/// (phrase match), dotted versions (AND of parts), and plain terms (prefix
/// match). Returns `None` when nothing searchable remains.
pub fn build_fts5_query(query: &str) -> Option<String> {
    let mut positive: Vec<String> = Vec::new();
    let mut negative: Vec<String> = Vec::new();

    let s: Vec<char> = query.trim().chars().collect();
    let mut i = 0;
    while i < s.len() {
        while i < s.len() && s[i].is_whitespace() {
            i += 1;
        }
        if i >= s.len() {
            break;
        }

        let negated = s[i] == '-';
        if negated {
            i += 1;
        }

        if i < s.len() && s[i] == '"' {
            let start = i + 1;
            i += 1;
            while i < s.len() && s[i] != '"' {
                i += 1;
            }
            let phrase: String = s[start..i].iter().collect();
            i += 1;
            let sanitized = sanitize_fts5_phrase(phrase.trim());
            if !sanitized.is_empty() {
                let fts_phrase = format!("\"{sanitized}\"");
                if negated {
                    negative.push(fts_phrase);
                } else {
                    positive.push(fts_phrase);
                }
            }
        } else {
            let start = i;
            while i < s.len() && !s[i].is_whitespace() && s[i] != '"' {
                i += 1;
            }
            let term: String = s[start..i].iter().collect();

            if is_hyphenated_token(&term) {
                let sanitized = sanitize_hyphenated_term(&term);
                if !sanitized.is_empty() {
                    let fts_phrase = format!("\"{sanitized}\"");
                    if negated {
                        negative.push(fts_phrase);
                    } else {
                        positive.push(fts_phrase);
                    }
                }
            } else if is_dotted_token(&term) {
                let parts = sanitize_dotted_parts(&term);
                if !parts.is_empty() {
                    if negated {
                        negative.push(format!("({})", parts.join(" AND ")));
                    } else {
                        positive.extend(parts);
                    }
                }
            } else {
                let sanitized = sanitize_fts5_term(&term);
                if !sanitized.is_empty() {
                    let fts_term = format!("\"{sanitized}\"*");
                    if negated {
                        negative.push(fts_term);
                    } else {
                        positive.push(fts_term);
                    }
                }
            }
        }
    }

    if positive.is_empty() {
        return None;
    }

    let mut result = positive.join(" AND ");
    for neg in negative {
        result = format!("{result} NOT {neg}");
    }
    Some(result)
}

impl Store {
    /// BM25 full-text search. Scores are mapped from FTS5's negative BM25 to
    /// a stable `[0, 1)` via `|x| / (1 + |x|)` (higher is better).
    pub fn search_fts(
        &self,
        query: &str,
        limit: usize,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let Some(fts_query) = build_fts5_query(query) else {
            return Ok(Vec::new());
        };

        let fts_limit = if collection.is_some() {
            limit.saturating_mul(10)
        } else {
            limit
        };

        let collection_filter = if collection.is_some() {
            " AND d.collection = ?2"
        } else {
            ""
        };
        let sql = format!(
            "WITH fts_matches AS (
               SELECT rowid, bm25(documents_fts, 1.5, 4.0, 1.0) as bm25_score
               FROM documents_fts
               WHERE documents_fts MATCH ?1
               ORDER BY bm25_score ASC
               LIMIT {fts_limit}
             )
             SELECT
               'qmd://' || d.collection || '/' || d.path,
               d.collection || '/' || d.path,
               d.title,
               content.doc,
               d.hash,
               d.collection,
               fm.bm25_score
             FROM fts_matches fm
             JOIN documents d ON d.id = fm.rowid
             JOIN content ON content.hash = d.hash
             WHERE d.active = 1{collection_filter}
             ORDER BY fm.bm25_score ASC
             LIMIT {limit}"
        );

        let mut stmt = self.conn().prepare(&sql)?;
        let map_row = |r: &rusqlite::Row<'_>| -> rusqlite::Result<SearchResult> {
            let hash: String = r.get(4)?;
            let bm25: f64 = r.get(6)?;
            Ok(SearchResult {
                filepath: r.get(0)?,
                display_path: r.get(1)?,
                title: r.get(2)?,
                body: r.get(3)?,
                docid: super::docs::docid(&hash),
                hash,
                collection_name: r.get(5)?,
                context: None,
                score: bm25.abs() / (1.0 + bm25.abs()),
            })
        };
        let rows: Vec<SearchResult> = match collection {
            Some(c) => stmt
                .query_map(params![fts_query, c], map_row)?
                .collect::<std::result::Result<_, _>>()?,
            None => stmt
                .query_map(params![fts_query], map_row)?
                .collect::<std::result::Result<_, _>>()?,
        };

        let mut results = rows;
        for r in &mut results {
            r.context = self.context_for_file(&r.filepath)?;
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_terms_are_prefix_matched_and_anded() {
        assert_eq!(
            build_fts5_query("hello world").as_deref(),
            Some("\"hello\"* AND \"world\"*")
        );
    }

    #[test]
    fn quoted_phrases_are_exact() {
        assert_eq!(
            build_fts5_query("\"machine learning\"").as_deref(),
            Some("\"machine learning\"")
        );
    }

    #[test]
    fn negation_uses_not() {
        assert_eq!(
            build_fts5_query("performance -sports").as_deref(),
            Some("\"performance\"* NOT \"sports\"*")
        );
    }

    #[test]
    fn hyphenated_tokens_become_phrases() {
        assert_eq!(
            build_fts5_query("multi-agent memory").as_deref(),
            Some("\"multi agent\" AND \"memory\"*")
        );
        assert_eq!(
            build_fts5_query("DEC-0054").as_deref(),
            Some("\"dec 0054\"")
        );
        assert_eq!(
            build_fts5_query("-multi-agent memory").as_deref(),
            Some("\"memory\"* NOT \"multi agent\"")
        );
    }

    #[test]
    fn dotted_versions_are_anded_parts() {
        assert_eq!(
            build_fts5_query("2026.4.10").as_deref(),
            Some("\"2026\"* AND \"4\"* AND \"10\"*")
        );
    }

    #[test]
    fn punctuation_only_terms_are_dropped() {
        assert_eq!(
            build_fts5_query("foo _ bar ''").as_deref(),
            Some("\"foo\"* AND \"bar\"*")
        );
        assert_eq!(
            build_fts5_query("\"foo _ bar\"").as_deref(),
            Some("\"foo bar\"")
        );
    }

    #[test]
    fn negative_only_query_is_unsearchable() {
        assert_eq!(build_fts5_query("-sports"), None);
        assert_eq!(build_fts5_query("  "), None);
        assert_eq!(build_fts5_query("!!!"), None);
    }
}
