//! FTS5 query building and BM25 search, ported from TS `store.ts`
//! (`sanitizeFTS5Term`, `buildFTS5Query`, `searchFTS`).

use std::sync::LazyLock;

use regex::Regex;
use rusqlite::params_from_iter;
use rusqlite::types::Value as SqlValue;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::metadata::MetadataFilter;
use crate::store::Store;
use crate::store::sections::{SectionRef, section_id};

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
    /// Set when the hit is one section of its file (a collection with
    /// `sectionLevel`); `body` is then the section's text.
    pub section: Option<SectionRef>,
}

impl SearchResult {
    /// The hit's identity, which fusion and dedupe key on: the file's
    /// virtual path, or for a section [`section_id`] of it.
    pub fn key(&self) -> String {
        match &self.section {
            Some(s) => section_id(&self.filepath, s.seq),
            None => self.filepath.clone(),
        }
    }
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

/// How the positive terms of a lex query combine. `Any` is the default;
/// [`Store::search_fts`] keeps `All`, the TS behaviour.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TermMatch {
    /// Every term must be present (FTS5 `AND`; the TS qmd behaviour).
    All,
    /// Any term may match (FTS5 `OR`); BM25 ranks documents holding more of
    /// them higher.
    #[default]
    Any,
}

impl TermMatch {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Any => "any",
        }
    }
}

impl std::str::FromStr for TermMatch {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "all" | "and" => Ok(Self::All),
            "any" | "or" => Ok(Self::Any),
            other => Err(format!("unknown term match '{other}': use all or any")),
        }
    }
}

/// [`build_fts5_query_with`] in [`TermMatch::All`] mode.
pub fn build_fts5_query(query: &str) -> Option<String> {
    build_fts5_query_with(query, TermMatch::All)
}

/// Parse lex query syntax into an FTS5 MATCH expression.
///
/// Supports quoted phrases (exact), `-term` negation, hyphenated compounds
/// (phrase match), dotted versions (AND of parts), and plain terms (prefix
/// match). Positive items combine by `mode`; a phrase, compound or dotted
/// version stays one item either way, and negations always exclude. Returns
/// `None` when nothing searchable remains.
pub fn build_fts5_query_with(query: &str, mode: TermMatch) -> Option<String> {
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
                    } else if mode == TermMatch::Any && parts.len() > 1 {
                        positive.push(format!("({})", parts.join(" AND ")));
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

    let mut result = match mode {
        TermMatch::All => positive.join(" AND "),
        TermMatch::Any if positive.len() > 1 && !negative.is_empty() => {
            format!("({})", positive.join(" OR "))
        }
        TermMatch::Any => positive.join(" OR "),
    };
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
        self.search_fts_with(query, limit, collection, TermMatch::All)
    }

    /// [`Store::search_fts`] with the terms combined by `mode`.
    pub fn search_fts_with(
        &self,
        query: &str,
        limit: usize,
        collection: Option<&str>,
        mode: TermMatch,
    ) -> Result<Vec<SearchResult>> {
        self.search_fts_filtered(query, limit, collection, mode, None)
    }

    /// [`Store::search_fts_with`] restricted to documents a metadata
    /// `filter` admits.
    pub fn search_fts_filtered(
        &self,
        query: &str,
        limit: usize,
        collection: Option<&str>,
        mode: TermMatch,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        let collections: Vec<String> = collection.map(str::to_owned).into_iter().collect();
        self.search_fts_in(query, limit, &collections, mode, filter)
    }

    /// BM25 search over `collections` (empty searches every collection),
    /// restricted to documents a metadata `filter` admits. A search
    /// restricted either way ranks the whole match set, not a top-k window,
    /// so a match below stronger ones outside the restriction is still
    /// found.
    pub fn search_fts_in(
        &self,
        query: &str,
        limit: usize,
        collections: &[String],
        mode: TermMatch,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        let Some(fts_query) = build_fts5_query_with(query, mode) else {
            return Ok(Vec::new());
        };
        let scope = Scope {
            collections,
            filter,
        };

        let levels = self.section_levels()?;
        let mut results = if levels.is_empty() {
            self.search_documents_fts(&fts_query, limit, &scope, false)?
        } else {
            let in_sections = |c: &String| levels.contains_key(c.as_str());
            let mut results = Vec::new();
            if collections.is_empty() || !collections.iter().all(in_sections) {
                results = self.search_documents_fts(&fts_query, limit, &scope, true)?;
            }
            if collections.is_empty() || collections.iter().any(in_sections) {
                results.extend(self.search_sections_fts(&fts_query, limit, &scope)?);
                results.sort_by(|a, b| b.score.total_cmp(&a.score));
                results.truncate(limit);
            }
            results
        };
        for r in &mut results {
            r.context = self.context_for_file(
                &r.collection_name,
                super::relative_path(&r.filepath, &r.collection_name),
            )?;
        }
        Ok(results)
    }

    /// BM25 over `documents_fts`, one result per file. `skip_sections`
    /// drops files of section collections, whose rows a TS trigger may have
    /// re-added since the last section sync.
    fn search_documents_fts(
        &self,
        fts_query: &str,
        limit: usize,
        scope: &Scope<'_>,
        skip_sections: bool,
    ) -> Result<Vec<SearchResult>> {
        let (mut collection_filter, params) = scope.where_sql(fts_query);
        if skip_sections {
            collection_filter.push_str(
                " AND NOT EXISTS (SELECT 1 FROM store_config \
                 WHERE key = 'section_level:' || d.collection)",
            );
        }
        let fts_limit = scope.fts_limit(limit);
        let sql = format!(
            "WITH fts_matches AS MATERIALIZED (
               SELECT rowid, bm25(documents_fts, 1.5, 4.0, 1.0) as bm25_score
               FROM documents_fts
               WHERE documents_fts MATCH ?1
               ORDER BY bm25_score ASC
               {fts_limit}
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
                section: None,
            })
        };
        let rows: Vec<SearchResult> = stmt
            .query_map(params_from_iter(params), map_row)?
            .collect::<std::result::Result<_, _>>()?;
        Ok(rows)
    }
}

/// What a BM25 search is restricted to: a set of collections (empty is
/// every collection), a metadata filter.
pub(crate) struct Scope<'a> {
    pub collections: &'a [String],
    pub filter: Option<&'a MetadataFilter>,
}

impl Scope<'_> {
    /// The `AND ...` restriction over `documents d` and every binding of the
    /// statement, `fts_query` first as `?1`.
    pub fn where_sql(&self, fts_query: &str) -> (String, Vec<SqlValue>) {
        let mut sql = String::new();
        let mut params = vec![SqlValue::Text(fts_query.to_owned())];
        if !self.collections.is_empty() {
            sql.push_str(" AND d.collection IN (SELECT value FROM json_each(?2))");
            params.push(super::metadata::collections_json(self.collections));
        }
        if let Some(filter) = self.filter {
            let compiled = super::metadata::search_filter_sql(filter);
            sql.push_str(&compiled.sql);
            params.extend(compiled.params);
        }
        (sql, params)
    }

    /// The candidate window of the FTS CTE: `limit` rows when nothing
    /// restricts the search, since the top of the index is then the answer.
    /// A collection or a metadata filter ranks the whole match set, so a
    /// match in scope is found however many stronger ones lie outside it.
    /// The CTE is `MATERIALIZED` so that FTS5 still runs before the
    /// restriction when no window forces that order (the query-planner
    /// hazard the TS implementation documents).
    pub fn fts_limit(&self, limit: usize) -> String {
        if self.filter.is_some() || !self.collections.is_empty() {
            String::new()
        } else {
            format!("LIMIT {limit}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_collection_scope_ranks_the_whole_match_set() {
        let store = Store::open_in_memory().unwrap();
        let add = |collection: &str, path: &str, body: &str| {
            let hash = crate::store::hash_content(body);
            store.insert_content(&hash, body, "t").unwrap();
            store
                .insert_document(collection, path, "Doc", &hash, "t", "t")
                .unwrap();
        };
        let limit = 5;
        for i in 0..limit * 10 + 10 {
            add(
                "big",
                &format!("{i}.md"),
                &format!("quorum quorum quorum {i}"),
            );
        }
        let filler = "Ordinary prose about unrelated matters goes on here. ".repeat(40);
        add("small", "one.md", &format!("{filler}Mentions quorum once."));

        let hits = store
            .search_fts_with("quorum", limit, Some("small"), TermMatch::Any)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].display_path, "small/one.md");

        add(
            "other",
            "two.md",
            &format!("{filler}Quorum, mentioned once."),
        );
        let set = ["small".to_owned(), "other".to_owned()];
        let hits = store
            .search_fts_in("quorum", limit, &set, TermMatch::Any, None)
            .unwrap();
        let mut found: Vec<&str> = hits.iter().map(|h| h.display_path.as_str()).collect();
        found.sort_unstable();
        assert_eq!(found, ["other/two.md", "small/one.md"]);

        let unscoped = store
            .search_fts_with("quorum", limit, None, TermMatch::Any)
            .unwrap();
        assert!(unscoped.iter().all(|h| h.collection_name == "big"));
    }

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
    fn any_mode_ors_items_and_keeps_groups_whole() {
        let any = |q| build_fts5_query_with(q, TermMatch::Any);
        assert_eq!(
            any("hello world").as_deref(),
            Some("\"hello\"* OR \"world\"*")
        );
        assert_eq!(
            any("\"machine learning\" multi-agent 2026.4").as_deref(),
            Some("\"machine learning\" OR \"multi agent\" OR (\"2026\"* AND \"4\"*)")
        );
        assert_eq!(
            any("cache store -redis").as_deref(),
            Some("(\"cache\"* OR \"store\"*) NOT \"redis\"*")
        );
        assert_eq!(
            any("cache -redis").as_deref(),
            Some("\"cache\"* NOT \"redis\"*")
        );
        assert_eq!(any("-redis"), None);
    }

    #[test]
    fn term_match_parses_both_spellings() {
        assert_eq!("any".parse::<TermMatch>(), Ok(TermMatch::Any));
        assert_eq!("OR".parse::<TermMatch>(), Ok(TermMatch::Any));
        assert_eq!("all".parse::<TermMatch>(), Ok(TermMatch::All));
        assert!("some".parse::<TermMatch>().is_err());
    }

    #[test]
    fn negative_only_query_is_unsearchable() {
        assert_eq!(build_fts5_query("-sports"), None);
        assert_eq!(build_fts5_query("  "), None);
        assert_eq!(build_fts5_query("!!!"), None);
    }
}
