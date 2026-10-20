//! Hybrid query pipeline, ported from TS `store.ts` (`reciprocalRankFusion`,
//! `hybridQuery`, `structuredSearch`, lex/semantic validators) and
//! `cli/qmd.ts` (`parseStructuredQuery`).
//!
//! LLM query expansion and reranking are not implemented: a bare query fans out
//! to BM25 + vector search on the original text, and results are scored
//! `1/rrf_rank` (the TS `--no-rerank` path). The strong-signal probe runs so
//! its bypass decision is reported, but with no expansion stage there is
//! nothing for it to skip yet.

use std::collections::HashMap;

use crate::chunk::chunk_document_default;
use crate::embed::{Embedder, format_query_for_embedding};
use crate::error::{Error, Result};
use crate::snippet::extract_intent_terms;
use crate::store::{SearchResult, Store};

/// Top normalized BM25 score required for the strong-signal bypass.
pub const STRONG_SIGNAL_MIN_SCORE: f64 = 0.85;
/// Minimum top-minus-runner-up gap for the strong-signal bypass.
pub const STRONG_SIGNAL_MIN_GAP: f64 = 0.15;
/// RRF rank constant.
pub const RRF_K: f64 = 60.0;
/// Fused candidates kept for chunking (keeps ranks 31-40 visible to a
/// future reranker).
pub const CANDIDATE_LIMIT: usize = 40;
/// Per-list backend fetch limit.
pub const PER_LIST_LIMIT: usize = 20;
/// Weight of intent terms relative to query terms in best-chunk selection.
pub const INTENT_WEIGHT_CHUNK: f64 = 0.5;

/// Type of a structured sub-query, controlling backend routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Keyword variant, routes to FTS only.
    Lex,
    /// Semantic variant, routes to vector search only.
    Vec,
    /// Hypothetical document, routes to vector search only.
    Hyde,
}

impl QueryType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lex => "lex",
            Self::Vec => "vec",
            Self::Hyde => "hyde",
        }
    }
}

/// One typed line of a query document.
#[derive(Debug, Clone)]
pub struct TypedQuery {
    pub query_type: QueryType,
    pub query: String,
    /// 1-based source line in the query document.
    pub line: usize,
}

/// A parsed multi-line query document.
#[derive(Debug, Clone)]
pub struct StructuredQuery {
    pub searches: Vec<TypedQuery>,
    pub intent: Option<String>,
}

fn strip_prefix_ci<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    if line
        .get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
    {
        Some(line[prefix.len()..].trim())
    } else {
        None
    }
}

/// Parse structured search query syntax.
///
/// Lines starting with `lex:`, `vec:`, or `hyde:` are routed directly; one
/// optional `intent:` line adds a domain hint. Returns `None` for a plain
/// query (single line without prefix, or an `expand:` query, both of which
/// take the default pipeline).
///
/// # Errors
/// Fails on mixed `expand:`/typed lines, multiple plain lines, repeated or
/// lone `intent:` lines, and empty line bodies (same messages as the TS CLI).
pub fn parse_structured_query(query: &str) -> Result<Option<StructuredQuery>> {
    struct Line {
        trimmed: String,
        number: usize,
    }
    let lines: Vec<Line> = query
        .split('\n')
        .enumerate()
        .map(|(i, l)| Line {
            trimmed: l.trim().to_owned(),
            number: i + 1,
        })
        .filter(|l| !l.trimmed.is_empty())
        .collect();
    if lines.is_empty() {
        return Ok(None);
    }

    let mut typed: Vec<TypedQuery> = Vec::new();
    let mut intent: Option<String> = None;

    for line in &lines {
        if let Some(text) = strip_prefix_ci(&line.trimmed, "expand:") {
            if lines.len() > 1 {
                return Err(Error::Other(format!(
                    "Line {} starts with expand:, but query documents cannot mix expand \
                     with typed lines. Submit a single expand query instead.",
                    line.number
                )));
            }
            if text.is_empty() {
                return Err(Error::Other("expand: query must include text.".to_owned()));
            }
            return Ok(None);
        }

        if let Some(text) = strip_prefix_ci(&line.trimmed, "intent:") {
            if intent.is_some() {
                return Err(Error::Other(format!(
                    "Line {}: only one intent: line is allowed per query document.",
                    line.number
                )));
            }
            if text.is_empty() {
                return Err(Error::Other(format!(
                    "Line {}: intent: must include text.",
                    line.number
                )));
            }
            intent = Some(text.to_owned());
            continue;
        }

        let prefixed = [
            (QueryType::Lex, "lex:"),
            (QueryType::Vec, "vec:"),
            (QueryType::Hyde, "hyde:"),
        ]
        .into_iter()
        .find_map(|(t, p)| strip_prefix_ci(&line.trimmed, p).map(|text| (t, text)));
        if let Some((query_type, text)) = prefixed {
            if text.is_empty() {
                return Err(Error::Other(format!(
                    "Line {} ({}:) must include text.",
                    line.number,
                    query_type.as_str()
                )));
            }
            if text.contains('\r') || text.contains('\n') {
                return Err(Error::Other(format!(
                    "Line {} ({}:) contains a newline. Keep each query on a single line.",
                    line.number,
                    query_type.as_str()
                )));
            }
            typed.push(TypedQuery {
                query_type,
                query: text.to_owned(),
                line: line.number,
            });
            continue;
        }

        if lines.len() == 1 {
            return Ok(None);
        }
        return Err(Error::Other(format!(
            "Line {} is missing a lex:/vec:/hyde:/intent: prefix. Each line in a \
             query document must start with one.",
            line.number
        )));
    }

    if intent.is_some() && typed.is_empty() {
        return Err(Error::Other(
            "intent: cannot appear alone. Add at least one lex:, vec:, or hyde: line.".to_owned(),
        ));
    }

    Ok((!typed.is_empty()).then_some(StructuredQuery {
        searches: typed,
        intent,
    }))
}

/// Validate that a vec/hyde query doesn't use lex-only syntax.
/// Returns an error message if invalid.
pub fn validate_semantic_query(query: &str) -> Option<String> {
    let bytes = query.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'-'
            && (i == 0 || bytes[i - 1].is_ascii_whitespace())
            && bytes
                .get(i + 1)
                .is_some_and(|&c| c.is_ascii_alphanumeric() || c == b'_' || c == b'"')
        {
            return Some(
                "Negation (-term) is not supported in vec/hyde queries. \
                 Use lex for exclusions."
                    .to_owned(),
            );
        }
    }
    None
}

/// Validate lex query syntax. Returns an error message if invalid.
pub fn validate_lex_query(query: &str) -> Option<String> {
    if query.contains('\r') || query.contains('\n') {
        return Some(
            "Lex queries must be a single line. Remove newline characters or \
             split into separate lex: lines."
                .to_owned(),
        );
    }
    if query.matches('"').count() % 2 == 1 {
        return Some(
            "Lex query has an unmatched double quote (\"). Add the closing \
             quote or remove it."
                .to_owned(),
        );
    }
    None
}

/// Fuse ranked result lists with RRF: each list contributes
/// `weight / (k + rank + 1)` per document (rank 0-based), plus a top-rank
/// bonus on the best rank across lists (rank 0 → +0.05, rank 1-2 → +0.02).
/// Results are keyed by filepath and sorted by fused score.
pub fn reciprocal_rank_fusion(
    lists: &[Vec<SearchResult>],
    weights: &[f64],
    k: f64,
) -> Vec<SearchResult> {
    struct Entry {
        result: SearchResult,
        score: f64,
        top_rank: usize,
    }
    let mut entries: Vec<Entry> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();

    for (list_idx, list) in lists.iter().enumerate() {
        let weight = weights.get(list_idx).copied().unwrap_or(1.0);
        for (rank, result) in list.iter().enumerate() {
            let contribution = weight / (k + rank as f64 + 1.0);
            if let Some(&i) = index.get(&result.filepath) {
                entries[i].score += contribution;
                entries[i].top_rank = entries[i].top_rank.min(rank);
            } else {
                index.insert(result.filepath.clone(), entries.len());
                entries.push(Entry {
                    result: result.clone(),
                    score: contribution,
                    top_rank: rank,
                });
            }
        }
    }

    for entry in &mut entries {
        if entry.top_rank == 0 {
            entry.score += 0.05;
        } else if entry.top_rank <= 2 {
            entry.score += 0.02;
        }
    }

    entries.sort_by(|a, b| b.score.total_cmp(&a.score));
    entries
        .into_iter()
        .map(|e| {
            let mut r = e.result;
            r.score = e.score;
            r
        })
        .collect()
}

/// Options shared by [`hybrid_query`] and [`structured_search`].
#[derive(Debug, Clone)]
pub struct HybridOptions {
    /// Collections to search; empty searches every collection.
    pub collections: Vec<String>,
    pub limit: usize,
    pub min_score: f64,
    pub candidate_limit: usize,
    /// Domain intent hint for disambiguation (chunk selection weighting;
    /// disables the strong-signal bypass).
    pub intent: Option<String>,
}

impl Default for HybridOptions {
    fn default() -> Self {
        Self {
            collections: Vec::new(),
            limit: 10,
            min_score: 0.0,
            candidate_limit: CANDIDATE_LIMIT,
            intent: None,
        }
    }
}

/// One ranked list of up to [`PER_LIST_LIMIT`] hits restricted to
/// `collections` (empty = all). `fetch(k, collection)` is a store search.
///
/// One collection filters in SQL. Several take one wide unfiltered pass,
/// which usually holds enough hits from the set; when it does not, each
/// collection is searched and the hits merged by score (BM25 and vector
/// scores are absolute, so they compare across collections). Filtering only
/// the fused top `limit` instead can leave nothing, however many matches the
/// set holds.
fn ranked_list(
    collections: &[String],
    fetch: impl Fn(usize, Option<&str>) -> Result<Vec<SearchResult>>,
) -> Result<Vec<SearchResult>> {
    match collections {
        [] => fetch(PER_LIST_LIMIT, None),
        [one] => fetch(PER_LIST_LIMIT, Some(one)),
        many => {
            let wide_k = PER_LIST_LIMIT * 4;
            let wide = fetch(wide_k, None)?;
            let exhausted = wide.len() < wide_k;
            let mut list: Vec<SearchResult> = wide
                .into_iter()
                .filter(|r| many.contains(&r.collection_name))
                .collect();
            if list.len() < PER_LIST_LIMIT && !exhausted {
                list.clear();
                for c in many {
                    list.extend(fetch(PER_LIST_LIMIT, Some(c))?);
                }
                list.sort_by(|a, b| b.score.total_cmp(&a.score));
            }
            list.truncate(PER_LIST_LIMIT);
            Ok(list)
        }
    }
}

/// One hybrid search hit.
#[derive(Debug, Clone)]
pub struct HybridResult {
    /// Virtual path `qmd://collection/path`.
    pub filepath: String,
    pub display_path: String,
    pub title: String,
    pub body: String,
    /// Best chunk text (keyword overlap; the future rerank input).
    pub best_chunk: String,
    /// Byte offset of the best chunk in `body`.
    pub best_chunk_pos: usize,
    /// `1 / rrf_rank` (RRF ordering preserved; there is no rerank blend).
    pub score: f64,
    pub context: Option<String>,
    pub docid: String,
}

/// What the pipeline did, for CLI progress/debug output.
#[derive(Debug, Clone, Default)]
pub struct HybridReport {
    /// Top BM25 score when the strong-signal bypass fired.
    pub strong_signal: Option<f64>,
    /// Ranked lists that entered fusion.
    pub lists_fused: usize,
    /// Queries embedded for vector lookups.
    pub queries_embedded: usize,
    /// Vector search was applicable but skipped (no embedding backend).
    pub vector_skipped: bool,
}

/// Results plus a report of the pipeline path taken.
#[derive(Debug)]
pub struct HybridResponse {
    pub results: Vec<HybridResult>,
    pub report: HybridReport,
}

/// Hybrid search on a bare query: BM25 + vector search on the original text,
/// RRF fusion, best-chunk selection, `1/rrf_rank` scoring.
///
/// `embedder` may be `None` (BM25-only build); vector lists are then skipped
/// and reported via [`HybridReport::vector_skipped`].
///
/// # Errors
/// Propagates store and embedding failures.
pub fn hybrid_query(
    store: &Store,
    embedder: Option<&dyn Embedder>,
    query: &str,
    options: &HybridOptions,
) -> Result<HybridResponse> {
    let collections = &options.collections;
    let mut report = HybridReport::default();

    let initial_fts = ranked_list(collections, |k, c| store.search_fts(query, k, c))?;
    let top_score = initial_fts.first().map_or(0.0, |r| r.score);
    let second_score = initial_fts.get(1).map_or(0.0, |r| r.score);
    if options.intent.is_none()
        && !initial_fts.is_empty()
        && top_score >= STRONG_SIGNAL_MIN_SCORE
        && (top_score - second_score) >= STRONG_SIGNAL_MIN_GAP
    {
        report.strong_signal = Some(top_score);
    }

    let mut lists: Vec<Vec<SearchResult>> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    if !initial_fts.is_empty() {
        lists.push(initial_fts);
        weights.push(2.0);
    }

    if store.vec_table_exists()? {
        if let Some(embedder) = embedder {
            let formatted = format_query_for_embedding(query, embedder.model_uri());
            if let Some(embedding) = embedder.embed(&formatted)? {
                report.queries_embedded = 1;
                let vec_results =
                    ranked_list(collections, |k, c| store.search_vec(&embedding, k, c))?;
                if !vec_results.is_empty() {
                    lists.push(vec_results);
                    weights.push(2.0);
                }
            }
        } else {
            report.vector_skipped = true;
        }
    }

    report.lists_fused = lists.len();
    let fused = reciprocal_rank_fusion(&lists, &weights, RRF_K);
    let results = finalize_candidates(store, fused, query, options)?;
    Ok(HybridResponse { results, report })
}

/// Structured search: execute pre-typed queries without expansion.
/// `lex` routes to FTS, `vec`/`hyde` to vector search (one batch embed);
/// the first ranked list gets 2x RRF weight.
///
/// # Errors
/// Fails on invalid sub-query syntax, and propagates store/embedding errors.
pub fn structured_search(
    store: &Store,
    embedder: Option<&dyn Embedder>,
    searches: &[TypedQuery],
    options: &HybridOptions,
) -> Result<HybridResponse> {
    let mut report = HybridReport::default();
    if searches.is_empty() {
        return Ok(HybridResponse {
            results: Vec::new(),
            report,
        });
    }

    for search in searches {
        let error = match search.query_type {
            QueryType::Lex => validate_lex_query(&search.query),
            QueryType::Vec | QueryType::Hyde => validate_semantic_query(&search.query),
        };
        if let Some(error) = error {
            return Err(Error::Other(format!(
                "Line {} ({}): {error}",
                search.line,
                search.query_type.as_str()
            )));
        }
    }

    let collections = &options.collections;
    let mut lists: Vec<Vec<SearchResult>> = Vec::new();

    for search in searches {
        if search.query_type == QueryType::Lex {
            let fts_results =
                ranked_list(collections, |k, c| store.search_fts(&search.query, k, c))?;
            if !fts_results.is_empty() {
                lists.push(fts_results);
            }
        }
    }

    let vec_searches: Vec<&TypedQuery> = searches
        .iter()
        .filter(|s| matches!(s.query_type, QueryType::Vec | QueryType::Hyde))
        .collect();
    if !vec_searches.is_empty() && store.vec_table_exists()? {
        if let Some(embedder) = embedder {
            let texts: Vec<String> = vec_searches
                .iter()
                .map(|s| format_query_for_embedding(&s.query, embedder.model_uri()))
                .collect();
            report.queries_embedded = texts.len();
            let embeddings = embedder.embed_batch(&texts)?;
            for embedding in embeddings.iter().flatten() {
                let vec_results =
                    ranked_list(collections, |k, c| store.search_vec(embedding, k, c))?;
                if !vec_results.is_empty() {
                    lists.push(vec_results);
                }
            }
        } else {
            report.vector_skipped = true;
        }
    }

    let weights: Vec<f64> = (0..lists.len())
        .map(|i| if i == 0 { 2.0 } else { 1.0 })
        .collect();
    report.lists_fused = lists.len();
    let fused = reciprocal_rank_fusion(&lists, &weights, RRF_K);

    let primary_query = searches
        .iter()
        .find(|s| s.query_type == QueryType::Lex)
        .or_else(|| searches.iter().find(|s| s.query_type == QueryType::Vec))
        .unwrap_or(&searches[0])
        .query
        .clone();

    let results = finalize_candidates(store, fused, &primary_query, options)?;
    Ok(HybridResponse { results, report })
}

/// Steps shared by both pipelines: slice to `candidate_limit`, chunk each
/// candidate and pick the best chunk by keyword overlap (intent terms at
/// [`INTENT_WEIGHT_CHUNK`]), score `1/rrf_rank`, filter by `min_score`,
/// slice to `limit`. Fused results are already unique per file.
fn finalize_candidates(
    store: &Store,
    fused: Vec<SearchResult>,
    primary_query: &str,
    options: &HybridOptions,
) -> Result<Vec<HybridResult>> {
    let mut candidates = fused;
    candidates.truncate(options.candidate_limit);

    let query_terms: Vec<String> = primary_query
        .to_lowercase()
        .split_whitespace()
        .filter(|t| t.chars().count() > 2)
        .map(str::to_owned)
        .collect();
    let intent_terms: Vec<String> = options
        .intent
        .as_deref()
        .map(extract_intent_terms)
        .unwrap_or_default();

    let mut results = Vec::new();
    for (i, cand) in candidates.into_iter().enumerate() {
        let rrf_score = 1.0 / (i as f64 + 1.0);
        if rrf_score < options.min_score {
            break;
        }

        let chunks = chunk_document_default(&cand.body);
        let (best_chunk, best_chunk_pos) = if chunks.is_empty() {
            (cand.body.clone(), 0)
        } else {
            let mut best_idx = 0usize;
            let mut best_score = -1.0f64;
            for (idx, chunk) in chunks.iter().enumerate() {
                let chunk_lower = chunk.text.to_lowercase();
                let mut score = query_terms
                    .iter()
                    .filter(|t| chunk_lower.contains(t.as_str()))
                    .count() as f64;
                for term in &intent_terms {
                    if chunk_lower.contains(term.as_str()) {
                        score += INTENT_WEIGHT_CHUNK;
                    }
                }
                if score > best_score {
                    best_score = score;
                    best_idx = idx;
                }
            }
            (chunks[best_idx].text.clone(), chunks[best_idx].pos)
        };

        results.push(HybridResult {
            context: store.context_for_file(&cand.filepath)?,
            filepath: cand.filepath,
            display_path: cand.display_path,
            title: cand.title,
            body: cand.body,
            best_chunk,
            best_chunk_pos,
            score: rrf_score,
            docid: cand.docid,
        });
        if results.len() == options.limit {
            break;
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranked(path: &str, score: f64) -> SearchResult {
        SearchResult {
            filepath: format!("qmd://c/{path}"),
            display_path: format!("c/{path}"),
            title: path.to_owned(),
            hash: "h".to_owned(),
            docid: "d".to_owned(),
            collection_name: "c".to_owned(),
            body: String::new(),
            context: None,
            score,
        }
    }

    #[test]
    fn plain_and_expand_queries_are_unstructured() {
        assert!(parse_structured_query("CAP theorem").unwrap().is_none());
        assert!(parse_structured_query("").unwrap().is_none());
        assert!(
            parse_structured_query("expand: CAP theorem")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn typed_lines_parse_with_intent() {
        let parsed = parse_structured_query(
            "intent: databases\nlex: CAP\nvec: consistency\nhyde: A doc about CAP",
        )
        .unwrap()
        .unwrap();
        assert_eq!(parsed.intent.as_deref(), Some("databases"));
        assert_eq!(parsed.searches.len(), 3);
        assert_eq!(parsed.searches[0].query_type, QueryType::Lex);
        assert_eq!(parsed.searches[0].query, "CAP");
        assert_eq!(parsed.searches[0].line, 2);
        assert_eq!(parsed.searches[2].query_type, QueryType::Hyde);
    }

    #[test]
    fn prefixes_are_case_insensitive_and_whitespace_tolerant() {
        let parsed = parse_structured_query("LEX:   CAP theorem  ")
            .unwrap()
            .unwrap();
        assert_eq!(parsed.searches[0].query, "CAP theorem");
    }

    #[test]
    fn multibyte_text_at_a_prefix_boundary_is_plain() {
        assert!(parse_structured_query("café au lait").unwrap().is_none());
        assert!(parse_structured_query("日本語の検索").unwrap().is_none());
    }

    #[test]
    fn parse_errors_match_ts_rules() {
        assert!(parse_structured_query("CAP\nconsistency").is_err());
        assert!(parse_structured_query("expand: CAP\nlex: theorem").is_err());
        assert!(parse_structured_query("expand:").is_err());
        assert!(parse_structured_query("intent: a\nintent: b\nlex: q").is_err());
        assert!(parse_structured_query("intent: databases").is_err());
        assert!(parse_structured_query("intent:\nlex: q").is_err());
        assert!(parse_structured_query("lex:\nvec: q").is_err());
    }

    #[test]
    fn validators_flag_bad_syntax() {
        assert!(validate_lex_query("\"unbalanced quote").is_some());
        assert!(validate_lex_query("a\nb").is_some());
        assert!(validate_lex_query("\"machine learning\" -sports").is_none());
        assert!(validate_semantic_query("-negated term").is_some());
        assert!(validate_semantic_query("stop -word").is_some());
        assert!(validate_semantic_query("real-time write-ahead log").is_none());
    }

    #[test]
    fn rrf_fuses_weights_and_top_rank_bonus() {
        let lists = vec![
            vec![ranked("a", 0.9), ranked("b", 0.5)],
            vec![ranked("b", 0.8), ranked("a", 0.7)],
        ];
        let fused = reciprocal_rank_fusion(&lists, &[2.0, 1.0], 60.0);
        assert_eq!(fused.len(), 2);
        let a = fused.iter().find(|r| r.display_path == "c/a").unwrap();
        let b = fused.iter().find(|r| r.display_path == "c/b").unwrap();
        assert!((a.score - (2.0 / 61.0 + 1.0 / 62.0 + 0.05)).abs() < 1e-12);
        assert!((b.score - (2.0 / 62.0 + 1.0 / 61.0 + 0.05)).abs() < 1e-12);
        assert_eq!(fused[0].display_path, "c/a", "higher-weighted rank wins");
    }

    #[test]
    fn rrf_top_rank_bonus_tiers() {
        let lists = vec![vec![
            ranked("a", 0.9),
            ranked("b", 0.8),
            ranked("c", 0.7),
            ranked("d", 0.6),
        ]];
        let fused = reciprocal_rank_fusion(&lists, &[1.0], 60.0);
        assert!((fused[0].score - (1.0 / 61.0 + 0.05)).abs() < 1e-12);
        assert!((fused[1].score - (1.0 / 62.0 + 0.02)).abs() < 1e-12);
        assert!((fused[2].score - (1.0 / 63.0 + 0.02)).abs() < 1e-12);
        assert!((fused[3].score - (1.0 / 64.0)).abs() < 1e-12);
    }
}
