//! Hybrid query pipeline, ported from TS `store.ts` (`reciprocalRankFusion`,
//! `hybridQuery`, `structuredSearch`, lex/semantic validators) and
//! `cli/qmd.ts` (`parseStructuredQuery`).
//!
//! LLM query expansion is not implemented: a bare query fans out to BM25 +
//! vector search on the original text. Results are scored `1/rrf_rank` (the
//! TS `--no-rerank` path) unless a [`Reranker`] is passed. The strong-signal
//! probe runs so its bypass decision is reported, but with no expansion stage
//! there is nothing for it to skip yet.

use std::collections::HashMap;

use serde::Serialize;

use crate::chunk::{Chunk, chunk_document_for, token_chunks_at};
use crate::embed::{Embedder, format_query_for_embedding};
use crate::error::{Error, Result};
use crate::rerank::{Reranker, blend, blend_fused_weight, rerank_cached};
use crate::snippet::extract_intent_terms;
use crate::store::{SearchResult, Store, TermMatch};

/// Top normalized BM25 score required for the strong-signal bypass.
pub const STRONG_SIGNAL_MIN_SCORE: f64 = 0.85;
/// Minimum top-minus-runner-up gap for the strong-signal bypass.
pub const STRONG_SIGNAL_MIN_GAP: f64 = 0.15;
/// RRF rank constant.
pub const RRF_K: f64 = 60.0;
/// Fused candidates kept for chunking (keeps ranks 31-40 visible to the
/// reranker).
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

/// The backend a ranked list came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// BM25 over FTS5.
    Fts,
    /// Vector KNN.
    Vec,
}

/// What produced one ranked list: a backend and the query it ran.
#[derive(Debug, Clone, Serialize)]
pub struct ListSource {
    pub backend: Backend,
    /// `original` for the plain query, else the typed line's `lex`, `vec` or
    /// `hyde`.
    pub kind: &'static str,
    pub query: String,
}

/// One ranked list's share of a result's fused score.
#[derive(Debug, Clone, Serialize)]
pub struct Contribution {
    #[serde(flatten)]
    pub source: ListSource,
    /// 1-based rank in that list.
    pub rank: usize,
    /// The list's RRF weight.
    pub weight: f64,
    /// The backend's normalized score (BM25 `|s|/(1+|s|)`, or cosine
    /// similarity).
    pub backend_score: f64,
    /// `weight / (k + rank)`.
    pub rrf: f64,
}

/// How the rerank blend set a result's final score.
#[derive(Debug, Clone, Serialize)]
pub struct RerankTrace {
    pub rerank_score: f64,
    /// Weight of `1/rrf_rank`; the rerank score gets `1 - fused_weight`.
    pub fused_weight: f64,
}

/// Why a result sits where it does: each list's contribution, the fused RRF
/// score and rank, and the final score. Built beside fusion, never fed back
/// into it, so asking for it changes no ranking or score.
#[derive(Debug, Clone, Serialize)]
pub struct Explain {
    pub contributions: Vec<Contribution>,
    /// Sum of the contributions.
    pub rrf_base: f64,
    /// +0.05 for a first place in any list, +0.02 for a second or third.
    pub rrf_bonus: f64,
    /// `rrf_base + rrf_bonus`, the score fusion sorted by.
    pub rrf_score: f64,
    /// 1-based rank after fusion.
    pub rrf_rank: usize,
    /// `1 / rrf_rank`, the final score when nothing reranked.
    pub position_score: f64,
    /// Present when the reranker ran.
    pub rerank: Option<RerankTrace>,
    pub final_score: f64,
}

impl std::fmt::Display for Explain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for c in &self.contributions {
            let query = if c.source.kind == "original" {
                String::new()
            } else {
                format!(" \"{}\"", preview(&c.source.query, 40))
            };
            writeln!(
                f,
                "  {:<3} {}{query} #{} x{} score={:.4} rrf={:.5}",
                match c.source.backend {
                    Backend::Fts => "fts",
                    Backend::Vec => "vec",
                },
                c.source.kind,
                c.rank,
                c.weight,
                c.backend_score,
                c.rrf
            )?;
        }
        writeln!(
            f,
            "  rrf: base={:.5} + bonus={:.2} = {:.5}, fused rank {}",
            self.rrf_base, self.rrf_bonus, self.rrf_score, self.rrf_rank
        )?;
        match &self.rerank {
            None => write!(
                f,
                "  final: 1/{} = {:.4} (no rerank)",
                self.rrf_rank, self.final_score
            ),
            Some(r) => write!(
                f,
                "  final: {:.2} x 1/{} + {:.2} x rerank {:.4} = {:.4}",
                r.fused_weight,
                self.rrf_rank,
                1.0 - r.fused_weight,
                r.rerank_score,
                self.final_score
            ),
        }
    }
}

fn preview(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_owned()
    } else {
        format!(
            "{}...",
            text.chars().take(max_chars - 3).collect::<String>()
        )
    }
}

/// Per-file traces of [`reciprocal_rank_fusion`] over the same lists, keyed
/// by filepath, with the RRF fields set and the rank and final fields left
/// for [`finalize_candidates`]. The sums run in fusion's order, so
/// `rrf_score` equals the fused score exactly.
fn rrf_traces(
    lists: &[Vec<SearchResult>],
    weights: &[f64],
    sources: &[ListSource],
    k: f64,
) -> HashMap<String, Explain> {
    let mut traces: HashMap<String, (Explain, usize)> = HashMap::new();
    for (list_idx, list) in lists.iter().enumerate() {
        let weight = weights.get(list_idx).copied().unwrap_or(1.0);
        for (rank, result) in list.iter().enumerate() {
            let contribution = Contribution {
                source: sources[list_idx].clone(),
                rank: rank + 1,
                weight,
                backend_score: result.score,
                rrf: weight / (k + rank as f64 + 1.0),
            };
            let (trace, top_rank) = traces.entry(result.filepath.clone()).or_insert_with(|| {
                (
                    Explain {
                        contributions: Vec::new(),
                        rrf_base: 0.0,
                        rrf_bonus: 0.0,
                        rrf_score: 0.0,
                        rrf_rank: 0,
                        position_score: 0.0,
                        rerank: None,
                        final_score: 0.0,
                    },
                    rank,
                )
            });
            trace.rrf_base += contribution.rrf;
            *top_rank = (*top_rank).min(rank);
            trace.contributions.push(contribution);
        }
    }
    traces
        .into_iter()
        .map(|(file, (mut trace, top_rank))| {
            trace.rrf_bonus = match top_rank {
                0 => 0.05,
                1 | 2 => 0.02,
                _ => 0.0,
            };
            trace.rrf_score = trace.rrf_base + trace.rrf_bonus;
            (file, trace)
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
    /// How BM25 terms combine, in the probe and in every `lex` sub-query.
    pub term_match: TermMatch,
    /// Which chunk of each candidate the reranker reads.
    pub rerank_chunk: RerankChunk,
    /// Attach an [`Explain`] trace to each result.
    pub explain: bool,
}

impl Default for HybridOptions {
    fn default() -> Self {
        Self {
            collections: Vec::new(),
            limit: 10,
            min_score: 0.0,
            candidate_limit: CANDIDATE_LIMIT,
            intent: None,
            term_match: TermMatch::default(),
            rerank_chunk: RerankChunk::default(),
            explain: false,
        }
    }
}

/// Which chunk of a candidate the reranker reads. Without a query embedding,
/// and for a candidate with no stored vectors, every rule reads the
/// [`Self::Overlap`] chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RerankChunk {
    /// Most query terms (TS step 5).
    #[default]
    Overlap,
    /// The stored chunk whose embedding is nearest the query's — for a vector
    /// hit, the chunk vector search matched.
    Nearest,
    /// The `n` nearest stored chunks, each reranked; the best score counts.
    BestOfNearest(usize),
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
    /// Best chunk text: the keyword-overlap chunk, or when reranked the
    /// [`RerankChunk`] chunk that scored best.
    pub best_chunk: String,
    /// Byte offset of the best chunk in `body`.
    pub best_chunk_pos: usize,
    /// `1 / rrf_rank`, or its blend with the rerank score when reranked.
    pub score: f64,
    /// 1-based rank in the fused list.
    pub rrf_rank: usize,
    /// The reranker's score, when reranked.
    pub rerank_score: Option<f64>,
    pub context: Option<String>,
    pub docid: String,
    /// Set when [`HybridOptions::explain`] asked for it.
    pub explain: Option<Explain>,
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
    /// Candidates scored by the reranker (0 when it did not run).
    pub reranked: usize,
    /// Of those, scored by the model rather than the cache.
    pub rerank_fresh: usize,
}

/// Results plus a report of the pipeline path taken.
#[derive(Debug)]
pub struct HybridResponse {
    pub results: Vec<HybridResult>,
    pub report: HybridReport,
}

/// Hybrid search on a bare query: BM25 + vector search on the original text,
/// RRF fusion, best-chunk selection, `1/rrf_rank` scoring or, with a
/// `reranker`, the position-aware rerank blend.
///
/// `embedder` may be `None` (BM25-only build); vector lists are then skipped
/// and reported via [`HybridReport::vector_skipped`].
///
/// # Errors
/// Propagates store and embedding failures.
pub fn hybrid_query(
    store: &Store,
    embedder: Option<&dyn Embedder>,
    reranker: Option<&dyn Reranker>,
    query: &str,
    options: &HybridOptions,
) -> Result<HybridResponse> {
    let collections = &options.collections;
    let mut report = HybridReport::default();

    let initial_fts = ranked_list(collections, |k, c| {
        store.search_fts_with(query, k, c, options.term_match)
    })?;
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
    let mut sources: Vec<ListSource> = Vec::new();
    let original = |backend| ListSource {
        backend,
        kind: "original",
        query: query.to_owned(),
    };
    if !initial_fts.is_empty() {
        lists.push(initial_fts);
        weights.push(2.0);
        sources.push(original(Backend::Fts));
    }

    let mut query_embedding = None;
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
                    sources.push(original(Backend::Vec));
                }
                query_embedding = Some(embedding);
            }
        } else {
            report.vector_skipped = true;
        }
    }

    report.lists_fused = lists.len();
    let fused = reciprocal_rank_fusion(&lists, &weights, RRF_K);
    let traces = options
        .explain
        .then(|| rrf_traces(&lists, &weights, &sources, RRF_K));
    let rerank = reranker.map(|r| RerankInput {
        reranker: r,
        query,
        embedding: query_embedding.as_deref(),
    });
    let results = finalize_candidates(store, fused, traces, query, rerank, options, &mut report)?;
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
    reranker: Option<&dyn Reranker>,
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
    let mut sources: Vec<ListSource> = Vec::new();
    let typed = |backend, search: &TypedQuery| ListSource {
        backend,
        kind: search.query_type.as_str(),
        query: search.query.clone(),
    };

    for search in searches {
        if search.query_type == QueryType::Lex {
            let fts_results = ranked_list(collections, |k, c| {
                store.search_fts_with(&search.query, k, c, options.term_match)
            })?;
            if !fts_results.is_empty() {
                lists.push(fts_results);
                sources.push(typed(Backend::Fts, search));
            }
        }
    }

    let vec_searches: Vec<&TypedQuery> = searches
        .iter()
        .filter(|s| matches!(s.query_type, QueryType::Vec | QueryType::Hyde))
        .collect();
    let mut rerank_embedding: Option<Vec<f32>> = None;
    if !vec_searches.is_empty() && store.vec_table_exists()? {
        if let Some(embedder) = embedder {
            let texts: Vec<String> = vec_searches
                .iter()
                .map(|s| format_query_for_embedding(&s.query, embedder.model_uri()))
                .collect();
            report.queries_embedded = texts.len();
            let mut embeddings = embedder.embed_batch(&texts)?;
            for (search, embedding) in vec_searches.iter().zip(&embeddings) {
                let Some(embedding) = embedding else { continue };
                let vec_results =
                    ranked_list(collections, |k, c| store.search_vec(embedding, k, c))?;
                if !vec_results.is_empty() {
                    lists.push(vec_results);
                    sources.push(typed(Backend::Vec, search));
                }
            }
            let first_vec = vec_searches
                .iter()
                .position(|s| s.query_type == QueryType::Vec)
                .unwrap_or(0);
            rerank_embedding = embeddings.get_mut(first_vec).and_then(Option::take);
        } else {
            report.vector_skipped = true;
        }
    }

    let weights: Vec<f64> = (0..lists.len())
        .map(|i| if i == 0 { 2.0 } else { 1.0 })
        .collect();
    report.lists_fused = lists.len();
    let fused = reciprocal_rank_fusion(&lists, &weights, RRF_K);
    let traces = options
        .explain
        .then(|| rrf_traces(&lists, &weights, &sources, RRF_K));

    let primary_query = searches
        .iter()
        .find(|s| s.query_type == QueryType::Lex)
        .or_else(|| searches.iter().find(|s| s.query_type == QueryType::Vec))
        .unwrap_or(&searches[0])
        .query
        .clone();

    let rerank_query = searches
        .iter()
        .find(|s| s.query_type == QueryType::Vec)
        .map_or(primary_query.as_str(), |s| s.query.as_str());
    let rerank = reranker.map(|r| RerankInput {
        reranker: r,
        query: rerank_query,
        embedding: rerank_embedding.as_deref(),
    });
    let results = finalize_candidates(
        store,
        fused,
        traces,
        &primary_query,
        rerank,
        options,
        &mut report,
    )?;
    Ok(HybridResponse { results, report })
}

/// What the rerank step reads: the model, the query it is asked, and that
/// query's embedding when one was made (for [`RerankChunk::Nearest`]).
#[derive(Clone, Copy)]
struct RerankInput<'a> {
    reranker: &'a dyn Reranker,
    query: &'a str,
    embedding: Option<&'a [f32]>,
}

/// Steps shared by both pipelines: slice to `candidate_limit`, chunk each
/// candidate with its collection's geometry and pick the best chunk by
/// keyword overlap (intent terms at [`INTENT_WEIGHT_CHUNK`]), score — `1/rrf_rank`, or
/// with a reranker the [`blend`] of that rank and the best rerank score among
/// the [`RerankChunk`] chunks, reordered — then filter by `min_score` and
/// slice to `limit`. Fused results are already unique per file. `traces`,
/// when given, are completed and attached to the results they describe.
fn finalize_candidates(
    store: &Store,
    fused: Vec<SearchResult>,
    mut traces: Option<HashMap<String, Explain>>,
    primary_query: &str,
    rerank: Option<RerankInput<'_>>,
    options: &HybridOptions,
    report: &mut HybridReport,
) -> Result<Vec<HybridResult>> {
    let mut candidates = fused;
    candidates.truncate(options.candidate_limit);
    if rerank.is_none() {
        let ranks_kept = candidates
            .len()
            .min(options.limit)
            .min(rank_floor(options.min_score));
        candidates.truncate(ranks_kept);
    }

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

    let nearest = match (rerank.and_then(|r| r.embedding), options.rerank_chunk) {
        (Some(embedding), RerankChunk::Nearest) => Some((embedding, 1)),
        (Some(embedding), RerankChunk::BestOfNearest(n)) => Some((embedding, n.max(1))),
        _ => None,
    };

    let mut geometry = HashMap::new();
    let mut results = Vec::with_capacity(candidates.len());
    let mut rerank_chunks: Vec<Vec<Chunk>> = Vec::new();
    for (i, cand) in candidates.into_iter().enumerate() {
        if !geometry.contains_key(&cand.collection_name) {
            let params = store.collection_chunk_params(&cand.collection_name)?;
            geometry.insert(cand.collection_name.clone(), params);
        }
        let chunks = chunk_document_for(&cand.body, &geometry[&cand.collection_name]);
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

        if rerank.is_some() {
            let mut offered = Vec::new();
            if let Some((embedding, n)) = nearest {
                let positions: Vec<usize> = store
                    .chunk_distances(&cand.hash, embedding)?
                    .into_iter()
                    .take(n)
                    .map(|(pos, _)| pos)
                    .collect();
                if !positions.is_empty() {
                    let params = store.vector_chunk_params(&cand.hash)?;
                    offered = token_chunks_at(&cand.body, &params, &positions);
                }
            }
            if offered.is_empty() {
                offered.push(Chunk {
                    text: best_chunk.clone(),
                    pos: best_chunk_pos,
                });
            }
            rerank_chunks.push(offered);
        }

        let explain = traces
            .as_mut()
            .and_then(|t| t.remove(&cand.filepath))
            .map(|mut trace| {
                trace.rrf_rank = i + 1;
                trace.position_score = 1.0 / (i as f64 + 1.0);
                trace.final_score = trace.position_score;
                trace
            });
        results.push(HybridResult {
            explain,
            context: None,
            filepath: cand.filepath,
            display_path: cand.display_path,
            title: cand.title,
            body: cand.body,
            best_chunk,
            best_chunk_pos,
            score: 1.0 / (i as f64 + 1.0),
            rrf_rank: i + 1,
            rerank_score: None,
            docid: cand.docid,
        });
    }

    if let Some(rerank) = rerank
        && !results.is_empty()
    {
        let texts: Vec<String> = rerank_chunks
            .iter()
            .flatten()
            .map(|c| c.text.clone())
            .collect();
        let (scores, fresh) = rerank_cached(store, rerank.reranker, rerank.query, &texts)?;
        report.reranked = results.len();
        report.rerank_fresh = fresh;
        let mut scores = scores.into_iter();
        for (i, (result, offered)) in results.iter_mut().zip(rerank_chunks).enumerate() {
            let mut best: Option<(f64, Chunk)> = None;
            for (chunk, score) in offered.into_iter().zip(scores.by_ref()) {
                if best.as_ref().is_none_or(|(s, _)| score > *s) {
                    best = Some((score, chunk));
                }
            }
            let (rerank_score, chunk) = best.expect("every result offers a chunk");
            result.best_chunk = chunk.text;
            result.best_chunk_pos = chunk.pos;
            result.rerank_score = Some(rerank_score);
            result.score = blend(i + 1, rerank_score);
            if let Some(trace) = &mut result.explain {
                trace.rerank = Some(RerankTrace {
                    rerank_score,
                    fused_weight: blend_fused_weight(i + 1),
                });
                trace.final_score = result.score;
            }
        }
        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        results.retain(|r| r.score >= options.min_score);
        results.truncate(options.limit);
    }

    for result in &mut results {
        result.context = store.context_for_file(&result.filepath)?;
    }
    Ok(results)
}

/// How many fused ranks can clear `min_score` when the score is `1/rank`.
fn rank_floor(min_score: f64) -> usize {
    if min_score <= 0.0 {
        usize::MAX
    } else {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let ranks = (1.0 / min_score).floor() as usize;
        ranks
    }
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
    fn rrf_traces_reproduce_the_fused_scores_exactly() {
        let lists = vec![
            vec![ranked("a", 0.9), ranked("b", 0.5), ranked("c", 0.4)],
            vec![ranked("c", 0.8), ranked("d", 0.7), ranked("e", 0.6)],
            vec![
                ranked("e", 0.3),
                ranked("f", 0.2),
                ranked("g", 0.1),
                ranked("a", 0.05),
                ranked("h", 0.01),
            ],
        ];
        let weights = [2.0, 2.0, 1.0];
        let sources: Vec<ListSource> = [Backend::Fts, Backend::Vec, Backend::Vec]
            .into_iter()
            .map(|backend| ListSource {
                backend,
                kind: "original",
                query: "q".to_owned(),
            })
            .collect();
        let fused = reciprocal_rank_fusion(&lists, &weights, RRF_K);
        let traces = rrf_traces(&lists, &weights, &sources, RRF_K);
        assert_eq!(traces.len(), fused.len());
        for r in &fused {
            let trace = &traces[&r.filepath];
            assert_eq!(
                trace.rrf_score.to_bits(),
                r.score.to_bits(),
                "{}",
                r.filepath
            );
        }
        let a = &traces["qmd://c/a"];
        assert_eq!(a.contributions.len(), 2);
        assert_eq!(a.contributions[1].rank, 4);
        assert_eq!(a.contributions[1].backend_score, 0.05);
        assert_eq!(a.rrf_bonus, 0.05);
        assert_eq!(traces["qmd://c/d"].rrf_bonus, 0.02);
        assert_eq!(traces["qmd://c/g"].rrf_bonus, 0.02);
        assert_eq!(traces["qmd://c/h"].rrf_bonus, 0.0);
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
