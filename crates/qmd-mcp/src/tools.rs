//! The four tools: `query`, `get`, `multi_get`, `status`.
//!
//! Each runs on the index thread (see [`crate::worker`]) and returns the rows
//! the CLI would: a text summary alongside structured content, as the
//! TypeScript server does.

use anyhow::{Result, anyhow, bail};
use qmd::query::{
    HybridOptions, HybridResult, QueryType, TypedQuery, hybrid_query, structured_search,
};
use qmd::querylog::{self, Arm, Via};
use qmd::snippet::{add_line_numbers, extract_snippet_at};
use qmd::store::{DocumentLookup, DocumentResult, TermMatch};
use serde::{Deserialize, Serialize};

use crate::worker::Index;

/// `multi_get` skips anything larger than this unless told otherwise.
pub const DEFAULT_MULTI_GET_MAX_BYTES: i64 = 65_536;

/// Snippet width, matching the CLI's JSON output.
const SNIPPET_LEN: usize = 300;

/// Which backend a sub-query routes to.
#[derive(Debug, Clone, Copy, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum SubSearchType {
    /// BM25 keywords: prefix match, "quoted phrases", -negation.
    Lex,
    /// A natural-language question, matched by meaning.
    Vec,
    /// A hypothetical answer passage, matched by meaning.
    Hyde,
}

impl From<SubSearchType> for QueryType {
    fn from(value: SubSearchType) -> Self {
        match value {
            SubSearchType::Lex => Self::Lex,
            SubSearchType::Vec => Self::Vec,
            SubSearchType::Hyde => Self::Hyde,
        }
    }
}

/// One typed sub-query of a `query` call.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct SubSearch {
    /// lex = BM25 keywords; vec = a natural-language question; hyde = a
    /// hypothetical answer passage.
    pub r#type: SubSearchType,
    /// The query text. For lex: keywords, "quoted phrases", -negation. For
    /// vec: a natural-language question. For hyde: 50-100 words of the answer
    /// you expect.
    pub query: String,
}

/// Arguments to `query`.
#[derive(Debug, Clone, Default, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct QueryParams {
    /// Plain-text query: BM25 and vector search both run on it and the two
    /// rankings are fused (RRF). The right default. Mutually exclusive with
    /// `searches`.
    #[serde(default)]
    pub query: Option<String>,
    /// Typed sub-queries run as given. The first gets 2x weight in the
    /// fusion, so lead with your strongest signal. Mutually exclusive with
    /// `query`; at most 10.
    #[serde(default)]
    pub searches: Option<Vec<SubSearch>>,
    /// Max results (default: 10).
    #[serde(default)]
    pub limit: Option<usize>,
    /// Minimum relevance score (default: 0).
    #[serde(default)]
    pub min_score: Option<f64>,
    /// Maximum candidates to chunk and rank (default: 40; lower is faster and
    /// may miss results).
    #[serde(default)]
    pub candidate_limit: Option<usize>,
    /// Restrict to these collections (OR match). Defaults to every collection
    /// the index includes by default.
    #[serde(default)]
    pub collections: Option<Vec<String>>,
    /// Background context that disambiguates the query — for example
    /// query="performance", intent="web page load times". Weights chunk and
    /// snippet selection; does not search on its own.
    #[serde(default)]
    pub intent: Option<String>,
    /// How BM25 terms combine, in a plain query and in every lex sub-query:
    /// "all" (every term must appear) or "any" (any may; documents holding
    /// more rank higher). Defaults to the index config, else "any".
    #[serde(default, rename = "match")]
    pub term_match: Option<TermMatchParam>,
    /// Rerank the fused candidates with the cross-encoder (slower). Defaults
    /// to the index config, else false.
    #[serde(default)]
    pub rerank: Option<bool>,
    /// Attach to each result how it got its rank: its rank and score in each
    /// BM25 and vector list, the fused RRF score and rank, and the rerank
    /// blend when reranking ran (default: false). Changes no ranking.
    #[serde(default)]
    pub explain: Option<bool>,
}

/// BM25 term combination for `query`.
#[derive(Debug, Clone, Copy, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum TermMatchParam {
    All,
    Any,
}

impl From<TermMatchParam> for TermMatch {
    fn from(value: TermMatchParam) -> Self {
        match value {
            TermMatchParam::All => Self::All,
            TermMatchParam::Any => Self::Any,
        }
    }
}

/// One row of `query` output. Field names match `qmd query --format json`.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct QueryHit {
    /// `#abc123` — pass it to `get` or `multi_get`.
    pub docid: String,
    /// `collection/path`.
    pub file: String,
    pub title: String,
    pub score: f64,
    /// Folder context, when the collection defines one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// 1-indexed line of the best match in the source document.
    pub line: usize,
    /// The matching lines, with line numbers.
    pub snippet: String,
    /// The ranking trace, when `explain` asked for it: `contributions` (one
    /// per ranked list: `backend`, `kind`, `query`, `rank`, `weight`,
    /// `backend_score`, `rrf`), `rrf_base`, `rrf_bonus`, `rrf_score`,
    /// `rrf_rank`, `position_score`, `rerank` (`rerank_score`,
    /// `fused_weight`) and `final_score`, all unrounded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explain: Option<serde_json::Value>,
}

/// The `query` tool's structured content.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct QueryOutput {
    pub results: Vec<QueryHit>,
}

/// Run a hybrid search. Mirrors `qmd query`, including its collection
/// defaults.
///
/// # Errors
/// Fails on an unknown collection and on `query`/`searches` misuse, and
/// propagates store and embedding failures.
pub fn query(index: &mut Index, params: &QueryParams) -> Result<(QueryOutput, String)> {
    let has_query = params.query.as_ref().is_some_and(|q| !q.trim().is_empty());
    let searches = params.searches.as_deref().unwrap_or(&[]);
    if !has_query && searches.is_empty() {
        bail!("provide either 'query' (plain text) or 'searches' (typed sub-queries)");
    }
    if has_query && !searches.is_empty() {
        bail!("'query' and 'searches' are mutually exclusive; provide only one");
    }
    if searches.len() > 10 {
        bail!("at most 10 sub-queries ({} given)", searches.len());
    }

    let names = resolve_collections(index, params.collections.as_deref())?;

    let typed: Vec<TypedQuery> = searches
        .iter()
        .enumerate()
        .map(|(i, s)| TypedQuery {
            query_type: s.r#type.into(),
            query: s.query.clone(),
            line: i + 1,
        })
        .collect();

    let settings = index.config.search_settings();
    let rerank = params.rerank.or(settings.rerank).unwrap_or(false);
    let options = HybridOptions {
        collections: names,
        limit: params.limit.unwrap_or(10),
        min_score: params.min_score.unwrap_or(0.0),
        candidate_limit: params
            .candidate_limit
            .unwrap_or(qmd::query::CANDIDATE_LIMIT),
        intent: params.intent.clone(),
        term_match: params
            .term_match
            .map(TermMatch::from)
            .or(settings.term_match)
            .unwrap_or_default(),
        explain: params.explain.unwrap_or(false),
        ..HybridOptions::default()
    };

    let display_query = if typed.is_empty() {
        params.query.clone().unwrap_or_default()
    } else {
        typed
            .iter()
            .find(|s| s.query_type == QueryType::Lex)
            .or_else(|| typed.iter().find(|s| s.query_type == QueryType::Vec))
            .map_or_else(String::new, |s| s.query.clone())
    };

    let needs_embedder = index.store.vec_table_exists()?
        && (typed.is_empty()
            || typed
                .iter()
                .any(|s| matches!(s.query_type, QueryType::Vec | QueryType::Hyde)));
    if needs_embedder {
        index.ensure_embedder()?;
    }
    if rerank {
        index.ensure_reranker()?;
    }
    let reranker = if rerank { index.reranker() } else { None };
    let embedder = index.embedder();
    let store = &index.store;
    let response = if typed.is_empty() {
        let text = params.query.as_deref().unwrap_or_default();
        hybrid_query(store, embedder, reranker, text, &options).map_err(|e| anyhow!("{e}"))?
    } else {
        structured_search(store, embedder, reranker, &typed, &options)
            .map_err(|e| anyhow!("{e}"))?
    };

    let mut results = response.results;
    results.truncate(options.limit);

    let hits: Vec<QueryHit> = results
        .iter()
        .map(|r| hit(r, &display_query, params.intent.as_deref()))
        .collect();
    let traces: Vec<Option<String>> = results
        .iter()
        .map(|r| r.explain.as_ref().map(ToString::to_string))
        .collect();
    let summary = summarize(&hits, &traces, &display_query);

    let (arm, logged) = if typed.is_empty() {
        (Arm::Hybrid, params.query.clone().unwrap_or_default())
    } else {
        let lines: Vec<String> = typed
            .iter()
            .map(|s| format!("{}: {}", s.query_type.as_str(), s.query))
            .collect();
        (Arm::Structured, lines.join("\n"))
    };
    let paths: Vec<String> = results.iter().map(|r| r.filepath.clone()).collect();
    querylog::append(&index.db_path, arm, Via::Mcp, &logged, &paths);

    Ok((QueryOutput { results: hits }, summary))
}

fn hit(r: &HybridResult, display_query: &str, intent: Option<&str>) -> QueryHit {
    let snippet = extract_snippet_at(
        &r.body,
        display_query,
        SNIPPET_LEN,
        Some(r.best_chunk_pos),
        Some(r.best_chunk.len()),
        intent,
    );
    QueryHit {
        docid: format!("#{}", r.docid),
        file: r.display_path.clone(),
        title: r.title.clone(),
        score: (r.score * 100.0).round() / 100.0,
        context: r.context.clone(),
        line: snippet.line,
        snippet: snippet.numbered(),
        explain: r
            .explain
            .as_ref()
            .map(|e| serde_json::to_value(e).expect("explain serializes")),
    }
}

/// The text block: each hit, with its trace (as `qmd query --explain` prints
/// it) between the header and the snippet when there is one.
fn summarize(hits: &[QueryHit], traces: &[Option<String>], display_query: &str) -> String {
    if hits.is_empty() {
        return format!("No results for: {display_query}");
    }
    let mut out = format!("{} results for: {display_query}\n", hits.len());
    for (h, trace) in hits.iter().zip(traces) {
        out.push_str(&format!("\n{} {} ({:.2})\n", h.docid, h.file, h.score));
        if let Some(context) = &h.context {
            out.push_str(&format!("Context: {context}\n"));
        }
        if let Some(trace) = trace {
            out.push_str(&format!("Explain:\n{trace}\n"));
        }
        out.push_str(&format!("{}\n", h.snippet));
    }
    out
}

/// Validate a requested collection filter, defaulting to the include-by-default set.
fn resolve_collections(index: &Index, requested: Option<&[String]>) -> Result<Vec<String>> {
    Ok(index
        .config
        .query_collections(requested.unwrap_or_default())?)
}

/// Arguments to `get`.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GetParams {
    /// File path or `#docid` from a search result. Accepts a line-range
    /// suffix: `repo/doc.md:100` starts at line 100, `repo/doc.md:100:40`
    /// (or `#abc123:100:40`) reads 40 lines from line 100.
    pub file: String,
    /// Start at this line (1-indexed). Wins over a `:line` suffix.
    #[serde(default)]
    pub from_line: Option<usize>,
    /// Maximum lines to return. Wins over a `:from:count` suffix.
    #[serde(default)]
    pub max_lines: Option<usize>,
    /// Prefix each line with its number (default: true).
    #[serde(default)]
    pub line_numbers: Option<bool>,
}

/// A document ready to be returned as an embedded resource.
pub struct Document {
    pub uri: String,
    pub display_path: String,
    pub title: String,
    pub text: String,
}

/// Fetch one document, honouring `:line` and `:from:count` suffixes.
///
/// # Errors
/// Fails when no document matches, listing similar paths when there are any.
pub fn get(index: &Index, params: &GetParams) -> Result<Document> {
    let (lookup, from, lines) = parse_range(&params.file, params.from_line, params.max_lines);
    let doc = find(index, &lookup, &params.file)?;

    let start_line = from.unwrap_or(1);
    let mut text = doc.body.clone().unwrap_or_default();
    if from.is_some() || lines.is_some() {
        let all: Vec<&str> = text.split('\n').collect();
        let start = start_line - 1;
        let end = lines.map_or(all.len(), |l| start.saturating_add(l).min(all.len()));
        text = all.get(start..end).unwrap_or_default().join("\n");
    }
    if params.line_numbers.unwrap_or(true) && !text.is_empty() {
        text = add_line_numbers(&text, start_line);
    }
    Ok(document(&doc, text))
}

/// Look one document up, turning a miss into the CLI's "did you mean" error.
fn find(index: &Index, lookup: &str, requested: &str) -> Result<DocumentResult> {
    match index.store.find_document(lookup, true)? {
        DocumentLookup::Found(doc) => Ok(*doc),
        DocumentLookup::NotFound { similar } => {
            let mut msg = format!("document not found: {requested}");
            if !similar.is_empty() {
                msg.push_str("\n\nDid you mean one of these?");
                for s in &similar {
                    msg.push_str(&format!("\n  - {s}"));
                }
            }
            bail!(msg)
        }
    }
}

/// Build the resource form of a document, prepending its folder context.
fn document(doc: &DocumentResult, text: String) -> Document {
    let text = match &doc.context {
        Some(context) => format!("<!-- Context: {context} -->\n\n{text}"),
        None => text,
    };
    Document {
        uri: format!("qmd://{}", doc.display_path),
        display_path: doc.display_path.clone(),
        title: doc.title.clone(),
        text,
    }
}

/// Split a `:line` / `:from:count` suffix off a lookup string. Explicit
/// arguments win over the suffix.
fn parse_range(
    file: &str,
    from: Option<usize>,
    lines: Option<usize>,
) -> (String, Option<usize>, Option<usize>) {
    let mut lookup = file.to_owned();
    let mut from = from;
    let mut lines = lines;
    let parts: Vec<&str> = file.rsplitn(3, ':').collect();
    if parts.len() == 3
        && let (Ok(count), Ok(line)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>())
        && !parts[2].is_empty()
    {
        lookup = parts[2].to_owned();
        from = from.or(Some(line));
        lines = lines.or(Some(count));
    } else if let Some((rest, last)) = file.rsplit_once(':')
        && let Ok(line) = last.parse::<usize>()
        && !rest.is_empty()
    {
        lookup = rest.to_owned();
        from = from.or(Some(line));
    }
    (lookup, from.map(|f| f.max(1)), lines)
}

/// Arguments to `multi_get`.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MultiGetParams {
    /// A glob (`docs/2026-08*.md`), a `#docid`, or a comma-separated list of
    /// paths and docids.
    pub pattern: String,
    /// Maximum lines per document.
    #[serde(default)]
    pub max_lines: Option<usize>,
    /// Skip documents larger than this many bytes (default: 65536).
    #[serde(default)]
    pub max_bytes: Option<i64>,
    /// Prefix each line with its number (default: true).
    #[serde(default)]
    pub line_numbers: Option<bool>,
}

/// One entry of a `multi_get` response: a document, or a note about one.
pub enum MultiGetEntry {
    Doc(Box<Document>),
    /// A document too large to inline, or lookups that failed.
    Note(String),
}

/// Fetch several documents by glob, comma-separated list, or docids.
///
/// # Errors
/// Fails only when nothing matched at all; per-document failures come back as
/// a leading [`MultiGetEntry::Note`], as the TypeScript server's `errors`
/// block does.
pub fn multi_get(index: &Index, params: &MultiGetParams) -> Result<Vec<MultiGetEntry>> {
    let max_bytes = params.max_bytes.unwrap_or(DEFAULT_MULTI_GET_MAX_BYTES);
    let mut errors = Vec::new();
    let docs = resolve_pattern(index, &params.pattern, &mut errors)?;
    if docs.is_empty() && errors.is_empty() {
        bail!("no documents matched: {}", params.pattern);
    }

    let mut entries = Vec::new();
    if !errors.is_empty() {
        entries.push(MultiGetEntry::Note(format!(
            "Errors:\n{}",
            errors.join("\n")
        )));
    }
    for doc in docs {
        if doc.body_length > max_bytes {
            entries.push(MultiGetEntry::Note(format!(
                "[SKIPPED: {} — {} bytes exceeds maxBytes {max_bytes}. Use `get` with \
                 file=\"{}\" to retrieve it.]",
                doc.display_path, doc.body_length, doc.display_path
            )));
            continue;
        }
        let mut text = doc.body.clone().unwrap_or_default();
        if let Some(max) = params.max_lines {
            let all: Vec<&str> = text.split('\n').collect();
            if all.len() > max {
                text = format!(
                    "{}\n\n[... truncated {} more lines]",
                    all[..max].join("\n"),
                    all.len() - max
                );
            }
        }
        if params.line_numbers.unwrap_or(true) {
            text = add_line_numbers(&text, 1);
        }
        entries.push(MultiGetEntry::Doc(Box::new(document(&doc, text))));
    }
    Ok(entries)
}

/// Expand a `multi_get` pattern into documents.
///
/// A comma splits the pattern into independent lookups; a term containing `*`
/// or `?` is globbed against every active document; anything else is an
/// ordinary `get`-style lookup. Terms that match nothing are collected into
/// `errors` rather than failing the call.
fn resolve_pattern(
    index: &Index,
    pattern: &str,
    errors: &mut Vec<String>,
) -> Result<Vec<DocumentResult>> {
    let mut docs: Vec<DocumentResult> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for term in pattern.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let found = if term.contains('*') || term.contains('?') {
            let matched = index.store.glob_documents(term)?;
            if matched.is_empty() {
                errors.push(format!("no files matched pattern: {term}"));
            }
            matched
        } else {
            match index.store.find_document(term, true)? {
                DocumentLookup::Found(doc) => vec![*doc],
                DocumentLookup::NotFound { .. } => {
                    errors.push(format!("file not found: {term}"));
                    continue;
                }
            }
        };
        for doc in found {
            if !seen.contains(&doc.filepath) {
                seen.push(doc.filepath.clone());
                docs.push(doc);
            }
        }
    }
    Ok(docs)
}

/// One collection in `status` output.
///
/// Deliberately carries no filesystem path: TS prints `col.path` per
/// collection, which across many collections would spell a machine's
/// directory layout into text a model reads and may quote.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct CollectionStatus {
    pub name: String,
    /// The glob the collection indexes, e.g. `**/*.md`.
    pub pattern: String,
    pub documents: i64,
}

/// The `status` tool's structured content.
#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StatusOutput {
    pub total_documents: i64,
    pub embedded_chunks: i64,
    pub needs_embedding: usize,
    pub has_vector_index: bool,
    pub collections: Vec<CollectionStatus>,
}

/// Index counts and the live collection list.
///
/// # Errors
/// Propagates store failures.
pub fn status(index: &Index) -> Result<(StatusOutput, String)> {
    let store = &index.store;
    let total_documents: i64 =
        store
            .conn()
            .query_row("SELECT COUNT(*) FROM documents WHERE active = 1", [], |r| {
                r.get(0)
            })?;
    let embedded_chunks: i64 =
        store
            .conn()
            .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))?;
    let needs_embedding = store.pending_embedding_docs(None, &index.model)?.len();
    let has_vector_index = store.vec_table_exists()? && embedded_chunks > 0;
    let collections: Vec<CollectionStatus> = store
        .list_collections_with_stats()?
        .into_iter()
        .map(|c| CollectionStatus {
            name: c.name,
            pattern: c.pattern,
            documents: c.active_count,
        })
        .collect();

    let mut summary = format!(
        "QMD index status:\n  Documents: {total_documents}\n  Embedded chunks: \
         {embedded_chunks}\n  Needs embedding: {needs_embedding}\n  Vector search: \
         {}\n  Collections: {}",
        if has_vector_index {
            "available"
        } else {
            "unavailable"
        },
        collections.len()
    );
    for c in &collections {
        summary.push_str(&format!(
            "\n    - {}: {} documents ({})",
            c.name, c.documents, c.pattern
        ));
    }
    Ok((
        StatusOutput {
            total_documents,
            embedded_chunks,
            needs_embedding,
            has_vector_index,
            collections,
        },
        summary,
    ))
}
