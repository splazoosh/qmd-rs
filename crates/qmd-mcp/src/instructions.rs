//! Server instructions, built from the live index at startup.
//!
//! These are injected into the connecting agent's system prompt, so every byte
//! is paid on every session of every client that connects. That is the
//! argument both for writing them carefully and for keeping them to collection
//! *names*: the per-collection paths and counts are a `status` call away, and
//! the paths would spell out the machine's directory layout into text a model
//! may quote.
//!
//! They describe *this* pipeline, not the TypeScript one: there is no query
//! expansion and no rerank here, so nothing claims them.

use anyhow::Result;

use crate::worker::Index;

/// The `instructions` string returned from `initialize`.
///
/// # Errors
/// Propagates store failures.
pub fn build(index: &Index) -> Result<String> {
    let store = &index.store;
    let total: i64 =
        store
            .conn()
            .query_row("SELECT COUNT(*) FROM documents WHERE active = 1", [], |r| {
                r.get(0)
            })?;
    let vectors: i64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))?;
    let pending = store.pending_embedding_docs(None, &index.model)?.len();
    let collections = store.list_collections_with_stats()?;
    let global_context = store.store_global_context()?;

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "QMD is a local search engine over {total} markdown documents, indexed on this machine."
    ));
    if let Some(context) = &global_context {
        lines.push(format!("Context: {context}"));
    }

    if !collections.is_empty() {
        let names: Vec<&str> = collections.iter().map(|c| c.name.as_str()).collect();
        lines.push(String::new());
        lines.push(format!(
            "Collections (scope with the `collections` parameter): {}",
            names.join(", ")
        ));
        lines.push("Call `status` for per-collection document counts.".to_owned());
    }

    if vectors == 0 {
        lines.push(String::new());
        lines.push(
            "Note: nothing is embedded yet, so `vec` and `hyde` sub-queries return \
             nothing and a plain query is BM25 only. Run `qmd embed`."
                .to_owned(),
        );
    } else if pending > 0 {
        lines.push(String::new());
        lines.push(format!(
            "Note: {pending} documents are indexed but not embedded, so semantic \
             search does not see them. Run `qmd embed`."
        ));
    }

    lines.push(String::new());
    lines.push("Search with `query`, in one of two ways:".to_owned());
    lines.push(
        "  - `query`: plain text. Runs BM25 and vector search on it and fuses the two \
         rankings (RRF). The right default."
            .to_owned(),
    );
    lines.push(
        "  - `searches`: typed sub-queries you compose yourself. The first gets double \
         weight, so lead with your strongest signal."
            .to_owned(),
    );
    lines.push(
        "      lex  — BM25 keywords: prefix match, \"quoted phrases\", -negation.".to_owned(),
    );
    lines.push("      vec  — a natural-language question, matched by meaning.".to_owned());
    lines.push(
        "      hyde — 50-100 words of the answer you expect, matched by meaning against \
         the corpus."
            .to_owned(),
    );
    lines.push(
        "  Pass `intent` alongside either one to disambiguate a query whose words are \
         used in several domains here. It weights chunk and snippet selection and does \
         not search on its own."
            .to_owned(),
    );

    lines.push(String::new());
    lines.push("Read what you find:".to_owned());
    lines.push(
        "  - `get` — one document, by the `file` path or the `#docid` from a result. \
         Accepts a line-range suffix: `repo/doc.md:100` from line 100, `repo/doc.md:100:40` \
         for 40 lines from there."
            .to_owned(),
    );
    lines.push(
        "  - `multi_get` — several, by glob (`docs/2026-08*.md`), comma-separated list, \
         or docids."
            .to_owned(),
    );
    lines.push(
        "  - The `qmd://{path}` resource returns the same document as `get` with no \
         range arguments."
            .to_owned(),
    );

    lines.push(String::new());
    lines.push("Worth knowing:".to_owned());
    lines.push(
        "  - Result paths are `collection/path`; the collection is the first segment.".to_owned(),
    );
    lines.push(
        "  - Every result carries `line`, the 1-indexed line of its best match. To read \
         around it: `get(file, fromLine = max(1, line - 20), maxLines = 80)`."
            .to_owned(),
    );
    lines.push(
        "  - The index is a snapshot. It is refreshed by `qmd update && qmd embed` and \
         will not include edits made since."
            .to_owned(),
    );

    Ok(lines.join("\n"))
}
