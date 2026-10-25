//! Cross-encoder reranking (TS `store.ts` `hybridQuery` steps 6–7): each
//! candidate's best chunk is scored against the query, the score is cached in
//! `llm_cache`, and blended with the fused rank by position.
//!
//! The [`Reranker`] trait keeps the pipeline free of llama.cpp, as
//! [`crate::embed::Embedder`] does; [`MockReranker`] serves tests.

#[cfg(feature = "llama")]
mod llama;

#[cfg(feature = "llama")]
pub use llama::LlamaReranker;

use std::collections::HashMap;

use rusqlite::{OptionalExtension, params};
use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::store::{Store, now_iso};

/// A cross-encoder scoring query–document relevance.
pub trait Reranker {
    /// Model URI this reranker serves (part of the cache key).
    fn model_uri(&self) -> &str;

    /// Relevance in `[0, 1]` of each document to `query`, in input order.
    ///
    /// # Errors
    /// Propagates backend failures.
    fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>>;
}

/// Position-aware blend of fused rank and rerank score. The fused order
/// carries more weight at the top, where retrieval is usually right, and the
/// reranker more further down:
///
/// | `rrf_rank` | fused weight | rerank weight |
/// |---|---|---|
/// | 1–3 | 0.75 | 0.25 |
/// | 4–10 | 0.60 | 0.40 |
/// | 11+ | 0.40 | 0.60 |
///
/// `rrf_rank` is 1-based; the fused term is `1 / rrf_rank`.
pub fn blend(rrf_rank: usize, rerank_score: f64) -> f64 {
    let fused_weight = match rrf_rank {
        0..=3 => 0.75,
        4..=10 => 0.60,
        _ => 0.40,
    };
    #[allow(clippy::cast_precision_loss)]
    let fused = 1.0 / rrf_rank.max(1) as f64;
    fused_weight * fused + (1.0 - fused_weight) * rerank_score
}

fn cache_key(model: &str, query: &str, document: &str) -> String {
    let mut hasher = Sha256::new();
    for part in ["rerank", model, query, document] {
        hasher.update(part.as_bytes());
        hasher.update([0u8]);
    }
    hex::encode(hasher.finalize())
}

/// Rerank `documents` against `query`, reading and filling the per-chunk
/// score cache in `llm_cache` (keyed by model, query and chunk text). Returns
/// the scores and how many came from the model rather than the cache.
///
/// # Errors
/// Propagates store and backend failures.
pub fn rerank_cached(
    store: &Store,
    reranker: &dyn Reranker,
    query: &str,
    documents: &[String],
) -> Result<(Vec<f64>, usize)> {
    let keys: Vec<String> = documents
        .iter()
        .map(|d| cache_key(reranker.model_uri(), query, d))
        .collect();
    let mut scores: Vec<Option<f64>> = Vec::with_capacity(documents.len());
    for key in &keys {
        let cached: Option<String> = store
            .conn()
            .query_row(
                "SELECT result FROM llm_cache WHERE hash = ?1",
                params![key],
                |r| r.get(0),
            )
            .optional()?;
        scores.push(cached.and_then(|v| v.parse().ok()));
    }

    let mut missing: Vec<String> = Vec::new();
    let mut slot: HashMap<&str, usize> = HashMap::new();
    for (doc, score) in documents.iter().zip(&scores) {
        if score.is_none() && !slot.contains_key(doc.as_str()) {
            slot.insert(doc, missing.len());
            missing.push(doc.clone());
        }
    }
    if !missing.is_empty() {
        let fresh = reranker.rerank(query, &missing)?;
        let now = now_iso();
        for (i, doc) in documents.iter().enumerate() {
            if scores[i].is_none() {
                let value = fresh.get(slot[doc.as_str()]).copied().unwrap_or(0.0);
                scores[i] = Some(value);
                store.conn().execute(
                    "INSERT OR REPLACE INTO llm_cache (hash, result, created_at) \
                     VALUES (?1, ?2, ?3)",
                    params![keys[i], value.to_string(), now],
                )?;
            }
        }
    }
    Ok((
        scores.into_iter().map(Option::unwrap_or_default).collect(),
        missing.len(),
    ))
}

/// Deterministic test reranker: the share of the query's words (over two
/// characters) that occur in the document.
#[derive(Debug, Clone, Default)]
pub struct MockReranker;

impl Reranker for MockReranker {
    fn model_uri(&self) -> &str {
        "mock:reranker"
    }

    fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>> {
        let terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .filter(|t| t.chars().count() > 2)
            .map(str::to_owned)
            .collect();
        Ok(documents
            .iter()
            .map(|d| {
                if terms.is_empty() {
                    return 0.0;
                }
                let lower = d.to_lowercase();
                #[allow(clippy::cast_precision_loss)]
                let share = terms.iter().filter(|t| lower.contains(t.as_str())).count() as f64
                    / terms.len() as f64;
                share
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn blend_weights_follow_position() {
        assert!((blend(1, 0.0) - 0.75).abs() < 1e-12);
        assert!((blend(2, 1.0) - (0.75 * 0.5 + 0.25)).abs() < 1e-12);
        assert!((blend(4, 1.0) - (0.60 * 0.25 + 0.40)).abs() < 1e-12);
        assert!((blend(11, 1.0) - (0.40 / 11.0 + 0.60)).abs() < 1e-12);
        assert!(blend(11, 0.95) > blend(4, 0.05));
    }

    struct Counting {
        calls: Cell<usize>,
    }

    impl Reranker for Counting {
        fn model_uri(&self) -> &str {
            "count"
        }

        fn rerank(&self, _query: &str, documents: &[String]) -> Result<Vec<f64>> {
            self.calls.set(self.calls.get() + documents.len());
            #[allow(clippy::cast_precision_loss)]
            Ok(documents.iter().map(|d| d.len() as f64 / 10.0).collect())
        }
    }

    #[test]
    fn scores_are_cached_per_model_query_and_chunk() {
        let store = Store::open_in_memory().unwrap();
        let reranker = Counting {
            calls: Cell::new(0),
        };
        let docs = vec!["abc".to_owned(), "abcde".to_owned(), "abc".to_owned()];
        let (scores, fresh) = rerank_cached(&store, &reranker, "q", &docs).unwrap();
        assert_eq!(scores, vec![0.3, 0.5, 0.3]);
        assert_eq!((fresh, reranker.calls.get()), (2, 2));

        let (again, fresh) = rerank_cached(&store, &reranker, "q", &docs).unwrap();
        assert_eq!(again, scores);
        assert_eq!((fresh, reranker.calls.get()), (0, 2));

        let (_, fresh) = rerank_cached(&store, &reranker, "other", &docs).unwrap();
        assert_eq!(fresh, 2);
    }

    #[test]
    fn mock_scores_term_share() {
        let scores = MockReranker
            .rerank("cache eviction policy", &["The cache policy".to_owned()])
            .unwrap();
        assert!((scores[0] - 2.0 / 3.0).abs() < 1e-12);
    }
}
