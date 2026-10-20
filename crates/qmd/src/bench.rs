//! Benchmark fixture format and scoring, ported from TS `src/bench`
//! (`types.ts`, `score.ts`): precision@k, recall@1/3/5, MRR, and F1 for
//! search results against ground-truth expected files.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// One benchmark query with its expected results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkQuery {
    pub id: String,
    pub query: String,
    /// Difficulty/type grouping (`exact`, `semantic`, `topical`,
    /// `cross-domain`, `alias`).
    #[serde(rename = "type")]
    pub query_type: String,
    pub description: String,
    /// Paths (relative to the collection) that should appear in results.
    pub expected_files: Vec<String>,
    /// How many of `expected_files` should appear in the top-k results.
    pub expected_in_top_k: usize,
}

/// A benchmark fixture: queries with expected results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkFixture {
    pub description: String,
    pub version: u32,
    /// Optional collection to search within.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    pub queries: Vec<BenchmarkQuery>,
}

impl BenchmarkFixture {
    /// Load a fixture from a JSON file.
    ///
    /// # Errors
    /// Fails on I/O or JSON parse errors.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        serde_json::from_str(&raw)
            .map_err(|e| Error::Other(format!("invalid fixture {}: {e}", path.display())))
    }
}

/// Scores for one query's results against its expected files.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreMetrics {
    pub precision_at_k: f64,
    pub recall: f64,
    pub recall_at_1: f64,
    pub recall_at_3: f64,
    pub recall_at_5: f64,
    pub mrr: f64,
    pub f1: f64,
    pub hits_at_k: usize,
    pub matched_files: Vec<String>,
    pub unmatched_expected_files: Vec<String>,
}

/// Normalize a file path for comparison: strip a `qmd://collection` prefix,
/// lowercase, and trim slashes.
pub fn normalize_path(path: &str) -> String {
    let path = match path.strip_prefix("qmd://") {
        Some(rest) => rest.split_once('/').map_or(rest, |(_, p)| p),
        None => path,
    };
    path.to_lowercase().trim_matches('/').to_owned()
}

/// Whether two paths refer to the same file (normalized suffix comparison on
/// a `/` boundary, so `a.md` does not match `data.md`).
pub fn paths_match(result: &str, expected: &str) -> bool {
    let nr = normalize_path(result);
    let ne = normalize_path(expected);
    if nr.is_empty() || ne.is_empty() {
        return false;
    }
    nr == ne || nr.ends_with(&format!("/{ne}")) || ne.ends_with(&format!("/{nr}"))
}

fn hits_within(result_files: &[String], expected_files: &[String], k: usize) -> usize {
    let top_k = &result_files[..k.min(result_files.len())];
    expected_files
        .iter()
        .filter(|e| top_k.iter().any(|r| paths_match(r, e)))
        .count()
}

#[allow(clippy::cast_precision_loss)]
fn ratio(hits: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

/// Score a ranked result list against the expected files.
#[allow(clippy::cast_precision_loss)]
pub fn score_results(
    result_files: &[String],
    expected_files: &[String],
    top_k: usize,
) -> ScoreMetrics {
    let hits_at_k = hits_within(result_files, expected_files, top_k);

    let mut matched_files = Vec::new();
    let mut unmatched_expected_files = Vec::new();
    for expected in expected_files {
        if result_files.iter().any(|r| paths_match(r, expected)) {
            matched_files.push(expected.clone());
        } else {
            unmatched_expected_files.push(expected.clone());
        }
    }

    let mrr = result_files
        .iter()
        .position(|r| expected_files.iter().any(|e| paths_match(r, e)))
        .map_or(0.0, |i| 1.0 / (i + 1) as f64);

    let denominator = top_k.min(expected_files.len());
    let precision_at_k = ratio(hits_at_k, denominator);
    let recall = ratio(matched_files.len(), expected_files.len());
    let f1 = if precision_at_k + recall > 0.0 {
        2.0 * (precision_at_k * recall) / (precision_at_k + recall)
    } else {
        0.0
    };

    ScoreMetrics {
        precision_at_k,
        recall,
        recall_at_1: ratio(
            hits_within(result_files, expected_files, 1),
            expected_files.len(),
        ),
        recall_at_3: ratio(
            hits_within(result_files, expected_files, 3),
            expected_files.len(),
        ),
        recall_at_5: ratio(
            hits_within(result_files, expected_files, 5),
            expected_files.len(),
        ),
        mrr,
        f1,
        hits_at_k,
        matched_files,
        unmatched_expected_files,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn normalize_strips_scheme_case_and_slashes() {
        assert_eq!(normalize_path("Docs/README.md"), "docs/readme.md");
        assert_eq!(
            normalize_path("qmd://notes/docs/readme.md"),
            "docs/readme.md"
        );
        assert_eq!(normalize_path("/docs/readme.md/"), "docs/readme.md");
        assert_eq!(normalize_path("readme.md"), "readme.md");
    }

    #[test]
    fn paths_match_by_normalized_suffix() {
        assert!(paths_match("a.md", "a.md"));
        assert!(paths_match("A.md", "a.MD"));
        assert!(paths_match("deep/dir/a.md", "dir/a.md"));
        assert!(paths_match("dir/a.md", "deep/dir/a.md"));
        assert!(paths_match("qmd://notes/dir/a.md", "dir/a.md"));
        assert!(!paths_match("a.md", "b.md"));
        assert!(!paths_match("data.md", "a.md"));
        assert!(!paths_match("", "a.md"));
    }

    #[test]
    fn perfect_score_when_all_expected_in_top_k() {
        let s = score_results(&strings(&["a.md", "b.md"]), &strings(&["a.md", "b.md"]), 2);
        assert_eq!(s.precision_at_k, 1.0);
        assert_eq!(s.recall, 1.0);
        assert_eq!(s.mrr, 1.0);
        assert_eq!(s.f1, 1.0);
        assert_eq!(s.hits_at_k, 2);
        assert!(s.unmatched_expected_files.is_empty());
    }

    #[test]
    fn zero_score_when_none_found() {
        let s = score_results(&strings(&["x.md"]), &strings(&["a.md"]), 3);
        assert_eq!(s.precision_at_k, 0.0);
        assert_eq!(s.recall, 0.0);
        assert_eq!(s.mrr, 0.0);
        assert_eq!(s.f1, 0.0);
        assert_eq!(s.unmatched_expected_files, strings(&["a.md"]));
    }

    #[test]
    fn partial_credit_when_found_outside_top_k() {
        let s = score_results(&strings(&["x.md", "y.md", "a.md"]), &strings(&["a.md"]), 2);
        assert_eq!(s.hits_at_k, 0);
        assert_eq!(s.recall, 1.0);
        assert!((s.mrr - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn mrr_uses_first_relevant_rank() {
        let s = score_results(&strings(&["x.md", "a.md"]), &strings(&["a.md"]), 3);
        assert!((s.mrr - 0.5).abs() < 1e-9);
    }

    #[test]
    fn recall_at_cutoffs_and_empty_results() {
        let s = score_results(
            &strings(&["a.md", "x.md", "b.md", "y.md", "c.md"]),
            &strings(&["a.md", "b.md", "c.md"]),
            5,
        );
        assert!((s.recall_at_1 - 1.0 / 3.0).abs() < 1e-9);
        assert!((s.recall_at_3 - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(s.recall_at_5, 1.0);
        assert_eq!(s.matched_files, strings(&["a.md", "b.md", "c.md"]));

        let empty = score_results(&[], &strings(&["a.md"]), 3);
        assert_eq!(empty.recall, 0.0);
        assert_eq!(empty.mrr, 0.0);
    }

    #[test]
    fn fixture_parses_from_repo_json() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/eval-fixture.json");
        let fixture = BenchmarkFixture::load(&path).unwrap();
        assert_eq!(fixture.version, 1);
        assert_eq!(fixture.collection.as_deref(), Some("eval-docs"));
        assert!(!fixture.queries.is_empty());
        assert!(fixture.queries.iter().all(|q| !q.expected_files.is_empty()));
    }
}
