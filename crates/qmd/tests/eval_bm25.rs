//! BM25 evaluation gate, ported from TS `test/eval-bm25.test.ts`.
//!
//! Indexes the 6-document eval corpus in `fixtures/eval-docs` and checks
//! Hit@k rates against the TS baselines.

mod common;

use common::{EvalQuery, by_difficulty, indexed_store};
use qmd::Store;

fn hit_rate(store: &Store, queries: &[&EvalQuery], top_k: usize) -> f64 {
    let mut hits = 0usize;
    for eq in queries {
        let results = store.search_fts(eq.query, 5, None).unwrap();
        if results
            .iter()
            .take(top_k)
            .any(|r| r.filepath.to_lowercase().contains(eq.expected_doc))
        {
            hits += 1;
        }
    }
    hits as f64 / queries.len() as f64
}

#[test]
fn easy_queries_hit3_at_least_80_percent() {
    let store = indexed_store();
    assert!(hit_rate(&store, &by_difficulty("easy"), 3) >= 0.8);
}

#[test]
fn medium_queries_hit3_at_least_15_percent() {
    let store = indexed_store();
    assert!(hit_rate(&store, &by_difficulty("medium"), 3) >= 0.15);
}

#[test]
fn hard_queries_hit5_at_least_15_percent() {
    let store = indexed_store();
    assert!(hit_rate(&store, &by_difficulty("hard"), 5) >= 0.15);
}

#[test]
fn overall_hit3_at_least_40_percent() {
    let store = indexed_store();
    let all: Vec<&EvalQuery> = common::EVAL_QUERIES.iter().collect();
    let rate = hit_rate(&store, &all, 3);
    println!("BM25 overall Hit@3: {rate:.3} (TS baseline ~0.50)");
    assert!(rate >= 0.4);
}
