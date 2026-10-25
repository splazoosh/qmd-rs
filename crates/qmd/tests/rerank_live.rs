//! The real cross-encoder (Qwen3-Reranker-0.6B) on the eval corpus.
//!
//! Requires the `llama` feature and the rerank model (~640 MB, downloaded on
//! first use); `#[ignore]`d so plain `cargo test` stays model-free:
//!
//! ```text
//! cargo test --release --test rerank_live -- --ignored --nocapture
//! ```

#![cfg(feature = "llama")]

mod common;

use qmd::models::DEFAULT_RERANK_MODEL;
use qmd::rerank::{LlamaReranker, Reranker as _};

#[test]
#[ignore = "requires the rerank model"]
fn reranker_scores_the_answering_passage_highest() {
    let reranker = LlamaReranker::load(DEFAULT_RERANK_MODEL).expect("load rerank model");
    let read = |name: &str| std::fs::read_to_string(common::eval_docs_dir().join(name)).unwrap();
    let docs = vec![
        read("distributed-systems-overview.md"),
        read("startup-fundraising-memo.md"),
        read("machine-learning-primer.md"),
    ];
    let cases = [
        ("consistency vs availability tradeoffs", 0),
        ("raising money for startup", 1),
        ("how to prevent models from memorizing data", 2),
    ];
    for (query, expected) in cases {
        let started = std::time::Instant::now();
        let scores = reranker.rerank(query, &docs).expect("rerank");
        println!("{query}: {scores:.3?} in {:?}", started.elapsed());
        assert!(scores.iter().all(|s| (0.0..=1.0).contains(s)));
        let best = (0..docs.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap();
        assert_eq!(best, expected, "{query}");
        assert!(
            scores[expected] > 0.5,
            "{query}: answering passage is a yes"
        );
    }
}
