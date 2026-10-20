//! Hybrid search evaluation gates with the real EmbeddingGemma model,
//! ported from TS `test/eval.test.ts` ("Hybrid Search (RRF)" describe block).
//! Hybrid must score at or above the TS `--no-rerank` baseline gates.
//!
//! Requires the `llama` feature and a local model (downloaded on first use);
//! `#[ignore]`d so plain `cargo test` stays model-free. Run with:
//!
//! ```text
//! cargo test --release --test eval_hybrid -- --ignored --nocapture
//! ```

#![cfg(feature = "llama")]

mod common;

use common::{EvalQuery, by_difficulty, indexed_store};
use qmd::Store;
use qmd::embed::{
    EmbedOptions, Embedder as _, LlamaEmbedder, format_query_for_embedding, generate_embeddings,
};
use qmd::models::DEFAULT_EMBED_MODEL;
use qmd::query::{HybridOptions, hybrid_query};

/// Serializes model tests within this binary: two concurrent llama contexts
/// in one process produce corrupted embeddings (observed as wrong eval hits).
static MODEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn embedded_store() -> (Store, LlamaEmbedder) {
    let store = indexed_store();
    let embedder = LlamaEmbedder::load(DEFAULT_EMBED_MODEL).expect("load embed model");
    let result = generate_embeddings(&store, &embedder, &EmbedOptions::default()).unwrap();
    assert_eq!(result.docs_processed, 6);
    assert_eq!(result.errors, 0, "failures: {:?}", result.failures);
    (store, embedder)
}

fn hybrid_hit(store: &Store, embedder: &LlamaEmbedder, eq: &EvalQuery, top_k: usize) -> bool {
    let response = hybrid_query(store, Some(embedder), eq.query, &HybridOptions::default())
        .expect("hybrid query");
    response
        .results
        .iter()
        .take(top_k)
        .any(|r| r.filepath.to_lowercase().contains(eq.expected_doc))
}

fn hit_rate(store: &Store, embedder: &LlamaEmbedder, queries: &[&EvalQuery], top_k: usize) -> f64 {
    let hits = queries
        .iter()
        .filter(|eq| hybrid_hit(store, embedder, eq, top_k))
        .count();
    hits as f64 / queries.len() as f64
}

#[test]
#[ignore = "requires the real embedding model (run locally)"]
fn hybrid_eval_gates() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let (store, embedder) = embedded_store();

    let easy = hit_rate(&store, &embedder, &by_difficulty("easy"), 3);
    let medium = hit_rate(&store, &embedder, &by_difficulty("medium"), 3);
    let hard = hit_rate(&store, &embedder, &by_difficulty("hard"), 5);
    let standard: Vec<&EvalQuery> = common::EVAL_QUERIES
        .iter()
        .filter(|q| q.difficulty != "fusion")
        .collect();
    let overall = hit_rate(&store, &embedder, &standard, 3);

    println!("Hybrid eval — easy Hit@3: {easy:.3} (gate 0.80)");
    println!("Hybrid eval — medium Hit@3: {medium:.3} (gate 0.50)");
    println!("Hybrid eval — hard Hit@5: {hard:.3} (gate 0.35)");
    println!("Hybrid eval — overall Hit@3: {overall:.3} (gate 0.60)");

    assert!(easy >= 0.8, "easy Hit@3 {easy:.3} below gate 0.80");
    assert!(medium >= 0.5, "medium Hit@3 {medium:.3} below gate 0.50");
    assert!(hard >= 0.35, "hard Hit@5 {hard:.3} below gate 0.35");
    assert!(overall >= 0.6, "overall Hit@3 {overall:.3} below gate 0.60");
}

/// Fusion queries carry both lexical and semantic signal: RRF must combine
/// the weak individual signals, beating (or matching) both single backends.
#[test]
#[ignore = "requires the real embedding model (run locally)"]
fn hybrid_eval_fusion_gates() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let (store, embedder) = embedded_store();
    let fusion = by_difficulty("fusion");

    let mut hybrid_hits = 0usize;
    let mut bm25_hits = 0usize;
    let mut vec_hits = 0usize;
    for eq in &fusion {
        if hybrid_hit(&store, &embedder, eq, 3) {
            hybrid_hits += 1;
        }
        let bm25 = store.search_fts(eq.query, 5, None).unwrap();
        if bm25
            .iter()
            .take(3)
            .any(|r| r.filepath.to_lowercase().contains(eq.expected_doc))
        {
            bm25_hits += 1;
        }
        let formatted = format_query_for_embedding(eq.query, DEFAULT_EMBED_MODEL);
        let embedding = embedder.embed(&formatted).unwrap().unwrap();
        let vec = store.search_vec(&embedding, 5, None).unwrap();
        if vec
            .iter()
            .take(3)
            .any(|r| r.filepath.to_lowercase().contains(eq.expected_doc))
        {
            vec_hits += 1;
        }
    }

    let n = fusion.len() as f64;
    let hybrid_rate = hybrid_hits as f64 / n;
    let bm25_rate = bm25_hits as f64 / n;
    let vec_rate = vec_hits as f64 / n;
    println!(
        "Hybrid eval — fusion Hit@3: hybrid {hybrid_rate:.3} vs bm25 {bm25_rate:.3} / \
         vector {vec_rate:.3} (gates: ≥0.50 and ≥ max of both)"
    );

    assert!(
        hybrid_rate >= 0.5,
        "fusion Hit@3 {hybrid_rate:.3} below gate 0.50"
    );
    assert!(
        hybrid_rate >= bm25_rate.max(vec_rate),
        "fusion must match or beat the best single backend"
    );
}
