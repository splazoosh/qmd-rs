//! Vector search evaluation gates with the real EmbeddingGemma model,
//! ported from TS `test/eval.test.ts` ("Vector Search" describe block).
//!
//! Requires the `llama` feature and a local model (downloaded on first use);
//! `#[ignore]`d so plain `cargo test` stays model-free. Run with:
//!
//! ```text
//! cargo test --release --test eval_vector -- --ignored --nocapture
//! ```

#![cfg(feature = "llama")]

mod common;

use common::{EvalQuery, by_difficulty, indexed_store};
use qmd::Store;
use qmd::embed::{
    EmbedOptions, Embedder as _, LlamaEmbedder, format_query_for_embedding, generate_embeddings,
};
use qmd::models::DEFAULT_EMBED_MODEL;

/// Serializes model tests within this binary: two concurrent llama contexts
/// in one process produce corrupted embeddings (observed in eval_hybrid).
static MODEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn embedded_store() -> (Store, LlamaEmbedder) {
    let store = indexed_store();
    let embedder = LlamaEmbedder::load(DEFAULT_EMBED_MODEL).expect("load embed model");
    let result = generate_embeddings(&store, &embedder, &EmbedOptions::default()).unwrap();
    assert_eq!(result.docs_processed, 6);
    assert_eq!(result.errors, 0, "failures: {:?}", result.failures);
    (store, embedder)
}

fn hit_rate(store: &Store, embedder: &LlamaEmbedder, queries: &[&EvalQuery], top_k: usize) -> f64 {
    let mut hits = 0usize;
    for eq in queries {
        let formatted = format_query_for_embedding(eq.query, DEFAULT_EMBED_MODEL);
        let embedding = embedder.embed(&formatted).unwrap().unwrap();
        let results = store.search_vec(&embedding, 5, None).unwrap();
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
#[ignore = "requires the real embedding model (run locally)"]
fn vector_eval_gates() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let (store, embedder) = embedded_store();

    let easy = hit_rate(&store, &embedder, &by_difficulty("easy"), 3);
    let medium = hit_rate(&store, &embedder, &by_difficulty("medium"), 3);
    let hard = hit_rate(&store, &embedder, &by_difficulty("hard"), 5);
    let all: Vec<&EvalQuery> = common::EVAL_QUERIES.iter().collect();
    let overall = hit_rate(&store, &embedder, &all, 3);

    println!("Vector eval — easy Hit@3: {easy:.3} (gate 0.60)");
    println!("Vector eval — medium Hit@3: {medium:.3} (gate 0.40)");
    println!("Vector eval — hard Hit@5: {hard:.3} (gate 0.30)");
    println!("Vector eval — overall Hit@3: {overall:.3} (gate 0.50, TS baseline ~0.70)");

    assert!(easy >= 0.6, "easy Hit@3 {easy:.3} below gate 0.60");
    assert!(medium >= 0.4, "medium Hit@3 {medium:.3} below gate 0.40");
    assert!(hard >= 0.3, "hard Hit@5 {hard:.3} below gate 0.30");
    assert!(overall >= 0.5, "overall Hit@3 {overall:.3} below gate 0.50");
}

/// Sanity parity check: document/query template
/// cosines behave (paraphrase > matching query > unrelated).
#[test]
#[ignore = "requires the real embedding model (run locally)"]
fn embedding_sanity_cosines() {
    let _guard = MODEL_LOCK.lock().unwrap();
    let embedder = LlamaEmbedder::load(DEFAULT_EMBED_MODEL).expect("load embed model");
    let embed = |text: &str| embedder.embed(text).unwrap().unwrap();
    let cosine = |a: &[f32], b: &[f32]| {
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        dot / (na * nb)
    };

    let doc = embed("title: none | text: The quick brown fox jumps over the lazy dog.");
    assert_eq!(doc.len(), 768, "EmbeddingGemma is 768-dim");
    let paraphrase = embed("title: none | text: A fast auburn fox leaps above a sleepy dog.");
    let query = embed("task: search result | query: fox jumping over dog");
    let unrelated =
        embed("title: none | text: SQLite is a small, fast, self-contained SQL database engine.");

    let c_para = cosine(&doc, &paraphrase);
    let c_query = cosine(&doc, &query);
    let c_unrelated = cosine(&doc, &unrelated);
    println!("cos(doc, paraphrase) = {c_para:.4}");
    println!("cos(doc, query)      = {c_query:.4}");
    println!("cos(doc, unrelated)  = {c_unrelated:.4}");
    assert!(c_para > c_query && c_query > c_unrelated);
    assert!(c_unrelated < 0.4);
}
