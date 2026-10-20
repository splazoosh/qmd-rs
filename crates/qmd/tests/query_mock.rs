//! End-to-end hybrid `query` pipeline with the deterministic mock embedder —
//! no model downloads (the TS `CI=true` analogue).

mod common;

use qmd::embed::{EmbedOptions, MockEmbedder, generate_embeddings};
use qmd::query::{
    HybridOptions, QueryType, TypedQuery, hybrid_query, parse_structured_query, structured_search,
};

fn embedded_store() -> (qmd::Store, MockEmbedder) {
    let store = common::indexed_store();
    let mock = MockEmbedder::default();
    let result = generate_embeddings(&store, &mock, &EmbedOptions::default()).unwrap();
    assert_eq!(result.errors, 0);
    (store, mock)
}

#[test]
fn hybrid_query_fuses_bm25_and_vector_lists() {
    let (store, mock) = embedded_store();

    let response = hybrid_query(
        &store,
        Some(&mock),
        "CAP theorem",
        &HybridOptions::default(),
    )
    .unwrap();

    assert!(!response.results.is_empty());
    assert!(response.results[0].filepath.contains("distributed-systems"));
    assert_eq!(response.report.lists_fused, 2, "FTS + vector lists");
    assert_eq!(response.report.queries_embedded, 1);
    assert!(!response.report.vector_skipped);

    for (i, r) in response.results.iter().enumerate() {
        assert!((r.score - 1.0 / (i as f64 + 1.0)).abs() < 1e-12);
    }
    let mut files: Vec<&str> = response
        .results
        .iter()
        .map(|r| r.filepath.as_str())
        .collect();
    files.sort_unstable();
    files.dedup();
    assert_eq!(files.len(), response.results.len());
    for r in &response.results {
        assert!(!r.best_chunk.is_empty());
        assert!(r.body.contains(&r.best_chunk));
    }
}

#[test]
fn hybrid_query_without_embedder_degrades_to_bm25() {
    let (store, _mock) = embedded_store();
    let response = hybrid_query(&store, None, "CAP theorem", &HybridOptions::default()).unwrap();
    assert!(response.report.vector_skipped);
    assert_eq!(response.report.lists_fused, 1);
    assert!(response.results[0].filepath.contains("distributed-systems"));
}

#[test]
fn hybrid_query_reports_strong_signal_and_intent_disables_it() {
    let (store, mock) = embedded_store();

    let strong = hybrid_query(
        &store,
        Some(&mock),
        "Project Phoenix retrospective",
        &HybridOptions::default(),
    )
    .unwrap();
    assert!(strong.report.strong_signal.is_some(), "expected bypass");

    let with_intent = hybrid_query(
        &store,
        Some(&mock),
        "Project Phoenix retrospective",
        &HybridOptions {
            intent: Some("product launches".to_owned()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        with_intent.report.strong_signal.is_none(),
        "an intent must disable the bypass"
    );
}

#[test]
fn hybrid_query_respects_limit_and_min_score() {
    let (store, mock) = embedded_store();
    let response = hybrid_query(
        &store,
        Some(&mock),
        "the",
        &HybridOptions {
            limit: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(response.results.len() <= 2);

    let filtered = hybrid_query(
        &store,
        Some(&mock),
        "data",
        &HybridOptions {
            min_score: 0.6,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(filtered.results.len() <= 1);
}

#[test]
fn several_collections_filter_before_ranking_not_after() {
    use qmd::store::{hash_content, now_iso};
    let store = qmd::Store::open_in_memory().unwrap();
    let now = now_iso();
    let add = |collection: &str, path: &str, body: String| {
        let hash = hash_content(&body);
        store.insert_content(&hash, &body, &now).unwrap();
        store
            .insert_document(collection, path, path, &hash, &now, &now)
            .unwrap();
    };
    for i in 0..100 {
        add(
            "noise",
            &format!("n{i}.md"),
            format!("zebra zebra zebra {i}"),
        );
    }
    add(
        "a",
        "a.md",
        "one zebra among much other prose here".to_owned(),
    );
    add(
        "b",
        "b.md",
        "a single zebra in a longer document body".to_owned(),
    );

    let response = hybrid_query(
        &store,
        None,
        "zebra",
        &HybridOptions {
            collections: vec!["a".to_owned(), "b".to_owned()],
            ..Default::default()
        },
    )
    .unwrap();
    let mut files: Vec<&str> = response
        .results
        .iter()
        .map(|r| r.filepath.as_str())
        .collect();
    files.sort_unstable();
    assert_eq!(files, ["qmd://a/a.md", "qmd://b/b.md"]);
}

#[test]
fn structured_search_routes_by_type() {
    let (store, mock) = embedded_store();

    let parsed = parse_structured_query(
        "intent: distributed databases\nlex: CAP theorem\nvec: consistency availability tradeoffs",
    )
    .unwrap()
    .unwrap();
    assert_eq!(parsed.intent.as_deref(), Some("distributed databases"));

    let response = structured_search(
        &store,
        Some(&mock),
        &parsed.searches,
        &HybridOptions {
            intent: parsed.intent.clone(),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!response.results.is_empty());
    assert!(response.results[0].filepath.contains("distributed-systems"));
    assert_eq!(response.report.queries_embedded, 1, "one vec line embedded");

    let lex_only = structured_search(
        &store,
        None,
        &[TypedQuery {
            query_type: QueryType::Lex,
            query: "overfitting".to_owned(),
            line: 1,
        }],
        &HybridOptions::default(),
    )
    .unwrap();
    assert_eq!(lex_only.report.queries_embedded, 0);
    assert!(
        !lex_only.report.vector_skipped,
        "no vec/hyde lines, so nothing was skipped"
    );
    assert!(lex_only.results[0].filepath.contains("machine-learning"));
}

#[test]
fn structured_search_validates_sub_queries() {
    let (store, mock) = embedded_store();
    let bad_lex = TypedQuery {
        query_type: QueryType::Lex,
        query: "\"unbalanced".to_owned(),
        line: 1,
    };
    let err =
        structured_search(&store, Some(&mock), &[bad_lex], &HybridOptions::default()).unwrap_err();
    assert!(err.to_string().contains("unmatched double quote"), "{err}");

    let bad_vec = TypedQuery {
        query_type: QueryType::Vec,
        query: "consistency -availability".to_owned(),
        line: 2,
    };
    let err =
        structured_search(&store, Some(&mock), &[bad_vec], &HybridOptions::default()).unwrap_err();
    assert!(err.to_string().contains("Negation"), "{err}");
}

#[test]
fn hybrid_query_on_empty_index_is_empty() {
    let store = qmd::Store::open_in_memory().unwrap();
    let mock = MockEmbedder::default();
    let response =
        hybrid_query(&store, Some(&mock), "anything", &HybridOptions::default()).unwrap();
    assert!(response.results.is_empty());
    assert_eq!(response.report.lists_fused, 0);
}
