//! End-to-end hybrid `query` pipeline with the deterministic mock embedder —
//! no model downloads (the TS `CI=true` analogue).

mod common;

use qmd::embed::{EmbedOptions, MockEmbedder, generate_embeddings};
use qmd::query::{
    HybridOptions, QueryType, TypedQuery, hybrid_query, parse_structured_query, structured_search,
};

/// Options with every BM25 term required. Mock vectors are noise, so under
/// `any` a document at vector rank 0 that the OR list also holds wins the
/// fusion; tests of fusion and rerank mechanics need the one-document list.
fn all_terms() -> HybridOptions {
    HybridOptions {
        term_match: qmd::store::TermMatch::All,
        ..Default::default()
    }
}

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

    let response = hybrid_query(&store, Some(&mock), None, "CAP theorem", &all_terms()).unwrap();

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
    let response =
        hybrid_query(&store, None, None, "CAP theorem", &HybridOptions::default()).unwrap();
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
        None,
        "Project Phoenix retrospective",
        &HybridOptions::default(),
    )
    .unwrap();
    assert!(strong.report.strong_signal.is_some(), "expected bypass");

    let with_intent = hybrid_query(
        &store,
        Some(&mock),
        None,
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
        None,
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
        None,
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
        None,
        &parsed.searches,
        &HybridOptions {
            intent: parsed.intent.clone(),
            ..all_terms()
        },
    )
    .unwrap();
    assert!(!response.results.is_empty());
    assert!(response.results[0].filepath.contains("distributed-systems"));
    assert_eq!(response.report.queries_embedded, 1, "one vec line embedded");

    let lex_only = structured_search(
        &store,
        None,
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
    let err = structured_search(
        &store,
        Some(&mock),
        None,
        &[bad_lex],
        &HybridOptions::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("unmatched double quote"), "{err}");

    let bad_vec = TypedQuery {
        query_type: QueryType::Vec,
        query: "consistency -availability".to_owned(),
        line: 2,
    };
    let err = structured_search(
        &store,
        Some(&mock),
        None,
        &[bad_vec],
        &HybridOptions::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("Negation"), "{err}");
}

#[test]
fn hybrid_query_on_empty_index_is_empty() {
    let store = qmd::Store::open_in_memory().unwrap();
    let mock = MockEmbedder::default();
    let response = hybrid_query(
        &store,
        Some(&mock),
        None,
        "anything",
        &HybridOptions::default(),
    )
    .unwrap();
    assert!(response.results.is_empty());
    assert_eq!(response.report.lists_fused, 0);
}

#[test]
fn any_term_match_finds_what_all_terms_cannot() {
    let (store, mock) = embedded_store();
    let query = "Sequoia overfitting";
    assert!(store.search_fts(query, 10, None).unwrap().is_empty());
    let any = store
        .search_fts_with(query, 10, None, qmd::store::TermMatch::Any)
        .unwrap();
    assert!(any.len() >= 2, "each word matches its own document");

    let options = HybridOptions {
        term_match: qmd::store::TermMatch::Any,
        ..Default::default()
    };
    let lex = TypedQuery {
        query_type: QueryType::Lex,
        query: query.to_owned(),
        line: 1,
    };
    let response = structured_search(&store, Some(&mock), None, &[lex], &options).unwrap();
    assert!(response.results.len() >= 2, "lex lines follow the mode too");
}

#[test]
fn rerank_reorders_by_blend_and_reports() {
    use qmd::rerank::MockReranker;
    let (store, mock) = embedded_store();
    let options = all_terms();
    let plain = hybrid_query(&store, Some(&mock), None, "CAP theorem", &options).unwrap();
    let reranked = hybrid_query(
        &store,
        Some(&mock),
        Some(&MockReranker),
        "CAP theorem",
        &options,
    )
    .unwrap();
    assert_eq!(plain.report.reranked, 0);
    assert_eq!(reranked.report.reranked, reranked.report.rerank_fresh);
    assert!(reranked.report.reranked >= reranked.results.len());
    assert!(
        reranked.results[0].filepath.contains("distributed-systems"),
        "the document holding both words stays first"
    );
    assert!(
        reranked
            .results
            .windows(2)
            .all(|w| w[0].score >= w[1].score),
        "results are sorted by the blended score"
    );

    let again = hybrid_query(
        &store,
        Some(&mock),
        Some(&MockReranker),
        "CAP theorem",
        &options,
    )
    .unwrap();
    assert_eq!(again.report.rerank_fresh, 0);
}

#[test]
fn explain_traces_each_result_and_changes_no_ranking() {
    use qmd::query::{Backend, HybridResult};
    use qmd::rerank::MockReranker;
    let (store, mock) = embedded_store();
    let plain = HybridOptions::default();
    let explained = HybridOptions {
        explain: true,
        ..HybridOptions::default()
    };
    let ranking = |rs: &[HybridResult]| -> Vec<(String, u64)> {
        rs.iter()
            .map(|r| (r.filepath.clone(), r.score.to_bits()))
            .collect()
    };

    for reranker in [None, Some(&MockReranker as &dyn qmd::rerank::Reranker)] {
        let off = hybrid_query(&store, Some(&mock), reranker, "CAP theorem", &plain).unwrap();
        let on = hybrid_query(&store, Some(&mock), reranker, "CAP theorem", &explained).unwrap();
        assert_eq!(ranking(&off.results), ranking(&on.results));
        assert!(off.results.iter().all(|r| r.explain.is_none()));
        for r in &on.results {
            let e = r.explain.as_ref().expect("every result is traced");
            assert_eq!(e.rrf_rank, r.rrf_rank);
            assert_eq!(e.final_score.to_bits(), r.score.to_bits());
            assert!(!e.contributions.is_empty());
            let base: f64 = e.contributions.iter().map(|c| c.rrf).sum();
            assert!((base - e.rrf_base).abs() < 1e-12);
            assert_eq!(e.rerank.is_some(), reranker.is_some());
            if let Some(t) = &e.rerank {
                assert_eq!(Some(t.rerank_score), r.rerank_score);
            }
            let text = e.to_string();
            assert!(text.contains("rrf: base="), "{text}");
            assert!(
                text.contains(if reranker.is_some() {
                    "x rerank"
                } else {
                    "(no rerank)"
                }),
                "{text}"
            );
        }
        let top = on.results[0].explain.as_ref().unwrap();
        assert!(
            top.contributions
                .iter()
                .any(|c| c.source.backend == Backend::Fts)
        );
    }

    let parsed = parse_structured_query("lex: CAP theorem\nvec: consistency tradeoffs")
        .unwrap()
        .unwrap();
    let on = structured_search(&store, Some(&mock), None, &parsed.searches, &explained).unwrap();
    let kinds: Vec<&str> = on
        .results
        .iter()
        .flat_map(|r| r.explain.as_ref().unwrap().contributions.iter())
        .map(|c| c.source.kind)
        .collect();
    assert!(
        kinds.contains(&"lex") && kinds.contains(&"vec"),
        "{kinds:?}"
    );
    let json = serde_json::to_value(on.results[0].explain.as_ref().unwrap()).unwrap();
    assert!(json["contributions"][0]["backend"].is_string(), "{json}");
}

/// Two concepts: `zebra`/`striped` on one axis and, in documents only,
/// `animal` on another, so a query naming both lands on the zebra chunk.
struct ConceptEmbedder;

impl qmd::embed::Embedder for ConceptEmbedder {
    fn model_uri(&self) -> &str {
        "mock:concepts"
    }

    fn embed_batch(&self, texts: &[String]) -> qmd::error::Result<Vec<Option<Vec<f32>>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let has = |w: &str| f32::from(u8::from(t.contains(w)));
                let animal = if t.starts_with("task:") {
                    0.0
                } else {
                    has("animal")
                };
                Some(vec![has("zebra").max(has("striped")), animal, 1e-3])
            })
            .collect())
    }
}

impl qmd::chunk::Tokenizer for ConceptEmbedder {
    fn count_tokens(&self, text: &str) -> qmd::error::Result<usize> {
        Ok(text.chars().count().div_ceil(4))
    }

    fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> qmd::error::Result<String> {
        Ok(text.chars().take(max_tokens * 4).collect())
    }
}

#[test]
fn rerank_chunk_rules_choose_what_the_reranker_reads() {
    use qmd::query::RerankChunk;
    use qmd::rerank::MockReranker;
    use qmd::store::{hash_content, now_iso};

    let store = qmd::Store::open_in_memory().unwrap();
    let now = now_iso();
    let body = format!(
        "# Notes\n\n{}{}\n\n## Later\n\n{}",
        "The animal kept walking. ".repeat(72),
        "Nothing more to say here. ".repeat(22),
        "A zebra grazed quietly. ".repeat(60),
    );
    let hash = hash_content(&body);
    store.insert_content(&hash, &body, &now).unwrap();
    store
        .insert_document("c", "notes.md", "Notes", &hash, &now, &now)
        .unwrap();
    let embedder = ConceptEmbedder;
    generate_embeddings(&store, &embedder, &EmbedOptions::default()).unwrap();
    let stored = store.chunk_distances(&hash, &[1.0, 0.0, 1e-3]).unwrap();
    assert_eq!(stored.len(), 2);
    let zebra_pos = stored[0].0;
    assert!(zebra_pos > 0);

    let read = |rule| {
        let options = HybridOptions {
            rerank_chunk: rule,
            ..Default::default()
        };
        let response = hybrid_query(
            &store,
            Some(&embedder),
            Some(&MockReranker),
            "striped animal",
            &options,
        )
        .unwrap();
        let top = response.results.into_iter().next().unwrap();
        assert!(body.contains(&top.best_chunk));
        assert_eq!(top.rrf_rank, 1);
        (
            top.best_chunk_pos,
            top.best_chunk,
            top.rerank_score.unwrap(),
        )
    };

    let (pos, _, score) = read(RerankChunk::Overlap);
    assert_eq!((pos, score), (0, 0.5), "the chunk holding `animal`");

    let (pos, chunk, score) = read(RerankChunk::Nearest);
    assert_eq!(
        (pos, score),
        (zebra_pos, 0.0),
        "the chunk vector search matched"
    );
    assert!(chunk.contains("zebra") && !chunk.contains("animal"));

    let (pos, _, score) = read(RerankChunk::BestOfNearest(2));
    assert_eq!((pos, score), (0, 0.5), "the farther chunk scored higher");
}

#[test]
fn nearest_chunk_is_cut_with_the_geometry_its_vectors_were() {
    use qmd::chunk::{ChunkParams, chunk_document_by_tokens_for};
    use qmd::query::RerankChunk;
    use qmd::rerank::MockReranker;
    use qmd::store::{hash_content, now_iso};

    let store = qmd::Store::open_in_memory().unwrap();
    let now = now_iso();
    let body = format!(
        "# Notes\n\n{}\n\n## Later\n\n{}",
        "The animal kept walking. ".repeat(40),
        "A zebra grazed quietly. ".repeat(40),
    );
    let hash = hash_content(&body);
    store.insert_content(&hash, &body, &now).unwrap();
    let mut small = qmd::config::Collection::new("/a", "**/*.md");
    small.chunk_tokens = Some(100);
    store.upsert_store_collection("a", &small).unwrap();
    for collection in ["a", "b"] {
        store
            .insert_document(collection, "notes.md", "Notes", &hash, &now, &now)
            .unwrap();
    }
    let embedder = ConceptEmbedder;
    generate_embeddings(&store, &embedder, &EmbedOptions::default()).unwrap();
    let stored =
        chunk_document_by_tokens_for(&body, &embedder, &ChunkParams::for_size(100)).unwrap();

    let options = HybridOptions {
        collections: vec!["b".to_owned()],
        rerank_chunk: RerankChunk::Nearest,
        ..Default::default()
    };
    let response = hybrid_query(
        &store,
        Some(&embedder),
        Some(&MockReranker),
        "striped animal",
        &options,
    )
    .unwrap();
    let top = response.results.into_iter().next().unwrap();
    let chunk = stored.iter().find(|c| c.pos == top.best_chunk_pos).unwrap();
    assert_eq!(top.best_chunk, chunk.text);
}
