//! End-to-end embed + vsearch flow with the deterministic mock embedder —
//! no model downloads (the TS `CI=true` analogue).

mod common;

use qmd::embed::{
    EmbedOptions, Embedder as _, MockEmbedder, embedding_fingerprint, format_doc_for_embedding,
    generate_embeddings,
};
use qmd::store::extract_title;

#[test]
fn embed_vsearch_roundtrip_with_mock_embedder() {
    let store = common::indexed_store();
    let mock = MockEmbedder::default();

    let result = generate_embeddings(&store, &mock, &EmbedOptions::default()).unwrap();
    assert_eq!(result.docs_processed, 6);
    assert!(result.chunks_embedded >= 6, "at least one chunk per doc");
    assert_eq!(result.errors, 0);
    assert!(store.vec_table_exists().unwrap());

    let again = generate_embeddings(&store, &mock, &EmbedOptions::default()).unwrap();
    assert_eq!(again.docs_processed, 0);
    assert_eq!(again.chunks_embedded, 0);

    let body = std::fs::read_to_string(common::eval_docs_dir().join("machine-learning-primer.md"))
        .unwrap();
    let title = extract_title(&body, "machine-learning-primer.md");
    let chunks = qmd::chunk::chunk_document_by_tokens_default(&body, &mock).unwrap();
    let formatted = format_doc_for_embedding(&chunks[0].text, Some(&title), mock.model_uri());
    let query_vec = mock.embed(&formatted).unwrap().unwrap();

    let results = store.search_vec(&query_vec, 5, None).unwrap();
    assert!(!results.is_empty());
    assert!(results[0].filepath.contains("machine-learning"));
    assert!(results[0].score > 0.999, "exact chunk match scores ~1");
    if results.len() > 1 {
        assert!(results[1].score < 0.5);
    }

    assert!(
        store
            .search_vec(&query_vec, 5, Some("no-such-collection"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn force_and_model_change_invalidate_embeddings() {
    let store = common::indexed_store();
    let mock = MockEmbedder::default();
    generate_embeddings(&store, &mock, &EmbedOptions::default()).unwrap();

    let other = MockEmbedder::new("mock:other-model", 64);
    assert_ne!(
        embedding_fingerprint(mock.model_uri()),
        embedding_fingerprint(other.model_uri())
    );
    let pending = store
        .pending_embedding_docs(None, other.model_uri())
        .unwrap();
    assert_eq!(pending.len(), 6);

    let forced = generate_embeddings(
        &store,
        &mock,
        &EmbedOptions {
            force: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(forced.docs_processed, 6);
    assert!(forced.chunks_embedded >= 6);
}
