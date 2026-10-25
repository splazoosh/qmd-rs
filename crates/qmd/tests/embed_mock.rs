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

#[test]
fn collection_chunk_size_rechunks_and_replaces_the_old_set() {
    let store = common::indexed_store();
    let mock = MockEmbedder::default();
    let default_run = generate_embeddings(&store, &mock, &EmbedOptions::default()).unwrap();

    let mut collection = qmd::config::Collection::new("/eval-docs", "**/*.md");
    collection.chunk_tokens = Some(128);
    store
        .upsert_store_collection("eval-docs", &collection)
        .unwrap();

    let small = generate_embeddings(&store, &mock, &EmbedOptions::default()).unwrap();
    assert_eq!(
        small.docs_processed, 6,
        "a new size re-embeds the collection"
    );
    assert!(small.chunks_embedded > default_run.chunks_embedded);

    let fingerprint = qmd::embed::embedding_fingerprint_for(
        mock.model_uri(),
        &qmd::chunk::ChunkParams::for_size(128),
    );
    let (stored, stale): (i64, i64) = store
        .conn()
        .query_row(
            "SELECT COUNT(*), SUM(embed_fingerprint != ?1) FROM content_vectors",
            [&fingerprint],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(stored as usize, small.chunks_embedded);
    assert_eq!(stale, 0, "no chunk of the 900-token set survives");
    assert!(
        store
            .pending_embedding_docs(None, mock.model_uri())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn collection_chunk_boundary_reembeds_and_replaces_the_old_set() {
    use qmd::chunk::{ChunkBoundary, ChunkParams};
    let store = common::indexed_store();
    let mock = MockEmbedder::default();
    generate_embeddings(&store, &mock, &EmbedOptions::default()).unwrap();

    let mut collection = qmd::config::Collection::new("/eval-docs", "**/*.md");
    collection.chunk_boundary = Some(ChunkBoundary::Heading);
    store
        .upsert_store_collection("eval-docs", &collection)
        .unwrap();
    assert_eq!(
        store.collection_chunk_params("eval-docs").unwrap(),
        ChunkParams::default().with_boundary(ChunkBoundary::Heading)
    );

    let heading = generate_embeddings(&store, &mock, &EmbedOptions::default()).unwrap();
    assert_eq!(heading.docs_processed, 6, "a new boundary re-embeds");
    let fingerprint = qmd::embed::embedding_fingerprint_for(
        mock.model_uri(),
        &ChunkParams::default().with_boundary(ChunkBoundary::Heading),
    );
    let stale: i64 = store
        .conn()
        .query_row(
            "SELECT SUM(embed_fingerprint != ?1) FROM content_vectors",
            [&fingerprint],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stale, 0, "no window-cut chunk survives");
}
