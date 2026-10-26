//! End-to-end embed + vsearch flow with the deterministic mock embedder —
//! no model downloads (the TS `CI=true` analogue).

mod common;

use qmd::embed::{
    EmbedOptions, EmbedStop, Embedder as _, MockEmbedder, embedding_fingerprint,
    format_doc_for_embedding, generate_embeddings,
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

#[test]
fn cleanup_removes_orphaned_vectors_and_leaves_nothing_to_re_embed() {
    let store = common::indexed_store();
    let mock = MockEmbedder::default();
    generate_embeddings(&store, &mock, &EmbedOptions::default()).unwrap();
    assert_eq!(
        store.preview_cleanup().unwrap(),
        qmd::store::CleanupStats::default()
    );

    store
        .deactivate_document("eval-docs", "machine-learning-primer.md")
        .unwrap();
    let preview = store.preview_cleanup().unwrap();
    assert!(preview.orphaned_vectors > 0, "{preview:?}");
    assert_eq!(preview.inactive_docs, 1);
    assert_eq!(preview.orphaned_content, 1);

    let removed = store.cleanup().unwrap();
    assert_eq!(removed, preview);
    assert_eq!(
        store.preview_cleanup().unwrap(),
        qmd::store::CleanupStats::default()
    );
    let dangling: i64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM vectors_vec", [], |r| r.get(0))
        .unwrap();
    let recorded: i64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        dangling, recorded,
        "vec0 rows and their records leave together"
    );

    let again = generate_embeddings(&store, &mock, &EmbedOptions::default()).unwrap();
    assert_eq!(
        again.docs_processed, 0,
        "the active documents keep their vectors"
    );
}

/// A backend whose batch calls always fail and whose single calls fail
/// `fail_first` times for each text `fails` picks, then succeed.
struct Flaky {
    inner: MockEmbedder,
    fails: fn(&str) -> bool,
    fail_first: usize,
    attempts: std::sync::Mutex<std::collections::HashMap<String, usize>>,
}

impl Flaky {
    fn new(fails: fn(&str) -> bool, fail_first: usize) -> Self {
        Self {
            inner: MockEmbedder::default(),
            fails,
            fail_first,
            attempts: std::sync::Mutex::default(),
        }
    }

    fn single_calls(&self) -> usize {
        self.attempts.lock().unwrap().values().sum()
    }
}

impl qmd::embed::Embedder for Flaky {
    fn model_uri(&self) -> &str {
        self.inner.model_uri()
    }

    fn embed_batch(&self, texts: &[String]) -> qmd::Result<Vec<Option<Vec<f32>>>> {
        if let [text] = texts {
            if (self.fails)(text) {
                let mut attempts = self.attempts.lock().unwrap();
                let n = attempts.entry(text.clone()).or_default();
                *n += 1;
                if *n <= self.fail_first {
                    return Err(qmd::Error::Other(format!("attempt {n} failed")));
                }
            }
            return self.inner.embed_batch(texts);
        }
        Err(qmd::Error::Other("batch call failed".to_owned()))
    }
}

impl qmd::chunk::Tokenizer for Flaky {
    fn count_tokens(&self, text: &str) -> qmd::Result<usize> {
        self.inner.count_tokens(text)
    }

    fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> qmd::Result<String> {
        self.inner.truncate_to_tokens(text, max_tokens)
    }
}

fn pending(store: &qmd::Store) -> usize {
    store
        .pending_embedding_docs(None, MockEmbedder::default().model_uri())
        .unwrap()
        .len()
}

#[test]
fn a_failed_batch_falls_back_to_single_chunks_and_retries_each_up_to_three_times() {
    let store = common::indexed_store();
    let recovers = Flaky::new(|t| !t.contains("dimension probe"), 2);
    let result = generate_embeddings(&store, &recovers, &EmbedOptions::default()).unwrap();
    assert_eq!(result.errors, 0, "{:?}", result.failures);
    assert!(result.stopped.is_none());
    assert_eq!(pending(&store), 0);
    assert_eq!(recovers.single_calls(), 3 * result.chunks_embedded);

    let store = common::indexed_store();
    let never = Flaky::new(|t| t.contains("Remote Work Policy"), usize::MAX);
    let result = generate_embeddings(&store, &never, &EmbedOptions::default()).unwrap();
    assert!(result.errors > 0);
    assert!(
        result
            .failures
            .iter()
            .all(|f| f.path == "remote-work-policy.md" && f.attempts == 3),
        "{:?}",
        result.failures
    );
    assert!(result.stopped.is_none(), "one bad document is no abort");
    assert_eq!(
        pending(&store),
        1,
        "its partial set is removed, so it stays pending"
    );
}

#[test]
fn embedding_stops_once_most_chunks_fail_and_leaves_the_rest_pending() {
    let store = qmd::Store::open_in_memory().unwrap();
    let now = qmd::store::now_iso();
    for i in 0..80 {
        let body = format!("# Note {i}\n\nA short note numbered {i}.\n");
        let hash = qmd::store::hash_content(&body);
        store.insert_content(&hash, &body, &now).unwrap();
        store
            .insert_document("notes", &format!("n{i:02}.md"), "Note", &hash, &now, &now)
            .unwrap();
    }
    let broken = Flaky::new(|t| t.contains("note numbered"), usize::MAX);
    let result = generate_embeddings(&store, &broken, &EmbedOptions::default()).unwrap();
    assert_eq!(
        result.stopped,
        Some(EmbedStop::ErrorRate {
            failed: 32,
            attempted: 32
        })
    );
    assert_eq!(result.chunks_embedded, 0);
    assert_eq!(broken.single_calls(), 32, "no retries once stopping");
    assert_eq!(pending(&store), 80);
}

#[test]
fn chunks_of_a_removed_partial_set_still_count_as_attempted_for_the_error_rate() {
    let store = qmd::Store::open_in_memory().unwrap();
    let now = qmd::store::now_iso();
    let filler = "Plain words that embed without trouble. ".repeat(100);
    for i in 0..40 {
        let body = format!("# Doc {i}\n\npoisoned {i}\n\n{filler}\n\n{filler}\n\n{filler}\n");
        let hash = qmd::store::hash_content(&body);
        store.insert_content(&hash, &body, &now).unwrap();
        store
            .insert_document("docs", &format!("d{i:02}.md"), "Doc", &hash, &now, &now)
            .unwrap();
    }
    let one_bad_chunk_each = Flaky::new(|t| t.contains("poisoned"), usize::MAX);
    let options = EmbedOptions {
        max_docs_per_batch: Some(4),
        ..Default::default()
    };
    let result = generate_embeddings(&store, &one_bad_chunk_each, &options).unwrap();
    assert_eq!(result.errors, 40, "{:?}", result.stopped);
    assert!(
        result.stopped.is_none(),
        "one chunk in four failing is no abort"
    );
}

#[test]
fn embedding_stops_at_its_time_cap_and_resumes_on_the_next_run() {
    let store = common::indexed_store();
    let mock = MockEmbedder::default();
    let capped = EmbedOptions {
        max_duration: Some(std::time::Duration::ZERO),
        ..Default::default()
    };
    let result = generate_embeddings(&store, &mock, &capped).unwrap();
    assert_eq!(
        result.stopped,
        Some(EmbedStop::TimeLimit(std::time::Duration::ZERO))
    );
    assert_eq!(result.chunks_embedded, 0);
    assert_eq!(pending(&store), 6);

    let result = generate_embeddings(&store, &mock, &EmbedOptions::default()).unwrap();
    assert!(result.stopped.is_none());
    assert_eq!(pending(&store), 0);
}

#[test]
fn embed_deletes_the_vectors_an_edited_document_leaves_behind() {
    let store = common::indexed_store();
    let mock = MockEmbedder::default();
    generate_embeddings(&store, &mock, &EmbedOptions::default()).unwrap();

    let body = "# Remote Work Policy\n\nRewritten.\n";
    let hash = qmd::store::hash_content(body);
    store.insert_content(&hash, body, "2026-01-01").unwrap();
    store
        .conn()
        .execute(
            "UPDATE documents SET hash = ?1 WHERE path = 'remote-work-policy.md'",
            [&hash],
        )
        .unwrap();
    let orphaned = store.preview_cleanup().unwrap().orphaned_vectors;
    assert!(orphaned > 0);

    let result = generate_embeddings(&store, &mock, &EmbedOptions::default()).unwrap();
    assert_eq!(result.orphans_removed, orphaned);
    assert_eq!(result.docs_processed, 1);
    assert_eq!(store.preview_cleanup().unwrap().orphaned_vectors, 0);
    let count = |table: &str| -> i64 {
        store
            .conn()
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    };
    assert_eq!(
        count("vectors_vec"),
        count("content_vectors"),
        "vec0 rows and their records leave together"
    );
}
