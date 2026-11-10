//! Metadata persistence, `qmd update` extraction and backfill, and discovery.
//! Ported from upstream's `test/metadata-store.test.ts` and
//! `test/metadata-discovery.test.ts`.

use std::path::Path;

use qmd::Store;
use qmd::index::reindex_collection;
use qmd::metadata::report::{Limit, MetadataKeySummary, SQL_BINDING_BUDGET, ValueSort};
use qmd::metadata::{
    ExtractionResult, ListMetadataOptions, ListMetadataResult, METADATA_EXTRACTION_VERSION,
    MetadataError, MetadataFilter, MetadataMatch, Scalar, ValueType, parse_metadata_json,
};
use qmd::store::{TermMatch, hash_content};
use serde_json::{Value, json};

struct Fixture {
    store: Store,
    next: usize,
}

impl Fixture {
    fn new() -> Self {
        Self {
            store: Store::open_in_memory().unwrap(),
            next: 0,
        }
    }

    /// An active document with extracted metadata; every body holds "doc".
    fn doc(&mut self, collection: &str, metadata: Value) -> i64 {
        self.next += 1;
        let path = format!("doc-{}.md", self.next);
        let body = format!("# doc {n}\n\nbody of doc {n}\n", n = self.next);
        let hash = hash_content(&body);
        self.store.insert_content(&hash, &body, "t").unwrap();
        self.store
            .insert_document(collection, &path, &path, &hash, "t", "t")
            .unwrap();
        let (id, _, _) = self
            .store
            .find_active_document(collection, &path)
            .unwrap()
            .unwrap();
        self.store
            .replace_document_metadata(id, &extraction(metadata))
            .unwrap();
        id
    }

    fn list(&self, options: &ListMetadataOptions) -> ListMetadataResult {
        self.store.list_metadata(options).unwrap()
    }

    fn sql(&self, sql: &str, id: i64) {
        self.store.conn().execute(sql, [id]).unwrap();
    }
}

fn extraction(metadata: Value) -> ExtractionResult {
    ExtractionResult {
        metadata: parse_metadata_json(Some(&metadata.to_string())),
        error: None,
        extraction_version: METADATA_EXTRACTION_VERSION,
    }
}

fn matching(json: Value) -> ListMetadataOptions {
    ListMetadataOptions {
        r#match: Some(MetadataMatch::parse(&json).unwrap()),
        ..Default::default()
    }
}

fn filtered(json: Value) -> ListMetadataOptions {
    ListMetadataOptions {
        filter: Some(MetadataFilter::parse(&json).unwrap()),
        ..Default::default()
    }
}

fn in_collections(names: &[&str]) -> ListMetadataOptions {
    ListMetadataOptions {
        collections: Some(names.iter().map(|n| (*n).to_owned()).collect()),
        ..Default::default()
    }
}

fn summary<'a>(result: &'a ListMetadataResult, key: &str) -> &'a MetadataKeySummary {
    result
        .keys
        .iter()
        .find(|s| s.key == key)
        .unwrap_or_else(|| panic!("key {key} missing"))
}

fn keys(result: &ListMetadataResult) -> Vec<&str> {
    result.keys.iter().map(|s| s.key.as_str()).collect()
}

/// The first type's value window as `(value, documents)` JSON pairs.
fn values(result: &ListMetadataResult, key: &str) -> Vec<(Value, u64)> {
    summary(result, key).types[0]
        .values
        .iter()
        .map(|v| (v.value.to_json(), v.documents))
        .collect()
}

fn pairs(expected: &[(Value, u64)]) -> Vec<(Value, u64)> {
    expected.to_vec()
}

#[test]
fn replacing_metadata_keeps_one_row_per_value_and_records_empty_and_failed_extractions() {
    let mut f = Fixture::new();
    let id = f.doc("c", json!({"topics": ["a", "b"], "status": "x"}));
    let rows = |f: &Fixture| -> i64 {
        f.store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM document_metadata_values WHERE document_id = ?1",
                [id],
                |r| r.get(0),
            )
            .unwrap()
    };
    assert_eq!(rows(&f), 3);
    f.store
        .replace_document_metadata(id, &extraction(json!({})))
        .unwrap();
    assert_eq!(rows(&f), 0);
    assert_eq!(
        f.store.count_documents_pending_metadata(None).unwrap(),
        0,
        "empty is extracted"
    );

    f.store
        .replace_document_metadata(id, &extraction(json!({"status": "x"})))
        .unwrap();
    let failed = ExtractionResult {
        error: Some("boom".into()),
        ..extraction(json!({}))
    };
    f.store.replace_document_metadata(id, &failed).unwrap();
    assert_eq!(rows(&f), 0, "a failed extraction leaves no stale values");
    assert_eq!(f.store.count_documents_pending_metadata(None).unwrap(), 0);
    assert_eq!(f.store.metadata_failures(None).unwrap().len(), 1);

    assert!(
        f.store
            .sync_document_metadata(id, "---\nqmd:\n  metadata:\n    a: 1\n---\n", "x.md", true)
            .unwrap()
            .is_none()
    );
    let fresh = f
        .store
        .sync_document_metadata(id, "---\nqmd:\n  metadata:\n    a: 1\n---\n", "x.md", false)
        .unwrap()
        .unwrap();
    assert!(fresh.error.is_none());
    assert_eq!(rows(&f), 1);
}

#[test]
fn two_paths_sharing_content_keep_their_own_metadata() {
    let store = Store::open_in_memory().unwrap();
    let body = "same body";
    let hash = hash_content(body);
    store.insert_content(&hash, body, "t").unwrap();
    for (path, status) in [("a.md", "published"), ("b.md", "draft")] {
        store
            .insert_document("c", path, path, &hash, "t", "t")
            .unwrap();
        let (id, _, _) = store.find_active_document("c", path).unwrap().unwrap();
        store
            .replace_document_metadata(id, &extraction(json!({"status": status})))
            .unwrap();
    }
    let found = store
        .metadata_by_filepath(&["qmd://c/a.md".to_owned(), "qmd://c/b.md".to_owned()])
        .unwrap();
    assert_eq!(
        serde_json::to_value(&found["qmd://c/a.md"]).unwrap(),
        json!({"status": "published"})
    );
    assert_eq!(
        serde_json::to_value(&found["qmd://c/b.md"]).unwrap(),
        json!({"status": "draft"})
    );
}

fn write(root: &Path, rel: &str, content: &str) {
    std::fs::write(root.join(rel), content).unwrap();
}

fn meta(status: &str) -> String {
    format!("---\nqmd:\n  metadata:\n    status: {status}\n---\n\n# Doc\n\nbody\n")
}

fn status_of(store: &Store, path: &str) -> Option<Value> {
    let found = store
        .metadata_by_filepath(&[format!("qmd://t/{path}")])
        .unwrap();
    found
        .get(&format!("qmd://t/{path}"))
        .map(|m| serde_json::to_value(m).unwrap())
}

fn failed_paths(store: &Store) -> Vec<String> {
    let failures = store.metadata_failures(None).unwrap();
    failures.into_iter().map(|f| f.path).collect()
}

#[test]
fn metadata_failures_are_the_current_failed_active_documents_of_the_scope() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let bad = "---\ntitle: a: b\n---\n\n# Bad\n";
    for name in ["b.md", "a.md", "stale.md", "gone.md", "fixed.md"] {
        write(root, name, bad);
    }
    write(root, "good.md", &meta("draft"));
    let store = Store::open_in_memory().unwrap();
    reindex_collection(&store, root, "**/*.md", "one", &[], None).unwrap();
    let other = tempfile::tempdir().unwrap();
    write(other.path(), "z.md", bad);
    reindex_collection(&store, other.path(), "**/*.md", "two", &[], None).unwrap();

    std::fs::remove_file(root.join("gone.md")).unwrap();
    write(root, "fixed.md", &meta("draft"));
    reindex_collection(&store, root, "**/*.md", "one", &[], None).unwrap();
    store
        .conn()
        .execute_batch(
            "UPDATE document_metadata SET extraction_version = 0 WHERE document_id =
               (SELECT id FROM documents WHERE path = 'stale.md')",
        )
        .unwrap();

    let all = store.metadata_failures(None).unwrap();
    let names: Vec<String> = all
        .iter()
        .map(|f| format!("{}/{}", f.collection, f.path))
        .collect();
    assert_eq!(names, ["one/a.md", "one/b.md", "two/z.md"]);
    assert!(all[0].error.starts_with("invalid frontmatter YAML"));
    assert_eq!(
        store.count_documents_pending_metadata(None).unwrap(),
        1,
        "stale.md: update will retry it"
    );
    assert_eq!(store.count_metadata_failures(None).unwrap(), 3);
    let two = ["two".to_owned()];
    assert_eq!(store.count_metadata_failures(Some(&two)).unwrap(), 1);
    let scoped = store.metadata_failures(Some(&two)).unwrap();
    assert_eq!(scoped.len(), 1);
    assert!(
        scoped[0]
            .to_string()
            .starts_with("two/z.md: invalid frontmatter YAML")
    );

    let retried = reindex_collection(&store, root, "**/*.md", "one", &[], None).unwrap();
    assert_eq!(retried.metadata_errors, 1, "stale.md, and no other");
    assert_eq!(store.count_documents_pending_metadata(None).unwrap(), 0);
    assert_eq!(store.count_metadata_failures(None).unwrap(), 4);
}

#[test]
fn update_extracts_new_changed_and_unchanged_files_and_backfills() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write(root, "a.md", &meta("draft"));
    write(root, "plain.md", "# Plain\n\nno frontmatter\n");
    write(
        root,
        "bad.md",
        "---\nqmd:\n  metadata:\n    bad: null\n---\n\n# Bad\n",
    );
    let store = Store::open_in_memory().unwrap();
    let first = reindex_collection(&store, root, "**/*.md", "t", &[], None).unwrap();
    assert_eq!(
        first.metadata_errors, 1,
        "a bad block is counted, not fatal"
    );
    assert_eq!(first.indexed, 3);
    assert_eq!(status_of(&store, "a.md"), Some(json!({"status": "draft"})));
    assert_eq!(status_of(&store, "plain.md"), Some(json!({})));
    assert_eq!(
        store.count_documents_pending_metadata(None).unwrap(),
        0,
        "bad.md failed, and update will not retry it"
    );
    assert_eq!(failed_paths(&store), ["bad.md"]);

    write(root, "a.md", &meta("published"));
    let second = reindex_collection(&store, root, "**/*.md", "t", &[], None).unwrap();
    assert_eq!((second.updated, second.metadata_errors), (1, 0));
    assert_eq!(
        status_of(&store, "a.md"),
        Some(json!({"status": "published"}))
    );

    let vectors_before: i64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM content", [], |r| r.get(0))
        .unwrap();
    store.conn().execute_batch("DELETE FROM document_metadata WHERE document_id IN (SELECT id FROM documents WHERE path = 'a.md');
         UPDATE document_metadata SET extraction_version = 0 WHERE document_id IN (SELECT id FROM documents WHERE path = 'plain.md');").unwrap();
    assert_eq!(store.count_documents_pending_metadata(None).unwrap(), 2);
    let third = reindex_collection(&store, root, "**/*.md", "t", &[], None).unwrap();
    assert_eq!((third.unchanged, third.updated, third.indexed), (3, 0, 0));
    assert_eq!(
        status_of(&store, "a.md"),
        Some(json!({"status": "published"}))
    );
    assert_eq!(third.metadata_errors, 0, "an unchanged file is not retried");
    assert_eq!(store.count_documents_pending_metadata(None).unwrap(), 0);
    assert_eq!(failed_paths(&store), ["bad.md"], "bad.md stays failed");
    let vectors_after: i64 = store
        .conn()
        .query_row("SELECT COUNT(*) FROM content", [], |r| r.get(0))
        .unwrap();
    assert_eq!(vectors_before, vectors_after);

    std::fs::rename(root.join("a.md"), root.join("a.tmp")).unwrap();
    reindex_collection(&store, root, "**/*.md", "t", &[], None).unwrap();
    assert_eq!(status_of(&store, "a.md"), None);
    std::fs::rename(root.join("a.tmp"), root.join("a.md")).unwrap();
    reindex_collection(&store, root, "**/*.md", "t", &[], None).unwrap();
    assert_eq!(
        status_of(&store, "a.md"),
        Some(json!({"status": "published"}))
    );
}

#[test]
fn deleting_documents_cascades_their_metadata() {
    let mut f = Fixture::new();
    let id = f.doc("c", json!({"status": "x"}));
    f.doc("gone", json!({"status": "y"}));
    let count = |f: &Fixture, table: &str| -> i64 {
        f.store
            .conn()
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    };
    f.sql("DELETE FROM documents WHERE id = ?1", id);
    assert_eq!(
        (
            count(&f, "document_metadata"),
            count(&f, "document_metadata_values")
        ),
        (1, 1)
    );
    f.store.remove_collection("gone").unwrap();
    assert_eq!(
        (
            count(&f, "document_metadata"),
            count(&f, "document_metadata_values")
        ),
        (0, 0)
    );
}

#[test]
fn counts_documents_not_values() {
    let mut f = Fixture::new();
    f.doc("notes", json!({"topics": ["a", "b"], "status": "draft"}));
    f.doc("notes", json!({"topics": ["a"], "status": "published"}));
    f.doc("notes", json!({"status": "published"}));
    let r = f.list(&ListMetadataOptions::default());
    assert_eq!((r.documents, r.filtered_documents), (3, None));
    assert_eq!(keys(&r), ["status", "topics"]);
    let topics = summary(&r, "topics");
    assert_eq!(topics.documents, 2);
    assert!(topics.types[0].multi_valued);
    assert_eq!(topics.types[0].distinct_values, 2);
    assert_eq!(
        values(&r, "topics"),
        pairs(&[(json!("a"), 2), (json!("b"), 1)])
    );
    assert!(!summary(&r, "status").types[0].multi_valued);
    assert_eq!(
        values(&r, "status"),
        pairs(&[(json!("published"), 2), (json!("draft"), 1)])
    );
}

#[test]
fn discovery_applies_the_gate_filtered_search_applies() {
    let mut f = Fixture::new();
    f.doc("notes", json!({}));
    let empty = f.list(&ListMetadataOptions::default());
    assert_eq!(
        (
            empty.documents,
            empty.total_keys,
            empty.keys.len(),
            empty.remaining_keys
        ),
        (1, 0, 0, 0)
    );

    let mut f = Fixture::new();
    f.doc("notes", json!({"status": "visible"}));
    let pending = f.doc("notes", json!({"status": "pending"}));
    let errored = f.doc("notes", json!({"status": "errored"}));
    let stale = f.doc("notes", json!({"status": "stale"}));
    let inactive = f.doc("notes", json!({"status": "inactive"}));
    f.sql(
        "DELETE FROM document_metadata WHERE document_id = ?1",
        pending,
    );
    f.sql(
        "UPDATE document_metadata SET extraction_error = 'boom' WHERE document_id = ?1",
        errored,
    );
    f.sql(
        "UPDATE document_metadata SET extraction_version = 0 WHERE document_id = ?1",
        stale,
    );
    f.sql("UPDATE documents SET active = 0 WHERE id = ?1", inactive);
    let r = f.list(&ListMetadataOptions::default());
    assert_eq!(r.documents, 4, "the denominator counts active documents");
    assert_eq!(values(&r, "status"), pairs(&[(json!("visible"), 1)]));
}

fn scoped() -> Fixture {
    let mut f = Fixture::new();
    f.doc("notes", json!({"status": "published", "priority": 3}));
    f.doc("notes", json!({"status": "draft", "priority": 1}));
    f.doc("work", json!({"status": "published", "priority": 5}));
    f.doc("work", json!({"status": "archived"}));
    f
}

#[test]
fn collections_scope_counts_and_the_denominator() {
    let mut f = scoped();
    let all = f.list(&ListMetadataOptions::default());
    assert_eq!((all.documents, summary(&all, "status").documents), (4, 4));
    assert_eq!(
        summary(&all, "status").types[0].collections,
        ["notes", "work"]
    );
    let notes = f.list(&in_collections(&["notes"]));
    assert_eq!(notes.documents, 2);
    assert_eq!(
        values(&notes, "status"),
        pairs(&[(json!("draft"), 1), (json!("published"), 1)])
    );
    assert_eq!(summary(&notes, "status").types[0].collections, ["notes"]);
    f.doc("other", json!({"status": "elsewhere"}));
    let both = f.list(&in_collections(&["notes", "work"]));
    assert_eq!(both.documents, 4);
    assert!(
        !values(&both, "status")
            .iter()
            .any(|(v, _)| v == "elsewhere")
    );
    let missing = f.list(&in_collections(&["missing"]));
    assert_eq!(
        (missing.documents, missing.total_keys, missing.keys.len()),
        (0, 0, 0)
    );
}

#[test]
fn a_filter_narrows_the_documents_counted() {
    let f = scoped();
    let r = f.list(&filtered(
        json!({"field": "status", "operator": "eq", "value": "published"}),
    ));
    assert_eq!((r.documents, r.filtered_documents), (4, Some(2)));
    assert_eq!(summary(&r, "priority").documents, 2);
    assert_eq!(
        values(&r, "priority"),
        pairs(&[(json!(3), 1), (json!(5), 1)])
    );
    assert_eq!(values(&r, "status"), pairs(&[(json!("published"), 2)]));
    let r = f.list(&filtered(json!({"operator": "and", "operands": [
        {"field": "status", "operator": "eq", "value": "published"},
        {"field": "priority", "operator": "gte", "value": 4},
    ]})));
    assert_eq!(r.filtered_documents, Some(1));
    assert_eq!(summary(&r, "status").types[0].collections, ["work"]);
}

fn matched() -> Fixture {
    let mut f = Fixture::new();
    f.doc("notes", json!({"topics": ["typescript", "sqlite"], "lead": "docs-team", "mem-kind": "fact", "priority": 3, "reviewed": true}));
    f.doc("notes", json!({"topics": ["typescript", "PostgreSQL"], "lead": "search-team", "reviewers": ["docs-team", "security-team"], "mem-scope": "user", "priority": 10, "reviewed": false}));
    f
}

fn keys_of(f: &Fixture, json: Value) -> Vec<String> {
    keys(&f.list(&matching(json)))
        .into_iter()
        .map(str::to_owned)
        .collect()
}

#[test]
fn a_match_on_the_key_field_selects_keys() {
    let f = matched();
    let r = f.list(&matching(
        json!({"field": "key", "operator": "eq", "value": "topics"}),
    ));
    assert_eq!(keys(&r), ["topics"]);
    assert_eq!(
        values(&r, "topics"),
        pairs(&[
            (json!("typescript"), 2),
            (json!("PostgreSQL"), 1),
            (json!("sqlite"), 1)
        ])
    );
    assert_eq!(
        keys_of(
            &f,
            json!({"field": "key", "operator": "prefix", "value": "mem-"})
        ),
        ["mem-kind", "mem-scope"]
    );
    assert_eq!(
        keys_of(
            &f,
            json!({"field": "key", "operator": "suffix", "value": "-scope"})
        ),
        ["mem-scope"]
    );
    assert_eq!(
        keys_of(
            &f,
            json!({"field": "key", "operator": "contains", "value": "review"})
        ),
        ["reviewed", "reviewers"]
    );
    assert_eq!(
        keys_of(
            &f,
            json!({"field": "key", "operator": "in", "value": ["tags", "topics", "labels"]})
        ),
        ["topics"]
    );
    assert_eq!(
        keys_of(
            &f,
            json!({"field": "key", "operator": "nin", "value": ["topics", "lead", "priority", "reviewed"]})
        ),
        ["mem-kind", "mem-scope", "reviewers"]
    );
    let none = f.list(&matching(
        json!({"field": "key", "operator": "prefix", "value": "missing-"}),
    ));
    assert!(none.keys.is_empty());
    assert_eq!(none.documents, 2);
    assert_eq!(
        keys_of(
            &f,
            json!({"field": "key", "operator": "eq", "value": "TOPICS", "caseInsensitive": true})
        ),
        ["topics"]
    );
    assert!(keys_of(&f, json!({"field": "key", "operator": "eq", "value": 3})).is_empty());
    assert!(keys_of(&f, json!({"field": "key", "operator": "gte", "value": 0})).is_empty());
    assert!(
        keys_of(
            &f,
            json!({"field": "key", "operator": "in", "value": [true]})
        )
        .is_empty()
    );
    let every: Vec<String> = keys(&f.list(&ListMetadataOptions::default()))
        .into_iter()
        .map(str::to_owned)
        .collect();
    assert_eq!(
        keys_of(
            &f,
            json!({"field": "key", "operator": "type", "value": "string"})
        ),
        every
    );
    assert!(
        keys_of(
            &f,
            json!({"field": "key", "operator": "type", "value": "number"})
        )
        .is_empty()
    );
}

#[test]
fn a_match_on_the_value_field_is_a_typed_reverse_lookup() {
    let mut f = matched();
    let r = f.list(&matching(
        json!({"field": "value", "operator": "eq", "value": "docs-team"}),
    ));
    assert_eq!(keys(&r), ["lead", "reviewers"]);
    assert_eq!(
        (
            summary(&r, "lead").documents,
            summary(&r, "lead").types[0].distinct_values
        ),
        (1, 1)
    );
    assert_eq!(values(&r, "reviewers"), pairs(&[(json!("docs-team"), 1)]));

    assert_eq!(
        keys_of(&f, json!({"field": "value", "operator": "eq", "value": 10})),
        ["priority"]
    );
    assert!(
        keys_of(
            &f,
            json!({"field": "value", "operator": "eq", "value": "10"})
        )
        .is_empty()
    );
    assert_eq!(
        keys_of(
            &f,
            json!({"field": "value", "operator": "eq", "value": true})
        ),
        ["reviewed"]
    );
    assert!(
        keys_of(
            &f,
            json!({"field": "value", "operator": "contains", "value": "true"})
        )
        .is_empty()
    );
    assert!(
        keys_of(
            &f,
            json!({"field": "value", "operator": "prefix", "value": "1"})
        )
        .is_empty()
    );
    assert_eq!(
        keys_of(
            &f,
            json!({"field": "value", "operator": "type", "value": "boolean"})
        ),
        ["reviewed"]
    );
    assert_eq!(
        keys_of(
            &f,
            json!({"field": "value", "operator": "type", "value": "number"})
        ),
        ["priority"]
    );
    assert_eq!(
        keys_of(
            &f,
            json!({"operator": "not", "operand": {"field": "value", "operator": "type", "value": "string"}})
        ),
        ["priority", "reviewed"]
    );

    let r = f.list(&matching(json!({"operator": "and", "operands": [
        {"field": "key", "operator": "eq", "value": "topics"},
        {"field": "value", "operator": "prefix", "value": "type"},
    ]})));
    assert_eq!(values(&r, "topics"), pairs(&[(json!("typescript"), 2)]));
    assert_eq!(summary(&r, "topics").types[0].remaining_values, 0);
    assert!(
        summary(&r, "topics").types[0].multi_valued,
        "array-ness describes the key"
    );

    let r = f.list(&matching(json!({"operator": "and", "operands": [
        {"field": "key", "operator": "eq", "value": "priority"},
        {"field": "value", "operator": "gte", "value": 5},
    ]})));
    assert_eq!(values(&r, "priority"), pairs(&[(json!(10), 1)]));
    let range = summary(&r, "priority").types[0].range.unwrap();
    assert_eq!((range.min, range.median, range.max), (10.0, 10.0, 10.0));

    let r = f.list(&matching(json!({"operator": "and", "operands": [
        {"field": "key", "operator": "eq", "value": "topics"},
        {"operator": "not", "operand": {"field": "value", "operator": "in", "value": ["typescript"]}},
    ]})));
    assert_eq!(
        values(&r, "topics"),
        pairs(&[(json!("PostgreSQL"), 1), (json!("sqlite"), 1)])
    );

    let r = f.list(&matching(json!({"operator": "or", "operands": [
        {"field": "key", "operator": "eq", "value": "lead"},
        {"field": "value", "operator": "eq", "value": "docs-team"},
    ]})));
    assert_eq!(keys(&r), ["lead", "reviewers"]);
    assert_eq!(
        values(&r, "lead"),
        pairs(&[(json!("docs-team"), 1), (json!("search-team"), 1)])
    );

    let sql = |ci: bool| {
        let r = f.list(&matching(json!({"field": "value", "operator": "contains", "value": "sql", "caseInsensitive": ci})));
        values(&r, "topics")
    };
    assert_eq!(sql(false), pairs(&[(json!("sqlite"), 1)]));
    assert_eq!(
        sql(true),
        pairs(&[(json!("PostgreSQL"), 1), (json!("sqlite"), 1)])
    );

    let r = f.list(&ListMetadataOptions {
        r#match: Some(
            MetadataMatch::parse(&json!({"field": "key", "operator": "eq", "value": "topics"}))
                .unwrap(),
        ),
        filter: Some(
            MetadataFilter::parse(&json!({"field": "reviewed", "operator": "eq", "value": true}))
                .unwrap(),
        ),
        ..Default::default()
    });
    assert_eq!(r.filtered_documents, Some(1));
    assert_eq!(
        values(&r, "topics"),
        pairs(&[(json!("sqlite"), 1), (json!("typescript"), 1)])
    );

    f.doc("work", json!({"priority": "high"}));
    let r = f.list(&matching(json!({"operator": "and", "operands": [
        {"field": "key", "operator": "eq", "value": "priority"},
        {"field": "value", "operator": "type", "value": "number"},
    ]})));
    let priority = summary(&r, "priority");
    assert_eq!(
        priority
            .types
            .iter()
            .map(|t| t.value_type)
            .collect::<Vec<_>>(),
        [ValueType::Number]
    );
    assert_eq!(
        (priority.documents, priority.types[0].collections.clone()),
        (2, vec!["notes".to_owned()])
    );
}

#[test]
fn entry_text_predicates_and_their_negations_partition_the_rows() {
    let mut f = Fixture::new();
    let labels = [
        json!(""),
        json!("draft"),
        json!("PUBLISHED"),
        json!("abc\u{0}XYZ"),
        json!("\u{0}"),
        json!("\u{1F600}XYZ"),
        json!(7),
        json!(true),
    ];
    for label in &labels {
        f.doc("notes", json!({"label": label}));
    }
    for op in ["contains", "prefix", "suffix"] {
        for operand in ["ft", "PUB", "XYZ", "\u{0}", "\u{1F600}", "nomatch"] {
            for ci in [false, true] {
                let fold = |s: &str| {
                    if ci {
                        s.to_ascii_lowercase()
                    } else {
                        s.to_owned()
                    }
                };
                let expected: Vec<bool> = labels
                    .iter()
                    .map(|v| {
                        v.as_str().is_some_and(|s| {
                            let (s, n) = (fold(s), fold(operand));
                            match op {
                                "contains" => s.contains(&n),
                                "prefix" => s.starts_with(&n),
                                _ => s.ends_with(&n),
                            }
                        })
                    })
                    .collect();
                let condition = json!({"field": "value", "operator": op, "value": operand, "caseInsensitive": ci});
                let negated = json!({"operator": "not", "operand": condition});
                for (predicate, want) in [
                    (condition.clone(), expected.clone()),
                    (negated.clone(), expected.iter().map(|b| !b).collect()),
                    (
                        json!({"operator": "or", "operands": [condition.clone(), negated]}),
                        vec![true; labels.len()],
                    ),
                ] {
                    let compiled = MetadataMatch::parse(&predicate).unwrap().compile("mv");
                    let sql = format!(
                        "SELECT {} FROM document_metadata_values mv ORDER BY document_id",
                        compiled.sql
                    );
                    let mut stmt = f.store.conn().prepare(&sql).unwrap();
                    let got: Vec<bool> = stmt
                        .query_map(rusqlite::params_from_iter(compiled.params), |r| {
                            r.get::<_, i64>(0)
                        })
                        .unwrap()
                        .map(|r| r.unwrap() == 1)
                        .collect();
                    assert_eq!(got, want, "{predicate}");
                    let report = f.list(&ListMetadataOptions {
                        value_limit: Some(Limit::Unbounded),
                        ..matching(predicate)
                    });
                    let documents = report.keys.first().map_or(0, |k| k.documents);
                    assert_eq!(documents, want.iter().filter(|b| **b).count() as u64);
                }
                let not_label = MetadataFilter::parse(&json!({"operator": "not", "operand":
                    {"field": "label", "operator": op, "value": operand, "caseInsensitive": ci}}))
                .unwrap();
                for scope in [
                    qmd::metadata::filter::FilterScope::Candidates,
                    qmd::metadata::filter::FilterScope::Corpus,
                ] {
                    let compiled = not_label.compile("d", scope);
                    let sql = format!("SELECT {} FROM documents d ORDER BY id", compiled.sql);
                    let mut stmt = f.store.conn().prepare(&sql).unwrap();
                    let got: Vec<bool> = stmt
                        .query_map(rusqlite::params_from_iter(compiled.params), |r| {
                            r.get::<_, i64>(0)
                        })
                        .unwrap()
                        .map(|r| r.unwrap() == 1)
                        .collect();
                    assert_eq!(got, expected.iter().map(|b| !b).collect::<Vec<_>>());
                }
            }
        }
    }
}

fn five_keys() -> Fixture {
    let mut f = Fixture::new();
    for (index, key) in ["a", "b", "c", "d", "e"].iter().enumerate() {
        for _ in 0..=index {
            f.doc("notes", json!({*key: "x"}));
        }
    }
    f
}

fn key_window(limit: Option<Limit>, offset: Option<u64>) -> ListMetadataOptions {
    ListMetadataOptions {
        key_limit: limit,
        key_offset: offset,
        ..Default::default()
    }
}

#[test]
fn the_key_window_pages_in_report_order() {
    let mut f = five_keys();
    let all = f.list(&ListMetadataOptions::default());
    assert_eq!(
        (keys(&all), all.total_keys, all.remaining_keys),
        (vec!["e", "d", "c", "b", "a"], 5, 0)
    );
    let pages: Vec<(Vec<String>, u64)> = [0, 2, 4]
        .into_iter()
        .map(|offset| {
            let r = f.list(&key_window(Some(Limit::At(2)), Some(offset)));
            assert_eq!(r.total_keys, 5);
            (
                keys(&r).into_iter().map(str::to_owned).collect(),
                r.remaining_keys,
            )
        })
        .collect();
    assert_eq!(
        pages,
        [
            (vec!["e".to_owned(), "d".to_owned()], 3),
            (vec!["c".to_owned(), "b".to_owned()], 1),
            (vec!["a".to_owned()], 0)
        ]
    );
    let past = f.list(&key_window(None, Some(10)));
    assert_eq!(
        (past.keys.len(), past.total_keys, past.remaining_keys),
        (0, 5, 0)
    );
    assert_eq!(
        keys(&f.list(&key_window(Some(Limit::Unbounded), None))).len(),
        5
    );
    let r = f.list(&ListMetadataOptions {
        key_limit: Some(Limit::At(2)),
        ..matching(json!({"field": "key", "operator": "in", "value": ["a", "c", "e"]}))
    });
    assert_eq!(
        (keys(&r), r.total_keys, r.remaining_keys),
        (vec!["e", "c"], 3, 1)
    );

    f.doc("work", json!({"e": ["y", "z"], "d": 1}));
    let r = f.list(&key_window(Some(Limit::At(1)), None));
    let e = summary(&r, "e");
    assert_eq!(
        (
            e.documents,
            e.types[0].multi_valued,
            e.types[0].distinct_values
        ),
        (6, true, 3)
    );
    assert_eq!(e.types[0].collections, ["notes", "work"]);
    assert_eq!(r.keys.len(), 1);
}

#[test]
fn keys_order_by_coverage_then_bytes_and_collections_by_bytes() {
    let mut f = Fixture::new();
    f.doc(
        "work",
        json!({"_id": 1, "a": 1, "B": 1, "\u{e9}": 1, "Z": 1}),
    );
    let names = ["B", "Z", "_id", "a", "\u{e9}"];
    assert_eq!(keys(&f.list(&ListMetadataOptions::default())), names);
    let paged: Vec<String> = [0, 2, 4]
        .into_iter()
        .flat_map(|o| {
            keys(&f.list(&key_window(Some(Limit::At(2)), Some(o))))
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(paged, names);
    let overview = |limit| {
        f.store.metadata_overviews(limit).unwrap()["work"]
            .keys
            .iter()
            .map(|k| k.key.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(overview(None), names);
    assert_eq!(overview(Some(3)), names[..3]);

    for collection in ["\u{10000}", "\u{e000}", "notes", "B"] {
        f.doc(collection, json!({"provenance": "shared"}));
    }
    let r = f.list(&matching(
        json!({"field": "key", "operator": "eq", "value": "provenance"}),
    ));
    assert_eq!(
        summary(&r, "provenance").types[0].collections,
        ["B", "notes", "\u{e000}", "\u{10000}"]
    );

    let mut f = Fixture::new();
    f.doc("notes", json!({"zeta": 1, "alpha": 1, "mid": 1}));
    f.doc("notes", json!({"zeta": 1, "alpha": 1}));
    f.doc("notes", json!({"zeta": 1}));
    assert_eq!(
        keys(&f.list(&ListMetadataOptions::default())),
        ["zeta", "alpha", "mid"]
    );
}

fn tags() -> Fixture {
    let mut f = Fixture::new();
    for tag in ["a", "a", "a", "b", "b", "c", "d", "d", "d", "d"] {
        f.doc("notes", json!({"tag": tag}));
    }
    f
}

fn value_window(
    limit: Option<Limit>,
    offset: Option<u64>,
    sort: ValueSort,
    min_count: Option<u64>,
) -> ListMetadataOptions {
    ListMetadataOptions {
        value_limit: limit,
        value_offset: offset,
        sort,
        min_count,
        ..Default::default()
    }
}

#[test]
fn the_value_window_sorts_pages_and_trims() {
    let f = tags();
    let tag = |o: &ListMetadataOptions| f.list(o).keys[0].types[0].clone();
    let names = |t: &qmd::metadata::report::MetadataKeyTypeSummary| {
        t.values
            .iter()
            .map(|v| v.value.to_json())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        values(&f.list(&ListMetadataOptions::default()), "tag"),
        pairs(&[
            (json!("d"), 4),
            (json!("a"), 3),
            (json!("b"), 2),
            (json!("c"), 1)
        ])
    );
    assert_eq!(
        values(
            &f.list(&value_window(None, None, ValueSort::Value, None)),
            "tag"
        ),
        pairs(&[
            (json!("a"), 3),
            (json!("b"), 2),
            (json!("c"), 1),
            (json!("d"), 4)
        ])
    );
    let limited = tag(&value_window(
        Some(Limit::At(2)),
        None,
        ValueSort::Count,
        None,
    ));
    assert_eq!(
        (
            names(&limited),
            limited.distinct_values,
            limited.remaining_values
        ),
        (vec![json!("d"), json!("a")], 4, 2)
    );
    let all = tag(&value_window(
        Some(Limit::Unbounded),
        None,
        ValueSort::Count,
        None,
    ));
    assert_eq!((all.values.len(), all.remaining_values), (4, 0));
    let pages: Vec<_> = [0, 2, 4]
        .into_iter()
        .map(|o| {
            tag(&value_window(
                Some(Limit::At(2)),
                Some(o),
                ValueSort::Count,
                None,
            ))
        })
        .collect();
    assert_eq!(
        pages.iter().map(names).collect::<Vec<_>>(),
        [
            vec![json!("d"), json!("a")],
            vec![json!("b"), json!("c")],
            vec![]
        ]
    );
    assert_eq!(
        pages.iter().map(|t| t.remaining_values).collect::<Vec<_>>(),
        [2, 0, 0]
    );
    assert!(pages.iter().all(|t| t.distinct_values == 4));
    assert_eq!(
        names(&tag(&value_window(
            Some(Limit::At(2)),
            Some(1),
            ValueSort::Value,
            None
        ))),
        [json!("b"), json!("c")]
    );
    let common = tag(&value_window(
        Some(Limit::At(1)),
        None,
        ValueSort::Count,
        Some(3),
    ));
    assert_eq!(
        (
            names(&common),
            common.distinct_values,
            common.remaining_values,
            common.documents
        ),
        (vec![json!("d")], 2, 1, 10)
    );
    let none = tag(&value_window(None, None, ValueSort::Count, Some(99)));
    assert_eq!(
        (
            none.values.len(),
            none.distinct_values,
            none.remaining_values
        ),
        (0, 0, 0)
    );
}

#[test]
fn numbers_report_min_median_max_over_every_value_row() {
    let range_of = |entries: &[Value]| {
        let mut f = Fixture::new();
        for e in entries {
            f.doc("notes", json!({"n": e}));
        }
        let r = f.list(&ListMetadataOptions::default());
        let t = &summary(&r, "n").types[0];
        let range = t.range.unwrap();
        (
            (range.min, range.median, range.max),
            t.values
                .iter()
                .map(|v| (v.value.to_json(), v.documents))
                .collect::<Vec<_>>(),
        )
    };
    let (range, window) = range_of(&[json!(5), json!(1), json!(3)]);
    assert_eq!(range, (1.0, 3.0, 5.0));
    assert_eq!(
        window,
        pairs(&[(json!(1), 1), (json!(3), 1), (json!(5), 1)])
    );
    assert_eq!(
        range_of(&[json!(1), json!(2), json!(3), json!(10)]).0,
        (1.0, 2.5, 10.0)
    );
    let (range, window) = range_of(&[json!([1, 1, 1]), json!([9])]);
    assert_eq!(
        range,
        (1.0, 1.0, 9.0),
        "array elements each count; repeats were de-duplicated at extraction"
    );
    assert_eq!(window, pairs(&[(json!(1), 1), (json!(9), 1)]));

    let tiny = f64::from_bits(1);
    for (lower, upper, median) in [
        (1e308, 1.5e308, 1.25e308),
        (-1.5e308, -1e308, -1.25e308),
        (-1e308, 1e308, 0.0),
        (tiny, 2.0 * tiny, 2.0 * tiny),
        (-2.0 * tiny, -tiny, -2.0 * tiny),
        (f64::MAX, f64::MAX, f64::MAX),
    ] {
        let mut f = Fixture::new();
        f.doc("c", json!({"n": lower}));
        f.doc("c", json!({"n": upper}));
        let r = f.list(&ListMetadataOptions::default());
        let range = summary(&r, "n").types[0].range.unwrap();
        assert_eq!((range.min, range.median, range.max), (lower, median, upper));
    }

    let mut f = Fixture::new();
    f.doc("notes", json!({"status": "x", "reviewed": true}));
    let r = f.list(&ListMetadataOptions::default());
    assert!(
        summary(&r, "status").types[0].range.is_none()
            && summary(&r, "reviewed").types[0].range.is_none()
    );
}

#[test]
fn a_key_whose_documents_disagree_splits_by_type() {
    let mut f = Fixture::new();
    f.doc("notes", json!({"priority": 3}));
    f.doc("notes", json!({"priority": 1}));
    f.doc("work", json!({"priority": "high"}));
    let r = f.list(&ListMetadataOptions::default());
    let p = summary(&r, "priority");
    assert_eq!(p.documents, 3);
    assert_eq!(
        p.types
            .iter()
            .map(|t| (t.value_type, t.documents))
            .collect::<Vec<_>>(),
        [(ValueType::Number, 2), (ValueType::String, 1)]
    );
    let range = p.types[0].range.unwrap();
    assert_eq!((range.min, range.median, range.max), (1.0, 2.0, 3.0));
    assert_eq!(
        (
            p.types[0].collections.clone(),
            p.types[1].collections.clone()
        ),
        (vec!["notes".to_owned()], vec!["work".to_owned()])
    );
    assert_eq!(p.types[1].values[0].value, Scalar::String("high".into()));

    let mut f = Fixture::new();
    f.doc("notes", json!({"flag": true}));
    f.doc("notes", json!({"flag": "yes"}));
    let r = f.list(&ListMetadataOptions::default());
    assert_eq!(
        summary(&r, "flag")
            .types
            .iter()
            .map(|t| t.value_type)
            .collect::<Vec<_>>(),
        [ValueType::Boolean, ValueType::String]
    );
}

#[test]
fn options_outside_their_domain_and_over_the_binding_budget_are_refused() {
    let mut f = Fixture::new();
    f.doc("notes", json!({"status": "x"}));
    let error = |o: ListMetadataOptions| f.store.list_metadata(&o).unwrap_err().to_string();
    assert_eq!(
        error(key_window(Some(Limit::At(0)), None)),
        "Invalid keyLimit: expected a positive integer or Infinity, received 0"
    );
    assert_eq!(
        error(key_window(None, Some(1 << 53))),
        "Invalid keyOffset: expected a non-negative integer, received 9007199254740992"
    );
    assert_eq!(
        error(value_window(None, None, ValueSort::Count, Some(0))),
        "Invalid minCount: expected a positive integer, received 0"
    );
    assert!(
        error(value_window(
            Some(Limit::At(1 << 60)),
            None,
            ValueSort::Count,
            None
        ))
        .starts_with("Invalid valueLimit")
    );
    let largest = qmd::metadata::report::MAX_SAFE_INTEGER;
    assert!(
        f.list(&key_window(Some(Limit::At(largest)), Some(largest)))
            .keys
            .is_empty()
    );
    let far = f.list(&value_window(
        Some(Limit::At(largest)),
        Some(largest),
        ValueSort::Count,
        None,
    ));
    assert!(far.keys[0].types[0].values.is_empty());
    let near = f.list(&value_window(
        Some(Limit::At(largest)),
        None,
        ValueSort::Count,
        None,
    ));
    assert_eq!(near.keys[0].types[0].values.len(), 1);

    let members: Vec<String> = (0..64).map(|i| format!("v{i}")).collect();
    let group = |cond: Value| json!({"operator": "and", "operands": vec![cond; 31]});
    let wide_filter = json!({"operator": "and", "operands": vec![group(json!({"field": "status", "operator": "all", "value": members})); 7]});
    let wide_match = json!({"operator": "or", "operands": vec![json!({"operator": "or", "operands": vec![json!({"field": "value", "operator": "in", "value": members}); 31]}); 7]});
    assert_eq!(f.list(&filtered(wide_filter.clone())).total_keys, 0);
    assert_eq!(f.list(&matching(wide_match.clone())).total_keys, 0);
    let both = ListMetadataOptions {
        filter: Some(MetadataFilter::parse(&wide_filter).unwrap()),
        r#match: Some(MetadataMatch::parse(&wide_match).unwrap()),
        ..Default::default()
    };
    match f.store.list_metadata(&both) {
        Err(qmd::Error::Metadata(MetadataError::BindingBudget { bindings, budget })) => {
            assert_eq!(budget, SQL_BINDING_BUDGET);
            assert!(bindings > budget);
        }
        other => panic!("expected the binding budget, got {other:?}"),
    }
}

#[test]
fn a_report_reads_one_snapshot_while_another_connection_writes() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("index.sqlite");
    let mut f = Fixture {
        store: Store::open(&db).unwrap(),
        next: 0,
    };
    let id = f.doc("notes", json!({"priority": 1}));
    let writer = Store::open(&db).unwrap();
    f.store.conn().execute_batch("BEGIN").unwrap();
    let before = f.list(&ListMetadataOptions::default());
    writer
        .replace_document_metadata(id, &extraction(json!({"priority": 100})))
        .unwrap();
    let during = f.list(&ListMetadataOptions::default());
    f.store.conn().execute_batch("COMMIT").unwrap();
    assert_eq!(before, during);
    let after = f.list(&ListMetadataOptions::default());
    assert_eq!(
        after.keys[0].types[0].values[0].value,
        Scalar::Number(100.0)
    );
}

#[test]
fn overviews_agree_with_full_reports_through_lifecycle_changes() {
    let mut f = Fixture::new();
    f.doc("notes", json!({"tags": ["a", "b", "c"], "priority": 1}));
    let changed = f.doc("notes", json!({"tags": 7, "priority": "high"}));
    f.doc(
        "work",
        json!({"tags": true, "\u{e9}": "x", "\u{1F600}": "x"}),
    );
    let stale = f.doc("work", json!({"old": "hidden"}));
    let inactive = f.doc("work", json!({"deleted": "hidden"}));
    f.sql(
        "UPDATE document_metadata SET extraction_version = 0 WHERE document_id = ?1",
        stale,
    );
    f.sql("UPDATE documents SET active = 0 WHERE id = ?1", inactive);
    for phase in ["initial", "replaced", "renamed", "deleted"] {
        match phase {
            "replaced" => f
                .store
                .replace_document_metadata(changed, &extraction(json!({"tags": [true, false]})))
                .unwrap(),
            "renamed" => {
                f.store
                    .conn()
                    .execute(
                        "UPDATE documents SET collection = 'renamed' WHERE collection = 'notes'",
                        [],
                    )
                    .unwrap();
            }
            "deleted" => f.sql("DELETE FROM documents WHERE id = ?1", changed),
            _ => {}
        }
        for limit in [Some(1), Some(10), None] {
            let overviews = f.store.metadata_overviews(limit).unwrap();
            for collection in ["notes", "renamed", "work", "missing"] {
                let report = f.list(&ListMetadataOptions {
                    collections: Some(vec![collection.to_owned()]),
                    key_limit: Some(limit.map_or(Limit::Unbounded, Limit::At)),
                    ..Default::default()
                });
                let got = overviews.get(collection).cloned().unwrap_or_default();
                assert_eq!(got.total_keys, report.total_keys, "{phase} {collection}");
                let expected: Vec<(String, u64, Vec<ValueType>)> = report
                    .keys
                    .iter()
                    .map(|k| {
                        (
                            k.key.clone(),
                            k.documents,
                            k.types.iter().map(|t| t.value_type).collect(),
                        )
                    })
                    .collect();
                let actual: Vec<(String, u64, Vec<ValueType>)> = got
                    .keys
                    .iter()
                    .map(|k| (k.key.clone(), k.documents, k.types.clone()))
                    .collect();
                assert_eq!(actual, expected, "{phase} {collection} {limit:?}");
            }
        }
    }
}

#[test]
fn status_helpers_count_keys_and_documents_in_scope() {
    let mut f = Fixture::new();
    f.doc("notes", json!({"status": "published", "priority": 3}));
    f.doc("notes", json!({"status": "draft"}));
    f.doc("notes", json!({}));
    f.doc("work", json!({"priority": "high", "source": "jira"}));
    let pending = f.doc("work", json!({"status": "ghost"}));
    f.sql(
        "DELETE FROM document_metadata WHERE document_id = ?1",
        pending,
    );
    let overviews = f.store.metadata_overviews(None).unwrap();
    let flat = |c: &str| -> (u64, Vec<(String, u64, Vec<ValueType>)>) {
        let o = &overviews[c];
        (
            o.total_keys,
            o.keys
                .iter()
                .map(|k| (k.key.clone(), k.documents, k.types.clone()))
                .collect(),
        )
    };
    assert_eq!(
        flat("notes"),
        (
            2,
            vec![
                ("status".into(), 2, vec![ValueType::String]),
                ("priority".into(), 1, vec![ValueType::Number])
            ]
        )
    );
    assert_eq!(
        flat("work"),
        (
            2,
            vec![
                ("priority".into(), 1, vec![ValueType::String]),
                ("source".into(), 1, vec![ValueType::String])
            ]
        )
    );
    assert!(!overviews.contains_key("missing"));
    assert_eq!(
        f.store.metadata_overviews(Some(1)).unwrap()["notes"]
            .keys
            .len(),
        1
    );
    let work = ["work".to_owned()];
    assert_eq!(f.store.count_metadata_keys(None).unwrap(), 3);
    assert_eq!(f.store.count_metadata_keys(Some(&work)).unwrap(), 2);
    assert_eq!(
        f.store
            .count_metadata_keys(Some(&["missing".to_owned()]))
            .unwrap(),
        0
    );
    assert_eq!(f.store.count_documents_with_metadata(None).unwrap(), 3);
    assert_eq!(
        f.store
            .count_documents_with_metadata(Some(&["notes".to_owned()]))
            .unwrap(),
        2
    );
    assert_eq!(f.store.count_documents_pending_metadata(None).unwrap(), 1);
    assert_eq!(
        f.store
            .count_documents_pending_metadata(Some(&["notes".to_owned()]))
            .unwrap(),
        0
    );
    assert_eq!(
        f.store
            .count_documents_pending_metadata(Some(&work))
            .unwrap(),
        1
    );
}

#[test]
fn every_reported_value_is_reachable_through_an_eq_filter() {
    let mut f = Fixture::new();
    f.doc(
        "notes",
        json!({"topics": ["a", "b"], "priority": 3, "reviewed": true, "lead": "docs-team"}),
    );
    f.doc(
        "notes",
        json!({"topics": ["b"], "priority": 1.5, "reviewed": false}),
    );
    f.doc("work", json!({"topics": ["c"], "priority": 3}));
    let pending = f.doc("notes", json!({"topics": ["ghost"]}));
    f.sql(
        "DELETE FROM document_metadata WHERE document_id = ?1",
        pending,
    );
    for scope in [
        None,
        Some(vec!["notes".to_owned()]),
        Some(vec!["notes".to_owned(), "work".to_owned()]),
    ] {
        let r = f.list(&ListMetadataOptions {
            collections: scope.clone(),
            value_limit: Some(Limit::Unbounded),
            ..Default::default()
        });
        assert!(!r.keys.is_empty());
        for s in &r.keys {
            for t in &s.types {
                for v in &t.values {
                    let eq = MetadataFilter::parse(
                        &json!({"field": s.key, "operator": "eq", "value": v.value.to_json()}),
                    )
                    .unwrap();
                    let mut hits = 0;
                    for c in scope
                        .clone()
                        .unwrap_or_else(|| vec!["notes".to_owned(), "work".to_owned()])
                    {
                        hits += f
                            .store
                            .search_fts_filtered("doc", 100, Some(&c), TermMatch::Any, Some(&eq))
                            .unwrap()
                            .len();
                    }
                    assert_eq!(hits as u64, v.documents, "{} = {:?}", s.key, v.value);
                }
            }
        }
    }
}
