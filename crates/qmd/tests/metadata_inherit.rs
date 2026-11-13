//! Metadata inherited from the global, collection and path-prefix levels:
//! filtered search admits exactly the documents under a level, a change to a
//! level reaches them on the next reindex with no file touched, and results
//! name where each inherited key came from.

use std::path::Path;

use qmd::Store;
use qmd::config::{Collection, CollectionConfig};
use qmd::index::reindex_collection;
use qmd::metadata::MetadataFilter;
use qmd::store::TermMatch;
use serde_json::{Value, json};

fn write(root: &Path, rel: &str, content: &str) {
    let p = root.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, content).unwrap();
}

struct Fixture {
    _dir: tempfile::TempDir,
    root: std::path::PathBuf,
    store: Store,
    config: CollectionConfig,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        write(&root, "top.md", "# Top\n\nkeyword\n");
        write(&root, "docs/a.md", "# A\n\nkeyword\n");
        write(
            &root,
            "docs/api/b.md",
            "---\nqmd:\n  metadata:\n    status: final\n---\n\n# B\n\nkeyword\n",
        );
        write(&root, "docs-old/c.md", "# C\n\nkeyword\n");
        let mut config = CollectionConfig::default();
        config.collections.insert(
            "n".to_owned(),
            Collection::new(root.to_string_lossy(), "**/*.md"),
        );
        Self {
            _dir: dir,
            root,
            store: Store::open_in_memory().unwrap(),
            config,
        }
    }

    fn set(&mut self, collection: Option<&str>, prefix: &str, key: &str, value: Value) {
        assert!(self.config.set_metadata(collection, prefix, key, value));
    }

    fn update(&self) -> qmd::index::ReindexResult {
        self.store.sync_config_to_db(&self.config).unwrap();
        reindex_collection(&self.store, &self.root, "**/*.md", "n", &[], None).unwrap()
    }

    fn matching(&self, filter: Value) -> Vec<String> {
        let filter = MetadataFilter::parse(&filter).unwrap();
        let mut paths: Vec<String> = self
            .store
            .search_fts_filtered("keyword", 50, None, TermMatch::Any, Some(&filter))
            .unwrap()
            .into_iter()
            .map(|r| r.display_path)
            .collect();
        paths.sort();
        paths
    }

    fn resolved(&self, path: &str) -> (Value, Value) {
        let key = format!("qmd://n/{path}");
        let found = self
            .store
            .metadata_by_filepath(std::slice::from_ref(&key))
            .unwrap();
        let r = &found[&key];
        (
            serde_json::to_value(&r.metadata).unwrap(),
            serde_json::to_value(&r.sources).unwrap(),
        )
    }
}

fn eq(field: &str, value: &str) -> Value {
    json!({"field": field, "operator": "eq", "value": value})
}

#[test]
fn a_filter_on_an_inherited_key_admits_exactly_the_documents_under_it() {
    let mut f = Fixture::new();
    f.set(None, "/", "team", json!("core"));
    f.set(Some("n"), "/", "status", json!("draft"));
    f.set(Some("n"), "docs", "area", json!("docs"));
    f.set(Some("n"), "/docs/api", "status", json!("api"));
    f.update();

    assert_eq!(
        f.matching(eq("team", "core")),
        [
            "n/docs-old/c.md",
            "n/docs/a.md",
            "n/docs/api/b.md",
            "n/top.md"
        ]
    );
    assert_eq!(
        f.matching(eq("area", "docs")),
        ["n/docs/a.md", "n/docs/api/b.md"]
    );
    assert_eq!(f.matching(eq("status", "final")), ["n/docs/api/b.md"]);
    assert_eq!(
        f.matching(eq("status", "draft")),
        ["n/docs-old/c.md", "n/docs/a.md", "n/top.md"]
    );
    assert!(f.matching(eq("status", "api")).is_empty());

    let (metadata, sources) = f.resolved("docs/api/b.md");
    assert_eq!(
        metadata,
        json!({"status": "final", "team": "core", "area": "docs"})
    );
    assert_eq!(
        sources,
        json!({"team": "global", "area": "qmd://n/docs/"}),
        "status is the file's own, so it has no source"
    );
}

#[test]
fn a_changed_level_reaches_its_documents_with_no_file_touched() {
    let mut f = Fixture::new();
    f.update();
    let unchanged = f.update();
    assert_eq!(unchanged.unchanged, 4);
    assert!(f.matching(eq("area", "docs")).is_empty());

    f.set(Some("n"), "/docs", "area", json!("docs"));
    f.update();
    assert_eq!(
        f.matching(eq("area", "docs")),
        ["n/docs/a.md", "n/docs/api/b.md"]
    );

    f.set(Some("n"), "/docs", "area", json!("manuals"));
    f.update();
    assert!(f.matching(eq("area", "docs")).is_empty());
    assert_eq!(f.matching(eq("area", "manuals")).len(), 2);

    assert_eq!(f.config.remove_metadata(Some("n"), "/docs", &[]), 1);
    assert!(f.config.collections["n"].metadata.is_none());
    f.update();
    assert!(f.matching(eq("area", "manuals")).is_empty());
    assert_eq!(f.resolved("docs/a.md"), (json!({}), json!({})));
}

#[test]
fn an_index_without_inherited_metadata_resolves_as_before() {
    let f = Fixture::new();
    f.update();
    let hashes: i64 = f
        .store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM document_metadata WHERE inherited_hash IS NOT NULL
               OR sources_json IS NOT NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(hashes, 0);
    assert_eq!(f.matching(eq("status", "final")), ["n/docs/api/b.md"]);
}

#[test]
fn an_invalid_level_fails_the_update_naming_it() {
    let mut f = Fixture::new();
    f.set(Some("n"), "/docs", "bad", json!(null));
    f.store.sync_config_to_db(&f.config).unwrap();
    let err = reindex_collection(&f.store, &f.root, "**/*.md", "n", &[], None).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("qmd://n/docs/"), "{message}");
    assert!(message.contains("null is not supported"), "{message}");
}

#[test]
fn discovery_counts_inherited_documents_per_value() {
    let mut f = Fixture::new();
    f.set(Some("n"), "/", "status", json!("draft"));
    f.update();
    let report = f
        .store
        .list_metadata(&qmd::metadata::ListMetadataOptions::default())
        .unwrap();
    let status = report.keys.iter().find(|k| k.key == "status").unwrap();
    let values = serde_json::to_value(&status.types[0].values).unwrap();
    assert_eq!(
        values,
        json!([
            {"value": "draft", "documents": 3, "inherited": [{"source": "qmd://n/", "documents": 3}]},
            {"value": "final", "documents": 1},
        ])
    );
}

#[test]
fn a_renamed_collection_names_its_new_address_as_the_source() {
    let mut f = Fixture::new();
    f.set(Some("n"), "/docs", "status", json!("draft"));
    f.set(None, "/", "team", json!("core"));
    f.update();
    f.store.rename_collection("n", "m").unwrap();
    let key = "qmd://m/docs/a.md".to_owned();
    let found = f
        .store
        .metadata_by_filepath(std::slice::from_ref(&key))
        .unwrap();
    assert_eq!(
        serde_json::to_value(&found[&key].sources).unwrap(),
        json!({"team": "global", "status": "qmd://m/docs/"})
    );
    let sources: Vec<String> = f
        .store
        .conn()
        .prepare("SELECT DISTINCT source FROM document_metadata_values WHERE source IS NOT NULL ORDER BY source")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(sources, ["global", "qmd://m/docs/"]);
}

#[test]
fn an_unreadable_stored_level_fails_the_update_naming_it() {
    let mut f = Fixture::new();
    f.set(Some("n"), "/", "status", json!("draft"));
    f.update();
    f.store
        .conn()
        .execute(
            "UPDATE store_config SET value = '{' WHERE key = 'metadata:n'",
            [],
        )
        .unwrap();
    let err = reindex_collection(&f.store, &f.root, "**/*.md", "n", &[], None).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("stored metadata for qmd://n/"),
        "{message}"
    );
    assert_eq!(
        f.resolved("top.md").1,
        json!({"status": "qmd://n/"}),
        "the failed update keeps what the document inherited"
    );
}

#[test]
fn an_unreadable_stored_level_leaves_search_and_listing_working() {
    let mut f = Fixture::new();
    f.set(Some("n"), "/", "status", json!("draft"));
    f.update();
    f.store
        .conn()
        .execute(
            "UPDATE store_config SET value = '{' WHERE key = 'metadata:n'",
            [],
        )
        .unwrap();
    assert_eq!(f.store.list_collections_with_stats().unwrap().len(), 1);
    assert_eq!(f.matching(eq("status", "draft")).len(), 3);
}
