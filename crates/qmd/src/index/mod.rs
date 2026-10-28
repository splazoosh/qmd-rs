//! Collection indexing: scan the filesystem and sync it into the store.
//! Ported from TS `reindexCollection`.

pub mod drift;
pub mod walk;

use std::collections::HashSet;
use std::path::Path;
use std::time::SystemTime;

use crate::error::Result;
use crate::store::{Store, now_iso};

/// Counters returned by [`reindex_collection`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReindexResult {
    pub indexed: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub orphaned_cleaned: usize,
    /// Directory entries the walk could not read. When non-zero no document
    /// was deactivated, since a missing file may only be unreadable.
    pub walk_errors: usize,
}

fn system_time_iso(t: std::io::Result<SystemTime>, fallback: &str) -> String {
    t.map(|st| {
        chrono::DateTime::<chrono::Utc>::from(st)
            .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
    })
    .unwrap_or_else(|_| fallback.to_owned())
}

/// Progress callback: (relative file, current, total).
pub type ProgressFn<'a> = &'a mut dyn FnMut(&str, usize, usize);

/// Scan `collection_path` for files matching `pattern` (minus
/// `ignore_patterns`) and sync them into the store: insert new documents,
/// update changed ones, deactivate missing ones, clean orphaned content.
///
/// The whole sync is one transaction unless the caller already holds one.
///
/// # Errors
/// Fails when `collection_path` is not a readable directory, on an invalid
/// glob, and on store failures.
pub fn reindex_collection(
    store: &Store,
    collection_path: &Path,
    pattern: &str,
    collection_name: &str,
    ignore_patterns: &[String],
    on_progress: Option<ProgressFn<'_>>,
) -> Result<ReindexResult> {
    let walk = walk::walk_files(collection_path, pattern, ignore_patterns)?;
    let tx = if store.conn().is_autocommit() {
        Some(store.conn().unchecked_transaction()?)
    } else {
        None
    };
    let result = sync_files(store, collection_path, collection_name, &walk, on_progress)?;
    if let Some(tx) = tx {
        tx.commit()?;
    }
    Ok(result)
}

fn sync_files(
    store: &Store,
    collection_path: &Path,
    collection_name: &str,
    walk: &walk::Walk,
    mut on_progress: Option<ProgressFn<'_>>,
) -> Result<ReindexResult> {
    let now = now_iso();
    let files = &walk.files;
    let total = files.len();
    let walked: HashSet<&str> = files.iter().map(String::as_str).collect();

    let mut result = ReindexResult {
        walk_errors: walk.errors,
        ..ReindexResult::default()
    };
    let mut seen_paths: HashSet<String> = HashSet::with_capacity(total);
    let mut processed = 0usize;

    for relative in files {
        processed += 1;
        if let Some(cb) = on_progress.as_deref_mut() {
            cb(relative, processed, total);
        }

        let full_path = collection_path.join(relative);
        seen_paths.insert(relative.clone());

        let Ok(content) = std::fs::read_to_string(&full_path) else {
            continue;
        };
        if content.trim().is_empty() {
            continue;
        }

        let hash = crate::store::hash_content(&content);
        let title = crate::store::extract_title(&content, relative);

        match store.find_or_migrate_document(collection_name, relative, &walked)? {
            Some((id, existing_hash, existing_title)) => {
                if existing_hash == hash {
                    if existing_title == title {
                        result.unchanged += 1;
                    } else {
                        store.update_document_title(id, &title, &now)?;
                        result.updated += 1;
                    }
                } else {
                    store.insert_content(&hash, &content, &now)?;
                    let modified = system_time_iso(
                        std::fs::metadata(&full_path).and_then(|m| m.modified()),
                        &now,
                    );
                    store.update_document(id, &title, &hash, &modified)?;
                    result.updated += 1;
                }
            }
            None => {
                store.insert_content(&hash, &content, &now)?;
                let metadata = std::fs::metadata(&full_path);
                let created = system_time_iso(
                    metadata
                        .as_ref()
                        .map_err(clone_io_err)
                        .and_then(|m| m.created()),
                    &now,
                );
                let modified = system_time_iso(
                    metadata
                        .as_ref()
                        .map_err(clone_io_err)
                        .and_then(|m| m.modified()),
                    &now,
                );
                store.insert_document(
                    collection_name,
                    relative,
                    &title,
                    &hash,
                    &created,
                    &modified,
                )?;
                result.indexed += 1;
            }
        }
    }

    if walk.errors > 0 {
        eprintln!(
            "qmd: {} entries under {} could not be read; no documents deactivated",
            walk.errors,
            collection_path.display()
        );
    } else {
        for path in store.active_document_paths(collection_name)? {
            if !seen_paths.contains(&path) {
                store.deactivate_document(collection_name, &path)?;
                result.removed += 1;
            }
        }
    }

    result.orphaned_cleaned = store.cleanup_orphaned_content()?;
    store.sync_sections(Some(collection_name))?;
    Ok(result)
}

fn clone_io_err(e: &std::io::Error) -> std::io::Error {
    std::io::Error::new(e.kind(), e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(root: &Path, rel: &str, content: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, content).unwrap();
    }

    #[test]
    fn incremental_reindex_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(root, "a.md", "# Doc A\ncontent a\n");
        write(root, "sub/b.md", "# Doc B\ncontent b\n");

        let store = Store::open_in_memory().unwrap();

        let r1 = reindex_collection(&store, root, "**/*.md", "test", &[], None).unwrap();
        assert_eq!(
            (r1.indexed, r1.updated, r1.unchanged, r1.removed),
            (2, 0, 0, 0)
        );

        let r2 = reindex_collection(&store, root, "**/*.md", "test", &[], None).unwrap();
        assert_eq!(
            (r2.indexed, r2.updated, r2.unchanged, r2.removed),
            (0, 0, 2, 0)
        );

        write(root, "a.md", "# Doc A v2\nnew content\n");
        fs::remove_file(root.join("sub/b.md")).unwrap();
        write(root, "c.md", "# Doc C\ncontent c\n");
        let r3 = reindex_collection(&store, root, "**/*.md", "test", &[], None).unwrap();
        assert_eq!(
            (r3.indexed, r3.updated, r3.unchanged, r3.removed),
            (1, 1, 0, 1)
        );
        assert_eq!(r3.orphaned_cleaned, 1);

        let hits = store.search_fts("content", 10, None).unwrap();
        let paths: Vec<&str> = hits.iter().map(|h| h.display_path.as_str()).collect();
        assert!(paths.contains(&"test/a.md"));
        assert!(paths.contains(&"test/c.md"));
        assert!(!paths.contains(&"test/sub/b.md"));

        let content_rows: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM content", [], |r| r.get(0))
            .unwrap();
        assert_eq!(content_rows, 3);
    }

    #[test]
    fn missing_root_errors_without_deactivating() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("coll");
        write(&root, "a.md", "# A\n");
        let store = Store::open_in_memory().unwrap();
        reindex_collection(&store, &root, "**/*.md", "t", &[], None).unwrap();

        fs::remove_dir_all(&root).unwrap();
        assert!(reindex_collection(&store, &root, "**/*.md", "t", &[], None).is_err());
        assert_eq!(store.active_document_count("t").unwrap(), 1);
    }

    #[test]
    fn empty_files_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "empty.md", "   \n");
        write(dir.path(), "real.md", "# Real\n");
        let store = Store::open_in_memory().unwrap();
        let r = reindex_collection(&store, dir.path(), "**/*.md", "t", &[], None).unwrap();
        assert_eq!(r.indexed, 1);
    }

    #[test]
    fn title_only_change_counts_as_update() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "a.md", "# One\nsame body hash differs though\n");
        let store = Store::open_in_memory().unwrap();
        reindex_collection(&store, dir.path(), "**/*.md", "t", &[], None).unwrap();
        write(dir.path(), "a.md", "# Two\nsame body hash differs though\n");
        let r = reindex_collection(&store, dir.path(), "**/*.md", "t", &[], None).unwrap();
        assert_eq!((r.indexed, r.updated), (0, 1));
    }
}
