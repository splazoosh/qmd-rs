//! Read-only drift detection: how far the index has fallen behind the disk.
//!
//! [`drift_collection`] answers the question [`reindex_collection`] answers as
//! a by-product of doing the work — new, changed, and gone files — without
//! writing a row. File selection, the empty-file skip, and the content hash
//! are the same, so the counts predict what an update would do.
//!
//! [`reindex_collection`]: super::reindex_collection

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::error::Result;
use crate::store::Store;

/// Counters returned by [`drift_collection`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DriftResult {
    /// On disk, not in the index.
    pub new: usize,
    /// In both, content hash differs.
    pub changed: usize,
    /// In the index, no longer on disk (or now excluded).
    pub gone: usize,
    /// In both, content hash identical.
    pub unchanged: usize,
}

impl DriftResult {
    /// Whether an update would change nothing.
    pub fn is_clean(&self) -> bool {
        self.new == 0 && self.changed == 0 && self.gone == 0
    }

    /// Files an update would touch.
    pub fn drifted(&self) -> usize {
        self.new + self.changed + self.gone
    }
}

/// Compare `collection_path` against what the store holds for
/// `collection_name`. Reads the filesystem and the `documents` table; writes
/// nothing.
pub fn drift_collection(
    store: &Store,
    collection_path: &Path,
    pattern: &str,
    collection_name: &str,
    ignore_patterns: &[String],
) -> Result<DriftResult> {
    let indexed: HashMap<String, String> = store
        .active_document_hashes(collection_name)?
        .into_iter()
        .collect();
    let mut by_lowercase: HashMap<String, Vec<(&str, &str)>> = HashMap::new();
    for (path, hash) in &indexed {
        by_lowercase
            .entry(path.to_lowercase())
            .or_default()
            .push((path.as_str(), hash.as_str()));
    }

    let walk = super::walk::walk_files(collection_path, pattern, ignore_patterns)?;
    let files = &walk.files;
    let walked: HashSet<&str> = files.iter().map(String::as_str).collect();
    let mut result = DriftResult::default();
    let mut seen: HashSet<&str> = HashSet::with_capacity(files.len());

    for relative in files {
        let matched = indexed
            .get_key_value(relative)
            .map(|(p, h)| (p.as_str(), h.as_str()))
            .or_else(|| {
                by_lowercase
                    .get(&relative.to_lowercase())?
                    .iter()
                    .copied()
                    .find(|(p, _)| !walked.contains(p))
            });
        if let Some((path, _)) = matched {
            seen.insert(path);
        }

        let Ok(content) = std::fs::read_to_string(collection_path.join(relative)) else {
            continue;
        };
        if content.trim().is_empty() {
            continue;
        }

        match matched {
            Some((_, indexed_hash)) => {
                if indexed_hash == crate::store::hash_content(&content) {
                    result.unchanged += 1;
                } else {
                    result.changed += 1;
                }
            }
            None => result.new += 1,
        }
    }

    if walk.errors == 0 {
        result.gone = indexed
            .keys()
            .filter(|p| !seen.contains(p.as_str()))
            .count();
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::reindex_collection;
    use std::fs;

    fn write(root: &Path, rel: &str, content: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, content).unwrap();
    }

    fn drift(store: &Store, root: &Path) -> DriftResult {
        drift_collection(store, root, "**/*.md", "t", &[]).unwrap()
    }

    #[test]
    fn reports_new_changed_and_gone_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(root, "a.md", "# A\nbody a\n");
        write(root, "b.md", "# B\nbody b\n");

        let store = Store::open_in_memory().unwrap();
        reindex_collection(&store, root, "**/*.md", "t", &[], None).unwrap();
        assert!(drift(&store, root).is_clean());

        write(root, "a.md", "# A\nbody a rewritten\n");
        fs::remove_file(root.join("b.md")).unwrap();
        write(root, "c.md", "# C\nbody c\n");

        let d = drift(&store, root);
        assert_eq!((d.new, d.changed, d.gone, d.unchanged), (1, 1, 1, 0));

        assert_eq!(drift(&store, root), d);
        assert_eq!(store.active_document_count("t").unwrap(), 2);

        reindex_collection(&store, root, "**/*.md", "t", &[], None).unwrap();
        assert!(drift(&store, root).is_clean());
    }

    #[test]
    fn empty_and_excluded_files_are_not_drift() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(root, "a.md", "# A\nbody\n");
        let store = Store::open_in_memory().unwrap();
        reindex_collection(&store, root, "**/*.md", "t", &[], None).unwrap();

        write(root, "blank.md", "   \n");
        write(root, "target/gen.md", "# generated\n");
        let d =
            drift_collection(&store, root, "**/*.md", "t", &["**/target/**".to_owned()]).unwrap();
        assert!(d.is_clean());
        assert_eq!(d.unchanged, 1);
    }
}
