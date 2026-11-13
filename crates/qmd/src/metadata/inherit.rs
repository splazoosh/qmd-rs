//! Metadata declared above a file: for every collection (global), for one
//! collection, or for a path prefix inside one. A document inherits it,
//! resolved at index time into the same rows its frontmatter writes, so every
//! filter and report reads one table.
//!
//! The most specific level wins, per key: global, then the collection, then
//! each longer matching prefix, then the file's own frontmatter. A value
//! replaces the one beneath it whole, arrays included, so a file can drop an
//! inherited tag by declaring the key itself.

use indexmap::IndexMap;
use sha2::{Digest, Sha256};

use super::{DocumentMetadata, MetadataValue};
use crate::store::collections::{ensure_leading_slash, path_under_prefix};

/// Where each inherited key of a document came from, by key. A key absent
/// here was declared by the file. Labels are [`level_label`]'s.
pub type MetadataSources = IndexMap<String, String>;

/// A document's metadata and the levels its inherited keys came from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedMetadata {
    pub metadata: DocumentMetadata,
    pub sources: MetadataSources,
}

/// The label a level carries wherever a value is explained: `global`, or the
/// `qmd://` address of the collection root or prefix with a trailing `/`
/// (`qmd://notes/`, `qmd://notes/docs/`). `prefix` is normalized (`/`, `/docs`).
pub fn level_label(collection: Option<&str>, prefix: &str) -> String {
    match collection {
        None => "global".to_owned(),
        Some(c) => {
            let p = prefix.trim_matches('/');
            if p.is_empty() {
                format!("qmd://{c}/")
            } else {
                format!("qmd://{c}/{p}/")
            }
        }
    }
}

/// The levels that apply inside one collection, least specific first.
#[derive(Debug, Clone, Default)]
pub struct InheritedLayers {
    /// (normalized prefix, or `None` for global; label; metadata).
    layers: Vec<(Option<String>, String, DocumentMetadata)>,
}

impl InheritedLayers {
    /// Layers for `collection` from its global and prefix levels. Prefixes
    /// are normalized (`/` is the collection root).
    pub fn new(
        collection: &str,
        global: Option<DocumentMetadata>,
        prefixes: impl IntoIterator<Item = (String, DocumentMetadata)>,
    ) -> Self {
        let mut layers: Vec<(Option<String>, String, DocumentMetadata)> = global
            .filter(|m| !m.is_empty())
            .map(|m| (None, level_label(None, "/"), m))
            .into_iter()
            .collect();
        let mut prefixed: Vec<(String, DocumentMetadata)> = prefixes
            .into_iter()
            .filter(|(_, m)| !m.is_empty())
            .collect();
        prefixed.sort_by_key(|(p, _)| p.trim_end_matches('/').len());
        layers.extend(
            prefixed
                .into_iter()
                .map(|(p, m)| (Some(p.clone()), level_label(Some(collection), &p), m)),
        );
        Self { layers }
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// What a document at collection-relative `path` inherits.
    pub fn resolve(&self, path: &str) -> Inherited {
        let path = ensure_leading_slash(path);
        let mut entries: IndexMap<String, (MetadataValue, String)> = IndexMap::new();
        for (prefix, label, metadata) in &self.layers {
            if prefix
                .as_deref()
                .is_some_and(|p| !path_under_prefix(&path, p))
            {
                continue;
            }
            for (key, value) in metadata {
                entries.insert(key.clone(), (value.clone(), label.clone()));
            }
        }
        Inherited { entries }
    }
}

/// The keys one document inherits, each with its value and level.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Inherited {
    entries: IndexMap<String, (MetadataValue, String)>,
}

impl Inherited {
    /// A digest of what is inherited, stored beside the extraction so a
    /// change to any level re-resolves the document on the next `qmd update`
    /// with no file touched. `None` when nothing is inherited, which is what
    /// an index built before inheritance holds.
    pub fn fingerprint(&self) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        let json = serde_json::to_string(&self.entries).expect("entries serialize");
        Some(hex::encode(Sha256::digest(json.as_bytes())))
    }

    /// The file's own metadata over what it inherits: the file's keys in
    /// declaration order, then inherited keys it does not declare.
    pub fn merge(&self, own: DocumentMetadata) -> ResolvedMetadata {
        let mut resolved = ResolvedMetadata {
            metadata: own,
            sources: MetadataSources::new(),
        };
        for (key, (value, label)) in &self.entries {
            if !resolved.metadata.contains_key(key) {
                resolved.metadata.insert(key.clone(), value.clone());
                resolved.sources.insert(key.clone(), label.clone());
            }
        }
        resolved
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::Scalar;

    fn md(pairs: &[(&str, &str)]) -> DocumentMetadata {
        pairs
            .iter()
            .map(|(k, v)| {
                (
                    (*k).to_owned(),
                    MetadataValue::Scalar(Scalar::String((*v).to_owned())),
                )
            })
            .collect()
    }

    #[test]
    fn the_most_specific_level_wins_per_key() {
        let layers = InheritedLayers::new(
            "notes",
            Some(md(&[("team", "core"), ("status", "g")])),
            [
                ("/docs/api".to_owned(), md(&[("status", "api")])),
                ("/".to_owned(), md(&[("status", "c"), ("area", "notes")])),
                ("/docs".to_owned(), md(&[("status", "docs")])),
            ],
        );
        let inherited = layers.resolve("docs/api/x.md");
        let resolved = inherited.merge(md(&[("area", "own")]));
        assert_eq!(
            resolved.metadata,
            md(&[("area", "own"), ("team", "core"), ("status", "api")])
        );
        assert_eq!(
            resolved.sources,
            MetadataSources::from([
                ("team".to_owned(), "global".to_owned()),
                ("status".to_owned(), "qmd://notes/docs/api/".to_owned()),
            ])
        );
        let beside = layers
            .resolve("docs-old/x.md")
            .merge(DocumentMetadata::new());
        assert_eq!(beside.sources["status"], "qmd://notes/");
    }

    #[test]
    fn nothing_inherited_has_no_fingerprint() {
        let layers = InheritedLayers::new("n", None, [("/a".to_owned(), md(&[("k", "v")]))]);
        assert_eq!(layers.resolve("b/x.md").fingerprint(), None);
        let a = layers.resolve("a/x.md").fingerprint();
        assert!(a.is_some());
        let changed = InheritedLayers::new("n", None, [("/a".to_owned(), md(&[("k", "w")]))]);
        assert_ne!(changed.resolve("a/x.md").fingerprint(), a);
    }
}
