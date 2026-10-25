//! Collection configuration: `index.yml` parsing and directory resolution.
//!
//! Ported from TS `collections.ts`. The YAML file lives at
//! `{config_dir}/{index_name}.yml` where `config_dir` resolves as
//! `QMD_CONFIG_DIR` > `$XDG_CONFIG_HOME/qmd` > `~/.config/qmd`.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::chunk::{ChunkBoundary, ChunkParams};
use crate::error::{Error, Result};
use crate::store::TermMatch;

/// Context definitions for a collection: path prefix -> description.
pub type ContextMap = IndexMap<String, String>;

/// A single collection configuration (one entry in `index.yml`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Collection {
    /// Absolute path to index.
    pub path: String,
    /// Glob pattern, e.g. `**/*.md`.
    #[serde(default = "default_pattern")]
    pub pattern: String,
    /// Glob patterns to exclude.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    /// Optional context definitions (path prefix -> text).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<ContextMap>,
    /// Optional shell command run during `qmd update`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub update: Option<String>,
    /// Include in queries by default (default: true).
    #[serde(
        default,
        rename = "includeByDefault",
        skip_serializing_if = "Option::is_none"
    )]
    pub include_by_default: Option<bool>,
    /// Chunk size in tokens for this collection's embeddings and best-chunk
    /// selection (default 900). Changing it re-embeds the collection.
    #[serde(
        default,
        rename = "chunkTokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub chunk_tokens: Option<usize>,
    /// Where chunks are cut: `window` (default, TS) or `heading`, which starts
    /// a chunk at a heading with no overlap ([`ChunkBoundary`]). Changing it
    /// re-embeds the collection.
    #[serde(
        default,
        rename = "chunkBoundary",
        skip_serializing_if = "Option::is_none"
    )]
    pub chunk_boundary: Option<ChunkBoundary>,
}

pub fn default_pattern() -> String {
    "**/*.md".to_owned()
}

impl Collection {
    pub fn new(path: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            pattern: pattern.into(),
            ignore: None,
            context: None,
            update: None,
            include_by_default: None,
            chunk_tokens: None,
            chunk_boundary: None,
        }
    }

    pub fn is_included_by_default(&self) -> bool {
        self.include_by_default != Some(false)
    }

    /// Chunk geometry for this collection.
    pub fn chunk_params(&self) -> ChunkParams {
        self.chunk_tokens
            .map_or_else(ChunkParams::default, ChunkParams::for_size)
            .with_boundary(self.chunk_boundary.unwrap_or_default())
    }
}

/// Per-role model URI overrides (`models:` section of `index.yml`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelsConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank: Option<String>,
}

/// Search defaults (`search:` section of `index.yml`); a CLI flag or MCP
/// argument overrides each one per query.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchConfig {
    /// How BM25 terms combine: `any` (default) or `all`.
    #[serde(default, rename = "match", skip_serializing_if = "Option::is_none")]
    pub term_match: Option<TermMatch>,
    /// Rerank hybrid results with the cross-encoder (default false).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank: Option<bool>,
}

/// The complete configuration file structure.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CollectionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_context: Option<String>,
    #[serde(default)]
    pub collections: IndexMap<String, Collection>,
    /// Model URI overrides (resolution order: config > env var > default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<ModelsConfig>,
    /// Search defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search: Option<SearchConfig>,
}

/// A collection paired with its name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedCollection {
    pub name: String,
    pub collection: Collection,
}

/// Resolve the qmd config directory.
pub fn config_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("QMD_CONFIG_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(xdg).join("qmd");
    }
    home_dir().join(".config").join("qmd")
}

/// Resolve the user's home directory.
///
/// # Panics
/// Panics if no home directory can be determined (no supported platform hits
/// this in practice).
pub fn home_dir() -> PathBuf {
    std::env::home_dir().expect("home directory must be resolvable")
}

/// Resolve the qmd cache directory (`$XDG_CACHE_HOME/qmd` or `~/.cache/qmd`).
pub fn cache_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".cache"));
    base.join("qmd")
}

/// Path of the config file for an index name (default `index`).
pub fn config_file_path(index_name: &str) -> PathBuf {
    config_dir().join(format!("{index_name}.yml"))
}

/// Find a project-local config by walking upward from `start_dir`.
///
/// The local config lives at `.qmd/index.yaml` or `.qmd/index.yml`; when the
/// CLI uses it, both config and index DB stay inside the project instead of
/// the global `~/.config` / `~/.cache` locations.
pub fn find_local_config_path(start_dir: &Path) -> Option<PathBuf> {
    let mut dir = start_dir;
    loop {
        let qmd_dir = dir.join(".qmd");
        for name in ["index.yaml", "index.yml"] {
            let candidate = qmd_dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        dir = dir.parent()?;
    }
}

/// The SQLite index path paired with a local `.qmd/index.yml` config file.
pub fn local_db_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("index.sqlite")
}

/// Load configuration from a YAML file. Returns an empty config if the file
/// does not exist.
pub fn load_config(path: &Path) -> Result<CollectionConfig> {
    if !path.exists() {
        return Ok(CollectionConfig::default());
    }
    let content = std::fs::read_to_string(path)?;
    if content.trim().is_empty() {
        return Ok(CollectionConfig::default());
    }
    serde_yaml_ng::from_str(&content)
        .map_err(|e| Error::Config(format!("failed to parse {}: {e}", path.display())))
}

/// Save configuration to a YAML file, creating parent directories as needed.
pub fn save_config(path: &Path, config: &CollectionConfig) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let yaml = serde_yaml_ng::to_string(config)
        .map_err(|e| Error::Config(format!("failed to serialize config: {e}")))?;
    std::fs::write(path, yaml)?;
    Ok(())
}

impl CollectionConfig {
    /// The `search:` section, empty when absent.
    pub fn search_settings(&self) -> SearchConfig {
        self.search.clone().unwrap_or_default()
    }

    /// List all collections with their names.
    pub fn list(&self) -> Vec<NamedCollection> {
        self.collections
            .iter()
            .map(|(name, collection)| NamedCollection {
                name: name.clone(),
                collection: collection.clone(),
            })
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<&Collection> {
        self.collections.get(name)
    }

    /// The collections a query searches: those `requested`, checked to exist,
    /// or else the defaults. Empty means "every collection", so when every
    /// configured collection is excluded from the defaults this is an error
    /// naming the fix, not a silent search of the excluded ones (TS qmd
    /// searches them; this port does not).
    pub fn query_collections(&self, requested: &[String]) -> Result<Vec<String>> {
        if requested.is_empty() {
            let defaults = self.default_collection_names();
            if defaults.is_empty() && !self.collections.is_empty() {
                return Err(Error::Other(
                    "no collection is included by default; name one with -c/--collection \
                     (MCP: the `collections` argument)"
                        .to_owned(),
                ));
            }
            return Ok(defaults);
        }
        if let Some(missing) = requested.iter().find(|n| self.get(n).is_none()) {
            return Err(Error::Other(format!("collection not found: {missing}")));
        }
        Ok(requested.to_vec())
    }

    /// Names of collections included in queries by default.
    pub fn default_collection_names(&self) -> Vec<String> {
        self.collections
            .iter()
            .filter(|(_, c)| c.is_included_by_default())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Add or update a collection, preserving any existing context.
    pub fn add_collection(&mut self, name: &str, path: &str, pattern: &str, ignore: &[String]) {
        let context = self.collections.get(name).and_then(|c| c.context.clone());
        let mut collection = Collection::new(path, pattern);
        collection.context = context;
        collection.ignore = (!ignore.is_empty()).then(|| ignore.to_vec());
        self.collections.insert(name.to_owned(), collection);
    }

    /// Remove a collection. Returns false if it did not exist.
    pub fn remove_collection(&mut self, name: &str) -> bool {
        self.collections.shift_remove(name).is_some()
    }
}

/// Validate a collection name: alphanumeric, hyphens, underscores.
pub fn is_valid_collection_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_config_round_trip() {
        let config = CollectionConfig::default();
        let yaml = serde_yaml_ng::to_string(&config).unwrap();
        let parsed: CollectionConfig = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(config, parsed);
    }

    #[test]
    fn parses_search_section_and_chunk_size() {
        let yaml = "search:\n  match: any\n  rerank: true\ncollections:\n  notes:\n    path: /n\n    chunkTokens: 300\n    chunkBoundary: heading\n  docs:\n    path: /d\n";
        let config: CollectionConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let search = config.search_settings();
        assert_eq!(search.term_match, Some(TermMatch::Any));
        assert_eq!(search.rerank, Some(true));
        assert_eq!(
            config.get("notes").unwrap().chunk_params(),
            ChunkParams::for_size(300).with_boundary(ChunkBoundary::Heading)
        );
        assert!(config.get("docs").unwrap().chunk_params().is_default());
        assert_eq!(
            CollectionConfig::default().search_settings(),
            SearchConfig::default()
        );
    }

    #[test]
    fn parses_ts_style_yaml() {
        let yaml = r#"
global_context: "company notes"
collections:
  notes:
    path: /home/user/notes
    pattern: "**/*.md"
    ignore:
      - "Sessions/**"
    context:
      "/": "personal notes"
    includeByDefault: false
"#;
        let config: CollectionConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let notes = config.get("notes").unwrap();
        assert_eq!(notes.path, "/home/user/notes");
        assert_eq!(
            notes.ignore.as_deref(),
            Some(&["Sessions/**".to_owned()][..])
        );
        assert!(!notes.is_included_by_default());
        assert_eq!(config.global_context.as_deref(), Some("company notes"));
    }

    #[test]
    fn query_collections_never_widens_to_excluded_ones() {
        let yaml =
            "collections:\n  a:\n    path: /a\n    includeByDefault: false\n  b:\n    path: /b\n";
        let mut config: CollectionConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(config.query_collections(&[]).unwrap(), ["b"]);
        assert_eq!(config.query_collections(&["a".to_owned()]).unwrap(), ["a"]);
        assert!(config.query_collections(&["zz".to_owned()]).is_err());

        config.collections.get_mut("b").unwrap().include_by_default = Some(false);
        let err = config.query_collections(&[]).unwrap_err().to_string();
        assert!(
            err.contains("no collection is included by default"),
            "{err}"
        );

        let empty = CollectionConfig::default();
        assert!(empty.query_collections(&[]).unwrap().is_empty());
    }

    #[test]
    fn pattern_defaults_when_missing() {
        let yaml = "collections:\n  a:\n    path: /tmp/a\n";
        let config: CollectionConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(config.get("a").unwrap().pattern, "**/*.md");
    }

    #[test]
    fn add_collection_preserves_context() {
        let mut config = CollectionConfig::default();
        config.add_collection("a", "/tmp/a", "**/*.md", &[]);
        let mut ctx = ContextMap::new();
        ctx.insert("/".to_owned(), "ctx".to_owned());
        config.collections.get_mut("a").unwrap().context = Some(ctx.clone());
        config.add_collection("a", "/tmp/b", "**/*.txt", &["x/**".to_owned()]);
        let a = config.get("a").unwrap();
        assert_eq!(a.path, "/tmp/b");
        assert_eq!(a.context, Some(ctx));
        assert_eq!(a.ignore.as_deref(), Some(&["x/**".to_owned()][..]));
    }

    #[test]
    fn local_config_discovery_walks_up() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("a").join("b");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(find_local_config_path(&nested), None);

        let qmd_dir = root.path().join(".qmd");
        std::fs::create_dir_all(&qmd_dir).unwrap();
        let yml = qmd_dir.join("index.yml");
        std::fs::write(&yml, "collections: {}\n").unwrap();
        assert_eq!(find_local_config_path(&nested), Some(yml.clone()));

        let yaml = qmd_dir.join("index.yaml");
        std::fs::write(&yaml, "collections: {}\n").unwrap();
        assert_eq!(find_local_config_path(&nested), Some(yaml.clone()));

        assert_eq!(local_db_path(&yaml), qmd_dir.join("index.sqlite"));
    }

    #[test]
    fn collection_name_validation() {
        assert!(is_valid_collection_name("my-notes_2"));
        assert!(!is_valid_collection_name("bad name"));
        assert!(!is_valid_collection_name(""));
        assert!(!is_valid_collection_name("a/b"));
    }
}
