//! `store_collections` mirror and config→DB sync, ported from TS `store.ts`
//! (`upsertStoreCollection`, `syncConfigToDb`, `listCollections`,
//! `removeCollection`, `getContextForFile`).

use rusqlite::{OptionalExtension, params};
use sha2::{Digest, Sha256};

use crate::chunk::{ChunkBoundary, ChunkParams};
use crate::config::{Collection, CollectionConfig, ContextMap, NamedCollection};
use crate::error::Result;
use crate::store::Store;

/// A collection merged with its database statistics.
#[derive(Debug, Clone)]
pub struct CollectionStats {
    pub name: String,
    pub path: String,
    pub pattern: String,
    pub ignore: Option<Vec<String>>,
    pub doc_count: i64,
    pub active_count: i64,
    pub last_modified: Option<String>,
    pub include_by_default: bool,
}

impl Store {
    /// All collections mirrored in `store_collections`.
    pub fn store_collections(&self) -> Result<Vec<NamedCollection>> {
        let mut stmt = self.conn().prepare(
            "SELECT name, path, pattern, ignore_patterns, include_by_default,
                    update_command, context
             FROM store_collections",
        )?;
        let rows = stmt.query_map([], row_to_named_collection)?;
        let mut named: Vec<NamedCollection> = rows.collect::<std::result::Result<_, _>>()?;
        for n in &mut named {
            n.collection.chunk_tokens = self.chunk_tokens_setting(&n.name)?;
            n.collection.chunk_boundary = self.chunk_boundary_setting(&n.name)?;
        }
        Ok(named)
    }

    /// One mirrored collection by name.
    pub fn store_collection(&self, name: &str) -> Result<Option<NamedCollection>> {
        let named = self
            .conn()
            .query_row(
                "SELECT name, path, pattern, ignore_patterns, include_by_default,
                        update_command, context
                 FROM store_collections WHERE name = ?1",
                params![name],
                row_to_named_collection,
            )
            .optional()?;
        named
            .map(|mut n| {
                n.collection.chunk_tokens = self.chunk_tokens_setting(name)?;
                n.collection.chunk_boundary = self.chunk_boundary_setting(name)?;
                Ok(n)
            })
            .transpose()
    }

    /// Insert or update one mirrored collection row.
    pub fn upsert_store_collection(&self, name: &str, collection: &Collection) -> Result<()> {
        let ignore_json = collection
            .ignore
            .as_ref()
            .map(|v| serde_json::to_string(v).expect("Vec<String> serializes"));
        let context_json = collection
            .context
            .as_ref()
            .map(|c| serde_json::to_string(c).expect("ContextMap serializes"));
        self.conn().execute(
            "INSERT INTO store_collections
               (name, path, pattern, ignore_patterns, include_by_default, update_command, context)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(name) DO UPDATE SET
               path = excluded.path,
               pattern = excluded.pattern,
               ignore_patterns = excluded.ignore_patterns,
               include_by_default = excluded.include_by_default,
               update_command = excluded.update_command,
               context = excluded.context",
            params![
                name,
                collection.path,
                collection.pattern,
                ignore_json,
                i64::from(collection.is_included_by_default()),
                collection.update,
                context_json,
            ],
        )?;
        self.set_chunk_settings(name, collection)
    }

    /// Chunk size and boundary are kept in `store_config`
    /// (`chunk_tokens:<collection>`, `chunk_boundary:<collection>`) rather
    /// than new `store_collections` columns, so the table stays the TS layout.
    fn set_chunk_settings(&self, name: &str, collection: &Collection) -> Result<()> {
        self.set_collection_setting(
            "chunk_tokens",
            name,
            collection.chunk_tokens.map(|t| t.to_string()),
        )?;
        self.set_collection_setting(
            "chunk_boundary",
            name,
            collection.chunk_boundary.map(|b| b.as_str().to_owned()),
        )
    }

    fn clear_chunk_settings(&self, name: &str) -> Result<()> {
        self.set_chunk_settings(name, &Collection::new("", ""))
    }

    fn set_collection_setting(
        &self,
        prefix: &str,
        name: &str,
        value: Option<String>,
    ) -> Result<()> {
        let key = format!("{prefix}:{name}");
        match value {
            Some(value) => self.conn().execute(
                "INSERT INTO store_config (key, value) VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![key, value],
            )?,
            None => self
                .conn()
                .execute("DELETE FROM store_config WHERE key = ?1", params![key])?,
        };
        Ok(())
    }

    fn collection_setting(&self, prefix: &str, name: &str) -> Result<Option<String>> {
        Ok(self
            .conn()
            .query_row(
                "SELECT value FROM store_config WHERE key = ?1",
                params![format!("{prefix}:{name}")],
                |r| r.get(0),
            )
            .optional()?)
    }

    fn chunk_tokens_setting(&self, name: &str) -> Result<Option<usize>> {
        Ok(self
            .collection_setting("chunk_tokens", name)?
            .and_then(|v| v.parse().ok()))
    }

    fn chunk_boundary_setting(&self, name: &str) -> Result<Option<ChunkBoundary>> {
        Ok(self
            .collection_setting("chunk_boundary", name)?
            .and_then(|v| v.parse().ok()))
    }

    /// Chunk geometry for a collection: its `chunkTokens` and
    /// `chunkBoundary`, else the defaults.
    pub fn collection_chunk_params(&self, name: &str) -> Result<ChunkParams> {
        Ok(self
            .chunk_tokens_setting(name)?
            .map_or_else(ChunkParams::default, ChunkParams::for_size)
            .with_boundary(self.chunk_boundary_setting(name)?.unwrap_or_default()))
    }

    /// Sync external config into `store_collections`. External config always
    /// wins; skipped when the config hash is unchanged.
    pub fn sync_config_to_db(&self, config: &CollectionConfig) -> Result<()> {
        let config_json = serde_json::to_string(config).expect("config serializes");
        let hash = hex::encode(Sha256::digest(config_json.as_bytes()));

        let existing: Option<String> = self
            .conn()
            .query_row(
                "SELECT value FROM store_config WHERE key = 'config_hash'",
                [],
                |r| r.get(0),
            )
            .optional()?;
        if existing.as_deref() == Some(hash.as_str()) {
            return Ok(());
        }

        for (name, coll) in &config.collections {
            self.upsert_store_collection(name, coll)?;
        }

        let db_names: Vec<String> = {
            let mut stmt = self.conn().prepare("SELECT name FROM store_collections")?;
            let rows = stmt.query_map([], |r| r.get(0))?;
            rows.collect::<std::result::Result<_, _>>()?
        };
        for name in db_names {
            if !config.collections.contains_key(&name) {
                self.conn().execute(
                    "DELETE FROM store_collections WHERE name = ?1",
                    params![name],
                )?;
                self.clear_chunk_settings(&name)?;
            }
        }

        match &config.global_context {
            Some(value) => {
                self.conn().execute(
                    "INSERT INTO store_config (key, value) VALUES ('global_context', ?1)
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    params![value],
                )?;
            }
            None => {
                self.conn()
                    .execute("DELETE FROM store_config WHERE key = 'global_context'", [])?;
            }
        }

        self.conn().execute(
            "INSERT INTO store_config (key, value) VALUES ('config_hash', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![hash],
        )?;
        Ok(())
    }

    /// The stored global context, if any.
    pub fn store_global_context(&self) -> Result<Option<String>> {
        let value: Option<String> = self
            .conn()
            .query_row(
                "SELECT value FROM store_config WHERE key = 'global_context'",
                [],
                |r| r.get(0),
            )
            .optional()?;
        Ok(value.filter(|v| !v.is_empty()))
    }

    /// Collections with document counts merged from the database.
    pub fn list_collections_with_stats(&self) -> Result<Vec<CollectionStats>> {
        let collections = self.store_collections()?;
        let mut result = Vec::with_capacity(collections.len());
        for named in collections {
            let (doc_count, active_count, last_modified): (i64, i64, Option<String>) =
                self.conn().query_row(
                    "SELECT COUNT(d.id),
                            COALESCE(SUM(CASE WHEN d.active = 1 THEN 1 ELSE 0 END), 0),
                            MAX(d.modified_at)
                     FROM documents d WHERE d.collection = ?1",
                    params![named.name],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )?;
            result.push(CollectionStats {
                include_by_default: named.collection.is_included_by_default(),
                name: named.name,
                path: named.collection.path,
                pattern: named.collection.pattern,
                ignore: named.collection.ignore,
                doc_count,
                active_count,
                last_modified,
            });
        }
        Ok(result)
    }

    /// Remove a collection: delete its documents, clean orphaned content, and
    /// drop the mirror row. Returns (deleted docs, cleaned hashes).
    pub fn remove_collection(&self, name: &str) -> Result<(usize, usize)> {
        let deleted = self
            .conn()
            .execute("DELETE FROM documents WHERE collection = ?1", params![name])?;
        let cleaned = self.conn().execute(
            "DELETE FROM content
             WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
            [],
        )?;
        self.conn().execute(
            "DELETE FROM store_collections WHERE name = ?1",
            params![name],
        )?;
        self.clear_chunk_settings(name)?;
        Ok((deleted, cleaned))
    }

    /// Rename a collection in the database: its documents (their FTS rows
    /// follow by trigger), its mirror row and its chunk settings. Vectors are
    /// keyed by content hash, so they stay; a hash also held by another
    /// collection may change home and re-embed if the two chunk differently.
    /// Returns the number of documents moved.
    pub fn rename_collection(&self, old: &str, new: &str) -> Result<usize> {
        let tx = self.conn().unchecked_transaction()?;
        let moved = tx.execute(
            "UPDATE documents SET collection = ?2 WHERE collection = ?1",
            params![old, new],
        )?;
        tx.execute(
            "UPDATE store_collections SET name = ?2 WHERE name = ?1",
            params![old, new],
        )?;
        for prefix in ["chunk_tokens", "chunk_boundary"] {
            tx.execute(
                "UPDATE store_config SET key = ?2 WHERE key = ?1",
                params![format!("{prefix}:{old}"), format!("{prefix}:{new}")],
            )?;
        }
        tx.commit()?;
        Ok(moved)
    }

    /// Context for a file: global context plus all matching path-prefix
    /// contexts (most general first), joined with blank lines.
    pub fn context_for_file(&self, filepath: &str) -> Result<Option<String>> {
        let Some((collection_name, relative_path)) = crate::store::parse_virtual_path(filepath)
        else {
            return Ok(None);
        };
        let Some(named) = self.store_collection(&collection_name)? else {
            return Ok(None);
        };

        let exists: Option<i64> = self
            .conn()
            .query_row(
                "SELECT 1 FROM documents WHERE collection = ?1 AND path = ?2 AND active = 1 LIMIT 1",
                params![collection_name, relative_path],
                |r| r.get(0),
            )
            .optional()?;
        if exists.is_none() {
            return Ok(None);
        }

        let mut contexts: Vec<String> = Vec::new();
        if let Some(global) = self.store_global_context()? {
            contexts.push(global);
        }
        if let Some(ctx_map) = &named.collection.context {
            let normalized_path = ensure_leading_slash(&relative_path);
            let mut matching: Vec<(String, &String)> = ctx_map
                .iter()
                .filter_map(|(prefix, context)| {
                    let normalized_prefix = ensure_leading_slash(prefix);
                    path_under_prefix(&normalized_path, &normalized_prefix)
                        .then_some((normalized_prefix, context))
                })
                .collect();
            matching.sort_by_key(|(prefix, _)| prefix.len());
            contexts.extend(matching.into_iter().map(|(_, c)| c.clone()));
        }

        Ok((!contexts.is_empty()).then(|| contexts.join("\n\n")))
    }
}

fn ensure_leading_slash(path: &str) -> String {
    if path.starts_with('/') {
        path.to_owned()
    } else {
        format!("/{path}")
    }
}

/// Whether `path` is `prefix` or lies under it on a segment boundary: `/docs`
/// covers `/docs/a.md`, not `/docs-old/a.md`.
fn path_under_prefix(path: &str, prefix: &str) -> bool {
    let prefix = prefix.trim_end_matches('/');
    path.strip_prefix(prefix)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

fn row_to_named_collection(r: &rusqlite::Row<'_>) -> rusqlite::Result<NamedCollection> {
    let ignore_json: Option<String> = r.get(3)?;
    let context_json: Option<String> = r.get(6)?;
    let include_by_default: i64 = r.get(4)?;
    Ok(NamedCollection {
        name: r.get(0)?,
        collection: Collection {
            path: r.get(1)?,
            pattern: r.get(2)?,
            ignore: ignore_json.and_then(|j| serde_json::from_str::<Vec<String>>(&j).ok()),
            include_by_default: (include_by_default == 0).then_some(false),
            update: r.get(5)?,
            context: context_json.and_then(|j| serde_json::from_str::<ContextMap>(&j).ok()),
            chunk_tokens: None,
            chunk_boundary: None,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_prefix_matches_on_segment_boundary() {
        assert!(path_under_prefix("/docs/a.md", "/docs"));
        assert!(path_under_prefix("/docs/a.md", "/docs/"));
        assert!(path_under_prefix("/docs", "/docs"));
        assert!(path_under_prefix("/docs/a.md", "/"));
        assert!(!path_under_prefix("/docs-old/a.md", "/docs"));
    }

    #[test]
    fn removing_a_collection_clears_its_chunk_settings() {
        let store = Store::open_in_memory().unwrap();
        let mut collection = Collection::new("/n", "**/*.md");
        collection.chunk_tokens = Some(300);
        collection.chunk_boundary = Some(ChunkBoundary::Heading);
        store.upsert_store_collection("n", &collection).unwrap();
        store.remove_collection("n").unwrap();
        assert!(store.collection_chunk_params("n").unwrap().is_default());
    }

    #[test]
    fn renaming_a_collection_moves_its_documents_and_settings() {
        let store = Store::open_in_memory().unwrap();
        let mut collection = Collection::new("/n", "**/*.md");
        collection.chunk_tokens = Some(300);
        store.upsert_store_collection("n", &collection).unwrap();
        let params = store.collection_chunk_params("n").unwrap();
        let hash = crate::store::hash_content("# Raft\n\nleader election");
        store
            .insert_content(&hash, "# Raft\n\nleader election", "2026-01-01")
            .unwrap();
        store
            .insert_document("n", "raft.md", "Raft", &hash, "2026-01-01", "2026-01-01")
            .unwrap();

        assert_eq!(store.rename_collection("n", "m").unwrap(), 1);
        assert!(store.store_collection("n").unwrap().is_none());
        assert!(store.store_collection("m").unwrap().is_some());
        assert_eq!(store.collection_chunk_params("m").unwrap(), params);
        assert!(store.collection_chunk_params("n").unwrap().is_default());
        let hits = store.search_fts("leader", 5, Some("m")).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].filepath, "qmd://m/raft.md");
    }
}
