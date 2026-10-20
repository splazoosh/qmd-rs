//! Vector storage and search: `content_vectors` + the `vectors_vec` vec0
//! virtual table, ported from TS `store.ts` (`ensureVecTableInternal`,
//! `insertEmbedding`, `clearAllEmbeddings`, `getPendingEmbeddingDocs`,
//! `removeIncompleteEmbeddings`, `searchVec`).

use std::collections::HashMap;

use rusqlite::params;

use crate::embed::embedding_fingerprint;
use crate::error::{Error, Result};
use crate::store::{SearchResult, Store};

/// A document (by content hash) that still needs embeddings.
#[derive(Debug, Clone)]
pub struct PendingEmbeddingDoc {
    pub hash: String,
    /// A sample path for display.
    pub path: String,
    /// Body size in bytes (for batch budgeting).
    pub bytes: u64,
}

/// Serialize an f32 vector to the little-endian blob layout sqlite-vec uses.
fn embedding_blob(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl Store {
    /// Whether the `vectors_vec` virtual table exists.
    pub fn vec_table_exists(&self) -> Result<bool> {
        let n: i64 = self.conn().query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='vectors_vec'",
            [],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    /// Create (or verify) the vec0 table for the given dimensionality.
    ///
    /// # Errors
    /// Fails when existing vectors have a different dimensionality (re-embed
    /// with `embed -f`) or sqlite-vec is unavailable.
    pub fn ensure_vec_table(&self, dimensions: usize) -> Result<()> {
        if !self.vec_available() {
            return Err(Error::Other(
                "vector operations require the sqlite-vec extension".to_owned(),
            ));
        }
        let existing_sql: Option<String> = self
            .conn()
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='vectors_vec'",
                [],
                |r| r.get(0),
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;

        if let Some(sql) = existing_sql {
            let existing_dims = sql
                .split("float[")
                .nth(1)
                .and_then(|rest| rest.split(']').next())
                .and_then(|d| d.parse::<usize>().ok());
            let has_hash_seq = sql.contains("hash_seq");
            let has_cosine = sql.contains("distance_metric=cosine");
            if existing_dims == Some(dimensions) && has_hash_seq && has_cosine {
                return Ok(());
            }
            if let Some(dims) = existing_dims
                && dims != dimensions
            {
                return Err(Error::Other(format!(
                    "embedding dimension mismatch: existing vectors are {dims}d but the \
                     current model produces {dimensions}d. Run 'qmd embed -f' to re-embed."
                )));
            }
            self.conn()
                .execute_batch("DROP TABLE IF EXISTS vectors_vec")?;
        }
        self.conn().execute_batch(&format!(
            "CREATE VIRTUAL TABLE vectors_vec USING vec0(hash_seq TEXT PRIMARY KEY, \
             embedding float[{dimensions}] distance_metric=cosine)"
        ))?;
        Ok(())
    }

    /// Insert one chunk embedding into `content_vectors` and `vectors_vec`.
    ///
    /// All three writes share one savepoint, so a failed vector insert (e.g. a
    /// dimension mismatch) cannot leave a `content_vectors` row without its
    /// vector. vec0 tables ignore `OR REPLACE`, hence the explicit DELETE +
    /// INSERT.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_embedding(
        &self,
        hash: &str,
        seq: i64,
        pos: i64,
        embedding: &[f32],
        model: &str,
        embedded_at: &str,
        total_chunks: i64,
        fingerprint: &str,
    ) -> Result<()> {
        let hash_seq = format!("{hash}_{seq}");
        let tx = self.conn().unchecked_transaction()?;
        tx.execute(
            "INSERT OR REPLACE INTO content_vectors \
             (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                hash,
                seq,
                pos,
                model,
                fingerprint,
                total_chunks,
                embedded_at
            ],
        )?;
        tx.execute(
            "DELETE FROM vectors_vec WHERE hash_seq = ?",
            params![hash_seq],
        )?;
        tx.execute(
            "INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)",
            params![hash_seq, embedding_blob(embedding)],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Clear embeddings for the whole index or one collection.
    ///
    /// A scoped clear only removes vectors whose hash is referenced
    /// exclusively by active documents of that collection; `vectors_vec` is
    /// dropped when it ends up empty (recreated on the next embed with the
    /// current dimensions).
    pub fn clear_all_embeddings(&self, collection: Option<&str>) -> Result<()> {
        let Some(collection) = collection else {
            self.conn()
                .execute_batch("DELETE FROM content_vectors; DROP TABLE IF EXISTS vectors_vec;")?;
            return Ok(());
        };

        const EXCLUSIVE_HASHES: &str = "
            SELECT DISTINCT d.hash
            FROM documents d
            WHERE d.collection = ?1 AND d.active = 1
              AND NOT EXISTS (
                SELECT 1 FROM documents d2
                WHERE d2.hash = d.hash AND d2.active = 1 AND d2.collection != d.collection
              )";

        if self.vec_table_exists()? {
            let hash_seqs: Vec<String> = {
                let mut stmt = self.conn().prepare(&format!(
                    "SELECT cv.hash || '_' || cv.seq FROM content_vectors cv \
                     WHERE cv.hash IN ({EXCLUSIVE_HASHES})"
                ))?;
                let rows = stmt.query_map(params![collection], |r| r.get(0))?;
                rows.collect::<std::result::Result<_, _>>()?
            };
            let mut del = self
                .conn()
                .prepare("DELETE FROM vectors_vec WHERE hash_seq = ?")?;
            for hash_seq in hash_seqs {
                del.execute(params![hash_seq])?;
            }
        }

        self.conn().execute(
            &format!("DELETE FROM content_vectors WHERE hash IN ({EXCLUSIVE_HASHES})"),
            params![collection],
        )?;

        let remaining: i64 =
            self.conn()
                .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))?;
        if remaining == 0 {
            self.conn()
                .execute_batch("DROP TABLE IF EXISTS vectors_vec")?;
        }
        Ok(())
    }

    /// Documents (by unique content hash) that still need embeddings for the
    /// given model + current fingerprint, ordered by path.
    pub fn pending_embedding_docs(
        &self,
        collection: Option<&str>,
        model: &str,
    ) -> Result<Vec<PendingEmbeddingDoc>> {
        let fingerprint = embedding_fingerprint(model);
        let collection_filter = if collection.is_some() {
            "AND d.collection = ?3"
        } else {
            ""
        };
        let sql = format!(
            "SELECT d.hash, MIN(d.path) as path, length(CAST(c.doc AS BLOB)) as bytes
             FROM documents d
             JOIN content c ON d.hash = c.hash
             LEFT JOIN (
               SELECT hash, model, COUNT(*) AS chunk_count, MAX(total_chunks) AS expected_chunks
               FROM content_vectors
               WHERE model = ?1 AND embed_fingerprint = ?2
               GROUP BY hash, model, embed_fingerprint
             ) v ON d.hash = v.hash
             WHERE d.active = 1
               AND (v.hash IS NULL OR v.chunk_count < v.expected_chunks)
               {collection_filter}
             GROUP BY d.hash
             ORDER BY MIN(d.path)"
        );
        let mut stmt = self.conn().prepare(&sql)?;
        let map_row = |r: &rusqlite::Row<'_>| -> rusqlite::Result<PendingEmbeddingDoc> {
            Ok(PendingEmbeddingDoc {
                hash: r.get(0)?,
                path: r.get(1)?,
                bytes: r.get::<_, i64>(2)?.max(0).unsigned_abs(),
            })
        };
        let rows = match collection {
            Some(c) => stmt.query_map(params![model, fingerprint, c], map_row)?,
            None => stmt.query_map(params![model, fingerprint], map_row)?,
        };
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// Fetch document bodies for a set of content hashes.
    pub fn content_bodies(&self, hashes: &[String]) -> Result<HashMap<String, String>> {
        if hashes.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders = vec!["?"; hashes.len()].join(",");
        let mut stmt = self.conn().prepare(&format!(
            "SELECT hash, doc FROM content WHERE hash IN ({placeholders})"
        ))?;
        let rows = stmt.query_map(rusqlite::params_from_iter(hashes.iter()), |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// Drop partial chunk sets (a hash whose stored chunk count differs from
    /// the expected count), so they are re-selected next run. Returns the
    /// number of removed chunk rows.
    pub fn remove_incomplete_embeddings(
        &self,
        expected_chunks_by_hash: &HashMap<String, usize>,
        model: &str,
    ) -> Result<usize> {
        let mut removed = 0usize;
        for (hash, expected) in expected_chunks_by_hash {
            let seqs: Vec<i64> = {
                let mut stmt = self
                    .conn()
                    .prepare("SELECT seq FROM content_vectors WHERE hash = ? AND model = ?")?;
                let rows = stmt.query_map(params![hash, model], |r| r.get(0))?;
                rows.collect::<std::result::Result<_, _>>()?
            };
            if seqs.is_empty() || seqs.len() == *expected {
                continue;
            }
            let mut del_vec = self
                .conn()
                .prepare("DELETE FROM vectors_vec WHERE hash_seq = ?")?;
            for seq in &seqs {
                del_vec.execute(params![format!("{hash}_{seq}")])?;
            }
            self.conn().execute(
                "DELETE FROM content_vectors WHERE hash = ? AND model = ?",
                params![hash, model],
            )?;
            removed += seqs.len();
        }
        Ok(removed)
    }

    /// Vector KNN search over document chunks with a precomputed query
    /// embedding. Results are deduped by file (best chunk wins) with
    /// `score = 1 - cosine_distance`.
    ///
    /// Two-step queries by design: vec0 virtual tables hang when combined
    /// with JOINs in one statement (TS qmd PR #23) — do not merge these.
    pub fn search_vec(
        &self,
        query_embedding: &[f32],
        limit: usize,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        if !self.vec_table_exists()? {
            return Ok(Vec::new());
        }

        let mut k = limit
            .saturating_mul(if collection.is_some() { 10 } else { 3 })
            .clamp(1, MAX_KNN_K);
        let best = loop {
            let (fetched, best) = self.vec_candidates(query_embedding, k, collection)?;
            if best.len() >= limit || fetched < k || k == MAX_KNN_K {
                break best;
            }
            k = k.saturating_mul(4).min(MAX_KNN_K);
        };

        let mut hits: Vec<(VecRow, f64)> = best.into_values().collect();
        hits.sort_by(|a, b| a.1.total_cmp(&b.1));
        hits.truncate(limit);

        let mut results = Vec::with_capacity(hits.len());
        for (row, distance) in hits {
            results.push(SearchResult {
                docid: super::docs::docid(&row.hash),
                context: self.context_for_file(&row.filepath)?,
                filepath: row.filepath,
                display_path: row.display_path,
                title: row.title,
                hash: row.hash,
                collection_name: row.collection_name,
                body: row.body,
                score: 1.0 - distance,
            });
        }
        Ok(results)
    }

    /// The two steps of [`Self::search_vec`] for one `k`: returns how many
    /// neighbours vec0 gave, and the surviving rows deduped by file (closest
    /// chunk wins).
    fn vec_candidates(
        &self,
        query_embedding: &[f32],
        k: usize,
        collection: Option<&str>,
    ) -> Result<(usize, BestByFile)> {
        let knn: Vec<(String, f64)> = {
            let mut stmt = self.conn().prepare(
                "SELECT hash_seq, distance FROM vectors_vec WHERE embedding MATCH ?1 AND k = ?2",
            )?;
            #[allow(clippy::cast_possible_wrap)]
            let rows = stmt.query_map(params![embedding_blob(query_embedding), k as i64], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })?;
            rows.collect::<std::result::Result<_, _>>()?
        };
        let mut best: BestByFile = HashMap::new();
        if knn.is_empty() {
            return Ok((0, best));
        }

        let distance_by_key: HashMap<&str, f64> =
            knn.iter().map(|(k, d)| (k.as_str(), *d)).collect();
        let placeholders = vec!["?"; knn.len()].join(",");
        let collection_filter = if collection.is_some() {
            format!(" AND d.collection = ?{}", knn.len() + 1)
        } else {
            String::new()
        };
        let sql = format!(
            "SELECT
               cv.hash || '_' || cv.seq,
               cv.hash,
               'qmd://' || d.collection || '/' || d.path,
               d.collection || '/' || d.path,
               d.title,
               d.collection,
               content.doc
             FROM content_vectors cv
             JOIN documents d ON d.hash = cv.hash AND d.active = 1
             JOIN content ON content.hash = d.hash
             WHERE cv.hash || '_' || cv.seq IN ({placeholders}){collection_filter}"
        );
        let mut stmt = self.conn().prepare(&sql)?;
        let params_vec: Vec<&str> = knn
            .iter()
            .map(|(k, _)| k.as_str())
            .chain(collection.iter().copied())
            .collect();
        let rows = stmt.query_map(rusqlite::params_from_iter(params_vec.iter()), |r| {
            Ok(VecRow {
                hash_seq: r.get(0)?,
                hash: r.get(1)?,
                filepath: r.get(2)?,
                display_path: r.get(3)?,
                title: r.get(4)?,
                collection_name: r.get(5)?,
                body: r.get(6)?,
            })
        })?;

        for row in rows {
            let row = row?;
            let distance = distance_by_key
                .get(row.hash_seq.as_str())
                .copied()
                .unwrap_or(1.0);
            match best.get(&row.filepath) {
                Some((_, existing)) if *existing <= distance => {}
                _ => {
                    best.insert(row.filepath.clone(), (row, distance));
                }
            }
        }
        Ok((knn.len(), best))
    }
}

/// sqlite-vec rejects KNN queries with k above 4096.
const MAX_KNN_K: usize = 4096;

/// One chunk row joined to its document, as [`Store::search_vec`] reads it.
struct VecRow {
    hash_seq: String,
    hash: String,
    filepath: String,
    display_path: String,
    title: String,
    collection_name: String,
    body: String,
}

/// Rows keyed by file path, each with its closest chunk's distance.
type BestByFile = HashMap<String, (VecRow, f64)>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::hash_content;

    fn store_with_doc(collection: &str, path: &str, body: &str) -> (Store, String) {
        let store = Store::open_in_memory().unwrap();
        let hash = insert_doc(&store, collection, path, body);
        (store, hash)
    }

    fn insert_doc(store: &Store, collection: &str, path: &str, body: &str) -> String {
        let now = crate::store::now_iso();
        let hash = hash_content(body);
        store.insert_content(&hash, body, &now).unwrap();
        store
            .insert_document(collection, path, "T", &hash, &now, &now)
            .unwrap();
        hash
    }

    fn unit(dims: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0; dims];
        v[hot] = 1.0;
        v
    }

    #[test]
    fn ensure_vec_table_verifies_and_rejects_dimension_changes() {
        let store = Store::open_in_memory().unwrap();
        store.ensure_vec_table(4).unwrap();
        store.ensure_vec_table(4).unwrap();
        let err = store.ensure_vec_table(8).unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"), "{err}");
    }

    #[test]
    fn insert_search_and_dedupe_by_best_chunk() {
        let (store, hash) = store_with_doc("c", "a.md", "doc body");
        let hash_b = insert_doc(&store, "c", "b.md", "other body");
        store.ensure_vec_table(4).unwrap();
        let now = crate::store::now_iso();

        store
            .insert_embedding(&hash, 0, 0, &unit(4, 1), "m", &now, 2, "fp")
            .unwrap();
        store
            .insert_embedding(&hash, 1, 10, &unit(4, 0), "m", &now, 2, "fp")
            .unwrap();
        store
            .insert_embedding(&hash_b, 0, 0, &unit(4, 2), "m", &now, 1, "fp")
            .unwrap();

        let results = store.search_vec(&unit(4, 0), 5, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].display_path, "c/a.md");
        assert!(
            (results[0].score - 1.0).abs() < 1e-5,
            "exact match scores 1"
        );
        assert!(results[1].score < 0.5);

        let none = store.search_vec(&unit(4, 0), 5, Some("missing")).unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn collection_filter_widens_knn_past_closer_neighbours() {
        let store = Store::open_in_memory().unwrap();
        store.ensure_vec_table(4).unwrap();
        let now = crate::store::now_iso();
        for i in 0..12 {
            let hash = insert_doc(&store, "a", &format!("{i}.md"), &format!("body {i}"));
            store
                .insert_embedding(&hash, 0, 0, &unit(4, 0), "m", &now, 1, "fp")
                .unwrap();
        }
        let far = insert_doc(&store, "b", "far.md", "far body");
        store
            .insert_embedding(&far, 0, 0, &unit(4, 1), "m", &now, 1, "fp")
            .unwrap();

        let results = store.search_vec(&unit(4, 0), 1, Some("b")).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].display_path, "b/far.md");
    }

    #[test]
    fn search_without_vec_table_returns_empty() {
        let (store, _) = store_with_doc("c", "a.md", "doc body");
        assert!(store.search_vec(&unit(4, 0), 5, None).unwrap().is_empty());
    }

    #[test]
    fn pending_docs_respect_fingerprint_and_completeness() {
        let (store, hash) = store_with_doc("c", "a.md", "doc body");
        let model = "m";
        let fingerprint = embedding_fingerprint(model);

        let pending = store.pending_embedding_docs(None, model).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].hash, hash);
        assert!(pending[0].bytes > 0);

        store.ensure_vec_table(4).unwrap();
        let now = crate::store::now_iso();
        store
            .insert_embedding(&hash, 0, 0, &unit(4, 0), model, &now, 2, &fingerprint)
            .unwrap();
        assert_eq!(store.pending_embedding_docs(None, model).unwrap().len(), 1);

        store
            .insert_embedding(&hash, 1, 5, &unit(4, 1), model, &now, 2, &fingerprint)
            .unwrap();
        assert!(
            store
                .pending_embedding_docs(None, model)
                .unwrap()
                .is_empty()
        );

        assert_eq!(store.pending_embedding_docs(None, "m2").unwrap().len(), 1);
    }

    #[test]
    fn remove_incomplete_embeddings_drops_partial_sets() {
        let (store, hash) = store_with_doc("c", "a.md", "doc body");
        store.ensure_vec_table(4).unwrap();
        let now = crate::store::now_iso();
        store
            .insert_embedding(&hash, 0, 0, &unit(4, 0), "m", &now, 3, "fp")
            .unwrap();

        let mut expected = HashMap::new();
        expected.insert(hash.clone(), 3usize);
        let removed = store.remove_incomplete_embeddings(&expected, "m").unwrap();
        assert_eq!(removed, 1);
        let n: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn clear_embeddings_globally_and_per_collection() {
        let (store, hash_a) = store_with_doc("c1", "a.md", "body a");
        let hash_b = insert_doc(&store, "c2", "b.md", "body b");
        insert_doc(&store, "c2", "shared.md", "body a");
        store.ensure_vec_table(4).unwrap();
        let now = crate::store::now_iso();
        store
            .insert_embedding(&hash_a, 0, 0, &unit(4, 0), "m", &now, 1, "fp")
            .unwrap();
        store
            .insert_embedding(&hash_b, 0, 0, &unit(4, 1), "m", &now, 1, "fp")
            .unwrap();

        store.clear_all_embeddings(Some("c1")).unwrap();
        let n: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 2, "shared hash must survive scoped clear");

        store.clear_all_embeddings(Some("c2")).unwrap();
        let remaining: Vec<String> = {
            let mut stmt = store
                .conn()
                .prepare("SELECT hash FROM content_vectors")
                .unwrap();
            let rows = stmt.query_map([], |r| r.get(0)).unwrap();
            rows.collect::<std::result::Result<_, _>>().unwrap()
        };
        assert_eq!(remaining, vec![hash_a.clone()]);

        store.clear_all_embeddings(None).unwrap();
        assert!(!store.vec_table_exists().unwrap());
    }
}
