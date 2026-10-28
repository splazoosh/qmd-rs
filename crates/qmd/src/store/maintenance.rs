//! `qmd cleanup`, ported from TS `store.ts` (`previewCleanup`, `runCleanup`).

use rusqlite::params;

use crate::error::Result;
use crate::store::Store;

/// What a cleanup removes, or would remove.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CleanupStats {
    /// Rows of `llm_cache` (cached rerank scores).
    pub cached_scores: usize,
    /// Embedded chunks whose content no active document holds.
    pub orphaned_vectors: usize,
    /// Documents marked inactive (gone from disk).
    pub inactive_docs: usize,
    /// Content rows no document references once the inactive ones are gone.
    pub orphaned_content: usize,
}

/// Embedded chunks of content that no active document holds.
const ORPHANED_VECTORS: &str = "FROM content_vectors AS cv
     WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1)";

impl Store {
    /// What [`Self::cleanup`] would remove, without writing.
    pub fn preview_cleanup(&self) -> Result<CleanupStats> {
        let count = |sql: &str| -> Result<usize> {
            let n: i64 = self.conn().query_row(sql, [], |r| r.get(0))?;
            Ok(usize::try_from(n).unwrap_or(0))
        };
        Ok(CleanupStats {
            cached_scores: count("SELECT COUNT(*) FROM llm_cache")?,
            orphaned_vectors: count(&format!("SELECT COUNT(*) {ORPHANED_VECTORS}"))?,
            inactive_docs: count("SELECT COUNT(*) FROM documents WHERE active = 0")?,
            orphaned_content: count(
                "SELECT COUNT(*) FROM content
                 WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
            )?,
        })
    }

    /// Clear the rerank cache, delete orphaned vectors, inactive documents and
    /// orphaned content in one transaction, then compact FTS5 and `VACUUM`.
    /// Nothing an active document needs is touched, so nothing needs
    /// re-embedding afterwards.
    ///
    /// # Errors
    /// Propagates SQLite failures; `VACUUM` fails while another connection
    /// holds a transaction open past the busy timeout.
    pub fn cleanup(&self) -> Result<CleanupStats> {
        let tx = self.conn().unchecked_transaction()?;
        let cached_scores = tx.execute("DELETE FROM llm_cache", [])?;
        let orphaned_vectors = self.delete_orphaned_vectors_in(&tx)?;
        let inactive_docs = tx.execute("DELETE FROM documents WHERE active = 0", [])?;
        let orphaned_content = tx.execute(
            "DELETE FROM content WHERE hash NOT IN (SELECT DISTINCT hash FROM documents)",
            [],
        )?;
        tx.commit()?;
        self.sync_sections(None)?;

        self.conn()
            .execute_batch("INSERT INTO documents_fts(documents_fts) VALUES('optimize');")?;
        if !self.section_levels()?.is_empty() {
            self.conn()
                .execute_batch("INSERT INTO sections_fts(sections_fts) VALUES('optimize');")?;
        }
        self.conn().execute_batch("VACUUM;")?;
        Ok(CleanupStats {
            cached_scores,
            orphaned_vectors,
            inactive_docs,
            orphaned_content,
        })
    }

    /// Delete the embedded chunks of content no active document holds: what
    /// an edited document's old hash leaves behind. `embed` runs this, so
    /// orphans do not build up between cleanups. Returns the chunks removed.
    ///
    /// # Errors
    /// Propagates SQLite failures.
    pub fn delete_orphaned_vectors(&self) -> Result<usize> {
        let tx = self.conn().unchecked_transaction()?;
        let removed = self.delete_orphaned_vectors_in(&tx)?;
        tx.commit()?;
        Ok(removed)
    }

    fn delete_orphaned_vectors_in(&self, tx: &rusqlite::Transaction<'_>) -> Result<usize> {
        let orphans: Vec<(String, i64)> = {
            let mut stmt = tx.prepare(&format!("SELECT cv.hash, cv.seq {ORPHANED_VECTORS}"))?;
            let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            rows.collect::<std::result::Result<_, _>>()?
        };
        if orphans.is_empty() {
            return Ok(0);
        }
        if self.vec_table_exists()? {
            let mut del = tx.prepare("DELETE FROM vectors_vec WHERE hash_seq = ?1")?;
            for (hash, seq) in &orphans {
                del.execute(params![format!("{hash}_{seq}")])?;
            }
        }
        tx.execute(
            "DELETE FROM content_vectors WHERE NOT EXISTS (SELECT 1 FROM documents d
               WHERE d.hash = content_vectors.hash AND d.active = 1)",
            [],
        )?;
        Ok(orphans.len())
    }
}
