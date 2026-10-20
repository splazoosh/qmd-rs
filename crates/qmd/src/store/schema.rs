//! Schema DDL and FTS sync triggers, ported verbatim from TS
//! `initializeDatabase()` / `applyFtsSyncTriggers()`.

use rusqlite::Connection;

use crate::error::Result;
use crate::store::STORE_SCHEMA_VERSION;

pub fn initialize(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "DROP TABLE IF EXISTS path_contexts;
         DROP TABLE IF EXISTS collections;",
    )?;

    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS content (
          hash TEXT PRIMARY KEY,
          doc TEXT NOT NULL,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          collection TEXT NOT NULL,
          path TEXT NOT NULL,
          title TEXT NOT NULL,
          hash TEXT NOT NULL,
          created_at TEXT NOT NULL,
          modified_at TEXT NOT NULL,
          active INTEGER NOT NULL DEFAULT 1,
          FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE,
          UNIQUE(collection, path)
        );
        CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection, active);
        CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
        CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path, active);

        CREATE TABLE IF NOT EXISTS llm_cache (
          hash TEXT PRIMARY KEY,
          result TEXT NOT NULL,
          created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS content_vectors (
          hash TEXT NOT NULL,
          seq INTEGER NOT NULL DEFAULT 0,
          pos INTEGER NOT NULL DEFAULT 0,
          model TEXT NOT NULL,
          embed_fingerprint TEXT NOT NULL DEFAULT '',
          total_chunks INTEGER NOT NULL DEFAULT 1,
          embedded_at TEXT NOT NULL,
          PRIMARY KEY (hash, seq)
        );

        CREATE TABLE IF NOT EXISTS store_collections (
          name TEXT PRIMARY KEY,
          path TEXT NOT NULL,
          pattern TEXT NOT NULL DEFAULT '**/*.md',
          ignore_patterns TEXT,
          include_by_default INTEGER DEFAULT 1,
          update_command TEXT,
          context TEXT
        );

        CREATE TABLE IF NOT EXISTS store_config (
          key TEXT PRIMARY KEY,
          value TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
          filepath, title, body,
          tokenize='porter unicode61'
        );
        "#,
    )?;

    apply_fts_sync_triggers(conn)?;
    Ok(())
}

fn user_version(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("PRAGMA user_version", [], |r| r.get(0))?)
}

/// Apply FTS sync triggers once per schema version.
///
/// DROP+CREATE is gated behind `PRAGMA user_version` inside one IMMEDIATE
/// transaction so concurrent first-time opens do not interleave the pair
/// (same rationale as the TS implementation).
fn apply_fts_sync_triggers(conn: &Connection) -> Result<()> {
    if user_version(conn)? >= STORE_SCHEMA_VERSION {
        return Ok(());
    }
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let result = (|| -> Result<()> {
        if user_version(conn)? < STORE_SCHEMA_VERSION {
            conn.execute_batch(
                r#"
                DROP TRIGGER IF EXISTS documents_ai;
                DROP TRIGGER IF EXISTS documents_ad;
                DROP TRIGGER IF EXISTS documents_au;

                CREATE TRIGGER documents_ai AFTER INSERT ON documents
                WHEN new.active = 1
                BEGIN
                  INSERT INTO documents_fts(rowid, filepath, title, body)
                  SELECT
                    new.id,
                    new.collection || '/' || new.path,
                    new.title,
                    (SELECT doc FROM content WHERE hash = new.hash)
                  WHERE new.active = 1;
                END;

                CREATE TRIGGER documents_ad AFTER DELETE ON documents BEGIN
                  DELETE FROM documents_fts WHERE rowid = old.id;
                END;

                CREATE TRIGGER documents_au AFTER UPDATE ON documents
                BEGIN
                  DELETE FROM documents_fts WHERE rowid = old.id AND new.active = 0;

                  INSERT OR REPLACE INTO documents_fts(rowid, filepath, title, body)
                  SELECT
                    new.id,
                    new.collection || '/' || new.path,
                    new.title,
                    (SELECT doc FROM content WHERE hash = new.hash)
                  WHERE new.active = 1;
                END;
                "#,
            )?;
            conn.pragma_update(None, "user_version", STORE_SCHEMA_VERSION)?;
        }
        Ok(())
    })();
    match result {
        Ok(()) => {
            conn.execute_batch("COMMIT")?;
            Ok(())
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}
