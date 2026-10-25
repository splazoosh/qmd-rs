//! SQLite store: connection management, schema, documents, FTS search.
//!
//! Mirrors the public surface of TS `store.ts` for the ported commands. The
//! on-disk layout is byte-compatible with TS qmd v2.6.3.

mod collections;
mod docs;
mod fts;
mod schema;
mod vectors;

pub use collections::CollectionStats;
pub use docs::{
    DocumentLookup, DocumentResult, LsEntry, docid, extract_title, hash_content, is_docid,
    normalize_docid,
};
pub use fts::{
    SearchResult, TermMatch, build_fts5_query, build_fts5_query_with, sanitize_fts5_term,
};
pub use vectors::PendingEmbeddingDoc;

use std::path::{Path, PathBuf};
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use rusqlite::Connection;

use crate::error::{Error, Result};

/// `PRAGMA user_version` value guarding trigger (re)application.
pub const STORE_SCHEMA_VERSION: i64 = 1;

/// Default `busy_timeout` in milliseconds (`QMD_SQLITE_BUSY_TIMEOUT` overrides).
const DEFAULT_BUSY_TIMEOUT_MS: i64 = 120_000;

static PRODUCTION_MODE: AtomicBool = AtomicBool::new(false);

/// Allow [`default_db_path`] to resolve the real `~/.cache/qmd` index.
///
/// Ported safety rail from TS: tests never touch the global index because the
/// default path resolution fails unless the binary opted in.
pub fn enable_production_mode() {
    PRODUCTION_MODE.store(true, Ordering::Relaxed);
}

/// Resolve the default database path for an index name.
///
/// `INDEX_PATH` always wins (test override). Otherwise requires
/// [`enable_production_mode`] to have been called.
///
/// # Errors
/// Returns [`Error::DbPathNotSet`] outside production mode.
pub fn default_db_path(index_name: &str) -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("INDEX_PATH") {
        return Ok(PathBuf::from(path));
    }
    if !PRODUCTION_MODE.load(Ordering::Relaxed) {
        return Err(Error::DbPathNotSet);
    }
    let dir = crate::config::cache_dir();
    std::fs::create_dir_all(&dir)?;
    Ok(dir.join(format!("{index_name}.sqlite")))
}

/// Register sqlite-vec as an auto extension so every new connection loads it.
fn register_sqlite_vec() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        type SqliteExtensionInit = unsafe extern "C" fn(
            *mut rusqlite::ffi::sqlite3,
            *mut *mut std::ffi::c_char,
            *const rusqlite::ffi::sqlite3_api_routines,
        ) -> std::ffi::c_int;
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
                *const (),
                SqliteExtensionInit,
            >(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

/// An open qmd index database.
pub struct Store {
    conn: Connection,
    vec_available: bool,
}

impl Store {
    /// Open (or create) a store at `path`, applying pragmas and schema.
    pub fn open(path: &Path) -> Result<Self> {
        register_sqlite_vec();
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    /// Open an in-memory store (tests).
    pub fn open_in_memory() -> Result<Self> {
        register_sqlite_vec();
        let conn = Connection::open_in_memory()?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self> {
        let busy_timeout = std::env::var("QMD_SQLITE_BUSY_TIMEOUT")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .filter(|v| *v >= 0)
            .unwrap_or(DEFAULT_BUSY_TIMEOUT_MS);
        conn.pragma_update(None, "busy_timeout", busy_timeout)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;

        let vec_available = conn
            .query_row("SELECT vec_version()", [], |r| r.get::<_, String>(0))
            .is_ok();

        schema::initialize(&conn)?;
        Ok(Self {
            conn,
            vec_available,
        })
    }

    /// Whether the sqlite-vec extension is loaded (vector search available).
    pub fn vec_available(&self) -> bool {
        self.vec_available
    }

    /// Raw connection access for advanced callers and tests.
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

/// Current time in the ISO-8601 format the TS implementation writes
/// (`Date.toISOString()`: millisecond precision, `Z` suffix).
pub fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Build a `qmd://collection/path` virtual path.
pub fn build_virtual_path(collection: &str, path: &str) -> String {
    format!("qmd://{collection}/{path}")
}

/// Parse a `qmd://collection/path` virtual path into (collection, path).
pub fn parse_virtual_path(input: &str) -> Option<(String, String)> {
    let rest = input.strip_prefix("qmd://")?;
    let rest = rest.trim_start_matches('/');
    let (collection, path) = rest.split_once('/')?;
    if collection.is_empty() || path.is_empty() {
        return None;
    }
    Some((collection.to_owned(), path.to_owned()))
}

/// Normalize path separators to forward slashes.
pub fn normalize_path_separators(path: &str) -> String {
    path.replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_path_round_trip() {
        let vp = build_virtual_path("notes", "a/b.md");
        assert_eq!(vp, "qmd://notes/a/b.md");
        assert_eq!(
            parse_virtual_path(&vp),
            Some(("notes".to_owned(), "a/b.md".to_owned()))
        );
        assert_eq!(parse_virtual_path("not-a-virtual-path"), None);
        assert_eq!(parse_virtual_path("qmd://only-collection"), None);
    }

    #[test]
    fn store_opens_with_vec_and_fts() {
        let store = Store::open_in_memory().unwrap();
        assert!(store.vec_available());
        let n: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'documents_fts'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
        let v: i64 = store
            .conn()
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(v, STORE_SCHEMA_VERSION);
    }
}
