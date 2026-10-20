//! Append-only log of the queries this index is asked.
//!
//! One JSON object per line next to the index database, recording the query
//! text, the arm that answered it, the surface it came through, and the paths
//! returned — a collection point for a later evaluation, deliberately without
//! scoring or analysis. Logging never fails a query: a log that cannot be
//! written is silently skipped.
//!
//! `QMD_QUERY_LOG=0` (or `off`/`false`) turns it off. Past [`ROTATE_BYTES`] the
//! file moves to `<name>.1`, replacing the previous one, so at most two files
//! of that size are kept.

use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Size at which the log is rotated.
pub const ROTATE_BYTES: u64 = 10 * 1024 * 1024;

/// Which retrieval arm answered: `search`, `vsearch`, `query` (plain text), or
/// the MCP `searches` form (typed sub-queries).
#[derive(Clone, Copy)]
pub enum Arm {
    Bm25,
    Vector,
    Hybrid,
    Structured,
}

impl Arm {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bm25 => "bm25",
            Self::Vector => "vector",
            Self::Hybrid => "hybrid",
            Self::Structured => "structured",
        }
    }
}

/// The surface a query came through.
#[derive(Clone, Copy)]
pub enum Via {
    Cli,
    Mcp,
}

impl Via {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::Mcp => "mcp",
        }
    }
}

/// The log file for an index database: `<stem>-queries.jsonl` beside it.
pub fn log_path(db_path: &Path) -> PathBuf {
    let stem = db_path
        .file_stem()
        .map_or_else(|| "index".to_owned(), |s| s.to_string_lossy().into_owned());
    db_path.with_file_name(format!("{stem}-queries.jsonl"))
}

/// Whether `QMD_QUERY_LOG` leaves logging on.
pub fn enabled() -> bool {
    enabled_by(std::env::var("QMD_QUERY_LOG").ok().as_deref())
}

fn enabled_by(value: Option<&str>) -> bool {
    !value.is_some_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        )
    })
}

/// Append one entry, unless logging is off. Errors are swallowed by design.
pub fn append(db_path: &Path, arm: Arm, via: Via, query: &str, paths: &[String]) {
    if enabled() {
        write_entry(db_path, arm, via, query, paths, ROTATE_BYTES);
    }
}

fn write_entry(db_path: &Path, arm: Arm, via: Via, query: &str, paths: &[String], cap: u64) {
    let entry = serde_json::json!({
        "ts": crate::store::now_iso(),
        "arm": arm.as_str(),
        "via": via.as_str(),
        "query": query,
        "results": paths,
    });
    let path = log_path(db_path);
    if std::fs::metadata(&path).is_ok_and(|m| m.len() >= cap) {
        let mut rotated = path.clone().into_os_string();
        rotated.push(".1");
        let _ = std::fs::rename(&path, rotated);
    }
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| f.write_all(format!("{entry}\n").as_bytes()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_one_line_per_query_beside_the_index() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.sqlite");
        let staleness = ["qmd://notes/staleness.md".to_owned()];
        write_entry(
            &db,
            Arm::Hybrid,
            Via::Mcp,
            "how do we handle staleness",
            &staleness,
            ROTATE_BYTES,
        );
        write_entry(&db, Arm::Bm25, Via::Cli, "drift", &[], ROTATE_BYTES);

        let log = log_path(&db);
        assert_eq!(log, dir.path().join("index-queries.jsonl"));
        let lines: Vec<String> = std::fs::read_to_string(&log)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect();
        assert_eq!(lines.len(), 2);
        let first: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(first["arm"], "hybrid");
        assert_eq!(first["via"], "mcp");
        assert_eq!(first["query"], "how do we handle staleness");
        assert_eq!(first["results"][0], "qmd://notes/staleness.md");
        assert!(first["ts"].as_str().is_some_and(|t| t.ends_with('Z')));
    }

    #[test]
    fn a_full_log_rotates_to_one_backup() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("index.sqlite");
        for query in ["one", "two", "three"] {
            write_entry(&db, Arm::Bm25, Via::Cli, query, &[], 1);
        }
        let log = std::fs::read_to_string(log_path(&db)).unwrap();
        let backup = std::fs::read_to_string(dir.path().join("index-queries.jsonl.1")).unwrap();
        assert!(log.contains("three") && log.lines().count() == 1, "{log}");
        assert!(
            backup.contains("two") && !backup.contains("one"),
            "{backup}"
        );
    }

    #[test]
    fn the_switch_turns_logging_off_and_nothing_else_does() {
        for off in ["0", "off", "FALSE", "no"] {
            assert!(!enabled_by(Some(off)), "{off}");
        }
        for on in [None, Some("1"), Some(""), Some("yes")] {
            assert!(enabled_by(on), "{on:?}");
        }
    }
}
