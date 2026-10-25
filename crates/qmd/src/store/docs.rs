//! Documents/content CRUD, identity helpers (hash, docid, title extraction),
//! and document lookup for `get`/`ls`. Ported from TS `store.ts`.

use std::collections::HashSet;
use std::sync::LazyLock;

use globset::GlobBuilder;
use regex::Regex;
use rusqlite::{OptionalExtension, params};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::store::Store;

/// SHA-256 of the file body, hex-encoded (TS `hashContent`).
pub fn hash_content(content: &str) -> String {
    hex::encode(Sha256::digest(content.as_bytes()))
}

/// First 6 characters of the content hash (TS `getDocid`).
pub fn docid(hash: &str) -> String {
    hash.chars().take(6).collect()
}

/// Strip surrounding quotes and a leading `#` from a docid reference.
pub fn normalize_docid(input: &str) -> String {
    let mut s = input.trim();
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        s = &s[1..s.len() - 1];
    }
    s.strip_prefix('#').unwrap_or(s).to_owned()
}

/// Whether the input looks like a docid reference (6+ hex chars after
/// normalization).
pub fn is_docid(input: &str) -> bool {
    let normalized = normalize_docid(input);
    normalized.len() >= 6 && normalized.chars().all(|c| c.is_ascii_hexdigit())
}

static MD_HEADING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^##?\s+(.+)$").expect("valid regex"));
static MD_H2: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^##\s+(.+)$").expect("valid regex"));
static ORG_TITLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?mi)^#\+TITLE:\s*(.+)$").expect("valid regex"));
static ORG_HEADING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\*+\s+(.+)$").expect("valid regex"));

/// Extract a document title: first `#`/`##` heading for markdown (with the
/// "Notes" special case), `#+TITLE:` or first heading for org, otherwise the
/// filename without extension.
pub fn extract_title(content: &str, filename: &str) -> String {
    let ext = filename
        .rfind('.')
        .map(|i| filename[i..].to_ascii_lowercase())
        .unwrap_or_default();

    let extracted = match ext.as_str() {
        ".md" => MD_HEADING.captures(content).and_then(|m| {
            let title = m[1].trim().to_owned();
            if title == "📝 Notes" || title == "Notes" {
                MD_H2
                    .captures(content)
                    .map(|n| n[1].trim().to_owned())
                    .or(Some(title))
            } else {
                Some(title)
            }
        }),
        ".org" => ORG_TITLE
            .captures(content)
            .or_else(|| ORG_HEADING.captures(content))
            .map(|m| m[1].trim().to_owned()),
        _ => None,
    };

    extracted.unwrap_or_else(|| {
        let base = filename.rsplit('/').next().unwrap_or(filename);
        let stem = base.rfind('.').map(|i| &base[..i]).unwrap_or(base);
        if stem.is_empty() {
            filename.to_owned()
        } else {
            stem.to_owned()
        }
    })
}

/// The `SELECT` prefix every `get`-style lookup shares. Column order is the
/// one [`map_doc_row`] reads.
fn doc_select(include_body: bool) -> String {
    let body_col = if include_body {
        ", content.doc as body"
    } else {
        ", NULL as body"
    };
    format!(
        "SELECT
           'qmd://' || d.collection || '/' || d.path,
           d.collection || '/' || d.path,
           d.title, d.hash, d.collection, d.modified_at,
           LENGTH(content.doc){body_col}
         FROM documents d
         JOIN content ON content.hash = d.hash"
    )
}

/// Map a row of [`doc_select`]. `context` is filled in by the caller, which
/// knows whether the lookup wants hierarchical folder context.
fn map_doc_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<DocumentResult> {
    let hash: String = r.get(3)?;
    Ok(DocumentResult {
        filepath: r.get(0)?,
        display_path: r.get(1)?,
        title: r.get(2)?,
        docid: docid(&hash),
        hash,
        collection_name: r.get(4)?,
        modified_at: r.get(5)?,
        body_length: r.get(6)?,
        body: r.get(7)?,
        context: None,
    })
}

/// A document row as returned by `get`-style lookups.
#[derive(Debug, Clone)]
pub struct DocumentResult {
    /// Virtual path `qmd://collection/path`.
    pub filepath: String,
    /// `collection/path` display form.
    pub display_path: String,
    pub title: String,
    pub context: Option<String>,
    pub hash: String,
    pub docid: String,
    pub collection_name: String,
    pub modified_at: String,
    pub body_length: i64,
    pub body: Option<String>,
}

/// Outcome of a `get` lookup.
#[derive(Debug)]
pub enum DocumentLookup {
    Found(Box<DocumentResult>),
    NotFound { similar: Vec<String> },
}

/// One row of `ls` output.
#[derive(Debug, Clone)]
pub struct LsEntry {
    pub path: String,
    pub title: String,
    pub modified_at: String,
    pub size: i64,
}

impl Store {
    /// Insert content (content-addressable; duplicate hashes are skipped).
    pub fn insert_content(&self, hash: &str, content: &str, created_at: &str) -> Result<()> {
        self.conn().execute(
            "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?1, ?2, ?3)",
            params![hash, content, created_at],
        )?;
        Ok(())
    }

    /// Insert a new document (or reactivate/update on `(collection, path)`
    /// conflict). FTS stays in sync via triggers.
    pub fn insert_document(
        &self,
        collection: &str,
        path: &str,
        title: &str,
        hash: &str,
        created_at: &str,
        modified_at: &str,
    ) -> Result<()> {
        self.conn().execute(
            "INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1)
             ON CONFLICT(collection, path) DO UPDATE SET
               title = excluded.title,
               hash = excluded.hash,
               modified_at = excluded.modified_at,
               active = 1",
            params![collection, path, title, hash, created_at, modified_at],
        )?;
        Ok(())
    }

    /// Find an active document by collection and path.
    pub fn find_active_document(
        &self,
        collection: &str,
        path: &str,
    ) -> Result<Option<(i64, String, String)>> {
        Ok(self
            .conn()
            .query_row(
                "SELECT id, hash, title FROM documents
                 WHERE collection = ?1 AND path = ?2 AND active = 1",
                params![collection, path],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?)
    }

    /// Find an active document, falling back to a case-insensitive path match
    /// (renames the row in place when found under different casing). A row
    /// whose own path is in `walked` still has its file, so it is never taken:
    /// on a case-sensitive filesystem `README.md` and `readme.md` are two
    /// documents.
    pub fn find_or_migrate_document(
        &self,
        collection: &str,
        path: &str,
        walked: &HashSet<&str>,
    ) -> Result<Option<(i64, String, String)>> {
        if let Some(found) = self.find_active_document(collection, path)? {
            return Ok(Some(found));
        }
        let legacy: Option<i64> = {
            let mut stmt = self.conn().prepare(
                "SELECT id, path FROM documents
                 WHERE collection = ?1 AND path COLLATE NOCASE = ?2 AND active = 1
                 ORDER BY id",
            )?;
            let rows = stmt.query_map(params![collection, path], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })?;
            let mut found = None;
            for row in rows {
                let (id, legacy_path) = row?;
                if !walked.contains(legacy_path.as_str()) {
                    found = Some(id);
                    break;
                }
            }
            found
        };
        let Some(id) = legacy else { return Ok(None) };
        let changed = self.conn().execute(
            "UPDATE OR IGNORE documents SET path = ?1 WHERE id = ?2 AND active = 1",
            params![path, id],
        )?;
        if changed == 0 {
            return Ok(None);
        }
        self.find_active_document(collection, path)
    }

    /// Update title and `modified_at` for a document.
    pub fn update_document_title(&self, id: i64, title: &str, modified_at: &str) -> Result<()> {
        self.conn().execute(
            "UPDATE documents SET title = ?1, modified_at = ?2 WHERE id = ?3",
            params![title, modified_at, id],
        )?;
        Ok(())
    }

    /// Update hash, title, and `modified_at` when content changed in place.
    pub fn update_document(
        &self,
        id: i64,
        title: &str,
        hash: &str,
        modified_at: &str,
    ) -> Result<()> {
        self.conn().execute(
            "UPDATE documents SET title = ?1, hash = ?2, modified_at = ?3 WHERE id = ?4",
            params![title, hash, modified_at, id],
        )?;
        Ok(())
    }

    /// Mark a document inactive (soft delete).
    pub fn deactivate_document(&self, collection: &str, path: &str) -> Result<()> {
        self.conn().execute(
            "UPDATE documents SET active = 0 WHERE collection = ?1 AND path = ?2 AND active = 1",
            params![collection, path],
        )?;
        Ok(())
    }

    /// All active document paths in a collection.
    pub fn active_document_paths(&self, collection: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn()
            .prepare("SELECT path FROM documents WHERE collection = ?1 AND active = 1")?;
        let rows = stmt.query_map(params![collection], |r| r.get(0))?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// All active documents in a collection as (path, content hash) pairs.
    pub fn active_document_hashes(&self, collection: &str) -> Result<Vec<(String, String)>> {
        let mut stmt = self
            .conn()
            .prepare("SELECT path, hash FROM documents WHERE collection = ?1 AND active = 1")?;
        let rows = stmt.query_map(params![collection], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// Delete content rows not referenced by any document (active or not).
    pub fn cleanup_orphaned_content(&self) -> Result<usize> {
        Ok(self.conn().execute(
            "DELETE FROM content WHERE hash NOT IN (SELECT DISTINCT hash FROM documents)",
            [],
        )?)
    }

    /// Find a document's virtual path by short docid prefix.
    pub fn find_document_by_docid(&self, docid_ref: &str) -> Result<Option<String>> {
        let short = normalize_docid(docid_ref);
        if short.is_empty() {
            return Ok(None);
        }
        let pattern = format!("{short}%");
        Ok(self
            .conn()
            .query_row(
                "SELECT 'qmd://' || d.collection || '/' || d.path
                 FROM documents d
                 WHERE d.hash LIKE ?1 AND d.active = 1
                 LIMIT 1",
                params![pattern],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// Active document paths within edit distance of `query` (for "did you
    /// mean" suggestions).
    pub fn find_similar_files(
        &self,
        query: &str,
        max_distance: usize,
        limit: usize,
    ) -> Result<Vec<String>> {
        let mut stmt = self
            .conn()
            .prepare("SELECT path FROM documents WHERE active = 1")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let query_lower = query.to_lowercase();
        let mut scored: Vec<(usize, String)> = Vec::new();
        for row in rows {
            let path = row?;
            let dist = levenshtein(&path.to_lowercase(), &query_lower);
            if dist <= max_distance {
                scored.push((dist, path));
            }
        }
        scored.sort_by_key(|a| a.0);
        scored.truncate(limit);
        Ok(scored.into_iter().map(|(_, p)| p).collect())
    }

    /// Find a document by virtual path, path suffix, collection-relative
    /// path, or docid. Ported from TS `findDocument`.
    pub fn find_document(&self, query: &str, include_body: bool) -> Result<DocumentLookup> {
        let mut filepath = match query.rfind(':') {
            Some(i)
                if query[i + 1..].chars().all(|c| c.is_ascii_digit())
                    && !query[i + 1..].is_empty() =>
            {
                &query[..i]
            }
            _ => query,
        }
        .to_owned();
        filepath = crate::store::normalize_path_separators(&filepath);

        if is_docid(&filepath) {
            match self.find_document_by_docid(&filepath)? {
                Some(virtual_path) => filepath = virtual_path,
                None => return Ok(DocumentLookup::NotFound { similar: vec![] }),
            }
        }

        let select = doc_select(include_body);
        let map_row = map_doc_row;

        let mut doc: Option<DocumentResult> = self
            .conn()
            .query_row(
                &format!(
                    "{select} WHERE 'qmd://' || d.collection || '/' || d.path = ?1 AND d.active = 1"
                ),
                params![filepath],
                map_row,
            )
            .optional()?;

        if doc.is_none() && !filepath.starts_with("qmd://") {
            let absolute = looks_absolute(&filepath);
            for named in self.store_collections()? {
                let coll_path = crate::store::normalize_path_separators(&named.collection.path);
                let coll_prefix = format!("{}/", coll_path.trim_end_matches('/'));
                let relative = if let Some(rel) = filepath.strip_prefix(&coll_prefix) {
                    Some(rel.to_owned())
                } else if !absolute {
                    Some(filepath.clone())
                } else {
                    None
                };
                if let Some(rel) = relative {
                    doc = self
                        .conn()
                        .query_row(
                            &format!(
                                "{select} WHERE d.collection = ?1 AND d.path = ?2 AND d.active = 1"
                            ),
                            params![named.name, rel],
                            map_row,
                        )
                        .optional()?;
                    if doc.is_some() {
                        break;
                    }
                }
            }
        }

        if doc.is_none() {
            doc = self
                .conn()
                .query_row(
                    &format!(
                        "{select} WHERE 'qmd://' || d.collection || '/' || d.path LIKE ?1 ESCAPE '\\' \
                         AND d.active = 1 \
                         ORDER BY LENGTH(d.path), d.collection, d.path LIMIT 1"
                    ),
                    params![format!("%/{}", escape_like(&filepath))],
                    map_row,
                )
                .optional()?;
        }

        match doc {
            Some(mut d) => {
                d.context = self.context_for_file(&d.filepath)?;
                Ok(DocumentLookup::Found(Box::new(d)))
            }
            None => Ok(DocumentLookup::NotFound {
                similar: self.find_similar_files(&filepath, 5, 5)?,
            }),
        }
    }

    /// Active documents whose path matches `pattern`, ordered by path.
    ///
    /// The pattern is tried against three forms of each document, as TS
    /// `matchFilesByGlob` does: the `qmd://` virtual path, the
    /// collection-relative path, and `collection/path`. `*` does not cross a
    /// `/`; `**` does.
    ///
    /// # Errors
    /// Fails on an invalid glob, and propagates store failures.
    pub fn glob_documents(&self, pattern: &str) -> Result<Vec<DocumentResult>> {
        let glob = GlobBuilder::new(pattern)
            .literal_separator(true)
            .build()
            .map_err(|e| Error::Config(format!("invalid glob pattern '{pattern}': {e}")))?
            .compile_matcher();
        let sql = format!(
            "{} WHERE d.active = 1 ORDER BY d.collection, d.path",
            doc_select(true)
        );
        let mut stmt = self.conn().prepare(&sql)?;
        let rows = stmt.query_map([], map_doc_row)?;
        let mut matched = Vec::new();
        for row in rows {
            let doc = row?;
            let relative = doc
                .display_path
                .split_once('/')
                .map_or("", |(_, rest)| rest);
            if glob.is_match(&doc.filepath)
                || glob.is_match(relative)
                || glob.is_match(&doc.display_path)
            {
                matched.push(doc);
            }
        }
        for doc in &mut matched {
            doc.context = self.context_for_file(&doc.filepath)?;
        }
        Ok(matched)
    }

    /// List active files in a collection, optionally under a path prefix.
    pub fn ls_entries(&self, collection: &str, path_prefix: Option<&str>) -> Result<Vec<LsEntry>> {
        let sql = "SELECT d.path, d.title, d.modified_at, LENGTH(ct.doc)
                   FROM documents d
                   JOIN content ct ON d.hash = ct.hash
                   WHERE d.collection = ?1 AND d.path LIKE ?2 ESCAPE '\\' AND d.active = 1
                   ORDER BY d.path";
        let like = format!("{}%", escape_like(path_prefix.unwrap_or("")));
        let mut stmt = self.conn().prepare(sql)?;
        let rows = stmt.query_map(params![collection, like], |r| {
            Ok(LsEntry {
                path: r.get(0)?,
                title: r.get(1)?,
                modified_at: r.get(2)?,
                size: r.get(3)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// Count of active documents in a collection.
    pub fn active_document_count(&self, collection: &str) -> Result<i64> {
        Ok(self.conn().query_row(
            "SELECT COUNT(*) FROM documents WHERE collection = ?1 AND active = 1",
            params![collection],
            |r| r.get(0),
        )?)
    }
}

/// Escape `%`, `_` and `\` for a `LIKE ... ESCAPE '\'` pattern.
fn escape_like(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(c, '%' | '_' | '\\') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Absolute on this platform, rooted at `/`, or starting with a drive letter
/// and a colon and a slash, so a path from either platform is not read as
/// relative.
fn looks_absolute(path: &str) -> bool {
    let b = path.as_bytes();
    std::path::Path::new(path).is_absolute()
        || path.starts_with('/')
        || (b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && b[2] == b'/')
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_sha256_hex() {
        assert_eq!(
            hash_content("hello"),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        assert_eq!(docid(&hash_content("hello")), "2cf24d");
    }

    #[test]
    fn docid_normalization() {
        assert_eq!(normalize_docid("#abc123"), "abc123");
        assert_eq!(normalize_docid("\"#abc123\""), "abc123");
        assert_eq!(normalize_docid("'abc123'"), "abc123");
        assert!(is_docid("#abc123"));
        assert!(is_docid("deadbeef"));
        assert!(!is_docid("abc12"));
        assert!(!is_docid("not-hex"));
    }

    #[test]
    fn title_extraction() {
        assert_eq!(extract_title("# Hello World\nbody", "x.md"), "Hello World");
        assert_eq!(extract_title("## Sub Heading\nbody", "x.md"), "Sub Heading");
        assert_eq!(
            extract_title("no heading here", "docs/my-file.md"),
            "my-file"
        );
        assert_eq!(
            extract_title("# Notes\n\n## Real Title\n", "x.md"),
            "Real Title"
        );
        assert_eq!(extract_title("#+TITLE: Org Doc\n* h\n", "x.org"), "Org Doc");
        assert_eq!(extract_title("* First Heading\n", "x.org"), "First Heading");
        assert_eq!(extract_title("plain text", "readme.txt"), "readme");
    }

    fn add_doc(store: &Store, collection: &str, path: &str) {
        let body = format!("# {path}\n{collection}\n");
        let hash = hash_content(&body);
        store.insert_content(&hash, &body, "t").unwrap();
        store
            .insert_document(collection, path, path, &hash, "t", "t")
            .unwrap();
    }

    fn found(store: &Store, query: &str) -> Option<String> {
        match store.find_document(query, false).unwrap() {
            DocumentLookup::Found(d) => Some(d.display_path),
            DocumentLookup::NotFound { .. } => None,
        }
    }

    #[test]
    fn find_document_prefers_exact_and_anchors_suffix() {
        let store = Store::open_in_memory().unwrap();
        let coll = crate::config::Collection {
            path: "/root/notes".to_owned(),
            pattern: "**/*.md".to_owned(),
            ignore: None,
            context: None,
            update: None,
            include_by_default: None,
            chunk_tokens: None,
            chunk_boundary: None,
        };
        store.upsert_store_collection("c", &coll).unwrap();
        add_doc(&store, "c", "x/sub/a.md");
        add_doc(&store, "c", "sub/a.md");
        add_doc(&store, "c", "my_a.md");

        assert_eq!(found(&store, "sub/a.md").as_deref(), Some("c/sub/a.md"));
        assert_eq!(found(&store, "sub\\a.md").as_deref(), Some("c/sub/a.md"));
        assert_eq!(
            found(&store, "\\root\\notes\\sub\\a.md").as_deref(),
            Some("c/sub/a.md")
        );
        assert_eq!(
            found(&store, "c/x/sub/a.md").as_deref(),
            Some("c/x/sub/a.md")
        );
        assert_eq!(found(&store, "a.md").as_deref(), Some("c/sub/a.md"));
        assert_eq!(found(&store, "y_a.md"), None);
        assert_eq!(found(&store, "my_a.md").as_deref(), Some("c/my_a.md"));
    }

    #[test]
    fn ls_prefix_is_literal() {
        let store = Store::open_in_memory().unwrap();
        add_doc(&store, "c", "a_b/x.md");
        add_doc(&store, "c", "aXb/y.md");
        let paths: Vec<String> = store
            .ls_entries("c", Some("a_b/"))
            .unwrap()
            .into_iter()
            .map(|e| e.path)
            .collect();
        assert_eq!(paths, vec!["a_b/x.md"]);
    }

    #[test]
    fn case_migration_skips_rows_whose_file_still_exists() {
        let store = Store::open_in_memory().unwrap();
        add_doc(&store, "c", "README.md");
        let both: HashSet<&str> = ["README.md", "readme.md"].into_iter().collect();
        assert!(
            store
                .find_or_migrate_document("c", "readme.md", &both)
                .unwrap()
                .is_none()
        );
        assert!(
            store
                .find_active_document("c", "README.md")
                .unwrap()
                .is_some()
        );

        let renamed: HashSet<&str> = ["readme.md"].into_iter().collect();
        assert!(
            store
                .find_or_migrate_document("c", "readme.md", &renamed)
                .unwrap()
                .is_some()
        );
        assert!(
            store
                .find_active_document("c", "README.md")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn levenshtein_basics() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
    }
}
