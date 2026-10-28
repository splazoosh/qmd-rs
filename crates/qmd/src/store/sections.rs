//! Section documents: a collection with `sectionLevel` is searched as its
//! sections, not its files ([`crate::chunk::split_sections`]).
//!
//! Each section is a row of `sections` and a row of `sections_fts`, so BM25
//! ranks sections with their own length and term statistics. The file's
//! `documents` row stays (identity, `get`, `ls`, embedding); only its
//! `documents_fts` row is removed while the collection is in section mode,
//! and restored when it leaves it. Both tables are created on the first
//! opt-in, so an index that never opts in keeps the TS layout untouched.

use std::collections::{BTreeSet, HashMap};

use rusqlite::{OptionalExtension, params};
use serde::Serialize;

use crate::chunk::split_sections;
use crate::error::Result;
use crate::store::{SearchResult, Store};

/// `store_config` key prefix of a collection's section level.
pub(crate) const SECTION_LEVEL: &str = "section_level";

/// Which section of its file a result or `get` lookup is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SectionRef {
    /// 1-based ordinal in the file: `qmd://c/a.md#3` is the third.
    pub seq: usize,
    /// The heading's level; 0 for text before the first heading.
    pub level: u8,
    /// The heading's text; empty at level 0.
    pub heading: String,
    /// Byte span in the file.
    pub start: usize,
    pub end: usize,
    /// First and last line in the file (1-based).
    pub start_line: usize,
    pub end_line: usize,
}

/// A section's identity: its file's path (virtual or display form) and
/// `#seq`. `get` accepts it.
pub fn section_id(filepath: &str, seq: usize) -> String {
    format!("{filepath}#{seq}")
}

/// Split `input` into a lookup and a section ordinal when it ends in
/// `#<digits>` after a non-empty path.
pub fn parse_section_id(input: &str) -> Option<(&str, usize)> {
    let (base, seq) = input.rsplit_once('#')?;
    if base.is_empty() || base == "#" || seq.is_empty() || !seq.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    Some((base, seq.parse().ok().filter(|&n| n > 0)?))
}

/// The FTS title of a section: the document title, then the headings that
/// lead to the section, a repeat of the title dropped.
fn section_title(doc_title: &str, trail: &[String]) -> String {
    let mut parts: Vec<&str> = vec![doc_title];
    for t in trail {
        if parts.last() != Some(&t.as_str()) {
            parts.push(t);
        }
    }
    parts.join(" › ")
}

/// `content[start..end]`, or all of it when the span does not fit (a stale
/// row the hash checks should already exclude).
pub(crate) fn slice_section(content: &str, start: usize, end: usize) -> String {
    content.get(start..end).unwrap_or(content).to_owned()
}

const SECTION_COLUMNS: &str =
    "s.seq, s.heading_level, s.heading, s.start_pos, s.end_pos, s.start_line, s.end_line";

fn map_section(r: &rusqlite::Row<'_>, at: usize) -> rusqlite::Result<SectionRef> {
    let n = |i: usize| -> rusqlite::Result<usize> {
        Ok(usize::try_from(r.get::<_, i64>(at + i)?).unwrap_or(0))
    };
    Ok(SectionRef {
        seq: n(0)?,
        level: u8::try_from(r.get::<_, i64>(at + 1)?).unwrap_or(0),
        heading: r.get(at + 2)?,
        start: n(3)?,
        end: n(4)?,
        start_line: n(5)?,
        end_line: n(6)?,
    })
}

impl Store {
    /// A collection's section level, when it is set to section documents.
    pub fn section_level(&self, collection: &str) -> Result<Option<u8>> {
        Ok(self
            .collection_setting(SECTION_LEVEL, collection)?
            .and_then(|v| v.parse().ok()))
    }

    /// Every collection set to section documents, with its level.
    pub fn section_levels(&self) -> Result<HashMap<String, u8>> {
        let mut stmt = self.conn().prepare(
            "SELECT key, value FROM store_config WHERE key LIKE 'section\\_level:%' ESCAPE '\\'",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        let mut levels = HashMap::new();
        for row in rows {
            let (key, value) = row?;
            if let (Some(name), Ok(level)) = (key.split_once(':').map(|k| k.1), value.parse()) {
                levels.insert(name.to_owned(), level);
            }
        }
        Ok(levels)
    }

    fn sections_table_exists(&self) -> Result<bool> {
        let n: i64 = self.conn().query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'sections'",
            [],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    /// Bring section rows in line with the collections' section levels and
    /// documents: drop the rows of documents that changed, went inactive or
    /// left section mode (restoring a file's `documents_fts` row when its
    /// collection left), cut the active documents of section collections
    /// that have none (`collection` only, when given), and remove those
    /// files' `documents_fts` rows, which the TS triggers re-add on every
    /// document update. Returns the number of documents cut.
    ///
    /// # Errors
    /// Propagates SQLite failures.
    pub fn sync_sections(&self, collection: Option<&str>) -> Result<usize> {
        let levels = self.section_levels()?;
        if levels.is_empty() && !self.sections_table_exists()? {
            return Ok(0);
        }
        let tx = if self.conn().is_autocommit() {
            Some(self.conn().unchecked_transaction()?)
        } else {
            None
        };
        self.conn().execute_batch(
            "CREATE TABLE IF NOT EXISTS sections (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               doc_id INTEGER NOT NULL,
               collection TEXT NOT NULL,
               path TEXT NOT NULL,
               hash TEXT NOT NULL,
               level INTEGER NOT NULL,
               seq INTEGER NOT NULL,
               heading_level INTEGER NOT NULL,
               heading TEXT NOT NULL,
               start_pos INTEGER NOT NULL,
               end_pos INTEGER NOT NULL,
               start_line INTEGER NOT NULL,
               end_line INTEGER NOT NULL,
               UNIQUE(doc_id, seq)
             );
             CREATE INDEX IF NOT EXISTS idx_sections_doc ON sections(doc_id, start_pos);
             CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
               filepath, title, body,
               tokenize='porter unicode61'
             );",
        )?;

        let stale: Vec<(i64, i64)> = {
            let mut stmt = self.conn().prepare(
                "SELECT s.id, s.doc_id FROM sections s
                 LEFT JOIN documents d ON d.id = s.doc_id
                 WHERE d.id IS NULL OR d.active = 0 OR d.hash != s.hash
                    OR d.collection != s.collection OR d.path != s.path
                    OR s.level != COALESCE((SELECT CAST(value AS INTEGER) FROM store_config
                                            WHERE key = 'section_level:' || d.collection), 0)",
            )?;
            let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            rows.collect::<std::result::Result<_, _>>()?
        };
        {
            let mut del_fts = self
                .conn()
                .prepare("DELETE FROM sections_fts WHERE rowid = ?1")?;
            let mut del = self.conn().prepare("DELETE FROM sections WHERE id = ?1")?;
            for (id, _) in &stale {
                del_fts.execute(params![id])?;
                del.execute(params![id])?;
            }
            let mut restore = self.conn().prepare(
                "INSERT OR REPLACE INTO documents_fts (rowid, filepath, title, body)
                 SELECT d.id, d.collection || '/' || d.path, d.title, c.doc
                 FROM documents d JOIN content c ON c.hash = d.hash
                 WHERE d.id = ?1 AND d.active = 1 AND NOT EXISTS
                   (SELECT 1 FROM store_config WHERE key = 'section_level:' || d.collection)",
            )?;
            let doc_ids: BTreeSet<i64> = stale.iter().map(|(_, d)| *d).collect();
            for doc_id in doc_ids {
                restore.execute(params![doc_id])?;
            }
        }

        let mut cut = 0;
        for (name, &level) in &levels {
            if collection.is_some_and(|c| c != name) {
                continue;
            }
            cut += self.cut_sections(name, level)?;
            self.conn().execute(
                "DELETE FROM documents_fts WHERE rowid IN
                   (SELECT id FROM documents WHERE collection = ?1)",
                params![name],
            )?;
        }
        if let Some(tx) = tx {
            tx.commit()?;
        }
        Ok(cut)
    }

    /// Cut the active documents of `collection` that have no section rows.
    fn cut_sections(&self, collection: &str, level: u8) -> Result<usize> {
        let docs: Vec<(i64, String, String, String, String)> = {
            let mut stmt = self.conn().prepare(
                "SELECT d.id, d.path, d.title, d.hash, c.doc
                 FROM documents d JOIN content c ON c.hash = d.hash
                 WHERE d.collection = ?1 AND d.active = 1
                   AND NOT EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.id)",
            )?;
            let rows = stmt.query_map(params![collection], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
            })?;
            rows.collect::<std::result::Result<_, _>>()?
        };
        let mut insert = self.conn().prepare(
            "INSERT INTO sections (doc_id, collection, path, hash, level, seq, heading_level,
               heading, start_pos, end_pos, start_line, end_line)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        )?;
        let mut insert_fts = self.conn().prepare(
            "INSERT INTO sections_fts (rowid, filepath, title, body) VALUES (?1, ?2, ?3, ?4)",
        )?;
        for (doc_id, path, title, hash, doc) in &docs {
            let filepath = format!("{collection}/{path}");
            for (i, s) in split_sections(doc, level).iter().enumerate() {
                #[allow(clippy::cast_possible_wrap)]
                insert.execute(params![
                    doc_id,
                    collection,
                    path,
                    hash,
                    level,
                    i as i64 + 1,
                    s.level,
                    s.heading,
                    s.start as i64,
                    s.end as i64,
                    s.start_line as i64,
                    s.end_line as i64,
                ])?;
                let id = self.conn().last_insert_rowid();
                insert_fts.execute(params![
                    id,
                    filepath,
                    section_title(title, &s.trail),
                    &doc[s.start..s.end],
                ])?;
            }
        }
        Ok(docs.len())
    }

    /// BM25 over `sections_fts`: one result per section, its text the body.
    pub(crate) fn search_sections_fts(
        &self,
        fts_query: &str,
        limit: usize,
        collection: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        if !self.sections_table_exists()? {
            return Ok(Vec::new());
        }
        let fts_limit = if collection.is_some() {
            limit.saturating_mul(10)
        } else {
            limit
        };
        let collection_filter = if collection.is_some() {
            " AND d.collection = ?2"
        } else {
            ""
        };
        let sql = format!(
            "WITH fts_matches AS (
               SELECT rowid, bm25(sections_fts, 1.5, 4.0, 1.0) as bm25_score
               FROM sections_fts
               WHERE sections_fts MATCH ?1
               ORDER BY bm25_score ASC
               LIMIT {fts_limit}
             )
             SELECT
               'qmd://' || d.collection || '/' || d.path,
               d.collection || '/' || d.path,
               d.title, content.doc, d.hash, d.collection, fm.bm25_score,
               {SECTION_COLUMNS}
             FROM fts_matches fm
             JOIN sections s ON s.id = fm.rowid
             JOIN documents d ON d.id = s.doc_id AND d.hash = s.hash
             JOIN content ON content.hash = d.hash
             WHERE d.active = 1{collection_filter}
             ORDER BY fm.bm25_score ASC
             LIMIT {limit}"
        );
        let mut stmt = self.conn().prepare(&sql)?;
        let map_row = |r: &rusqlite::Row<'_>| -> rusqlite::Result<SearchResult> {
            let hash: String = r.get(4)?;
            let bm25: f64 = r.get(6)?;
            let doc: String = r.get(3)?;
            let section = map_section(r, 7)?;
            Ok(SearchResult {
                filepath: r.get(0)?,
                display_path: r.get(1)?,
                title: r.get(2)?,
                body: slice_section(&doc, section.start, section.end),
                docid: super::docs::docid(&hash),
                hash,
                collection_name: r.get(5)?,
                context: None,
                score: bm25.abs() / (1.0 + bm25.abs()),
                section: Some(section),
            })
        };
        let rows: Vec<SearchResult> = match collection {
            Some(c) => stmt
                .query_map(params![fts_query, c], map_row)?
                .collect::<std::result::Result<_, _>>()?,
            None => stmt
                .query_map(params![fts_query], map_row)?
                .collect::<std::result::Result<_, _>>()?,
        };
        Ok(rows)
    }

    /// The section of document `doc_id` (content `hash`) holding byte `pos`.
    pub(crate) fn section_containing(
        &self,
        doc_id: i64,
        hash: &str,
        pos: i64,
    ) -> Result<Option<SectionRef>> {
        if !self.sections_table_exists()? {
            return Ok(None);
        }
        Ok(self
            .conn()
            .query_row(
                &format!(
                    "SELECT {SECTION_COLUMNS} FROM sections s
                     WHERE s.doc_id = ?1 AND s.hash = ?2 AND s.start_pos <= ?3
                     ORDER BY s.start_pos DESC LIMIT 1"
                ),
                params![doc_id, hash, pos],
                |r| map_section(r, 0),
            )
            .optional()?)
    }

    /// Section `seq` of the active file `path` in `collection`. A pair, not a
    /// virtual path, because a collection name may hold a `/`.
    pub fn section_of_file(
        &self,
        collection: &str,
        path: &str,
        seq: usize,
    ) -> Result<Option<SectionRef>> {
        if !self.sections_table_exists()? {
            return Ok(None);
        }
        #[allow(clippy::cast_possible_wrap)]
        Ok(self
            .conn()
            .query_row(
                &format!(
                    "SELECT {SECTION_COLUMNS} FROM sections s
                     JOIN documents d ON d.id = s.doc_id AND d.hash = s.hash
                     WHERE d.collection = ?1 AND d.path = ?2
                       AND d.active = 1 AND s.seq = ?3"
                ),
                params![collection, path, seq as i64],
                |r| map_section(r, 0),
            )
            .optional()?)
    }

    /// Section rows and how many documents they cover, per collection.
    pub fn section_counts(&self) -> Result<HashMap<String, (usize, usize)>> {
        if !self.sections_table_exists()? {
            return Ok(HashMap::new());
        }
        let mut stmt = self.conn().prepare(
            "SELECT collection, COUNT(*), COUNT(DISTINCT doc_id) FROM sections GROUP BY collection",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                (
                    usize::try_from(r.get::<_, i64>(1)?).unwrap_or(0),
                    usize::try_from(r.get::<_, i64>(2)?).unwrap_or(0),
                ),
            ))
        })?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Collection;
    use crate::store::hash_content;

    fn add(store: &Store, collection: &str, path: &str, body: &str) {
        let hash = hash_content(body);
        store.insert_content(&hash, body, "t").unwrap();
        store
            .insert_document(collection, path, "Doc", &hash, "t", "t")
            .unwrap();
    }

    fn set_level(store: &Store, collection: &str, level: Option<u8>) {
        let mut c = Collection::new("/x", "**/*.md");
        c.section_level = level;
        store.upsert_store_collection(collection, &c).unwrap();
        store.sync_sections(None).unwrap();
    }

    fn fts_rows(store: &Store, table: &str) -> i64 {
        store
            .conn()
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn section_ids_parse() {
        assert_eq!(
            parse_section_id("qmd://c/a.md#3"),
            Some(("qmd://c/a.md", 3))
        );
        assert_eq!(parse_section_id("#abc123#2"), Some(("#abc123", 2)));
        assert_eq!(parse_section_id("#abc123"), None);
        assert_eq!(parse_section_id("c/a.md#0"), None);
        assert_eq!(parse_section_id("c/a.md#x"), None);
        assert_eq!(
            section_title("Guide", &["Guide".into(), "Install".into()]),
            "Guide › Install"
        );
    }

    #[test]
    fn bm25_ranks_a_dense_short_section_above_a_long_one() {
        let store = Store::open_in_memory().unwrap();
        let filler = "Ordinary prose about unrelated matters goes on here. ".repeat(60);
        let doc = format!(
            "# Manual\n\n## Overview\n\n{filler}Mentions quorum once.\n\n{filler}\n\n\
             ## Quorum\n\nQuorum rules: a quorum is a majority; quorum loss halts writes.\n\n\
             ## Appendix\n\n{filler}\n"
        );
        add(&store, "s", "manual.md", &doc);
        set_level(&store, "s", Some(2));
        assert_eq!(fts_rows(&store, "documents_fts"), 0, "the file row is gone");

        let hits = store
            .search_fts_with("quorum", 10, None, crate::store::TermMatch::Any)
            .unwrap();
        assert_eq!(hits.len(), 2, "two sections of one file both return");
        assert_eq!(hits[0].section.as_ref().unwrap().heading, "Quorum");
        assert_eq!(hits[1].section.as_ref().unwrap().heading, "Overview");
        assert!(hits[0].score > hits[1].score);
        assert!(hits[0].body.starts_with("## Quorum"));
        assert_eq!(
            hits[0].key(),
            "qmd://s/manual.md#2",
            "the title joined Overview"
        );
        let quorum = hits[0].section.as_ref().unwrap();
        assert_eq!(&doc[quorum.start..quorum.end], hits[0].body);
    }

    #[test]
    fn switching_out_restores_file_documents_and_edits_recut() {
        let store = Store::open_in_memory().unwrap();
        add(&store, "s", "a.md", "## One\n\nalpha\n\n## Two\n\nbeta\n");
        add(&store, "f", "b.md", "## Other\n\nalpha\n");
        set_level(&store, "s", Some(2));
        set_level(&store, "f", None);
        assert_eq!(fts_rows(&store, "sections"), 2);
        assert_eq!(fts_rows(&store, "documents_fts"), 1);
        let hits = store.search_fts("alpha", 10, None).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(
            hits.iter()
                .any(|h| h.section.is_none() && h.display_path == "f/b.md")
        );
        assert!(
            hits.iter()
                .any(|h| h.section.as_ref().is_some_and(|s| s.seq == 1))
        );
        assert_eq!(store.search_fts("alpha", 10, Some("f")).unwrap().len(), 1);

        let body = "## One\n\nalpha\n\n## Two\n\nbeta\n\n## Three\n\ngamma\n";
        let hash = hash_content(body);
        store.insert_content(&hash, body, "t").unwrap();
        store
            .conn()
            .execute(
                "UPDATE documents SET hash = ?1 WHERE path = 'a.md'",
                params![hash],
            )
            .unwrap();
        store.sync_sections(Some("s")).unwrap();
        assert_eq!(fts_rows(&store, "sections"), 3);
        assert_eq!(fts_rows(&store, "documents_fts"), 1);

        set_level(&store, "s", None);
        assert_eq!(fts_rows(&store, "sections"), 0);
        assert_eq!(fts_rows(&store, "sections_fts"), 0);
        assert_eq!(fts_rows(&store, "documents_fts"), 2);
        let hits = store.search_fts("gamma", 10, None).unwrap();
        assert_eq!(hits[0].display_path, "s/a.md");
        assert!(hits[0].section.is_none());
    }

    #[test]
    fn an_index_that_never_opts_in_gets_no_section_tables() {
        let store = Store::open_in_memory().unwrap();
        add(&store, "f", "b.md", "## Other\n\nalpha\n");
        set_level(&store, "f", None);
        assert!(!store.sections_table_exists().unwrap());
        assert_eq!(store.section_of_file("f", "b.md", 1).unwrap(), None);
    }

    #[test]
    fn a_real_path_ending_in_hash_digits_is_not_read_as_a_section() {
        use crate::store::DocumentLookup;
        let store = Store::open_in_memory().unwrap();
        add(&store, "s", "log", "## One\n\nalpha\n\n## Two\n\nbeta\n");
        add(&store, "f", "log#2", "plain\n");
        set_level(&store, "s", Some(2));
        set_level(&store, "f", None);
        let found = |q: &str| match store.find_document(q, true).unwrap() {
            DocumentLookup::Found(d) => d,
            DocumentLookup::NotFound { .. } => panic!("{q} not found"),
        };
        let file = found("log#2");
        assert_eq!(file.display_path, "f/log#2");
        assert!(file.section.is_none());
        let section = found("s/log#2");
        assert_eq!(section.display_path, "s/log");
        assert_eq!(section.section.unwrap().seq, 2);
    }

    #[test]
    fn a_section_identity_is_not_shadowed_by_a_suffix_match() {
        use crate::store::DocumentLookup;
        let store = Store::open_in_memory().unwrap();
        let body = "## One\n\nalpha\n\n## Two\n\nbeta\n";
        add(&store, "s", "dir/log", body);
        add(&store, "f", "a/s/dir/log#2", "plain\n");
        set_level(&store, "s", Some(2));
        set_level(&store, "f", None);
        let found = |q: &str| match store.find_document(q, true).unwrap() {
            DocumentLookup::Found(d) => (d.display_path, d.section.map(|s| s.seq)),
            DocumentLookup::NotFound { .. } => panic!("{q} not found"),
        };
        let section = ("s/dir/log".to_owned(), Some(2));
        assert_eq!(found("s/dir/log#2"), section);
        assert_eq!(found("dir/log#2"), section);
        assert_eq!(found(&format!("#{}#2", &hash_content(body)[..6])), section);
        assert_eq!(found("log#2"), ("f/a/s/dir/log#2".to_owned(), None));
    }

    #[test]
    fn a_docid_section_is_its_own_documents_when_display_paths_collide() {
        use crate::store::DocumentLookup;
        let store = Store::open_in_memory().unwrap();
        let one = "## One\n\nbeta\n";
        add(&store, "a", "b/x", "## One\n\nalpha\n\n## Two\n\ngamma\n");
        add(&store, "a/b", "x", one);
        set_level(&store, "a", Some(2));
        set_level(&store, "a/b", Some(2));
        let id = &hash_content(one)[..6];
        match store.find_document(&format!("{id}#1"), true).unwrap() {
            DocumentLookup::Found(d) => {
                assert_eq!(d.collection_name, "a/b");
                assert_eq!(d.body.as_deref(), Some(one));
            }
            DocumentLookup::NotFound { .. } => panic!("{id}#1 not found"),
        }
        assert!(matches!(
            store.find_document(&format!("{id}#2"), true).unwrap(),
            DocumentLookup::NotFound { .. }
        ));
    }

    #[test]
    fn renaming_a_section_collection_recuts_under_the_new_name() {
        let store = Store::open_in_memory().unwrap();
        add(&store, "s", "a.md", "## One\n\nalpha\n");
        set_level(&store, "s", Some(2));
        store.rename_collection("s", "t").unwrap();
        assert_eq!(fts_rows(&store, "documents_fts"), 0);
        let hits = store.search_fts("alpha", 10, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].key(), "qmd://t/a.md#1");
        assert!(store.section_of_file("t", "a.md", 1).unwrap().is_some());
    }
}
