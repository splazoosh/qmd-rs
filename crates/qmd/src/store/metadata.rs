//! Document metadata in the index: schema, persistence, the extraction gate
//! filtered search applies, and discovery. Ported from TS
//! `src/metadata-store.ts`; the tables are upstream's, so an index stays
//! shareable with TS qmd.
//!
//! Metadata attaches to document identity (`documents.id`), not content: two
//! paths sharing one hash may carry different metadata, and extraction never
//! touches content or vectors. `document_metadata` records extraction state
//! per document (empty, failed or current), so filtered search tells "no
//! metadata" from "not yet extracted"; `document_metadata_values` holds one
//! indexed row per scalar.

use std::collections::HashMap;

use rusqlite::types::Value as SqlValue;
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};

use crate::error::Result;
use crate::metadata::filter::{CompiledSql, FilterScope};
use crate::metadata::report::{
    MetadataFailure, MetadataKeySummary, MetadataKeyTypeSummary, MetadataValueCount, NumberRange,
    SQL_BINDING_BUDGET, SourceCount, ValueSort, Window,
};
use crate::metadata::{
    ExtractionResult, Inherited, ListMetadataOptions, ListMetadataResult,
    METADATA_EXTRACTION_VERSION, MetadataError, MetadataFilter, MetadataKeyOverview,
    MetadataOverview, MetadataSources, ResolvedMetadata, Scalar, ValueType,
    extract_document_metadata, parse_metadata_json,
};
use crate::store::{Store, now_iso};

pub(crate) fn initialize_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS document_metadata (
           document_id INTEGER PRIMARY KEY,
           metadata_json TEXT NOT NULL DEFAULT '{}',
           extraction_version INTEGER NOT NULL,
           extraction_error TEXT,
           extracted_at TEXT NOT NULL,
           FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
         );
         CREATE TABLE IF NOT EXISTS document_metadata_values (
           document_id INTEGER NOT NULL,
           key TEXT NOT NULL,
           ordinal INTEGER NOT NULL,
           value_type TEXT NOT NULL,
           text_value TEXT,
           number_value REAL,
           boolean_value INTEGER,
           PRIMARY KEY (document_id, key, ordinal),
           FOREIGN KEY (document_id)
             REFERENCES document_metadata(document_id)
             ON DELETE CASCADE,
           CHECK (value_type IN ('string', 'number', 'boolean')),
           CHECK (
             (value_type = 'string' AND text_value IS NOT NULL AND number_value IS NULL AND boolean_value IS NULL)
             OR (value_type = 'number' AND number_value IS NOT NULL AND text_value IS NULL AND boolean_value IS NULL)
             OR (value_type = 'boolean' AND boolean_value IN (0, 1) AND text_value IS NULL AND number_value IS NULL)
           )
         );
         CREATE INDEX IF NOT EXISTS idx_metadata_text_lookup
           ON document_metadata_values(key, text_value, document_id)
           WHERE value_type = 'string';
         CREATE INDEX IF NOT EXISTS idx_metadata_number_lookup
           ON document_metadata_values(key, number_value, document_id)
           WHERE value_type = 'number';
         CREATE INDEX IF NOT EXISTS idx_metadata_boolean_lookup
           ON document_metadata_values(key, boolean_value, document_id)
           WHERE value_type = 'boolean';",
    )?;
    for (table, column) in [
        ("document_metadata", "inherited_hash"),
        ("document_metadata", "sources_json"),
        ("document_metadata_values", "source"),
    ] {
        let present: bool = conn.query_row(
            "SELECT EXISTS (SELECT 1 FROM pragma_table_info(?1) WHERE name = ?2)",
            params![table, column],
            |r| r.get(0),
        )?;
        if !present {
            conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} TEXT"))?;
        }
    }
    Ok(())
}

/// Documents whose metadata filtered search may read: a current,
/// error-free extraction. A predicate over a documents-table alias `d`.
pub(crate) const CURRENT_EXTRACTION: &str = "EXISTS (SELECT 1 FROM document_metadata dm
     WHERE dm.document_id = d.id AND dm.extraction_version = 1 AND dm.extraction_error IS NULL)";

const _: () = assert!(
    METADATA_EXTRACTION_VERSION == 1,
    "CURRENT_EXTRACTION spells the version"
);

/// The SQL a search appends to its `WHERE` over `documents d` to apply
/// `filter`: the extraction gate, then the filter compiled for candidates.
pub(crate) fn search_filter_sql(filter: &MetadataFilter) -> CompiledSql {
    let compiled = filter.compile("d", FilterScope::Candidates);
    CompiledSql {
        sql: format!(" AND {CURRENT_EXTRACTION} AND {}", compiled.sql),
        params: compiled.params,
    }
}

/// A collection scope bound as one JSON parameter, so no variable limit applies.
fn collections_json(collections: &[String]) -> SqlValue {
    SqlValue::Text(serde_json::to_string(collections).expect("names serialize"))
}

fn count_of(conn: &Connection, sql: &str, params: &[SqlValue]) -> Result<u64> {
    let n: i64 = conn.query_row(sql, params_from_iter(params.iter()), |r| r.get(0))?;
    Ok(u64::try_from(n).unwrap_or(0))
}

/// `SELECT columns` over the active documents whose current extraction failed.
fn failures_query(columns: &str, collections: Option<&[String]>) -> (String, Vec<SqlValue>) {
    let mut sql = format!(
        "SELECT {columns}
         FROM documents d JOIN document_metadata dm ON dm.document_id = d.id
         WHERE d.active = 1 AND dm.extraction_error IS NOT NULL
           AND dm.extraction_version = {METADATA_EXTRACTION_VERSION}"
    );
    let mut params = Vec::new();
    if let Some(names) = collections {
        sql.push_str(" AND d.collection IN (SELECT value FROM json_each(?))");
        params.push(collections_json(names));
    }
    (sql, params)
}

impl Store {
    /// Extract and store metadata for document `id`, replacing what it had.
    /// With `only_if_stale`, a document whose extraction is current is left
    /// alone and `None` returned.
    ///
    /// # Errors
    /// Propagates SQLite failures.
    pub fn sync_document_metadata(
        &self,
        id: i64,
        content: &str,
        path: &str,
        only_if_stale: bool,
    ) -> Result<Option<ExtractionResult>> {
        self.sync_document_metadata_inheriting(
            id,
            content,
            path,
            only_if_stale,
            &Inherited::default(),
        )
    }

    /// [`Self::sync_document_metadata`] for a document that inherits
    /// `inherited`: the file's frontmatter over it. A document is stale when
    /// its extraction is from another version or what it inherits has
    /// changed since it was resolved. A failed extraction inherits nothing,
    /// as it declares nothing: filtered search excludes it either way.
    ///
    /// # Errors
    /// Propagates SQLite failures.
    pub fn sync_document_metadata_inheriting(
        &self,
        id: i64,
        content: &str,
        path: &str,
        only_if_stale: bool,
        inherited: &Inherited,
    ) -> Result<Option<ExtractionResult>> {
        let fingerprint = inherited.fingerprint();
        if only_if_stale {
            let current: Option<bool> = self
                .conn()
                .query_row(
                    "SELECT extraction_version = ?2 AND inherited_hash IS ?3
                     FROM document_metadata WHERE document_id = ?1",
                    params![id, METADATA_EXTRACTION_VERSION, fingerprint],
                    |r| r.get(0),
                )
                .optional()?;
            if current == Some(true) {
                return Ok(None);
            }
        }
        let mut extraction = extract_document_metadata(content, path);
        let sources = if extraction.error.is_none() {
            let resolved = inherited.merge(std::mem::take(&mut extraction.metadata));
            extraction.metadata = resolved.metadata;
            resolved.sources
        } else {
            MetadataSources::new()
        };
        self.write_document_metadata(id, &extraction, &sources, fingerprint.as_deref())?;
        Ok(Some(extraction))
    }

    /// Replace a document's metadata rows in one transaction (the caller's,
    /// when it holds one). A failed
    /// extraction stores empty metadata and the error, so stale metadata
    /// never survives a bad edit.
    ///
    /// # Errors
    /// Propagates SQLite failures.
    pub fn replace_document_metadata(&self, id: i64, extraction: &ExtractionResult) -> Result<()> {
        self.write_document_metadata(id, extraction, &MetadataSources::new(), None)
    }

    /// Store a document's resolved metadata: `sources` names the level of
    /// each inherited key, `fingerprint` what it inherited.
    fn write_document_metadata(
        &self,
        id: i64,
        extraction: &ExtractionResult,
        sources: &MetadataSources,
        fingerprint: Option<&str>,
    ) -> Result<()> {
        let tx = if self.conn().is_autocommit() {
            Some(self.conn().unchecked_transaction()?)
        } else {
            None
        };
        let conn = self.conn();
        conn.execute(
            "INSERT INTO document_metadata (document_id, metadata_json, extraction_version,
               extraction_error, extracted_at, inherited_hash, sources_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(document_id) DO UPDATE SET
               metadata_json = excluded.metadata_json,
               extraction_version = excluded.extraction_version,
               extraction_error = excluded.extraction_error,
               extracted_at = excluded.extracted_at,
               inherited_hash = excluded.inherited_hash,
               sources_json = excluded.sources_json",
            params![
                id,
                serde_json::to_string(&extraction.metadata).expect("metadata serializes"),
                extraction.extraction_version,
                extraction.error,
                now_iso(),
                fingerprint,
                (!sources.is_empty())
                    .then(|| serde_json::to_string(sources).expect("sources serialize")),
            ],
        )?;
        conn.execute(
            "DELETE FROM document_metadata_values WHERE document_id = ?1",
            params![id],
        )?;
        {
            let mut insert = conn.prepare(
                "INSERT INTO document_metadata_values
                   (document_id, key, ordinal, value_type, text_value, number_value, boolean_value, source)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?;
            for (key, value) in &extraction.metadata {
                let source = sources.get(key);
                for (ordinal, scalar) in value.scalars().iter().enumerate() {
                    let (text, number, boolean) = match scalar {
                        Scalar::String(s) => (Some(s.as_str()), None, None),
                        Scalar::Number(n) => (None, Some(*n), None),
                        Scalar::Boolean(b) => (None, None, Some(i64::from(*b))),
                    };
                    insert.execute(params![
                        id,
                        key,
                        i64::try_from(ordinal).expect("bounded array"),
                        scalar.value_type().as_str(),
                        text,
                        number,
                        boolean,
                        source,
                    ])?;
                }
            }
        }
        if let Some(tx) = tx {
            tx.commit()?;
        }
        Ok(())
    }

    /// Active documents with no extraction at the current version: what
    /// filtered search cannot see until `qmd update` runs. Scoped to
    /// `collections` when given. One that failed at the current version is
    /// not pending but in [`Self::metadata_failures`]; one that failed at an
    /// older version is pending, since `qmd update` retries it.
    ///
    /// # Errors
    /// Propagates SQLite failures.
    pub fn count_documents_pending_metadata(&self, collections: Option<&[String]>) -> Result<u64> {
        let mut sql = format!(
            "SELECT COUNT(*) FROM documents d WHERE d.active = 1 AND NOT EXISTS
               (SELECT 1 FROM document_metadata dm WHERE dm.document_id = d.id
                  AND dm.extraction_version = {METADATA_EXTRACTION_VERSION})"
        );
        let mut params = Vec::new();
        if let Some(names) = collections {
            sql.push_str(" AND d.collection IN (SELECT value FROM json_each(?))");
            params.push(collections_json(names));
        }
        count_of(self.conn(), &sql, &params)
    }

    /// Active documents whose current extraction failed, by collection and
    /// path: what filtered search cannot see until the file is fixed, however
    /// many times `qmd update` runs. Scoped to `collections` when given.
    ///
    /// # Errors
    /// Propagates SQLite failures.
    pub fn metadata_failures(
        &self,
        collections: Option<&[String]>,
    ) -> Result<Vec<MetadataFailure>> {
        let (mut sql, params) =
            failures_query("d.collection, d.path, dm.extraction_error", collections);
        sql.push_str(" ORDER BY d.collection, d.path");
        let mut stmt = self.conn().prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(params.iter()), |r| {
            Ok(MetadataFailure {
                collection: r.get(0)?,
                path: r.get(1)?,
                error: r.get(2)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// How many documents [`Self::metadata_failures`] lists.
    ///
    /// # Examples
    /// ```
    /// let store = qmd::Store::open_in_memory()?;
    /// assert_eq!(store.count_metadata_failures(None)?, 0);
    /// let scope = ["notes".to_owned()];
    /// assert_eq!(store.count_metadata_failures(Some(&scope))?, 0);
    /// # Ok::<(), qmd::Error>(())
    /// ```
    ///
    /// # Errors
    /// Propagates SQLite failures.
    pub fn count_metadata_failures(&self, collections: Option<&[String]>) -> Result<u64> {
        let (sql, params) = failures_query("COUNT(*)", collections);
        count_of(self.conn(), &sql, &params)
    }

    /// Stored metadata for result paths (`qmd://collection/path`), in one
    /// query. Paths with no metadata row are absent.
    ///
    /// # Errors
    /// Propagates SQLite failures.
    pub fn metadata_by_filepath(
        &self,
        filepaths: &[String],
    ) -> Result<HashMap<String, ResolvedMetadata>> {
        if filepaths.is_empty() {
            return Ok(HashMap::new());
        }
        let mut stmt = self.conn().prepare(
            "SELECT 'qmd://' || d.collection || '/' || d.path, dm.metadata_json, dm.sources_json
             FROM documents d JOIN document_metadata dm ON dm.document_id = d.id
             WHERE d.active = 1
               AND 'qmd://' || d.collection || '/' || d.path IN (SELECT value FROM json_each(?1))",
        )?;
        let json = serde_json::to_string(filepaths).expect("paths serialize");
        let rows = stmt.query_map(params![json], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<String>>(2)?,
            ))
        })?;
        let mut out = HashMap::new();
        for row in rows {
            let (path, json, sources) = row?;
            out.insert(
                path,
                ResolvedMetadata {
                    metadata: parse_metadata_json(json.as_deref()),
                    sources: sources
                        .and_then(|s| serde_json::from_str(&s).ok())
                        .unwrap_or_default(),
                },
            );
        }
        Ok(out)
    }

    /// Summarize metadata keys, types and value counts for the documents in
    /// scope. Discovery sees what filtering sees: the same extraction gate,
    /// active-document rule and collection scope, so every value reported is
    /// one an `eq` filter matches. Counts are documents, not values. The
    /// report is read in one transaction, so every count comes from one
    /// snapshot while another connection writes.
    ///
    /// # Errors
    /// [`MetadataError`] for an option outside its domain or a filter and
    /// match over the binding budget; SQLite failures.
    pub fn list_metadata(&self, options: &ListMetadataOptions) -> Result<ListMetadataResult> {
        let window = options.window()?;
        let tx = if self.conn().is_autocommit() {
            Some(self.conn().unchecked_transaction()?)
        } else {
            None
        };
        let result = read_report(self.conn(), options, &window)?;
        if let Some(tx) = tx {
            tx.commit()?;
        }
        Ok(result)
    }

    /// Every collection's key names, coverage and types in coverage order,
    /// the first `key_limit` per collection (`None`: all). One statement, no
    /// values: what `collection list`, `status` and MCP `status` print.
    ///
    /// # Errors
    /// Propagates SQLite failures.
    pub fn metadata_overviews(
        &self,
        key_limit: Option<u64>,
    ) -> Result<HashMap<String, MetadataOverview>> {
        let region = Region::matched(&eligible_cte(None, None), None);
        let limit = key_limit.map_or(-1, |n| i64::try_from(n).unwrap_or(i64::MAX));
        let sql = format!(
            "{},
             key_coverages AS (
               SELECT e.collection AS collection, mv.key, COUNT(*) AS documents,
                 SUM(mv.value_type = 'string') AS string_documents,
                 SUM(mv.value_type = 'number') AS number_documents,
                 SUM(mv.value_type = 'boolean') AS boolean_documents,
                 COUNT(*) OVER (PARTITION BY e.collection) AS total_keys,
                 ROW_NUMBER() OVER (PARTITION BY e.collection ORDER BY COUNT(*) DESC, mv.key) AS rank
               {}
               WHERE mv.ordinal = 0
               GROUP BY e.collection, mv.key
             )
             SELECT collection, key, documents, string_documents, number_documents, boolean_documents, total_keys
             FROM key_coverages WHERE rank <= ? OR ? = -1 ORDER BY collection, rank",
            region.with_sql, region.from_sql
        );
        let mut params = region.params();
        params.push(SqlValue::Integer(limit));
        params.push(SqlValue::Integer(limit));
        let mut stmt = self.conn().prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(params.iter()), |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
                [
                    r.get::<_, i64>(3)?,
                    r.get::<_, i64>(4)?,
                    r.get::<_, i64>(5)?,
                ],
                r.get::<_, i64>(6)?,
            ))
        })?;
        let mut overviews: HashMap<String, MetadataOverview> = HashMap::new();
        for row in rows {
            let (collection, key, documents, by_type, total_keys) = row?;
            let overview = overviews
                .entry(collection)
                .or_insert_with(|| MetadataOverview {
                    total_keys: u64::try_from(total_keys).unwrap_or(0),
                    keys: Vec::new(),
                });
            let mut types: Vec<(ValueType, i64)> =
                ValueType::ALL.into_iter().zip(by_type).collect();
            types.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.as_str().cmp(b.0.as_str())));
            overview.keys.push(MetadataKeyOverview {
                key,
                documents: u64::try_from(documents).unwrap_or(0),
                types: types
                    .into_iter()
                    .filter(|(_, n)| *n > 0)
                    .map(|(t, _)| t)
                    .collect(),
            });
        }
        Ok(overviews)
    }

    /// Distinct keys declared by active, extracted documents in scope.
    ///
    /// # Errors
    /// Propagates SQLite failures.
    pub fn count_metadata_keys(&self, collections: Option<&[String]>) -> Result<u64> {
        let region = Region::matched(&eligible_cte(collections, None), None);
        let sql = format!(
            "{} SELECT COUNT(DISTINCT mv.key) {} WHERE mv.ordinal = 0",
            region.with_sql, region.from_sql
        );
        count_of(self.conn(), &sql, &region.params())
    }

    /// Active, extracted documents in scope declaring at least one key.
    ///
    /// # Errors
    /// Propagates SQLite failures.
    pub fn count_documents_with_metadata(&self, collections: Option<&[String]>) -> Result<u64> {
        let eligible = eligible_cte(collections, None);
        let sql = format!(
            "{} SELECT COUNT(*) FROM eligible e
             WHERE EXISTS (SELECT 1 FROM document_metadata_values mv WHERE mv.document_id = e.document_id)",
            eligible.with_sql
        );
        count_of(self.conn(), &sql, &eligible.with_params)
    }
}

/// The value rows discovery aggregates over: `with_sql` defines the
/// `eligible` CTE (and, once a key window applies, `selected_keys`);
/// `from_sql` joins `document_metadata_values mv` to them, narrowed by a
/// predicate when one is set. Each query adds its own SELECT, WHERE and
/// GROUP BY around the two.
#[derive(Clone)]
struct Region {
    with_sql: String,
    with_params: Vec<SqlValue>,
    from_sql: String,
    from_params: Vec<SqlValue>,
}

impl Region {
    /// Eligible documents' values, narrowed to the entries `predicate`
    /// admits. The predicate sits in the JOIN so queries can add a WHERE.
    fn matched(eligible: &Region, predicate: Option<&CompiledSql>) -> Region {
        let mut region = Region {
            with_sql: eligible.with_sql.clone(),
            with_params: eligible.with_params.clone(),
            from_sql: " FROM document_metadata_values mv JOIN eligible e ON e.document_id = mv.document_id"
                .to_owned(),
            from_params: Vec::new(),
        };
        if let Some(p) = predicate {
            region.from_sql.push_str(&format!(" AND {}", p.sql));
            region.from_params.extend(p.params.iter().cloned());
        }
        region
    }

    fn params(&self) -> Vec<SqlValue> {
        self.with_params
            .iter()
            .chain(&self.from_params)
            .cloned()
            .collect()
    }

    fn bindings(&self) -> usize {
        self.with_params.len() + self.from_params.len()
    }
}

/// Documents filtered search can see: active, current and error-free, in
/// scope, and passing the filter, compiled for the corpus (a document set
/// built once per statement and probed).
fn eligible_cte(collections: Option<&[String]>, filter: Option<&MetadataFilter>) -> Region {
    let mut with_sql = "WITH eligible AS (
       SELECT d.id AS document_id, d.collection
       FROM documents d
       JOIN document_metadata dm ON dm.document_id = d.id
       WHERE d.active = 1
         AND dm.extraction_version = 1
         AND dm.extraction_error IS NULL"
        .to_owned();
    let mut with_params = Vec::new();
    if let Some(names) = collections {
        with_sql.push_str(" AND d.collection IN (SELECT value FROM json_each(?))");
        with_params.push(collections_json(names));
    }
    if let Some(filter) = filter {
        let compiled = filter.compile("d", FilterScope::Corpus);
        with_sql.push_str(&format!(" AND {}", compiled.sql));
        with_params.extend(compiled.params);
    }
    with_sql.push_str("\n     )");
    Region {
        with_sql,
        with_params,
        from_sql: String::new(),
        from_params: Vec::new(),
    }
}

/// Narrow a region to one window of its keys in report order (documents
/// descending, then key in BINARY collation), each carrying its `rank`.
/// MATERIALIZED, so a small LIMIT does not invite the planner to re-run the
/// ranking once per value row. SQLite reads `LIMIT -1` as no limit.
fn key_window(region: &Region, window: &Window) -> Region {
    let with_sql = format!(
        "{},
         selected_keys AS MATERIALIZED (
           SELECT mv.key, ROW_NUMBER() OVER (ORDER BY COUNT(DISTINCT mv.document_id) DESC, mv.key ASC) AS rank
           {}
           GROUP BY mv.key
           ORDER BY rank
           LIMIT ? OFFSET ?
         )",
        region.with_sql, region.from_sql
    );
    let mut with_params = region.params();
    with_params.push(SqlValue::Integer(window.key_limit.unwrap_or(-1)));
    with_params.push(SqlValue::Integer(window.key_offset));
    Region {
        with_sql,
        with_params,
        from_sql: format!(
            "{} JOIN selected_keys sk ON sk.key = mv.key",
            region.from_sql
        ),
        from_params: region.from_params.clone(),
    }
}

/// Every entry of these keys, bound as one JSON parameter.
fn key_set(keys: &[String]) -> CompiledSql {
    CompiledSql {
        sql: "mv.key IN (SELECT value FROM json_each(?))".to_owned(),
        params: vec![SqlValue::Text(
            serde_json::to_string(keys).expect("keys serialize"),
        )],
    }
}

/// Selected names as JSON, minCount, offset and the two endpoint operands.
const VALUE_QUERY_BINDINGS: usize = 5;

fn read_report(
    conn: &Connection,
    options: &ListMetadataOptions,
    window: &Window,
) -> Result<ListMetadataResult> {
    let collections = options.collections.as_deref();
    let eligible = eligible_cte(collections, options.filter.as_ref());
    let compiled_match = options.r#match.as_ref().map(|m| m.compile("mv"));
    let matched = Region::matched(&eligible, compiled_match.as_ref());
    let windowed = key_window(&matched, window);
    let bindings = windowed
        .bindings()
        .max(matched.bindings() + VALUE_QUERY_BINDINGS);
    if bindings > SQL_BINDING_BUDGET {
        return Err(MetadataError::BindingBudget {
            bindings,
            budget: SQL_BINDING_BUDGET,
        }
        .into());
    }

    let mut documents_sql = "SELECT COUNT(*) FROM documents d WHERE d.active = 1".to_owned();
    let mut documents_params = Vec::new();
    if let Some(names) = collections {
        documents_sql.push_str(" AND d.collection IN (SELECT value FROM json_each(?))");
        documents_params.push(collections_json(names));
    }
    let mut result = ListMetadataResult {
        documents: count_of(conn, &documents_sql, &documents_params)?,
        filtered_documents: None,
        total_keys: 0,
        keys: Vec::new(),
        remaining_keys: 0,
    };
    if options.filter.is_some() {
        let sql = format!("{} SELECT COUNT(*) FROM eligible", eligible.with_sql);
        result.filtered_documents = Some(count_of(conn, &sql, &eligible.with_params)?);
    }
    let sql = format!(
        "{} SELECT COUNT(DISTINCT mv.key) {}",
        matched.with_sql, matched.from_sql
    );
    result.total_keys = count_of(conn, &sql, &matched.params())?;
    if result.total_keys == 0 {
        return Ok(result);
    }

    let stats = type_stats(conn, &windowed)?;
    if stats.is_empty() {
        return Ok(result);
    }
    let mut key_names: Vec<String> = Vec::new();
    for row in &stats {
        if key_names.last() != Some(&row.key) {
            key_names.push(row.key.clone());
        }
    }
    let selected = key_set(&key_names);
    let region = Region {
        from_sql: format!("{} AND {}", matched.from_sql, selected.sql),
        from_params: matched
            .from_params
            .iter()
            .cloned()
            .chain(selected.params.clone())
            .collect(),
        ..matched.clone()
    };
    let medians = number_medians(conn, &region)?;

    let mut summaries: Vec<(String, Vec<MetadataKeyTypeSummary>)> = Vec::new();
    for row in stats {
        let mut collections: Vec<String> =
            serde_json::from_str(&row.collections).unwrap_or_default();
        collections.sort();
        let range = (row.value_type == ValueType::Number).then(|| NumberRange {
            min: row.min.unwrap_or(0.0),
            median: medians.get(&row.key).copied().unwrap_or(0.0),
            max: row.max.unwrap_or(0.0),
        });
        let summary = MetadataKeyTypeSummary {
            value_type: row.value_type,
            multi_valued: false,
            documents: row.documents,
            distinct_values: 0,
            values: Vec::new(),
            remaining_values: 0,
            range,
            collections,
        };
        match summaries.last_mut() {
            Some((k, types)) if *k == row.key => types.push(summary),
            _ => summaries.push((row.key, vec![summary])),
        }
    }
    let position: HashMap<(String, ValueType), (usize, usize)> = summaries
        .iter()
        .enumerate()
        .flat_map(|(i, (key, types))| {
            types
                .iter()
                .enumerate()
                .map(move |(j, t)| ((key.clone(), t.value_type), (i, j)))
        })
        .collect();

    let key_region = Region::matched(&eligible, Some(&selected));
    for (key, t, multi) in arrayness(conn, &key_region)? {
        if let Some(&(i, j)) = position.get(&(key, t)) {
            summaries[i].1[j].multi_valued = multi;
        }
    }
    for row in value_counts(conn, &region, window)? {
        if let Some(&(i, j)) = position.get(&(row.key, row.value_type)) {
            let s = &mut summaries[i].1[j];
            s.distinct_values = row.distinct_values;
            if row.in_window {
                s.values.push(MetadataValueCount {
                    value: row.value,
                    documents: row.documents,
                    inherited: Vec::new(),
                });
            }
        }
    }
    for (key, value_type, value, count) in value_sources(conn, &region)? {
        if let Some(&(i, j)) = position.get(&(key, value_type))
            && let Some(v) = summaries[i].1[j]
                .values
                .iter_mut()
                .find(|v| v.value == value)
        {
            v.inherited.push(count);
        }
    }
    let value_offset = u64::try_from(window.value_offset).unwrap_or(0);
    for (key, mut types) in summaries {
        for t in &mut types {
            t.remaining_values = t
                .distinct_values
                .saturating_sub(value_offset)
                .saturating_sub(t.values.len() as u64);
        }
        types.sort_by(|a, b| {
            b.documents
                .cmp(&a.documents)
                .then(a.value_type.as_str().cmp(b.value_type.as_str()))
        });
        result.keys.push(MetadataKeySummary {
            documents: types.iter().map(|t| t.documents).sum(),
            key,
            types,
        });
    }
    let key_offset = u64::try_from(window.key_offset).unwrap_or(0);
    result.remaining_keys = result
        .total_keys
        .saturating_sub(key_offset)
        .saturating_sub(result.keys.len() as u64);
    Ok(result)
}

struct TypeStats {
    key: String,
    value_type: ValueType,
    documents: u64,
    min: Option<f64>,
    max: Option<f64>,
    collections: String,
}

fn value_type_of(s: &str) -> ValueType {
    ValueType::parse(s).unwrap_or(ValueType::String)
}

/// Per key and type over the key window, in the window's rank order.
/// Collection names are sorted after reading (aggregate ORDER BY needs
/// SQLite 3.44, newer than upstream's floor).
fn type_stats(conn: &Connection, region: &Region) -> Result<Vec<TypeStats>> {
    let sql = format!(
        "{}
         SELECT mv.key, mv.value_type,
           COUNT(DISTINCT mv.document_id),
           MIN(mv.number_value),
           MAX(mv.number_value),
           json_group_array(DISTINCT e.collection)
         {}
         GROUP BY mv.key, mv.value_type
         ORDER BY sk.rank, mv.value_type",
        region.with_sql, region.from_sql
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(region.params()), |r| {
        Ok(TypeStats {
            key: r.get(0)?,
            value_type: value_type_of(&r.get::<_, String>(1)?),
            documents: u64::try_from(r.get::<_, i64>(2)?).unwrap_or(0),
            min: r.get(3)?,
            max: r.get(4)?,
            collections: r.get(5)?,
        })
    })?;
    Ok(rows.collect::<std::result::Result<_, _>>()?)
}

fn arrayness(conn: &Connection, region: &Region) -> Result<Vec<(String, ValueType, bool)>> {
    let sql = format!(
        "{} SELECT mv.key, mv.value_type, MAX(mv.ordinal) > 0 {} GROUP BY mv.key, mv.value_type",
        region.with_sql, region.from_sql
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(region.params()), |r| {
        Ok((
            r.get::<_, String>(0)?,
            value_type_of(&r.get::<_, String>(1)?),
            r.get::<_, i64>(2)? == 1,
        ))
    })?;
    Ok(rows.collect::<std::result::Result<_, _>>()?)
}

/// Median over value rows per key: the middle row for an odd count, the
/// midpoint of the two middle rows for an even one, taken here because
/// SQL's AVG sums first and the sum of two finite doubles can overflow.
fn number_medians(conn: &Connection, region: &Region) -> Result<HashMap<String, f64>> {
    let sql = format!(
        "{}
         SELECT key, number_value
         FROM (
           SELECT mv.key, mv.number_value,
             ROW_NUMBER() OVER (PARTITION BY mv.key ORDER BY mv.number_value) AS position,
             COUNT(*) OVER (PARTITION BY mv.key) AS total
           {}
           WHERE mv.value_type = 'number'
         )
         WHERE position IN ((total + 1) / 2, (total + 2) / 2)
         ORDER BY key, number_value",
        region.with_sql, region.from_sql
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(region.params()), |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
    })?;
    let mut medians: HashMap<String, f64> = HashMap::new();
    for row in rows {
        let (key, value) = row?;
        medians
            .entry(key)
            .and_modify(|lower| *lower = midpoint(*lower, value))
            .or_insert(value);
    }
    Ok(medians)
}

/// The correctly rounded midpoint when the sum is representable (which
/// keeps adjacent subnormals exact); halving each side first only where the
/// sum overflows.
fn midpoint(lower: f64, upper: f64) -> f64 {
    let sum = lower + upper;
    if sum.is_finite() {
        sum / 2.0
    } else {
        lower / 2.0 + upper / 2.0
    }
}

/// Documents inheriting each value, per level, most first. Rows a file
/// declares carry no source, so an index without inherited metadata yields
/// none.
fn value_sources(
    conn: &Connection,
    region: &Region,
) -> Result<Vec<(String, ValueType, Scalar, SourceCount)>> {
    let sql = format!(
        "{}
         SELECT mv.key, mv.value_type, mv.text_value, mv.number_value, mv.boolean_value,
           mv.source, COUNT(DISTINCT mv.document_id) AS documents
         {}
         WHERE mv.source IS NOT NULL
         GROUP BY mv.key, mv.value_type, mv.text_value, mv.number_value, mv.boolean_value, mv.source
         ORDER BY mv.key, mv.value_type, documents DESC, mv.source",
        region.with_sql, region.from_sql
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(region.params()), |r| {
        let value_type = value_type_of(&r.get::<_, String>(1)?);
        let value = match value_type {
            ValueType::String => Scalar::String(r.get(2)?),
            ValueType::Number => Scalar::Number(r.get(3)?),
            ValueType::Boolean => Scalar::Boolean(r.get::<_, i64>(4)? == 1),
        };
        Ok((
            r.get::<_, String>(0)?,
            value_type,
            value,
            SourceCount {
                source: r.get(5)?,
                documents: u64::try_from(r.get::<_, i64>(6)?).unwrap_or(0),
            },
        ))
    })?;
    Ok(rows.collect::<std::result::Result<_, _>>()?)
}

struct ValueCountRow {
    key: String,
    value_type: ValueType,
    value: Scalar,
    documents: u64,
    distinct_values: u64,
    in_window: bool,
}

/// Distinct values grouped once for both the total and the window. The
/// first row of each partition carries its total even when the window is
/// past the end; only rows in the window become values. One typed column is
/// non-null per partition, so ordering by all three is stable.
fn value_counts(conn: &Connection, region: &Region, window: &Window) -> Result<Vec<ValueCountRow>> {
    let value_order = "mv.text_value, mv.number_value, mv.boolean_value";
    let rank_order = match window.sort {
        ValueSort::Count => format!("COUNT(DISTINCT mv.document_id) DESC, {value_order}"),
        ValueSort::Value => value_order.to_owned(),
    };
    let mut params = region.params();
    params.push(SqlValue::Integer(window.min_count));
    params.push(SqlValue::Integer(window.value_offset));
    let mut window_sql = "rank > ?".to_owned();
    if let Some(limit) = window.value_limit {
        window_sql.push_str(" AND rank <= CAST(? AS INTEGER) + CAST(? AS INTEGER)");
        params.push(SqlValue::Integer(window.value_offset));
        params.push(SqlValue::Integer(limit));
    }
    let sql = format!(
        "{},
         value_counts AS (
           SELECT mv.key, mv.value_type, mv.text_value, mv.number_value, mv.boolean_value,
             COUNT(DISTINCT mv.document_id) AS documents,
             COUNT(*) OVER (PARTITION BY mv.key, mv.value_type) AS distinct_values,
             ROW_NUMBER() OVER (PARTITION BY mv.key, mv.value_type ORDER BY {rank_order}) AS rank
           {}
           GROUP BY mv.key, mv.value_type, mv.text_value, mv.number_value, mv.boolean_value
           HAVING COUNT(DISTINCT mv.document_id) >= ?
         ),
         value_window AS (
           SELECT *, ({window_sql}) AS in_window FROM value_counts
         )
         SELECT key, value_type, text_value, number_value, boolean_value, documents, distinct_values, in_window
         FROM value_window
         WHERE rank = 1 OR in_window
         ORDER BY key, value_type, rank",
        region.with_sql, region.from_sql
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(params), |r| {
        let value_type = value_type_of(&r.get::<_, String>(1)?);
        let value = match value_type {
            ValueType::String => Scalar::String(r.get(2)?),
            ValueType::Number => Scalar::Number(r.get(3)?),
            ValueType::Boolean => Scalar::Boolean(r.get::<_, i64>(4)? == 1),
        };
        Ok(ValueCountRow {
            key: r.get(0)?,
            value_type,
            value,
            documents: u64::try_from(r.get::<_, i64>(5)?).unwrap_or(0),
            distinct_values: u64::try_from(r.get::<_, i64>(6)?).unwrap_or(0),
            in_window: r.get::<_, i64>(7)? != 0,
        })
    })?;
    Ok(rows.collect::<std::result::Result<_, _>>()?)
}
