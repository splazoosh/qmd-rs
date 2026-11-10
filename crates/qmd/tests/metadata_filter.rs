//! Metadata filters: validation, SQL semantics in both scopes, and filtered
//! BM25, section and vector search. Ported from upstream's
//! `test/metadata-filter.test.ts` and `test/metadata-search.test.ts`.

use qmd::Store;
use qmd::metadata::filter::{FilterScope, limits};
use qmd::metadata::{
    DocumentMetadata, ExtractionResult, METADATA_EXTRACTION_VERSION, MetadataFilter, MetadataMatch,
    parse_metadata_json,
};
use qmd::store::{TermMatch, hash_content};
use serde_json::{Value, json};

fn md(json: Value) -> DocumentMetadata {
    parse_metadata_json(Some(&json.to_string()))
}

/// Insert an active document whose body mentions `doc`, with `metadata`
/// extracted. Returns its id.
fn insert(store: &Store, collection: &str, path: &str, body: &str, metadata: Value) -> i64 {
    let hash = hash_content(body);
    store.insert_content(&hash, body, "t").unwrap();
    store
        .insert_document(collection, path, path, &hash, "t", "t")
        .unwrap();
    let (id, _, _) = store
        .find_active_document(collection, path)
        .unwrap()
        .unwrap();
    store
        .replace_document_metadata(
            id,
            &ExtractionResult {
                metadata: md(metadata),
                error: None,
                extraction_version: METADATA_EXTRACTION_VERSION,
            },
        )
        .unwrap();
    id
}

fn filter(json: Value) -> MetadataFilter {
    MetadataFilter::parse(&json).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn accepts_every_condition_operator_shape() {
    let conditions = [
        json!({"field": "status", "operator": "eq", "value": "published"}),
        json!({"field": "status", "operator": "ne", "value": "draft"}),
        json!({"field": "priority", "operator": "gt", "value": 3}),
        json!({"field": "priority", "operator": "gte", "value": 3}),
        json!({"field": "priority", "operator": "lt", "value": 10}),
        json!({"field": "name", "operator": "lte", "value": "m"}),
        json!({"field": "topics", "operator": "in", "value": ["a", "b"]}),
        json!({"field": "topics", "operator": "nin", "value": [1, 2]}),
        json!({"field": "flags", "operator": "all", "value": [true, false]}),
        json!({"field": "status", "operator": "exists", "value": false}),
        json!({"field": "topics", "operator": "contains", "value": "vec"}),
        json!({"field": "topics", "operator": "prefix", "value": "sql"}),
        json!({"field": "lead", "operator": "suffix", "value": "-team"}),
        json!({"field": "priority", "operator": "type", "value": "number"}),
    ];
    for condition in conditions {
        assert_eq!(filter(condition.clone()).to_json(), condition);
    }
    let nested = json!({"operator": "and", "operands": [
        {"field": "topics", "operator": "all", "value": ["typescript", "programming"]},
        {"operator": "or", "operands": [
            {"field": "status", "operator": "eq", "value": "published"},
            {"operator": "not", "operand": {"field": "audience", "operator": "eq", "value": "internal"}},
        ]},
    ]});
    assert_eq!(filter(nested.clone()).to_json(), nested);
    assert_eq!(
        filter(json!({"field": "topics", "operator": "in", "value": ["a", "b", "a"]})).to_json(),
        json!({"field": "topics", "operator": "in", "value": ["a", "b"]})
    );
}

#[test]
fn case_insensitive_applies_to_string_operands_only() {
    for accepted in [
        json!({"field": "status", "operator": "eq", "value": "Published", "caseInsensitive": true}),
        json!({"field": "status", "operator": "ne", "value": "Draft", "caseInsensitive": false}),
        json!({"field": "name", "operator": "gt", "value": "M", "caseInsensitive": true}),
        json!({"field": "topics", "operator": "in", "value": ["SQL", "TypeScript"], "caseInsensitive": true}),
        json!({"field": "topics", "operator": "all", "value": ["SQL"], "caseInsensitive": true}),
        json!({"field": "topics", "operator": "contains", "value": "SQL", "caseInsensitive": true}),
        json!({"field": "topics", "operator": "prefix", "value": "SQL", "caseInsensitive": true}),
        json!({"field": "topics", "operator": "suffix", "value": "SQL", "caseInsensitive": true}),
    ] {
        assert_eq!(filter(accepted.clone()).to_json(), accepted);
    }
    for (input, expected) in [
        (
            json!({"field": "priority", "operator": "eq", "value": 3, "caseInsensitive": true}),
            "at $.caseInsensitive: 'caseInsensitive' applies to string values only",
        ),
        (
            json!({"field": "reviewed", "operator": "in", "value": [true], "caseInsensitive": true}),
            "string values only",
        ),
        (
            json!({"field": "reviewed", "operator": "exists", "value": true, "caseInsensitive": true}),
            "does not apply to 'exists'",
        ),
        (
            json!({"field": "priority", "operator": "type", "value": "number", "caseInsensitive": true}),
            "does not apply to 'type'",
        ),
        (
            json!({"field": "status", "operator": "eq", "value": "x", "caseInsensitive": "yes"}),
            "must be a boolean",
        ),
    ] {
        let error = MetadataFilter::parse(&input).unwrap_err().to_string();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn rejects_invalid_shapes_naming_the_json_path() {
    let cases = [
        (
            json!("not-an-object"),
            "at $: each filter node must be an object",
        ),
        (
            json!({"field": "a", "value": 1}),
            "at $: missing 'operator'",
        ),
        (
            json!({"operator": "equal", "field": "a", "value": 1}),
            "unknown operator 'equal'",
        ),
        (
            json!({"operator": "and", "operands": []}),
            "non-empty 'operands'",
        ),
        (
            json!({"operator": "and", "operands": "nope"}),
            "'operands' array",
        ),
        (
            json!({"operator": "not", "operands": [{"field": "a", "operator": "eq", "value": 1}]}),
            "unknown property 'operands'",
        ),
        (json!({"operator": "not"}), "exactly one 'operand'"),
        (
            json!({"operator": "and", "operands": [{"operator": "eq"}]}),
            "at $.operands[0]",
        ),
        (
            json!({"operator": "eq", "value": 1}),
            "non-empty string 'field'",
        ),
        (
            json!({"operator": "eq", "field": "a"}),
            "requires a 'value'",
        ),
        (
            json!({"operator": "eq", "field": "a", "value": 1, "extra": true}),
            "unknown property 'extra'",
        ),
        (
            json!({"operator": "and", "operands": [{"field": "a", "operator": "eq", "value": 1}], "field": "a"}),
            "unknown property 'field'",
        ),
        (
            json!({"operator": "eq", "key": "a", "value": 1}),
            "unknown property 'key'",
        ),
        (
            json!({"operator": "gt", "field": "a", "value": true}),
            "string or number value",
        ),
        (
            json!({"operator": "eq", "field": "a", "value": {"nested": 1}}),
            "string, number, or boolean",
        ),
        (
            json!({"operator": "in", "field": "a", "value": "x"}),
            "array value",
        ),
        (
            json!({"operator": "in", "field": "a", "value": []}),
            "non-empty array",
        ),
        (
            json!({"operator": "in", "field": "a", "value": [1, "two"]}),
            "homogeneous array",
        ),
        (
            json!({"operator": "exists", "field": "a", "value": "yes"}),
            "boolean value",
        ),
        (
            json!({"operator": "contains", "field": "a", "value": ""}),
            "at $.value: 'contains' requires a non-empty string",
        ),
        (
            json!({"operator": "prefix", "field": "a", "value": 3}),
            "'prefix' requires a non-empty string",
        ),
        (
            json!({"operator": "suffix", "field": "a", "value": ["x"]}),
            "string, number, or boolean",
        ),
        (
            json!({"operator": "type", "field": "a", "value": "integer"}),
            "at $.value: 'type' requires one of: string, number, boolean",
        ),
        (
            json!({"operator": "type", "field": "a", "value": 1}),
            "'type' requires one of",
        ),
    ];
    for (input, expected) in cases {
        let error = MetadataFilter::parse(&input).unwrap_err().to_string();
        assert!(error.starts_with("Invalid metadata filter at $"), "{error}");
        assert!(error.contains(expected), "{input}: {error}");
    }
}

#[test]
fn rejects_excessive_depth_node_count_operands_and_values() {
    let condition = json!({"field": "a", "operator": "eq", "value": 1});
    let mut deep = condition.clone();
    for _ in 0..=limits::MAX_DEPTH {
        deep = json!({"operator": "not", "operand": deep});
    }
    assert!(
        MetadataFilter::parse(&deep)
            .unwrap_err()
            .to_string()
            .contains("nesting depth")
    );
    let wide = json!({"operator": "or", "operands": vec![condition.clone(); limits::MAX_GROUP_OPERANDS + 1]});
    assert!(
        MetadataFilter::parse(&wide)
            .unwrap_err()
            .to_string()
            .contains("operands")
    );
    let group =
        json!({"operator": "and", "operands": vec![condition.clone(); limits::MAX_GROUP_OPERANDS]});
    let many = json!({"operator": "and", "operands": vec![group; limits::MAX_GROUP_OPERANDS]});
    assert!(
        MetadataFilter::parse(&many)
            .unwrap_err()
            .to_string()
            .contains("nodes")
    );
    let values: Vec<usize> = (0..=limits::MAX_MEMBERSHIP_VALUES).collect();
    let long = json!({"field": "a", "operator": "in", "value": values});
    assert!(
        MetadataFilter::parse(&long)
            .unwrap_err()
            .to_string()
            .contains("values")
    );
}

#[test]
fn a_match_takes_key_and_value_fields_and_names_itself() {
    let accepted = json!({"operator": "and", "operands": [
        {"field": "key", "operator": "prefix", "value": "mem-"},
        {"operator": "or", "operands": [
            {"field": "value", "operator": "gte", "value": 3},
            {"field": "value", "operator": "type", "value": "boolean"},
            {"operator": "not", "operand": {"field": "value", "operator": "in", "value": ["Draft"], "caseInsensitive": true}},
        ]},
    ]});
    assert_eq!(MetadataMatch::parse(&accepted).unwrap().to_json(), accepted);
    let cases = [
        (
            json!({"field": "topics", "operator": "eq", "value": "x"}),
            "Invalid metadata match at $: 'topics' is not a field of a metadata entry, expected 'key' or 'value'",
        ),
        (
            json!({"field": "value", "operator": "exists", "value": true}),
            "Invalid metadata match at $: 'exists' has no meaning for a single metadata entry",
        ),
        (
            json!({"field": "key", "operator": "all", "value": ["a"]}),
            "Invalid metadata match at $: 'all' has no meaning for a single metadata entry",
        ),
        (
            json!({"operator": "not", "operand": {"field": "value", "operator": "eq"}}),
            "Invalid metadata match at $.operand: 'eq' requires a 'value'",
        ),
        (
            json!("nope"),
            "Invalid metadata match at $: each filter node must be an object",
        ),
    ];
    for (input, expected) in cases {
        assert_eq!(
            MetadataMatch::parse(&input).unwrap_err().to_string(),
            expected
        );
    }
    assert!(
        MetadataFilter::parse(&json!("nope"))
            .unwrap_err()
            .to_string()
            .starts_with("Invalid metadata filter at $:")
    );
}

#[test]
fn a_match_compiles_over_its_row_alias_with_every_operand_bound() {
    use rusqlite::types::Value as Sql;
    let compiled = MetadataMatch::parse(&json!({"operator": "and", "operands": [
        {"field": "key", "operator": "eq", "value": "k'; --"},
        {"field": "value", "operator": "suffix", "value": "V'; --", "caseInsensitive": true},
        {"field": "value", "operator": "nin", "value": [1, 2]},
    ]}))
    .unwrap()
    .compile("row");
    assert!(!compiled.sql.contains("'; --") && !compiled.sql.contains("EXISTS"));
    assert!(compiled.sql.contains("row.key = ?"));
    assert!(compiled.sql.contains("lower(row.text_value)"));
    assert!(compiled.sql.contains("row.number_value NOT IN (?, ?)"));
    assert_eq!(
        compiled.params,
        [
            Sql::Text("k'; --".into()),
            Sql::Integer(6),
            Sql::Text("v'; --".into()),
            Sql::Real(1.0),
            Sql::Real(2.0),
        ]
    );
}

/// The paths a filter admits, checked to agree across both scopes.
fn match_paths(store: &Store, json: Value) -> Vec<String> {
    let parsed = filter(json.clone());
    let run = |scope| {
        let compiled = parsed.compile("d", scope);
        let sql = format!(
            "SELECT d.path FROM documents d
             JOIN document_metadata dm ON dm.document_id = d.id
               AND dm.extraction_version = 1 AND dm.extraction_error IS NULL
             WHERE d.active = 1 AND {} ORDER BY d.path",
            compiled.sql
        );
        let mut stmt = store.conn().prepare(&sql).unwrap();
        stmt.query_map(rusqlite::params_from_iter(compiled.params), |r| r.get(0))
            .unwrap()
            .collect::<Result<Vec<String>, _>>()
            .unwrap()
    };
    let candidates = run(FilterScope::Candidates);
    assert_eq!(
        run(FilterScope::Corpus),
        candidates,
        "corpus scope for {json}"
    );
    candidates
}

fn docs(entries: &[(&str, Value)]) -> Store {
    let store = Store::open_in_memory().unwrap();
    for (path, metadata) in entries {
        insert(
            &store,
            "c",
            path,
            &format!("body of {path}"),
            metadata.clone(),
        );
    }
    store
}

#[test]
fn comparisons_match_by_type_without_coercion() {
    let s = docs(&[
        ("str.md", json!({"status": "published"})),
        ("num.md", json!({"status": 1})),
        ("bool.md", json!({"status": true})),
    ]);
    assert_eq!(
        match_paths(
            &s,
            json!({"field": "status", "operator": "eq", "value": "published"})
        ),
        ["str.md"]
    );
    assert_eq!(
        match_paths(&s, json!({"field": "status", "operator": "eq", "value": 1})),
        ["num.md"]
    );
    assert_eq!(
        match_paths(
            &s,
            json!({"field": "status", "operator": "eq", "value": true})
        ),
        ["bool.md"]
    );
    assert!(
        match_paths(
            &s,
            json!({"field": "status", "operator": "eq", "value": "1"})
        )
        .is_empty()
    );

    let s = docs(&[
        ("draft.md", json!({"status": "draft"})),
        ("published.md", json!({"status": "published"})),
        ("missing.md", json!({"other": "x"})),
        ("typed.md", json!({"status": 1})),
    ]);
    assert_eq!(
        match_paths(
            &s,
            json!({"field": "status", "operator": "ne", "value": "draft"})
        ),
        ["published.md"]
    );

    let s = docs(&[
        ("low.md", json!({"priority": 1})),
        ("mid.md", json!({"priority": 3})),
        ("high.md", json!({"priority": 7})),
        ("alpha.md", json!({"name": "alpha"})),
        ("zulu.md", json!({"name": "zulu"})),
    ]);
    assert_eq!(
        match_paths(
            &s,
            json!({"field": "priority", "operator": "gte", "value": 3})
        ),
        ["high.md", "mid.md"]
    );
    assert_eq!(
        match_paths(
            &s,
            json!({"field": "priority", "operator": "lt", "value": 3})
        ),
        ["low.md"]
    );
    assert_eq!(
        match_paths(
            &s,
            json!({"field": "name", "operator": "gt", "value": "alpha"})
        ),
        ["zulu.md"]
    );
    assert!(
        match_paths(
            &s,
            json!({"field": "priority", "operator": "gte", "value": "3"})
        )
        .is_empty()
    );
}

#[test]
fn membership_and_presence() {
    let s = docs(&[
        ("ts.md", json!({"topics": ["typescript", "programming"]})),
        ("go.md", json!({"topics": ["go"]})),
        ("none.md", json!({"other": "x"})),
    ]);
    let topics = |op: &str, value: Value| {
        match_paths(
            &s,
            json!({"field": "topics", "operator": op, "value": value}),
        )
    };
    assert_eq!(topics("in", json!(["typescript", "rust"])), ["ts.md"]);
    assert_eq!(topics("nin", json!(["typescript", "rust"])), ["go.md"]);
    assert_eq!(
        topics("all", json!(["typescript", "programming"])),
        ["ts.md"]
    );
    assert!(topics("all", json!(["typescript", "rust"])).is_empty());
    assert_eq!(topics("exists", json!(true)), ["go.md", "ts.md"]);
    assert_eq!(topics("exists", json!(false)), ["none.md"]);

    let s = docs(&[
        ("flagged.md", json!({"reviewed": true})),
        ("unflagged.md", json!({"reviewed": false})),
    ]);
    assert_eq!(
        match_paths(
            &s,
            json!({"field": "reviewed", "operator": "in", "value": [true]})
        ),
        ["flagged.md"]
    );
    assert_eq!(
        match_paths(
            &s,
            json!({"field": "reviewed", "operator": "nin", "value": [true]})
        ),
        ["unflagged.md"]
    );
}

#[test]
fn groups_compose_and_conditions_on_an_array_are_independent() {
    let s = docs(&[
        ("a.md", json!({"status": "published", "priority": 5})),
        ("b.md", json!({"status": "published", "priority": 1})),
        ("c.md", json!({"status": "draft", "priority": 9})),
    ]);
    assert_eq!(
        match_paths(
            &s,
            json!({"operator": "and", "operands": [
                {"field": "status", "operator": "eq", "value": "published"},
                {"field": "priority", "operator": "gte", "value": 3},
            ]})
        ),
        ["a.md"]
    );
    assert_eq!(
        match_paths(
            &s,
            json!({"operator": "or", "operands": [
                {"field": "priority", "operator": "gte", "value": 9},
                {"field": "priority", "operator": "lte", "value": 1},
            ]})
        ),
        ["b.md", "c.md"]
    );
    assert_eq!(
        match_paths(
            &s,
            json!({"operator": "not", "operand": {"field": "status", "operator": "eq", "value": "draft"}})
        ),
        ["a.md", "b.md"]
    );

    let s = docs(&[
        ("wide.md", json!({"priority": [1, 20]})),
        ("narrow.md", json!({"priority": [5]})),
    ]);
    assert_eq!(
        match_paths(
            &s,
            json!({"operator": "and", "operands": [
                {"field": "priority", "operator": "gte", "value": 3},
                {"field": "priority", "operator": "lt", "value": 10},
            ]})
        ),
        ["narrow.md", "wide.md"]
    );
}

#[test]
fn text_operators_match_string_elements_whole_characters_and_past_nul() {
    let s = docs(&[
        ("sqlite.md", json!({"topics": ["sqlite", "search"]})),
        ("vec.md", json!({"topics": ["sqlite-vec", "embeddings"]})),
        ("pg.md", json!({"topics": "postgres"})),
        ("num.md", json!({"topics": 42})),
        ("bool.md", json!({"topics": true})),
    ]);
    let t =
        |op: &str, v: &str| match_paths(&s, json!({"field": "topics", "operator": op, "value": v}));
    assert_eq!(t("prefix", "sql"), ["sqlite.md", "vec.md"]);
    assert_eq!(t("suffix", "vec"), ["vec.md"]);
    assert_eq!(t("contains", "arch"), ["sqlite.md"]);
    assert!(t("contains", "4").is_empty());
    assert!(t("prefix", "tru").is_empty());

    let s = docs(&[
        ("exact.md", json!({"code": "abc"})),
        ("emoji.md", json!({"code": "\u{1F600}abc"})),
    ]);
    let c =
        |op: &str, v: &str| match_paths(&s, json!({"field": "code", "operator": op, "value": v}));
    assert_eq!(c("prefix", "abc"), ["exact.md"]);
    assert_eq!(c("suffix", "abc"), ["emoji.md", "exact.md"]);
    assert!(c("prefix", "abcd").is_empty());
    assert!(c("suffix", "zabc").is_empty());
    assert_eq!(c("prefix", "\u{1F600}a"), ["emoji.md"]);
    assert_eq!(c("suffix", "\u{1F600}abc"), ["emoji.md"]);

    let s = docs(&[
        ("nul.md", json!({"code": "abc\u{0}XYZ"})),
        ("plain.md", json!({"code": "abc"})),
    ]);
    let c =
        |op: &str, v: &str| match_paths(&s, json!({"field": "code", "operator": op, "value": v}));
    assert_eq!(c("suffix", "XYZ"), ["nul.md"]);
    assert_eq!(c("suffix", "abc"), ["plain.md"]);
    assert_eq!(c("prefix", "abc\u{0}XYZ"), ["nul.md"]);
    assert_eq!(c("suffix", "abc\u{0}XYZ"), ["nul.md"]);
    assert_eq!(c("prefix", "abc\u{0}"), ["nul.md"]);
    assert_eq!(c("contains", "\u{0}X"), ["nul.md"]);
    assert_eq!(c("eq", "abc\u{0}XYZ"), ["nul.md"]);
    assert_eq!(
        match_paths(
            &s,
            json!({"field": "code", "operator": "suffix", "value": "xyz", "caseInsensitive": true})
        ),
        ["nul.md"]
    );
}

#[test]
fn type_tests_the_stored_type() {
    let s = docs(&[
        ("num.md", json!({"priority": 3})),
        ("nums.md", json!({"priority": [1, 2]})),
        ("str.md", json!({"priority": "high"})),
        ("bool.md", json!({"priority": true})),
        ("none.md", json!({"other": 1})),
    ]);
    let t = |v: &str| {
        match_paths(
            &s,
            json!({"field": "priority", "operator": "type", "value": v}),
        )
    };
    assert_eq!(t("number"), ["num.md", "nums.md"]);
    assert_eq!(t("string"), ["str.md"]);
    assert_eq!(t("boolean"), ["bool.md"]);
    assert_eq!(
        match_paths(
            &s,
            json!({"operator": "not", "operand": {"field": "priority", "operator": "type", "value": "string"}})
        ),
        ["bool.md", "none.md", "num.md", "nums.md"]
    );
}

#[test]
fn case_insensitive_folds_ascii_for_every_string_operator() {
    let s = docs(&[
        (
            "upper.md",
            json!({"status": "PUBLISHED", "topics": ["PostgreSQL", "MySQL"]}),
        ),
        (
            "lower.md",
            json!({"status": "published", "topics": ["sqlite"]}),
        ),
        ("draft.md", json!({"status": "Draft", "topics": ["Search"]})),
    ]);
    let ci = |field: &str, op: &str, value: Value| {
        match_paths(
            &s,
            json!({"field": field, "operator": op, "value": value, "caseInsensitive": true}),
        )
    };
    assert!(
        match_paths(
            &s,
            json!({"field": "status", "operator": "eq", "value": "Published"})
        )
        .is_empty()
    );
    assert_eq!(
        ci("status", "eq", json!("Published")),
        ["lower.md", "upper.md"]
    );
    assert_eq!(ci("status", "ne", json!("published")), ["draft.md"]);
    assert_eq!(ci("status", "in", json!(["DRAFT"])), ["draft.md"]);
    assert_eq!(
        ci("status", "nin", json!(["DRAFT"])),
        ["lower.md", "upper.md"]
    );
    assert_eq!(
        ci("topics", "all", json!(["postgresql", "mysql"])),
        ["upper.md"]
    );
    assert_eq!(
        ci("topics", "contains", json!("sql")),
        ["lower.md", "upper.md"]
    );
    assert_eq!(ci("topics", "prefix", json!("postgres")), ["upper.md"]);
    assert_eq!(ci("topics", "suffix", json!("SQL")), ["upper.md"]);
    assert_eq!(ci("status", "lt", json!("M")), ["draft.md"]);

    let s = docs(&[
        ("upper.md", json!({"city": "\u{C9}VORA"})),
        ("lower.md", json!({"city": "\u{E9}vora"})),
    ]);
    let city = |v: &str| {
        match_paths(
            &s,
            json!({"field": "city", "operator": "eq", "value": v, "caseInsensitive": true}),
        )
    };
    assert_eq!(city("\u{C9}vora"), ["upper.md"]);
    assert_eq!(city("\u{E9}VORA"), ["lower.md"]);
}

#[test]
fn injection_payloads_stay_data() {
    let payload = "key'; DROP TABLE documents; --";
    let s = docs(&[("safe.md", json!({payload: "v'; DROP TABLE documents; --"}))]);
    assert_eq!(
        match_paths(
            &s,
            json!({"field": payload, "operator": "eq", "value": "v'; DROP TABLE documents; --"})
        ),
        ["safe.md"]
    );
    let compiled = filter(json!({"field": payload, "operator": "contains", "value": "'; --"}))
        .compile("d", FilterScope::Candidates);
    assert!(!compiled.sql.contains("DROP") && !compiled.sql.contains("'; --"));
}

#[test]
fn filtered_bm25_returns_only_admitted_documents_past_the_unfiltered_window() {
    let store = Store::open_in_memory().unwrap();
    for i in 0..20 {
        insert(
            &store,
            "c",
            &format!("draft{i}.md"),
            "keyword keyword keyword keyword",
            json!({"status": "draft"}),
        );
    }
    insert(
        &store,
        "c",
        "published.md",
        "keyword and a great deal of other prose here",
        json!({"status": "published"}),
    );
    let published = filter(json!({"field": "status", "operator": "eq", "value": "published"}));
    for collection in [None, Some("c")] {
        let hits = store
            .search_fts_filtered("keyword", 5, collection, TermMatch::Any, Some(&published))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].display_path, "c/published.md");
    }
    let nested = filter(json!({"operator": "and", "operands": [
        {"field": "status", "operator": "ne", "value": "published"},
        {"operator": "not", "operand": {"field": "status", "operator": "exists", "value": false}},
    ]}));
    let hits = store
        .search_fts_filtered("keyword", 50, None, TermMatch::Any, Some(&nested))
        .unwrap();
    assert_eq!(hits.len(), 20);
}

#[test]
fn filtered_search_excludes_pending_stale_and_errored_documents() {
    let store = Store::open_in_memory().unwrap();
    insert(&store, "c", "ok.md", "doc", json!({"status": "ok"}));
    let pending = insert(
        &store,
        "c",
        "pending.md",
        "doc pending",
        json!({"status": "ok"}),
    );
    let stale = insert(
        &store,
        "c",
        "stale.md",
        "doc stale",
        json!({"status": "ok"}),
    );
    let errored = insert(
        &store,
        "c",
        "errored.md",
        "doc errored",
        json!({"status": "ok"}),
    );
    let conn = store.conn();
    conn.execute(
        "DELETE FROM document_metadata WHERE document_id = ?1",
        [pending],
    )
    .unwrap();
    conn.execute(
        "UPDATE document_metadata SET extraction_version = 0 WHERE document_id = ?1",
        [stale],
    )
    .unwrap();
    conn.execute(
        "UPDATE document_metadata SET extraction_error = 'boom' WHERE document_id = ?1",
        [errored],
    )
    .unwrap();
    let ok = filter(json!({"field": "status", "operator": "eq", "value": "ok"}));
    let hits = store
        .search_fts_filtered("doc", 10, None, TermMatch::Any, Some(&ok))
        .unwrap();
    assert_eq!(
        hits.iter()
            .map(|h| h.display_path.as_str())
            .collect::<Vec<_>>(),
        ["c/ok.md"]
    );
    assert_eq!(
        store
            .search_fts_filtered("doc", 10, None, TermMatch::Any, None)
            .unwrap()
            .len(),
        4
    );
    assert_eq!(store.count_documents_pending_metadata(None).unwrap(), 2);
    assert_eq!(store.metadata_failures(None).unwrap().len(), 1);
    assert_eq!(
        store
            .count_documents_pending_metadata(Some(&["other".to_owned()]))
            .unwrap(),
        0
    );
}

#[test]
fn a_section_collection_admits_or_rejects_a_file_s_sections_together() {
    let store = Store::open_in_memory().unwrap();
    let body = |status: &str| {
        format!(
            "---\nqmd:\n  metadata:\n    status: {status}\n---\n\n## One\n\nquorum alpha\n\n## Two\n\nquorum beta\n"
        )
    };
    insert(
        &store,
        "s",
        "keep.md",
        &body("published"),
        json!({"status": "published"}),
    );
    insert(
        &store,
        "s",
        "drop.md",
        &body("draft"),
        json!({"status": "draft"}),
    );
    let mut c = qmd::config::Collection::new("/x", "**/*.md");
    c.section_level = Some(2);
    store.upsert_store_collection("s", &c).unwrap();
    store.sync_sections(None).unwrap();

    assert_eq!(
        store
            .search_fts_filtered("quorum", 10, None, TermMatch::Any, None)
            .unwrap()
            .len(),
        4
    );
    let published = filter(json!({"field": "status", "operator": "eq", "value": "published"}));
    for collection in [None, Some("s")] {
        let hits = store
            .search_fts_filtered("quorum", 10, collection, TermMatch::Any, Some(&published))
            .unwrap();
        let mut keys: Vec<String> = hits.iter().map(qmd::store::SearchResult::key).collect();
        keys.sort();
        assert_eq!(keys, ["qmd://s/keep.md#2", "qmd://s/keep.md#3"]);
    }
}

fn unit(hot: usize) -> Vec<f32> {
    let mut v = vec![0.0; 4];
    v[hot] = 1.0;
    v
}

fn embed(store: &Store, body: &str, vector: &[f32]) {
    store
        .insert_embedding(&hash_content(body), 0, 0, vector, "m", "t", 1, "fp")
        .unwrap();
}

#[test]
fn filtered_vector_search_is_exact_over_the_admitted_documents() {
    let store = Store::open_in_memory().unwrap();
    store.ensure_vec_table(4).unwrap();
    for i in 0..30 {
        let body = format!("draft {i}");
        insert(
            &store,
            "c",
            &format!("d{i}.md"),
            &body,
            json!({"status": "draft"}),
        );
        embed(&store, &body, &unit(0));
    }
    insert(
        &store,
        "c",
        "far.md",
        "far body",
        json!({"status": "published"}),
    );
    embed(&store, "far body", &unit(1));
    insert(
        &store,
        "c",
        "shared-yes.md",
        "shared body",
        json!({"status": "published"}),
    );
    insert(
        &store,
        "c",
        "shared-no.md",
        "shared body",
        json!({"status": "draft"}),
    );
    embed(&store, "shared body", &unit(2));

    let published = filter(json!({"field": "status", "operator": "eq", "value": "published"}));
    let hits = store
        .search_vec_filtered(&unit(0), 1, None, Some(&published))
        .unwrap();
    assert_eq!(hits.len(), 1, "the far document is still found");
    let hits = store
        .search_vec_filtered(&unit(0), 10, Some("c"), Some(&published))
        .unwrap();
    let mut paths: Vec<&str> = hits.iter().map(|h| h.display_path.as_str()).collect();
    paths.sort_unstable();
    assert_eq!(paths, ["c/far.md", "c/shared-yes.md"]);
    assert!(
        store
            .search_vec_filtered(&unit(0), 10, Some("other"), Some(&published))
            .unwrap()
            .is_empty()
    );
    let none = filter(json!({"field": "status", "operator": "eq", "value": "missing"}));
    assert!(
        store
            .search_vec_filtered(&unit(0), 10, None, Some(&none))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn metadata_by_filepath_skips_inactive_documents() {
    let store = Store::open_in_memory().unwrap();
    let a = insert(&store, "c", "a.md", "alpha", json!({}));
    insert(&store, "c", "b.md", "beta", json!({}));
    let declared = ExtractionResult {
        metadata: parse_metadata_json(Some(r#"{"status":"ok","n":3}"#)),
        error: None,
        extraction_version: METADATA_EXTRACTION_VERSION,
    };
    store.replace_document_metadata(a, &declared).unwrap();
    let paths = [
        "qmd://c/a.md".to_owned(),
        "qmd://c/b.md".to_owned(),
        "qmd://c/z.md".to_owned(),
    ];
    let found = store.metadata_by_filepath(&paths).unwrap();
    assert_eq!(
        serde_json::to_string(&found["qmd://c/a.md"]).unwrap(),
        r#"{"status":"ok","n":3}"#
    );
    assert!(found["qmd://c/b.md"].is_empty());
    assert!(!found.contains_key("qmd://c/z.md"));
    store.deactivate_document("c", "a.md").unwrap();
    assert!(
        !store
            .metadata_by_filepath(&paths)
            .unwrap()
            .contains_key("qmd://c/a.md")
    );
}
