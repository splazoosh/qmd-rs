//! `--filter` and `qmd collection metadata` against the real binary, on
//! upstream's `test/metadata-cli.test.ts` fixture: six notes, five with
//! `qmd.metadata`, one of them spelling `priority` as a string.

use std::path::Path;
use std::process::{Command, Output};

fn qmd(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_qmd"))
        .args(args)
        .current_dir(dir)
        .env("QMD_MOCK_EMBEDDER", "1")
        .env_remove("INDEX_PATH")
        .env("QMD_CONFIG_DIR", dir.join("config-unused"))
        .env("XDG_CACHE_HOME", dir.join("cache-unused"))
        .env_remove("QMD_EMBED_MODEL")
        .output()
        .expect("qmd binary runs")
}

fn ok(dir: &Path, args: &[&str]) -> String {
    let out = qmd(dir, args);
    assert!(
        out.status.success(),
        "qmd {args:?} failed\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Exit 1, nothing on stdout; returns stderr.
fn fails(dir: &Path, args: &[&str]) -> String {
    let out = qmd(dir, args);
    assert_eq!(out.status.code(), Some(1), "qmd {args:?}");
    assert!(
        out.stdout.is_empty(),
        "qmd {args:?}: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn note(dir: &Path, name: &str, metadata: &[&str], title: &str) {
    let mut lines = vec!["---", "qmd:", "  metadata:"];
    lines.extend_from_slice(metadata);
    let block = if metadata.is_empty() {
        String::new()
    } else {
        format!("{}\n---\n\n", lines.join("\n"))
    };
    std::fs::write(
        dir.join("notes").join(name),
        format!("{block}# {title}\n\ncli filter keyword body\n"),
    )
    .unwrap();
}

fn fixture(dir: &Path) {
    ok(dir, &["init"]);
    std::fs::create_dir_all(dir.join("notes")).unwrap();
    note(
        dir,
        "published.md",
        &[
            "    status: published",
            "    topics: [typescript, programming]",
        ],
        "Published doc",
    );
    note(dir, "draft.md", &["    status: draft"], "Draft doc");
    note(dir, "plain.md", &[], "Plain doc");
    note(
        dir,
        "discovery.md",
        &[
            "    status: published",
            "    topics: [typescript, sqlite, search, architecture, embeddings, mcp, cli, testing, agents, indexing, chunking, reranking]",
            "    priority: 5",
            "    lead: docs-team",
        ],
        "Discovery doc",
    );
    note(
        dir,
        "priority.md",
        &[
            "    status: draft",
            "    priority: 1",
            "    reviewed: false",
            "    reviewers: [docs-team, security-team]",
        ],
        "Priority doc",
    );
    note(
        dir,
        "conflict.md",
        &["    status: archived", "    priority: high"],
        "Conflict doc",
    );
    ok(dir, &["collection", "add", "notes", "--name", "notes"]);
}

const KEY_TOPICS: &str = r#"{"field":"key","operator":"eq","value":"topics"}"#;
const KEY_PRIORITY: &str = r#"{"field":"key","operator":"eq","value":"priority"}"#;

fn value_lines(out: &str) -> usize {
    out.lines()
        .filter(|l| l.starts_with("  ") && !l.starts_with("   "))
        .count()
}

#[test]
fn filtered_search_and_discovery() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fixture(dir);

    let json =
        |args: &[&str]| -> serde_json::Value { serde_json::from_str(&ok(dir, args)).unwrap() };
    let published = r#"{"field":"status","operator":"eq","value":"published"}"#;
    let rows = json(&[
        "search",
        "cli filter keyword",
        "--format",
        "json",
        "--filter",
        published,
    ]);
    let mut files: Vec<&str> = rows
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["file"].as_str().unwrap())
        .collect();
    files.sort_unstable();
    assert_eq!(files, ["notes/discovery.md", "notes/published.md"]);
    let row = rows
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["file"] == "notes/published.md")
        .unwrap();
    assert_eq!(
        row["metadata"],
        serde_json::json!({"status": "published", "topics": ["typescript", "programming"]})
    );
    let nested = r#"{"operator":"and","operands":[{"field":"topics","operator":"all","value":["typescript","programming"]},{"operator":"not","operand":{"field":"status","operator":"eq","value":"draft"}}]}"#;
    let rows = json(&[
        "search",
        "cli filter keyword",
        "--format",
        "json",
        "--filter",
        nested,
    ]);
    assert_eq!(rows.as_array().unwrap().len(), 1);
    let none = r#"{"field":"status","operator":"eq","value":"missing"}"#;
    assert_eq!(
        json(&[
            "search",
            "cli filter keyword",
            "--format",
            "json",
            "--filter",
            none
        ]),
        serde_json::json!([])
    );
    let rows = json(&[
        "search",
        "cli filter keyword",
        "--format",
        "json",
        "-n",
        "10",
    ]);
    let plain = rows
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["file"] == "notes/plain.md")
        .unwrap();
    assert!(plain.get("metadata").is_none(), "{plain}");
    ok(dir, &["embed"]);
    for command in ["vsearch", "query"] {
        let rows = json(&[
            command,
            "cli filter keyword",
            "--format",
            "json",
            "--min-score",
            "0",
            "--filter",
            published,
        ]);
        let rows = rows.as_array().unwrap();
        assert!(
            !rows.is_empty() && rows.iter().all(|r| r["metadata"]["status"] == "published"),
            "{command}: {rows:?}"
        );
    }
    let error = fails(dir, &["search", "x", "--filter", "{not json"]);
    assert!(error.contains("Invalid --filter JSON"), "{error}");
    let error = fails(
        dir,
        &[
            "search",
            "x",
            "--filter",
            r#"{"field":"status","operator":"equal","value":"x"}"#,
        ],
    );
    assert!(
        error.contains("Invalid metadata filter at $: unknown operator 'equal'"),
        "{error}"
    );

    let out = ok(dir, &["collection", "metadata", "notes"]);
    assert!(
        out.starts_with("status  string  5 of 6 documents  3 distinct\n  draft      2\n  published  2\n  archived   1\n"),
        "{out}"
    );
    assert!(
        out.contains("topics  string[]  2 of 6 documents  13 distinct"),
        "{out}"
    );
    assert!(
        out.contains("reviewed  boolean  1 of 6 documents\n  false 1"),
        "{out}"
    );
    assert!(out.contains("reviewers  string[]  1 of 6 documents  2 distinct\n  docs-team      1\n  security-team  1"), "{out}");
    assert_eq!(
        ok(dir, &["collection", "metadata"]),
        out,
        "the default collections"
    );

    assert_eq!(
        ok(
            dir,
            &["collection", "metadata", "notes", "--match", KEY_PRIORITY]
        ),
        "priority  number | string  3 of 6 documents\n  number  2 docs  min 1  median 3  max 5\n  string  1 doc   high (1)\n"
    );

    let topics = ok(
        dir,
        &["collection", "metadata", "notes", "--match", KEY_TOPICS],
    );
    assert_eq!(value_lines(&topics), 10);
    assert!(topics.contains("\n  typescript    2\n"), "{topics}");
    assert!(
        topics
            .contains("3 more values, use --value-limit <n>, --value-offset <n>, or --all-values")
    );
    let limited = ok(
        dir,
        &[
            "collection",
            "metadata",
            "notes",
            "--match",
            KEY_TOPICS,
            "--value-limit",
            "2",
        ],
    );
    assert!(limited.contains("11 more values"), "{limited}");
    let paged = ok(
        dir,
        &[
            "collection",
            "metadata",
            "notes",
            "--match",
            KEY_TOPICS,
            "--value-limit",
            "2",
            "--value-offset",
            "11",
        ],
    );
    assert_eq!(value_lines(&paged), 2);
    assert!(!paged.contains("more values"));
    let all = ok(
        dir,
        &[
            "collection",
            "metadata",
            "notes",
            "--match",
            KEY_TOPICS,
            "--all-values",
        ],
    );
    assert_eq!(value_lines(&all), 13);

    let headers = |out: &str| -> Vec<String> {
        out.lines()
            .filter(|l| !l.starts_with(' ') && l.contains(" of 6 documents"))
            .map(|l| l.split("  ").next().unwrap().to_owned())
            .collect()
    };
    let everything = ok(dir, &["collection", "metadata", "notes"]);
    assert_eq!(headers(&everything).len(), 6);
    let page = ok(
        dir,
        &["collection", "metadata", "notes", "--key-limit", "2"],
    );
    assert_eq!(headers(&page), ["status", "priority"]);
    assert!(page.contains("4 more keys, use --key-limit <n>, --key-offset <n>, or --all-keys"));
    let page = ok(
        dir,
        &[
            "collection",
            "metadata",
            "notes",
            "--key-limit",
            "2",
            "--key-offset",
            "2",
        ],
    );
    assert_eq!(headers(&page), ["topics", "lead"]);
    let page = ok(
        dir,
        &[
            "collection",
            "metadata",
            "notes",
            "--key-limit",
            "2",
            "--key-offset",
            "4",
        ],
    );
    assert_eq!(headers(&page), ["reviewed", "reviewers"]);
    assert!(!page.contains("more keys"));
    let past = ok(
        dir,
        &["collection", "metadata", "notes", "--key-offset", "6"],
    );
    assert!(
        past.contains("No keys at --key-offset 6, 6 keys in total."),
        "{past}"
    );
    assert_eq!(
        headers(&ok(
            dir,
            &[
                "collection",
                "metadata",
                "notes",
                "--all-keys",
                "--key-limit",
                "1"
            ]
        ))
        .len(),
        6
    );

    assert_eq!(
        ok(
            dir,
            &[
                "collection",
                "metadata",
                "notes",
                "--match",
                r#"{"field":"value","operator":"eq","value":"docs-team"}"#
            ]
        ),
        "lead  string  1 of 6 documents  1 distinct\n  docs-team  1\n\nreviewers  string[]  1 of 6 documents  1 distinct\n  docs-team  1\n"
    );
    assert_eq!(
        ok(
            dir,
            &[
                "collection",
                "metadata",
                "notes",
                "--match",
                KEY_PRIORITY,
                "--filter",
                published
            ]
        ),
        "filter: 2 of 6 documents\n\npriority  number  1 of 2 documents  1 distinct\n  min 5  median 5  max 5\n  5 (1)\n"
    );
    let sorted = ok(
        dir,
        &[
            "collection",
            "metadata",
            "notes",
            "--match",
            KEY_TOPICS,
            "--sort",
            "value",
            "--value-limit",
            "2",
        ],
    );
    assert!(
        sorted.contains("  agents        1\n  architecture  1\n"),
        "{sorted}"
    );
    let common = ok(
        dir,
        &[
            "collection",
            "metadata",
            "notes",
            "--match",
            KEY_TOPICS,
            "--min-count",
            "2",
        ],
    );
    assert!(
        common.contains("topics  string[]  2 of 6 documents  1 distinct\n  typescript  2\n"),
        "{common}"
    );
    let composed = r#"{"operator":"and","operands":[{"field":"key","operator":"eq","value":"topics"},{"operator":"or","operands":[{"field":"value","operator":"prefix","value":"sql"},{"field":"value","operator":"contains","value":"ARCH","caseInsensitive":true}]}]}"#;
    assert_eq!(
        ok(
            dir,
            &["collection", "metadata", "notes", "--match", composed]
        ),
        "topics  string[]  1 of 6 documents  3 distinct\n  architecture  1\n  search        1\n  sqlite        1\n"
    );
    assert_eq!(
        ok(
            dir,
            &[
                "collection",
                "metadata",
                "notes",
                "--match",
                r#"{"field":"value","operator":"type","value":"number"}"#
            ]
        ),
        "priority  number  2 of 6 documents  2 distinct\n  min 1  median 3  max 5\n  1 (1)  5 (1)\n"
    );
    let nothing = ok(
        dir,
        &[
            "collection",
            "metadata",
            "notes",
            "--match",
            r#"{"field":"key","operator":"prefix","value":"missing-"}"#,
        ],
    );
    assert!(nothing.contains("No metadata matches"), "{nothing}");
    assert_eq!(
        ok(
            dir,
            &[
                "collection",
                "metadata",
                "notes",
                "--filter",
                r#"{"field":"status","operator":"exists","value":false}"#
            ]
        ),
        "filter: 1 of 6 documents\n\nNo metadata matches. Run 'qmd collection metadata' without --match or --filter to see which keys exist.\n"
    );

    assert!(fails(dir, &["collection", "metadata", "missing"]).contains("missing"));
    assert!(
        fails(
            dir,
            &[
                "collection",
                "metadata",
                "notes",
                "--filter",
                r#"{"field":"status","operator":"equal","value":"x"}"#
            ]
        )
        .contains("Invalid metadata filter at $")
    );
    let bad = fails(
        dir,
        &["collection", "metadata", "notes", "--match", "{nope"],
    );
    assert!(
        bad.contains("Invalid --match JSON")
            && bad
                .contains(r#"Example: --match '{"field":"key","operator":"eq","value":"topics"}'"#),
        "{bad}"
    );
    assert!(fails(dir, &["collection", "metadata", "notes", "--match", r#"{"field":"topics","operator":"eq","value":"x"}"#])
        .contains("Invalid metadata match at $: 'topics' is not a field of a metadata entry, expected 'key' or 'value'"));
    assert!(
        fails(
            dir,
            &[
                "collection",
                "metadata",
                "notes",
                "--match",
                r#"{"field":"value","operator":"exists","value":true}"#
            ]
        )
        .contains("'exists' has no meaning for a single metadata entry")
    );
    assert!(
        fails(
            dir,
            &["collection", "metadata", "notes", "--value-limit", "0"]
        )
        .contains("Invalid --value-limit value: 0")
    );
    assert!(
        fails(
            dir,
            &["collection", "metadata", "notes", "--key-limit", "x"]
        )
        .contains("Invalid --key-limit value: x")
    );
    let offset = fails(dir, &["collection", "metadata", "notes", "--key-offset=-1"]);
    assert!(
        offset.contains("Invalid --key-offset value: -1")
            && offset.contains("--key-offset must be a non-negative safe integer"),
        "{offset}"
    );
    assert!(
        fails(
            dir,
            &["collection", "metadata", "notes", "--min-count", "x"]
        )
        .contains("Invalid --min-count value: x")
    );
    assert!(
        fails(dir, &["collection", "metadata", "notes", "--sort", "size"])
            .contains("Invalid --sort value: size")
    );
    for flag in [
        "--key-limit",
        "--key-offset",
        "--value-limit",
        "--value-offset",
        "--min-count",
    ] {
        let error = fails(
            dir,
            &["collection", "metadata", "notes", flag, "9007199254740993"],
        );
        assert!(
            error.contains(&format!("Invalid {flag} value: 9007199254740993"))
                && error.contains("safe integer"),
            "{error}"
        );
    }
    for format in ["json", "files"] {
        assert!(
            fails(
                dir,
                &["collection", "metadata", "notes", "--format", format]
            )
            .contains("is not supported by 'qmd collection metadata'")
        );
    }
    assert!(
        ok(dir, &["collection", "metadata", "notes", "--format", "cli"])
            .contains("topics  string[]")
    );
    assert!(
        fails(dir, &["collection", "metadata", "notes", "-n", "3"])
            .contains("Use --value-limit <n> for values per key, or --key-limit <n> for keys")
    );
    assert!(
        fails(dir, &["collection", "metadata", "notes", "--all"])
            .contains("Use --all-values, --all-keys, or both")
    );

    let list = ok(dir, &["collection", "list"]);
    assert!(
        list.contains("  Metadata: status, priority, topics, lead, reviewed, +1 more\n"),
        "{list}"
    );
    let show = ok(dir, &["collection", "show", "notes"]);
    assert_eq!(
        &show[show.find("  Metadata:").unwrap()..],
        [
            "  Metadata: 6 keys, 5 of 6 documents",
            "    status    string           5 docs   3 distinct  draft (2), published (2), archived (1)",
            r#"    priority  number | string  3 docs   3 distinct  types disagree, see: qmd collection metadata notes --match '{"field":"key","operator":"eq","value":"priority"}'"#,
            "    topics    string[]         2 docs  13 distinct  typescript (2), agents (1), architecture (1), ...",
            "    lead      string           1 doc    1 distinct",
            "    reviewed  boolean          1 doc    1 distinct  false 1",
            "    1 more key, see 'qmd collection metadata notes'",
            "",
        ]
        .join("\n")
    );
    let status = ok(dir, &["status"]);
    assert!(
        status.contains(
            "  Metadata: 6 keys across 5 files (explore with 'qmd collection metadata')\n"
        ),
        "{status}"
    );

    note(dir, "broken.md", &["    status: null"], "Broken doc");
    let out = qmd(dir, &["update"]);
    assert!(String::from_utf8_lossy(&out.stderr).contains("1 file(s) have invalid frontmatter"),);
    let out = qmd(dir, &["collection", "metadata", "notes"]);
    assert!(
        String::from_utf8_lossy(&out.stderr)
            .contains("Warning: 1 document(s) have invalid frontmatter")
    );
    let status = ok(dir, &["status"]);
    assert!(status.contains("Metadata: 1 with invalid frontmatter"));
    assert!(!status.contains("need extraction"), "{status}");
    assert!(ok(dir, &["collection", "show", "notes"]).contains("1 with invalid frontmatter"));
}

fn update_warning(dir: &Path) -> String {
    let out = qmd(dir, &["update"]);
    assert!(out.status.success());
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Frontmatter that does not parse, and has no `qmd` key.
fn broken(dir: &Path, name: &str) {
    std::fs::write(
        dir.join("notes").join(name),
        "---\ntitle: a: b\n---\n\n# Broken\n",
    )
    .unwrap();
}

#[test]
fn update_names_a_failing_file_on_every_run_until_it_is_fixed() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fixture(dir);
    assert_eq!(
        ok(dir, &["collection", "metadata", "--errors"]).trim(),
        "No files have invalid frontmatter."
    );
    broken(dir, "yaml.md");
    let line = "notes/yaml.md: invalid frontmatter YAML: ";

    let first = update_warning(dir);
    assert!(first.contains(line), "{first}");
    assert!(first.contains("at line 2 column"), "{first}");
    assert!(!first.contains("qmd.metadata"), "{first}");

    let second = update_warning(dir);
    assert!(second.contains(line), "{second}");
    let listed = ok(dir, &["collection", "metadata", "--errors"]);
    assert!(listed.starts_with(line), "{listed}");
    assert_eq!(listed.lines().count(), 1, "{listed}");

    std::fs::write(dir.join("notes/yaml.md"), "# Fixed\n").unwrap();
    assert_eq!(update_warning(dir), "");
}

#[test]
fn update_bounds_the_list_and_errors_gives_all_of_it() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fixture(dir);
    for i in 0..12 {
        broken(dir, &format!("bad-{i:02}.md"));
    }
    let warning = update_warning(dir);
    assert!(warning.contains("12 file(s)"), "{warning}");
    assert_eq!(warning.matches("notes/bad-").count(), 10, "{warning}");
    assert!(
        warning.contains("... and 2 more: 'qmd collection metadata --errors'"),
        "{warning}"
    );
    let listed = ok(dir, &["collection", "metadata", "--errors"]);
    assert_eq!(listed.lines().count(), 12, "{listed}");
}

#[test]
fn errors_covers_every_collection_unless_named() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fixture(dir);
    broken(dir, "yaml.md");
    std::fs::create_dir_all(dir.join("quiet")).unwrap();
    ok(dir, &["collection", "add", "quiet", "--name", "quiet"]);
    ok(dir, &["collection", "exclude", "notes"]);
    update_warning(dir);

    assert!(ok(dir, &["collection", "metadata", "--errors"]).starts_with("notes/yaml.md: "));
    assert_eq!(
        ok(dir, &["collection", "metadata", "--errors", "quiet"]).trim(),
        "No files have invalid frontmatter."
    );
    fails(dir, &["collection", "metadata", "--errors", "missing"]);

    let refused = qmd(dir, &["collection", "metadata", "--errors", "--all-keys"]);
    assert_eq!(refused.status.code(), Some(2));
    for flag in [["--match", "{}"], ["--format", "json"]] {
        let args = ["collection", "metadata", "--errors", flag[0], flag[1]];
        assert_eq!(qmd(dir, &args).status.code(), Some(2), "{flag:?}");
    }
}
