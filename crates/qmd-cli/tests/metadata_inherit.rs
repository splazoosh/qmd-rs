//! `qmd metadata set/list/rm` against the real binary: an inherited key
//! filters `search`, JSON results name its level, discovery counts it, and
//! removing it takes effect on the next `qmd update` with no file touched.

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

fn write(dir: &Path, rel: &str, body: &str) {
    let p = dir.join("notes").join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, body).unwrap();
}

fn files(dir: &Path, filter: &str) -> Vec<String> {
    let out = ok(
        dir,
        &["search", "keyword", "--format", "json", "--filter", filter],
    );
    let rows: serde_json::Value = serde_json::from_str(&out).unwrap_or_default();
    let mut files: Vec<String> = rows
        .as_array()
        .map(|a| {
            a.iter()
                .map(|r| r["file"].as_str().unwrap().to_owned())
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

#[test]
fn inherited_metadata_from_set_to_rm() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    ok(dir, &["init"]);
    write(dir, "top.md", "# Top\n\nkeyword\n");
    write(dir, "guides/a.md", "# A\n\nkeyword\n");
    write(
        dir,
        "guides/b.md",
        "---\nqmd:\n  metadata:\n    audience: staff\n---\n\n# B\n\nkeyword\n",
    );
    ok(dir, &["collection", "add", "notes", "--name", "notes"]);
    let staff = r#"{"field":"audience","operator":"eq","value":"staff"}"#;
    let public = r#"{"field":"audience","operator":"eq","value":"public"}"#;
    assert_eq!(files(dir, staff), ["notes/guides/b.md"]);

    let set = ok(
        dir,
        &[
            "metadata",
            "set",
            "qmd://notes/guides",
            "audience=public",
            "tags=[a, b]",
        ],
    );
    assert!(set.contains("qmd update"), "{set}");
    ok(dir, &["metadata", "set", "/", "team=ops"]);
    let list = ok(dir, &["metadata", "list"]);
    assert!(list.contains("/ (global)\n  team: \"ops\""), "{list}");
    assert!(
        list.contains("qmd://notes/guides\n  audience: \"public\"\n  tags: [\"a\",\"b\"]"),
        "{list}"
    );
    assert!(files(dir, public).is_empty());

    ok(dir, &["update"]);
    assert_eq!(files(dir, public), ["notes/guides/a.md"]);
    assert_eq!(
        files(dir, staff),
        ["notes/guides/b.md"],
        "the file's own value wins"
    );
    let team = r#"{"field":"team","operator":"eq","value":"ops"}"#;
    assert_eq!(files(dir, team).len(), 3);

    let out = ok(
        dir,
        &["search", "keyword", "--format", "json", "--filter", public],
    );
    let rows: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(
        rows[0]["metadataSources"],
        serde_json::json!({"audience": "qmd://notes/guides/", "tags": "qmd://notes/guides/", "team": "global"})
    );

    let report = ok(
        dir,
        &[
            "collection",
            "metadata",
            "notes",
            "--match",
            r#"{"field":"key","operator":"eq","value":"audience"}"#,
        ],
    );
    assert!(
        report.contains("inherited 1 from qmd://notes/guides/"),
        "{report}"
    );

    let bad = qmd(dir, &["metadata", "set", "/", "x=[[1]]"]);
    assert_eq!(bad.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("nested arrays"));

    ok(dir, &["metadata", "rm", "qmd://notes/guides", "audience"]);
    ok(dir, &["update"]);
    assert!(files(dir, public).is_empty());
    let rm_missing = qmd(dir, &["metadata", "rm", "qmd://notes/guides", "audience"]);
    assert_eq!(rm_missing.status.code(), Some(1));
    ok(dir, &["metadata", "rm", "/"]);
    ok(dir, &["metadata", "rm", "qmd://notes/guides"]);
    assert!(ok(dir, &["metadata", "list"]).starts_with("No inherited metadata"));
}
