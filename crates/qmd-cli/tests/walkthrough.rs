//! Fresh-machine walkthrough smoke test: init → collection add → update →
//! embed → search / vsearch / query, run against the real binary in a temp
//! directory with a project-local `.qmd/` index.
//!
//! Model-free: `QMD_MOCK_EMBEDDER=1` selects the deterministic mock backend,
//! so this runs in plain `cargo test` with no downloads.

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

#[test]
fn fresh_machine_walkthrough() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    let out = ok(dir, &["init"]);
    assert!(out.contains("ready to go with new local index"), "{out}");
    assert!(dir.join(".qmd").join("index.yml").is_file());
    assert!(dir.join(".qmd").join("index.sqlite").is_file());
    let gitignore = std::fs::read_to_string(dir.join(".qmd").join(".gitignore")).unwrap();
    assert_eq!(
        gitignore, "*\n",
        "the index and query log must stay out of git"
    );

    let docs = dir.join("docs");
    std::fs::create_dir_all(docs.join("sub")).unwrap();
    std::fs::write(
        docs.join("raft.md"),
        "# Raft Consensus\n\nRaft elects a leader; followers replicate the log.\n",
    )
    .unwrap();
    std::fs::write(
        docs.join("bread.md"),
        "# Sourdough Basics\n\nFeed the starter, fold the dough, bake at high heat.\n",
    )
    .unwrap();
    std::fs::write(
        docs.join("sub").join("tea.md"),
        "# Brewing Tea\n\nSteep green tea at 80 degrees for two minutes.\n",
    )
    .unwrap();

    let out = ok(dir, &["collection", "add", "docs", "--name", "docs"]);
    assert!(out.contains("Indexed: 3 new"), "{out}");

    let out = ok(dir, &["update"]);
    assert!(out.contains("3 unchanged"), "{out}");

    let out = ok(dir, &["status"]);
    assert!(out.contains("3 files indexed"), "{out}");
    assert!(out.contains("0 embedded"), "{out}");
    assert!(out.contains("need embedding"), "{out}");

    let out = ok(dir, &["search", "leader election raft", "--format", "json"]);
    assert!(out.contains("docs/raft.md"), "{out}");

    let out = qmd(dir, &["vsearch", "raft", "--format", "json"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("run 'qmd embed' first"), "{stderr}");

    let out = ok(dir, &["embed"]);
    assert!(out.contains("Embedded"), "{out}");

    let out = ok(
        dir,
        &[
            "vsearch",
            "raft",
            "--all",
            "--min-score=-1",
            "--format",
            "json",
        ],
    );
    assert!(out.contains("docs/raft.md"), "{out}");

    let out = ok(dir, &["ls", "docs"]);
    assert!(out.contains("qmd://docs/sub/tea.md"), "{out}");
    let out = ok(dir, &["get", "qmd://docs/bread.md"]);
    assert!(out.contains("Sourdough"), "{out}");

    let out = ok(dir, &["query", "sourdough starter", "--format", "json"]);
    assert!(out.contains("docs/bread.md"), "{out}");

    let out = ok(dir, &["status"]);
    assert!(!out.contains("need embedding"), "{out}");

    let log = std::fs::read_to_string(dir.join(".qmd").join("index-queries.jsonl")).unwrap();
    let entries: Vec<serde_json::Value> = log
        .lines()
        .map(|l| serde_json::from_str(l).expect("one JSON object per line"))
        .collect();
    let arms: Vec<&str> = entries.iter().filter_map(|e| e["arm"].as_str()).collect();
    assert_eq!(arms, ["bm25", "vector", "hybrid"], "{log}");
    assert_eq!(entries[2]["query"], "sourdough starter");
    assert!(
        entries[2]["results"][0]
            .as_str()
            .is_some_and(|p| p.starts_with("qmd://docs/")),
        "{log}"
    );

    let out = ok(dir, &["drift"]);
    assert!(out.contains("Index is current with 3 files"), "{out}");

    std::fs::write(docs.join("raft.md"), "# Raft\n\nRewritten.\n").unwrap();
    std::fs::write(docs.join("new.md"), "# New\n\nAdded after indexing.\n").unwrap();
    std::fs::remove_file(docs.join("bread.md")).unwrap();

    let out = ok(dir, &["drift"]);
    assert!(out.contains("1 new, 1 changed, 1 gone"), "{out}");
    let out2 = ok(dir, &["drift"]);
    assert_eq!(out, out2);
    let out = ok(dir, &["search", "sourdough", "--format", "json"]);
    assert!(out.contains("docs/bread.md"), "{out}");

    ok(dir, &["update"]);
    let out = ok(dir, &["drift"]);
    assert!(out.contains("Index is current with 3 files"), "{out}");
}

/// `collection add --ignore` writes the exclusion into the config, so it
/// binds every later `qmd update` and not just the first index.
#[test]
fn ignore_patterns_survive_update() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    ok(dir, &["init"]);

    let docs = dir.join("docs");
    std::fs::create_dir_all(docs.join("target")).unwrap();
    std::fs::write(docs.join("keep.md"), "# Keep\n\nreal prose\n").unwrap();
    std::fs::write(docs.join("target").join("gen.md"), "# Generated\n\nnoise\n").unwrap();

    let out = ok(
        dir,
        &[
            "collection",
            "add",
            "docs",
            "--name",
            "docs",
            "--ignore",
            "**/target/**",
        ],
    );
    assert!(out.contains("Indexed: 1 new"), "{out}");

    let config = std::fs::read_to_string(dir.join(".qmd").join("index.yml")).unwrap();
    assert!(config.contains("**/target/**"), "{config}");

    std::fs::write(docs.join("target").join("more.md"), "# More\n\nnoise\n").unwrap();
    let out = ok(dir, &["update"]);
    assert!(out.contains("0 new, 0 updated, 1 unchanged"), "{out}");
    let out = ok(dir, &["drift"]);
    assert!(out.contains("Index is current"), "{out}");
}

#[test]
fn init_refuses_home_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let out = Command::new(env!("CARGO_BIN_EXE_qmd"))
        .arg("init")
        .current_dir(dir)
        .env("HOME", dir)
        .env("USERPROFILE", dir)
        .output()
        .expect("qmd binary runs");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Refusing to initialize a local index in $HOME"),
        "{stderr}"
    );
}
