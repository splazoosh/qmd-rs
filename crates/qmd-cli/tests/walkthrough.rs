//! Fresh-machine walkthrough smoke test: init → collection add → update →
//! embed → search / vsearch / query, run against the real binary in a temp
//! directory with a project-local `.qmd/` index.
//!
//! Model-free: `QMD_MOCK_EMBEDDER=1` selects the deterministic mock backend,
//! so this runs in plain `cargo test` with no downloads.

use std::path::Path;
use std::process::{Command, Output, Stdio};

fn command(dir: &Path, args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qmd"));
    command
        .args(args)
        .current_dir(dir)
        .env("QMD_MOCK_EMBEDDER", "1")
        .env_remove("INDEX_PATH")
        .env("QMD_CONFIG_DIR", dir.join("config-unused"))
        .env("XDG_CACHE_HOME", dir.join("cache-unused"))
        .env_remove("QMD_EMBED_MODEL");
    command
}

fn qmd(dir: &Path, args: &[&str]) -> Output {
    command(dir, args).output().expect("qmd binary runs")
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

/// Runs `qmd args` with the read end of its stdout closed before qmd writes,
/// as `head` does once it has its lines; returns the exit code and stderr.
fn run_with_closed_stdout(dir: &Path, args: &[&str]) -> (Option<i32>, String) {
    let mut child = command(dir, args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("qmd binary runs");
    drop(child.stdout.take());
    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(!stderr.contains("panicked"), "{stderr}");
    (out.status.code(), stderr)
}

#[test]
fn closed_stdout_exits_zero_for_a_read_only_command() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    ok(dir, &["init"]);
    let (code, stderr) = run_with_closed_stdout(dir, &["status"]);
    assert_eq!(code, Some(0), "stderr: {stderr}");
}

#[test]
fn closed_stdout_exits_141_for_a_state_changing_command() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    ok(dir, &["init"]);
    let docs = dir.join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(docs.join("a.md"), "# A\n\nalpha\n").unwrap();
    let (code, stderr) = run_with_closed_stdout(dir, &["collection", "add", "docs"]);
    assert_eq!(code, Some(141), "stderr: {stderr}");
}

/// A local index over `docs/` with three documents, embedded by the mock.
fn embedded_corpus(dir: &Path) {
    ok(dir, &["init"]);
    let docs = dir.join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(
        docs.join("raft.md"),
        "# Raft Consensus\n\nRaft elects a leader; followers replicate the log.\n",
    )
    .unwrap();
    std::fs::write(
        docs.join("bread.md"),
        "# Sourdough Basics\n\nFeed the starter, fold the dough.\n",
    )
    .unwrap();
    std::fs::write(docs.join("tea.md"), "# Brewing Tea\n\nSteep green tea.\n").unwrap();
    ok(dir, &["collection", "add", "docs", "--name", "docs"]);
    ok(dir, &["embed"]);
}

#[test]
fn query_explain_prints_the_trace_with_and_without_rerank() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    embedded_corpus(dir);

    let plain = ok(dir, &["query", "raft leader", "--format", "json"]);
    assert!(!plain.contains("\"explain\""), "{plain}");
    let plain: serde_json::Value = serde_json::from_str(&plain).unwrap();
    let ranking = |v: &serde_json::Value| -> Vec<(serde_json::Value, serde_json::Value)> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|r| (r["file"].clone(), r["score"].clone()))
            .collect()
    };

    for rerank in ["--no-rerank", "--rerank"] {
        let out = ok(dir, &["query", "raft leader", "--explain", rerank]);
        assert!(out.contains("Explain:"), "{out}");
        assert!(out.contains("fts original #1"), "{out}");
        assert!(out.contains("rrf: base="), "{out}");
        let blend = if rerank == "--rerank" {
            "x rerank"
        } else {
            "(no rerank)"
        };
        assert!(out.contains(blend), "{out}");

        let json = ok(
            dir,
            &[
                "query",
                "raft leader",
                "--explain",
                rerank,
                "--format",
                "json",
            ],
        );
        let rows: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(rows[0]["file"], "docs/raft.md");
        assert_eq!(rows[0]["explain"]["rrf_rank"], 1);
        assert_eq!(
            rows[0]["explain"]["rerank"].is_null(),
            rerank == "--no-rerank"
        );
        if rerank == "--no-rerank" {
            assert_eq!(
                ranking(&rows),
                ranking(&plain),
                "explain changes no ranking"
            );
        }
    }
}

#[test]
fn collection_rename_moves_documents_without_reindexing() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    embedded_corpus(dir);

    let out = ok(dir, &["collection", "rename", "docs", "notes"]);
    assert!(
        out.contains("Renamed collection 'docs' to 'notes' (3 documents)"),
        "{out}"
    );
    let config = std::fs::read_to_string(dir.join(".qmd").join("index.yml")).unwrap();
    assert!(
        config.contains("notes:") && !config.contains("docs:"),
        "{config}"
    );

    let out = ok(dir, &["update"]);
    assert!(out.contains("0 new, 0 updated, 3 unchanged"), "{out}");
    let out = ok(dir, &["status"]);
    assert!(!out.contains("need embedding"), "{out}");
    let out = ok(
        dir,
        &["query", "raft leader", "--format", "files", "-c", "notes"],
    );
    assert!(out.contains("notes/raft.md"), "{out}");

    assert!(
        !qmd(dir, &["collection", "rename", "missing", "x"])
            .status
            .success()
    );
    assert!(
        !qmd(dir, &["collection", "rename", "notes", "bad name"])
            .status
            .success()
    );
}

#[test]
fn collection_rename_leaves_the_index_untouched_when_the_config_cannot_be_saved() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    embedded_corpus(dir);
    let config = dir.join(".qmd").join("index.yml");
    let set_readonly = |readonly| {
        let mut perms = std::fs::metadata(&config).unwrap().permissions();
        perms.set_readonly(readonly);
        std::fs::set_permissions(&config, perms).unwrap();
    };

    set_readonly(true);
    let out = qmd(dir, &["collection", "rename", "docs", "notes"]);
    set_readonly(false);
    assert!(!out.status.success());

    let out = ok(dir, &["update"]);
    assert!(out.contains("0 new, 0 updated, 3 unchanged"), "{out}");
    let out = ok(dir, &["ls"]);
    assert!(
        out.contains("qmd://docs/") && !out.contains("notes"),
        "{out}"
    );
}

#[test]
fn collection_show_update_cmd_include_and_exclude() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    embedded_corpus(dir);

    let out = ok(dir, &["collection", "show", "docs"]);
    assert!(out.contains("Collection: docs"), "{out}");
    assert!(out.contains("Include:  yes (default)"), "{out}");
    assert!(
        out.contains("Chunks:   900 tokens, window boundary"),
        "{out}"
    );
    assert!(out.contains("Files:    3"), "{out}");
    assert!(!out.contains("Update:"), "{out}");

    let out = ok(
        dir,
        &["collection", "update-cmd", "docs", "echo", "hook-ran"],
    );
    assert!(
        out.contains("Set update command for 'docs': echo hook-ran"),
        "{out}"
    );
    let out = ok(dir, &["collection", "show", "docs"]);
    assert!(out.contains("Update:   echo hook-ran"), "{out}");
    let out = ok(dir, &["collection", "update-cmd", "docs"]);
    assert!(out.contains("Cleared update command"), "{out}");
    let config = std::fs::read_to_string(dir.join(".qmd").join("index.yml")).unwrap();
    assert!(!config.contains("update:"), "{config}");

    ok(dir, &["collection", "exclude", "docs"]);
    let out = ok(dir, &["collection", "show", "docs"]);
    assert!(out.contains("Include:  no"), "{out}");
    assert!(
        !qmd(dir, &["query", "raft"]).status.success(),
        "no collection is left to search by default"
    );
    let out = ok(dir, &["query", "raft", "-c", "docs", "--format", "files"]);
    assert!(out.contains("docs/raft.md"), "{out}");
    ok(dir, &["collection", "include", "docs"]);
    let out = ok(dir, &["query", "raft", "--format", "files"]);
    assert!(out.contains("docs/raft.md"), "{out}");

    assert!(
        !qmd(dir, &["collection", "show", "missing"])
            .status
            .success()
    );
}

#[test]
fn update_runs_a_local_update_command_only_when_trusted() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    embedded_corpus(dir);
    ok(
        dir,
        &["collection", "update-cmd", "docs", "echo", "hook-ran"],
    );

    let out = ok(dir, &["update"]);
    assert!(out.contains("Skipped update command"), "{out}");
    assert!(!out.contains("    hook-ran"), "{out}");

    let trusted = |args: &[&str]| {
        command(dir, args)
            .env("QMD_TRUST_LOCAL_CONFIG", "1")
            .output()
            .unwrap()
    };
    let out = trusted(&["update"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{stdout}");
    assert!(stdout.contains("    hook-ran"), "{stdout}");

    ok(
        dir,
        &[
            "collection",
            "update-cmd",
            "docs",
            "echo",
            "\"a b\">\"quoted out.txt\"",
        ],
    );
    let out = trusted(&["update"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let written = std::fs::read_to_string(dir.join("docs/quoted out.txt")).unwrap();
    assert!(
        written.contains("a b") && !written.contains('\\'),
        "{written}"
    );

    ok(dir, &["collection", "update-cmd", "docs", "exit", "3"]);
    let out = trusted(&["update"]);
    assert_eq!(
        out.status.code(),
        Some(3),
        "the command's status stops the update"
    );
}

#[test]
fn cleanup_removes_what_its_dry_run_reports() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    embedded_corpus(dir);

    let out = ok(dir, &["cleanup", "--dry-run"]);
    assert!(
        out.contains("Would remove 0 orphaned embedding chunks"),
        "{out}"
    );

    std::fs::remove_file(dir.join("docs").join("tea.md")).unwrap();
    ok(dir, &["update"]);
    let out = ok(dir, &["cleanup", "--dry-run"]);
    assert!(
        out.contains("Would remove 1 orphaned embedding chunks"),
        "{out}"
    );
    assert!(
        out.contains("Would remove 1 inactive document records"),
        "{out}"
    );

    let out = ok(dir, &["cleanup"]);
    assert!(out.contains("Removed 1 orphaned embedding chunks"), "{out}");
    assert!(out.contains("Removed 1 inactive document records"), "{out}");
    assert!(out.contains("FTS compacted, database vacuumed"), "{out}");
    let out = ok(dir, &["cleanup", "--dry-run"]);
    assert!(
        out.contains("Would remove 0 orphaned embedding chunks"),
        "{out}"
    );
    let out = ok(dir, &["status"]);
    assert!(!out.contains("need embedding"), "{out}");
}
