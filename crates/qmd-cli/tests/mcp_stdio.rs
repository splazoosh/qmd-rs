//! `qmd mcp` over a real pipe: the subcommand speaks the protocol on
//! stdin/stdout and exits cleanly when the client closes stdin.
//!
//! The in-process surface tests live in `qmd-mcp`
//! (`crates/qmd-mcp/tests/surface.rs`); what only the binary can show is that
//! the stdio transport is wired and that EOF terminates the process — the
//! behaviour TS needed `registerStdioEofShutdown` to get.
//!
//! No model is loaded: `initialize` and `tools/list` never reach the embedder.

use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

fn qmd(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_qmd"))
        .args(args)
        .current_dir(dir)
        .env("QMD_MOCK_EMBEDDER", "1")
        .env_remove("INDEX_PATH")
        .env("QMD_CONFIG_DIR", dir.join("config-unused"))
        .env("XDG_CACHE_HOME", dir.join("cache-unused"))
        .output()
        .expect("qmd binary runs")
}

#[test]
fn serves_the_protocol_on_stdio_and_exits_on_eof() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    assert!(qmd(dir, &["init"]).status.success());
    std::fs::create_dir_all(dir.join("docs")).unwrap();
    std::fs::write(
        dir.join("docs/raft.md"),
        "# Raft Consensus\n\nRaft elects a leader; followers replicate the log.\n",
    )
    .unwrap();
    assert!(
        qmd(dir, &["collection", "add", "docs", "--name", "docs"])
            .status
            .success()
    );

    let mut child = Command::new(env!("CARGO_BIN_EXE_qmd"))
        .arg("mcp")
        .current_dir(dir)
        .env("QMD_MOCK_EMBEDDER", "1")
        .env_remove("INDEX_PATH")
        .env("QMD_CONFIG_DIR", dir.join("config-unused"))
        .env("XDG_CACHE_HOME", dir.join("cache-unused"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("qmd mcp starts");

    {
        let stdin = child.stdin.as_mut().expect("stdin is piped");
        for line in [
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#,
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        ] {
            writeln!(stdin, "{line}").unwrap();
        }
        stdin.flush().unwrap();
    }
    drop(child.stdin.take());

    let out = child.wait_with_output().expect("qmd mcp exits");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "qmd mcp exited with {:?}\nstdout: {stdout}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(stdout.contains(r#""name":"qmd""#), "{stdout}");
    assert!(stdout.contains("Collections (scope with"), "{stdout}");
    assert!(stdout.contains("docs"), "{stdout}");
    for tool in ["query", "get", "multi_get", "status"] {
        assert!(stdout.contains(&format!(r#""name":"{tool}""#)), "{stdout}");
    }
}
