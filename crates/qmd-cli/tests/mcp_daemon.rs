//! `qmd mcp --http --daemon` and `qmd mcp stop` as a user runs them: the
//! daemon outlives the command that started it, is found again through its PID
//! file, refuses a second start, and a stale PID file is cleaned up rather
//! than acted on. Model-free.

use std::net::TcpListener;
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
        .output()
        .expect("qmd binary runs")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn project() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    assert!(qmd(dir, &["init"]).status.success());
    std::fs::create_dir_all(dir.join("docs")).unwrap();
    std::fs::write(dir.join("docs/raft.md"), "# Raft\n\nLeaders and logs.\n").unwrap();
    assert!(
        qmd(dir, &["collection", "add", "docs", "--name", "docs"])
            .status
            .success()
    );
    tmp
}

#[test]
fn the_daemon_starts_is_found_again_and_stops() {
    let tmp = project();
    let dir = tmp.path();
    let port = free_port().to_string();

    let started = qmd(dir, &["mcp", "--http", "--daemon", "--port", &port]);
    let _cleanup = scopeguard(dir);
    assert!(
        started.status.success(),
        "{}{}",
        stdout(&started),
        String::from_utf8_lossy(&started.stderr)
    );
    assert!(stdout(&started).contains(&format!("127.0.0.1:{port}/mcp")));
    assert!(dir.join(".qmd/mcp.pid").exists());

    let status = stdout(&qmd(dir, &["status"]));
    assert!(
        status.contains("MCP:") && status.contains("running"),
        "{status}"
    );

    let again = qmd(dir, &["mcp", "--http", "--daemon", "--port", &port]);
    assert!(!again.status.success());
    assert!(String::from_utf8_lossy(&again.stderr).contains("already running"));

    let stopped = qmd(dir, &["mcp", "stop"]);
    assert!(stopped.status.success());
    assert!(stdout(&stopped).contains("Stopped"), "{}", stdout(&stopped));
    assert!(!dir.join(".qmd/mcp.pid").exists());
    assert!(!stdout(&qmd(dir, &["status"])).contains("MCP:"));
    assert!(stdout(&qmd(dir, &["mcp", "stop"])).contains("Not running"));
}

#[test]
fn a_stale_pid_file_is_removed_not_acted_on() {
    let tmp = project();
    let dir = tmp.path();
    let stale = format!(r#"{{"pid":{},"port":{}}}"#, std::process::id(), free_port());
    std::fs::write(dir.join(".qmd/mcp.pid"), stale).unwrap();

    let out = qmd(dir, &["mcp", "stop"]);
    assert!(out.status.success());
    assert!(stdout(&out).contains("stale"), "{}", stdout(&out));
    assert!(!dir.join(".qmd/mcp.pid").exists());
}

#[test]
fn a_daemon_on_port_zero_is_refused() {
    let tmp = project();
    let dir = tmp.path();
    let out = qmd(dir, &["mcp", "--http", "--daemon", "--port", "0"]);
    assert!(!out.status.success());
    assert!(!dir.join(".qmd/mcp.pid").exists());
}

/// Runs `qmd mcp stop` in `dir` when dropped.
fn scopeguard(dir: &Path) -> impl Drop + '_ {
    struct Stop<'a>(&'a Path);
    impl Drop for Stop<'_> {
        fn drop(&mut self) {
            let _ = qmd(self.0, &["mcp", "stop"]);
        }
    }
    Stop(dir)
}
