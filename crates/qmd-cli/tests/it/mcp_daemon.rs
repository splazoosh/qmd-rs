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

/// `GET /health` as JSON text, or `None` if nothing answers.
fn health(port: &str) -> Option<String> {
    use std::io::{Read as _, Write as _};
    let mut stream = std::net::TcpStream::connect(format!("127.0.0.1:{port}")).ok()?;
    write!(
        stream,
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    Some(response.split_once("\r\n\r\n")?.1.to_owned())
}

fn post_query(port: &str, body: &str) -> String {
    use std::io::{Read as _, Write as _};
    let mut stream = std::net::TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
    write!(
        stream,
        "POST /query HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    response
}

/// Index the project and start a daemon whose embedder fails after one batch.
/// Returns its port and the PID file's text.
fn start_failing_daemon(dir: &Path) -> (String, String) {
    assert!(qmd(dir, &["update"]).status.success());
    assert!(qmd(dir, &["embed"]).status.success());
    let port = free_port().to_string();
    let started = Command::new(env!("CARGO_BIN_EXE_qmd"))
        .args(["mcp", "--http", "--daemon", "--port", &port])
        .current_dir(dir)
        .env("QMD_MOCK_EMBEDDER", "fail-after-1")
        .env_remove("INDEX_PATH")
        .env("QMD_CONFIG_DIR", dir.join("config-unused"))
        .env("XDG_CACHE_HOME", dir.join("cache-unused"))
        .output()
        .unwrap();
    assert!(started.status.success());
    let first = std::fs::read_to_string(dir.join(".qmd/mcp.pid")).unwrap();
    (port, first)
}

/// The `"pid"` in a PID file or a `/health` body.
fn pid_in(json: &str) -> u32 {
    let digits = json.split(r#""pid":"#).nth(1).unwrap();
    let end = digits.find(|c: char| !c.is_ascii_digit()).unwrap();
    digits[..end].parse().unwrap()
}

/// Wait for a process other than the one `first` names to answer on `port`.
fn await_replacement(port: &str, first: &str) -> u32 {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        if let Some(pid) = health(port).map(|h| pid_in(&h))
            && pid != pid_in(first)
        {
            return pid;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "no replacement answered"
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

#[test]
fn a_daemon_that_loses_its_embedder_hands_over_to_a_new_process() {
    let tmp = project();
    let dir = tmp.path();
    let _cleanup = scopeguard(dir);
    let (port, first) = start_failing_daemon(dir);
    let pid_file = dir.join(".qmd/mcp.pid");

    let query = r#"{"query":"leaders"}"#;
    let worked = post_query(&port, query);
    assert!(worked.starts_with("HTTP/1.1 200"), "{worked}");
    let failed = post_query(&port, query);
    assert!(failed.contains("retry in a few seconds"), "{failed}");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let moved = std::fs::read_to_string(&pid_file).is_ok_and(|now| now != first);
        if moved && health(&port).is_some_and(|h| h.contains(r#""status":"ok""#)) {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "no replacement answered"
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let recovered = post_query(&port, query);
    assert!(recovered.starts_with("HTTP/1.1 200"), "{recovered}");
    let log = std::fs::read_to_string(dir.join(".qmd/mcp.log")).unwrap();
    assert!(log.contains("mock embedder fault"), "{log}");

    let stopped = qmd(dir, &["mcp", "stop"]);
    assert!(stdout(&stopped).contains("Stopped"), "{}", stdout(&stopped));
    assert!(health(&port).is_none());
}

#[test]
fn a_status_during_the_handover_leaves_the_replacement_findable() {
    let tmp = project();
    let dir = tmp.path();
    let _cleanup = scopeguard(dir);
    let (port, first) = start_failing_daemon(dir);
    let query = r#"{"query":"leaders"}"#;
    assert!(post_query(&port, query).starts_with("HTTP/1.1 200"));

    let started = std::time::Instant::now();
    let replacement = std::thread::scope(|scope| {
        for _ in 0..3 {
            scope.spawn(|| {
                while started.elapsed() < std::time::Duration::from_secs(4) {
                    qmd(dir, &["status"]);
                }
            });
        }
        post_query(&port, query);
        await_replacement(&port, &first)
    });

    let recorded = std::fs::read_to_string(dir.join(".qmd/mcp.pid")).unwrap_or_default();
    assert!(recorded.contains(&replacement.to_string()), "{recorded:?}");
}

#[test]
fn a_stop_during_the_handover_is_refused_rather_than_undone() {
    let tmp = project();
    let dir = tmp.path();
    let _cleanup = scopeguard(dir);
    let (port, first) = start_failing_daemon(dir);
    let query = r#"{"query":"leaders"}"#;
    assert!(post_query(&port, query).starts_with("HTTP/1.1 200"));
    post_query(&port, query);

    let stop = qmd(dir, &["mcp", "stop"]);

    assert!(!stop.status.success(), "{}", stdout(&stop));
    await_replacement(&port, &first);
    assert!(stdout(&qmd(dir, &["mcp", "stop"])).contains("Stopped"));
}

#[test]
fn a_config_a_replacement_could_not_load_keeps_the_daemon_up() {
    let tmp = project();
    let dir = tmp.path();
    let _cleanup = scopeguard(dir);
    let (port, _) = start_failing_daemon(dir);
    let query = r#"{"query":"leaders"}"#;
    assert!(post_query(&port, query).starts_with("HTTP/1.1 200"));

    std::fs::write(dir.join(".qmd/index.yml"), "collections: [unclosed").unwrap();
    post_query(&port, query);
    let lexical = r#"{"searches":[{"type":"lex","query":"leaders"}]}"#;
    assert!(post_query(&port, lexical).starts_with("HTTP/1.1 200"));

    let body = health(&port).expect("the daemon ended itself");
    assert!(body.contains("could not restart"), "{body}");
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
