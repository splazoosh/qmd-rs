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

/// Index the project and start a daemon with `env` set, `QMD_MOCK_EMBEDDER`
/// among it. Returns its port and PID.
fn start_daemon(dir: &Path, env: &[(&str, &str)]) -> (String, u32) {
    assert!(qmd(dir, &["update"]).status.success());
    assert!(qmd(dir, &["embed"]).status.success());
    let port = free_port().to_string();
    let started = Command::new(env!("CARGO_BIN_EXE_qmd"))
        .args(["mcp", "--http", "--daemon", "--port", &port])
        .current_dir(dir)
        .env_remove("INDEX_PATH")
        .env_remove("QMD_LLAMA_GPU")
        .env_remove("QMD_MODEL_IDLE_SECONDS")
        .env("QMD_CONFIG_DIR", dir.join("config-unused"))
        .env("XDG_CACHE_HOME", dir.join("cache-unused"))
        .envs(env.iter().copied())
        .output()
        .unwrap();
    assert!(started.status.success());
    let pid = pid_in(&std::fs::read_to_string(dir.join(".qmd/mcp.pid")).unwrap());
    (port, pid)
}

/// The `"pid"` in a PID file or a `/health` body.
fn pid_in(json: &str) -> u32 {
    let digits = json.split(r#""pid":"#).nth(1).unwrap();
    let end = digits.find(|c: char| !c.is_ascii_digit()).unwrap();
    digits[..end].parse().unwrap()
}

fn log(dir: &Path) -> String {
    std::fs::read_to_string(dir.join(".qmd/mcp.log")).unwrap()
}

/// The PIDs of the model hosts the daemon's log says it started, in order.
fn host_pids(dir: &Path) -> Vec<u32> {
    log(dir)
        .lines()
        .filter_map(|line| {
            line.split_once("model host started (PID ")?
                .1
                .split_once(',')?
                .0
                .parse()
                .ok()
        })
        .collect()
}

fn is_running(pid: u32) -> bool {
    if cfg!(windows) {
        let filter = format!("PID eq {pid}");
        let out = Command::new("tasklist")
            .args(["/NH", "/FI", &filter])
            .output()
            .unwrap();
        stdout(&out).contains(&pid.to_string())
    } else {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .output()
            .unwrap()
            .status
            .success()
    }
}

/// Wait up to ten seconds for `done`.
fn eventually(what: &str, done: impl Fn() -> bool) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !done() {
        assert!(std::time::Instant::now() < deadline, "{what}");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

const QUERY: &str = r#"{"query":"leaders"}"#;

#[test]
fn a_daemon_whose_model_host_starts_failing_answers_from_a_new_one() {
    let tmp = project();
    let dir = tmp.path();
    let _cleanup = scopeguard(dir);
    let (port, pid) = start_daemon(dir, &[("QMD_MOCK_EMBEDDER", "fail-after-1")]);

    for _ in 0..3 {
        let answered = post_query(&port, QUERY);
        assert!(answered.starts_with("HTTP/1.1 200"), "{answered}");
    }

    let health = health(&port).unwrap();
    assert_eq!(pid_in(&health), pid);
    assert!(health.contains(r#""status":"ok""#), "{health}");
    assert!(log(dir).contains("mock embedder fault"), "{}", log(dir));
    assert_eq!(host_pids(dir).len(), 3);
}

#[test]
fn a_daemon_whose_model_host_dies_answers_from_a_new_one() {
    let tmp = project();
    let dir = tmp.path();
    let _cleanup = scopeguard(dir);
    let (port, pid) = start_daemon(dir, &[("QMD_MOCK_EMBEDDER", "abort-after-1")]);

    assert!(post_query(&port, QUERY).starts_with("HTTP/1.1 200"));
    let answered = post_query(&port, QUERY);
    assert!(answered.starts_with("HTTP/1.1 200"), "{answered}");

    assert_eq!(pid_in(&health(&port).unwrap()), pid);
    assert!(log(dir).contains("the model host stopped"), "{}", log(dir));
}

#[test]
fn under_auto_a_gpu_no_new_host_can_use_gives_way_to_the_cpu() {
    let tmp = project();
    let dir = tmp.path();
    let _cleanup = scopeguard(dir);
    let (port, _) = start_daemon(dir, &[("QMD_MOCK_EMBEDDER", "fail-on-gpu")]);

    let answered = post_query(&port, QUERY);

    assert!(answered.starts_with("HTTP/1.1 200"), "{answered}");
    assert!(log(dir).contains("GPU mode off"), "{}", log(dir));
}

#[test]
fn under_a_named_gpu_mode_the_loss_is_reported_and_lexical_search_goes_on() {
    let tmp = project();
    let dir = tmp.path();
    let _cleanup = scopeguard(dir);
    let env = [
        ("QMD_MOCK_EMBEDDER", "fail-on-gpu"),
        ("QMD_LLAMA_GPU", "cuda"),
    ];
    let (port, pid) = start_daemon(dir, &env);

    let failed = post_query(&port, QUERY);
    assert!(failed.contains("mock GPU fault"), "{failed}");
    eventually("no host is left, so no model is loaded", || {
        health(&port).is_some_and(|h| h.contains(r#""models":"unloaded""#))
    });
    let health = health(&port).unwrap();
    assert!(health.contains(r#""status":"degraded""#), "{health}");
    assert_eq!(pid_in(&health), pid);

    let lexical = r#"{"searches":[{"type":"lex","query":"leaders"}]}"#;
    assert!(post_query(&port, lexical).starts_with("HTTP/1.1 200"));
}

#[test]
fn an_idle_daemon_ends_its_model_host_and_the_next_query_starts_another() {
    let tmp = project();
    let dir = tmp.path();
    let _cleanup = scopeguard(dir);
    let env = [("QMD_MOCK_EMBEDDER", "1"), ("QMD_MODEL_IDLE_SECONDS", "1")];
    let (port, _) = start_daemon(dir, &env);
    assert!(host_pids(dir).is_empty());

    assert!(post_query(&port, QUERY).starts_with("HTTP/1.1 200"));
    let host = host_pids(dir)[0];
    assert!(is_running(host));

    eventually("the model host outlived the idle time", || {
        !is_running(host)
    });
    assert!(health(&port).unwrap().contains(r#""models":"unloaded""#));

    assert!(post_query(&port, QUERY).starts_with("HTTP/1.1 200"));
    assert_eq!(host_pids(dir).len(), 2);
}

#[test]
fn stopping_the_daemon_ends_its_model_host() {
    let tmp = project();
    let dir = tmp.path();
    let _cleanup = scopeguard(dir);
    let (port, _) = start_daemon(dir, &[("QMD_MOCK_EMBEDDER", "1")]);
    assert!(post_query(&port, QUERY).starts_with("HTTP/1.1 200"));
    let host = host_pids(dir)[0];

    assert!(qmd(dir, &["mcp", "stop"]).status.success());

    eventually("the model host outlived its daemon", || !is_running(host));
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
