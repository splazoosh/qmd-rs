//! `qmd mcp --http --daemon` and `qmd mcp stop`: one long-lived HTTP server per
//! index, found through a PID file.
//!
//! The PID file records the port as well as the PID, because a PID alone goes
//! stale after a reboot and may name an unrelated process. Nothing is treated
//! as running, and nothing is killed, unless `GET /health` on the recorded
//! port answers with the recorded PID.

use std::fs::{self, File};
use std::io::{Read as _, Write as _};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};
use serde_json::{Value, json};

use crate::commands::IndexLocation;

/// How long `--daemon` waits for the child to answer before giving up.
const START_TIMEOUT: Duration = Duration::from_secs(30);

/// The PID and log file of the daemon serving one index.
pub struct DaemonFiles {
    pub pid: PathBuf,
    pub log: PathBuf,
}

impl DaemonFiles {
    /// Beside the index: `mcp.pid` for the default global index,
    /// `mcp-{name}.pid` for a named one (upstream's names), and `mcp.pid` in
    /// `.qmd/` for a project-local index.
    pub fn for_index(location: &IndexLocation) -> Self {
        let (dir, stem) = match location {
            IndexLocation::Global(name) if name == "index" => {
                (qmd::config::cache_dir(), "mcp".to_owned())
            }
            IndexLocation::Global(name) => (qmd::config::cache_dir(), format!("mcp-{name}")),
            IndexLocation::Local(config) => (
                config
                    .parent()
                    .map_or_else(|| PathBuf::from("."), ToOwned::to_owned),
                "mcp".to_owned(),
            ),
        };
        Self {
            pid: dir.join(format!("{stem}.pid")),
            log: dir.join(format!("{stem}.log")),
        }
    }
}

/// What a PID file says, once `/health` has vouched for it.
pub struct Running {
    pub pid: u32,
    pub port: u16,
}

/// The daemon for this index, if one is answering. A PID file nobody answers
/// for is removed, unless a start holding the [`StartLock`] is still waiting
/// for its server to bind.
pub fn running(files: &DaemonFiles) -> Option<Running> {
    let text = fs::read_to_string(&files.pid).ok()?;
    let recorded: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    let pid = recorded["pid"].as_u64().and_then(|p| u32::try_from(p).ok());
    let port = recorded["port"]
        .as_u64()
        .and_then(|p| u16::try_from(p).ok());
    if let (Some(pid), Some(port)) = (pid, port)
        && health_pid(port) == Some(pid)
    {
        return Some(Running { pid, port });
    }
    if !starting(files) {
        let _ = fs::remove_file(&files.pid);
    }
    None
}

/// Whether a `start` is in progress for this index.
fn starting(files: &DaemonFiles) -> bool {
    is_fresh(&StartLock::path(files))
}

/// Whether `path` was written within [`START_TIMEOUT`].
fn is_fresh(path: &Path) -> bool {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age < START_TIMEOUT)
}

/// Remove the PID file only if it still names `pid`: a concurrent start may
/// have replaced it with a live daemon's.
fn remove_if_names(files: &DaemonFiles, pid: u32) {
    let Ok(text) = fs::read_to_string(&files.pid) else {
        return;
    };
    let recorded: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    if recorded["pid"].as_u64() == Some(u64::from(pid)) {
        let _ = fs::remove_file(&files.pid);
    }
}

/// Held for the length of one `start`, so two starts cannot race each other
/// for the PID file. A lock older than [`START_TIMEOUT`] is a crashed start's.
struct StartLock(PathBuf);

impl StartLock {
    fn path(files: &DaemonFiles) -> PathBuf {
        files.pid.with_extension("starting")
    }

    fn acquire(files: &DaemonFiles) -> Result<Self> {
        let path = Self::path(files);
        for _ in 0..2 {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(_) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if is_fresh(&path) {
                        bail!("another 'qmd mcp --http --daemon' is starting; try again shortly");
                    }
                    let _ = fs::remove_file(&path);
                }
                Err(e) => {
                    return Err(e).with_context(|| format!("failed to create {}", path.display()));
                }
            }
        }
        bail!("could not take {}", path.display())
    }
}

impl Drop for StartLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Start `qmd mcp --http` detached, record it, and wait until it answers.
pub fn start(location: &IndexLocation, port: u16) -> Result<()> {
    if port == 0 {
        bail!("--daemon needs a fixed --port: the PID file records the port to find it again");
    }
    let files = DaemonFiles::for_index(location);
    if let Some(dir) = files.pid.parent() {
        fs::create_dir_all(dir)?;
    }
    let _lock = StartLock::acquire(&files)?;
    if let Some(r) = running(&files) {
        bail!(
            "already running (PID {}, port {}). Run 'qmd mcp stop' first",
            r.pid,
            r.port
        );
    }
    let log = File::create(&files.log)
        .with_context(|| format!("failed to create {}", files.log.display()))?;

    let exe = std::env::current_exe().context("failed to locate the qmd binary")?;
    let mut command = Command::new(exe);
    if let IndexLocation::Global(name) = location {
        command.args(["--index", name]);
    }
    command
        .args(["mcp", "--http", "--port", &port.to_string()])
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    let mut child = spawn_detached(&mut command)?;
    let pid = child.id();
    fs::write(&files.pid, json!({ "pid": pid, "port": port }).to_string())
        .with_context(|| format!("failed to write {}", files.pid.display()))?;

    let deadline = Instant::now() + START_TIMEOUT;
    loop {
        if health_pid(port) == Some(pid) {
            break;
        }
        if let Some(status) = child.try_wait()? {
            remove_if_names(&files, pid);
            let tail = fs::read_to_string(&files.log).unwrap_or_default();
            bail!(
                "the server exited ({status}) before answering:\n{}",
                tail.trim_end()
            );
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            remove_if_names(&files, pid);
            bail!(
                "the server (PID {pid}) did not answer on port {port} within {}s; see {}",
                START_TIMEOUT.as_secs(),
                files.log.display()
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    println!("Started on http://127.0.0.1:{port}/mcp (PID {pid})");
    println!("Logs: {}", files.log.display());
    Ok(())
}

/// Stop the daemon for this index, if one is answering.
pub fn stop(location: &IndexLocation) -> Result<()> {
    let files = DaemonFiles::for_index(location);
    let had_file = files.pid.exists();
    let Some(r) = running(&files) else {
        if files.pid.exists() {
            bail!("a server is still starting and not answering yet; try again shortly");
        } else if had_file {
            println!("Cleaned up a stale PID file (the server was not running).");
        } else {
            println!("Not running (no PID file).");
        }
        return Ok(());
    };
    kill(r.pid)?;
    let deadline = Instant::now() + Duration::from_secs(5);
    while health_pid(r.port) == Some(r.pid) {
        if Instant::now() > deadline {
            bail!("PID {} is still answering on port {}", r.pid, r.port);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let _ = fs::remove_file(&files.pid);
    println!("Stopped the QMD MCP server (PID {}).", r.pid);
    Ok(())
}

/// Remove the PID file if it names this process: a foreground server that
/// was started as the daemon cleans up after itself on a clean exit.
pub fn release(location: &IndexLocation) {
    let files = DaemonFiles::for_index(location);
    let Ok(text) = fs::read_to_string(&files.pid) else {
        return;
    };
    let recorded: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    if recorded["pid"].as_u64() == Some(u64::from(std::process::id())) {
        let _ = fs::remove_file(&files.pid);
    }
}

/// The PID `GET /health` reports on a loopback port, or `None` if nothing
/// there answers as a qmd server. Plain HTTP/1.1 over std: this is the only
/// request the CLI makes, and it is to itself.
fn health_pid(port: u16) -> Option<u32> {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let timeout = Duration::from_secs(2);
    let mut stream = TcpStream::connect_timeout(&addr, timeout).ok()?;
    stream.set_read_timeout(Some(timeout)).ok()?;
    stream.set_write_timeout(Some(timeout)).ok()?;
    write!(
        stream,
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    let (head, body) = response.split_once("\r\n\r\n")?;
    if !head.starts_with("HTTP/1.1 200") {
        return None;
    }
    let body: Value = serde_json::from_str(body.trim()).ok()?;
    if body["status"] != "ok" {
        return None;
    }
    body["pid"].as_u64().and_then(|p| u32::try_from(p).ok())
}

/// Spawn with no console and outside the caller's job, so the server outlives
/// the terminal or agent session that started it.
#[cfg(windows)]
fn spawn_detached(command: &mut Command) -> Result<std::process::Child> {
    use std::os::windows::process::CommandExt as _;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const CREATE_BREAKAWAY_FROM_JOB: u32 = 0x0100_0000;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let base = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW;
    keep_own_stdio_out_of_children();
    match command
        .creation_flags(base | CREATE_BREAKAWAY_FROM_JOB)
        .spawn()
    {
        Ok(child) => Ok(child),
        Err(_) => command
            .creation_flags(base)
            .spawn()
            .context("failed to start the server"),
    }
}

/// Windows hands a child every inheritable handle the parent holds, not only
/// the three it is given. Our own stdout is often the caller's pipe, and a
/// daemon holding it open means whoever reads `qmd mcp --http --daemon`'s
/// output never sees EOF — an agent's shell tool waits forever. Unix closes
/// everything but fds 0–2 on exec, so there is no counterpart.
#[cfg(windows)]
fn keep_own_stdio_out_of_children() {
    use std::os::windows::io::AsRawHandle as _;
    const HANDLE_FLAG_INHERIT: u32 = 1;
    unsafe extern "system" {
        fn SetHandleInformation(handle: *mut std::ffi::c_void, mask: u32, flags: u32) -> i32;
    }
    for handle in [
        std::io::stdin().as_raw_handle(),
        std::io::stdout().as_raw_handle(),
        std::io::stderr().as_raw_handle(),
    ] {
        unsafe {
            SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0);
        }
    }
}

#[cfg(unix)]
fn spawn_detached(command: &mut Command) -> Result<std::process::Child> {
    use std::os::unix::process::CommandExt as _;
    command
        .process_group(0)
        .spawn()
        .context("failed to start the server")
}

#[cfg(windows)]
fn kill(pid: u32) -> Result<()> {
    let status = Command::new("taskkill")
        .args(["/F", "/PID", &pid.to_string()])
        .stdout(Stdio::null())
        .status()
        .context("failed to run taskkill")?;
    if !status.success() {
        bail!("taskkill could not stop PID {pid}");
    }
    Ok(())
}

#[cfg(unix)]
fn kill(pid: u32) -> Result<()> {
    let status = Command::new("kill")
        .arg(pid.to_string())
        .status()
        .context("failed to run kill")?;
    if !status.success() {
        bail!("kill could not stop PID {pid}");
    }
    Ok(())
}
