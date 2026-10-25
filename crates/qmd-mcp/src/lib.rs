//! An MCP server over a qmd index, on stdio or HTTP.
//!
//! Four tools (`query`, `get`, `multi_get`, `status`), the `qmd://{+path}`
//! resource, and instructions generated from the collections that are actually
//! in the index. Port of TS `src/mcp/server.ts`.
//!
//! The index itself is owned by one worker thread ([`worker`]) because neither
//! the SQLite connection nor the embedder is `Sync`. Nothing in the `qmd`
//! library is async.

pub mod http;
pub mod instructions;
pub mod server;
pub mod tools;
pub mod worker;

use std::path::Path;

use anyhow::Result;
use rmcp::ServiceExt as _;

pub use http::HttpServer;
pub use server::QmdServer;
pub use worker::{EmbedderFactory, Factories, Index, RerankerFactory, Worker};

/// Serve MCP over stdin/stdout until the client disconnects.
///
/// Returns when stdin reaches EOF, which is how a client signals shutdown.
///
/// # Errors
/// Fails if the index cannot be opened or the transport fails.
pub async fn serve_stdio(
    config_path: &Path,
    db_path: &Path,
    model: String,
    factories: Factories,
) -> Result<()> {
    let worker = Worker::spawn(config_path, db_path, model, factories)?;
    let server = QmdServer::new(worker).await?;
    let running = server.serve(rmcp::transport::stdio()).await?;
    running.waiting().await?;
    Ok(())
}

/// Serve MCP and the REST routes over HTTP on `addr` until `shutdown`
/// resolves. `on_bound` hears the address actually bound before the first
/// request is accepted.
///
/// # Errors
/// Fails if the index cannot be opened, the address is not loopback or is
/// taken, or the listener fails.
pub async fn serve_http(
    config_path: &Path,
    db_path: &Path,
    model: String,
    factories: Factories,
    addr: std::net::SocketAddr,
    on_bound: impl FnOnce(std::net::SocketAddr),
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let worker = Worker::spawn(config_path, db_path, model, factories)?;
    let server = QmdServer::new(worker).await?;
    let http = HttpServer::bind(server, addr).await?;
    on_bound(http.local_addr()?);
    http.serve(shutdown).await
}
