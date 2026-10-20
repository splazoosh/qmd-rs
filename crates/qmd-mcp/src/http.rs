//! The HTTP transport: one process, one index handle and one embedder, however
//! many clients connect.
//!
//! Routes, as upstream serves them:
//!
//! - `/mcp` — MCP over Streamable HTTP, with sessions. rmcp's transport.
//! - `POST /query` (alias `/search`) — the `query` tool's parameters in, its
//!   structured rows out, without the protocol.
//! - `GET /health` — liveness, plus the PID that `qmd mcp stop` checks a PID
//!   file against before it kills anything.
//!
//! Loopback only. The listener refuses a non-loopback address, and every route
//! sits behind a DNS-rebinding guard: the `Host` must name a loopback host and
//! an `Origin`, when a browser sends one, must too. Upstream guards the REST
//! routes the same way (its #881); rmcp's own host check covers only `/mcp`.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};
use axum::extract::{Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::{StreamableHttpServerConfig, StreamableHttpService};
use serde_json::json;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::server::QmdServer;
use crate::tools::{self, QueryParams};
use crate::worker::Index;

/// Default port, upstream's.
pub const DEFAULT_PORT: u16 = 8181;

/// How often a running server rebuilds the instructions new sessions receive.
const INSTRUCTIONS_REFRESH: Duration = Duration::from_secs(60);

#[derive(Clone)]
struct AppState {
    server: QmdServer,
    started: Instant,
}

/// A bound, not yet serving, HTTP server.
pub struct HttpServer {
    listener: TcpListener,
    server: QmdServer,
}

impl HttpServer {
    /// Bind `addr`. Port 0 picks a free port; [`Self::local_addr`] says which.
    ///
    /// # Errors
    /// Refuses a non-loopback address, and fails if the address is taken.
    pub async fn bind(server: QmdServer, addr: SocketAddr) -> Result<Self> {
        if !addr.ip().is_loopback() {
            bail!(
                "refusing to listen on {addr}: the server has no authentication, so it binds \
                 loopback addresses only"
            );
        }
        let listener = TcpListener::bind(addr)
            .await
            .with_context(|| format!("failed to listen on {addr}"))?;
        Ok(Self { listener, server })
    }

    /// The address actually bound.
    ///
    /// # Errors
    /// Propagates the socket error.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }

    /// Serve until `shutdown` resolves, then close every session.
    ///
    /// # Errors
    /// Propagates listener failures.
    pub async fn serve(self, shutdown: impl Future<Output = ()> + Send + 'static) -> Result<()> {
        let cancel = CancellationToken::new();
        let router = router(self.server.clone(), cancel.clone());

        let refresher = {
            let server = self.server.clone();
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(INSTRUCTIONS_REFRESH);
                tick.tick().await;
                loop {
                    tick.tick().await;
                    if let Err(e) = server.refresh_instructions().await {
                        eprintln!("qmd: keeping the previous instructions: {e:#}");
                    }
                }
            })
        };

        let result = axum::serve(self.listener, router)
            .with_graceful_shutdown(async move {
                shutdown.await;
                cancel.cancel();
            })
            .await;
        refresher.abort();
        result.context("the HTTP server failed")
    }
}

fn router(server: QmdServer, cancel: CancellationToken) -> Router {
    let mcp = {
        let server = server.clone();
        StreamableHttpService::new(
            move || Ok(server.clone()),
            Arc::new(LocalSessionManager::default()),
            StreamableHttpServerConfig::default().with_cancellation_token(cancel),
        )
    };
    let state = AppState {
        server,
        started: Instant::now(),
    };
    Router::new()
        .route("/health", get(health))
        .route("/query", post(query))
        .route("/search", post(query))
        .nest_service("/mcp", mcp)
        .with_state(state)
        .layer(middleware::from_fn(guard_and_log))
}

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "pid": std::process::id(),
        "uptime": state.started.elapsed().as_secs(),
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn query(State(state): State<AppState>, Json(params): Json<QueryParams>) -> Response {
    match state
        .server
        .worker()
        .run(move |index: &mut Index| tools::query(index, &params))
        .await
    {
        Ok((output, _summary)) => Json(output).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": format!("{e:#}") })),
        )
            .into_response(),
    }
}

/// Screen out DNS rebinding, then log the request to stderr — the daemon's
/// log file.
async fn guard_and_log(request: Request, next: Next) -> Response {
    let started = Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_owned();

    let verdict = screen(request.headers());

    let response = match verdict {
        Ok(()) => next.run(request).await,
        Err(reason) => (
            StatusCode::FORBIDDEN,
            Json(json!({
                "jsonrpc": "2.0",
                "error": { "code": -32003, "message": format!("Forbidden: {reason}") },
                "id": null,
            })),
        )
            .into_response(),
    };
    eprintln!(
        "{method} {path} {} ({}ms)",
        response.status().as_u16(),
        started.elapsed().as_millis()
    );
    response
}

/// Pass a loopback `Host` with no `Origin` or a loopback one; name what failed.
fn screen(headers: &header::HeaderMap) -> Result<(), String> {
    let get = |name| headers.get(name).and_then(|v| v.to_str().ok());
    match (get(header::HOST), get(header::ORIGIN)) {
        (None, _) => Err("no host".to_owned()),
        (Some(host), _) if !is_loopback_authority(host) => Err(format!("host {host}")),
        (_, Some(origin)) if !is_loopback_origin(origin) => Err(format!("origin {origin}")),
        _ => Ok(()),
    }
}

/// `localhost`, `127.0.0.1` or `[::1]`, with or without a port.
fn is_loopback_authority(authority: &str) -> bool {
    let host = if let Some(rest) = authority.strip_prefix('[') {
        rest.split_once(']').map_or(rest, |(h, _)| h)
    } else {
        authority.split_once(':').map_or(authority, |(h, _)| h)
    };
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// An `http`/`https` origin on a loopback host. `Origin: null` is refused.
fn is_loopback_origin(origin: &str) -> bool {
    origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
        .is_some_and(is_loopback_authority)
}

#[cfg(test)]
mod tests {
    use super::{is_loopback_authority, is_loopback_origin};

    #[test]
    fn loopback_authorities_pass_and_others_do_not() {
        for ok in [
            "localhost",
            "localhost:8181",
            "127.0.0.1:8181",
            "[::1]:8181",
            "LOCALHOST",
        ] {
            assert!(is_loopback_authority(ok), "{ok}");
        }
        for bad in [
            "evil.example",
            "evil.example:8181",
            "localhost.evil.example",
            "10.0.0.2",
        ] {
            assert!(!is_loopback_authority(bad), "{bad}");
        }
    }

    #[test]
    fn only_loopback_web_origins_pass() {
        assert!(is_loopback_origin("http://localhost:3000"));
        assert!(is_loopback_origin("https://127.0.0.1"));
        assert!(!is_loopback_origin("http://evil.example"));
        assert!(!is_loopback_origin("null"));
        assert!(!is_loopback_origin("file://localhost"));
    }
}
