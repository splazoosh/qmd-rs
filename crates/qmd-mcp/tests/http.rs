//! The HTTP transport: two MCP clients share one server, the REST routes answer
//! without the protocol, and the DNS-rebinding guard refuses a foreign `Host`
//! or `Origin`. Model-free, over a real socket on a free loopback port.

mod common;

use std::net::SocketAddr;

use common::Fixture;
use qmd_mcp::{HttpServer, QmdServer};
use rmcp::ServiceExt as _;
use rmcp::model::CallToolRequestParams;
use rmcp::transport::StreamableHttpClientTransport;
use serde_json::{Value, json};
use tokio::sync::oneshot;

/// Serve the fixture on a free port. Dropping the sender stops the server.
async fn start(fixture: &Fixture) -> (SocketAddr, oneshot::Sender<()>) {
    serve(fixture.worker()).await
}

#[tokio::test]
async fn two_mcp_clients_share_one_server() {
    let fixture = Fixture::build();
    let (addr, _stop) = start(&fixture).await;
    let url = format!("http://{addr}/mcp");

    let first = ().serve(StreamableHttpClientTransport::from_uri(url.clone())).await.unwrap();
    let second = ().serve(StreamableHttpClientTransport::from_uri(url)).await.unwrap();

    let instructions = first.peer_info().unwrap().instructions.clone().unwrap();
    assert!(instructions.contains("notes, recipes"), "{instructions}");

    for client in [&first, &second] {
        let result = client
            .call_tool(
                CallToolRequestParams::new("query").with_arguments(
                    json!({ "searches": [{ "type": "lex", "query": "raft" }] })
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
            )
            .await
            .unwrap();
        let rows = &result.structured_content.unwrap()["results"];
        assert_eq!(rows[0]["file"], "notes/raft.md", "{rows}");
    }

    first.cancel().await.unwrap();
    second.cancel().await.unwrap();
}

#[tokio::test]
async fn rest_routes_answer_without_the_protocol() {
    let fixture = Fixture::build();
    let (addr, _stop) = start(&fixture).await;
    let http = reqwest::Client::new();

    let health: Value = http
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(health["pid"], std::process::id());

    for route in ["query", "search"] {
        let response = http
            .post(format!("http://{addr}/{route}"))
            .json(&json!({ "searches": [{ "type": "lex", "query": "sourdough" }] }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "/{route}");
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["results"][0]["file"], "notes/sub/bread.md", "{body}");
    }

    let response = http
        .post(format!("http://{addr}/query"))
        .json(&json!({ "query": "raft", "collections": ["nope"] }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("nope"), "{body}");
}

#[tokio::test]
async fn a_foreign_host_or_origin_is_refused_on_every_route() {
    let fixture = Fixture::build();
    let (addr, _stop) = start(&fixture).await;
    let http = reqwest::Client::new();

    for (header, value) in [
        ("host", "evil.example:8181"),
        ("origin", "http://evil.example"),
    ] {
        for (method, route) in [("GET", "health"), ("POST", "query"), ("POST", "mcp")] {
            let response = http
                .request(method.parse().unwrap(), format!("http://{addr}/{route}"))
                .header(header, value)
                .json(&json!({}))
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                403,
                "{method} /{route} with {header}: {value}"
            );
        }
    }

    let local = http
        .get(format!("http://{addr}/health"))
        .header("origin", "http://localhost:3000")
        .send()
        .await
        .unwrap();
    assert_eq!(local.status(), 200);
}

#[tokio::test]
async fn a_collection_added_after_start_is_served() {
    let fixture = Fixture::build();
    let (addr, _stop) = start(&fixture).await;

    let root = fixture.config_path.parent().unwrap();
    let travel = root.join("travel");
    std::fs::create_dir_all(&travel).unwrap();
    std::fs::write(
        travel.join("oslo.md"),
        "# Oslo\n\nThe fjord and the ferries.\n",
    )
    .unwrap();
    let mut config = qmd::config::load_config(&fixture.config_path).unwrap();
    let mut added = config.collections["recipes"].clone();
    added.path = travel.to_string_lossy().into_owned();
    config.collections.insert("travel".to_owned(), added);
    qmd::config::save_config(&fixture.config_path, &config).unwrap();
    let store = qmd::store::Store::open(&fixture.db_path).unwrap();
    store.sync_config_to_db(&config).unwrap();
    qmd::index::reindex_collection(&store, &travel, "**/*.md", "travel", &[], None).unwrap();

    let response = reqwest::Client::new()
        .post(format!("http://{addr}/query"))
        .json(&json!({ "searches": [{ "type": "lex", "query": "fjord" }], "collections": ["travel"] }))
        .send()
        .await
        .unwrap();
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["results"][0]["file"], "travel/oslo.md", "{body}");
}

#[tokio::test]
async fn a_non_loopback_address_is_refused() {
    let fixture = Fixture::build();
    let server = QmdServer::new(fixture.worker()).await.unwrap();
    let Err(e) = HttpServer::bind(server, "0.0.0.0:0".parse().unwrap()).await else {
        panic!("bound a wildcard address");
    };
    assert!(format!("{e:#}").contains("loopback"), "{e:#}");
}

#[tokio::test]
async fn rest_query_filters_on_inherited_metadata() {
    let fixture = Fixture::with_inherited_metadata();
    let (addr, _stop) = start(&fixture).await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/query"))
        .json(&json!({
            "searches": [{ "type": "lex", "query": "quorum" }],
            "filter": {"field": "area", "operator": "eq", "value": "elections"},
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.unwrap();
    let rows = body["results"].as_array().unwrap();
    assert_eq!(rows.len(), 2, "{body}");
    for row in rows {
        assert_eq!(row["metadataSources"]["area"], "qmd://notes/meta/", "{row}");
    }
}

#[tokio::test]
async fn rest_routes_take_a_metadata_filter_and_report_metadata() {
    let fixture = Fixture::with_metadata();
    let (addr, _stop) = start(&fixture).await;
    let http = reqwest::Client::new();
    let post = |route: &str, body: Value| {
        http.post(format!("http://{addr}/{route}"))
            .json(&body)
            .send()
    };

    let published = json!({"field": "status", "operator": "eq", "value": "published"});
    for route in ["query", "search"] {
        let response = post(
            route,
            json!({ "searches": [{ "type": "lex", "query": "quorum" }], "filter": published }),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), 200, "/{route}");
        let body: Value = response.json().await.unwrap();
        let rows = body["results"].as_array().unwrap();
        assert_eq!(rows.len(), 1, "{body}");
        assert_eq!(rows[0]["metadata"]["status"], "published");
    }
    let response = post(
        "query",
        json!({ "query": "quorum", "filter": {"field": "status"} }),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 400);
    let response = post("query", json!({ "query": "quorum", "filter": [1] }))
        .await
        .unwrap();
    assert!(response.status().is_client_error());

    let response = post(
        "metadata",
        json!({ "match": {"field": "key", "operator": "eq", "value": "status"} }),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["keys"][0]["key"], "status", "{body}");
    assert_eq!(
        body["keys"][0]["types"][0]["values"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    for bad in [
        json!({ "match": {"field": "nope", "operator": "eq", "value": 1} }),
        json!({ "sort": "size" }),
    ] {
        let response = post("metadata", bad).await.unwrap();
        assert!(response.status().is_client_error(), "{}", response.status());
    }
}

/// Serve `worker` on a free port. Dropping the sender stops the server.
async fn serve(worker: qmd_mcp::Worker) -> (SocketAddr, oneshot::Sender<()>) {
    let server = QmdServer::new(worker).await.unwrap();
    let http = HttpServer::bind(server, "127.0.0.1:0".parse().unwrap())
        .await
        .unwrap();
    let addr = http.local_addr().unwrap();
    let (stop, stopped) = oneshot::channel::<()>();
    tokio::spawn(http.serve(async move {
        let _ = stopped.await;
    }));
    (addr, stop)
}

fn failing_after(batches: usize) -> qmd_mcp::EmbedderFactory {
    Box::new(move |_: &str| {
        Ok(Box::new(qmd::embed::MockEmbedder::failing_after(batches))
            as Box<dyn qmd::embed::EmbeddingBackend>)
    })
}

/// `POST /query` with plain text: the status and the body.
async fn plain_query(addr: SocketAddr) -> (u16, Value) {
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/query"))
        .json(&json!({ "query": "raft leader" }))
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

async fn health(addr: SocketAddr) -> Value {
    reqwest::get(format!("http://{addr}/health"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

#[tokio::test]
async fn health_reports_an_embedder_that_stopped_working_and_lexical_search_goes_on() {
    let fixture = Fixture::build();
    let (addr, _stop) = serve(fixture.worker_with(failing_after(1), None)).await;
    assert_eq!(health(addr).await["embedder"], "idle");

    assert_eq!(plain_query(addr).await.0, 200);
    let body = health(addr).await;
    assert_eq!(
        (&body["status"], &body["embedder"]),
        (&json!("ok"), &json!("ok"))
    );

    let (status, body) = plain_query(addr).await;
    assert_eq!(status, 400);
    let error = body["error"].as_str().unwrap();
    assert!(error.contains("mock embedder fault"), "{error}");
    assert!(error.contains("restarted"), "{error}");

    let body = health(addr).await;
    assert_eq!(body["status"], "degraded");
    assert_eq!(body["embedder"], "failed");
    assert!(
        body["embedderError"]
            .as_str()
            .unwrap()
            .contains("mock embedder fault")
    );

    let lexical = reqwest::Client::new()
        .post(format!("http://{addr}/query"))
        .json(&json!({ "searches": [{ "type": "lex", "query": "raft" }] }))
        .send()
        .await
        .unwrap();
    assert_eq!(lexical.status().as_u16(), 200);
}

#[tokio::test]
async fn a_recovered_embedder_clears_the_degraded_state() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let loaded_once = AtomicBool::new(false);
    let factory: qmd_mcp::EmbedderFactory = Box::new(move |_: &str| {
        if loaded_once.swap(true, Ordering::Relaxed) {
            Ok(Box::new(qmd::embed::MockEmbedder::default())
                as Box<dyn qmd::embed::EmbeddingBackend>)
        } else {
            anyhow::bail!("no memory for the model")
        }
    });
    let fixture = Fixture::build();
    let (addr, _stop) = serve(fixture.worker_with(factory, None)).await;

    assert_eq!(plain_query(addr).await.0, 400);
    assert_eq!(health(addr).await["status"], "degraded");
    assert_eq!(plain_query(addr).await.0, 200);
    assert_eq!(health(addr).await["status"], "ok");
}

/// A relaunch hook that counts its calls and returns `result`.
fn counting_relaunch(
    result: fn() -> anyhow::Result<()>,
) -> (
    qmd_mcp::Relaunch,
    std::sync::Arc<std::sync::atomic::AtomicUsize>,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = std::sync::Arc::new(AtomicUsize::new(0));
    let counted = calls.clone();
    let hook = Box::new(move || {
        counted.fetch_add(1, Ordering::Relaxed);
        result()
    });
    (hook, calls)
}

#[tokio::test]
async fn a_daemon_relaunches_only_after_it_has_embedded() {
    use std::sync::atomic::Ordering;
    let fixture = Fixture::build();

    let (hook, calls) = counting_relaunch(|| Ok(()));
    let (addr, _stop) = serve(fixture.worker_with(failing_after(0), Some(hook))).await;
    assert_eq!(plain_query(addr).await.0, 400);
    plain_query(addr).await;
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    let (hook, calls) = counting_relaunch(|| Ok(()));
    let (addr, _stop) = serve(fixture.worker_with(failing_after(1), Some(hook))).await;
    assert_eq!(plain_query(addr).await.0, 200);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let (_, body) = plain_query(addr).await;
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("retry in a few seconds"),
        "{body}"
    );
    plain_query(addr).await;
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn a_failed_relaunch_is_reported() {
    let fixture = Fixture::build();
    let (hook, _calls) = counting_relaunch(|| anyhow::bail!("no such binary"));
    let (addr, _stop) = serve(fixture.worker_with(failing_after(1), Some(hook))).await;
    plain_query(addr).await;
    plain_query(addr).await;
    reqwest::Client::new()
        .post(format!("http://{addr}/query"))
        .json(&json!({ "searches": [{ "type": "lex", "query": "raft" }] }))
        .send()
        .await
        .unwrap();
    let body = health(addr).await;
    assert_eq!(body["status"], "degraded");
    assert!(
        body["embedderError"]
            .as_str()
            .unwrap()
            .contains("no such binary"),
        "{body}"
    );
}
