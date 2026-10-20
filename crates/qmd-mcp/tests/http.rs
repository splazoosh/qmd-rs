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
    let server = QmdServer::new(fixture.worker()).await.unwrap();
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
