//! The tool surface, exercised by a real MCP client over the real protocol.
//!
//! Client and server are joined by an in-process duplex pipe rather than a
//! child process, so every assertion here goes through `initialize`,
//! `tools/list`, `tools/call` and `resources/read` exactly as a wired client
//! would. Model-free: the embedder factory returns qmd's deterministic
//! `MockEmbedder`.

mod common;

use common::Fixture;
use qmd_mcp::QmdServer;
use rmcp::model::{CallToolRequestParams, ReadResourceRequestParams};
use rmcp::service::RunningService;
use rmcp::{RoleClient, ServiceExt as _};
use serde_json::{Value, json};

/// Start a server on one end of a duplex pipe and an ordinary client on the
/// other. Dropping the client shuts both down.
async fn connect(fixture: &Fixture) -> (RunningService<RoleClient, ()>, String) {
    let server = QmdServer::new(fixture.worker()).await.unwrap();
    let instructions = server.instructions();
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        if let Ok(running) = server.serve(server_io).await {
            let _ = running.waiting().await;
        }
    });
    let client = ().serve(client_io).await.unwrap();
    (client, instructions)
}

/// The concatenated text of every text block in a tool result.
fn text_of(result: &rmcp::model::CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|block| match block {
            rmcp::model::ContentBlock::Text(t) => Some(t.text.clone()),
            rmcp::model::ContentBlock::Resource(r) => match &r.resource {
                rmcp::model::ResourceContents::TextResourceContents { text, .. } => {
                    Some(text.clone())
                }
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

async fn call(
    client: &RunningService<RoleClient, ()>,
    name: &str,
    args: Value,
) -> rmcp::model::CallToolResult {
    let params = CallToolRequestParams::new(name.to_owned())
        .with_arguments(args.as_object().unwrap().clone());
    client.call_tool(params).await.unwrap()
}

#[tokio::test]
async fn initialize_lists_the_four_tools_and_the_resource_template() {
    let fixture = Fixture::build();
    let (client, _) = connect(&fixture).await;

    let mut names: Vec<String> = client
        .list_all_tools()
        .await
        .unwrap()
        .into_iter()
        .map(|t| t.name.to_string())
        .collect();
    names.sort();
    assert_eq!(names, ["get", "multi_get", "query", "status"]);

    let templates = client.list_all_resource_templates().await.unwrap();
    assert_eq!(templates.len(), 1);
    assert_eq!(templates[0].uri_template, "qmd://{+path}");

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn instructions_name_the_collections_that_are_in_the_index() {
    let fixture = Fixture::build();
    let (client, instructions) = connect(&fixture).await;

    assert!(instructions.contains("notes, recipes"), "{instructions}");
    assert!(
        instructions.contains("3 markdown documents"),
        "{instructions}"
    );
    assert!(
        instructions.contains("Context: A test corpus."),
        "{instructions}"
    );
    assert!(
        !instructions.contains(&fixture.config_path.to_string_lossy().to_string()),
        "{instructions}"
    );

    let info = client.peer_info().expect("server info after initialize");
    assert_eq!(info.instructions.as_deref(), Some(instructions.as_str()));

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn query_returns_ranked_rows_as_structured_content() {
    let fixture = Fixture::build();
    let (client, _) = connect(&fixture).await;

    let result = call(&client, "query", json!({ "query": "raft leader election" })).await;
    let structured = result
        .structured_content
        .clone()
        .expect("structured content");
    let rows = structured["results"].as_array().unwrap();
    assert!(!rows.is_empty());
    assert_eq!(rows[0]["file"], "notes/raft.md");
    assert!(rows[0]["docid"].as_str().unwrap().starts_with('#'));
    assert!(rows[0]["line"].as_u64().unwrap() >= 1);
    assert!(text_of(&result).contains("notes/raft.md"));

    let typed = call(
        &client,
        "query",
        json!({ "searches": [{ "type": "lex", "query": "sourdough starter" }] }),
    )
    .await;
    let text = text_of(&typed);
    let rows = typed.structured_content.expect("structured content");
    assert_eq!(rows["results"][0]["file"], "notes/sub/bread.md");
    assert_eq!(
        rows["results"][0]["context"],
        "A test corpus.\n\nbaking notes"
    );
    assert!(
        text.contains("notes/sub/bread.md (") && text.contains("Context: A test corpus."),
        "{text}"
    );

    let scoped = call(
        &client,
        "query",
        json!({ "query": "simmer bones", "collections": ["recipes"] }),
    )
    .await;
    let rows = scoped.structured_content.expect("structured content");
    for row in rows["results"].as_array().unwrap() {
        assert!(row["file"].as_str().unwrap().starts_with("recipes/"));
    }

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn query_takes_match_and_rerank() {
    let fixture = Fixture::build();
    let (client, _) = connect(&fixture).await;

    let lex = |mode: &str| {
        json!({
            "searches": [{ "type": "lex", "query": "sourdough bones" }],
            "match": mode,
        })
    };
    let all = call(&client, "query", lex("all")).await;
    let rows = all.structured_content.expect("structured content");
    assert!(rows["results"].as_array().unwrap().is_empty());
    let any = call(&client, "query", lex("any")).await;
    let rows = any.structured_content.expect("structured content");
    assert_eq!(rows["results"].as_array().unwrap().len(), 2);

    let reranked = call(
        &client,
        "query",
        json!({ "query": "raft leader election", "rerank": true }),
    )
    .await;
    assert_ne!(reranked.is_error, Some(true), "{}", text_of(&reranked));
    let rows = reranked.structured_content.expect("structured content");
    assert_eq!(rows["results"][0]["file"], "notes/raft.md");

    let bad = call(
        &client,
        "query",
        json!({ "query": "raft", "match": "some" }),
    )
    .await;
    assert_eq!(bad.is_error, Some(true));

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn query_returns_the_explain_trace_when_asked() {
    let fixture = Fixture::build();
    let (client, _) = connect(&fixture).await;

    let plain = call(&client, "query", json!({ "query": "raft leader election" })).await;
    let plain_rows = plain
        .structured_content
        .clone()
        .expect("structured content");
    assert!(plain_rows["results"][0].get("explain").is_none());
    assert!(!text_of(&plain).contains("Explain:"));

    for rerank in [false, true] {
        let explained = call(
            &client,
            "query",
            json!({ "query": "raft leader election", "explain": true, "rerank": rerank }),
        )
        .await;
        assert_ne!(explained.is_error, Some(true), "{}", text_of(&explained));
        let rows = explained
            .structured_content
            .clone()
            .expect("structured content");
        let top = &rows["results"][0];
        assert_eq!(top["file"], "notes/raft.md");
        assert_eq!(top["explain"]["rrf_rank"], 1);
        assert!(
            top["explain"]["contributions"][0]["rank"]
                .as_u64()
                .is_some()
        );
        assert_eq!(top["explain"]["rerank"].is_null(), !rerank, "{top}");
        assert!(text_of(&explained).contains("Explain:"));
        if !rerank {
            let files = |v: &serde_json::Value| -> Vec<serde_json::Value> {
                v["results"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|r| r["file"].clone())
                    .collect()
            };
            assert_eq!(
                files(&rows),
                files(&plain_rows),
                "explain changes no ranking"
            );
        }
    }

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn query_rejects_both_arms_at_once_as_a_tool_error() {
    let fixture = Fixture::build();
    let (client, _) = connect(&fixture).await;

    let result = call(
        &client,
        "query",
        json!({ "query": "raft", "searches": [{ "type": "lex", "query": "raft" }] }),
    )
    .await;
    assert_eq!(result.is_error, Some(true));
    assert!(text_of(&result).contains("mutually exclusive"));

    let neither = call(&client, "query", json!({})).await;
    assert_eq!(neither.is_error, Some(true));

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn get_reads_a_document_by_path_docid_and_line_range() {
    let fixture = Fixture::build();
    let (client, _) = connect(&fixture).await;

    let whole = call(&client, "get", json!({ "file": "notes/raft.md" })).await;
    let text = text_of(&whole);
    assert!(text.contains("Raft elects a leader"), "{text}");
    assert!(text.contains("1: # Raft Consensus"), "{text}");

    let found = call(&client, "query", json!({ "query": "raft leader election" })).await;
    let docid = found.structured_content.unwrap()["results"][0]["docid"]
        .as_str()
        .unwrap()
        .to_owned();
    let by_docid = call(&client, "get", json!({ "file": docid })).await;
    assert!(text_of(&by_docid).contains("Raft elects a leader"));

    let window = call(&client, "get", json!({ "file": "notes/raft.md:3:1" })).await;
    let text = text_of(&window);
    assert!(text.contains("3: Raft elects a leader"), "{text}");
    assert!(!text.contains("# Raft Consensus"), "{text}");

    let raw = call(
        &client,
        "get",
        json!({ "file": "notes/raft.md", "lineNumbers": false }),
    )
    .await;
    assert!(!text_of(&raw).contains("1: #"));

    let contextual = call(&client, "get", json!({ "file": "notes/sub/bread.md" })).await;
    assert!(text_of(&contextual).contains("<!-- Context: "));

    let missing = call(&client, "get", json!({ "file": "notes/nope.md" })).await;
    assert_eq!(missing.is_error, Some(true));
    assert!(text_of(&missing).contains("document not found"));

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn multi_get_takes_globs_lists_and_a_size_limit() {
    let fixture = Fixture::build();
    let (client, _) = connect(&fixture).await;

    let globbed = call(&client, "multi_get", json!({ "pattern": "**/*.md" })).await;
    let text = text_of(&globbed);
    assert!(text.contains("Raft elects a leader"), "{text}");
    assert!(text.contains("Feed the starter"), "{text}");
    assert!(text.contains("Simmer bones"), "{text}");

    let listed = call(
        &client,
        "multi_get",
        json!({ "pattern": "notes/raft.md, notes/absent.md" }),
    )
    .await;
    let text = text_of(&listed);
    assert!(text.contains("Raft elects a leader"), "{text}");
    assert!(text.contains("file not found: notes/absent.md"), "{text}");

    let skipped = call(
        &client,
        "multi_get",
        json!({ "pattern": "**/*.md", "maxBytes": 10 }),
    )
    .await;
    let text = text_of(&skipped);
    assert!(text.contains("[SKIPPED: notes/raft.md"), "{text}");
    assert!(!text.contains("Raft elects a leader"), "{text}");

    let clipped = call(
        &client,
        "multi_get",
        json!({ "pattern": "notes/raft.md", "maxLines": 1 }),
    )
    .await;
    assert!(text_of(&clipped).contains("truncated"));

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn status_counts_the_index_without_naming_a_directory() {
    let fixture = Fixture::build();
    let (client, _) = connect(&fixture).await;

    let result = call(&client, "status", json!({})).await;
    let status = result.structured_content.expect("structured content");
    assert_eq!(status["totalDocuments"], 3);
    assert_eq!(status["hasVectorIndex"], true);
    assert_eq!(status["needsEmbedding"], 0);
    let names: Vec<&str> = status["collections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["notes", "recipes"]);

    let rendered = serde_json::to_string(&status).unwrap();
    let root = fixture.db_path.parent().unwrap().to_string_lossy();
    assert!(!rendered.contains(root.as_ref()), "{rendered}");

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn the_resource_reads_a_document_and_reports_a_miss_as_content() {
    let fixture = Fixture::build();
    let (client, _) = connect(&fixture).await;

    let read = client
        .read_resource(ReadResourceRequestParams::new("qmd://notes/raft.md"))
        .await
        .unwrap();
    let rmcp::model::ResourceContents::TextResourceContents { text, uri, .. } = &read.contents[0]
    else {
        panic!("expected text contents");
    };
    assert_eq!(uri, "qmd://notes/raft.md");
    assert!(text.contains("1: # Raft Consensus"), "{text}");

    let encoded = client
        .read_resource(ReadResourceRequestParams::new("qmd://notes/sub%2Fbread.md"))
        .await
        .unwrap();
    let rmcp::model::ResourceContents::TextResourceContents { text, .. } = &encoded.contents[0]
    else {
        panic!("expected text contents");
    };
    assert!(text.contains("Feed the starter"), "{text}");

    let missing = client
        .read_resource(ReadResourceRequestParams::new("qmd://notes/absent.md"))
        .await
        .unwrap();
    let rmcp::model::ResourceContents::TextResourceContents { text, .. } = &missing.contents[0]
    else {
        panic!("expected text contents");
    };
    assert!(text.contains("document not found"), "{text}");

    client.cancel().await.unwrap();
}
