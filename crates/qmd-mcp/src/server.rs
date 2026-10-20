//! The MCP server: four tools, one resource template, and instructions built
//! from the live index.

use std::sync::{Arc, PoisonError, RwLock};

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, ListResourceTemplatesResult,
    PaginatedRequestParams, ReadResourceRequestParams, ReadResourceResponse, ReadResourceResult,
    ResourceContents, ResourceTemplate, ServerCapabilities, ServerInfo,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData, RoleServer, ServerHandler, tool, tool_handler, tool_router};

use crate::tools::{self, Document, GetParams, MultiGetEntry, MultiGetParams, QueryParams};
use crate::worker::{Index, Worker};

/// URI template for the document resource.
const RESOURCE_TEMPLATE: &str = "qmd://{+path}";

const MARKDOWN: &str = "text/markdown";

/// The MCP server. Holds a handle to the index thread and the instructions
/// computed from the index. Clones share both, so an HTTP transport can hand
/// one to each session.
#[derive(Clone)]
pub struct QmdServer {
    worker: Worker,
    instructions: Arc<RwLock<String>>,
    tool_router: ToolRouter<Self>,
}

/// Turn an internal failure into a tool-level error the caller can read,
/// rather than a protocol error. Nothing here is the client's protocol fault:
/// a bad collection name or a missing document is a result, not a fault.
fn tool_error(e: &anyhow::Error) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(format!("{e:#}"))])
}

/// A document as an embedded resource block.
fn resource_block(doc: &Document) -> ContentBlock {
    ContentBlock::resource(
        ResourceContents::text(doc.text.clone(), doc.uri.clone()).with_mime_type(MARKDOWN),
    )
}

#[tool_router]
impl QmdServer {
    /// Build the server, computing its instructions from the index.
    ///
    /// # Errors
    /// Propagates store failures while reading the index.
    pub async fn new(worker: Worker) -> anyhow::Result<Self> {
        let instructions = worker
            .run(|index: &mut Index| crate::instructions::build(index))
            .await?;
        Ok(Self {
            worker,
            instructions: Arc::new(RwLock::new(instructions)),
            tool_router: Self::tool_router(),
        })
    }

    /// The `instructions` an agent receives on connect.
    pub fn instructions(&self) -> String {
        self.instructions
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Rebuild the instructions from the index as it is now, for sessions
    /// that connect after a collection was added or removed.
    ///
    /// # Errors
    /// Propagates store failures; the previous instructions stay in place.
    pub async fn refresh_instructions(&self) -> anyhow::Result<()> {
        let fresh = self
            .worker
            .run(|index: &mut Index| crate::instructions::build(index))
            .await?;
        *self
            .instructions
            .write()
            .unwrap_or_else(PoisonError::into_inner) = fresh;
        Ok(())
    }

    /// The handle to the index thread, shared by every clone.
    pub fn worker(&self) -> &Worker {
        &self.worker
    }

    #[tool(
        name = "query",
        description = "Search the knowledge base. Pass `query` (plain text) to run BM25 \
                       and vector search on it and fuse the two rankings, or `searches` \
                       to compose typed sub-queries yourself: lex (BM25 keywords, with \
                       \"quoted phrases\" and -negation), vec (a natural-language \
                       question), hyde (50-100 words of the answer you expect). The \
                       first sub-query gets 2x weight. Pass `intent` to disambiguate a \
                       query whose words are used in several domains here. Each result \
                       carries `line`, the 1-indexed line of its best match: read around \
                       it with get(file, fromLine = max(1, line - 20), maxLines = 80).",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn query(&self, Parameters(params): Parameters<QueryParams>) -> CallToolResult {
        match self
            .worker
            .run(move |index: &mut Index| tools::query(index, &params))
            .await
        {
            Ok((output, summary)) => {
                let structured = match serde_json::to_value(&output) {
                    Ok(value) => value,
                    Err(e) => return tool_error(&anyhow::anyhow!(e)),
                };
                let mut result = CallToolResult::success(vec![ContentBlock::text(summary)]);
                result.structured_content = Some(structured);
                result
            }
            Err(e) => tool_error(&e),
        }
    }

    #[tool(
        name = "get",
        description = "Retrieve one document by its path or its #docid from a search \
                       result. Accepts a line-range suffix on `file`: 'repo/doc.md:100' \
                       starts at line 100, 'repo/doc.md:100:40' reads 40 lines from \
                       there. Suggests similar paths when nothing matches.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn get(&self, Parameters(params): Parameters<GetParams>) -> CallToolResult {
        match self
            .worker
            .run(move |index: &mut Index| tools::get(index, &params))
            .await
        {
            Ok(doc) => CallToolResult::success(vec![resource_block(&doc)]),
            Err(e) => tool_error(&e),
        }
    }

    #[tool(
        name = "multi_get",
        description = "Retrieve several documents by glob pattern (e.g. \
                       'docs/2026-08*.md'), comma-separated list of paths or docids, or \
                       a single docid. Documents larger than `maxBytes` come back as a \
                       note naming them instead of their content.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn multi_get(&self, Parameters(params): Parameters<MultiGetParams>) -> CallToolResult {
        match self
            .worker
            .run(move |index: &mut Index| tools::multi_get(index, &params))
            .await
        {
            Ok(entries) => CallToolResult::success(
                entries
                    .iter()
                    .map(|entry| match entry {
                        MultiGetEntry::Doc(doc) => resource_block(doc),
                        MultiGetEntry::Note(text) => ContentBlock::text(text.clone()),
                    })
                    .collect(),
            ),
            Err(e) => tool_error(&e),
        }
    }

    #[tool(
        name = "status",
        description = "Report the state of the index: document count, how much of it is \
                       embedded, whether semantic search is available, and the \
                       collections that are live with their document counts.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn status(&self) -> CallToolResult {
        match self
            .worker
            .run(|index: &mut Index| tools::status(index))
            .await
        {
            Ok((output, summary)) => {
                let structured = match serde_json::to_value(&output) {
                    Ok(value) => value,
                    Err(e) => return tool_error(&anyhow::anyhow!(e)),
                };
                let mut result = CallToolResult::success(vec![ContentBlock::text(summary)]);
                result.structured_content = Some(structured);
                result
            }
            Err(e) => tool_error(&e),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for QmdServer {
    fn get_info(&self) -> ServerInfo {
        let mut implementation = Implementation::from_build_env();
        implementation.name = "qmd".to_owned();
        implementation.version = env!("CARGO_PKG_VERSION").to_owned();
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_server_info(implementation)
        .with_instructions(self.instructions())
    }

    /// Documents are discovered by searching, so there is no flat list — only
    /// the template, as the TypeScript server does.
    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, ErrorData> {
        let template = ResourceTemplate::new(RESOURCE_TEMPLATE, "document")
            .with_title("QMD Document")
            .with_description(
                "A markdown document from the local knowledge base. Use `query` to find one.",
            )
            .with_mime_type(MARKDOWN);
        Ok(ListResourceTemplatesResult::with_all_items(vec![template]))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, ErrorData> {
        let uri = request.uri.clone();
        let path = percent_decode(uri.strip_prefix("qmd://").unwrap_or(&uri));
        let params = GetParams {
            file: path,
            from_line: None,
            max_lines: None,
            line_numbers: Some(true),
        };
        let text = match self
            .worker
            .run(move |index: &mut Index| tools::get(index, &params))
            .await
        {
            Ok(doc) => doc.text,
            Err(e) => format!("{e:#}"),
        };
        Ok(ReadResourceResult::new(vec![
            ResourceContents::text(text, uri).with_mime_type(MARKDOWN),
        ])
        .into())
    }
}

/// Decode `%XX` escapes. Invalid escapes are left as written, so a literal
/// `%` in a filename still resolves.
fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
            if let Some(byte) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| input.to_owned())
}

#[cfg(test)]
mod tests {
    use super::percent_decode;

    #[test]
    fn decodes_escapes_and_leaves_invalid_ones() {
        assert_eq!(percent_decode("qmd-rs/docs/a%20b.md"), "qmd-rs/docs/a b.md");
        assert_eq!(percent_decode("a%zz.md"), "a%zz.md");
        assert_eq!(percent_decode("plain/path.md"), "plain/path.md");
    }
}
