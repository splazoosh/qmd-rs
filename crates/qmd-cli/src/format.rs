//! Output formatting: cli / json / files, ported from the TS formatter and
//! `outputResults`.

use std::io::IsTerminal;

use qmd::query::HybridResult;
use qmd::snippet::{add_line_numbers, extract_snippet_at};
use qmd::store::SearchResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    Cli,
    Json,
    Files,
}

impl OutputFormat {
    /// Default result limit: 20 for machine formats, 5 for cli.
    pub fn default_limit(self) -> usize {
        match self {
            Self::Cli => 5,
            Self::Json | Self::Files => 20,
        }
    }
}

/// ANSI color helpers, active only on a terminal.
pub struct Colors {
    pub cyan: &'static str,
    pub bold: &'static str,
    pub dim: &'static str,
    pub yellow: &'static str,
    pub green: &'static str,
    pub red: &'static str,
    pub reset: &'static str,
}

pub fn colors() -> Colors {
    if std::io::stdout().is_terminal() {
        Colors {
            cyan: "\x1b[36m",
            bold: "\x1b[1m",
            dim: "\x1b[2m",
            yellow: "\x1b[33m",
            green: "\x1b[32m",
            red: "\x1b[31m",
            reset: "\x1b[0m",
        }
    } else {
        Colors {
            cyan: "",
            bold: "",
            dim: "",
            yellow: "",
            green: "",
            red: "",
            reset: "",
        }
    }
}

pub fn print_empty_results(format: OutputFormat) {
    match format {
        OutputFormat::Json => println!("[]"),
        OutputFormat::Files => {}
        OutputFormat::Cli => println!("No results found."),
    }
}

/// Common shape for search/vsearch/query output rows. Hybrid results carry
/// a best-chunk hint that focuses snippet extraction.
pub struct ResultView<'a> {
    pub docid: &'a str,
    pub display_path: &'a str,
    pub title: &'a str,
    pub body: &'a str,
    pub score: f64,
    pub context: Option<&'a str>,
    pub chunk_pos: Option<usize>,
    pub chunk_len: Option<usize>,
}

impl<'a> From<&'a SearchResult> for ResultView<'a> {
    fn from(r: &'a SearchResult) -> Self {
        ResultView {
            docid: &r.docid,
            display_path: &r.display_path,
            title: &r.title,
            body: &r.body,
            score: r.score,
            context: r.context.as_deref(),
            chunk_pos: None,
            chunk_len: None,
        }
    }
}

impl<'a> From<&'a HybridResult> for ResultView<'a> {
    fn from(r: &'a HybridResult) -> Self {
        ResultView {
            docid: &r.docid,
            display_path: &r.display_path,
            title: &r.title,
            body: &r.body,
            score: r.score,
            context: r.context.as_deref(),
            chunk_pos: Some(r.best_chunk_pos),
            chunk_len: Some(r.best_chunk.len()),
        }
    }
}

/// Print BM25/vector search results. `results` must already be
/// min-score-filtered and limited.
pub fn print_search_results(
    results: &[SearchResult],
    query: &str,
    format: OutputFormat,
    full: bool,
) {
    let views: Vec<ResultView<'_>> = results.iter().map(ResultView::from).collect();
    print_result_views(&views, query, format, full, None);
}

/// Print hybrid query results, using each result's best-chunk hint for
/// snippet focus.
pub fn print_query_results(
    results: &[HybridResult],
    query: &str,
    format: OutputFormat,
    full: bool,
    intent: Option<&str>,
) {
    let views: Vec<ResultView<'_>> = results.iter().map(ResultView::from).collect();
    print_result_views(&views, query, format, full, intent);
}

fn print_result_views(
    results: &[ResultView<'_>],
    query: &str,
    format: OutputFormat,
    full: bool,
    intent: Option<&str>,
) {
    match format {
        OutputFormat::Json => {
            let output: Vec<serde_json::Value> = results
                .iter()
                .map(|r| {
                    let snippet =
                        extract_snippet_at(r.body, query, 300, r.chunk_pos, r.chunk_len, intent);
                    let mut obj = serde_json::Map::new();
                    obj.insert("docid".into(), format!("#{}", r.docid).into());
                    obj.insert(
                        "score".into(),
                        serde_json::json!((r.score * 100.0).round() / 100.0),
                    );
                    obj.insert("file".into(), r.display_path.into());
                    obj.insert("line".into(), serde_json::json!(snippet.line));
                    obj.insert("title".into(), r.title.into());
                    if let Some(ctx) = r.context {
                        obj.insert("context".into(), ctx.into());
                    }
                    if full {
                        obj.insert("body".into(), r.body.into());
                    } else {
                        obj.insert("snippet".into(), snippet.text.into());
                    }
                    serde_json::Value::Object(obj)
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&output).expect("json serializes")
            );
        }
        OutputFormat::Files => {
            for r in results {
                let ctx = r
                    .context
                    .map(|c| format!(",\"{}\"", c.replace('"', "\"\"")))
                    .unwrap_or_default();
                println!(
                    "#{},{:.2},{}{}",
                    r.docid,
                    r.score,
                    csv_field(r.display_path),
                    ctx
                );
            }
        }
        OutputFormat::Cli => {
            let c = colors();
            for (i, r) in results.iter().enumerate() {
                let snippet =
                    extract_snippet_at(r.body, query, 500, r.chunk_pos, r.chunk_len, intent);
                let snippet_body = snippet
                    .text
                    .split('\n')
                    .skip(1)
                    .collect::<Vec<_>>()
                    .join("\n")
                    .to_lowercase();
                let has_match = query
                    .to_lowercase()
                    .split_whitespace()
                    .any(|t| snippet_body.contains(t));
                let line_info = if has_match {
                    format!(":{}", snippet.line)
                } else {
                    String::new()
                };
                println!(
                    "{}qmd://{}{}{}{} {}#{}{}",
                    c.cyan, r.display_path, c.dim, line_info, c.reset, c.dim, r.docid, c.reset
                );
                if !r.title.is_empty() {
                    println!("{}Title: {}{}", c.bold, r.title, c.reset);
                }
                if let Some(ctx) = r.context {
                    println!("{}Context: {}{}", c.dim, ctx, c.reset);
                }
                println!("Score: {}{:.2}{}", c.bold, r.score, c.reset);
                println!();
                let content = if full { r.body } else { snippet.text.as_str() };
                println!("{content}");
                if i + 1 < results.len() {
                    println!("\n");
                }
            }
        }
    }
}

/// Format a byte count like the TS CLI (`B`, `K`, `M`, `G`).
pub fn format_bytes(bytes: i64) -> String {
    const UNITS: [&str; 4] = ["B", "K", "M", "G"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes}{}", UNITS[0])
    } else {
        format!("{value:.1}{}", UNITS[unit])
    }
}

/// Format an ISO-8601 timestamp as a relative age (`5m ago`), like the TS
/// `formatTimeAgo`. Returns `None` when the timestamp does not parse.
pub fn format_time_ago(iso: &str) -> Option<String> {
    let then = chrono::DateTime::parse_from_rfc3339(iso).ok()?;
    let seconds = (chrono::Utc::now() - then.with_timezone(&chrono::Utc))
        .num_seconds()
        .max(0);
    Some(if seconds < 60 {
        format!("{seconds}s ago")
    } else if seconds < 3600 {
        format!("{}m ago", seconds / 60)
    } else if seconds < 86_400 {
        format!("{}h ago", seconds / 3600)
    } else {
        format!("{}d ago", seconds / 86_400)
    })
}

/// A CSV field, quoted only when it holds a comma, quote or line break.
fn csv_field(s: &str) -> std::borrow::Cow<'_, str> {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\"")).into()
    } else {
        s.into()
    }
}

/// Re-export used by commands for get output.
pub fn numbered(text: &str, start_line: usize) -> String {
    add_line_numbers(text, start_line)
}
