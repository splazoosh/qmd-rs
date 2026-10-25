//! qmd CLI: clap dispatch for every subcommand.

mod archive;
mod commands;
mod daemon;
mod format;

use std::sync::atomic::{AtomicI32, Ordering};

use anyhow::Result;
use clap::{Parser, Subcommand};

use format::OutputFormat;

#[derive(Parser)]
#[command(
    name = "qmd",
    version,
    about = "On-device hybrid search for markdown knowledge bases"
)]
struct Cli {
    /// Index name (config file {index}.yml, database {index}.sqlite).
    /// Without this flag, a project-local .qmd/index.yml found by walking up
    /// from the current directory takes precedence over the global index.
    #[arg(long, global = true)]
    index: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create a project-local index (.qmd/) in the current directory
    Init,
    /// Show index, document, and collection status
    Status,
    /// Manage collections
    Collection {
        #[command(subcommand)]
        command: CollectionCommand,
    },
    /// Mirror one document with its provenance, or check one already held.
    /// Re-running against an archived URL compares rather than overwrites.
    Archive {
        #[command(subcommand)]
        command: Option<ArchiveCommand>,
        /// URL to archive, or to re-check if it is already archived
        url: Option<String>,
        /// Accept a changed document as a new revision beside the old one
        #[arg(long)]
        update: bool,
        /// CSS selector for an element to drop before extraction, repeatable.
        /// Recorded on the item, so the extract stays reproducible
        #[arg(long = "drop")]
        drop: Vec<String>,
    },
    /// Re-index all collections
    Update,
    /// Report new, changed, and gone files without touching the index
    Drift,
    /// Hybrid search: BM25 + vector search fused with RRF. Also accepts
    /// multi-line query documents (lex:/vec:/hyde: lines, one intent: line)
    Query {
        /// The search query (or query document)
        query: String,
        /// Domain intent hint for disambiguation
        #[arg(long)]
        intent: Option<String>,
        /// Rerank the fused candidates with the cross-encoder (overrides the
        /// config's search.rerank)
        #[arg(long, overrides_with = "no_rerank")]
        rerank: bool,
        /// Do not rerank, whatever the config says
        #[arg(long)]
        no_rerank: bool,
        #[command(flatten)]
        output: SearchOptions,
    },
    /// Full-text (BM25) search
    Search {
        /// The search query
        query: String,
        #[command(flatten)]
        output: SearchOptions,
    },
    /// Vector similarity search
    Vsearch {
        /// The search query
        query: String,
        #[command(flatten)]
        output: SearchOptions,
    },
    /// Generate embeddings for indexed documents
    Embed {
        /// Re-embed everything (clears existing embeddings first)
        #[arg(short = 'f', long)]
        force: bool,
        /// Restrict to one collection
        #[arg(short = 'c', long = "collection")]
        collection: Option<String>,
    },
    /// Download the models needed by the active configuration
    Pull {
        /// Re-download even when the cached copy is up to date
        #[arg(long)]
        refresh: bool,
    },
    /// Print a document by path or docid
    Get {
        /// Path (qmd://collection/path, collection-relative, or #docid),
        /// optionally with :line or :line:count suffix
        file: String,
        /// Start line (1-indexed)
        #[arg(long)]
        from: Option<usize>,
        /// Max lines to print
        #[arg(short = 'l', long = "lines")]
        lines: Option<usize>,
        /// Disable line numbers (on by default)
        #[arg(long)]
        no_line_numbers: bool,
    },
    /// List collections or files in a collection
    Ls {
        /// Collection name or collection/path prefix
        path: Option<String>,
    },
    /// Serve this index over MCP on stdin/stdout, for an agent to search.
    /// Exits when the client closes stdin. With --http, serve MCP and REST
    /// on a loopback port instead, for every client to share
    Mcp {
        #[command(subcommand)]
        command: Option<McpCommand>,
        /// Serve over HTTP (/mcp, POST /query, GET /health) on 127.0.0.1
        #[arg(long)]
        http: bool,
        /// Port for --http
        #[arg(long, default_value_t = qmd_mcp::http::DEFAULT_PORT, requires = "http")]
        port: u16,
        /// Run the --http server in the background, recorded in a PID file
        #[arg(long, requires = "http")]
        daemon: bool,
    },
}

#[derive(Subcommand)]
enum McpCommand {
    /// Stop the background server for this index
    Stop,
}

#[derive(Subcommand)]
enum ArchiveCommand {
    /// List everything held
    List,
    /// Show one item: provenance, terms, and every fetch of it
    Show {
        /// The archived URL
        url: String,
    },
    /// Re-run the recorded extractor over the stored bytes and compare hashes.
    /// Offline: this reads the store, not the source
    Verify {
        /// One URL, or all items when omitted
        url: Option<String>,
    },
    /// Register the archive as a collection so its extracts are searchable
    Index,
}

#[derive(Subcommand)]
enum CollectionCommand {
    /// Add a collection and index it
    Add {
        /// Directory to index
        path: String,
        /// Collection name (default: directory basename)
        #[arg(long)]
        name: Option<String>,
        /// Glob pattern
        #[arg(long, default_value = "**/*.md")]
        mask: String,
        /// Glob pattern to exclude, repeatable (stored in the config, so it
        /// also applies to every later `qmd update`)
        #[arg(long = "ignore")]
        ignore: Vec<String>,
        /// Chunk size in tokens for this collection (default 900; 64 to 1800)
        #[arg(long)]
        chunk_tokens: Option<usize>,
        /// Where chunks are cut: window (default) or heading, which starts a
        /// chunk at a heading with no overlap
        #[arg(long, value_name = "window|heading")]
        chunk_boundary: Option<qmd::chunk::ChunkBoundary>,
    },
    /// List all collections
    List,
    /// Remove a collection and its documents
    Remove {
        /// Collection name
        name: String,
    },
}

#[derive(clap::Args)]
struct SearchOptions {
    /// Number of results
    #[arg(short = 'n')]
    limit: Option<usize>,
    /// Minimum score threshold (default 0; 0.3 for vsearch)
    #[arg(long = "min-score")]
    min_score: Option<f64>,
    /// Return all results
    #[arg(long)]
    all: bool,
    /// Show full document content instead of snippets
    #[arg(long)]
    full: bool,
    /// Output format
    #[arg(long, value_enum, default_value_t = OutputFormat::Cli)]
    format: OutputFormat,
    /// Filter by collection (repeatable)
    #[arg(short = 'c', long = "collection")]
    collections: Vec<String>,
    /// How BM25 terms combine: any (default) or all (search and query;
    /// overrides the config's search.match)
    #[arg(long = "match", value_name = "all|any")]
    term_match: Option<qmd::store::TermMatch>,
}

fn main() {
    quiet_on_closed_stdout();
    qmd::store::enable_production_mode();
    let cli = Cli::parse();
    if !cli.command.is_read_only() {
        CLOSED_STDOUT_EXIT.store(Exit::CLOSED_STDOUT_CUT_SHORT, Ordering::Relaxed);
    }
    let location = commands::IndexLocation::resolve(cli.index.as_deref());
    let result: Result<()> = match cli.command {
        Command::Init => commands::init(),
        Command::Status => commands::status(&location),
        Command::Collection { command } => match command {
            CollectionCommand::Add {
                path,
                name,
                mask,
                ignore,
                chunk_tokens,
                chunk_boundary,
            } => commands::collection_add(
                &location,
                &path,
                name.as_deref(),
                &mask,
                &ignore,
                chunk_tokens,
                chunk_boundary,
            ),
            CollectionCommand::List => commands::collection_list(&location),
            CollectionCommand::Remove { name } => commands::collection_remove(&location, &name),
        },
        Command::Archive {
            command,
            url,
            update,
            drop,
        } => match (command, url) {
            (Some(ArchiveCommand::List), _) => archive::list(),
            (Some(ArchiveCommand::Show { url }), _) => archive::show(&url),
            (Some(ArchiveCommand::Verify { url }), _) => archive::verify(url.as_deref()),
            (Some(ArchiveCommand::Index), _) => archive::index(&location),
            (None, Some(url)) => archive::add(&url, update, &drop),
            (None, None) => Err(anyhow::anyhow!(
                "give a URL to archive, or one of: list, show, verify, index"
            )),
        },
        Command::Update => commands::update(&location),
        Command::Drift => commands::drift(&location),
        Command::Query {
            query,
            intent,
            rerank,
            no_rerank,
            output,
        } => {
            let rerank = match (rerank, no_rerank) {
                (true, _) => Some(true),
                (_, true) => Some(false),
                _ => None,
            };
            commands::query(&location, &query, intent.as_deref(), rerank, &output)
        }
        Command::Search { query, output } => commands::search(&location, &query, &output),
        Command::Vsearch { query, output } => commands::vsearch(&location, &query, &output),
        Command::Embed { force, collection } => {
            commands::embed(&location, force, collection.as_deref())
        }
        Command::Pull { refresh } => commands::pull(&location, refresh),
        Command::Get {
            file,
            from,
            lines,
            no_line_numbers,
        } => commands::get(&location, &file, from, lines, !no_line_numbers),
        Command::Ls { path } => commands::ls(&location, path.as_deref()),
        Command::Mcp {
            command: Some(McpCommand::Stop),
            ..
        } => daemon::stop(&location),
        Command::Mcp {
            http: true,
            daemon: true,
            port,
            ..
        } => daemon::start(&location, port),
        Command::Mcp { http, port, .. } => commands::mcp(&location, http.then_some(port)),
    };
    if let Err(e) = result {
        eprintln!("Error: {e:#}");
        let code = e.downcast_ref::<Exit>().map_or(1, |exit| exit.code);
        std::process::exit(code);
    }
}

/// The status `quiet_on_closed_stdout` exits with: 0 until a command that
/// changes state is parsed.
static CLOSED_STDOUT_EXIT: AtomicI32 = AtomicI32::new(0);

/// Ends the process with no panic message when `println!` panics because the
/// reader of stdout has gone (`qmd status | head`): status 0 for a read-only
/// command, `Exit::CLOSED_STDOUT_CUT_SHORT` for one whose work stopped partway.
/// Any other panic goes to the default hook.
fn quiet_on_closed_stdout() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let message = info.payload().downcast_ref::<String>();
        if message.is_some_and(|m| is_closed_stdout(m)) {
            std::process::exit(CLOSED_STDOUT_EXIT.load(Ordering::Relaxed));
        }
        default(info);
    }));
}

impl Command {
    /// Whether the command leaves the index, config, archive, model cache and
    /// daemon as it found them. The rerank cache a query fills does not count.
    fn is_read_only(&self) -> bool {
        match self {
            Command::Status
            | Command::Drift
            | Command::Query { .. }
            | Command::Search { .. }
            | Command::Vsearch { .. }
            | Command::Get { .. }
            | Command::Ls { .. } => true,
            Command::Collection { command } => matches!(command, CollectionCommand::List),
            Command::Archive { command, .. } => matches!(
                command,
                Some(
                    ArchiveCommand::List
                        | ArchiveCommand::Show { .. }
                        | ArchiveCommand::Verify { .. }
                )
            ),
            Command::Mcp {
                command, daemon, ..
            } => command.is_none() && !daemon,
            Command::Init | Command::Update | Command::Embed { .. } | Command::Pull { .. } => false,
        }
    }
}

/// Whether a panic message is std's "failed printing to stdout" for a broken
/// pipe. The OS text is localised on Windows; the trailing os error code is not.
fn is_closed_stdout(message: &str) -> bool {
    let Some(error) = message.strip_prefix("failed printing to stdout: ") else {
        return false;
    };
    error
        .rsplit_once("(os error ")
        .and_then(|(_, code)| code.strip_suffix(')')?.parse().ok())
        .is_some_and(|code| {
            std::io::Error::from_raw_os_error(code).kind() == std::io::ErrorKind::BrokenPipe
        })
}

/// An error that ends the process with its own exit code rather than 1, for
/// an outcome a script must tell apart from failure.
#[derive(Debug)]
pub struct Exit {
    pub code: i32,
    pub message: String,
}

impl Exit {
    /// `qmd archive <url>` found the extract changed and, without `--update`,
    /// stored nothing.
    pub const CHANGED_NOT_STORED: i32 = 3;
    /// A command that changes state lost the reader of its stdout and stopped
    /// partway: 128 + SIGPIPE, what a shell reports for a Unix tool killed by it.
    pub const CLOSED_STDOUT_CUT_SHORT: i32 = 141;
}

impl std::fmt::Display for Exit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Exit {}
