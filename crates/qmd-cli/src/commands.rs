//! Command implementations for the CLI.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, anyhow, bail};

use qmd::config::{self, CollectionConfig};
use qmd::index::reindex_collection;
use qmd::store::{DocumentLookup, Store, default_db_path};

use crate::SearchOptions;
use crate::format::{self, colors, format_bytes, format_time_ago};
use qmd::querylog::{self, Arm, Via};

/// Where the active index lives: the global config/cache directories, or a
/// project-local `.qmd/` directory discovered by walking up from the cwd.
pub enum IndexLocation {
    /// Global index by name (`{config_dir}/{name}.yml`, `{cache_dir}/{name}.sqlite`).
    Global(String),
    /// Project-local index: path of the `.qmd/index.yml` config file.
    Local(PathBuf),
}

impl IndexLocation {
    /// Resolve the active index. An explicit `--index` always selects the
    /// global index of that name; otherwise a project-local `.qmd/` config
    /// takes precedence over the global default.
    pub fn resolve(explicit_index: Option<&str>) -> Self {
        if let Some(name) = explicit_index {
            return Self::Global(name.to_owned());
        }
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        match config::find_local_config_path(&cwd) {
            Some(config_path) => Self::Local(config_path),
            None => Self::Global("index".to_owned()),
        }
    }

    fn config_path(&self) -> PathBuf {
        match self {
            Self::Global(name) => config::config_file_path(name),
            Self::Local(config_path) => config_path.clone(),
        }
    }

    fn db_path(&self) -> Result<PathBuf> {
        if let Some(path) = std::env::var_os("INDEX_PATH") {
            return Ok(PathBuf::from(path));
        }
        match self {
            Self::Global(name) => Ok(default_db_path(name)?),
            Self::Local(config_path) => Ok(config::local_db_path(config_path)),
        }
    }
}

/// Config file + open store, with the YAML config synced into the DB mirror.
struct App {
    config_path: PathBuf,
    db_path: PathBuf,
    config: CollectionConfig,
    store: Store,
}

fn open(location: &IndexLocation) -> Result<App> {
    let app = open_without_sync(location)?;
    app.store.sync_config_to_db(&app.config)?;
    Ok(app)
}

/// Open without mirroring the YAML config into the database, for commands
/// that must not write.
fn open_without_sync(location: &IndexLocation) -> Result<App> {
    let config_path = location.config_path();
    let config = config::load_config(&config_path)?;
    let db_path = location.db_path()?;
    let store = Store::open(&db_path)
        .with_context(|| format!("failed to open index at {}", db_path.display()))?;
    Ok(App {
        config_path,
        db_path,
        config,
        store,
    })
}

/// `qmd init`: create a project-local index under `./.qmd/`.
pub fn init() -> Result<()> {
    let cwd = std::env::current_dir().context("failed to resolve current directory")?;
    let home = config::home_dir();
    if same_directory(&cwd, &home) {
        bail!(
            "Refusing to initialize a local index in $HOME. The global index is \
             automatically created; run 'qmd collection add <path>' for the global \
             index, or run 'qmd init' inside a project folder."
        );
    }

    let qmd_dir = cwd.join(".qmd");
    let yaml_path = qmd_dir.join("index.yaml");
    let config_path = if yaml_path.exists() {
        yaml_path
    } else {
        qmd_dir.join("index.yml")
    };
    std::fs::create_dir_all(&qmd_dir)
        .with_context(|| format!("failed to create {}", qmd_dir.display()))?;
    let gitignore = qmd_dir.join(".gitignore");
    if !gitignore.exists() {
        std::fs::write(&gitignore, "*\n")
            .with_context(|| format!("failed to write {}", gitignore.display()))?;
    }

    let mut cfg = config::load_config(&config_path)?;
    if !config_path.exists() {
        cfg.models = Some(qmd::config::ModelsConfig {
            embed: Some(qmd::models::resolve_embed_model(None)),
            ..Default::default()
        });
        config::save_config(&config_path, &cfg)?;
    }

    let db_path = config::local_db_path(&config_path);
    let store = Store::open(&db_path)
        .with_context(|| format!("failed to create index at {}", db_path.display()))?;
    store.sync_config_to_db(&cfg)?;

    println!("ready to go with new local index");
    Ok(())
}

fn same_directory(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// `qmd status`: index location, document/vector counts, collections, models.
pub fn status(location: &IndexLocation) -> Result<()> {
    let app = open(location)?;
    let c = colors();

    let index_size = std::fs::metadata(&app.db_path)
        .map(|m| i64::try_from(m.len()).unwrap_or(i64::MAX))
        .ok();

    let total_docs: i64 =
        app.store
            .conn()
            .query_row("SELECT COUNT(*) FROM documents WHERE active = 1", [], |r| {
                r.get(0)
            })?;
    let vector_count: i64 =
        app.store
            .conn()
            .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))?;
    let model = embed_model(&app.config);
    let pending = app.store.pending_embedding_docs(None, &model)?.len();
    let most_recent: Option<String> = app.store.conn().query_row(
        "SELECT MAX(modified_at) FROM documents WHERE active = 1",
        [],
        |r| r.get(0),
    )?;

    println!("{}QMD Status{}\n", c.bold, c.reset);
    println!("Config: {}", app.config_path.display());
    println!("Index:  {}", app.db_path.display());
    if let Some(size) = index_size {
        println!("Size:   {}", format_bytes(size));
    }
    match embed_on_gpu()? {
        Some(true) => println!("Embed:  GPU"),
        Some(false) => println!("Embed:  {}CPU — {CPU_EMBED_WARNING}{}", c.yellow, c.reset),
        None => {}
    }
    if let Some(r) = crate::daemon::running(&crate::daemon::DaemonFiles::for_index(location)) {
        println!(
            "MCP:    {}running{} (PID {}, http://127.0.0.1:{}/mcp)",
            c.green, c.reset, r.pid, r.port
        );
    }
    println!();

    println!("{}Documents{}", c.bold, c.reset);
    println!("  Total:    {total_docs} files indexed");
    println!("  Vectors:  {vector_count} embedded");
    if pending > 0 {
        println!(
            "  {}Pending:  {pending} need embedding{} (run 'qmd embed')",
            c.yellow, c.reset
        );
    }
    if let Some(latest) = &most_recent
        && let Some(ago) = format_time_ago(latest)
    {
        println!("  Updated:  {ago}");
    }

    let collections = app.store.list_collections_with_stats()?;
    if collections.is_empty() {
        println!(
            "\n{}No collections. Run 'qmd collection add .' to index markdown files.{}",
            c.dim, c.reset
        );
    } else {
        println!("\n{}Collections{}", c.bold, c.reset);
        for coll in &collections {
            let last_mod = coll
                .last_modified
                .as_deref()
                .and_then(format_time_ago)
                .unwrap_or_else(|| "never".to_owned());
            println!(
                "  {}{}{} {}(qmd://{}/){}",
                c.cyan, coll.name, c.reset, c.dim, coll.name, c.reset
            );
            println!("    {}Pattern:{}  {}", c.dim, c.reset, coll.pattern);
            println!(
                "    {}Files:{}    {} (updated {last_mod})",
                c.dim, c.reset, coll.active_count
            );
        }

        println!("\n{}Examples{}", c.bold, c.reset);
        let first = &collections[0].name;
        println!("  {}# List files in a collection{}", c.dim, c.reset);
        println!("  qmd ls {first}");
        println!("  {}# Get a document{}", c.dim, c.reset);
        println!("  qmd get qmd://{first}/path/to/file.md");
        println!("  {}# Search within a collection{}", c.dim, c.reset);
        println!("  qmd search \"query\" -c {first}");
    }

    let hf_link = |uri: &str| {
        qmd::models::parse_hf_uri(uri)
            .map(|r| format!("https://huggingface.co/{}", r.repo))
            .unwrap_or_else(|| uri.to_owned())
    };
    println!("\n{}Models{}", c.bold, c.reset);
    println!("  Embedding:   {}", hf_link(&model));
    let hf = qmd::models::hf::HfSetup::from_env();
    println!("  Download:    {}", hf.route());
    println!("  HF token:    {}", hf.token_status());

    Ok(())
}

pub fn collection_add(
    location: &IndexLocation,
    path: &str,
    name: Option<&str>,
    mask: &str,
    ignore: &[String],
    chunk_tokens: Option<usize>,
    chunk_boundary: Option<qmd::chunk::ChunkBoundary>,
) -> Result<()> {
    let resolved =
        std::fs::canonicalize(path).with_context(|| format!("directory not found: {path}"))?;
    let resolved_str = resolved
        .to_string_lossy()
        .trim_start_matches(r"\\?\")
        .replace('\\', "/");

    let coll_name = match name {
        Some(n) => n.to_owned(),
        None => resolved_str
            .split('/')
            .rfind(|s| !s.is_empty())
            .unwrap_or("root")
            .to_owned(),
    };
    if !config::is_valid_collection_name(&coll_name) {
        bail!("invalid collection name '{coll_name}' (use letters, numbers, hyphens, underscores)");
    }

    let mut app = open(location)?;
    if app.config.get(&coll_name).is_some() {
        bail!("collection '{coll_name}' already exists (use a different --name)");
    }
    if let Some(existing) = app
        .config
        .list()
        .into_iter()
        .find(|c| c.collection.path == resolved_str && c.collection.pattern == mask)
    {
        bail!(
            "a collection already exists for this path and pattern: {} — \
             run 'qmd update' to re-index it",
            existing.name
        );
    }

    app.config
        .add_collection(&coll_name, &resolved_str, mask, ignore);
    if let Some(tokens) = chunk_tokens {
        let clamped = qmd::chunk::ChunkParams::for_size(tokens).size_tokens;
        if clamped != tokens {
            bail!(
                "--chunk-tokens must be {} to {}",
                qmd::chunk::MIN_CHUNK_TOKENS,
                qmd::chunk::MAX_CHUNK_TOKENS
            );
        }
        if let Some(c) = app.config.collections.get_mut(&coll_name) {
            c.chunk_tokens = Some(tokens);
        }
    }
    if let Some(c) = app.config.collections.get_mut(&coll_name) {
        c.chunk_boundary = chunk_boundary;
    }
    config::save_config(&app.config_path, &app.config)?;
    app.store.sync_config_to_db(&app.config)?;

    println!("Creating collection '{coll_name}'...");
    let result = reindex_collection(
        &app.store,
        std::path::Path::new(&resolved_str),
        mask,
        &coll_name,
        ignore,
        None,
    )?;
    println!(
        "Indexed: {} new, {} updated, {} unchanged, {} removed",
        result.indexed, result.updated, result.unchanged, result.removed
    );
    println!("✓ Collection '{coll_name}' created successfully");
    Ok(())
}

pub fn collection_list(location: &IndexLocation) -> Result<()> {
    let app = open(location)?;
    let collections = app.store.list_collections_with_stats()?;
    if collections.is_empty() {
        println!("No collections found. Run 'qmd collection add .' to create one.");
        return Ok(());
    }
    let c = colors();
    println!(
        "{}Collections ({}):{}\n",
        c.bold,
        collections.len(),
        c.reset
    );
    for coll in collections {
        let exclude_tag = if coll.include_by_default {
            ""
        } else {
            " [excluded]"
        };
        println!(
            "{}{}{} {}(qmd://{}/){}{}",
            c.cyan, coll.name, c.reset, c.dim, coll.name, c.reset, exclude_tag
        );
        println!("  {}Pattern:{}  {}", c.dim, c.reset, coll.pattern);
        if let Some(ignore) = &coll.ignore
            && !ignore.is_empty()
        {
            println!("  {}Ignore:{}   {}", c.dim, c.reset, ignore.join(", "));
        }
        println!("  {}Path:{}     {}", c.dim, c.reset, coll.path);
        println!("  {}Files:{}    {}", c.dim, c.reset, coll.active_count);
        if let Some(modified) = &coll.last_modified {
            println!("  {}Updated:{}  {}", c.dim, c.reset, modified);
        }
        println!();
    }
    Ok(())
}

pub fn collection_remove(location: &IndexLocation, name: &str) -> Result<()> {
    let mut app = open(location)?;
    if app.config.get(name).is_none() {
        bail!("collection not found: {name} (run 'qmd collection list')");
    }
    let (deleted, cleaned) = app.store.remove_collection(name)?;
    app.config.remove_collection(name);
    config::save_config(&app.config_path, &app.config)?;
    app.store.sync_config_to_db(&app.config)?;
    println!("✓ Removed collection '{name}'");
    println!("  Deleted {deleted} documents");
    if cleaned > 0 {
        println!("  Cleaned up {cleaned} orphaned content hashes");
    }
    Ok(())
}

fn require_collection(app: &App, name: &str) -> Result<()> {
    if app.config.get(name).is_none() {
        bail!("collection not found: {name} (run 'qmd collection list')");
    }
    Ok(())
}

pub fn collection_rename(location: &IndexLocation, old: &str, new: &str) -> Result<()> {
    let mut app = open(location)?;
    require_collection(&app, old)?;
    if !config::is_valid_collection_name(new) {
        bail!("invalid collection name '{new}' (use letters, numbers, hyphens, underscores)");
    }
    if app.config.get(new).is_some() {
        bail!("collection '{new}' already exists; choose another name or remove it first");
    }
    let moved = app.store.rename_collection(old, new)?;
    app.config.rename_collection(old, new);
    if let Err(e) = config::save_config(&app.config_path, &app.config) {
        app.store
            .rename_collection(new, old)
            .context("config save failed and the database rename could not be undone")?;
        return Err(e.into());
    }
    app.store.sync_config_to_db(&app.config)?;
    println!("✓ Renamed collection '{old}' to '{new}' ({moved} documents)");
    println!("  Virtual paths: qmd://{old}/ → qmd://{new}/");
    Ok(())
}

pub fn collection_show(location: &IndexLocation, name: &str) -> Result<()> {
    let app = open(location)?;
    require_collection(&app, name)?;
    let coll = &app.config.collections[name];
    let stats = app
        .store
        .list_collections_with_stats()?
        .into_iter()
        .find(|s| s.name == name);
    let params = coll.chunk_params();
    println!("Collection: {name}");
    println!("  Path:     {}", coll.path);
    println!("  Pattern:  {}", coll.pattern);
    if let Some(ignore) = coll.ignore.as_ref().filter(|i| !i.is_empty()) {
        println!("  Ignore:   {}", ignore.join(", "));
    }
    println!(
        "  Include:  {}",
        if coll.is_included_by_default() {
            "yes (default)"
        } else {
            "no"
        }
    );
    if let Some(update) = &coll.update {
        println!("  Update:   {update}");
    }
    if let Some(context) = &coll.context {
        println!("  Contexts: {}", context.len());
    }
    println!(
        "  Chunks:   {} tokens, {} boundary",
        params.size_tokens,
        coll.chunk_boundary.unwrap_or_default().as_str()
    );
    if let Some(stats) = stats {
        println!("  Files:    {}", stats.active_count);
        if let Some(modified) = &stats.last_modified {
            println!("  Updated:  {modified}");
        }
    }
    Ok(())
}

pub fn collection_update_cmd(location: &IndexLocation, name: &str, command: &str) -> Result<()> {
    let mut app = open(location)?;
    require_collection(&app, name)?;
    let command = command.trim();
    let coll = app.config.collections.get_mut(name).expect("checked above");
    coll.update = (!command.is_empty()).then(|| command.to_owned());
    config::save_config(&app.config_path, &app.config)?;
    app.store.sync_config_to_db(&app.config)?;
    if command.is_empty() {
        println!("✓ Cleared update command for '{name}'");
    } else {
        println!("✓ Set update command for '{name}': {command}");
    }
    Ok(())
}

pub fn collection_include(location: &IndexLocation, name: &str, include: bool) -> Result<()> {
    let mut app = open(location)?;
    require_collection(&app, name)?;
    app.config
        .collections
        .get_mut(name)
        .expect("checked above")
        .include_by_default = (!include).then_some(false);
    config::save_config(&app.config_path, &app.config)?;
    app.store.sync_config_to_db(&app.config)?;
    println!(
        "✓ Collection '{name}' {} default queries",
        if include {
            "included in"
        } else {
            "excluded from"
        }
    );
    Ok(())
}

pub fn cleanup(location: &IndexLocation, dry_run: bool) -> Result<()> {
    let app = if dry_run {
        open_without_sync(location)?
    } else {
        open(location)?
    };
    let (stats, verb) = if dry_run {
        println!("Dry run — no changes made.\n");
        (app.store.preview_cleanup()?, "Would remove")
    } else {
        (app.store.cleanup()?, "Removed")
    };
    println!("{verb} {} cached rerank scores", stats.cached_scores);
    println!(
        "{verb} {} orphaned embedding chunks",
        stats.orphaned_vectors
    );
    println!("{verb} {} inactive document records", stats.inactive_docs);
    println!("{verb} {} orphaned content hashes", stats.orphaned_content);
    if dry_run {
        println!("Would compact FTS and vacuum the database");
    } else {
        println!("✓ FTS compacted, database vacuumed");
    }
    Ok(())
}

/// Run a collection's `update` command in its root before it is indexed, as
/// TS qmd does; a failure stops `qmd update` with the command's status.
/// Commands from a project-local config run only with
/// `QMD_TRUST_LOCAL_CONFIG=1`, since such a config can arrive with a checkout.
fn run_update_command(location: &IndexLocation, root: &Path, command: &str) -> Result<()> {
    let c = colors();
    if matches!(location, IndexLocation::Local(_))
        && std::env::var("QMD_TRUST_LOCAL_CONFIG").as_deref() != Ok("1")
    {
        println!(
            "{}Skipped update command (project-local config; set QMD_TRUST_LOCAL_CONFIG=1 to run it): {command}{}",
            c.yellow, c.reset
        );
        return Ok(());
    }
    println!("{}    Running update command: {command}{}", c.dim, c.reset);
    let output = shell_command(command)
        .current_dir(root)
        .stdin(std::process::Stdio::null())
        .output()
        .with_context(|| format!("failed to start update command: {command}"))?;
    for stream in [&output.stdout, &output.stderr] {
        for line in String::from_utf8_lossy(stream).trim().lines() {
            println!("    {line}");
        }
    }
    if !output.status.success() {
        let code = output.status.code().unwrap_or(1);
        return Err(crate::Exit {
            code,
            message: format!("update command failed with exit code {code}: {command}"),
        }
        .into());
    }
    Ok(())
}

/// `cmd /C` reads the rest of its command line verbatim and does not undo
/// `arg`'s MSVC quoting, so the command goes in raw.
#[cfg(windows)]
fn shell_command(command: &str) -> std::process::Command {
    use std::os::windows::process::CommandExt;
    let mut shell = std::process::Command::new("cmd");
    shell.arg("/C").raw_arg(command);
    shell
}

#[cfg(not(windows))]
fn shell_command(command: &str) -> std::process::Command {
    let mut shell = std::process::Command::new("sh");
    shell.arg("-c").arg(command);
    shell
}

pub fn update(location: &IndexLocation) -> Result<()> {
    let app = open(location)?;
    let collections = app.config.list();
    if collections.is_empty() {
        println!("No collections found. Run 'qmd collection add .' to index markdown files.");
        return Ok(());
    }
    let c = colors();
    println!(
        "{}Updating {} collection(s)...{}\n",
        c.bold,
        collections.len(),
        c.reset
    );
    let mut skipped = Vec::new();
    for (i, named) in collections.iter().enumerate() {
        println!(
            "{}[{}/{}]{} {}{}{} {}({}){}",
            c.cyan,
            i + 1,
            collections.len(),
            c.reset,
            c.bold,
            named.name,
            c.reset,
            c.dim,
            named.collection.pattern,
            c.reset
        );
        let root = std::path::Path::new(&named.collection.path);
        if !root.is_dir() {
            println!(
                "{}Skipped: {} not found{}\n",
                c.yellow, named.collection.path, c.reset
            );
            skipped.push(named.name.as_str());
            continue;
        }
        if let Some(command) = &named.collection.update {
            run_update_command(location, root, command)?;
        }
        let ignore = named.collection.ignore.clone().unwrap_or_default();
        let result = reindex_collection(
            &app.store,
            root,
            &named.collection.pattern,
            &named.name,
            &ignore,
            None,
        )
        .with_context(|| format!("collection {} ({})", named.name, named.collection.path))?;
        println!(
            "Indexed: {} new, {} updated, {} unchanged, {} removed",
            result.indexed, result.updated, result.unchanged, result.removed
        );
        if result.orphaned_cleaned > 0 {
            println!(
                "Cleaned up {} orphaned content hash(es)",
                result.orphaned_cleaned
            );
        }
        println!();
    }
    if skipped.is_empty() {
        println!("✓ All collections updated.");
    } else {
        println!(
            "{}✓ Updated, but skipped {} with a missing path: {}. Fix the path or run 'qmd collection remove <name>'.{}",
            c.yellow,
            skipped.len(),
            skipped.join(", "),
            c.reset
        );
    }
    Ok(())
}

/// `qmd drift`: how far the index has fallen behind the disk, without
/// touching it. Walks and hashes every file the collections cover; loads no
/// model and writes no document, content, or vector row.
pub fn drift(location: &IndexLocation) -> Result<()> {
    use qmd::index::drift::drift_collection;

    let app = open_without_sync(location)?;
    let collections = app.config.list();
    if collections.is_empty() {
        println!("No collections found. Run 'qmd collection add .' to index markdown files.");
        return Ok(());
    }
    let c = colors();
    println!(
        "{}Drift{} {}(index vs. disk){}\n",
        c.bold, c.reset, c.dim, c.reset
    );

    let mut total = qmd::index::drift::DriftResult::default();
    for named in &collections {
        let ignore = named.collection.ignore.clone().unwrap_or_default();
        let d = drift_collection(
            &app.store,
            std::path::Path::new(&named.collection.path),
            &named.collection.pattern,
            &named.name,
            &ignore,
        )
        .with_context(|| format!("collection {}", named.name))?;
        total.new += d.new;
        total.changed += d.changed;
        total.gone += d.gone;
        total.unchanged += d.unchanged;

        if d.is_clean() {
            println!(
                "  {}{:<24}{} {}clean{} ({})",
                c.cyan,
                named.name,
                c.reset,
                c.dim,
                c.reset,
                files(d.unchanged)
            );
        } else {
            println!(
                "  {}{:<24}{} {}{} new, {} changed, {} gone{} ({} unchanged)",
                c.cyan,
                named.name,
                c.reset,
                c.yellow,
                d.new,
                d.changed,
                d.gone,
                c.reset,
                d.unchanged
            );
        }
    }

    if total.is_clean() {
        println!(
            "\n✓ Index is current with {} on disk.",
            files(total.unchanged)
        );
    } else {
        println!(
            "\n{}{} drifted{} — {} new, {} changed, {} gone. \
             Run 'qmd update', then 'qmd embed'.",
            c.yellow,
            files(total.drifted()),
            c.reset,
            total.new,
            total.changed,
            total.gone
        );
    }
    Ok(())
}

/// Pluralize a file count for human-facing output.
fn files(n: usize) -> String {
    if n == 1 {
        "1 file".to_owned()
    } else {
        format!("{n} files")
    }
}

/// Validate `-c` filters against the config; default to include-by-default
/// collections. Returns the selected names.
fn resolve_collection_filter(app: &App, requested: &[String]) -> Result<Vec<String>> {
    Ok(app.config.query_collections(requested)?)
}

/// Drop results outside the selected collections (multi-collection filters
/// run post-hoc; single-collection filters run at the SQL level).
fn retain_collections<T>(results: &mut Vec<T>, names: &[String], filepath: impl Fn(&T) -> &str) {
    if names.len() > 1 {
        let prefixes: Vec<String> = names.iter().map(|n| format!("qmd://{n}/")).collect();
        results.retain(|r| prefixes.iter().any(|p| filepath(r).starts_with(p)));
    }
}

pub fn search(location: &IndexLocation, query: &str, opts: &SearchOptions) -> Result<()> {
    let app = open(location)?;
    let collection_names = resolve_collection_filter(&app, &opts.collections)?;
    let single_collection = (collection_names.len() == 1).then(|| collection_names[0].clone());

    let limit = opts.limit.unwrap_or_else(|| opts.format.default_limit());
    let limit = if opts.all { 100_000 } else { limit };
    let fetch_limit = if opts.all { 100_000 } else { limit.max(50) * 2 };

    let term_match = opts
        .term_match
        .or(app.config.search_settings().term_match)
        .unwrap_or_default();
    let mut results =
        app.store
            .search_fts_with(query, fetch_limit, single_collection.as_deref(), term_match)?;
    retain_collections(&mut results, &collection_names, |r| &r.filepath);

    let min_score = opts.min_score.unwrap_or(0.0);
    results.retain(|r| r.score >= min_score);
    results.truncate(limit);

    log_query(&app, Arm::Bm25, query, &results, |r| &r.filepath);
    if results.is_empty() {
        format::print_empty_results(opts.format);
        return Ok(());
    }
    format::print_search_results(&results, query, opts.format, opts.full);
    Ok(())
}

/// Record one answered query in the index's query log.
fn log_query<T>(app: &App, arm: Arm, query: &str, results: &[T], filepath: impl Fn(&T) -> &str) {
    let paths: Vec<String> = results.iter().map(|r| filepath(r).to_owned()).collect();
    querylog::append(&app.db_path, arm, Via::Cli, query, &paths);
}

/// Resolve the active embed model URI: config > env > default. The mock
/// seam overrides this so embed, status, and search agree on the
/// embedding fingerprint.
fn embed_model(config: &CollectionConfig) -> String {
    if mock_embedder_requested() {
        use qmd::embed::Embedder as _;
        return qmd::embed::MockEmbedder::default().model_uri().to_owned();
    }
    qmd::models::resolve_embed_model(config.models.as_ref())
}

/// Whether `QMD_MOCK_EMBEDDER` selects the deterministic mock backend
/// (test seam, mirroring the TS CI mode that disables LLM operations).
fn mock_embedder_requested() -> bool {
    std::env::var("QMD_MOCK_EMBEDDER").is_ok_and(|v| {
        let v = v.trim().to_ascii_lowercase();
        !v.is_empty() && !matches!(v.as_str(), "0" | "false" | "off")
    })
}

#[cfg(feature = "llama")]
fn load_embedder(model: &str) -> Result<Box<dyn qmd::embed::EmbeddingBackend>> {
    if mock_embedder_requested() {
        return Ok(Box::new(qmd::embed::MockEmbedder::default()));
    }
    let embedder = qmd::embed::LlamaEmbedder::load(model).with_context(|| {
        format!(
            "failed to load embedding model {model} — \
             run 'qmd pull' to (re)download it"
        )
    })?;
    Ok(Box::new(embedder))
}

/// Load the reranker for a model URI. `QMD_MOCK_EMBEDDER` selects the mock
/// reranker too, so the test seam needs no model.
fn load_reranker(model: &str) -> Result<Box<dyn qmd::rerank::Reranker>> {
    if mock_embedder_requested() {
        return Ok(Box::new(qmd::rerank::MockReranker));
    }
    load_llama_reranker(model)
}

#[cfg(feature = "llama")]
fn load_llama_reranker(model: &str) -> Result<Box<dyn qmd::rerank::Reranker>> {
    let reranker = qmd::rerank::LlamaReranker::load(model).with_context(|| {
        format!("failed to load rerank model {model} — run 'qmd pull' to (re)download it")
    })?;
    Ok(Box::new(reranker))
}

#[cfg(not(feature = "llama"))]
fn load_llama_reranker(_model: &str) -> Result<Box<dyn qmd::rerank::Reranker>> {
    bail!("this build has no model backend: rebuild qmd-cli with its default features to rerank")
}

/// Where `load_embedder` will run: `Some(true)` GPU, `Some(false)` CPU, `None`
/// for the mock embedder or a build with no embedding backend.
#[cfg(feature = "llama")]
fn embed_on_gpu() -> Result<Option<bool>> {
    if mock_embedder_requested() {
        return Ok(None);
    }
    Ok(Some(qmd::embed::gpu_offload()?))
}

#[cfg(not(feature = "llama"))]
fn embed_on_gpu() -> Result<Option<bool>> {
    Ok(None)
}

/// A CPU embed of the whole tree takes hours where the GPU takes minutes, and
/// nothing else about a CPU-only binary looks different.
const CPU_EMBED_WARNING: &str = "this qmd has no GPU offload and embeds on the CPU, \
     roughly 15x slower (a build without the default 'cuda' feature, or no \
     usable CUDA device); rebuild with plain 'cargo build --release'";

#[cfg(not(feature = "llama"))]
fn load_embedder(_model: &str) -> Result<Box<dyn qmd::embed::EmbeddingBackend>> {
    if mock_embedder_requested() {
        return Ok(Box::new(qmd::embed::MockEmbedder::default()));
    }
    bail!(
        "this build has no embedding backend: rebuild qmd-cli with its default \
         features to use embed/vsearch"
    )
}

pub fn vsearch(location: &IndexLocation, query: &str, opts: &SearchOptions) -> Result<()> {
    use qmd::embed::format_query_for_embedding;

    let app = open(location)?;
    let collection_names = resolve_collection_filter(&app, &opts.collections)?;
    let single_collection = (collection_names.len() == 1).then(|| collection_names[0].clone());

    let embedded: i64 = if app.store.vec_table_exists()? {
        app.store
            .conn()
            .query_row("SELECT COUNT(*) FROM content_vectors", [], |r| r.get(0))?
    } else {
        0
    };
    if embedded == 0 {
        bail!("no embeddings in this index yet — run 'qmd embed' first");
    }

    let model = embed_model(&app.config);
    let embedder = load_embedder(&model)?;
    let formatted = format_query_for_embedding(query, &model);
    let embedding = embedder
        .embed(&formatted)?
        .ok_or_else(|| anyhow!("failed to embed query"))?;

    let limit = opts.limit.unwrap_or_else(|| opts.format.default_limit());
    let limit = if opts.all { 100_000 } else { limit };
    let fetch_limit = if opts.all { 100_000 } else { limit.max(20) };

    let mut results =
        app.store
            .search_vec(&embedding, fetch_limit, single_collection.as_deref())?;
    retain_collections(&mut results, &collection_names, |r| &r.filepath);

    let min_score = opts.min_score.unwrap_or(0.3);
    results.retain(|r| r.score >= min_score);
    results.truncate(limit);

    log_query(&app, Arm::Vector, query, &results, |r| &r.filepath);
    if results.is_empty() {
        format::print_empty_results(opts.format);
        return Ok(());
    }
    format::print_search_results(&results, query, opts.format, opts.full);
    Ok(())
}

pub fn query(
    location: &IndexLocation,
    query: &str,
    intent: Option<&str>,
    rerank: Option<bool>,
    explain: bool,
    opts: &SearchOptions,
) -> Result<()> {
    use qmd::query::{
        HybridOptions, QueryType, hybrid_query, parse_structured_query, structured_search,
    };

    let app = open(location)?;
    let collection_names = resolve_collection_filter(&app, &opts.collections)?;

    let parsed = parse_structured_query(query).map_err(|e| anyhow!("{e}"))?;
    let intent = intent
        .map(str::to_owned)
        .or_else(|| parsed.as_ref().and_then(|p| p.intent.clone()));

    let needs_embedder = app.store.vec_table_exists()?
        && parsed.as_ref().is_none_or(|p| {
            p.searches
                .iter()
                .any(|s| matches!(s.query_type, QueryType::Vec | QueryType::Hyde))
        });
    let embedder: Option<Box<dyn qmd::embed::EmbeddingBackend>> =
        if needs_embedder && (cfg!(feature = "llama") || mock_embedder_requested()) {
            let model = embed_model(&app.config);
            Some(load_embedder(&model)?)
        } else {
            None
        };
    let embedder_dyn = embedder.as_deref().map(|e| e as &dyn qmd::embed::Embedder);
    let settings = app.config.search_settings();
    let reranker = if rerank.or(settings.rerank).unwrap_or(false) {
        Some(load_reranker(&qmd::models::resolve_rerank_model(
            app.config.models.as_ref(),
        ))?)
    } else {
        None
    };

    let options = HybridOptions {
        collections: collection_names,
        limit: if opts.all {
            500
        } else {
            opts.limit.unwrap_or_else(|| opts.format.default_limit())
        },
        min_score: opts.min_score.unwrap_or(0.0),
        intent: intent.clone(),
        term_match: opts.term_match.or(settings.term_match).unwrap_or_default(),
        explain,
        ..Default::default()
    };

    let c = colors();
    let display_query;
    let response = if let Some(parsed) = &parsed {
        let type_labels: Vec<&str> = parsed
            .searches
            .iter()
            .map(|s| s.query_type.as_str())
            .collect();
        eprintln!(
            "{}Structured search: {} queries ({}){}",
            c.dim,
            parsed.searches.len(),
            type_labels.join("+"),
            c.reset
        );
        if let Some(intent) = &intent {
            eprintln!("{}├─ intent: {intent}{}", c.dim, c.reset);
        }
        for s in &parsed.searches {
            let mut preview = s.query.clone();
            if preview.chars().count() > 72 {
                preview = format!("{}...", preview.chars().take(69).collect::<String>());
            }
            eprintln!(
                "{}├─ {}: {preview}{}",
                c.dim,
                s.query_type.as_str(),
                c.reset
            );
        }
        eprintln!("{}└─ Searching...{}", c.dim, c.reset);

        display_query = parsed
            .searches
            .iter()
            .find(|s| s.query_type == QueryType::Lex)
            .or_else(|| {
                parsed
                    .searches
                    .iter()
                    .find(|s| s.query_type == QueryType::Vec)
            })
            .map_or_else(|| query.to_owned(), |s| s.query.clone());
        structured_search(
            &app.store,
            embedder_dyn,
            reranker.as_deref(),
            &parsed.searches,
            &options,
        )
        .map_err(|e| anyhow!("{e}"))?
    } else {
        display_query = query.to_owned();
        hybrid_query(
            &app.store,
            embedder_dyn,
            reranker.as_deref(),
            query,
            &options,
        )
        .map_err(|e| anyhow!("{e}"))?
    };

    if let Some(score) = response.report.strong_signal {
        eprintln!(
            "{}Strong BM25 signal ({score:.2}) — expansion bypass (expansion is not \
             implemented yet){}",
            c.dim, c.reset
        );
    }
    if response.report.reranked > 0 {
        eprintln!(
            "{}Reranked {} candidates ({} by the model, the rest cached){}",
            c.dim, response.report.reranked, response.report.rerank_fresh, c.reset
        );
    }
    if response.report.vector_skipped {
        let reason = if cfg!(feature = "llama") {
            "no embeddings in the index — run 'qmd embed'"
        } else {
            "no embedding backend in this build"
        };
        eprintln!("{}Note: vector search skipped ({reason}){}", c.dim, c.reset);
    }

    let results = response.results;
    log_query(&app, Arm::Hybrid, query, &results, |r| &r.filepath);
    if results.is_empty() {
        format::print_empty_results(opts.format);
        return Ok(());
    }
    format::print_query_results(
        &results,
        &display_query,
        opts.format,
        opts.full,
        intent.as_deref(),
    );
    Ok(())
}

pub fn embed(location: &IndexLocation, force: bool, collection: Option<&str>) -> Result<()> {
    use std::io::Write as _;

    use qmd::embed::{EmbedOptions, generate_embeddings};

    let app = open(location)?;
    if let Some(name) = collection
        && app.config.get(name).is_none()
    {
        bail!("collection not found: {name}");
    }

    let model = embed_model(&app.config);
    let c = colors();
    println!("{}Generating embeddings{} ({model})", c.bold, c.reset);
    if embed_on_gpu()? == Some(false) {
        eprintln!("{}warning:{} {CPU_EMBED_WARNING}", c.yellow, c.reset);
    }
    let embedder = load_embedder(&model)?;

    let options = EmbedOptions {
        force,
        collection: collection.map(str::to_owned),
        on_progress: Some(Box::new(|p: &qmd::embed::EmbedProgress| {
            print!(
                "\r  {} / {} chunks embedded ({} errors)",
                p.chunks_embedded, p.total_chunks, p.errors
            );
            let _ = std::io::stdout().flush();
        })),
        ..Default::default()
    };
    let result = generate_embeddings(&app.store, embedder.as_ref(), &options)?;
    if result.docs_processed == 0 {
        println!("✓ All documents already embedded.");
        return Ok(());
    }
    println!();
    println!(
        "✓ Embedded {} chunk(s) across {} document(s) in {:.1}s",
        result.chunks_embedded,
        result.docs_processed,
        result.duration_ms as f64 / 1000.0
    );
    if !result.failures.is_empty() {
        println!("⚠ {} chunk(s) failed:", result.failures.len());
        for f in &result.failures {
            println!("  {} #{} ({}): {}", f.path, f.seq, f.hash, f.reason);
        }
        bail!("embedding completed with {} error(s)", result.errors);
    }
    Ok(())
}

pub fn pull(location: &IndexLocation, refresh: bool) -> Result<()> {
    let config = config::load_config(&location.config_path())?;
    let mut models = vec![embed_model(&config)];
    if config.search_settings().rerank == Some(true) {
        models.push(qmd::models::resolve_rerank_model(config.models.as_ref()));
    }
    let c = colors();
    println!("{}Pulling models{}", c.bold, c.reset);
    let cache_dir = qmd::models::model_cache_dir();
    for model in &models {
        pull_one(model, &cache_dir, refresh)?;
    }
    Ok(())
}

fn pull_one(model: &str, cache_dir: &Path, refresh: bool) -> Result<()> {
    let result = qmd::models::pull_model(model, cache_dir, refresh)?;
    if let Some(failure) = &result.cli_failure {
        eprintln!("{failure}; used the built-in download instead");
    }
    let note = match (&result.route, result.refreshed) {
        (None, _) => "cached/checked".to_owned(),
        (Some(route), true) => format!("refreshed by {route}"),
        (Some(route), false) => format!("fetched by {route}"),
    };
    println!(
        "- {} -> {} ({}, {note})",
        result.model,
        result.path.display(),
        format_bytes(i64::try_from(result.size_bytes).unwrap_or(i64::MAX))
    );
    Ok(())
}

pub fn get(
    location: &IndexLocation,
    file: &str,
    mut from: Option<usize>,
    mut lines: Option<usize>,
    line_numbers: bool,
) -> Result<()> {
    let app = open(location)?;

    let mut input = file.to_owned();
    if let Some((rest, l, count)) = parse_line_suffix(&input) {
        input = rest;
        if from.is_none() {
            from = Some(l);
        }
        if let Some(count) = count
            && lines.is_none()
        {
            lines = Some(count);
        }
    }
    let from = from.map(|f| f.max(1));

    let doc = match app.store.find_document(&input, true)? {
        DocumentLookup::Found(doc) => doc,
        DocumentLookup::NotFound { similar } => {
            let mut msg = format!("document not found: {file}");
            if !similar.is_empty() {
                msg.push_str("\nSimilar files:");
                for s in &similar {
                    msg.push_str(&format!("\n  {s}"));
                }
            }
            return Err(anyhow!(msg));
        }
    };

    let mut output = doc.body.clone().unwrap_or_default();
    let start_line = from.unwrap_or(1);
    if from.is_some() || lines.is_some() {
        let all_lines: Vec<&str> = output.split('\n').collect();
        let start = start_line - 1;
        let end = lines
            .map(|l| start.saturating_add(l).min(all_lines.len()))
            .unwrap_or(all_lines.len());
        output = all_lines.get(start..end).unwrap_or_default().join("\n");
    }
    if line_numbers && !output.is_empty() {
        output = format::numbered(&output, start_line);
    }

    println!("qmd://{}  #{}", doc.display_path, doc.docid);
    if let Some(ctx) = &doc.context {
        println!("Folder Context: {ctx}");
    }
    println!("---\n");
    println!("{output}");
    Ok(())
}

fn parse_line_suffix(input: &str) -> Option<(String, usize, Option<usize>)> {
    let parts: Vec<&str> = input.rsplitn(3, ':').collect();
    if parts.len() == 3
        && let (Ok(count), Ok(line)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>())
        && !parts[2].is_empty()
    {
        return Some((parts[2].to_owned(), line, Some(count)));
    }
    let (rest, last) = input.rsplit_once(':')?;
    let line = last.parse::<usize>().ok()?;
    if rest.is_empty() {
        return None;
    }
    Some((rest.to_owned(), line, None))
}

/// The model loaders the MCP index thread calls on first use.
fn factories() -> qmd_mcp::Factories {
    qmd_mcp::Factories {
        embedder: Box::new(load_embedder),
        reranker: Box::new(load_reranker),
    }
}

/// `qmd mcp`: serve the active index over MCP on stdin/stdout, or with
/// `http_port` over HTTP on that loopback port until interrupted.
///
/// The CLI supplies the embedder factory so the `llama` feature and its native
/// build stay on this side of the seam; `qmd-mcp` never links llama.cpp.
pub fn mcp(location: &IndexLocation, http_port: Option<u16>) -> Result<()> {
    let config_path = location.config_path();
    let db_path = location.db_path()?;
    let config = config::load_config(&config_path)?;
    let model = embed_model(&config);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to start the async runtime for the MCP server")?;
    let Some(port) = http_port else {
        return runtime.block_on(qmd_mcp::serve_stdio(
            &config_path,
            &db_path,
            model,
            factories(),
        ));
    };
    let served = runtime.block_on(qmd_mcp::serve_http(
        &config_path,
        &db_path,
        model,
        factories(),
        std::net::SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, port)),
        |addr| eprintln!("QMD MCP server listening on http://{addr}/mcp"),
        async {
            let _ = tokio::signal::ctrl_c().await;
        },
    ));
    crate::daemon::release(location);
    served
}

pub fn ls(location: &IndexLocation, path: Option<&str>) -> Result<()> {
    let app = open(location)?;
    let c = colors();

    let Some(path_arg) = path else {
        let collections = app.config.list();
        if collections.is_empty() {
            println!("No collections found. Run 'qmd collection add .' to index files.");
            return Ok(());
        }
        println!("{}Collections:{}\n", c.bold, c.reset);
        for named in collections {
            let count = app.store.active_document_count(&named.name)?;
            println!(
                "  {}qmd://{}{}{}/{}  {}({count} files){}",
                c.dim, c.reset, c.cyan, named.name, c.reset, c.dim, c.reset
            );
        }
        return Ok(());
    };

    let trimmed = path_arg.strip_prefix("qmd://").unwrap_or(path_arg);
    let trimmed = trimmed.trim_start_matches('/').trim_end_matches('/');
    let (collection_name, path_prefix) = match trimmed.split_once('/') {
        Some((coll, prefix)) => (coll, Some(prefix)),
        None => (trimmed, None),
    };

    if app.config.get(collection_name).is_none() {
        bail!("collection not found: {collection_name} (run 'qmd ls' to see collections)");
    }

    let files = app.store.ls_entries(collection_name, path_prefix)?;
    if files.is_empty() {
        match path_prefix {
            Some(prefix) => println!("No files found under qmd://{collection_name}/{prefix}"),
            None => println!("No files found in collection: {collection_name}"),
        }
        return Ok(());
    }

    let max_size_width = files
        .iter()
        .map(|f| format_bytes(f.size).len())
        .max()
        .unwrap_or(0);
    for file in files {
        let size = format_bytes(file.size);
        let time = file.modified_at.get(..16).unwrap_or(&file.modified_at);
        println!(
            "{size:>max_size_width$}  {time}  {}qmd://{collection_name}/{}{}{}",
            c.dim, c.reset, file.path, c.reset
        );
    }
    Ok(())
}
