//! `qmd archive`: mirror one document with its provenance, or check one already
//! held.
//!
//! The same invocation does both. Against a URL not yet archived it stores;
//! against one already archived it fetches, extracts and compares, and says
//! whether the extract still hashes the same. Nothing is overwritten without
//! `--update`, and a re-check is one document, asked for — there is no bulk
//! re-check here on purpose.

use anyhow::{Context as _, Result};
use qmd_archive::extract::Rules;
use qmd_archive::fetch::HttpFetcher;
use qmd_archive::manifest::Manifest;
use qmd_archive::store::{Archive, ArchiveOptions, Outcome};
use qmd_archive::terms::Redistribution;

use crate::commands::IndexLocation;
use crate::format::colors;

/// The collection the archive is indexed as. Only the extracts are indexed —
/// the raw responses and manifests are storage, not documents.
pub const COLLECTION: &str = "archive";
pub const MASK: &str = "**/extract.md";

fn open() -> Archive {
    Archive::open(Archive::default_root())
}

fn short(hash: &str) -> &str {
    &hash[..hash.len().min(12)]
}

/// Archive `url`, or check it if it is already held.
pub fn add(url: &str, update: bool, drop: &[String]) -> Result<()> {
    let archive = open();
    let options = ArchiveOptions {
        update,
        rules: (!drop.is_empty()).then(|| Rules {
            drop: drop.to_vec(),
        }),
    };

    let c = colors();
    println!("{}", url);
    println!("  identified as {}", archive.user_agent());
    if archive.contact().is_none() {
        println!(
            "{}  no contact sent — set QMD_ARCHIVE_CONTACT to a URL that reaches you{}",
            c.yellow, c.reset
        );
    }
    let report = archive
        .archive(&HttpFetcher, url, &options, chrono::Utc::now())
        .with_context(|| format!("archiving {url}"))?;

    println!(
        "  raw      {:>9} bytes  {}  (not expected to reproduce)",
        report.raw_bytes,
        short(&report.raw_sha256)
    );
    println!(
        "  extract  {:>9} bytes  {}",
        report.extract_bytes,
        short(&report.extract_sha256)
    );
    println!(
        "  terms    {}  — {}",
        match report.redistribution {
            Redistribution::Permitted => "redistribution permitted",
            Redistribution::Forbidden => "redistribution forbidden",
            Redistribution::Unstated => "redistribution unstated",
        },
        report.terms_note
    );

    match &report.outcome {
        Outcome::Stored { revision } => println!(
            "{}✓ stored as {revision} in {}{}",
            c.green,
            report.dir.display(),
            c.reset
        ),
        Outcome::Unchanged { revision } => println!(
            "{}✓ unchanged — the extract still hashes the same as {revision}{}",
            c.green, c.reset
        ),
        Outcome::Changed {
            previous,
            stored_revision: Some(revision),
        } => println!(
            "{}! changed — extract moved from {}; stored as {revision}{}",
            c.yellow,
            short(previous),
            c.reset
        ),
        Outcome::Changed {
            stored_revision: None,
            ..
        } => {}
    }

    for warning in &report.warnings {
        println!("{}  warning: {warning}{}", c.yellow, c.reset);
    }
    if let Outcome::Changed {
        previous,
        stored_revision: None,
    } = &report.outcome
    {
        return Err(crate::Exit {
            code: crate::Exit::CHANGED_NOT_STORED,
            message: format!(
                "changed — extract moved from {} to {}; nothing written. \
                 Re-run with --update to keep the new revision.",
                short(previous),
                short(&report.extract_sha256),
            ),
        }
        .into());
    }
    Ok(())
}

/// Everything held, one line each.
pub fn list() -> Result<()> {
    let archive = open();
    let items = archive.items()?;
    if items.is_empty() {
        println!("Nothing archived. Archive a document with 'qmd archive <url>'.");
        println!("Store: {}", archive.root().display());
        return Ok(());
    }

    println!("{} item(s) in {}\n", items.len(), archive.root().display());
    for (_, manifest) in &items {
        let revision = manifest.current_revision();
        println!("{}", manifest.url);
        println!(
            "  {} revision(s), current {}  extract {}  terms {:?}",
            manifest.revisions.len(),
            manifest.current,
            revision.map_or("-", |r| short(&r.extract.sha256)),
            manifest.terms.redistribution
        );
    }
    Ok(())
}

/// One item in full: provenance, terms, and every fetch of it.
pub fn show(url: &str) -> Result<()> {
    let archive = open();
    let manifest = archive.get(url)?;
    print_manifest(&manifest);

    let history = archive.history(url).unwrap_or_default();
    if !history.is_empty() {
        println!("\nFetches");
        println!(
            "  {:<20}  {:<10}  {:<14}  {:<14}",
            "when", "outcome", "raw", "extract"
        );
        for record in &history {
            println!(
                "  {:<20}  {:<10}  {:<14}  {:<14}  {}",
                record.at.format("%Y-%m-%d %H:%M:%SZ"),
                record.outcome,
                record.raw_sha256.as_deref().map_or("-", short),
                record.extract_sha256.as_deref().map_or("-", short),
                record.detail
            );
        }
    }
    Ok(())
}

fn print_manifest(manifest: &Manifest) {
    println!("{}", manifest.url);
    println!("  host             {}", manifest.host);
    println!("  first archived   {}", manifest.first_archived);
    println!("  last checked     {}", manifest.last_checked);
    println!("  fetched by       {:?}", manifest.fetch_method);
    println!("  identified as    {}", manifest.user_agent);
    println!("  current revision {}", manifest.current);

    if let Some(revision) = manifest.current_revision() {
        println!("  extractor        {}", revision.extract.extractor);
        println!(
            "  raw              {} bytes  sha256 {}  (not expected to reproduce)",
            revision.raw.bytes, revision.raw.sha256
        );
        println!(
            "  extract          {} bytes  sha256 {}",
            revision.extract.bytes, revision.extract.sha256
        );
        if let Some(charset) = &revision.extract.charset {
            println!("  read as          {charset}");
        }
        if !revision.headers.is_empty() {
            println!("  headers");
            for (name, value) in &revision.headers {
                println!("    {name}: {value}");
            }
        }
    }
    if !manifest.rules.drop.is_empty() {
        println!("  extraction rules drop {}", manifest.rules.drop.join(", "));
    }

    println!("\nTerms");
    println!("  redistribution   {:?}", manifest.terms.redistribution);
    println!("  verdict          {}", manifest.terms.note);
    println!(
        "  robots.txt       {}  {}",
        if manifest.terms.robots.readable {
            "readable"
        } else {
            "not readable"
        },
        manifest.terms.robots.note
    );
    if let Some(rule) = &manifest.terms.robots.matched_rule {
        println!("  robots rule      {rule}");
    }
    for signal in &manifest.terms.signals {
        println!(
            "  signal           [{:?}/{:?}] {} — {}",
            signal.scope, signal.stance, signal.locus, signal.statement
        );
    }
    if !manifest.not_recorded.is_empty() {
        println!("\nNot recorded");
        for absent in &manifest.not_recorded {
            println!("  {}: {}", absent.field, absent.reason);
        }
    }
}

/// Re-run each recorded extractor over the stored raw bytes. Offline.
pub fn verify(url: Option<&str>) -> Result<()> {
    let archive = open();
    let reports = match url {
        Some(url) => archive.verify(url)?,
        None => archive.verify_all()?,
    };
    if reports.is_empty() {
        println!("Nothing archived.");
        return Ok(());
    }

    let c = colors();
    let mut failures = 0;
    for report in &reports {
        if report.reproduces() {
            println!(
                "{}✓{} {} {}  {}  {}",
                c.green,
                c.reset,
                report.url,
                report.revision,
                short(&report.recorded),
                report.extractor
            );
        } else if let Some(recomputed) = &report.recomputed {
            failures += 1;
            println!(
                "{}✗{} {} {}  recorded {}  recomputed {}  {}",
                c.red,
                c.reset,
                report.url,
                report.revision,
                short(&report.recorded),
                short(recomputed),
                report.extractor
            );
        } else {
            failures += 1;
            println!(
                "{}?{} {} {}  recorded {} under {}, which this build no longer runs",
                c.yellow,
                c.reset,
                report.url,
                report.revision,
                short(&report.recorded),
                report.extractor
            );
        }
    }
    println!(
        "\n{} of {} extract(s) reproduce from the stored bytes.",
        reports.len() - failures,
        reports.len()
    );
    if failures > 0 {
        anyhow::bail!("{failures} extract(s) did not reproduce from the stored bytes");
    }
    Ok(())
}

/// Register the archive as an ordinary collection, so its extracts are
/// searchable through the query path that already exists. Running it again
/// is harmless: an `archive` collection already over this root is reported,
/// not added twice.
pub fn index(location: &IndexLocation) -> Result<()> {
    let root = Archive::default_root();
    std::fs::create_dir_all(&root)
        .with_context(|| format!("creating the archive root {}", root.display()))?;

    let config_path = match location {
        IndexLocation::Global(name) => qmd::config::config_file_path(name),
        IndexLocation::Local(path) => path.clone(),
    };
    let config = qmd::config::load_config(&config_path)?;
    if let Some(existing) = config.get(COLLECTION) {
        let same_root =
            std::fs::canonicalize(&existing.path).ok() == std::fs::canonicalize(&root).ok();
        if same_root && existing.pattern == MASK {
            println!(
                "The archive is already indexed as collection '{COLLECTION}' over {}. \
                 Run 'qmd update' to re-index it.",
                existing.path
            );
            return Ok(());
        }
        anyhow::bail!(
            "collection '{COLLECTION}' already exists over {} ({}), not the archive root {}",
            existing.path,
            existing.pattern,
            root.display()
        );
    }

    crate::commands::collection_add(
        location,
        &root.to_string_lossy(),
        Some(COLLECTION),
        MASK,
        &[],
        None,
        None,
        None,
    )
}
