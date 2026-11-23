//! `qmd archive rm` against the real binary: the item leaves the store, the
//! listing and an index of the archive, and its neighbour stays in all three.
//! The items are archived through a stub fetcher, so nothing touches the network.

use std::path::Path;
use std::process::{Command, Output};

use chrono::DateTime;
use qmd_archive::fetch::StubFetcher;
use qmd_archive::store::{Archive, ArchiveOptions};

const GONE: &str = "https://example.org/wrong-page.html";
const KEPT: &str = "https://example.org/right-page.html";

fn qmd(dir: &Path, archive: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_qmd"))
        .args(args)
        .current_dir(dir)
        .env("QMD_ARCHIVE_DIR", archive)
        .env("QMD_MOCK_EMBEDDER", "1")
        .env_remove("INDEX_PATH")
        .env("QMD_CONFIG_DIR", dir.join("config-unused"))
        .env("XDG_CACHE_HOME", dir.join("cache-unused"))
        .output()
        .expect("qmd binary runs")
}

fn ok(dir: &Path, archive: &Path, args: &[&str]) -> String {
    let out = qmd(dir, archive, args);
    assert!(
        out.status.success(),
        "qmd {args:?} failed\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn page(word: &str) -> Vec<u8> {
    format!(
        "<html><head><title>{word}</title></head><body><p>{word} survey notes.</p></body></html>"
    )
    .into_bytes()
}

#[test]
fn rm_removes_an_item_from_the_store_the_listing_and_the_index() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let root = dir.join("archive");
    let now = DateTime::from_timestamp(1_780_000_000, 0).unwrap();
    let robots = root.join("hosts").join("example.org");
    std::fs::create_dir_all(&robots).unwrap();
    std::fs::write(robots.join("robots.txt"), "User-agent: *\nAllow: /\n").unwrap();
    std::fs::write(robots.join("robots.fetched"), now.to_rfc3339()).unwrap();
    let archive = Archive::open(&root);
    let fetcher = StubFetcher::new();
    fetcher.push_ok(GONE, "text/html; charset=utf-8", &page("zygomorphic"));
    fetcher.push_ok(KEPT, "text/html; charset=utf-8", &page("quillwort"));
    for (url, secs) in [(GONE, 0), (KEPT, 10)] {
        let at = now + chrono::Duration::seconds(secs);
        archive
            .archive(&fetcher, url, &ArchiveOptions::default(), at)
            .unwrap();
    }

    ok(dir, &root, &["init"]);
    ok(dir, &root, &["archive", "index"]);
    ok(dir, &root, &["update"]);
    assert!(ok(dir, &root, &["search", "zygomorphic"]).contains("extract.md"));

    let out = ok(dir, &root, &["archive", "rm", GONE]);
    assert!(out.contains("1 extract(s) removed"), "{out}");

    let listed = ok(dir, &root, &["archive", "list"]);
    assert!(!listed.contains(GONE) && listed.contains(KEPT), "{listed}");
    assert!(
        !ok(dir, &root, &["search", "zygomorphic"]).contains("extract.md"),
        "the removed extract no longer comes back"
    );
    assert!(ok(dir, &root, &["search", "quillwort"]).contains("extract.md"));
    assert!(robots.join("robots.txt").is_file() && root.join("hosts.json").is_file());

    let again = qmd(dir, &root, &["archive", "rm", GONE]);
    assert!(!again.status.success());
    assert!(
        String::from_utf8_lossy(&again.stderr).contains("nothing archived"),
        "{}",
        String::from_utf8_lossy(&again.stderr)
    );
}
