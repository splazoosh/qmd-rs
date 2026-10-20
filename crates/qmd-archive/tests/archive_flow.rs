//! Archiving one document, and re-running the same command as a check.
//!
//! Everything here fetches through a stub: no test touches the network. What is
//! under test is the discipline — that a re-run compares instead of
//! overwriting, that a refusal is recorded and not retried, and that an item
//! carries every field it is supposed to carry.

use std::path::Path;

use chrono::{DateTime, Utc};
use qmd_archive::error::{Error, Refusal};
use qmd_archive::extract::Rules;
use qmd_archive::fetch::{Response, StubFetcher};
use qmd_archive::store::{Archive, ArchiveOptions, Outcome};
use qmd_archive::terms::Redistribution;

const URL: &str = "https://example.org/book/chapter.html";
const HOST: &str = "example.org";

fn at(seconds: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(1_780_000_000 + seconds, 0).unwrap()
}

/// A page whose only moving parts are a rotating obfuscation token and a
/// render-time footer — the two shapes observed on real hosts.
fn page(token: &str, render_secs: &str) -> Vec<u8> {
    format!(
        "<!DOCTYPE html><html><head><title>Chapter</title>\
         <script nonce=\"{token}\">var t={render_secs};</script></head>\
         <body><main><h1>Chapter</h1><p>The gate array is a state machine.</p>\
         <a class=\"__cf_email__\" data-cfemail=\"{token}\" \
         href=\"/cdn-cgi/l/email-protection#{token}\">[email&#160;protected]</a></main>\
         <div id=\"footer\">Page generated in {render_secs} seconds.</div></body></html>"
    )
    .into_bytes()
}

/// Pre-read robots for a host, so a test exercises one request rather than two.
/// The spacing between those two is tested on its own below.
fn seed_robots(root: &Path, host: &str, text: &str, now: DateTime<Utc>) {
    let dir = root.join("hosts").join(host);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("robots.txt"), text).unwrap();
    std::fs::write(dir.join("robots.fetched"), now.to_rfc3339()).unwrap();
}

fn ok_response(status: u16, body: Vec<u8>, headers: &[(&str, &str)]) -> Response {
    Response {
        final_url: URL.to_owned(),
        status,
        headers: headers
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect(),
        body,
    }
}

struct Fixture {
    _dir: tempfile::TempDir,
    archive: Archive,
    fetcher: StubFetcher,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::open(dir.path()).with_contact("https://example.invalid/archive");
    seed_robots(
        dir.path(),
        HOST,
        "User-agent: *\nDisallow: /private/\n",
        at(0),
    );
    Fixture {
        _dir: dir,
        archive,
        fetcher: StubFetcher::new(),
    }
}

#[test]
fn an_empty_contact_sends_and_records_the_user_agent_without_one() {
    let f = fixture();
    let archive = f.archive.clone().with_contact("");
    f.fetcher
        .push_ok(URL, "text/html; charset=utf-8", &page("aaaa", "0.0134"));

    archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap();

    let bare = format!("qmd-archive/{}", env!("CARGO_PKG_VERSION"));
    let requests = f.fetcher.requests.lock().unwrap();
    assert!(requests.iter().all(|(_, agent)| *agent == bare), "{requests:?}");
    assert_eq!(archive.get(URL).unwrap().user_agent, bare);
}

#[test]
fn a_first_archive_stores_the_raw_bytes_the_extract_and_a_manifest() {
    let f = fixture();
    f.fetcher
        .push_ok(URL, "text/html; charset=utf-8", &page("aaaa", "0.0134"));

    let report = f
        .archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap();

    assert_eq!(
        report.outcome,
        Outcome::Stored {
            revision: "v1".into()
        }
    );
    assert!(report.dir.join("v1/raw.bin").is_file());
    assert!(report.dir.join("v1/extract.md").is_file());
    assert!(report.dir.join("manifest.yml").is_file());

    let manifest = f.archive.get(URL).unwrap();
    let revision = manifest.current_revision().unwrap();
    assert_eq!(revision.status, 200);
    assert_eq!(revision.raw.sha256, report.raw_sha256);
    assert!(
        !revision.raw.expected_to_reproduce,
        "the raw hash must be marked as not expected to reproduce"
    );
    assert_eq!(revision.extract.sha256, report.extract_sha256);
    assert_eq!(revision.extract.extractor.to_string(), "html-text/2");
    assert_eq!(revision.extract.charset.as_deref(), Some("utf-8"));
    assert_eq!(
        manifest.user_agent,
        format!(
            "qmd-archive/{} (+https://example.invalid/archive)",
            env!("CARGO_PKG_VERSION")
        )
    );
    assert!(manifest.terms.robots.readable);

    assert!(
        manifest
            .not_recorded
            .iter()
            .any(|field| field.field.contains("browser") && !field.reason.is_empty()),
        "browser fetching is unimplemented and must say so on the item"
    );
}

#[test]
fn the_extract_file_carries_the_extract_and_nothing_else() {
    let f = fixture();
    f.fetcher
        .push_ok(URL, "text/html; charset=utf-8", &page("aaaa", "0.0134"));
    let report = f
        .archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap();

    let extract = std::fs::read_to_string(report.dir.join("v1/extract.md")).unwrap();
    assert!(extract.contains("# Chapter"));
    assert!(
        !extract.contains("1780000010") && !extract.to_lowercase().contains("fetched"),
        "provenance in the extract would move the hash on every fetch:\n{extract}"
    );
    assert!(!extract.contains("cfemail"), "{extract}");
}

#[test]
fn a_re_run_is_a_check_and_a_moving_raw_hash_does_not_move_the_extract() {
    let f = fixture();
    let rules = Rules {
        drop: vec!["#footer".to_owned()],
    };
    f.fetcher
        .push_ok(URL, "text/html; charset=utf-8", &page("aaaa", "0.0134"));
    let first = f
        .archive
        .archive(
            &f.fetcher,
            URL,
            &ArchiveOptions {
                update: false,
                rules: Some(rules),
            },
            at(10),
        )
        .unwrap();

    f.fetcher
        .push_ok(URL, "text/html; charset=utf-8", &page("bbbb", "0.0207"));
    let second = f
        .archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(100))
        .unwrap();

    assert_ne!(
        first.raw_sha256, second.raw_sha256,
        "the fetches must differ"
    );
    assert_eq!(first.extract_sha256, second.extract_sha256);
    assert_eq!(
        second.outcome,
        Outcome::Unchanged {
            revision: "v1".into()
        }
    );
    assert!(!second.dir.join("v2").exists(), "a check wrote a revision");

    let history = f.archive.history(URL).unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].outcome, "stored");
    assert_eq!(history[1].outcome, "unchanged");
    assert_ne!(history[0].raw_sha256, history[1].raw_sha256);
    assert_eq!(history[0].extract_sha256, history[1].extract_sha256);
}

#[test]
fn a_changed_document_is_reported_and_never_silently_overwritten() {
    let f = fixture();
    f.fetcher
        .push_ok(URL, "text/html; charset=utf-8", &page("aaaa", "0.0134"));
    let first = f
        .archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap();

    let rewritten = String::from_utf8(page("aaaa", "0.0134"))
        .unwrap()
        .replace("a state machine", "a state machine clocked at 7 MHz")
        .into_bytes();

    f.fetcher
        .push_ok(URL, "text/html; charset=utf-8", &rewritten);
    let check = f
        .archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(100))
        .unwrap();
    assert_eq!(
        check.outcome,
        Outcome::Changed {
            previous: first.extract_sha256.clone(),
            stored_revision: None,
        }
    );
    assert!(!check.dir.join("v2").exists());
    assert_eq!(f.archive.get(URL).unwrap().current, "v1");

    f.fetcher
        .push_ok(URL, "text/html; charset=utf-8", &rewritten);
    let accepted = f
        .archive
        .archive(
            &f.fetcher,
            URL,
            &ArchiveOptions {
                update: true,
                rules: None,
            },
            at(200),
        )
        .unwrap();
    assert_eq!(
        accepted.outcome,
        Outcome::Changed {
            previous: first.extract_sha256.clone(),
            stored_revision: Some("v2".to_owned()),
        }
    );

    assert!(accepted.dir.join("v1/extract.md").is_file());
    assert!(accepted.dir.join("v2/extract.md").is_file());
    let manifest = f.archive.get(URL).unwrap();
    assert_eq!(manifest.current, "v2");
    assert_eq!(manifest.revisions.len(), 2);
    assert_eq!(
        manifest.revision("v1").unwrap().extract.sha256,
        first.extract_sha256
    );
}

#[test]
fn the_recorded_extractor_regenerates_the_extract_from_the_stored_bytes() {
    let f = fixture();
    f.fetcher
        .push_ok(URL, "text/html; charset=utf-8", &page("aaaa", "0.0134"));
    f.archive
        .archive(
            &f.fetcher,
            URL,
            &ArchiveOptions {
                update: false,
                rules: Some(Rules {
                    drop: vec!["#footer".to_owned()],
                }),
            },
            at(10),
        )
        .unwrap();

    let reports = f.archive.verify(URL).unwrap();
    assert_eq!(reports.len(), 1);
    assert!(reports[0].reproduces(), "{reports:?}");
    assert_eq!(reports[0].extractor, "html-text/2+rules:52b141f6");
}

#[test]
fn robots_disallow_stops_the_fetch_and_is_recorded() {
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::open(dir.path());
    seed_robots(dir.path(), HOST, "User-agent: *\nDisallow: /book/\n", at(0));
    let fetcher = StubFetcher::new();
    fetcher.push_ok(URL, "text/html", b"<html>never read</html>");

    let error = archive
        .archive(&fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap_err();
    assert!(
        matches!(error, Error::Refused(Refusal::Robots { .. })),
        "{error}"
    );
    assert_eq!(fetcher.request_count(), 0, "a disallowed path was fetched");

    let history = archive.history(URL).unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].outcome, "refused");
    assert!(history[0].detail.contains("robots.txt disallows"));
    assert!(archive.get(URL).is_err(), "nothing should have been stored");
}

#[test]
fn a_429_is_recorded_held_against_the_host_and_never_retried() {
    let f = fixture();
    f.fetcher.push(
        URL,
        Ok(ok_response(
            429,
            b"slow down".to_vec(),
            &[("retry-after", "600")],
        )),
    );

    let error = f
        .archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::Refused(Refusal::RateLimited { status: 429, .. })
        ),
        "{error}"
    );
    assert_eq!(f.fetcher.request_count(), 1, "a 429 must not be retried");

    let second = f
        .archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(120))
        .unwrap_err();
    assert!(
        matches!(second, Error::Refused(Refusal::RetryAfter { .. })),
        "{second}"
    );
    assert_eq!(
        f.fetcher.request_count(),
        1,
        "the host was read while held off"
    );
}

#[test]
fn a_challenge_page_stops_the_run_rather_than_being_worked_around() {
    let f = fixture();
    f.fetcher.push(
        URL,
        Ok(ok_response(
            403,
            b"<html><title>Just a moment...</title></html>".to_vec(),
            &[],
        )),
    );

    let error = f
        .archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap_err();
    assert!(
        matches!(error, Error::Refused(Refusal::HumanGate { .. })),
        "{error}"
    );
    assert_eq!(f.fetcher.request_count(), 1);
    assert_eq!(f.archive.history(URL).unwrap()[0].outcome, "refused");
}

#[test]
fn terms_are_recorded_per_item_and_absence_is_stored_as_unstated() {
    let f = fixture();
    f.fetcher
        .push_ok(URL, "text/html; charset=utf-8", &page("aaaa", "0.0134"));
    let report = f
        .archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap();

    assert_eq!(report.redistribution, Redistribution::Unstated);
    let manifest = f.archive.get(URL).unwrap();
    assert_eq!(manifest.terms.redistribution, Redistribution::Unstated);
    assert!(
        !manifest.terms.note.is_empty(),
        "a verdict must say how it was reached"
    );
    assert!(manifest.terms.robots.readable);
    assert_eq!(
        manifest.terms.robots.matched_group.as_deref(),
        Some("*"),
        "the robots verdict belongs to the item, not just to the run"
    );
}

#[test]
fn every_archived_item_is_listed_with_its_manifest() {
    let f = fixture();
    f.fetcher
        .push_ok(URL, "text/html; charset=utf-8", &page("aaaa", "0.0134"));
    f.archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap();

    let items = f.archive.items().unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].1.url, URL);
    assert!(f.archive.verify_all().unwrap()[0].reproduces());
}

#[test]
fn a_revision_from_a_retired_extractor_version_is_unverifiable_not_mismatched() {
    let f = fixture();
    f.fetcher
        .push_ok(URL, "text/html; charset=utf-8", &page("aaaa", "0.0134"));
    f.archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap();

    let (dir, _) = f.archive.items().unwrap().remove(0);
    let path = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| std::fs::read_to_string(p).is_ok_and(|t| t.contains("html-text")))
        .unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, text.replace("version: 2", "version: 1")).unwrap();

    let report = f.archive.verify_all().unwrap().remove(0);
    assert_eq!(report.extractor, "html-text/1");
    assert_eq!(report.recomputed, None);
    assert!(!report.reproduces());
}

fn redirect_to(location: &str) -> Response {
    ok_response(301, Vec::new(), &[("location", location)])
}

#[test]
fn a_redirect_to_a_path_robots_disallows_is_refused_before_it_is_fetched() {
    let f = fixture();
    seed_robots(
        f.archive.root(),
        "other.example",
        "User-agent: *\nDisallow: /\n",
        at(0),
    );
    f.fetcher
        .push(URL, Ok(redirect_to("https://other.example/copy.html")));
    f.fetcher.push_ok(
        "https://other.example/copy.html",
        "text/html",
        b"<html>never read</html>",
    );

    let error = f
        .archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap_err();
    assert!(
        matches!(error, Error::Refused(Refusal::Robots { ref path, .. }) if path.contains("other.example")),
        "{error}"
    );
    assert_eq!(
        f.fetcher.request_count(),
        1,
        "the disallowed hop was fetched"
    );
}

#[test]
fn a_redirect_is_followed_one_checked_hop_at_a_time() {
    let f = fixture();
    seed_robots(f.archive.root(), "other.example", "User-agent: *\n", at(0));
    let target = "https://other.example/chapter.html";
    f.fetcher.push(URL, Ok(redirect_to(target)));
    f.fetcher
        .push_ok(target, "text/html; charset=utf-8", &page("aaaa", "0.0134"));

    let report = f
        .archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap();
    assert!(matches!(report.outcome, Outcome::Stored { .. }));
    let requested: Vec<_> = f
        .fetcher
        .requests
        .lock()
        .unwrap()
        .iter()
        .map(|(url, _)| url.clone())
        .collect();
    assert_eq!(requested, vec![URL.to_owned(), target.to_owned()]);
    let manifest = f.archive.get(URL).unwrap();
    assert_eq!(manifest.current_revision().unwrap().final_url, target);
}

#[test]
fn a_robots_txt_that_errors_is_a_complete_disallow_and_is_asked_again() {
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::open(dir.path());
    let fetcher = StubFetcher::new();
    let robots = "https://example.org/robots.txt";
    fetcher.push(robots, Ok(ok_response(503, b"down".to_vec(), &[])));
    fetcher.push_ok(URL, "text/html", b"<html>read on the second run</html>");

    let error = archive
        .archive(&fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap_err();
    assert!(
        matches!(error, Error::Refused(Refusal::Robots { .. })),
        "{error}"
    );
    assert_eq!(fetcher.request_count(), 1, "only robots.txt was read");

    fetcher.push_ok(robots, "text/plain", b"User-agent: *\n");
    archive
        .archive(&fetcher, URL, &ArchiveOptions::default(), at(100))
        .unwrap();
    assert_eq!(fetcher.requests.lock().unwrap()[1].0, robots);
}

#[test]
fn a_429_on_robots_txt_holds_the_host_off() {
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::open(dir.path());
    let fetcher = StubFetcher::new();
    fetcher.push(
        "https://example.org/robots.txt",
        Ok(ok_response(429, Vec::new(), &[("retry-after", "600")])),
    );

    let error = archive
        .archive(&fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::Refused(Refusal::RateLimited { status: 429, .. })
        ),
        "{error}"
    );
    let second = archive
        .archive(&fetcher, URL, &ArchiveOptions::default(), at(120))
        .unwrap_err();
    assert!(
        matches!(second, Error::Refused(Refusal::RetryAfter { .. })),
        "{second}"
    );
    assert_eq!(fetcher.request_count(), 1);
}

#[test]
fn each_revision_verifies_against_the_rules_it_was_extracted_with() {
    let f = fixture();
    f.fetcher
        .push_ok(URL, "text/html; charset=utf-8", &page("aaaa", "0.0134"));
    f.archive
        .archive(&f.fetcher, URL, &ArchiveOptions::default(), at(10))
        .unwrap();

    f.fetcher
        .push_ok(URL, "text/html; charset=utf-8", &page("aaaa", "0.0134"));
    let second = f
        .archive
        .archive(
            &f.fetcher,
            URL,
            &ArchiveOptions {
                update: true,
                rules: Some(Rules {
                    drop: vec!["#footer".to_owned()],
                }),
            },
            at(100),
        )
        .unwrap();
    assert!(matches!(
        second.outcome,
        Outcome::Changed {
            stored_revision: Some(_),
            ..
        }
    ));

    let reports = f.archive.verify(URL).unwrap();
    assert_eq!(reports.len(), 2);
    assert!(reports.iter().all(|r| r.reproduces()), "{reports:?}");
}

/// Costs five seconds of wall clock on purpose: it is the only way to show
/// that two reads of one host are actually spaced, rather than merely intended
/// to be. The gate is on disk, so this holds across processes too.
#[test]
fn two_reads_of_one_host_are_spaced_by_the_access_rule() {
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::open(dir.path());
    let fetcher = StubFetcher::new();
    fetcher.push_ok(
        "https://example.org/robots.txt",
        "text/plain",
        b"User-agent: *\nDisallow: /private/\n",
    );
    fetcher.push_ok(URL, "text/html; charset=utf-8", &page("aaaa", "0.0134"));

    let started = std::time::Instant::now();
    let report = archive
        .archive(&fetcher, URL, &ArchiveOptions::default(), Utc::now())
        .unwrap();
    let elapsed = started.elapsed();

    assert!(matches!(report.outcome, Outcome::Stored { .. }));
    assert_eq!(fetcher.request_count(), 2, "robots.txt, then the document");
    assert_eq!(
        fetcher.requests.lock().unwrap()[0].0,
        "https://example.org/robots.txt"
    );
    assert!(
        elapsed >= std::time::Duration::from_secs(5),
        "two reads of one host were {elapsed:?} apart"
    );
    assert!(
        fetcher.requests.lock().unwrap()[0]
            .1
            .starts_with("qmd-archive/"),
        "every request identifies itself"
    );
}
