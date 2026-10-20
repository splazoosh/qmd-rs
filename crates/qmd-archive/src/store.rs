//! The store: archiving a document, and re-running the same command as a check.
//!
//! One command does both, and that is the point. `archive <url>` on a URL never
//! seen stores it; on one already held it fetches again and *compares*, and
//! says whether the extract still hashes the same. It never silently
//! overwrites: a changed document needs `--update`, which adds a revision and
//! leaves the old one where a citation can still find it.
//!
//! Nothing here crawls. There is no link following, no bulk re-check, no
//! scheduled pass — a re-check is one document, asked for, exactly like the
//! first fetch was.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};

use crate::error::{Error, Refusal, Result};
use crate::extract::{self, Rules};
use crate::fetch::{self, FetchMethod, Fetcher};
use crate::gate::{Clearance, Gate};
use crate::item::ItemKey;
use crate::manifest::{
    AbsentField, ExtractRecord, FETCH_LOG_FILE, FetchRecord, MANIFEST_FILE, Manifest, RAW_FILE,
    RawRecord, Revision,
};
use crate::robots::{self, RobotsRecord};
use crate::terms;

/// How long a host's `robots.txt` is reused before being read again. Not a
/// background refresh: it is read, at most this often, on the way to a fetch
/// somebody asked for.
const ROBOTS_TTL_SECS: i64 = 24 * 60 * 60;

/// The longest the store will wait out the per-host spacing before refusing
/// instead. Rule 5's delay is deliberate and worth waiting; a wait longer than
/// this means something else is going on and the caller should hear about it.
const MAX_WAIT_SECS: u64 = 60;

/// Held by an archiving run for its whole length; see [`Archive::lock`].
const LOCK_FILE: &str = ".lock";

/// What a call to [`Archive::archive`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The URL was not held before; this is its first revision.
    Stored { revision: String },
    /// It was held, and the extract still hashes the same. The raw response
    /// may well have differed — that is the ordinary case, and the evidence is
    /// in the fetch log.
    Unchanged { revision: String },
    /// The extract hash moved. Nothing was written unless `update` was asked
    /// for, in which case a new revision was added beside the old one.
    Changed {
        previous: String,
        stored_revision: Option<String>,
    },
}

impl Outcome {
    pub fn is_check(&self) -> bool {
        !matches!(self, Outcome::Stored { .. })
    }
}

/// What one archive or check produced.
#[derive(Debug, Clone)]
pub struct Report {
    pub url: String,
    pub dir: PathBuf,
    pub outcome: Outcome,
    pub raw_sha256: String,
    pub raw_bytes: u64,
    pub extract_sha256: String,
    pub extract_bytes: u64,
    pub redistribution: terms::Redistribution,
    pub terms_note: String,
    /// Things worth a reader's attention that are not failures.
    pub warnings: Vec<String>,
}

/// Options for one archive call.
#[derive(Debug, Clone, Default)]
pub struct ArchiveOptions {
    /// Accept a changed document as a new revision. Without it a change is
    /// reported and nothing is written.
    pub update: bool,
    /// Per-item extraction rules to set or replace.
    pub rules: Option<Rules>,
}

/// The result of re-running a recorded extractor over stored bytes.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    pub url: String,
    pub revision: String,
    pub recorded: String,
    /// Absent when the recorded extractor version is not the one this build
    /// runs: the hash then cannot be checked, which is not a mismatch.
    pub recomputed: Option<String>,
    pub extractor: String,
}

impl VerifyReport {
    pub fn reproduces(&self) -> bool {
        self.recomputed.as_ref() == Some(&self.recorded)
    }
}

/// A directory of archived documents.
#[derive(Debug, Clone)]
pub struct Archive {
    root: PathBuf,
    contact: Option<String>,
}

impl Archive {
    /// The contact comes from `QMD_ARCHIVE_CONTACT`; unset or empty, none is sent.
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            contact: std::env::var("QMD_ARCHIVE_CONTACT")
                .ok()
                .filter(|contact| !contact.is_empty()),
        }
    }

    /// Set the URL sent in the user-agent. A URL — never a mailbox, which
    /// identifies a person. An empty one counts as none.
    pub fn with_contact(mut self, contact: impl Into<String>) -> Self {
        self.contact = Some(contact.into()).filter(|contact| !contact.is_empty());
        self
    }

    pub fn contact(&self) -> Option<&str> {
        self.contact.as_deref()
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn user_agent(&self) -> String {
        fetch::user_agent(self.contact())
    }

    /// `QMD_ARCHIVE_DIR`, else `$XDG_DATA_HOME/qmd/archive`, else
    /// `~/.local/share/qmd/archive`.
    ///
    /// Deliberately not under `~/.cache`: a cache is disposable, and this is
    /// the thing whose disappearance the archive exists to prevent.
    pub fn default_root() -> PathBuf {
        if let Some(dir) = std::env::var_os("QMD_ARCHIVE_DIR") {
            return PathBuf::from(dir);
        }
        let base = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::home_dir()
                    .expect("home directory must be resolvable")
                    .join(".local")
                    .join("share")
            });
        base.join("qmd").join("archive")
    }

    fn gate(&self) -> Gate {
        Gate::new(&self.root)
    }

    fn item_dir(&self, key: &ItemKey) -> PathBuf {
        self.root.join(key.dir_name())
    }

    /// Archive `url`, or check it if it is already held.
    pub fn archive(
        &self,
        fetcher: &dyn Fetcher,
        url: &str,
        options: &ArchiveOptions,
        now: DateTime<Utc>,
    ) -> Result<Report> {
        let key = ItemKey::parse(url)?;
        let _lock = self.lock()?;
        let dir = self.item_dir(&key);
        let existing = self.load_manifest(&dir)?;
        let user_agent = self.user_agent();
        let mut warnings = Vec::new();

        let rules = options
            .rules
            .clone()
            .or_else(|| existing.as_ref().map(|m| m.rules.clone()))
            .unwrap_or_default();

        let clock = Clock::start(now);
        let (response, robots) = match self.fetch_following(fetcher, &key, &user_agent, &clock) {
            Ok(found) => found,
            Err(Error::Refused(refusal)) => {
                let status = refusal_status(&refusal);
                self.record_refusal(&dir, &key, &user_agent, now, &refusal, status)?;
                return Err(refusal.into());
            }
            Err(other) => return Err(other),
        };

        if let Some(evidence) = fetch::human_gate(&response) {
            let refusal = Refusal::HumanGate {
                url: key.url.clone(),
                evidence,
            };
            self.record_refusal(
                &dir,
                &key,
                &user_agent,
                now,
                &refusal,
                Some(response.status),
            )?;
            return Err(refusal.into());
        }

        if response.status != 200 {
            let refusal = Refusal::Status {
                url: key.url.clone(),
                status: response.status,
            };
            self.record_refusal(
                &dir,
                &key,
                &user_agent,
                now,
                &refusal,
                Some(response.status),
            )?;
            return Err(refusal.into());
        }

        let raw_sha256 = hex::encode(Sha256::digest(&response.body));
        let extracted = extract::extract(&response.body, response.content_type(), &rules);
        let extract_sha256 = extracted.hash();

        if !extracted.rules_rejected.is_empty() {
            warnings.push(format!(
                "extraction rules that are not valid selectors did not run: {}",
                extracted.rules_rejected.join(", ")
            ));
        }
        if extracted.truncated_depth {
            warnings.push(
                "the document nests deeper than the extractor descends; part of it is not in \
                 the extract"
                    .to_owned(),
            );
        }

        let source = extract::decode_text(&response.body, response.content_type());
        let terms = terms::read(&source, &response.headers, robots);
        if terms.redistribution == terms::Redistribution::Forbidden {
            warnings.push(
                "redistribution is forbidden for this item; it is stored for reading and \
                 citation only"
                    .to_owned(),
            );
        }

        let previous_extract = existing
            .as_ref()
            .and_then(|m| m.current_revision())
            .map(|r| r.extract.sha256.clone());

        let outcome = match &previous_extract {
            None => Outcome::Stored {
                revision: "v1".to_owned(),
            },
            Some(previous) if *previous == extract_sha256 => Outcome::Unchanged {
                revision: existing
                    .as_ref()
                    .map(|m| m.current.clone())
                    .unwrap_or_default(),
            },
            Some(previous) => Outcome::Changed {
                previous: previous.clone(),
                stored_revision: options.update.then(|| {
                    existing
                        .as_ref()
                        .map_or("v1".to_owned(), Manifest::next_revision_id)
                }),
            },
        };

        let write_revision = match &outcome {
            Outcome::Stored { revision } => Some(revision.clone()),
            Outcome::Changed {
                stored_revision, ..
            } => stored_revision.clone(),
            Outcome::Unchanged { .. } => None,
        };

        if let Some(revision_id) = &write_revision {
            let revision_dir = dir.join(revision_id);
            create_dir(&revision_dir)?;
            write_file(&revision_dir.join(RAW_FILE), &response.body)?;
            write_file(&revision_dir.join(extracted.file_name()), &extracted.body)?;

            let revision = Revision {
                id: revision_id.clone(),
                fetched_at: now,
                final_url: response.final_url.clone(),
                status: response.status,
                headers: response.headers.iter().cloned().collect(),
                raw: RawRecord {
                    file: format!("{revision_id}/{RAW_FILE}"),
                    bytes: response.body.len() as u64,
                    sha256: raw_sha256.clone(),
                    expected_to_reproduce: false,
                },
                extract: ExtractRecord {
                    file: format!("{revision_id}/{}", extracted.file_name()),
                    bytes: extracted.body.len() as u64,
                    sha256: extract_sha256.clone(),
                    extractor: extracted.extractor.clone(),
                    charset: extracted.charset.clone(),
                    rules_applied: extracted.rules_applied.clone(),
                    rules_rejected: extracted.rules_rejected.clone(),
                    truncated_depth: extracted.truncated_depth,
                },
                rules: Some(rules.clone()),
            };

            let manifest = match existing {
                Some(mut manifest) => {
                    manifest.last_checked = now;
                    manifest.current = revision_id.clone();
                    manifest.user_agent = user_agent.clone();
                    manifest.rules = rules.clone();
                    manifest.terms = terms.clone();
                    manifest.revisions.push(revision);
                    manifest
                }
                None => Manifest {
                    url: key.url.clone(),
                    host: key.host.clone(),
                    first_archived: now,
                    last_checked: now,
                    current: revision_id.clone(),
                    fetch_method: FetchMethod::Request,
                    user_agent: user_agent.clone(),
                    rules: rules.clone(),
                    terms: terms.clone(),
                    revisions: vec![revision],
                    not_recorded: absent_fields(),
                },
            };
            self.write_manifest(&dir, &manifest)?;
        } else if let Some(mut manifest) = existing {
            manifest.last_checked = now;
            manifest.user_agent = user_agent.clone();
            manifest.terms = terms.clone();
            self.write_manifest(&dir, &manifest)?;
        }

        let (label, detail) = match &outcome {
            Outcome::Stored { .. } => ("stored", "first revision".to_owned()),
            Outcome::Unchanged { .. } => (
                "unchanged",
                if previous_extract.is_some() {
                    "the extract hashes the same as the stored revision".to_owned()
                } else {
                    String::new()
                },
            ),
            Outcome::Changed {
                previous,
                stored_revision,
            } => (
                "changed",
                match stored_revision {
                    Some(id) => format!("extract moved from {previous}; stored as {id}"),
                    None => format!(
                        "extract moved from {previous}; nothing written, re-run with --update \
                         to keep the new revision"
                    ),
                },
            ),
        };

        self.log_fetch(
            &dir,
            &FetchRecord {
                at: now,
                url: key.url.clone(),
                user_agent: user_agent.clone(),
                outcome: label.to_owned(),
                detail,
                status: Some(response.status),
                raw_bytes: Some(response.body.len() as u64),
                raw_sha256: Some(raw_sha256.clone()),
                extract_sha256: Some(extract_sha256.clone()),
                revision: write_revision.clone(),
            },
        )?;

        Ok(Report {
            url: key.url,
            dir,
            outcome,
            raw_sha256,
            raw_bytes: response.body.len() as u64,
            extract_sha256,
            extract_bytes: extracted.body.len() as u64,
            redistribution: terms.redistribution,
            terms_note: terms.note,
            warnings,
        })
    }

    /// Re-run the recorded extractor over the stored raw bytes and compare.
    /// Offline: this is the check that the store still says what it said, and
    /// that the extract can be regenerated from what was kept.
    pub fn verify(&self, url: &str) -> Result<Vec<VerifyReport>> {
        let key = ItemKey::parse(url)?;
        let dir = self.item_dir(&key);
        let manifest = self
            .load_manifest(&dir)?
            .ok_or_else(|| Error::NotArchived(key.url.clone()))?;
        self.verify_manifest(&dir, &manifest)
    }

    /// Verify every archived item.
    pub fn verify_all(&self) -> Result<Vec<VerifyReport>> {
        let mut reports = Vec::new();
        for (dir, manifest) in self.items()? {
            reports.extend(self.verify_manifest(&dir, &manifest)?);
        }
        Ok(reports)
    }

    fn verify_manifest(&self, dir: &Path, manifest: &Manifest) -> Result<Vec<VerifyReport>> {
        let mut reports = Vec::new();
        for revision in &manifest.revisions {
            let raw_path = dir.join(&revision.raw.file);
            let raw = std::fs::read(&raw_path).map_err(|source| Error::Io {
                path: raw_path,
                source,
            })?;
            let content_type = revision.headers.get("content-type").map(String::as_str);
            let rules = revision.rules_or(&manifest.rules);
            let recomputed = extract::extract(&raw, content_type, rules);
            let recorded = &revision.extract.extractor;
            let same_extractor = recomputed.extractor.name == recorded.name
                && recomputed.extractor.version == recorded.version;
            reports.push(VerifyReport {
                url: manifest.url.clone(),
                revision: revision.id.clone(),
                recorded: revision.extract.sha256.clone(),
                recomputed: same_extractor.then(|| recomputed.hash()),
                extractor: revision.extract.extractor.to_string(),
            });
        }
        Ok(reports)
    }

    /// Every manifest under the root, in path order.
    pub fn items(&self) -> Result<Vec<(PathBuf, Manifest)>> {
        let mut found = Vec::new();
        let mut stack = vec![self.root.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.file_name().is_some_and(|n| n == MANIFEST_FILE)
                    && let Some(parent) = path.parent()
                    && let Some(manifest) = self.load_manifest(parent)?
                {
                    found.push((parent.to_path_buf(), manifest));
                }
            }
        }
        found.sort_by(|a, b| a.1.url.cmp(&b.1.url));
        Ok(found)
    }

    /// The manifest for one URL.
    pub fn get(&self, url: &str) -> Result<Manifest> {
        let key = ItemKey::parse(url)?;
        self.load_manifest(&self.item_dir(&key))?
            .ok_or(Error::NotArchived(key.url))
    }

    /// The fetch log for one URL, oldest first.
    pub fn history(&self, url: &str) -> Result<Vec<FetchRecord>> {
        let key = ItemKey::parse(url)?;
        let path = self.item_dir(&key).join(FETCH_LOG_FILE);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::NotArchived(key.url));
            }
            Err(source) => return Err(Error::Io { path, source }),
        };
        Ok(text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect())
    }

    /// The response for `key`, following redirects one hop at a time. Every
    /// hop is a request like the first: its origin's robots.txt is read and
    /// obeyed, and its host is spaced by the gate, before it is asked for.
    /// Returns the final response and the robots record of the origin that
    /// served it.
    fn fetch_following(
        &self,
        fetcher: &dyn Fetcher,
        key: &ItemKey,
        user_agent: &str,
        clock: &Clock,
    ) -> Result<(fetch::Response, RobotsRecord)> {
        let mut hop = key.clone();
        for hops in 0..=fetch::MAX_REDIRECTS {
            let robots = self.robots_for(fetcher, &hop, user_agent, clock)?;
            if !robots.allowed {
                return Err(Refusal::Robots {
                    path: if hops == 0 {
                        hop.path_and_query.clone()
                    } else {
                        hop.url.clone()
                    },
                    agent: fetch::PRODUCT.to_owned(),
                }
                .into());
            }

            let gate = self.gate().with_host_delay(robots.crawl_delay_secs);
            clear_gate(&gate, &hop.host, clock)?;
            gate.record_request(&hop.host, clock.now())?;
            let response = fetcher.get(&hop.url, user_agent)?;

            if let Some(refusal) = fetch::rate_limit_refusal(&hop.host, response.status) {
                hold_off(&gate, &hop.host, &response, clock)?;
                return Err(refusal.into());
            }

            let Some(target) = response.redirect_target(&hop.url) else {
                return Ok((response, robots));
            };
            if hops == fetch::MAX_REDIRECTS {
                return Err(Refusal::Redirect {
                    url: key.url.clone(),
                    target,
                    reason: format!("more than {} redirects", fetch::MAX_REDIRECTS),
                }
                .into());
            }
            hop = ItemKey::parse(&target).map_err(|error| Refusal::Redirect {
                url: hop.url.clone(),
                target: target.clone(),
                reason: error.to_string(),
            })?;
        }
        unreachable!("the loop returns on its last hop")
    }

    /// The origin's `robots.txt`, read at most once a day and reused in
    /// between. Only an answer is cached: a 5xx or a transport failure is
    /// recorded as a complete disallow (RFC 9309 §2.3.1.4) and asked again
    /// next time, and a 429 holds the host off like any other.
    fn robots_for(
        &self,
        fetcher: &dyn Fetcher,
        key: &ItemKey,
        user_agent: &str,
        clock: &Clock,
    ) -> Result<RobotsRecord> {
        let host_dir = self.root.join("hosts").join(key.origin_dir());
        let text_path = host_dir.join("robots.txt");
        let stamp_path = host_dir.join("robots.fetched");
        let evaluate = |text: &str| robots::evaluate(text, fetch::PRODUCT, &key.path_and_query);

        let fresh = std::fs::read_to_string(&stamp_path)
            .ok()
            .and_then(|s| s.trim().parse::<DateTime<Utc>>().ok())
            .is_some_and(|fetched| (clock.now() - fetched).num_seconds() < ROBOTS_TTL_SECS);

        if fresh {
            return Ok(match std::fs::read_to_string(&text_path) {
                Ok(text) => evaluate(&text),
                Err(_) => RobotsRecord::unreadable("host returned no robots.txt"),
            });
        }

        let stamp = |record: RobotsRecord| -> Result<RobotsRecord> {
            create_dir(&host_dir)?;
            write_file(&stamp_path, clock.now().to_rfc3339().as_bytes())?;
            Ok(record)
        };

        let mut url = format!("{}/robots.txt", key.origin);
        let mut host = key.host.clone();
        let mut hops = 0;
        loop {
            let gate = self.gate();
            clear_gate(&gate, &host, clock)?;
            gate.record_request(&host, clock.now())?;

            let response = match fetcher.get(&url, user_agent) {
                Ok(response) => response,
                Err(Refusal::Status { status, .. }) => fetch::Response {
                    final_url: url.clone(),
                    status,
                    headers: Vec::new(),
                    body: Vec::new(),
                },
                Err(refusal) => return Ok(RobotsRecord::unreachable(&refusal.to_string())),
            };

            let status = response.status;
            if status == 200 {
                let text = String::from_utf8_lossy(&response.body).into_owned();
                create_dir(&host_dir)?;
                write_file(&text_path, text.as_bytes())?;
                return stamp(evaluate(&text));
            }
            if status == 429 {
                hold_off(&gate, &host, &response, clock)?;
                return Err(Refusal::RateLimited { host, status }.into());
            }
            if let Some(target) = response.redirect_target(&url) {
                hops += 1;
                let next_host = url::Url::parse(&target)
                    .ok()
                    .filter(|u| matches!(u.scheme(), "http" | "https"))
                    .and_then(|u| u.host_str().map(str::to_ascii_lowercase));
                match next_host {
                    Some(next) if hops <= fetch::MAX_REDIRECTS => {
                        url = target;
                        host = next;
                        continue;
                    }
                    _ => {
                        let _ = std::fs::remove_file(&text_path);
                        return stamp(RobotsRecord::unreadable(&format!(
                            "redirected to {target} after {hops} hop(s)"
                        )));
                    }
                }
            }
            if (400..500).contains(&status) {
                let _ = std::fs::remove_file(&text_path);
                return stamp(RobotsRecord::unreadable(&format!("HTTP {status}")));
            }
            return Ok(RobotsRecord::unreachable(&format!("HTTP {status}")));
        }
    }

    fn record_refusal(
        &self,
        dir: &Path,
        key: &ItemKey,
        user_agent: &str,
        now: DateTime<Utc>,
        refusal: &Refusal,
        status: Option<u16>,
    ) -> Result<()> {
        self.log_fetch(
            dir,
            &FetchRecord {
                at: now,
                url: key.url.clone(),
                user_agent: user_agent.to_owned(),
                outcome: "refused".to_owned(),
                detail: refusal.to_string(),
                status,
                raw_bytes: None,
                raw_sha256: None,
                extract_sha256: None,
                revision: None,
            },
        )
    }

    fn log_fetch(&self, dir: &Path, record: &FetchRecord) -> Result<()> {
        use std::io::Write;
        create_dir(dir)?;
        let path = dir.join(FETCH_LOG_FILE);
        let mut line = serde_json::to_string(record).map_err(|e| Error::Other(e.to_string()))?;
        line.push('\n');
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
        file.write_all(line.as_bytes())
            .map_err(|source| Error::Io { path, source })
    }

    fn load_manifest(&self, dir: &Path) -> Result<Option<Manifest>> {
        let path = dir.join(MANIFEST_FILE);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(Error::Io { path, source }),
        };
        serde_yaml_ng::from_str(&text)
            .map(Some)
            .map_err(|e| Error::Manifest {
                path,
                message: e.to_string(),
            })
    }

    fn write_manifest(&self, dir: &Path, manifest: &Manifest) -> Result<()> {
        create_dir(dir)?;
        let text = serde_yaml_ng::to_string(manifest).map_err(|e| Error::Other(e.to_string()))?;
        write_file(&dir.join(MANIFEST_FILE), text.as_bytes())
    }
}

/// Fields the request asks an item to carry that this build does not fill in,
/// each with the reason. Recorded on the item so a reader sees the decision
/// rather than a blank.
fn absent_fields() -> Vec<AbsentField> {
    vec![AbsentField {
        field: "fetch_method: browser".to_owned(),
        reason: "this build drives no browser engine, so every item is fetched by ordinary \
                 request; access rule 3 permits a stock engine, and adding one is separate work"
            .to_owned(),
    }]
}

impl Archive {
    /// Block until no other process is archiving into this root, and hold that
    /// until the returned file drops. One run at a time is what makes the
    /// gate's per-host spacing and the manifest's load-then-save hold across
    /// processes. A run is a few requests and at most one gate wait of
    /// [`MAX_WAIT_SECS`], which bounds how long a queued run blocks.
    pub(crate) fn lock(&self) -> Result<std::fs::File> {
        create_dir(&self.root)?;
        let path = self.root.join(LOCK_FILE);
        let io = |source| Error::Io {
            path: path.clone(),
            source,
        };
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(io)?;
        file.lock().map_err(io)?;
        Ok(file)
    }
}

fn create_dir(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Write `bytes` to `path` whole or not at all: to a sibling temporary file
/// first, then renamed over the target, so a crash or a concurrent reader
/// never sees a half-written manifest or gate.
pub(crate) fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);

    let io = |source| Error::Io {
        path: path.to_path_buf(),
        source,
    };
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temp = path.with_file_name(format!(
        ".{name}.{}-{}.tmp",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&temp, bytes).map_err(io)?;
    std::fs::rename(&temp, path).map_err(|source| {
        let _ = std::fs::remove_file(&temp);
        io(source)
    })
}

/// `now` as the caller gave it, advanced by the wall time this call has spent
/// since — so a request made after waiting out the gate is recorded when it
/// was made, and an injected clock stays consistent with itself.
struct Clock {
    base: DateTime<Utc>,
    started: std::time::Instant,
}

impl Clock {
    fn start(base: DateTime<Utc>) -> Self {
        Self {
            base,
            started: std::time::Instant::now(),
        }
    }

    fn now(&self) -> DateTime<Utc> {
        self.base + chrono::TimeDelta::from_std(self.started.elapsed()).unwrap_or_default()
    }
}

/// Wait out ordinary spacing, up to [`MAX_WAIT_SECS`]; refuse past that, and
/// refuse outright while a `Retry-After` holds.
fn clear_gate(gate: &Gate, host: &str, clock: &Clock) -> Result<()> {
    match gate.clearance(host, clock.now())? {
        Clearance::Ready => Ok(()),
        Clearance::WaitFor {
            seconds,
            elapsed_secs,
            ..
        } => {
            if seconds > MAX_WAIT_SECS {
                return Err(Refusal::RateGate {
                    host: host.to_owned(),
                    elapsed_secs,
                    minimum_secs: gate.minimum_secs(),
                }
                .into());
            }
            std::thread::sleep(std::time::Duration::from_secs(seconds));
            Ok(())
        }
    }
}

/// Hold `host` off for what a rate-limiting response asked, or a day when it
/// did not say.
fn hold_off(gate: &Gate, host: &str, response: &fetch::Response, clock: &Clock) -> Result<()> {
    let now = clock.now();
    let seconds = response
        .header("retry-after")
        .and_then(|value| fetch::retry_after_secs(value, now))
        .unwrap_or(ROBOTS_TTL_SECS as u64);
    gate.record_retry_after(host, now, seconds, &format!("HTTP {}", response.status))
}

/// The HTTP status a refusal carries, for the fetch log.
fn refusal_status(refusal: &Refusal) -> Option<u16> {
    match refusal {
        Refusal::RateLimited { status, .. } | Refusal::Status { status, .. } => Some(*status),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_archiving_run_waits_for_the_first() {
        let dir = tempfile::tempdir().unwrap();
        let first = Archive::open(dir.path()).lock().unwrap();

        let second = std::fs::File::open(dir.path().join(LOCK_FILE)).unwrap();
        assert!(
            matches!(second.try_lock(), Err(std::fs::TryLockError::WouldBlock)),
            "a second holder was let in"
        );
        drop(first);
        assert!(second.try_lock().is_ok(), "the lock outlived its holder");
    }
}
