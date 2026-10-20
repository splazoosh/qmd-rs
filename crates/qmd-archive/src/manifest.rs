//! What an item carries: the manifest beside the bytes, and the fetch log.
//!
//! The manifest is a file, not a row. The index qmd already owns is
//! byte-compatible with another implementation of it, and provenance columns
//! would spend that compatibility for a feature that does not need to live
//! there. So an item is a directory: raw bytes, extract, and this.
//!
//! Two rules the layout exists to enforce. The extract file holds the extract
//! and nothing else — no header, no timestamp — because anything else in it
//! would move the hash on every fetch. And a revision is never overwritten:
//! a citation naming an old extract hash stays resolvable.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::extract::{ExtractorId, Rules};
use crate::fetch::FetchMethod;
use crate::terms::TermsRecord;

pub const MANIFEST_FILE: &str = "manifest.yml";
pub const FETCH_LOG_FILE: &str = "fetches.jsonl";
pub const RAW_FILE: &str = "raw.bin";

/// Everything known about one archived document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// The URL as requested, normalised.
    pub url: String,
    pub host: String,
    pub first_archived: DateTime<Utc>,
    /// The last time the source was read, whatever the outcome.
    pub last_checked: DateTime<Utc>,
    /// The revision a bare citation means.
    pub current: String,
    pub fetch_method: FetchMethod,
    /// The identification sent with the most recent read. Recorded per item
    /// because it can be reconfigured, and an old item should stay honest about
    /// what was actually sent.
    pub user_agent: String,
    #[serde(default, skip_serializing_if = "Rules::is_empty")]
    pub rules: Rules,
    pub terms: TermsRecord,
    pub revisions: Vec<Revision>,
    /// Fields this item does not carry, each with the reason. An absent field
    /// is a recorded decision, not a blank.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub not_recorded: Vec<AbsentField>,
}

impl Manifest {
    pub fn revision(&self, id: &str) -> Option<&Revision> {
        self.revisions.iter().find(|r| r.id == id)
    }

    pub fn current_revision(&self) -> Option<&Revision> {
        self.revision(&self.current)
    }

    /// The next revision id: `v1`, `v2`, …
    pub fn next_revision_id(&self) -> String {
        format!("v{}", self.revisions.len() + 1)
    }
}

/// One stored fetch of the document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Revision {
    pub id: String,
    pub fetched_at: DateTime<Utc>,
    /// Where the bytes came from after redirects, which is not always where
    /// they were asked for.
    pub final_url: String,
    pub status: u16,
    /// The response headers worth keeping.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    pub raw: RawRecord,
    pub extract: ExtractRecord,
    /// The rules this revision was extracted with. `--update` can replace the
    /// item's rules, so each revision keeps its own to be verified against.
    /// Absent on revisions written before this field, which used the item's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rules: Option<Rules>,
}

impl Revision {
    /// The rules to re-extract this revision with.
    pub fn rules_or<'a>(&'a self, item_rules: &'a Rules) -> &'a Rules {
        self.rules.as_ref().unwrap_or(item_rules)
    }
}

/// The response as served.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawRecord {
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
    /// Always false, and stated rather than implied: a hosted page re-renders,
    /// carries the fetching machine's locale, or rotates tokens, so this hash
    /// records a fetch and not a document. It is kept because it is evidence of
    /// what arrived, not because it can be checked against anything later.
    pub expected_to_reproduce: bool,
}

/// The extract, and everything needed to produce it again.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractRecord {
    pub file: String,
    pub bytes: u64,
    /// The hash a citation names.
    pub sha256: String,
    pub extractor: ExtractorId,
    /// The encoding the raw bytes were read as, for text documents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub charset: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rules_applied: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rules_rejected: Vec<String>,
    /// Whether the document nested deeper than the extractor descends.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated_depth: bool,
}

/// A field an item does not carry, and why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbsentField {
    pub field: String,
    pub reason: String,
}

/// One line of `fetches.jsonl`: every read of this URL, ever, including the
/// ones that stored nothing. This is where a check leaves its evidence — the
/// raw hash that moved while the extract hash held.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchRecord {
    pub at: DateTime<Utc>,
    pub url: String,
    pub user_agent: String,
    /// `stored`, `unchanged`, `changed`, or `refused`.
    pub outcome: String,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extract_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}
