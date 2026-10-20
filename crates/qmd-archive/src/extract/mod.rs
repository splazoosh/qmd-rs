//! The extract: the content the document determines, and the hash a citation
//! uses.
//!
//! An extractor is named and versioned, runs offline over stored bytes, and is
//! deterministic — those three properties together are what let a later reader
//! regenerate an extract and check it against the recorded hash. Any change to
//! what an extractor emits is a version bump, because the old hash must keep
//! meaning what it meant.
//!
//! The per-item [`Rules`] are part of the identity too, hashed into it, so a
//! source that needed a hand-written selector is still reproducible from the
//! manifest alone. That is the difference this module is for: the rules are
//! data in the store, not a bespoke binary someone remembers to write.

mod charset;
mod html;

use std::fmt;

use scraper::Selector;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const HTML_TEXT: &str = "html-text";
pub const HTML_TEXT_VERSION: u32 = 2;
pub const IDENTITY: &str = "identity";
pub const IDENTITY_VERSION: u32 = 1;

/// Per-item extraction rules. Empty for almost every document; a source with
/// volatility in *visible* text needs a selector here, and recording it is what
/// keeps the extract reproducible.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rules {
    /// CSS selectors whose matching elements are dropped before extraction.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drop: Vec<String>,
}

impl Rules {
    pub fn is_empty(&self) -> bool {
        self.drop.is_empty()
    }

    /// Eight hex characters over the rule text, so the identity changes when
    /// the rules do.
    pub fn digest(&self) -> String {
        let joined = self.drop.join("\n");
        hex::encode(Sha256::digest(joined.as_bytes()))[..8].to_owned()
    }

    fn compile(&self) -> (Vec<(String, Selector)>, Vec<String>) {
        let mut compiled = Vec::new();
        let mut rejected = Vec::new();
        for source in &self.drop {
            match Selector::parse(source) {
                Ok(selector) => compiled.push((source.clone(), selector)),
                Err(_) => rejected.push(source.clone()),
            }
        }
        (compiled, rejected)
    }
}

/// What produced an extract, recorded in the manifest so it can be re-run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractorId {
    pub name: String,
    pub version: u32,
    /// Digest of the per-item rules, absent when there were none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rules: Option<String>,
}

impl fmt::Display for ExtractorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.name, self.version)?;
        if let Some(rules) = &self.rules {
            write!(f, "+rules:{rules}")?;
        }
        Ok(())
    }
}

/// One extraction.
#[derive(Debug, Clone)]
pub struct Extract {
    pub body: Vec<u8>,
    /// Whether the body is text, and so worth putting in front of the index.
    pub is_text: bool,
    pub extractor: ExtractorId,
    /// The encoding actually used to read the bytes, for text documents.
    pub charset: Option<String>,
    /// Rules that matched something. A rule that never fires is worth seeing.
    pub rules_applied: Vec<String>,
    /// Rules that would not parse as selectors, so did not run.
    pub rules_rejected: Vec<String>,
    /// Whether nesting exceeded the walker's bound and a subtree was skipped.
    pub truncated_depth: bool,
}

impl Extract {
    /// SHA-256 of the extract bytes, hex-encoded. This is the hash a citation
    /// names.
    pub fn hash(&self) -> String {
        hex::encode(Sha256::digest(&self.body))
    }

    /// The file name the extract is stored under. Text lands as `.md` so the
    /// existing collection mask picks it up; anything else is stored but not
    /// indexed, because the index is over text.
    pub fn file_name(&self) -> &'static str {
        if self.is_text {
            "extract.md"
        } else {
            "extract.bin"
        }
    }
}

/// Run the extractor appropriate to `content_type` over `raw`.
///
/// HTML goes through [`html`]; everything else is stored as itself, so its
/// extract hash equals its raw hash and reproduces trivially. That is not a
/// gap: a PDF or a JSON API response has no wrapper for an extractor to strip.
pub fn extract(raw: &[u8], content_type: Option<&str>, rules: &Rules) -> Extract {
    if is_html(content_type, raw) {
        let (source, charset) = charset::decode(raw, content_type);
        let (compiled, rules_rejected) = rules.compile();
        let result = html::extract(&source, &compiled);
        Extract {
            body: result.text.into_bytes(),
            is_text: true,
            extractor: ExtractorId {
                name: HTML_TEXT.to_owned(),
                version: HTML_TEXT_VERSION,
                rules: (!rules.is_empty()).then(|| rules.digest()),
            },
            charset: Some(charset),
            rules_applied: result.rules_applied,
            rules_rejected,
            truncated_depth: result.truncated_depth,
        }
    } else {
        let is_text = content_type.is_some_and(is_texty) && std::str::from_utf8(raw).is_ok();
        Extract {
            body: raw.to_vec(),
            is_text,
            extractor: ExtractorId {
                name: IDENTITY.to_owned(),
                version: IDENTITY_VERSION,
                rules: None,
            },
            charset: None,
            rules_applied: Vec::new(),
            rules_rejected: Vec::new(),
            truncated_depth: false,
        }
    }
}

/// Decode `raw` as text the way the extractor would read it, for readers of
/// the source (the terms) that must see the same characters.
pub fn decode_text(raw: &[u8], content_type: Option<&str>) -> String {
    charset::decode(raw, content_type).0
}

fn is_html(content_type: Option<&str>, raw: &[u8]) -> bool {
    if let Some(content_type) = content_type {
        let lower = content_type.to_ascii_lowercase();
        if lower.contains("text/html") || lower.contains("application/xhtml") {
            return true;
        }
        if !lower.is_empty() {
            return false;
        }
    }
    let head = &raw[..raw.len().min(1024)];
    let lower: String = head
        .iter()
        .map(|&b| (b as char).to_ascii_lowercase())
        .collect();
    lower.contains("<html") || lower.contains("<!doctype html")
}

fn is_texty(content_type: &str) -> bool {
    let lower = content_type.to_ascii_lowercase();
    lower.starts_with("text/")
        || lower.contains("json")
        || lower.contains("xml")
        || lower.contains("markdown")
}
