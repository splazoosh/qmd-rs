//! The terms a document arrived under, and a plain redistribution verdict.
//!
//! This is not paperwork. A licence stated on a site covers what its own text
//! scopes it to, which need not include a given file on that site, and a file
//! free to download can still be copyrighted. Free to read is not free to
//! redistribute.
//!
//! So a signal is stored with **where it was found and what it covers**, not
//! just what it said. A licence in the page's own head is a statement about
//! this document; the same words in a site footer are a statement about a site,
//! which may not reach this document. The verdict reflects that: a site-wide
//! grant alone never reads as permission, it reads as *unstated*, which is what
//! it is.

use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

use crate::robots::RobotsRecord;

/// The plain verdict. Absence of a statement is [`Unstated`], never blank.
///
/// [`Unstated`]: Redistribution::Unstated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Redistribution {
    Permitted,
    Forbidden,
    Unstated,
}

/// What a statement covers, judged by where it was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Scope {
    /// A statement about this document: the page's own head, or a response
    /// header for this URL.
    Page,
    /// A statement about the site it sits in — a footer, site chrome. It may
    /// not reach this document.
    SiteWide,
    /// Found in the body with nothing establishing what it covers.
    Unknown,
}

/// Which way a statement points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Stance {
    /// Names a licence that permits redistribution.
    Open,
    /// Reserves rights, or asks that no copy be published.
    Restrictive,
    /// A rights statement whose effect is not decidable from its text.
    Informational,
}

/// One licence-ish thing found, recorded verbatim with its provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TermsSignal {
    /// How it was found: `link-rel-license`, `meta`, `json-ld`, `anchor`,
    /// `http-header`, `x-robots-tag`, `text`.
    pub kind: String,
    /// Where in the document, for a reader who wants to go and look.
    pub locus: String,
    pub scope: Scope,
    pub stance: Stance,
    /// The statement itself, as it was written.
    pub statement: String,
}

/// Everything an item records about the terms it arrived under.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TermsRecord {
    pub redistribution: Redistribution,
    /// One line saying how the verdict was reached, including when a statement
    /// was found but judged not to cover this document.
    pub note: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub signals: Vec<TermsSignal>,
    pub robots: RobotsRecord,
}

/// Phrases that name a licence permitting redistribution.
const OPEN_MARKERS: &[&str] = &[
    "creativecommons.org/licenses",
    "creativecommons.org/publicdomain",
    "gnu.org/licenses/fdl",
    "gnu.org/licenses/gpl",
    "gnu.org/licenses/lgpl",
    "opensource.org/licenses",
    "spdx.org/licenses",
    "creative commons",
    "cc by",
    "cc0",
    "public domain",
    "gnu free documentation",
    "gfdl",
    "mit license",
    "apache license",
    "bsd license",
];

/// Phrases that reserve rights. `freeware` is here on purpose: free to
/// download is not free to redistribute.
const RESTRICTIVE_MARKERS: &[&str] = &[
    "all rights reserved",
    "may not be reproduced",
    "may not be redistributed",
    "may not be copied",
    "no part of this",
    "unauthorised reproduction",
    "unauthorized reproduction",
    "freeware",
    "noarchive",
];

/// Read the terms out of a served document.
///
/// `source` is the raw markup, not the extract: the statements live in
/// attributes and in script blocks the extractor deliberately throws away.
pub fn read(source: &str, headers: &[(String, String)], robots: RobotsRecord) -> TermsRecord {
    let mut signals = Vec::new();
    signals.extend(from_headers(headers));
    if looks_like_html(source) {
        signals.extend(from_html(source));
    }
    signals.sort_by(|a, b| (&a.kind, &a.locus).cmp(&(&b.kind, &b.locus)));
    signals.dedup_by(|a, b| a.statement == b.statement && a.kind == b.kind);

    let (redistribution, note) = decide(&signals);
    TermsRecord {
        redistribution,
        note,
        signals,
        robots,
    }
}

/// A record for something that was never fetched, or whose terms could not be
/// read. Still not blank.
pub fn unread(reason: &str, robots: RobotsRecord) -> TermsRecord {
    TermsRecord {
        redistribution: Redistribution::Unstated,
        note: format!("terms not read: {reason}"),
        signals: Vec::new(),
        robots,
    }
}

fn decide(signals: &[TermsSignal]) -> (Redistribution, String) {
    let has = |scope: Scope, stance: Stance| {
        signals
            .iter()
            .any(|s| s.scope == scope && s.stance == stance)
    };
    let page_open = has(Scope::Page, Stance::Open);
    let page_restrictive = has(Scope::Page, Stance::Restrictive);
    let site_open = has(Scope::SiteWide, Stance::Open);
    let site_restrictive = has(Scope::SiteWide, Stance::Restrictive);
    let loose_open = has(Scope::Unknown, Stance::Open);
    let loose_restrictive = has(Scope::Unknown, Stance::Restrictive);

    match (page_open, page_restrictive) {
        (true, false) => (
            Redistribution::Permitted,
            "a licence stated for this document permits redistribution".to_owned(),
        ),
        (true, true) => (
            Redistribution::Unstated,
            "this document carries both an open licence and a reservation of rights; \
             the conflict is not ours to resolve"
                .to_owned(),
        ),
        (false, true) => (
            Redistribution::Forbidden,
            "this document reserves rights".to_owned(),
        ),
        (false, false) => match (site_open, site_restrictive, loose_open, loose_restrictive) {
            (true, true, _, _) => (
                Redistribution::Unstated,
                "the site carries both an open licence and a reservation of rights, and \
                 neither is stated for this document"
                    .to_owned(),
            ),
            (true, false, _, _) => (
                Redistribution::Unstated,
                "an open licence was found, but at site scope: a site-wide grant is not \
                 evidence that it covers this document, and reading it as one is the \
                 mistake this field exists to prevent"
                    .to_owned(),
            ),
            (false, true, _, _) => (
                Redistribution::Forbidden,
                "the site reserves rights and this document states nothing else".to_owned(),
            ),
            (false, false, true, _) => (
                Redistribution::Unstated,
                "a licence was mentioned somewhere in the body with nothing establishing \
                 what it covers"
                    .to_owned(),
            ),
            (false, false, false, true) => (
                Redistribution::Forbidden,
                "a reservation of rights was found in the body".to_owned(),
            ),
            (false, false, false, false) => (
                Redistribution::Unstated,
                "no licence statement was found".to_owned(),
            ),
        },
    }
}

fn from_headers(headers: &[(String, String)]) -> Vec<TermsSignal> {
    let mut signals = Vec::new();
    for (name, value) in headers {
        let lower_name = name.to_ascii_lowercase();
        if lower_name == "link" && value.to_ascii_lowercase().contains("rel=\"license\"")
            || lower_name == "link" && value.to_ascii_lowercase().contains("rel=license")
        {
            signals.push(TermsSignal {
                kind: "http-header".to_owned(),
                locus: "Link header".to_owned(),
                scope: Scope::Page,
                stance: stance_of(value),
                statement: value.clone(),
            });
        }
        if lower_name == "x-robots-tag" {
            signals.push(TermsSignal {
                kind: "x-robots-tag".to_owned(),
                locus: "X-Robots-Tag header".to_owned(),
                scope: Scope::Page,
                stance: if value.to_ascii_lowercase().contains("noarchive") {
                    Stance::Restrictive
                } else {
                    Stance::Informational
                },
                statement: value.clone(),
            });
        }
    }
    signals
}

fn from_html(source: &str) -> Vec<TermsSignal> {
    let document = Html::parse_document(source);
    let mut signals = Vec::new();

    if let Ok(selector) = Selector::parse("head link[rel~=\"license\"]") {
        for element in document.select(&selector) {
            let href = element.value().attr("href").unwrap_or_default();
            signals.push(TermsSignal {
                kind: "link-rel-license".to_owned(),
                locus: "head link[rel=license]".to_owned(),
                scope: Scope::Page,
                stance: stance_of(href),
                statement: href.to_owned(),
            });
        }
    }

    if let Ok(selector) = Selector::parse("meta[name], meta[property]") {
        for element in document.select(&selector) {
            let name = element
                .value()
                .attr("name")
                .or_else(|| element.value().attr("property"))
                .unwrap_or_default()
                .to_ascii_lowercase();
            if !matches!(
                name.as_str(),
                "dc.rights"
                    | "dcterms.rights"
                    | "dc.license"
                    | "dcterms.license"
                    | "copyright"
                    | "rights"
                    | "license"
                    | "robots"
            ) {
                continue;
            }
            let content = element.value().attr("content").unwrap_or_default();
            if content.is_empty() {
                continue;
            }
            if name == "robots" && !content.to_ascii_lowercase().contains("noarchive") {
                continue;
            }
            signals.push(TermsSignal {
                kind: "meta".to_owned(),
                locus: format!("meta[name={name}]"),
                scope: Scope::Page,
                stance: stance_of(content),
                statement: content.to_owned(),
            });
        }
    }

    if let Ok(selector) = Selector::parse("script[type=\"application/ld+json\"]") {
        for element in document.select(&selector) {
            let text: String = element.text().collect();
            if let Some(statement) = json_ld_license(&text) {
                signals.push(TermsSignal {
                    kind: "json-ld".to_owned(),
                    locus: "script[type=application/ld+json]".to_owned(),
                    scope: Scope::Page,
                    stance: stance_of(&statement),
                    statement,
                });
            }
        }
    }

    if let Ok(selector) = Selector::parse("a[rel~=\"license\"]") {
        for element in document.select(&selector) {
            let href = element.value().attr("href").unwrap_or_default();
            let text: String = element.text().collect::<String>().trim().to_owned();
            let statement = if href.is_empty() {
                text.clone()
            } else {
                format!("{href} ({text})")
            };
            signals.push(TermsSignal {
                kind: "anchor".to_owned(),
                locus: format!("a[rel=license] in {}", region_of(&element)),
                scope: scope_of(&element),
                stance: stance_of(&statement),
                statement,
            });
        }
    }

    if let Ok(selector) = Selector::parse("body *") {
        for element in document.select(&selector) {
            if matches!(element.value().name(), "script" | "style") {
                continue;
            }
            let own_text: String = element
                .children()
                .filter_map(|child| child.value().as_text().map(|t| t.to_string()))
                .collect();
            let flat = collapse(&own_text);
            if flat.is_empty() || flat.len() > 400 {
                continue;
            }
            if let Some(stance) = phrase_stance(&flat) {
                let scope = match (scope_of(&element), stance) {
                    (Scope::Page, Stance::Open) => Scope::Unknown,
                    (scope, _) => scope,
                };
                signals.push(TermsSignal {
                    kind: "text".to_owned(),
                    locus: format!("{} text in {}", element.value().name(), region_of(&element)),
                    scope,
                    stance,
                    statement: flat,
                });
            }
        }
    }

    signals
}

/// A statement in site chrome is about the site; one in the document's own
/// content is about the document; anything else is honestly unknown.
fn scope_of(element: &scraper::ElementRef<'_>) -> Scope {
    for ancestor in element.ancestors() {
        let Some(value) = ancestor.value().as_element() else {
            continue;
        };
        let name = value.name();
        let id = value.attr("id").unwrap_or_default().to_ascii_lowercase();
        let class = value.attr("class").unwrap_or_default().to_ascii_lowercase();
        let chrome = |s: &str| {
            s.contains("footer") || s.contains("sitewide") || s.contains("site-wide") || s == "site"
        };
        if name == "footer" || chrome(&id) || chrome(&class) {
            return Scope::SiteWide;
        }
        if matches!(name, "main" | "article") {
            return Scope::Page;
        }
    }
    Scope::Unknown
}

fn region_of(element: &scraper::ElementRef<'_>) -> String {
    match scope_of(element) {
        Scope::SiteWide => "site chrome".to_owned(),
        Scope::Page => "the document body".to_owned(),
        Scope::Unknown => "an unclassified part of the page".to_owned(),
    }
}

fn stance_of(statement: &str) -> Stance {
    match phrase_stance(statement) {
        Some(stance) => stance,
        None => Stance::Informational,
    }
}

fn phrase_stance(statement: &str) -> Option<Stance> {
    let lower = statement.to_ascii_lowercase();
    let restrictive = RESTRICTIVE_MARKERS
        .iter()
        .any(|m| !occurrences(&lower, m).is_empty());
    let open = OPEN_MARKERS.iter().any(|m| {
        occurrences(&lower, m)
            .into_iter()
            .any(|at| !negated(&lower[..at]))
    });
    match (open, restrictive) {
        (_, true) => Some(Stance::Restrictive),
        (true, false) => Some(Stance::Open),
        (false, false) => None,
    }
}

/// Byte offsets where `marker` occurs as whole words: not preceded or followed
/// by a letter or digit, so "submit license" does not contain "mit license".
fn occurrences(lower: &str, marker: &str) -> Vec<usize> {
    let is_word = |c: Option<char>| c.is_some_and(char::is_alphanumeric);
    lower
        .match_indices(marker)
        .map(|(at, _)| at)
        .filter(|&at| {
            !is_word(lower[..at].chars().next_back())
                && !is_word(lower[at + marker.len()..].chars().next())
        })
        .collect()
}

/// Whether the few words before a marker negate it.
fn negated(before: &str) -> bool {
    const NEGATIONS: &[&str] = &["not", "no", "never", "isn't", "isnt", "nor", "non"];
    before
        .split(|c: char| c.is_whitespace() || c == '-')
        .rev()
        .filter(|word| !word.is_empty())
        .take(3)
        .any(|word| {
            NEGATIONS.contains(&word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\''))
        })
}

/// The `license` a JSON-LD block states for the document itself: on a
/// top-level node, on its `mainEntity`, or on a `@graph` node that is the page
/// or its article. A `license` nested anywhere else — inside an `image`, an
/// `author`, a `publisher` — is about that thing, not this document.
fn json_ld_license(text: &str) -> Option<String> {
    use serde_json::Value;
    const DOCUMENT_TYPES: &[&str] = &[
        "article",
        "newsarticle",
        "blogposting",
        "techarticle",
        "scholarlyarticle",
        "report",
        "webpage",
        "creativework",
        "book",
        "dataset",
    ];
    fn license_of(node: &Value) -> Option<String> {
        let map = node.as_object()?;
        let value = map
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("license"))?
            .1;
        match value {
            Value::String(text) => Some(text.clone()),
            Value::Object(inner) => ["@id", "url"]
                .iter()
                .find_map(|k| inner.get(*k)?.as_str().map(str::to_owned)),
            Value::Array(items) => items.iter().find_map(|i| i.as_str().map(str::to_owned)),
            _ => None,
        }
    }
    fn is_document(node: &Value) -> bool {
        let names = match node.get("@type") {
            Some(Value::String(t)) => vec![t.as_str()],
            Some(Value::Array(ts)) => ts.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        names
            .iter()
            .any(|t| DOCUMENT_TYPES.contains(&t.to_ascii_lowercase().as_str()))
    }
    fn top(node: &Value) -> Option<String> {
        if let Some(found) = license_of(node) {
            return Some(found);
        }
        if let Some(found) = node.get("mainEntity").and_then(license_of) {
            return Some(found);
        }
        node.get("@graph")?
            .as_array()?
            .iter()
            .filter(|n| is_document(n))
            .find_map(|n| license_of(n).or_else(|| n.get("mainEntity").and_then(license_of)))
    }

    let value: Value = serde_json::from_str(text).ok()?;
    match &value {
        Value::Array(items) => items.iter().find_map(top),
        _ => top(&value),
    }
}

fn collapse(raw: &str) -> String {
    raw.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn looks_like_html(source: &str) -> bool {
    let head = &source[..source.floor_char_boundary(1024)].to_ascii_lowercase();
    head.contains("<html") || head.contains("<!doctype html") || head.contains("<body")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn robots() -> RobotsRecord {
        RobotsRecord::unreadable("not fetched in this test")
    }

    #[test]
    fn nothing_found_is_unstated_and_says_so() {
        let record = read("<html><body><p>hello</p></body></html>", &[], robots());
        assert_eq!(record.redistribution, Redistribution::Unstated);
        assert!(record.note.contains("no licence statement"));
        assert!(record.signals.is_empty());
    }

    #[test]
    fn a_licence_for_this_document_permits_redistribution() {
        let source = "<html><head><link rel=\"license\" \
                      href=\"https://creativecommons.org/licenses/by/4.0/\"></head>\
                      <body><p>text</p></body></html>";
        let record = read(source, &[], robots());
        assert_eq!(record.redistribution, Redistribution::Permitted);
        assert_eq!(record.signals[0].scope, Scope::Page);
    }

    #[test]
    fn a_site_wide_grant_is_not_permission_for_this_document() {
        let source = "<html><body><main><p>the game</p></main>\
                      <footer>All content on this site is released under the \
                      <a rel=\"license\" href=\"https://www.gnu.org/licenses/fdl-1.3.html\">GFDL</a>.\
                      </footer></body></html>";
        let record = read(source, &[], robots());
        assert_eq!(record.redistribution, Redistribution::Unstated);
        assert!(record.note.contains("site scope"), "{}", record.note);
        assert!(record.signals.iter().any(|s| s.scope == Scope::SiteWide));
    }

    #[test]
    fn freeware_is_restrictive_because_free_to_read_is_not_free_to_redistribute() {
        let source = "<html><body><main><p>This game is freeware.</p></main></body></html>";
        let record = read(source, &[], robots());
        assert_eq!(record.redistribution, Redistribution::Forbidden);
        assert_eq!(record.signals[0].stance, Stance::Restrictive);
    }

    #[test]
    fn a_copyright_footer_forbids_redistribution() {
        let source =
            "<html><body><footer>&copy; 2026 Someone. All rights reserved.</footer></body></html>";
        let record = read(source, &[], robots());
        assert_eq!(record.redistribution, Redistribution::Forbidden);
    }

    #[test]
    fn a_conflict_is_reported_rather_than_resolved() {
        let source = "<html><head>\
                      <link rel=\"license\" href=\"https://creativecommons.org/licenses/by/4.0/\">\
                      <meta name=\"dc.rights\" content=\"All rights reserved\">\
                      </head><body><p>x</p></body></html>";
        let record = read(source, &[], robots());
        assert_eq!(record.redistribution, Redistribution::Unstated);
        assert!(record.note.contains("conflict"), "{}", record.note);
    }

    #[test]
    fn headers_carry_terms_too() {
        let headers = vec![
            (
                "link".to_owned(),
                "<https://creativecommons.org/licenses/by/4.0/>; rel=\"license\"".to_owned(),
            ),
            ("x-robots-tag".to_owned(), "noarchive".to_owned()),
        ];
        let record = read("<html><body>x</body></html>", &headers, robots());
        assert_eq!(record.redistribution, Redistribution::Unstated);
        assert!(record.note.contains("conflict"), "{}", record.note);
        assert!(record.signals.iter().any(|s| s.kind == "x-robots-tag"));
    }

    #[test]
    fn json_ld_is_read_even_though_the_extractor_throws_scripts_away() {
        let source = "<html><head><script type=\"application/ld+json\">\
                      {\"@type\":\"Article\",\"license\":\"https://creativecommons.org/publicdomain/zero/1.0/\"}\
                      </script></head><body>x</body></html>";
        let record = read(source, &[], robots());
        assert_eq!(record.redistribution, Redistribution::Permitted);
        assert_eq!(record.signals[0].kind, "json-ld");
    }

    #[test]
    fn a_licence_nested_inside_another_node_is_not_the_documents() {
        let source = "<html><head><script type=\"application/ld+json\">\
                      {\"@type\":\"Article\",\"image\":{\"@type\":\"ImageObject\",\
                      \"license\":\"https://creativecommons.org/licenses/by/4.0/\"}}\
                      </script></head><body>x</body></html>";
        assert_eq!(
            read(source, &[], robots()).redistribution,
            Redistribution::Unstated
        );

        let graph = "<html><head><script type=\"application/ld+json\">\
                     {\"@graph\":[{\"@type\":\"ImageObject\",\"license\":\"https://creativecommons.org/licenses/by/4.0/\"},\
                     {\"@type\":\"WebPage\",\"license\":{\"@id\":\"https://creativecommons.org/publicdomain/zero/1.0/\"}}]}\
                     </script></head><body>x</body></html>";
        let record = read(graph, &[], robots());
        assert_eq!(record.redistribution, Redistribution::Permitted);
        assert!(record.signals[0].statement.contains("publicdomain"));
    }

    #[test]
    fn licence_words_match_whole_words_and_respect_negation() {
        assert_eq!(phrase_stance("please submit licenses here"), None);
        assert_eq!(phrase_stance("submit license"), None);
        assert_eq!(
            phrase_stance("This work is not in the public domain."),
            None
        );
        assert_eq!(
            phrase_stance("Released under the MIT License."),
            Some(Stance::Open)
        );
        assert_eq!(
            phrase_stance("Dedicated to the public domain."),
            Some(Stance::Open)
        );
    }

    #[test]
    fn prose_naming_a_licence_is_not_permission() {
        let source = "<html><body><main><p>This page compares the MIT license with the GPL.\
                      </p></main></body></html>";
        let record = read(source, &[], robots());
        assert_eq!(record.redistribution, Redistribution::Unstated);
        assert_eq!(record.signals[0].scope, Scope::Unknown);
    }

    #[test]
    fn a_multibyte_character_at_the_sniff_boundary_does_not_panic() {
        let mut source = "a".repeat(1023);
        source.push('é');
        source.push_str("<html><body>x</body></html>");
        let record = read(&source, &[], robots());
        assert_eq!(record.redistribution, Redistribution::Unstated);
    }

    #[test]
    fn an_unread_document_still_records_a_verdict() {
        let record = unread("fetch refused", robots());
        assert_eq!(record.redistribution, Redistribution::Unstated);
        assert!(record.note.contains("fetch refused"));
    }
}
