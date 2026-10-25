//! `robots.txt`, read and obeyed.
//!
//! Two jobs, and the second is easy to forget: it decides whether a fetch may
//! happen at all, and it is itself part of what an item records about the terms
//! it arrived under. A `robots.txt` that could not be read is a different fact
//! from one that was read and said nothing, so both are stored.

use serde::{Deserialize, Serialize};

/// What the host's `robots.txt` said, as recorded on an item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RobotsRecord {
    /// Whether the file could be read at all (a 404 counts as readable and
    /// permissive, which is what the standard says it means).
    pub readable: bool,
    /// The group that matched our agent, verbatim, or absent when none did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched_group: Option<String>,
    /// The rule that decided this path, verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched_rule: Option<String>,
    pub allowed: bool,
    /// A `Crawl-delay` in the matching group, if it named one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crawl_delay_secs: Option<u64>,
    /// Why the verdict is what it is, in one line, for a reader of the manifest.
    pub note: String,
}

impl RobotsRecord {
    /// The record for a host whose `robots.txt` could not be read. Unreadable
    /// is not permission: it is recorded as such, and the fetch proceeds only
    /// because the standard treats an unavailable file as no restriction.
    pub fn unreadable(reason: &str) -> Self {
        Self {
            readable: false,
            matched_group: None,
            matched_rule: None,
            allowed: true,
            crawl_delay_secs: None,
            note: format!("robots.txt could not be read ({reason}); no restriction recorded"),
        }
    }

    /// The record for a host whose `robots.txt` exists but could not be
    /// reached — a 5xx or a transport failure. RFC 9309 §2.3.1.4 reads that as
    /// a complete disallow, not as permission.
    pub fn unreachable(reason: &str) -> Self {
        Self {
            readable: false,
            matched_group: None,
            matched_rule: None,
            allowed: false,
            crawl_delay_secs: None,
            note: format!("robots.txt was unreachable ({reason}); treated as a complete disallow"),
        }
    }
}

/// Decide `path` for `agent` against the text of a `robots.txt`.
///
/// Group selection is the conventional one: a group whose agent token is a
/// case-insensitive prefix of ours wins over the wildcard group, and every
/// group that matches is merged into one (RFC 9309 §2.2.1). Within the merged
/// group the longest matching pattern wins, and `Allow` breaks a tie with
/// `Disallow`.
pub fn evaluate(text: &str, agent: &str, path: &str) -> RobotsRecord {
    let groups = parse(text);
    let agent = agent.to_ascii_lowercase();

    let names_us = |group: &&Group| {
        group
            .agents
            .iter()
            .any(|name| name != "*" && agent.starts_with(name.as_str()))
    };
    let mut chosen: Vec<&Group> = groups.iter().filter(names_us).collect();
    if chosen.is_empty() {
        chosen = groups
            .iter()
            .filter(|group| group.agents.iter().any(|name| name == "*"))
            .collect();
    }

    let merged = (!chosen.is_empty()).then(|| Group {
        agents: chosen
            .iter()
            .flat_map(|g| g.agents.iter().cloned())
            .collect(),
        rules: chosen
            .iter()
            .flat_map(|g| {
                g.rules.iter().map(|r| Rule {
                    allow: r.allow,
                    pattern: r.pattern.clone(),
                })
            })
            .collect(),
        crawl_delay_secs: chosen.iter().filter_map(|g| g.crawl_delay_secs).max(),
    });

    let Some(group) = merged.as_ref() else {
        return RobotsRecord {
            readable: true,
            matched_group: None,
            matched_rule: None,
            allowed: true,
            crawl_delay_secs: None,
            note: "robots.txt has no group for this agent".to_owned(),
        };
    };

    let path = normalise(path);
    let mut decision: Option<(&Rule, usize)> = None;
    for rule in &group.rules {
        if let Some(len) = pattern_len(&normalise(&rule.pattern), &path) {
            let better = match decision {
                None => true,
                Some((current, current_len)) => {
                    len > current_len || (len == current_len && rule.allow && !current.allow)
                }
            };
            if better {
                decision = Some((rule, len));
            }
        }
    }

    let group_name = group.agents.join(", ");
    match decision {
        Some((rule, _)) => RobotsRecord {
            readable: true,
            matched_group: Some(group_name.clone()),
            matched_rule: Some(format!(
                "{}: {}",
                if rule.allow { "Allow" } else { "Disallow" },
                rule.pattern
            )),
            allowed: rule.allow,
            crawl_delay_secs: group.crawl_delay_secs,
            note: format!("matched the {group_name} group"),
        },
        None => RobotsRecord {
            readable: true,
            matched_group: Some(group_name.clone()),
            matched_rule: None,
            allowed: true,
            crawl_delay_secs: group.crawl_delay_secs,
            note: format!("the {group_name} group has no rule for this path"),
        },
    }
}

struct Group {
    agents: Vec<String>,
    rules: Vec<Rule>,
    crawl_delay_secs: Option<u64>,
}

struct Rule {
    allow: bool,
    pattern: String,
}

fn parse(text: &str) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    let mut in_agents = false;

    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let Some((field, value)) = line.split_once(':') else {
            continue;
        };
        let field = field.trim().to_ascii_lowercase();
        let value = value.trim();

        match field.as_str() {
            "user-agent" => {
                if !in_agents || groups.is_empty() {
                    groups.push(Group {
                        agents: Vec::new(),
                        rules: Vec::new(),
                        crawl_delay_secs: None,
                    });
                    in_agents = true;
                }
                if let Some(group) = groups.last_mut()
                    && !value.is_empty()
                {
                    group.agents.push(value.to_ascii_lowercase());
                }
            }
            "allow" | "disallow" => {
                in_agents = false;
                if let Some(group) = groups.last_mut()
                    && !(field == "disallow" && value.is_empty())
                {
                    group.rules.push(Rule {
                        allow: field == "allow",
                        pattern: value.to_owned(),
                    });
                }
            }
            "crawl-delay" => {
                in_agents = false;
                if let Some(group) = groups.last_mut() {
                    group.crawl_delay_secs =
                        value.parse::<f64>().ok().map(|delay| delay.ceil() as u64);
                }
            }
            _ => in_agents = false,
        }
    }
    groups
}

/// Length of `pattern` if it matches `path`, supporting `*` and a trailing `$`.
fn pattern_len(pattern: &str, path: &str) -> Option<usize> {
    let anchored_end = pattern.ends_with('$');
    let body = if anchored_end {
        &pattern[..pattern.len() - 1]
    } else {
        pattern
    };

    wildcard_match(body.as_bytes(), path.as_bytes(), anchored_end).then_some(body.len())
}

/// Whether `pattern` (with `*` for any run of octets) matches the start of
/// `path`, or all of it when `anchored_end`. The classic greedy match with a
/// single backtrack point: a later `*` supersedes an earlier one, so this is
/// exact, not a first-occurrence approximation, and linear in practice.
fn wildcard_match(pattern: &[u8], path: &[u8], anchored_end: bool) -> bool {
    let (mut p, mut s) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    loop {
        if p == pattern.len() && (!anchored_end || s == path.len()) {
            return true;
        }
        if p < pattern.len() && pattern[p] == b'*' {
            star = Some((p, s));
            p += 1;
        } else if p < pattern.len() && s < path.len() && pattern[p] == path[s] {
            p += 1;
            s += 1;
        } else if let Some((star_p, star_s)) = star
            && star_s < path.len()
        {
            star = Some((star_p, star_s + 1));
            p = star_p + 1;
            s = star_s + 1;
        } else {
            return false;
        }
    }
}

/// Percent-encoding normalised so a pattern and a path that spell the same
/// octets compare equal: an escaped unreserved character is decoded, other
/// escapes get upper-case hex, and a raw non-ASCII octet (a pattern typed in
/// UTF-8) is escaped. `*` and `$` are left alone.
fn normalise(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        let byte = bytes[i];
        if byte == b'%'
            && let Some(hex) = text.get(i + 1..i + 3)
            && let Ok(value) = u8::from_str_radix(hex, 16)
        {
            if value.is_ascii_alphanumeric() || b"-._~".contains(&value) {
                out.push(value as char);
            } else {
                out.push_str(&format!("%{value:02X}"));
            }
            i += 3;
            continue;
        }
        if byte.is_ascii() {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const AGENT: &str = "qmd-archive";

    #[test]
    fn a_disallow_for_everyone_binds_us() {
        let record = evaluate("User-agent: *\nDisallow: /private/\n", AGENT, "/private/x");
        assert!(!record.allowed);
        assert_eq!(record.matched_rule.as_deref(), Some("Disallow: /private/"));
        assert!(record.readable);
    }

    #[test]
    fn a_group_naming_us_wins_over_the_wildcard() {
        let text = "User-agent: *\nDisallow: /\n\nUser-agent: qmd-archive\nAllow: /\n";
        let record = evaluate(text, AGENT, "/book/page.shtml");
        assert!(record.allowed);
        assert_eq!(record.matched_group.as_deref(), Some("qmd-archive"));
    }

    #[test]
    fn the_longest_pattern_decides_and_allow_breaks_a_tie() {
        let text = "User-agent: *\nDisallow: /book/\nAllow: /book/public/\n";
        assert!(evaluate(text, AGENT, "/book/public/a").allowed);
        assert!(!evaluate(text, AGENT, "/book/private/a").allowed);
    }

    #[test]
    fn wildcards_and_end_anchors_are_honoured() {
        let text = "User-agent: *\nDisallow: /*.pdf$\n";
        assert!(!evaluate(text, AGENT, "/docs/manual.pdf").allowed);
        assert!(evaluate(text, AGENT, "/docs/manual.pdf.html").allowed);
    }

    #[test]
    fn an_end_anchor_after_a_wildcard_is_matched_at_the_end_not_the_first_occurrence() {
        let text = "User-agent: *\nDisallow: /*.pdf$\n";
        assert!(!evaluate(text, AGENT, "/a.pdf/b.pdf").allowed);
        let text = "User-agent: *\nDisallow: /foo*$\n";
        assert!(!evaluate(text, AGENT, "/foobar").allowed);
        assert!(!evaluate(text, AGENT, "/foo").allowed);
        assert!(evaluate(text, AGENT, "/fo").allowed);
    }

    #[test]
    fn an_empty_user_agent_binds_nobody() {
        let text = "User-agent:\nDisallow: /\n\nUser-agent: *\nAllow: /\n";
        assert!(evaluate(text, AGENT, "/x").allowed);
    }

    #[test]
    fn every_group_that_matches_is_merged() {
        let text = "User-agent: qmd-archive\nDisallow: /a/\n\n\
                    User-agent: *\nDisallow: /\n\n\
                    User-agent: qmd-archive\nDisallow: /b/\nCrawl-delay: 9\n";
        assert!(!evaluate(text, AGENT, "/a/x").allowed);
        let b = evaluate(text, AGENT, "/b/x");
        assert!(!b.allowed);
        assert_eq!(b.crawl_delay_secs, Some(9));
        assert!(
            evaluate(text, AGENT, "/c/x").allowed,
            "the wildcard group is not ours"
        );
    }

    #[test]
    fn percent_encoding_is_compared_as_octets() {
        let text = "User-agent: *\nDisallow: /%7euser/\nDisallow: /caf\u{e9}/\n";
        assert!(!evaluate(text, AGENT, "/~user/x").allowed);
        assert!(!evaluate(text, AGENT, "/caf%C3%A9/x").allowed);
        assert!(!evaluate(text, AGENT, "/caf%c3%a9/x").allowed);
    }

    #[test]
    fn an_empty_disallow_permits_everything() {
        let record = evaluate("User-agent: *\nDisallow:\n", AGENT, "/anything");
        assert!(record.allowed);
        assert!(record.matched_rule.is_none());
    }

    #[test]
    fn a_crawl_delay_is_carried_out_of_the_matching_group() {
        let text = "User-agent: *\nCrawl-delay: 10\nDisallow: /x\n";
        assert_eq!(evaluate(text, AGENT, "/y").crawl_delay_secs, Some(10));
    }

    #[test]
    fn unreadable_is_recorded_as_unreadable_not_as_permission() {
        let record = RobotsRecord::unreadable("connection refused");
        assert!(!record.readable);
        assert!(record.allowed);
        assert!(record.note.contains("could not be read"));
        assert!(!RobotsRecord::unreachable("HTTP 503").allowed);
    }
}
