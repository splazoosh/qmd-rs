//! Structural text extraction from HTML: the document's own content, without
//! the wrapper the host put around it.
//!
//! Most observed hash instability lives in markup rather than in text —
//! rotating email-obfuscation attributes, script nonces, per-request ids in
//! inline JSON. Emitting text and dropping script, style and every attribute
//! removes that whole class without a rule per source. What it cannot remove is
//! volatility in *visible* text (a footer clock, a "generated in 0.0123s"
//! line), because nothing in the document distinguishes that from content; the
//! per-item `drop` selectors in [`Rules`](super::Rules) are the escape hatch,
//! and they are recorded in the extract's identity so the result stays
//! reproducible.

use std::collections::BTreeSet;

use ego_tree::NodeRef;
use scraper::{ElementRef, Html, Node, Selector};

/// Elements whose content is not the document's text.
const SKIP: &[&str] = &[
    "script", "style", "noscript", "template", "svg", "math", "iframe", "object", "embed",
    "canvas", "audio", "video", "map", "form",
];

/// Elements that begin and end a block of text.
const BLOCK: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "caption",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "title",
    "tr",
    "ul",
];

/// Bound on nesting. Beyond this the subtree is not descended into: a stack
/// overflow aborts the process, which is a worse failure than a truncated
/// extract that the depth marker makes visible.
const MAX_DEPTH: usize = 512;

pub(super) struct HtmlExtract {
    pub text: String,
    pub rules_applied: Vec<String>,
    pub truncated_depth: bool,
}

/// Extract text from `document`, dropping any element matching one of
/// `drop_selectors` (given as `(source, compiled)` so the report can name the
/// rule the way the manifest stores it).
pub(super) fn extract(source: &str, drop_selectors: &[(String, Selector)]) -> HtmlExtract {
    let document = Html::parse_document(source);
    let mut walker = Walker {
        drop_selectors,
        applied: BTreeSet::new(),
        blocks: Vec::new(),
        current: String::new(),
        current_is_pre: false,
        truncated_depth: false,
    };
    walker.walk(document.tree.root(), 0);
    walker.flush();

    let mut text = walker.blocks.join("\n\n");
    if !text.is_empty() {
        text.push('\n');
    }
    HtmlExtract {
        text,
        rules_applied: walker.applied.into_iter().collect(),
        truncated_depth: walker.truncated_depth,
    }
}

struct Walker<'a> {
    drop_selectors: &'a [(String, Selector)],
    applied: BTreeSet<String>,
    blocks: Vec<String>,
    current: String,
    current_is_pre: bool,
    truncated_depth: bool,
}

impl Walker<'_> {
    fn flush(&mut self) {
        let normalised = if self.current_is_pre {
            normalise_pre(&self.current)
        } else {
            normalise_flow(&self.current)
        };
        if !normalised.is_empty() {
            self.blocks.push(normalised);
        }
        self.current.clear();
        self.current_is_pre = false;
    }

    fn walk(&mut self, node: NodeRef<'_, Node>, depth: usize) {
        if depth > MAX_DEPTH {
            self.truncated_depth = true;
            return;
        }
        match node.value() {
            Node::Text(text) => self.current.push_str(text.as_ref()),
            Node::Element(element) => {
                let name = element.name();
                if SKIP.contains(&name) {
                    return;
                }
                if let Some(element_ref) = ElementRef::wrap(node) {
                    for (source, selector) in self.drop_selectors {
                        if selector.matches(&element_ref) {
                            self.applied.insert(source.clone());
                            return;
                        }
                    }
                }
                if name == "br" {
                    let mark = if self.current_is_pre { '\n' } else { BR };
                    self.current.push(mark);
                    return;
                }

                let is_block = BLOCK.contains(&name);
                if is_block {
                    self.flush();
                    if name == "pre" {
                        self.current_is_pre = true;
                    }
                    if let Some(prefix) = block_prefix(name) {
                        self.current.push_str(prefix);
                    }
                }
                for child in node.children() {
                    self.walk(child, depth + 1);
                }
                if is_block {
                    self.flush();
                }
            }
            _ => {
                for child in node.children() {
                    self.walk(child, depth + 1);
                }
            }
        }
    }
}

/// Markdown-ish markers, so the extract keeps the document's own structure
/// where the document stated it. Nothing here is inferred from styling.
fn block_prefix(name: &str) -> Option<&'static str> {
    Some(match name {
        "title" | "h1" => "# ",
        "h2" => "## ",
        "h3" => "### ",
        "h4" => "#### ",
        "h5" => "##### ",
        "h6" => "###### ",
        "li" => "- ",
        "blockquote" => "> ",
        _ => return None,
    })
}

/// Marks a `<br>` in flow text. The HTML parser drops NUL from text or turns
/// it into U+FFFD, so the mark cannot collide with document content.
const BR: char = '\0';

/// Collapse whitespace runs, source newlines included, to a single space,
/// keeping only the line breaks the document asked for with `<br>`.
fn normalise_flow(raw: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    for line in raw.split(BR) {
        let mut out = String::new();
        for word in line.split_whitespace() {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(word);
        }
        if !out.is_empty() {
            lines.push(out);
        }
    }
    lines.join("\n")
}

/// Preformatted text keeps its shape; only line endings and trailing spaces
/// are normalised, because those vary with the server and not the document.
fn normalise_pre(raw: &str) -> String {
    let unified = raw.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<&str> = unified.split('\n').map(str::trim_end).collect();
    let start = lines.iter().position(|l| !l.is_empty()).unwrap_or(0);
    let end = lines
        .iter()
        .rposition(|l| !l.is_empty())
        .map_or(0, |i| i + 1);
    lines.get(start..end).unwrap_or(&[]).join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(source: &str) -> String {
        extract(source, &[]).text
    }

    #[test]
    fn rewrapping_the_source_does_not_change_the_extract() {
        let narrow = "<p>one two\nthree four</p>";
        let wide = "<p>one\ntwo three\n  four</p>";
        assert_eq!(text(narrow), "one two three four\n");
        assert_eq!(text(narrow), text(wide));
    }

    #[test]
    fn only_br_breaks_a_flow_line() {
        assert_eq!(text("<p>one<br>two\nthree</p>"), "one\ntwo three\n");
        assert_eq!(text("<pre>a<br>b\nc</pre>"), "a\nb\nc\n");
    }
}
