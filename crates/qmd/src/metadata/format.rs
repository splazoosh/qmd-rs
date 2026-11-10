//! Plain-text rendering of discovery reports, shared by `qmd collection
//! metadata`, `collection show` and the MCP `metadata` tool. Ported from TS
//! `src/metadata-format.ts`: the layout is upstream's, line for line.
//!
//! A string value prints bare only when the bare form is unambiguous: one
//! that is empty, padded, holds a quote, backslash, control character, line
//! separator or a delimiter of the compact `value (count), ...` list, or
//! reads as a JSON literal or as the list's remainder tail, prints as a JSON
//! string, so what the reader sees is exactly the value.

use super::report::{
    ListMetadataResult, MetadataKeySummary, MetadataKeyTypeSummary, MetadataValueCount,
};
use super::{Scalar, ValueType, js_number};

/// ANSI sequences; [`Colors::NONE`] for plain text.
#[derive(Debug, Clone, Copy)]
pub struct Colors {
    pub reset: &'static str,
    pub dim: &'static str,
    pub bold: &'static str,
    pub cyan: &'static str,
}

impl Colors {
    pub const NONE: Self = Self {
        reset: "",
        dim: "",
        bold: "",
        cyan: "",
    };
}

/// How the caller spells its window options, for the footers that point at
/// them.
#[derive(Debug, Clone)]
pub struct FormatOptions<'a> {
    /// Name the collections contributing each type of a type-split key.
    pub show_collections: bool,
    /// e.g. `--value-limit <n>, --value-offset <n>, or --all-values`.
    pub value_window_hint: &'a str,
    /// e.g. `--key-limit <n>, --key-offset <n>, or --all-keys`.
    pub key_window_hint: &'a str,
    pub key_offset: u64,
    /// e.g. `--key-offset`.
    pub key_offset_label: &'a str,
    /// Printed in place of the key blocks when no entry is in the result.
    pub empty_message: &'a str,
    pub colors: Colors,
}

/// The filter line when a filter is in play, one block per key separated by
/// blank lines, and the key footer; or the one-line empty state.
pub fn format_key_summaries(result: &ListMetadataResult, options: &FormatOptions<'_>) -> String {
    let c = options.colors;
    let mut blocks: Vec<String> = Vec::new();
    if let Some(filtered) = result.filtered_documents {
        blocks.push(format!(
            "{}filter:{} {} of {} documents",
            c.dim,
            c.reset,
            count(filtered),
            count(result.documents)
        ));
    }
    if result.total_keys == 0 {
        blocks.push(format!("{}{}{}", c.dim, options.empty_message, c.reset));
    } else if result.keys.is_empty() {
        blocks.push(format!(
            "{}No keys at {} {}, {} {} in total.{}",
            c.dim,
            options.key_offset_label,
            count(options.key_offset),
            count(result.total_keys),
            plural(result.total_keys, "key", "keys"),
            c.reset
        ));
    }
    blocks.extend(result.keys.iter().map(|s| key_block(s, result, options)));
    if result.remaining_keys > 0 {
        blocks.push(format!(
            "{}{} more {}, use {}{}",
            c.dim,
            count(result.remaining_keys),
            plural(result.remaining_keys, "key", "keys"),
            options.key_window_hint,
            c.reset
        ));
    }
    blocks.join("\n\n")
}

/// One key: a header of name, types, coverage and distinct count, then the
/// body per type.
fn key_block(
    summary: &MetadataKeySummary,
    result: &ListMetadataResult,
    options: &FormatOptions<'_>,
) -> String {
    let c = options.colors;
    let coverage = format!(
        "{} of {} documents",
        count(summary.documents),
        count(result.filtered_documents.unwrap_or(result.documents))
    );
    let mut header = vec![
        format!("{}{}{}{}", c.cyan, c.bold, summary.key, c.reset),
        format!("{}{}{}", c.dim, type_label(summary), c.reset),
        coverage,
    ];
    let mut lines = Vec::new();
    if let [only] = summary.types.as_slice() {
        if only.value_type != ValueType::Boolean {
            header.push(format!("{} distinct", count(only.distinct_values)));
        }
        lines.push(header.join("  "));
        lines.extend(type_body(only));
    } else {
        lines.push(header.join("  "));
        lines.extend(type_split(&summary.types, options));
    }
    let remaining: u64 = summary.types.iter().map(|t| t.remaining_values).sum();
    if remaining > 0 {
        lines.push(format!(
            "{}{} more {}, use {}{}",
            c.dim,
            count(remaining),
            plural(remaining, "value", "values"),
            options.value_window_hint,
            c.reset
        ));
    }
    lines.join("\n")
}

/// What `collection show` needs besides the report.
#[derive(Debug, Clone)]
pub struct OverviewOptions<'a> {
    /// Active, extracted documents declaring at least one key.
    pub documents_with_metadata: u64,
    /// Active documents awaiting extraction.
    pub pending_metadata: u64,
    /// Active documents whose extraction failed.
    pub failed_metadata: u64,
    /// e.g. `qmd collection metadata notes`.
    pub drill_down_hint: &'a str,
    pub colors: Colors,
}

/// The `Metadata:` section of `collection show`: a coverage line, the keys
/// as aligned rows with a value preview, and a pointer at the drill-down
/// when keys were left out.
pub fn format_overview(result: &ListMetadataResult, options: &OverviewOptions<'_>) -> String {
    let c = options.colors;
    let mut notes = Vec::new();
    if options.pending_metadata > 0 {
        notes.push(format!(
            "{} pending extraction",
            count(options.pending_metadata)
        ));
    }
    if options.failed_metadata > 0 {
        notes.push(format!(
            "{} with invalid frontmatter",
            count(options.failed_metadata)
        ));
    }
    let pending = if notes.is_empty() {
        String::new()
    } else {
        format!(" ({})", notes.join(", "))
    };
    if result.total_keys == 0 {
        return format!("  Metadata: none{pending}");
    }
    let mut lines = vec![format!(
        "  Metadata: {} {}, {} of {} documents{pending}",
        count(result.total_keys),
        plural(result.total_keys, "key", "keys"),
        count(options.documents_with_metadata),
        count(result.documents)
    )];
    let keys = &result.keys;
    let width = |f: &dyn Fn(&MetadataKeySummary) -> usize| keys.iter().map(f).max().unwrap_or(0);
    let key_width = width(&|s| s.key.chars().count());
    let type_width = width(&|s| type_label(s).chars().count());
    let documents_width = width(&|s| count(s.documents).len());
    let distinct_width = width(&|s| count(distinct_values(s)).len());
    for summary in keys {
        let mut columns = vec![
            format!("{}{}{}", c.cyan, pad_end(&summary.key, key_width), c.reset),
            format!(
                "{}{}{}",
                c.dim,
                pad_end(&type_label(summary), type_width),
                c.reset
            ),
            format!(
                "{} {}",
                pad_start(&count(summary.documents), documents_width),
                documents_label(summary.documents)
            ),
            format!(
                "{} distinct",
                pad_start(&count(distinct_values(summary)), distinct_width)
            ),
        ];
        let preview = value_preview(summary, options.drill_down_hint);
        if !preview.is_empty() {
            columns.push(preview);
        }
        lines.push(format!("    {}", columns.join("  ")));
    }
    if result.remaining_keys > 0 {
        lines.push(format!(
            "    {}{} more {}, see '{}'{}",
            c.dim,
            count(result.remaining_keys),
            plural(result.remaining_keys, "key", "keys"),
            options.drill_down_hint,
            c.reset
        ));
    }
    lines.join("\n")
}

/// One-line preview: strings list the window (nothing when every value is
/// unique), numbers give the range, booleans the two counts, and a type
/// conflict points at the drill-down.
fn value_preview(summary: &MetadataKeySummary, drill_down_hint: &str) -> String {
    let [only] = summary.types.as_slice() else {
        let r#match = serde_json::json!({ "field": "key", "operator": "eq", "value": summary.key });
        let quoted = r#match.to_string().replace('\'', "'\\''");
        return format!("types disagree, see: {drill_down_hint} --match '{quoted}'");
    };
    match only.value_type {
        ValueType::Boolean => boolean_counts(&only.values).replacen("  ", ", ", 1),
        ValueType::Number => {
            let range = only.range.expect("numbers carry a range");
            format!(
                "{} to {}, median {}",
                js_number(range.min),
                js_number(range.max),
                js_number(range.median)
            )
        }
        ValueType::String => {
            if only.distinct_values == only.documents {
                return String::new();
            }
            let mut preview: Vec<String> = only
                .values
                .iter()
                .map(|v| format!("{} ({})", format_value(&v.value), count(v.documents)))
                .collect();
            if only.remaining_values > 0 {
                preview.push("...".to_owned());
            }
            preview.join(", ")
        }
    }
}

fn distinct_values(summary: &MetadataKeySummary) -> u64 {
    summary.types.iter().map(|t| t.distinct_values).sum()
}

/// One type: vertical values for strings, a range then inline values for
/// numbers, one line for booleans.
fn type_body(t: &MetadataKeyTypeSummary) -> Vec<String> {
    match t.value_type {
        ValueType::Boolean => vec![format!("  {}", boolean_counts(&t.values))],
        ValueType::Number => {
            let mut lines = vec![format!("  {}", range_text(t))];
            if !t.values.is_empty() {
                lines.push(format!("  {}", inline_values(t).join("  ")));
            }
            lines
        }
        ValueType::String => {
            let value_width = t
                .values
                .iter()
                .map(|v| format_value(&v.value).chars().count())
                .max()
                .unwrap_or(0);
            let count_width = t
                .values
                .iter()
                .map(|v| count(v.documents).len())
                .max()
                .unwrap_or(0);
            t.values
                .iter()
                .map(|v| {
                    format!(
                        "  {}  {}",
                        pad_end(&format_value(&v.value), value_width),
                        pad_start(&count(v.documents), count_width)
                    )
                })
                .collect()
        }
    }
}

/// A key whose documents disagree on type: one row per type with its own
/// document count and a compact value summary, plus the contributing
/// collections when asked.
fn type_split(types: &[MetadataKeyTypeSummary], options: &FormatOptions<'_>) -> Vec<String> {
    let type_width = types
        .iter()
        .map(|t| t.value_type.as_str().len())
        .max()
        .unwrap_or(0);
    let documents_width = types
        .iter()
        .map(|t| count(t.documents).len())
        .max()
        .unwrap_or(0);
    let rows: Vec<(String, String)> = types
        .iter()
        .map(|t| {
            let values = match t.value_type {
                ValueType::Boolean => boolean_counts(&t.values),
                ValueType::Number => range_text(t),
                ValueType::String => {
                    let mut items: Vec<String> = t
                        .values
                        .iter()
                        .map(|v| format!("{} ({})", format_value(&v.value), count(v.documents)))
                        .collect();
                    if t.remaining_values > 0 {
                        items.push(format!("{} more", count(t.remaining_values)));
                    }
                    items.join(", ")
                }
            };
            let row = format!(
                "  {}  {} {}  {values}",
                pad_end(t.value_type.as_str(), type_width),
                pad_start(&count(t.documents), documents_width),
                documents_label(t.documents)
            );
            (row, t.collections.join(", "))
        })
        .collect();
    if !options.show_collections {
        return rows.into_iter().map(|(row, _)| row).collect();
    }
    let row_width = rows
        .iter()
        .map(|(row, _)| row.chars().count())
        .max()
        .unwrap_or(0);
    rows.into_iter()
        .map(|(row, collections)| format!("{}  {collections}", pad_end(&row, row_width)))
        .collect()
}

fn type_label(summary: &MetadataKeySummary) -> String {
    summary
        .types
        .iter()
        .map(|t| {
            if t.multi_valued {
                format!("{}[]", t.value_type)
            } else {
                t.value_type.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn range_text(t: &MetadataKeyTypeSummary) -> String {
    let range = t.range.expect("numbers carry a range");
    format!(
        "min {}  median {}  max {}",
        js_number(range.min),
        js_number(range.median),
        js_number(range.max)
    )
}

/// Numbers inline as `value (documents)`: in value order when the whole
/// distribution fits, else in the requested order so the commonest stay.
fn inline_values(t: &MetadataKeyTypeSummary) -> Vec<String> {
    let mut values: Vec<&MetadataValueCount> = t.values.iter().collect();
    if t.remaining_values == 0 {
        values.sort_by(|a, b| number_of(&a.value).total_cmp(&number_of(&b.value)));
    }
    values
        .iter()
        .map(|v| format!("{} ({})", format_value(&v.value), count(v.documents)))
        .collect()
}

fn number_of(scalar: &Scalar) -> f64 {
    match scalar {
        Scalar::Number(n) => *n,
        _ => 0.0,
    }
}

fn boolean_counts(values: &[MetadataValueCount]) -> String {
    let find = |b: bool| values.iter().find(|v| v.value == Scalar::Boolean(b));
    let mut parts = Vec::new();
    if let Some(v) = find(true) {
        parts.push(format!("true {}", count(v.documents)));
    }
    if let Some(v) = find(false) {
        parts.push(format!("false {}", count(v.documents)));
    }
    parts.join("  ")
}

/// Padded so `doc` and `docs` rows stay aligned.
fn documents_label(documents: u64) -> &'static str {
    if documents == 1 { "doc " } else { "docs" }
}

fn plural(n: u64, one: &'static str, many: &'static str) -> &'static str {
    if n == 1 { one } else { many }
}

pub fn format_value(value: &Scalar) -> String {
    match value {
        Scalar::String(s) if is_unambiguous_bare(s) => s.clone(),
        Scalar::String(s) => quoted(s),
        Scalar::Number(n) => js_number(*n),
        Scalar::Boolean(b) => b.to_string(),
    }
}

/// C0, DEL, C1 and the line and paragraph separators, none of which print
/// as themselves.
fn is_control(c: char) -> bool {
    matches!(c, '\u{0}'..='\u{1f}' | '\u{7f}'..='\u{9f}' | '\u{2028}' | '\u{2029}')
}

/// JavaScript's `\s`.
fn is_js_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{b}' | '\u{c}' | '\r' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200a}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202f}'
                | '\u{205f}'
                | '\u{3000}'
                | '\u{feff}'
    )
}

fn is_unambiguous_bare(text: &str) -> bool {
    if text.is_empty() || text.contains('"') || text.contains('\\') {
        return false;
    }
    if text.chars().any(is_control) {
        return false;
    }
    let chars: Vec<char> = text.chars().collect();
    let ambiguous_space = is_js_space(chars[0])
        || is_js_space(chars[chars.len() - 1])
        || chars
            .windows(2)
            .any(|w| is_js_space(w[0]) && is_js_space(w[1]))
        || chars.iter().any(|&c| c != ' ' && is_js_space(c));
    if ambiguous_space {
        return false;
    }
    if text.contains([',', '(', ')']) {
        return false;
    }
    let tail = text == "..."
        || text
            .strip_suffix(" more")
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
    if tail {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(text).is_err()
}

/// A JSON string with every [`is_control`] character escaped, not only the
/// ones JSON escapes.
fn quoted(text: &str) -> String {
    let json = serde_json::to_string(text).expect("a string serializes");
    let mut out = String::with_capacity(json.len());
    for c in json.chars() {
        if matches!(c, '\u{7f}'..='\u{9f}' | '\u{2028}' | '\u{2029}') {
            out.push_str(&format!("\\u{:04x}", u32::from(c)));
        } else {
            out.push(c);
        }
    }
    out
}

/// `1,234`, as `toLocaleString("en-US")` groups a count.
pub fn count(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, d) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(d);
    }
    out
}

fn pad_end(s: &str, width: usize) -> String {
    let len = s.chars().count();
    format!("{s}{}", " ".repeat(width.saturating_sub(len)))
}

fn pad_start(s: &str, width: usize) -> String {
    let len = s.chars().count();
    format!("{}{s}", " ".repeat(width.saturating_sub(len)))
}

#[cfg(test)]
mod tests {
    use super::super::report::NumberRange;
    use super::*;

    const OPTIONS: FormatOptions<'static> = FormatOptions {
        show_collections: false,
        value_window_hint: "--value-limit <n>",
        key_window_hint: "--key-limit <n>",
        key_offset: 0,
        key_offset_label: "--key-offset",
        empty_message: "No metadata matches.",
        colors: Colors::NONE,
    };

    fn strings(values: &[&str], remaining: u64) -> MetadataKeyTypeSummary {
        MetadataKeyTypeSummary {
            value_type: ValueType::String,
            multi_valued: false,
            documents: values.len() as u64,
            distinct_values: values.len() as u64 + remaining,
            values: values
                .iter()
                .map(|v| MetadataValueCount {
                    value: Scalar::String((*v).to_owned()),
                    documents: 1,
                })
                .collect(),
            remaining_values: remaining,
            range: None,
            collections: vec!["notes".to_owned()],
        }
    }

    fn result(types: Vec<MetadataKeyTypeSummary>) -> ListMetadataResult {
        let documents = types.iter().map(|t| t.documents).sum();
        ListMetadataResult {
            documents,
            filtered_documents: None,
            total_keys: 1,
            keys: vec![MetadataKeySummary {
                key: "label".to_owned(),
                documents,
                types,
            }],
            remaining_keys: 0,
        }
    }

    fn seven() -> MetadataKeyTypeSummary {
        MetadataKeyTypeSummary {
            value_type: ValueType::Number,
            multi_valued: false,
            documents: 1,
            distinct_values: 1,
            values: vec![MetadataValueCount {
                value: Scalar::Number(7.0),
                documents: 1,
            }],
            remaining_values: 0,
            range: Some(NumberRange {
                min: 7.0,
                median: 7.0,
                max: 7.0,
            }),
            collections: vec!["notes".to_owned()],
        }
    }

    /// The value column of each row of a one-type string key.
    fn printed(values: &[&str]) -> Vec<String> {
        format_key_summaries(&result(vec![strings(values, 0)]), &OPTIONS)
            .lines()
            .skip(1)
            .map(|line| {
                let line = &line[2..];
                line.trim_end_matches(|c: char| c.is_ascii_digit())
                    .trim_end_matches(' ')
                    .to_owned()
            })
            .collect()
    }

    #[test]
    fn plain_strings_print_bare_and_ambiguous_ones_quoted() {
        let plain = ["published", "docs team", "v1.2-rc", "é"];
        assert_eq!(printed(&plain), plain);
        let ambiguous = [
            "",
            "\"\"",
            " padded",
            "padded ",
            "two  spaces",
            "tab\tin",
            "line\nbreak",
            "42",
            "-1.5",
            "true",
            "null",
            "[1]",
            "back\\slash",
        ];
        for (shown, value) in printed(&ambiguous).iter().zip(ambiguous) {
            assert!(shown.starts_with('"') && shown.ends_with('"'), "{shown}");
            assert_eq!(serde_json::from_str::<String>(shown).unwrap(), value);
        }
    }

    #[test]
    fn every_control_character_is_escaped() {
        let controls = [
            "bell\u{7}",
            "del\u{7f}",
            "csi\u{9b}",
            "line\u{2028}sep",
            "para\u{2029}sep",
            "nul\u{0}",
        ];
        let output = format_key_summaries(&result(vec![strings(&controls, 0)]), &OPTIONS);
        assert!(!output.chars().any(|c| is_control(c) && c != '\n'));
        for (shown, value) in printed(&controls).iter().zip(controls) {
            assert_eq!(serde_json::from_str::<String>(shown).unwrap(), value);
        }
    }

    #[test]
    fn list_delimiters_and_tails_are_quoted_in_both_layouts() {
        let values = ["a (1), b", "c", "3 more", "...", "(x)", "plain value"];
        let output = format_key_summaries(&result(vec![strings(&values, 2), seven()]), &OPTIONS);
        assert_eq!(
            output.lines().nth(1).unwrap(),
            r#"  string  6 docs  "a (1), b" (1), c (1), "3 more" (1), "..." (1), "(x)" (1), plain value (1), 2 more"#
        );
        assert_eq!(
            printed(&values),
            [
                r#""a (1), b""#,
                "c",
                r#""3 more""#,
                r#""...""#,
                r#""(x)""#,
                "plain value"
            ]
        );
    }

    #[test]
    fn numbers_and_booleans_print_bare() {
        let numbers = MetadataKeyTypeSummary {
            value_type: ValueType::Number,
            multi_valued: false,
            documents: 2,
            distinct_values: 2,
            values: vec![
                MetadataValueCount {
                    value: Scalar::Number(1.5),
                    documents: 1,
                },
                MetadataValueCount {
                    value: Scalar::Number(-2.0),
                    documents: 1,
                },
            ],
            remaining_values: 0,
            range: Some(NumberRange {
                min: -2.0,
                median: -0.25,
                max: 1.5,
            }),
            collections: vec!["notes".to_owned()],
        };
        let mut r = result(vec![numbers]);
        r.keys[0].key = "n".to_owned();
        assert_eq!(
            format_key_summaries(&r, &OPTIONS),
            "n  number  2 of 2 documents  2 distinct\n  min -2  median -0.25  max 1.5\n  -2 (1)  1.5 (1)"
        );
    }

    #[test]
    fn empty_states_keep_the_filter_line() {
        let none = ListMetadataResult {
            documents: 3,
            filtered_documents: Some(1),
            total_keys: 0,
            keys: vec![],
            remaining_keys: 0,
        };
        assert_eq!(
            format_key_summaries(&none, &OPTIONS),
            "filter: 1 of 3 documents\n\nNo metadata matches."
        );
        let past = ListMetadataResult {
            documents: 3,
            filtered_documents: Some(2),
            total_keys: 4,
            keys: vec![],
            remaining_keys: 0,
        };
        let options = FormatOptions {
            key_offset: 10,
            ..OPTIONS
        };
        assert_eq!(
            format_key_summaries(&past, &options),
            "filter: 2 of 3 documents\n\nNo keys at --key-offset 10, 4 keys in total."
        );
    }

    #[test]
    fn a_type_conflict_hint_quotes_the_key_for_a_shell() {
        let mut r = result(vec![strings(&["a", "b"], 0), seven()]);
        r.keys[0].key = "lead's".to_owned();
        let overview = format_overview(
            &r,
            &OverviewOptions {
                documents_with_metadata: 3,
                pending_metadata: 0,
                failed_metadata: 0,
                drill_down_hint: "qmd collection metadata notes",
                colors: Colors::NONE,
            },
        );
        assert!(overview.ends_with(
            r#"types disagree, see: qmd collection metadata notes --match '{"field":"key","operator":"eq","value":"lead'\''s"}'"#
        ), "{overview}");
    }

    #[test]
    fn counts_group_thousands() {
        assert_eq!(count(0), "0");
        assert_eq!(count(999), "999");
        assert_eq!(count(1_000), "1,000");
        assert_eq!(count(1_234_567), "1,234,567");
    }
}
