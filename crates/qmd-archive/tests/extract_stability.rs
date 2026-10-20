//! The claim the archive rests on: two responses that differ can still yield
//! one extract, and that extract's hash is what a citation names.
//!
//! The fixtures are one page fetched twice, differing exactly where hosted
//! pages were observed to differ: a rotating script nonce, an inline build
//! timestamp, Cloudflare's rotating email-obfuscation hex, and a footer line
//! carrying a render time and a session id.

use qmd_archive::extract::{HTML_TEXT, Rules, extract};

const A: &[u8] = include_bytes!("../fixtures/rotating-a.html");
const B: &[u8] = include_bytes!("../fixtures/rotating-b.html");
const HTML: Option<&str> = Some("text/html; charset=utf-8");

#[test]
fn the_responses_really_do_differ() {
    assert_ne!(
        A, B,
        "the fixtures must not be identical or nothing is tested"
    );
}

#[test]
fn markup_volatility_is_stripped_without_any_per_source_rule() {
    let a = extract(A, HTML, &Rules::default());
    let b = extract(B, HTML, &Rules::default());
    let text_a = String::from_utf8(a.body).unwrap();
    let text_b = String::from_utf8(b.body).unwrap();

    for (line_a, line_b) in text_a.lines().zip(text_b.lines()) {
        if line_a.starts_with("Page generated in") {
            continue;
        }
        assert_eq!(
            line_a, line_b,
            "markup-level volatility reached the extract"
        );
    }
    assert!(
        !text_a.contains("cfemail"),
        "an attribute reached the extract"
    );
    assert!(
        !text_a.contains("__BUILD__"),
        "script content reached the extract"
    );
    assert!(
        text_a.contains("[email protected]"),
        "visible text was lost:\n{text_a}"
    );
}

#[test]
fn visible_text_volatility_needs_a_recorded_rule_and_then_reproduces() {
    let rules = Rules {
        drop: vec!["#footer".to_owned()],
    };
    let a = extract(A, HTML, &rules);
    let b = extract(B, HTML, &rules);

    assert_eq!(
        a.hash(),
        b.hash(),
        "extract did not reproduce across fetches"
    );
    assert_eq!(a.rules_applied, vec!["#footer".to_owned()]);
    assert!(a.rules_rejected.is_empty());
    assert_eq!(a.extractor.name, HTML_TEXT);
    assert_eq!(a.extractor.rules.as_deref(), Some(rules.digest().as_str()));
    assert!(
        !String::from_utf8(a.body)
            .unwrap()
            .contains("Page generated"),
        "the dropped element survived"
    );
}

#[test]
fn the_documents_own_structure_survives() {
    let text = String::from_utf8(extract(A, HTML, &Rules::default()).body).unwrap();
    assert!(text.starts_with("# The ULA, chapter one\n"), "{text}");
    assert!(text.contains("\n# Chapter one\n"), "{text}");
    assert!(text.contains("- ROM timing"), "{text}");
    assert!(
        text.contains("  LD A, 0\n  OUT (254), A"),
        "pre lost its shape:\n{text}"
    );
    assert!(
        text.contains("clocked at 7 MHz."),
        "nbsp or wrapping leaked:\n{text}"
    );
}

#[test]
fn identity_is_used_when_there_is_no_wrapper_to_strip() {
    let raw = b"{\"title\":\"a json api response\"}";
    let result = extract(raw, Some("application/json"), &Rules::default());
    assert_eq!(result.body, raw);
    assert_eq!(result.extractor.to_string(), "identity/1");
    assert!(result.is_text, "json should still reach the index");
}

#[test]
fn a_rule_that_does_not_parse_is_reported_rather_than_silently_ignored() {
    let rules = Rules {
        drop: vec!["#footer".to_owned(), "!!not a selector".to_owned()],
    };
    let result = extract(A, HTML, &rules);
    assert_eq!(result.rules_rejected, vec!["!!not a selector".to_owned()]);
    assert_eq!(result.rules_applied, vec!["#footer".to_owned()]);
}
