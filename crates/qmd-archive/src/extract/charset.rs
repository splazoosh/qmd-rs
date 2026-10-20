//! Deciding which bytes mean which characters, and saying so.
//!
//! The encoding is part of the extract's identity: the same bytes read as
//! windows-1252 and as UTF-8 are different documents, and an extract hash that
//! does not record which one was chosen cannot be reproduced. The label the
//! caller stores is the encoding actually used, not the one that was declared.

/// Decode `raw` to text, preferring the `Content-Type` charset, then a `<meta>`
/// declaration in the head, then UTF-8. Returns the text and the label of the
/// encoding used.
pub fn decode(raw: &[u8], content_type: Option<&str>) -> (String, String) {
    let for_label = |label: String| encoding_rs::Encoding::for_label(label.as_bytes());
    let encoding = content_type
        .and_then(charset_from_content_type)
        .and_then(for_label)
        .or_else(|| {
            charset_from_meta(raw)
                .and_then(for_label)
                .map(meta_override)
        })
        .unwrap_or(encoding_rs::UTF_8);

    let (text, actual, _had_errors) = encoding.decode(raw);
    (text.into_owned(), actual.name().to_ascii_lowercase())
}

fn charset_from_content_type(content_type: &str) -> Option<String> {
    let lower = content_type.to_ascii_lowercase();
    let idx = lower.find("charset=")?;
    let value = lower[idx + "charset=".len()..]
        .trim_start()
        .trim_start_matches('"')
        .split(['"', ';', ' '])
        .next()?
        .trim();
    (!value.is_empty()).then(|| value.to_owned())
}

/// The HTML standard's prescan rule: bytes that could be read as ASCII to
/// find the `<meta>` are not UTF-16, so that label means UTF-8, and
/// `x-user-defined` means windows-1252.
fn meta_override(encoding: &'static encoding_rs::Encoding) -> &'static encoding_rs::Encoding {
    if encoding == encoding_rs::UTF_16LE || encoding == encoding_rs::UTF_16BE {
        encoding_rs::UTF_8
    } else if encoding == encoding_rs::X_USER_DEFINED {
        encoding_rs::WINDOWS_1252
    } else {
        encoding
    }
}

/// Look for a `charset` declaration in the head. Only the first 4 KiB is
/// examined, which is where the HTML standard requires the declaration to be.
fn charset_from_meta(raw: &[u8]) -> Option<String> {
    let window = &raw[..raw.len().min(4096)];
    let lower: String = window
        .iter()
        .map(|&b| (b as char).to_ascii_lowercase())
        .collect();

    let mut from = 0;
    while let Some(rel) = lower[from..].find("charset") {
        let at = from + rel + "charset".len();
        let rest = lower[at..].trim_start();
        if let Some(rest) = rest.strip_prefix('=') {
            let value = rest
                .trim_start()
                .trim_start_matches(['"', '\''])
                .split(['"', '\'', ';', ' ', '>', '/'])
                .next()
                .unwrap_or("")
                .trim();
            if !value.is_empty() {
                return Some(value.to_owned());
            }
        }
        from = at;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_type_wins_over_meta() {
        let raw = b"<meta charset=\"utf-8\"><p>caf\xe9</p>";
        let (text, label) = decode(raw, Some("text/html; charset=windows-1252"));
        assert_eq!(label, "windows-1252");
        assert!(text.contains("café"), "{text}");
    }

    #[test]
    fn meta_is_used_when_the_header_is_silent() {
        let raw = b"<html><head><meta charset=iso-8859-1></head><body>caf\xe9</body></html>";
        let (text, label) = decode(raw, Some("text/html"));
        assert_eq!(label, "windows-1252", "iso-8859-1 labels the 1252 superset");
        assert!(text.contains("café"), "{text}");
    }

    #[test]
    fn a_utf16_meta_label_means_utf8() {
        let raw = "<meta charset=utf-16><p>héllo</p>".as_bytes();
        let (text, label) = decode(raw, Some("text/html"));
        assert_eq!(label, "utf-8");
        assert!(text.contains("héllo"), "{text}");
    }

    #[test]
    fn utf8_is_the_fallback_and_the_label_is_what_was_used() {
        let (text, label) = decode("héllo".as_bytes(), None);
        assert_eq!(label, "utf-8");
        assert_eq!(text, "héllo");
    }
}
