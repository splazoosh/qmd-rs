//! Extraction of `qmd.metadata` from a document's leading YAML frontmatter,
//! ported from TS `src/metadata.ts`. Never fails: a document without the
//! block yields empty metadata, an invalid one empty metadata plus an error.

use std::cell::Cell;
use std::fmt;
use std::marker::PhantomData;

use serde::de::{self, DeserializeSeed, EnumAccess, MapAccess, SeqAccess, VariantAccess, Visitor};

use super::limits::{
    MAX_ARRAY_LENGTH, MAX_ERROR_LENGTH, MAX_FRONTMATTER_BYTES, MAX_KEY_BYTES, MAX_KEYS,
    MAX_STRING_LENGTH, MAX_YAML_ALIAS_COUNT,
};
use super::{DocumentMetadata, METADATA_EXTRACTION_VERSION, MetadataValue, Scalar, js_length};

/// What extracting one document produced. `error` is set when the frontmatter
/// does not parse, whether or not it has a `qmd` key, or when its
/// `qmd.metadata` is invalid: the document still indexes, and is excluded from
/// filtered search until corrected and re-indexed.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractionResult {
    pub metadata: DocumentMetadata,
    pub error: Option<String>,
    pub extraction_version: i64,
}

impl ExtractionResult {
    fn success(metadata: DocumentMetadata) -> Self {
        Self {
            metadata,
            error: None,
            extraction_version: METADATA_EXTRACTION_VERSION,
        }
    }

    fn failure(message: &str) -> Self {
        Self {
            metadata: DocumentMetadata::new(),
            error: Some(truncate_error(message)),
            extraction_version: METADATA_EXTRACTION_VERSION,
        }
    }
}

const FRONTMATTER_EXTENSIONS: [&str; 3] = [".md", ".markdown", ".mdx"];

/// Extract `qmd.metadata` from `content`, the file at `path`. Only markdown
/// extensions are parsed.
pub fn extract_document_metadata(content: &str, path: &str) -> ExtractionResult {
    let has_extension = path
        .rfind('.')
        .is_some_and(|i| FRONTMATTER_EXTENSIONS.contains(&path[i..].to_lowercase().as_str()));
    if !has_extension {
        return ExtractionResult::success(DocumentMetadata::new());
    }
    let Some(yaml) = frontmatter_yaml(content) else {
        return ExtractionResult::success(DocumentMetadata::new());
    };
    if yaml.len() > MAX_FRONTMATTER_BYTES {
        return ExtractionResult::failure(&format!(
            "frontmatter exceeds {MAX_FRONTMATTER_BYTES} bytes"
        ));
    }
    let frontmatter = match parse_yaml(yaml) {
        Ok(node) => node,
        Err(e) => return ExtractionResult::failure(&format!("invalid frontmatter YAML: {e}")),
    };
    let Yaml::Map(frontmatter) = frontmatter else {
        return ExtractionResult::success(DocumentMetadata::new());
    };
    let Some(namespace) = lookup(&frontmatter, "qmd") else {
        return ExtractionResult::success(DocumentMetadata::new());
    };
    let Yaml::Map(namespace) = namespace else {
        return ExtractionResult::failure("frontmatter 'qmd' must be a mapping");
    };
    let Some(raw) = lookup(namespace, "metadata") else {
        return ExtractionResult::success(DocumentMetadata::new());
    };
    let Yaml::Map(raw) = raw else {
        return ExtractionResult::failure("frontmatter 'qmd.metadata' must be a mapping");
    };
    match normalize(raw) {
        Ok(metadata) => ExtractionResult::success(metadata),
        Err(message) => ExtractionResult::failure(&message),
    }
}

/// The YAML between a leading `---` line and a closing `---` or `...` line.
/// Tolerates a UTF-8 BOM and CRLF line endings; `None` without a complete
/// leading block.
fn frontmatter_yaml(content: &str) -> Option<&str> {
    let body = content.strip_prefix('\u{feff}').unwrap_or(content);
    let rest = body.strip_prefix("---")?;
    let open_end = rest.find('\n')?;
    let opener = rest[..open_end]
        .strip_suffix('\r')
        .unwrap_or(&rest[..open_end]);
    if !opener.chars().all(|c| c == ' ' || c == '\t') {
        return None;
    }
    let yaml = &rest[open_end + 1..];
    let mut start = 0;
    while start <= yaml.len() {
        let (line, terminated) = match yaml[start..].find('\n') {
            Some(i) => (&yaml[start..start + i], true),
            None => (&yaml[start..], false),
        };
        let line = if terminated {
            line.strip_suffix('\r').unwrap_or(line)
        } else {
            line
        };
        let marker = line
            .strip_prefix("---")
            .or_else(|| line.strip_prefix("..."));
        if marker.is_some_and(|tail| tail.chars().all(|c| c == ' ' || c == '\t')) {
            return Some(&yaml[..start]);
        }
        if !terminated {
            return None;
        }
        start += yaml[start..].find('\n').expect("terminated") + 1;
    }
    None
}

/// A parsed YAML node, mappings in source order.
#[derive(Debug, Clone)]
enum Yaml {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Seq(Vec<Yaml>),
    Map(Vec<(String, Yaml)>),
}

fn lookup<'a>(map: &'a [(String, Yaml)], key: &str) -> Option<&'a Yaml> {
    map.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// Parse YAML, bounding alias expansion: every node a visitor sees counts,
/// and a document may not yield more nodes than its bytes plus
/// [`MAX_YAML_ALIAS_COUNT`]. Without aliases a node takes at least a byte of
/// source, so only expansion reaches the bound.
fn parse_yaml(yaml: &str) -> Result<Yaml, String> {
    if yaml.trim().is_empty() {
        return Ok(Yaml::Null);
    }
    let budget = Cell::new(yaml.len() + MAX_YAML_ALIAS_COUNT);
    let source = format!("\n{yaml}");
    let deserializer = serde_yaml_ng::Deserializer::from_str(&source);
    NodeSeed { budget: &budget }
        .deserialize(deserializer)
        .map_err(|e| e.to_string())
}

struct NodeSeed<'a> {
    budget: &'a Cell<usize>,
}

impl NodeSeed<'_> {
    fn tick<E: de::Error>(&self) -> Result<(), E> {
        match self.budget.get().checked_sub(1) {
            Some(left) => {
                self.budget.set(left);
                Ok(())
            }
            None => Err(E::custom(
                "excessive alias count indicates a resource exhaustion attack",
            )),
        }
    }
}

impl<'de> DeserializeSeed<'de> for NodeSeed<'_> {
    type Value = Yaml;

    fn deserialize<D: de::Deserializer<'de>>(self, deserializer: D) -> Result<Yaml, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for NodeSeed<'_> {
    type Value = Yaml;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a YAML node")
    }

    fn visit_bool<E: de::Error>(self, v: bool) -> Result<Yaml, E> {
        self.tick()?;
        Ok(Yaml::Bool(v))
    }

    #[allow(clippy::cast_precision_loss)]
    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Yaml, E> {
        self.tick()?;
        Ok(Yaml::Number(v as f64))
    }

    #[allow(clippy::cast_precision_loss)]
    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Yaml, E> {
        self.tick()?;
        Ok(Yaml::Number(v as f64))
    }

    #[allow(clippy::cast_precision_loss)]
    fn visit_i128<E: de::Error>(self, v: i128) -> Result<Yaml, E> {
        self.tick()?;
        Ok(Yaml::Number(v as f64))
    }

    #[allow(clippy::cast_precision_loss)]
    fn visit_u128<E: de::Error>(self, v: u128) -> Result<Yaml, E> {
        self.tick()?;
        Ok(Yaml::Number(v as f64))
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Yaml, E> {
        self.tick()?;
        Ok(Yaml::Number(v))
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<Yaml, E> {
        self.tick()?;
        Ok(Yaml::String(v.to_owned()))
    }

    fn visit_string<E: de::Error>(self, v: String) -> Result<Yaml, E> {
        self.tick()?;
        Ok(Yaml::String(v))
    }

    fn visit_unit<E: de::Error>(self) -> Result<Yaml, E> {
        self.tick()?;
        Ok(Yaml::Null)
    }

    fn visit_none<E: de::Error>(self) -> Result<Yaml, E> {
        self.visit_unit()
    }

    fn visit_some<D: de::Deserializer<'de>>(self, deserializer: D) -> Result<Yaml, D::Error> {
        self.deserialize(deserializer)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Yaml, A::Error> {
        self.tick()?;
        let mut items = Vec::new();
        while let Some(item) = seq.next_element_seed(NodeSeed {
            budget: self.budget,
        })? {
            items.push(item);
        }
        Ok(Yaml::Seq(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Yaml, A::Error> {
        self.tick()?;
        let mut entries: Vec<(String, Yaml)> = Vec::new();
        while let Some(key) = map.next_key_seed(NodeSeed {
            budget: self.budget,
        })? {
            let key = key_string(&key);
            let value = map.next_value_seed(NodeSeed {
                budget: self.budget,
            })?;
            if entries.iter().any(|(k, _)| *k == key) {
                return Err(de::Error::custom("Map keys must be unique"));
            }
            entries.push((key, value));
        }
        Ok(Yaml::Map(entries))
    }

    /// A tagged node (`!tag value`): the tag is dropped and the value kept,
    /// as an unknown tag is in upstream's parser.
    fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<Yaml, A::Error> {
        let (_tag, variant) = data.variant_seed(PhantomData::<String>)?;
        variant.newtype_variant_seed(self)
    }
}

/// A mapping key as JavaScript's object conversion names it.
fn key_string(key: &Yaml) -> String {
    match key {
        Yaml::String(s) => s.clone(),
        Yaml::Number(n) => super::js_number(*n),
        Yaml::Bool(b) => b.to_string(),
        Yaml::Null => String::new(),
        Yaml::Seq(_) | Yaml::Map(_) => "[object]".to_owned(),
    }
}

/// Normalize raw `qmd.metadata`. All or nothing per document, so stale
/// partial metadata can never persist.
fn normalize(raw: &[(String, Yaml)]) -> Result<DocumentMetadata, String> {
    if raw.len() > MAX_KEYS {
        return Err(format!("metadata has {} keys (max {MAX_KEYS})", raw.len()));
    }
    let mut metadata = DocumentMetadata::with_capacity(raw.len());
    for (key, value) in raw {
        validate_key(key)?;
        let value = match value {
            Yaml::Seq(items) => MetadataValue::Array(normalize_array(key, items)?),
            other => MetadataValue::Scalar(normalize_scalar(key, other)?),
        };
        metadata.insert(key.clone(), value);
    }
    Ok(metadata)
}

fn validate_key(key: &str) -> Result<(), String> {
    if key.is_empty() {
        return Err("metadata keys must be non-empty strings".to_owned());
    }
    if key.len() > MAX_KEY_BYTES {
        return Err(format!("metadata key exceeds {MAX_KEY_BYTES} bytes"));
    }
    if key.chars().any(|c| c <= '\u{1f}' || c == '\u{7f}') {
        return Err("metadata keys must not contain control characters".to_owned());
    }
    Ok(())
}

fn normalize_scalar(key: &str, value: &Yaml) -> Result<Scalar, String> {
    match value {
        Yaml::String(s) => {
            if js_length(s) > MAX_STRING_LENGTH {
                return Err(format!(
                    "metadata key \"{key}\": string exceeds {MAX_STRING_LENGTH} characters"
                ));
            }
            Ok(Scalar::String(s.clone()))
        }
        Yaml::Number(n) => {
            if !n.is_finite() {
                return Err(format!("metadata key \"{key}\": numbers must be finite"));
            }
            Ok(Scalar::Number(*n))
        }
        Yaml::Bool(b) => Ok(Scalar::Boolean(*b)),
        Yaml::Null => Err(format!(
            "metadata key \"{key}\": null is not supported — omit the key instead"
        )),
        Yaml::Seq(_) | Yaml::Map(_) => Err(format!(
            "metadata key \"{key}\": unsupported value type — use strings, numbers, booleans, \
             or flat arrays of one of those"
        )),
    }
}

fn normalize_array(key: &str, items: &[Yaml]) -> Result<Vec<Scalar>, String> {
    if items.is_empty() {
        return Err(format!(
            "metadata key \"{key}\": empty arrays are not supported — omit the key instead"
        ));
    }
    if items.len() > MAX_ARRAY_LENGTH {
        return Err(format!(
            "metadata key \"{key}\": array exceeds {MAX_ARRAY_LENGTH} values"
        ));
    }
    let mut scalars = Vec::with_capacity(items.len());
    for item in items {
        if matches!(item, Yaml::Seq(_)) {
            return Err(format!(
                "metadata key \"{key}\": nested arrays are not supported"
            ));
        }
        scalars.push(normalize_scalar(key, item)?);
    }
    let first = scalars[0].value_type();
    if scalars.iter().any(|s| s.value_type() != first) {
        return Err(format!(
            "metadata key \"{key}\": mixed-type arrays are not supported"
        ));
    }
    Ok(dedupe(scalars))
}

/// Drop repeats, keeping first-seen order.
pub(crate) fn dedupe(scalars: Vec<Scalar>) -> Vec<Scalar> {
    let mut unique: Vec<Scalar> = Vec::with_capacity(scalars.len());
    for scalar in scalars {
        if !unique.contains(&scalar) {
            unique.push(scalar);
        }
    }
    unique
}

/// One line, at most [`MAX_ERROR_LENGTH`] UTF-16 units.
fn truncate_error(message: &str) -> String {
    let single = message.split_whitespace().collect::<Vec<_>>().join(" ");
    if js_length(&single) <= MAX_ERROR_LENGTH {
        return single;
    }
    let mut out = String::new();
    let mut units = 0;
    for c in single.chars() {
        units += c.len_utf16();
        if units > MAX_ERROR_LENGTH - 3 {
            break;
        }
        out.push(c);
    }
    out.push_str("...");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(frontmatter: &str) -> String {
        format!("---\n{frontmatter}---\n\n# Title\n\nBody text.\n")
    }

    fn extract(frontmatter: &str) -> ExtractionResult {
        extract_document_metadata(&doc(frontmatter), "doc.md")
    }

    fn json(result: &ExtractionResult) -> String {
        serde_json::to_string(&result.metadata).unwrap()
    }

    fn error(frontmatter: &str) -> String {
        let result = extract(frontmatter);
        assert!(result.metadata.is_empty());
        result.error.expect("an extraction error")
    }

    #[test]
    fn documents_without_the_block_yield_empty_metadata() {
        let none = extract_document_metadata("# Title\n\nBody.\n", "doc.md");
        assert_eq!(none, ExtractionResult::success(DocumentMetadata::new()));
        for frontmatter in [
            "title: Hello\ntags: [a, b]\n",
            "qmd:\n  other: true\n",
            "qmd:\n  metadata: {}\n",
        ] {
            let result = extract(frontmatter);
            assert!(
                result.metadata.is_empty() && result.error.is_none(),
                "{frontmatter}"
            );
        }
    }

    #[test]
    fn extracts_scalars_and_homogeneous_arrays_in_order() {
        let scalars = extract(
            "qmd:\n  metadata:\n    status: published\n    priority: 3\n    reviewed: true\n",
        );
        assert_eq!(
            json(&scalars),
            r#"{"status":"published","priority":3,"reviewed":true}"#
        );
        let arrays = extract(
            "qmd:\n  metadata:\n    topics: [typescript, programming]\n    scores: [1, 2.5, 3]\n    flags: [true, false]\n",
        );
        assert_eq!(
            json(&arrays),
            r#"{"topics":["typescript","programming"],"scores":[1,2.5,3],"flags":[true,false]}"#
        );
        let deduped = extract("qmd:\n  metadata:\n    topics: [b, a, b, c, a]\n");
        assert_eq!(json(&deduped), r#"{"topics":["b","a","c"]}"#);
        let prototype =
            extract("qmd:\n  metadata:\n    __proto__: inherited\n    constructor: built\n");
        assert_eq!(
            json(&prototype),
            r#"{"__proto__":"inherited","constructor":"built"}"#
        );
        let date = extract("qmd:\n  metadata:\n    published: 2024-01-15\n");
        assert_eq!(json(&date), r#"{"published":"2024-01-15"}"#);
    }

    #[test]
    fn tolerates_bom_crlf_and_dots_closer() {
        for content in [
            "\u{feff}---\nqmd:\n  metadata:\n    status: ok\n---\nBody\n",
            "---\r\nqmd:\r\n  metadata:\r\n    status: ok\r\n---\r\nBody\r\n",
            "---\nqmd:\n  metadata:\n    status: ok\n...\nBody\n",
        ] {
            let result = extract_document_metadata(content, "doc.md");
            assert_eq!(json(&result), r#"{"status":"ok"}"#, "{content:?}");
        }
        let unclosed =
            extract_document_metadata("---\nqmd:\n  metadata:\n    status: ok\n", "doc.md");
        assert_eq!(unclosed, ExtractionResult::success(DocumentMetadata::new()));
    }

    #[test]
    fn only_markdown_extensions_are_parsed() {
        let content = doc("qmd:\n  metadata:\n    status: ok\n");
        assert!(
            extract_document_metadata(&content, "script.ts")
                .metadata
                .is_empty()
        );
        assert!(
            extract_document_metadata(&content, "noext")
                .metadata
                .is_empty()
        );
        assert_eq!(
            extract_document_metadata(&content, "doc.markdown")
                .metadata
                .len(),
            1
        );
        assert_eq!(
            extract_document_metadata(&content, "doc.MDX")
                .metadata
                .len(),
            1
        );
    }

    #[test]
    fn a_yaml_error_gives_the_line_of_the_file() {
        let message = error("title: ok\nbad: a: b\n");
        assert!(message.contains("at line 3 column"), "{message}");
    }

    #[test]
    fn invalid_blocks_record_an_error() {
        assert!(error("qmd: [unclosed\n").starts_with("invalid frontmatter YAML"));
        assert!(error("qmd: hello\n").contains("'qmd' must be a mapping"));
        assert!(error("qmd:\n  metadata: hello\n").contains("'qmd.metadata' must be a mapping"));
        let cases = [
            (
                "qmd:\n  metadata:\n    nested:\n      a: 1\n",
                "unsupported value type",
            ),
            (
                "qmd:\n  metadata:\n    empty: null\n",
                "null is not supported",
            ),
            (
                "qmd:\n  metadata:\n    empty: []\n",
                "empty arrays are not supported",
            ),
            (
                "qmd:\n  metadata:\n    nested: [[1, 2]]\n",
                "nested arrays are not supported",
            ),
            (
                "qmd:\n  metadata:\n    mixed: [1, two]\n",
                "mixed-type arrays are not supported",
            ),
            ("qmd:\n  metadata:\n    bad: .inf\n", "finite"),
            (
                "qmd:\n  metadata:\n    a: 1\n    a: 2\n",
                "invalid frontmatter YAML",
            ),
        ];
        for (frontmatter, expected) in cases {
            assert!(error(frontmatter).contains(expected), "{frontmatter}");
        }
    }

    #[test]
    fn enforces_the_size_limits() {
        let long_key = "k".repeat(MAX_KEY_BYTES + 1);
        assert!(error(&format!("qmd:\n  metadata:\n    {long_key}: 1\n")).contains("key exceeds"));
        let long_string = "v".repeat(MAX_STRING_LENGTH + 1);
        assert!(
            error(&format!("qmd:\n  metadata:\n    long: \"{long_string}\"\n"))
                .contains("string exceeds")
        );
        let big: Vec<String> = (0..=MAX_ARRAY_LENGTH).map(|i| i.to_string()).collect();
        assert!(
            error(&format!(
                "qmd:\n  metadata:\n    big: [{}]\n",
                big.join(", ")
            ))
            .contains("array exceeds")
        );
        let many: Vec<String> = (0..=MAX_KEYS).map(|i| format!("    key{i}: 1")).collect();
        assert!(error(&format!("qmd:\n  metadata:\n{}\n", many.join("\n"))).contains("keys (max"));
        let filler = format!("filler: \"{}\"\n", "x".repeat(MAX_FRONTMATTER_BYTES));
        assert!(
            error(&format!("{filler}qmd:\n  metadata:\n    status: ok\n"))
                .contains("frontmatter exceeds")
        );
        let long = error(&format!(
            "qmd:\n  metadata:\n    valid: 1\n    \"{}x\": 1\n",
            "k".repeat(600)
        ));
        assert!(js_length(&long) <= MAX_ERROR_LENGTH);
    }

    #[test]
    fn bounds_yaml_alias_expansion() {
        let bomb = "a: &a [x, x, x, x, x, x, x, x, x, x]\n\
                    b: &b [*a, *a, *a, *a, *a, *a, *a, *a, *a, *a]\n\
                    c: &c [*b, *b, *b, *b, *b, *b, *b, *b, *b, *b]\n\
                    qmd:\n  metadata:\n    status: ok\n";
        assert!(error(bomb).starts_with("invalid frontmatter YAML"));
        let reuse = extract("base: &b draft\nqmd:\n  metadata:\n    status: *b\n");
        assert_eq!(json(&reuse), r#"{"status":"draft"}"#);
    }
}
