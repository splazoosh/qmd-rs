//! Document metadata: typed values a markdown document declares in a
//! namespaced frontmatter block, a filter over them that every search surface
//! accepts, and discovery of what the index holds. Ported from TS qmd at the
//! commit `docs/metadata.md` names; the on-disk tables are upstream's.
//!
//! A document opts in with
//!
//! ```yaml
//! ---
//! qmd:
//!   metadata:
//!     topics: [typescript, programming]
//!     status: published
//! ---
//! ```
//!
//! The frontmatter stays ordinary content: chunking, embedding, snippets and
//! line numbers do not change, and a document without the block behaves as
//! before. Extraction happens on `qmd update` ([`crate::index`]); storage,
//! filtered search and discovery live on [`crate::Store`].

mod extract;
pub mod filter;
pub mod format;
pub mod report;

use std::fmt;

use indexmap::IndexMap;
use serde::{Serialize, Serializer};

pub use extract::{ExtractionResult, extract_document_metadata};
pub use filter::{FilterError, MetadataFilter, MetadataMatch};
pub use report::{
    ListMetadataOptions, ListMetadataResult, MetadataError, MetadataFailure, MetadataKeyOverview,
    MetadataOverview,
};

/// Bump when extraction or normalization semantics change, so existing rows
/// are re-extracted on the next `qmd update`. Shared with TS qmd: both write
/// `document_metadata.extraction_version`.
pub const METADATA_EXTRACTION_VERSION: i64 = 1;

/// Defensive limits for metadata from untrusted repositories (upstream's).
pub mod limits {
    pub const MAX_FRONTMATTER_BYTES: usize = 64 * 1024;
    pub const MAX_KEYS: usize = 64;
    pub const MAX_KEY_BYTES: usize = 128;
    /// In UTF-16 code units, as JavaScript counts a string's length.
    pub const MAX_STRING_LENGTH: usize = 1024;
    pub const MAX_ARRAY_LENGTH: usize = 128;
    /// Slack over the source's byte count for nodes produced by YAML aliases.
    pub const MAX_YAML_ALIAS_COUNT: usize = 100;
    pub const MAX_ERROR_LENGTH: usize = 200;
}

/// The stored type of one metadata value. Arrays are homogeneous, so a key
/// holds one type per document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ValueType {
    Boolean,
    Number,
    String,
}

impl ValueType {
    pub const ALL: [Self; 3] = [Self::String, Self::Number, Self::Boolean];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Number => "number",
            Self::Boolean => "boolean",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.as_str() == s)
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    String(String),
    Number(f64),
    Boolean(bool),
}

impl Scalar {
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::String(_) => ValueType::String,
            Self::Number(_) => ValueType::Number,
            Self::Boolean(_) => ValueType::Boolean,
        }
    }

    fn from_json(value: &serde_json::Value) -> Option<Self> {
        match value {
            serde_json::Value::String(s) => Some(Self::String(s.clone())),
            serde_json::Value::Number(n) => n.as_f64().map(Self::Number),
            serde_json::Value::Bool(b) => Some(Self::Boolean(*b)),
            _ => None,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("a scalar serializes")
    }
}

impl Serialize for Scalar {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::String(s) => serializer.serialize_str(s),
            Self::Number(n) => serialize_number(*n, serializer),
            Self::Boolean(b) => serializer.serialize_bool(*b),
        }
    }
}

/// An integral number serializes as an integer, as `JSON.stringify` prints it.
pub(crate) fn serialize_number<S: Serializer>(n: f64, serializer: S) -> Result<S::Ok, S::Error> {
    const SAFE: f64 = 9_007_199_254_740_991.0;
    if n.fract() == 0.0 && n.abs() <= SAFE {
        #[allow(clippy::cast_possible_truncation)]
        return serializer.serialize_i64(n as i64);
    }
    serializer.serialize_f64(n)
}

#[allow(clippy::trivially_copy_pass_by_ref)]
pub(crate) fn serialize_number_ref<S: Serializer>(
    n: &f64,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serialize_number(*n, serializer)
}

/// A key's value in one document: a scalar or a flat array of one type.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MetadataValue {
    Scalar(Scalar),
    Array(Vec<Scalar>),
}

impl MetadataValue {
    fn from_json(value: &serde_json::Value) -> Option<Self> {
        match value {
            serde_json::Value::Array(items) => items
                .iter()
                .map(Scalar::from_json)
                .collect::<Option<Vec<_>>>()
                .map(Self::Array),
            other => Scalar::from_json(other).map(Self::Scalar),
        }
    }

    /// The value's scalars: one, or the array's elements in order.
    pub fn scalars(&self) -> &[Scalar] {
        match self {
            Self::Scalar(s) => std::slice::from_ref(s),
            Self::Array(items) => items,
        }
    }
}

/// A document's metadata, in declaration order.
pub type DocumentMetadata = IndexMap<String, MetadataValue>;

/// Parse a stored `metadata_json` column, tolerating an absent or malformed
/// row as empty metadata.
pub fn parse_metadata_json(json: Option<&str>) -> DocumentMetadata {
    let Some(json) = json.filter(|j| !j.is_empty()) else {
        return DocumentMetadata::new();
    };
    let Ok(raw) = serde_json::from_str::<IndexMap<String, serde_json::Value>>(json) else {
        return DocumentMetadata::new();
    };
    raw.iter()
        .filter_map(|(k, v)| MetadataValue::from_json(v).map(|v| (k.clone(), v)))
        .collect()
}

/// A number as JavaScript's `String(n)` prints it, so text output reads the
/// same as upstream's: `3`, `2.5`, `1e+21`, `1e-7`.
pub fn js_number(n: f64) -> String {
    if n.is_nan() {
        return "NaN".to_owned();
    }
    if n.is_infinite() {
        return if n > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    if n == 0.0 {
        return "0".to_owned();
    }
    let sci = format!("{:e}", n.abs());
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("exponent is an integer");
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let k = i32::try_from(digits.len()).expect("few digits");
    let point = exp + 1;
    let sign = if n < 0.0 { "-" } else { "" };
    let body = if (k..=21).contains(&point) {
        format!("{digits}{}", "0".repeat((point - k) as usize))
    } else if 0 < point && point <= 21 {
        let (int, frac) = digits.split_at(point as usize);
        format!("{int}.{frac}")
    } else if -6 < point && point <= 0 {
        format!("0.{}{digits}", "0".repeat((-point) as usize))
    } else {
        let e = point - 1;
        let e = if e >= 0 {
            format!("+{e}")
        } else {
            e.to_string()
        };
        if k == 1 {
            format!("{digits}e{e}")
        } else {
            format!("{}.{}e{e}", &digits[..1], &digits[1..])
        }
    };
    format!("{sign}{body}")
}

/// A string's length in UTF-16 code units, as JavaScript's `.length` counts it.
pub(crate) fn js_length(s: &str) -> usize {
    s.encode_utf16().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_print_as_javascript_prints_them() {
        let cases = [
            (3.0, "3"),
            (-2.0, "-2"),
            (2.5, "2.5"),
            (-0.25, "-0.25"),
            (1e21, "1e+21"),
            (1e20, "100000000000000000000"),
            (1e-7, "1e-7"),
            (1.5e-7, "1.5e-7"),
            (0.000_001, "0.000001"),
            (1.25e308, "1.25e+308"),
            (f64::MIN_POSITIVE * f64::EPSILON, "5e-324"),
            (123.456, "123.456"),
        ];
        for (n, expected) in cases {
            assert_eq!(js_number(n), expected, "{n:e}");
        }
    }

    #[test]
    fn stored_json_keeps_declaration_order_and_integral_numbers() {
        let metadata = parse_metadata_json(Some(r#"{"zeta":1,"alpha":[2.5,3],"mid":true}"#));
        assert_eq!(
            metadata.keys().collect::<Vec<_>>(),
            ["zeta", "alpha", "mid"]
        );
        assert_eq!(
            serde_json::to_string(&metadata).unwrap(),
            r#"{"zeta":1,"alpha":[2.5,3],"mid":true}"#
        );
        assert!(parse_metadata_json(Some("not json")).is_empty());
        assert!(parse_metadata_json(None).is_empty());
    }
}
