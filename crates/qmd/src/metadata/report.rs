//! Discovery's options and report: which keys, types and values the index
//! holds ([`crate::Store::list_metadata`]). Field names serialize as
//! upstream's `ListMetadataResult`.

use std::fmt;

use serde::Serialize;

use super::{MetadataFilter, MetadataMatch, Scalar, ValueType};

pub const DEFAULT_KEY_LIMIT: u64 = 50;
pub const DEFAULT_VALUE_LIMIT: u64 = 10;

/// Bound parameters one discovery statement may carry, under SQLite's
/// default variable limit (32,766) with headroom. A filter and a match each
/// pass the parser's limits alone; their combination is checked here.
pub const SQL_BINDING_BUDGET: usize = 30_000;

/// The largest integer JavaScript represents exactly, the bound upstream
/// puts on every window option.
pub const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// A document whose current extraction failed: it stays indexed, filtered
/// search excludes it, and `qmd update` will not retry it until it changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MetadataFailure {
    pub collection: String,
    pub path: String,
    pub error: String,
}

impl fmt::Display for MetadataFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}: {}", self.collection, self.path, self.error)
    }
}

/// How many keys or values a window shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Limit {
    At(u64),
    /// No window (upstream's `Infinity`).
    Unbounded,
}

/// Order of values within a key.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ValueSort {
    /// Documents descending, then value.
    #[default]
    Count,
    /// Value ascending.
    Value,
}

impl ValueSort {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "count" => Some(Self::Count),
            "value" => Some(Self::Value),
            _ => None,
        }
    }
}

/// What to report. Defaults: every collection, 50 keys, 10 values per key
/// and type, count order, no minimum.
#[derive(Debug, Clone, Default)]
pub struct ListMetadataOptions {
    /// Restrict to these collections; `None` is every collection.
    pub collections: Option<Vec<String>>,
    /// Report only the entries this matches.
    pub r#match: Option<MetadataMatch>,
    /// Count only the documents this admits.
    pub filter: Option<MetadataFilter>,
    pub key_limit: Option<Limit>,
    pub key_offset: Option<u64>,
    pub value_limit: Option<Limit>,
    pub value_offset: Option<u64>,
    pub sort: ValueSort,
    /// Drop values held by fewer documents than this.
    pub min_count: Option<u64>,
}

/// The window options with defaults applied and domains checked.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Window {
    pub key_limit: Option<i64>,
    pub key_offset: i64,
    pub value_limit: Option<i64>,
    pub value_offset: i64,
    pub sort: ValueSort,
    pub min_count: i64,
}

impl ListMetadataOptions {
    pub(crate) fn window(&self) -> Result<Window, MetadataError> {
        let limit = |value: Option<Limit>, option: &'static str, fallback: u64| match value {
            None => Ok(Some(to_i64(fallback))),
            Some(Limit::Unbounded) => Ok(None),
            Some(Limit::At(n)) if (1..=MAX_SAFE_INTEGER).contains(&n) => Ok(Some(to_i64(n))),
            Some(Limit::At(n)) => Err(MetadataError::Option {
                option,
                expected: "a positive integer or Infinity",
                received: n.to_string(),
            }),
        };
        let offset = |value: Option<u64>, option: &'static str| match value {
            None => Ok(0),
            Some(n) if n <= MAX_SAFE_INTEGER => Ok(to_i64(n)),
            Some(n) => Err(MetadataError::Option {
                option,
                expected: "a non-negative integer",
                received: n.to_string(),
            }),
        };
        let min_count = match self.min_count {
            None => 1,
            Some(n) if (1..=MAX_SAFE_INTEGER).contains(&n) => to_i64(n),
            Some(n) => {
                return Err(MetadataError::Option {
                    option: "minCount",
                    expected: "a positive integer",
                    received: n.to_string(),
                });
            }
        };
        Ok(Window {
            key_limit: limit(self.key_limit, "keyLimit", DEFAULT_KEY_LIMIT)?,
            key_offset: offset(self.key_offset, "keyOffset")?,
            value_limit: limit(self.value_limit, "valueLimit", DEFAULT_VALUE_LIMIT)?,
            value_offset: offset(self.value_offset, "valueOffset")?,
            sort: self.sort,
            min_count,
        })
    }
}

fn to_i64(n: u64) -> i64 {
    i64::try_from(n).expect("bounded by MAX_SAFE_INTEGER")
}

/// Discovery refused its input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataError {
    /// A window or ordering option outside its domain.
    Option {
        option: &'static str,
        expected: &'static str,
        received: String,
    },
    /// The filter and match together bind more parameters than
    /// [`SQL_BINDING_BUDGET`].
    BindingBudget { bindings: usize, budget: usize },
}

impl fmt::Display for MetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Option {
                option,
                expected,
                received,
            } => write!(
                f,
                "Invalid {option}: expected {expected}, received {received}"
            ),
            Self::BindingBudget { bindings, budget } => write!(
                f,
                "filter and match together bind {bindings} SQL parameters, over the budget of \
                 {budget}. Shorten their membership lists."
            ),
        }
    }
}

impl std::error::Error for MetadataError {}

/// The report.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ListMetadataResult {
    /// Active documents in scope.
    pub documents: u64,
    /// Documents in scope the filter admits; present only with a filter, and
    /// then the denominator of every coverage count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filtered_documents: Option<u64>,
    /// Keys with a matching entry, before the key window.
    pub total_keys: u64,
    /// The key window, by documents descending then key in byte order.
    pub keys: Vec<MetadataKeySummary>,
    /// Keys after the window.
    pub remaining_keys: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetadataKeySummary {
    pub key: String,
    /// Distinct documents holding a matching entry under any type.
    pub documents: u64,
    /// One per type present, by documents descending then name; more than
    /// one is a type conflict.
    pub types: Vec<MetadataKeyTypeSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MetadataKeyTypeSummary {
    #[serde(rename = "type")]
    pub value_type: ValueType,
    /// Some document holds more than one value for the key. A one-element
    /// array is indistinguishable from a scalar in the index.
    pub multi_valued: bool,
    /// Distinct documents holding a matching value of this type.
    pub documents: u64,
    /// Distinct matching values that meet `minCount`.
    pub distinct_values: u64,
    /// The value window.
    pub values: Vec<MetadataValueCount>,
    /// Distinct values after the window.
    pub remaining_values: u64,
    /// Numbers only, over every matching value row (array elements each count).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<NumberRange>,
    /// Collections contributing a matching value of this type, in byte order.
    pub collections: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct NumberRange {
    #[serde(serialize_with = "super::serialize_number_ref")]
    pub min: f64,
    #[serde(serialize_with = "super::serialize_number_ref")]
    pub median: f64,
    #[serde(serialize_with = "super::serialize_number_ref")]
    pub max: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetadataValueCount {
    pub value: Scalar,
    pub documents: u64,
    /// Of `documents`, those that inherit the value, by level, most first;
    /// the rest declare it themselves. Empty when none inherits it.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub inherited: Vec<SourceCount>,
}

/// Documents that inherit one value from one level
/// ([`crate::metadata::inherit::level_label`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceCount {
    pub source: String,
    pub documents: u64,
}

/// A key's name, coverage and types, without values: what status views show.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MetadataKeyOverview {
    pub key: String,
    pub documents: u64,
    /// By documents descending then name; more than one is a type conflict.
    pub types: Vec<ValueType>,
}

/// One collection's vocabulary size and its first keys by coverage.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MetadataOverview {
    pub total_keys: u64,
    pub keys: Vec<MetadataKeyOverview>,
}
