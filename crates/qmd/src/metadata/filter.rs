//! The metadata predicate: one recursive, `operator`-discriminated JSON AST,
//! strict validation, and parameterized SQL. Ported from TS
//! `src/metadata-filter.ts`; the JSON shape and error messages are upstream's.
//!
//! ```json
//! { "operator": "and", "operands": [ ... ] }
//! { "operator": "not", "operand": { ... } }
//! { "field": "status", "operator": "eq", "value": "published" }
//! { "field": "topics", "operator": "prefix", "value": "sql", "caseInsensitive": true }
//! ```
//!
//! The grammar evaluates two kinds of record. A [`MetadataFilter`] tests a
//! document, whose fields are its metadata keys; every search surface takes
//! one. A [`MetadataMatch`] tests one metadata entry, whose fields are `key`
//! and `value`; discovery takes one. Operands are typed: a string compares
//! only with strings, a number with numbers, a boolean with booleans, and a
//! mismatch never matches. Every user key and value binds as a parameter.

use std::fmt;

use rusqlite::types::Value as SqlValue;
use serde_json::{Map, Value};

use super::limits::{MAX_KEY_BYTES, MAX_STRING_LENGTH};
use super::{Scalar, ValueType, js_length};

/// Limits for recursive predicates from untrusted callers (upstream's).
pub mod limits {
    pub const MAX_DEPTH: usize = 16;
    pub const MAX_NODES: usize = 256;
    pub const MAX_GROUP_OPERANDS: usize = 32;
    pub const MAX_MEMBERSHIP_VALUES: usize = 64;
}
use limits::{MAX_DEPTH, MAX_GROUP_OPERANDS, MAX_MEMBERSHIP_VALUES, MAX_NODES};

/// A condition's operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
    Nin,
    All,
    Contains,
    Prefix,
    Suffix,
    Type,
    Exists,
}

impl Op {
    const ALL: [Self; 14] = [
        Self::Eq,
        Self::Ne,
        Self::Gt,
        Self::Gte,
        Self::Lt,
        Self::Lte,
        Self::In,
        Self::Nin,
        Self::All,
        Self::Contains,
        Self::Prefix,
        Self::Suffix,
        Self::Type,
        Self::Exists,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Gt => "gt",
            Self::Gte => "gte",
            Self::Lt => "lt",
            Self::Lte => "lte",
            Self::In => "in",
            Self::Nin => "nin",
            Self::All => "all",
            Self::Contains => "contains",
            Self::Prefix => "prefix",
            Self::Suffix => "suffix",
            Self::Type => "type",
            Self::Exists => "exists",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|op| op.as_str() == s)
    }

    fn is_ordered(self) -> bool {
        matches!(self, Self::Gt | Self::Gte | Self::Lt | Self::Lte)
    }

    fn sql(self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::Ne => "<>",
            Self::Gt => ">",
            Self::Gte => ">=",
            Self::Lt => "<",
            Self::Lte => "<=",
            _ => unreachable!("not a comparison"),
        }
    }
}

/// What a condition compares its field against.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// `eq` `ne` `gt` `gte` `lt` `lte` `contains` `prefix` `suffix`.
    Scalar(Scalar),
    /// `in` `nin` `all`: non-empty, one type, de-duplicated in order.
    List(Vec<Scalar>),
    Type(ValueType),
    Exists(bool),
}

/// One `{field, operator, value, caseInsensitive?}` node.
#[derive(Debug, Clone, PartialEq)]
pub struct Condition {
    pub field: String,
    pub op: Op,
    pub value: Operand,
    /// As given; folds ASCII letters on both sides when true.
    pub case_insensitive: Option<bool>,
}

impl Condition {
    fn folds(&self) -> bool {
        self.case_insensitive == Some(true)
    }
}

/// The recursive grammar: a condition, or `and`/`or`/`not` over predicates.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    Condition(Condition),
    And(Vec<Predicate>),
    Or(Vec<Predicate>),
    Not(Box<Predicate>),
}

/// A predicate over a document, whose fields are its metadata keys.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataFilter(pub Predicate);

/// A predicate over a metadata entry, whose fields are `key` and `value`.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataMatch(pub Predicate);

/// Which record a predicate is evaluated against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordType {
    Document,
    Entry,
}

/// A predicate failed validation at a JSON path such as `$.operands[1].value`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterError {
    pub path: String,
    /// The message without the `Invalid metadata ... at` prefix.
    pub detail: String,
    pub record: RecordType,
}

impl fmt::Display for FilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = match self.record {
            RecordType::Document => "filter",
            RecordType::Entry => "match",
        };
        write!(
            f,
            "Invalid metadata {what} at {}: {}",
            self.path, self.detail
        )
    }
}

impl std::error::Error for FilterError {}

impl MetadataFilter {
    /// Strictly validate an untrusted value: unknown operators and
    /// properties, operator-incompatible values and inputs past
    /// [`limits`] are rejected; membership lists are de-duplicated.
    ///
    /// # Errors
    /// [`FilterError`] naming the JSON path of the failing node.
    pub fn parse(input: &Value) -> Result<Self, FilterError> {
        let mut parser = Parser {
            nodes: 0,
            record: RecordType::Document,
        };
        parser.node(input, "$", 1).map(Self)
    }

    /// Parse a JSON string, as `--filter` receives it.
    ///
    /// # Errors
    /// The JSON parser's message, or a [`FilterError`]'s.
    pub fn parse_str(json: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
        Self::parse(&value).map_err(|e| e.to_string())
    }

    pub fn to_json(&self) -> Value {
        predicate_json(&self.0)
    }
}

impl MetadataMatch {
    /// [`MetadataFilter::parse`]'s grammar and limits, where a condition's
    /// `field` is `key` or `value`, and `exists` and `all`, which speak about
    /// a document's set of values, are rejected.
    ///
    /// # Errors
    /// [`FilterError`] naming the JSON path of the failing node.
    pub fn parse(input: &Value) -> Result<Self, FilterError> {
        let mut parser = Parser {
            nodes: 0,
            record: RecordType::Entry,
        };
        parser.node(input, "$", 1).map(Self)
    }

    pub fn to_json(&self) -> Value {
        predicate_json(&self.0)
    }
}

struct Parser {
    nodes: usize,
    record: RecordType,
}

impl Parser {
    fn err(&self, path: impl Into<String>, detail: impl Into<String>) -> FilterError {
        FilterError {
            path: path.into(),
            detail: detail.into(),
            record: self.record,
        }
    }

    fn node(&mut self, input: &Value, path: &str, depth: usize) -> Result<Predicate, FilterError> {
        if depth > MAX_DEPTH {
            return Err(self.err(
                path,
                format!("exceeds maximum nesting depth of {MAX_DEPTH}"),
            ));
        }
        self.nodes += 1;
        if self.nodes > MAX_NODES {
            return Err(self.err(path, format!("exceeds maximum of {MAX_NODES} nodes")));
        }
        let Value::Object(node) = input else {
            return Err(self.err(path, "each filter node must be an object"));
        };
        let Some(Value::String(operator)) = node.get("operator") else {
            return Err(self.err(path, "missing 'operator' property"));
        };
        match operator.as_str() {
            "and" | "or" => self.group(node, operator, path, depth),
            "not" => {
                self.reject_unknown(node, &["operator", "operand"], path)?;
                let Some(operand) = node.get("operand") else {
                    return Err(self.err(path, "'not' requires exactly one 'operand'"));
                };
                let inner = self.node(operand, &format!("{path}.operand"), depth + 1)?;
                Ok(Predicate::Not(Box::new(inner)))
            }
            other => match Op::parse(other) {
                Some(op) => self.condition(node, op, path).map(Predicate::Condition),
                None => {
                    let all: Vec<&str> = ["and", "or", "not"]
                        .into_iter()
                        .chain(Op::ALL.iter().map(|op| op.as_str()))
                        .collect();
                    Err(self.err(
                        path,
                        format!(
                            "unknown operator '{other}' — expected one of: {}",
                            all.join(", ")
                        ),
                    ))
                }
            },
        }
    }

    fn group(
        &mut self,
        node: &Map<String, Value>,
        operator: &str,
        path: &str,
        depth: usize,
    ) -> Result<Predicate, FilterError> {
        self.reject_unknown(node, &["operator", "operands"], path)?;
        let Some(Value::Array(operands)) = node.get("operands") else {
            return Err(self.err(path, format!("'{operator}' requires an 'operands' array")));
        };
        if operands.is_empty() {
            return Err(self.err(
                path,
                format!("'{operator}' requires a non-empty 'operands' array"),
            ));
        }
        if operands.len() > MAX_GROUP_OPERANDS {
            return Err(self.err(
                path,
                format!("'{operator}' exceeds maximum of {MAX_GROUP_OPERANDS} operands"),
            ));
        }
        let parsed = operands
            .iter()
            .enumerate()
            .map(|(i, operand)| self.node(operand, &format!("{path}.operands[{i}]"), depth + 1))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(if operator == "and" {
            Predicate::And(parsed)
        } else {
            Predicate::Or(parsed)
        })
    }

    fn condition(
        &self,
        node: &Map<String, Value>,
        op: Op,
        path: &str,
    ) -> Result<Condition, FilterError> {
        self.reject_unknown(
            node,
            &["field", "operator", "value", "caseInsensitive"],
            path,
        )?;
        let name = op.as_str();
        let field = match node.get("field") {
            Some(Value::String(f)) if !f.is_empty() => f.clone(),
            _ => {
                return Err(self.err(
                    path,
                    format!("'{name}' requires a non-empty string 'field'"),
                ));
            }
        };
        if field.len() > MAX_KEY_BYTES {
            return Err(self.err(path, format!("'field' exceeds {MAX_KEY_BYTES} bytes")));
        }
        if self.record == RecordType::Entry {
            if field != "key" && field != "value" {
                return Err(self.err(
                    path,
                    format!(
                        "'{field}' is not a field of a metadata entry, expected 'key' or 'value'"
                    ),
                ));
            }
            if matches!(op, Op::Exists | Op::All) {
                return Err(self.err(
                    path,
                    format!("'{name}' has no meaning for a single metadata entry"),
                ));
            }
        }
        let Some(value) = node.get("value") else {
            return Err(self.err(path, format!("'{name}' requires a 'value'")));
        };
        let case_insensitive = match node.get("caseInsensitive") {
            None => None,
            Some(Value::Bool(b)) => Some(*b),
            Some(_) => {
                return Err(self.err(
                    format!("{path}.caseInsensitive"),
                    "'caseInsensitive' must be a boolean",
                ));
            }
        };
        let value_path = format!("{path}.value");

        let operand = match op {
            Op::Exists => {
                let Value::Bool(present) = value else {
                    return Err(self.err(value_path, "'exists' requires a boolean value"));
                };
                Operand::Exists(*present)
            }
            Op::Type => {
                let Some(value_type) = value.as_str().and_then(ValueType::parse) else {
                    return Err(self.err(
                        value_path,
                        "'type' requires one of: string, number, boolean",
                    ));
                };
                Operand::Type(value_type)
            }
            Op::Contains | Op::Prefix | Op::Suffix => match self.scalar(value, &value_path)? {
                Scalar::String(text) if !text.is_empty() => Operand::Scalar(Scalar::String(text)),
                _ => {
                    return Err(self.err(
                        value_path,
                        format!("'{name}' requires a non-empty string value"),
                    ));
                }
            },
            Op::In | Op::Nin | Op::All => {
                Operand::List(self.membership(value, name, &value_path)?)
            }
            Op::Eq | Op::Ne | Op::Gt | Op::Gte | Op::Lt | Op::Lte => {
                let scalar = self.scalar(value, &value_path)?;
                if op.is_ordered() && matches!(scalar, Scalar::Boolean(_)) {
                    return Err(self.err(
                        value_path,
                        format!("'{name}' requires a string or number value"),
                    ));
                }
                Operand::Scalar(scalar)
            }
        };

        if case_insensitive.is_some() {
            let ci_path = format!("{path}.caseInsensitive");
            match &operand {
                Operand::Exists(_) | Operand::Type(_) => {
                    return Err(self.err(
                        ci_path,
                        format!("'caseInsensitive' does not apply to '{name}'"),
                    ));
                }
                Operand::Scalar(Scalar::String(_)) => {}
                Operand::List(items) if items.iter().all(|s| matches!(s, Scalar::String(_))) => {}
                _ => {
                    return Err(
                        self.err(ci_path, "'caseInsensitive' applies to string values only")
                    );
                }
            }
        }

        Ok(Condition {
            field,
            op,
            value: operand,
            case_insensitive,
        })
    }

    fn membership(
        &self,
        value: &Value,
        name: &str,
        path: &str,
    ) -> Result<Vec<Scalar>, FilterError> {
        let Value::Array(items) = value else {
            return Err(self.err(path, format!("'{name}' requires an array value")));
        };
        if items.is_empty() {
            return Err(self.err(path, format!("'{name}' requires a non-empty array value")));
        }
        if items.len() > MAX_MEMBERSHIP_VALUES {
            return Err(self.err(
                path,
                format!("'{name}' exceeds maximum of {MAX_MEMBERSHIP_VALUES} values"),
            ));
        }
        let scalars = items
            .iter()
            .enumerate()
            .map(|(i, item)| self.scalar(item, &format!("{path}[{i}]")))
            .collect::<Result<Vec<_>, _>>()?;
        let first = scalars[0].value_type();
        if scalars.iter().any(|s| s.value_type() != first) {
            return Err(self.err(
                path,
                format!("'{name}' requires a homogeneous array of one scalar type"),
            ));
        }
        Ok(super::extract::dedupe(scalars))
    }

    fn scalar(&self, value: &Value, path: &str) -> Result<Scalar, FilterError> {
        match value {
            Value::String(s) => {
                if js_length(s) > MAX_STRING_LENGTH {
                    return Err(self.err(
                        path,
                        format!("string exceeds {MAX_STRING_LENGTH} characters"),
                    ));
                }
                Ok(Scalar::String(s.clone()))
            }
            Value::Number(n) => match n.as_f64() {
                Some(f) if f.is_finite() => Ok(Scalar::Number(f)),
                _ => Err(self.err(path, "numbers must be finite")),
            },
            Value::Bool(b) => Ok(Scalar::Boolean(*b)),
            _ => Err(self.err(path, "expected a string, number, or boolean")),
        }
    }

    fn reject_unknown(
        &self,
        node: &Map<String, Value>,
        allowed: &[&str],
        path: &str,
    ) -> Result<(), FilterError> {
        match node.keys().find(|k| !allowed.contains(&k.as_str())) {
            Some(property) => Err(self.err(
                path,
                format!(
                    "unknown property '{property}' — allowed: {}",
                    allowed.join(", ")
                ),
            )),
            None => Ok(()),
        }
    }
}

fn predicate_json(predicate: &Predicate) -> Value {
    match predicate {
        Predicate::And(operands) | Predicate::Or(operands) => {
            let operator = if matches!(predicate, Predicate::And(_)) {
                "and"
            } else {
                "or"
            };
            serde_json::json!({
                "operator": operator,
                "operands": operands.iter().map(predicate_json).collect::<Vec<_>>(),
            })
        }
        Predicate::Not(operand) => {
            serde_json::json!({ "operator": "not", "operand": predicate_json(operand) })
        }
        Predicate::Condition(c) => {
            let value = match &c.value {
                Operand::Scalar(s) => s.to_json(),
                Operand::List(items) => Value::Array(items.iter().map(Scalar::to_json).collect()),
                Operand::Type(t) => Value::String(t.as_str().to_owned()),
                Operand::Exists(b) => Value::Bool(*b),
            };
            let mut node = serde_json::json!({
                "field": c.field,
                "operator": c.op.as_str(),
                "value": value,
            });
            if let Some(ci) = c.case_insensitive {
                node["caseInsensitive"] = Value::Bool(ci);
            }
            node
        }
    }
}

/// A parameterized SQL predicate and its bindings, in order.
#[derive(Debug, Clone, Default)]
pub struct CompiledSql {
    pub sql: String,
    pub params: Vec<SqlValue>,
}

/// The documents a compiled filter is applied to, which decides the SQL each
/// condition takes.
///
/// - `Candidates`: a few documents, as when search checks its results. Each
///   condition is a correlated `EXISTS` that seeks the document's own rows.
/// - `Corpus`: every document, as when discovery counts them. Each condition
///   is an uncorrelated `IN (SELECT document_id ...)` that SQLite evaluates
///   once per statement and probes per document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterScope {
    Candidates,
    Corpus,
}

/// How a `Candidates` subquery reaches one document's rows for a key. The
/// covering indexes are (key, <value>, document_id), so an unfolded equality
/// is one probe by value; any other condition seeks the document's own rows
/// through the primary key, its key term hidden from the planner by a unary
/// `+` so the plan does not depend on `sqlite_stat1`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RowSeek {
    ByValue,
    ByDocument,
}

fn equality_seek(case_insensitive: bool) -> RowSeek {
    if case_insensitive {
        RowSeek::ByDocument
    } else {
        RowSeek::ByValue
    }
}

/// The typed columns an operand is tested against.
struct Columns {
    value_type: String,
    text: String,
    number: String,
    boolean: String,
}

impl Columns {
    fn of_document() -> Self {
        Self {
            value_type: "mv.value_type".into(),
            text: "mv.text_value".into(),
            number: "mv.number_value".into(),
            boolean: "mv.boolean_value".into(),
        }
    }

    /// The column an operand compares against, folded when the condition
    /// ignores case.
    fn operand_column(&self, scalar: &Scalar, fold: bool) -> String {
        match scalar {
            Scalar::Number(_) => self.number.clone(),
            Scalar::Boolean(_) => self.boolean.clone(),
            Scalar::String(_) if fold => format!("lower({})", self.text),
            Scalar::String(_) => self.text.clone(),
        }
    }
}

impl MetadataFilter {
    /// One parameterized predicate over a documents-table `alias` (e.g. `d`).
    /// The caller restricts the surrounding query to active documents with a
    /// current, error-free extraction.
    pub fn compile(&self, alias: &str, scope: FilterScope) -> CompiledSql {
        let mut params = Vec::new();
        let sql = compile_filter(&self.0, alias, scope, &mut params);
        CompiledSql { sql, params }
    }
}

impl MetadataMatch {
    /// One parameterized predicate over a `document_metadata_values` row
    /// `alias` (e.g. `mv`). The caller scopes the rows to eligible documents.
    pub fn compile(&self, alias: &str) -> CompiledSql {
        let mut params = Vec::new();
        let sql = compile_match(&self.0, alias, &mut params);
        CompiledSql { sql, params }
    }
}

/// One condition over the document's rows for the key, whose parameter the
/// caller binds before `condition`'s.
fn condition_sql(
    alias: &str,
    scope: FilterScope,
    seek: RowSeek,
    condition: Option<&str>,
) -> String {
    if scope == FilterScope::Corpus {
        let where_sql = match condition {
            Some(c) => format!("mv.key = ? AND {c}"),
            None => "mv.key = ?".to_owned(),
        };
        return format!(
            "{alias}.id IN (SELECT mv.document_id FROM document_metadata_values mv WHERE {where_sql})"
        );
    }
    let key_term = if seek == RowSeek::ByValue {
        "mv.key = ?"
    } else {
        "+mv.key = ?"
    };
    let where_sql = match condition {
        Some(c) => format!("{key_term} AND {c}"),
        None => key_term.to_owned(),
    };
    format!(
        "EXISTS (SELECT 1 FROM document_metadata_values mv WHERE mv.document_id = {alias}.id AND {where_sql})"
    )
}

fn compile_filter(
    predicate: &Predicate,
    alias: &str,
    scope: FilterScope,
    params: &mut Vec<SqlValue>,
) -> String {
    let c = match predicate {
        Predicate::And(operands) | Predicate::Or(operands) => {
            let joiner = if matches!(predicate, Predicate::And(_)) {
                " AND "
            } else {
                " OR "
            };
            let parts: Vec<String> = operands
                .iter()
                .map(|o| compile_filter(o, alias, scope, params))
                .collect();
            return format!("({})", parts.join(joiner));
        }
        Predicate::Not(operand) => {
            return format!("NOT {}", compile_filter(operand, alias, scope, params));
        }
        Predicate::Condition(c) => c,
    };
    let columns = Columns::of_document();
    let fold = c.folds();
    let key = || SqlValue::Text(c.field.clone());

    match (&c.value, c.op) {
        (Operand::Exists(present), _) => {
            params.push(key());
            let sql = condition_sql(alias, scope, RowSeek::ByDocument, None);
            if *present { sql } else { format!("NOT {sql}") }
        }
        (Operand::Type(t), _) => {
            params.push(key());
            params.push(SqlValue::Text(t.as_str().to_owned()));
            condition_sql(
                alias,
                scope,
                RowSeek::ByDocument,
                Some(&format!("{} = ?", columns.value_type)),
            )
        }
        (Operand::Scalar(s), Op::Eq | Op::Gt | Op::Gte | Op::Lt | Op::Lte) => {
            let column = columns.operand_column(s, fold);
            params.push(key());
            params.push(bind(s, fold));
            let seek = if c.op == Op::Eq {
                equality_seek(fold)
            } else {
                RowSeek::ByDocument
            };
            condition_sql(
                alias,
                scope,
                seek,
                Some(&format!(
                    "{} = '{}' AND {column} {} ?",
                    columns.value_type,
                    s.value_type(),
                    c.op.sql()
                )),
            )
        }
        (Operand::Scalar(s), Op::Ne) => {
            let value_type = s.value_type();
            let column = columns.operand_column(s, fold);
            params.push(key());
            let present = condition_sql(
                alias,
                scope,
                RowSeek::ByDocument,
                Some(&format!("{} = '{value_type}'", columns.value_type)),
            );
            params.push(key());
            params.push(bind(s, fold));
            let equal = condition_sql(
                alias,
                scope,
                equality_seek(fold),
                Some(&format!(
                    "{} = '{value_type}' AND {column} = ?",
                    columns.value_type
                )),
            );
            format!("({present} AND NOT {equal})")
        }
        (Operand::List(items), Op::In | Op::Nin) => {
            let value_type = items[0].value_type();
            let column = columns.operand_column(&items[0], fold);
            let placeholders = vec!["?"; items.len()].join(", ");
            let member = format!(
                "{} = '{value_type}' AND {column} IN ({placeholders})",
                columns.value_type
            );
            if c.op == Op::In {
                params.push(key());
                params.extend(items.iter().map(|s| bind(s, fold)));
                return condition_sql(alias, scope, equality_seek(fold), Some(&member));
            }
            params.push(key());
            let present = condition_sql(
                alias,
                scope,
                RowSeek::ByDocument,
                Some(&format!("{} = '{value_type}'", columns.value_type)),
            );
            params.push(key());
            params.extend(items.iter().map(|s| bind(s, fold)));
            let member = condition_sql(alias, scope, equality_seek(fold), Some(&member));
            format!("({present} AND NOT {member})")
        }
        (Operand::List(items), _) => {
            let value_type = items[0].value_type();
            let column = columns.operand_column(&items[0], fold);
            let parts: Vec<String> = items
                .iter()
                .map(|s| {
                    params.push(key());
                    params.push(bind(s, fold));
                    condition_sql(
                        alias,
                        scope,
                        equality_seek(fold),
                        Some(&format!(
                            "{} = '{value_type}' AND {column} = ?",
                            columns.value_type
                        )),
                    )
                })
                .collect();
            format!("({})", parts.join(" AND "))
        }
        (Operand::Scalar(s), _) => {
            params.push(key());
            let column = columns.operand_column(s, fold);
            let Scalar::String(text) = s else {
                unreachable!("text operands are strings")
            };
            let test = text_test_sql(c.op, text, &column, fold, params);
            condition_sql(
                alias,
                scope,
                RowSeek::ByDocument,
                Some(&format!("{} = 'string' AND {test}", columns.value_type)),
            )
        }
    }
}

fn compile_match(predicate: &Predicate, alias: &str, params: &mut Vec<SqlValue>) -> String {
    let c = match predicate {
        Predicate::And(operands) | Predicate::Or(operands) => {
            let joiner = if matches!(predicate, Predicate::And(_)) {
                " AND "
            } else {
                " OR "
            };
            let parts: Vec<String> = operands
                .iter()
                .map(|o| compile_match(o, alias, params))
                .collect();
            return format!("({})", parts.join(joiner));
        }
        Predicate::Not(operand) => return format!("NOT {}", compile_match(operand, alias, params)),
        Predicate::Condition(c) => c,
    };
    let columns = if c.field == "key" {
        Columns {
            value_type: "'string'".into(),
            text: format!("{alias}.key"),
            number: "NULL".into(),
            boolean: "NULL".into(),
        }
    } else {
        Columns {
            value_type: format!("{alias}.value_type"),
            text: format!("{alias}.text_value"),
            number: format!("{alias}.number_value"),
            boolean: format!("{alias}.boolean_value"),
        }
    };
    let fold = c.folds();
    match &c.value {
        Operand::Type(t) => {
            params.push(SqlValue::Text(t.as_str().to_owned()));
            format!("{} = ?", columns.value_type)
        }
        Operand::Scalar(s) if matches!(c.op, Op::Contains | Op::Prefix | Op::Suffix) => {
            let column = columns.operand_column(s, fold);
            let Scalar::String(text) = s else {
                unreachable!("text operands are strings")
            };
            let test = text_test_sql(c.op, text, &column, fold, params);
            format!("({} = 'string' AND {test})", columns.value_type)
        }
        Operand::Scalar(s) => {
            let column = columns.operand_column(s, fold);
            params.push(bind(s, fold));
            format!(
                "({} = '{}' AND {column} {} ?)",
                columns.value_type,
                s.value_type(),
                c.op.sql()
            )
        }
        Operand::List(items) => {
            let column = columns.operand_column(&items[0], fold);
            let placeholders = vec!["?"; items.len()].join(", ");
            let membership = if c.op == Op::In { "IN" } else { "NOT IN" };
            params.extend(items.iter().map(|s| bind(s, fold)));
            format!(
                "({} = '{}' AND {column} {membership} ({placeholders}))",
                columns.value_type,
                items[0].value_type()
            )
        }
        Operand::Exists(_) => unreachable!("a match has no 'exists'"),
    }
}

/// Substring tests over a string column. Prefix and suffix compare UTF-8
/// bytes with the operand's byte length bound from Rust: SQLite's text
/// `length()` and `substr()` stop at an embedded NUL, blobs do not. UTF-8 is
/// self-synchronizing, so a byte prefix of a whole operand is a character
/// prefix. `COALESCE` keeps each test boolean (a substring of an empty blob
/// is NULL), so a match and its negation partition the rows.
fn text_test_sql(
    op: Op,
    text: &str,
    column: &str,
    fold: bool,
    params: &mut Vec<SqlValue>,
) -> String {
    let operand = if fold {
        text.to_ascii_lowercase()
    } else {
        text.to_owned()
    };
    if op == Op::Contains {
        params.push(SqlValue::Text(operand));
        return format!("instr({column}, ?) > 0");
    }
    params.push(SqlValue::Integer(
        i64::try_from(operand.len()).expect("bounded operand"),
    ));
    params.push(SqlValue::Text(operand));
    if op == Op::Prefix {
        format!("COALESCE(substr(CAST({column} AS BLOB), 1, ?) = CAST(? AS BLOB), 0)")
    } else {
        format!("COALESCE(substr(CAST({column} AS BLOB), -?) = CAST(? AS BLOB), 0)")
    }
}

/// Booleans bind as 0/1; folded strings bind folded as SQLite's `lower()`
/// folds the column, over ASCII letters only.
fn bind(scalar: &Scalar, fold: bool) -> SqlValue {
    match scalar {
        Scalar::Boolean(b) => SqlValue::Integer(i64::from(*b)),
        Scalar::Number(n) => SqlValue::Real(*n),
        Scalar::String(s) if fold => SqlValue::Text(s.to_ascii_lowercase()),
        Scalar::String(s) => SqlValue::Text(s.clone()),
    }
}
