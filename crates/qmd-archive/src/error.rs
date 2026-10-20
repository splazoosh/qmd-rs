//! Errors for the archive layer.

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{0}")]
    Plain(#[from] std::io::Error),

    #[error("not a usable http(s) url: {0}")]
    Url(String),

    #[error("malformed manifest at {path}: {message}")]
    Manifest { path: PathBuf, message: String },

    #[error("nothing archived for {0}")]
    NotArchived(String),

    /// The fetch was refused before it happened, or the response said stop.
    /// Never retried: the caller records it and exits.
    #[error("{0}")]
    Refused(#[from] Refusal),

    #[error("{0}")]
    Other(String),
}

/// A read this tool declined to make, or stopped at. Each variant maps to one
/// of the access rules in [`crate::fetch`]; every one of them is terminal for the run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Refusal {
    /// robots.txt disallows this path for our user-agent (rule 2 read as binding).
    #[error("robots.txt disallows {path} for {agent}")]
    Robots { path: String, agent: String },

    /// Rule 5: the per-host gate has not elapsed. Not slept through — refused.
    #[error("rate gate: {host} was read {elapsed_secs}s ago, minimum is {minimum_secs}s")]
    RateGate {
        host: String,
        elapsed_secs: u64,
        minimum_secs: u64,
    },

    /// Rule 5: a 429 or 503 named a time; we do not read that host before it.
    #[error("host {host} asked for {retry_after_secs}s; {remaining_secs}s remain")]
    RetryAfter {
        host: String,
        retry_after_secs: u64,
        remaining_secs: u64,
    },

    /// Rule 5: a 429 is the one response where persistence is the harm.
    #[error("HTTP {status} from {host}: rate-limited, not retried")]
    RateLimited { host: String, status: u16 },

    /// Rule 4: a CAPTCHA or interactive challenge is asking for a person.
    #[error("human gate at {url}: {evidence}")]
    HumanGate { url: String, evidence: String },

    /// A redirect this tool will not follow: too many hops, or a target that is
    /// not an http(s) URL.
    #[error("redirect from {url} to {target} not followed: {reason}")]
    Redirect {
        url: String,
        target: String,
        reason: String,
    },

    #[error("HTTP {status} for {url}")]
    Status { url: String, status: u16 },

    #[error("transport error for {url}: {message}")]
    Transport { url: String, message: String },
}
