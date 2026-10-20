//! Library error type.

/// Errors surfaced by the qmd library.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("config error: {0}")]
    Config(String),

    /// Default DB path requested outside production mode (test safety rail,
    /// ported from the TS implementation).
    #[error(
        "database path not set: tests must set INDEX_PATH or open a store with an explicit path"
    )]
    DbPathNotSet,

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
