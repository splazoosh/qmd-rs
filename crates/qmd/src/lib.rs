//! Core library for qmd: on-device hybrid search over markdown/text
//! knowledge bases (SQLite FTS5 + sqlite-vec + local GGUF embeddings).
//!
//! Surface: config parsing, store (schema/docs/FTS/vectors), indexing,
//! chunking, embeddings, snippets, and the hybrid query pipeline (structured
//! query parsing + RRF fusion). LLM query expansion and reranking are not
//! implemented; [`query`] holds their seams.

pub mod bench;
pub mod chunk;
pub mod config;
pub mod embed;
pub mod error;
pub mod index;
pub mod models;
pub mod query;
pub mod querylog;
pub mod snippet;
pub mod store;

pub use error::{Error, Result};
pub use store::Store;
