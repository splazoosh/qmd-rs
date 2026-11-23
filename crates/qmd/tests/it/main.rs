//! The integration tests, one module per area, linked as one binary.

mod common;

mod embed_mock;
mod eval_bm25;
mod eval_hybrid;
mod eval_vector;
mod hf_live;
mod metadata_discovery;
mod metadata_filter;
mod metadata_inherit;
mod query_mock;
mod rerank_live;
