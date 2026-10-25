//! Embedding backends and prompt formats, ported from TS `llm.ts`.
//!
//! The [`Embedder`] trait decouples the pipeline from llama.cpp so tests run
//! with a deterministic mock and no model downloads (the TS `CI=true` mode).

mod generate;
#[cfg(feature = "llama")]
mod llama;

pub use generate::{EmbedFailure, EmbedOptions, EmbedProgress, EmbedResult, generate_embeddings};
#[cfg(feature = "llama")]
pub(crate) use llama::backend as llama_backend;
#[cfg(feature = "llama")]
pub use llama::{LlamaEmbedder, gpu_offload};

use sha2::{Digest, Sha256};

use crate::chunk::{ChunkBoundary, ChunkParams, Tokenizer};
use crate::error::Result;

const EMBED_FINGERPRINT_PROBE_QUERY: &str = "__qmd_embedding_query_probe__";
const EMBED_FINGERPRINT_PROBE_TITLE: &str = "__qmd_embedding_title_probe__";
const EMBED_FINGERPRINT_PROBE_DOC: &str = "__qmd_embedding_document_probe__";

/// Whether the model URI names a Qwen3-Embedding model (different templates).
pub fn is_qwen3_embedding_model(model_uri: &str) -> bool {
    let lower = model_uri.to_lowercase();
    lower.contains("qwen") && lower.contains("embed")
}

/// Format a query for embedding (template is part of the fingerprint and must
/// match the TS implementation byte-for-byte).
pub fn format_query_for_embedding(query: &str, model_uri: &str) -> String {
    if is_qwen3_embedding_model(model_uri) {
        return format!(
            "Instruct: Retrieve relevant documents for the given query\nQuery: {query}"
        );
    }
    format!("task: search result | query: {query}")
}

/// Format a document (chunk) for embedding.
pub fn format_doc_for_embedding(text: &str, title: Option<&str>, model_uri: &str) -> String {
    if is_qwen3_embedding_model(model_uri) {
        return match title {
            Some(t) if !t.is_empty() => format!("{t}\n{text}"),
            _ => text.to_owned(),
        };
    }
    let title = match title {
        Some(t) if !t.is_empty() => t,
        _ => "none",
    };
    format!("title: {title} | text: {text}")
}

/// Embedding fingerprint: first 6 hex chars of SHA-256 over the significant
/// inputs (model URI, prompt templates, chunking constants). A change
/// invalidates stored vectors.
pub fn embedding_fingerprint(model: &str) -> String {
    embedding_fingerprint_for(model, &ChunkParams::default())
}

/// [`embedding_fingerprint`] for vectors chunked with `chunk`. The window is
/// derived from the size, so size, overlap and boundary fix the chunking. The
/// default boundary adds no line, so default vectors keep the TS fingerprint.
pub fn embedding_fingerprint_for(model: &str, chunk: &ChunkParams) -> String {
    let mut significant = vec![
        format!("model:{model}"),
        format!(
            "query:{}",
            format_query_for_embedding(EMBED_FINGERPRINT_PROBE_QUERY, model)
        ),
        format!(
            "doc:{}",
            format_doc_for_embedding(
                EMBED_FINGERPRINT_PROBE_DOC,
                Some(EMBED_FINGERPRINT_PROBE_TITLE),
                model
            )
        ),
        format!("chunk_tokens:{}", chunk.size_tokens),
        format!("chunk_overlap_tokens:{}", chunk.overlap_tokens),
    ];
    if chunk.boundary != ChunkBoundary::Window {
        significant.push(format!("chunk_boundary:{}", chunk.boundary.as_str()));
    }
    let digest = Sha256::digest(significant.join("\n").as_bytes());
    hex::encode(digest)[..6].to_owned()
}

/// An embedding backend.
pub trait Embedder {
    /// Model URI this embedder serves (used for fingerprints and metadata).
    fn model_uri(&self) -> &str;

    /// Embed a batch of already-formatted texts. A `None` entry means that
    /// text failed to embed (mirrors the TS per-text null results).
    ///
    /// # Errors
    /// Returns an error only for whole-batch failures.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Option<Vec<f32>>>>;

    /// Embed a single formatted text.
    ///
    /// # Errors
    /// Propagates backend failures.
    fn embed(&self, text: &str) -> Result<Option<Vec<f32>>> {
        Ok(self.embed_batch(&[text.to_owned()])?.pop().flatten())
    }
}

/// Full backend surface needed by [`generate_embeddings`]: embedding plus the
/// model tokenizer for token-accurate chunking.
pub trait EmbeddingBackend: Embedder + Tokenizer {}

impl<T: Embedder + Tokenizer> EmbeddingBackend for T {}

/// Deterministic mock embedder for tests: unit vectors derived from the
/// SHA-256 of the text (identical text embeds identically; distinct texts are
/// near-orthogonal), with a ~4 chars/token tokenizer.
pub struct MockEmbedder {
    model_uri: String,
    dimensions: usize,
}

impl Default for MockEmbedder {
    fn default() -> Self {
        Self::new("mock:test-embedder", 64)
    }
}

impl MockEmbedder {
    pub fn new(model_uri: impl Into<String>, dimensions: usize) -> Self {
        Self {
            model_uri: model_uri.into(),
            dimensions,
        }
    }

    fn vector_for(&self, text: &str) -> Vec<f32> {
        let mut values = Vec::with_capacity(self.dimensions);
        let mut counter = 0u32;
        while values.len() < self.dimensions {
            let mut hasher = Sha256::new();
            hasher.update(text.as_bytes());
            hasher.update(counter.to_le_bytes());
            let digest = hasher.finalize();
            for byte in digest {
                if values.len() == self.dimensions {
                    break;
                }
                values.push(f32::from(byte) / 127.5 - 1.0);
            }
            counter += 1;
        }
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut values {
                *v /= norm;
            }
        }
        values
    }
}

impl Embedder for MockEmbedder {
    fn model_uri(&self) -> &str {
        &self.model_uri
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Option<Vec<f32>>>> {
        Ok(texts.iter().map(|t| Some(self.vector_for(t))).collect())
    }
}

impl Tokenizer for MockEmbedder {
    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(text.chars().count().div_ceil(4))
    }

    fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> Result<String> {
        Ok(text.chars().take(max_tokens * 4).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::DEFAULT_EMBED_MODEL;

    #[test]
    fn query_and_doc_formats_match_ts_templates() {
        assert_eq!(
            format_query_for_embedding("fox jumping", DEFAULT_EMBED_MODEL),
            "task: search result | query: fox jumping"
        );
        assert_eq!(
            format_doc_for_embedding("body text", Some("My Title"), DEFAULT_EMBED_MODEL),
            "title: My Title | text: body text"
        );
        assert_eq!(
            format_doc_for_embedding("body text", None, DEFAULT_EMBED_MODEL),
            "title: none | text: body text"
        );
    }

    #[test]
    fn qwen3_models_use_instruct_templates() {
        let uri = "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/model.gguf";
        assert!(is_qwen3_embedding_model(uri));
        assert!(!is_qwen3_embedding_model(DEFAULT_EMBED_MODEL));
        assert!(format_query_for_embedding("q", uri).starts_with("Instruct: Retrieve relevant"));
        assert_eq!(format_doc_for_embedding("t", Some("T"), uri), "T\nt");
        assert_eq!(format_doc_for_embedding("t", None, uri), "t");
    }

    #[test]
    fn fingerprint_matches_ts_value_for_default_model() {
        assert_eq!(embedding_fingerprint(DEFAULT_EMBED_MODEL), "c37385");
        assert_ne!(
            embedding_fingerprint("hf:other/model/file.gguf"),
            embedding_fingerprint(DEFAULT_EMBED_MODEL)
        );
        let heading = ChunkParams::default().with_boundary(ChunkBoundary::Heading);
        assert_ne!(
            embedding_fingerprint_for(DEFAULT_EMBED_MODEL, &heading),
            embedding_fingerprint(DEFAULT_EMBED_MODEL)
        );
    }

    #[test]
    fn mock_embedder_is_deterministic_and_normalized() {
        let mock = MockEmbedder::default();
        let a = mock.embed("hello").unwrap().unwrap();
        let b = mock.embed("hello").unwrap().unwrap();
        let c = mock.embed("goodbye").unwrap().unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 64);
        let norm: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        let dot: f32 = a.iter().zip(&c).map(|(x, y)| x * y).sum();
        assert!(dot.abs() < 0.5, "distinct texts should be near-orthogonal");
    }
}
