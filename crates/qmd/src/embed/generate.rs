//! Embedding generation driver, ported from TS `generateEmbeddings`.
//!
//! Simpler than the original: no LLM session timeout and no >80% error-rate
//! abort; the per-chunk fallback and bounded retry of failed chunks are kept.

use std::collections::HashMap;
use std::time::Instant;

use crate::chunk::chunk_document_by_tokens_for;
use crate::embed::{EmbeddingBackend, embedding_fingerprint_for, format_doc_for_embedding};
use crate::error::Result;
use crate::store::{Store, extract_title, now_iso};

/// Documents per outer batch (chunking + body fetch granularity).
pub const DEFAULT_EMBED_MAX_DOCS_PER_BATCH: usize = 64;
/// Byte budget per outer batch.
pub const DEFAULT_EMBED_MAX_BATCH_BYTES: u64 = 64 * 1024 * 1024;
/// Chunks per embedding call.
const BATCH_SIZE: usize = 32;
/// Give up on a chunk after this many attempts.
const MAX_RETRY_ATTEMPTS: u32 = 3;

/// Options for [`generate_embeddings`].
#[derive(Default)]
pub struct EmbedOptions<'a> {
    /// Model URI override (defaults to the embedder's model).
    pub model: Option<String>,
    /// Clear existing embeddings first.
    pub force: bool,
    /// Restrict to one collection.
    pub collection: Option<String>,
    pub max_docs_per_batch: Option<usize>,
    pub max_batch_bytes: Option<u64>,
    /// Progress callback, invoked after every embedding sub-batch.
    #[allow(clippy::type_complexity)]
    pub on_progress: Option<Box<dyn Fn(&EmbedProgress) + 'a>>,
}

/// Progress snapshot for [`EmbedOptions::on_progress`].
#[derive(Debug, Clone)]
pub struct EmbedProgress {
    pub chunks_embedded: usize,
    pub total_chunks: usize,
    pub docs_total: usize,
    pub errors: usize,
}

/// A chunk that could not be embedded.
#[derive(Debug, Clone)]
pub struct EmbedFailure {
    pub path: String,
    pub hash: String,
    pub seq: usize,
    pub attempts: u32,
    pub reason: String,
}

/// Outcome of an embedding run.
#[derive(Debug, Clone, Default)]
pub struct EmbedResult {
    pub docs_processed: usize,
    pub chunks_embedded: usize,
    pub errors: usize,
    pub failures: Vec<EmbedFailure>,
    pub duration_ms: u128,
}

struct ChunkItem {
    hash: String,
    path: String,
    title: String,
    text: String,
    seq: usize,
    pos: usize,
    expected_total_chunks: usize,
    fingerprint: String,
}

/// Generate vector embeddings for all documents that need them.
///
/// # Errors
/// Fails on store errors or unrecoverable backend failures (per-chunk
/// failures are recorded in the result instead).
pub fn generate_embeddings(
    store: &Store,
    backend: &dyn EmbeddingBackend,
    options: &EmbedOptions<'_>,
) -> Result<EmbedResult> {
    let model = options
        .model
        .clone()
        .unwrap_or_else(|| backend.model_uri().to_owned());
    let now = now_iso();
    let started = Instant::now();
    let max_docs = options
        .max_docs_per_batch
        .unwrap_or(DEFAULT_EMBED_MAX_DOCS_PER_BATCH)
        .max(1);
    let max_bytes = options
        .max_batch_bytes
        .unwrap_or(DEFAULT_EMBED_MAX_BATCH_BYTES)
        .max(1);

    if options.force {
        store.clear_all_embeddings(options.collection.as_deref())?;
    }

    let docs_to_embed = store.pending_embedding_docs(options.collection.as_deref(), &model)?;
    if docs_to_embed.is_empty() {
        return Ok(EmbedResult::default());
    }
    let docs_total = docs_to_embed.len();

    let mut batches: Vec<Vec<crate::store::PendingEmbeddingDoc>> = Vec::new();
    let mut current: Vec<crate::store::PendingEmbeddingDoc> = Vec::new();
    let mut current_bytes = 0u64;
    for doc in docs_to_embed {
        let doc_bytes = doc.bytes;
        if current.len() >= max_docs
            || (!current.is_empty() && current_bytes + doc_bytes > max_bytes)
        {
            batches.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
        current_bytes += doc_bytes;
        current.push(doc);
    }
    if !current.is_empty() {
        batches.push(current);
    }

    let mut chunks_embedded = 0usize;
    let mut total_chunks = 0usize;
    let mut vector_table_initialized = false;
    let mut failures: HashMap<(String, usize), EmbedFailure> = HashMap::new();

    for batch in batches {
        let hashes: Vec<String> = batch.iter().map(|d| d.hash.clone()).collect();
        let bodies = store.content_bodies(&hashes)?;

        let mut batch_chunks: Vec<ChunkItem> = Vec::new();
        let mut expected_chunks_by_hash: HashMap<String, usize> = HashMap::new();
        for doc in &batch {
            let Some(body) = bodies.get(&doc.hash) else {
                continue;
            };
            if body.trim().is_empty() {
                continue;
            }
            let title = extract_title(body, &doc.path);
            let chunks = chunk_document_by_tokens_for(body, backend, &doc.chunk)?;
            let fingerprint = embedding_fingerprint_for(&model, &doc.chunk);
            store.clear_hash_embeddings(&doc.hash)?;
            let expected = chunks.len();
            for (seq, chunk) in chunks.into_iter().enumerate() {
                batch_chunks.push(ChunkItem {
                    hash: doc.hash.clone(),
                    path: doc.path.clone(),
                    title: title.clone(),
                    text: chunk.text,
                    seq,
                    pos: chunk.pos,
                    expected_total_chunks: expected,
                    fingerprint: fingerprint.clone(),
                });
            }
            expected_chunks_by_hash.insert(doc.hash.clone(), expected);
        }
        total_chunks += batch_chunks.len();
        if batch_chunks.is_empty() {
            continue;
        }

        if !vector_table_initialized {
            let first = &batch_chunks[0];
            let text = format_doc_for_embedding(&first.text, Some(&first.title), &model);
            let probe = backend.embed(&text)?.ok_or_else(|| {
                crate::error::Error::Other(
                    "failed to get embedding dimensions from first chunk".to_owned(),
                )
            })?;
            store.ensure_vec_table(probe.len())?;
            vector_table_initialized = true;
        }

        let insert_chunk = |chunk: &ChunkItem,
                            embedding: &[f32],
                            chunks_embedded: &mut usize,
                            failures: &mut HashMap<(String, usize), EmbedFailure>|
         -> Result<()> {
            #[allow(clippy::cast_possible_wrap)]
            store.insert_embedding(
                &chunk.hash,
                chunk.seq as i64,
                chunk.pos as i64,
                embedding,
                &model,
                &now,
                chunk.expected_total_chunks as i64,
                &chunk.fingerprint,
            )?;
            *chunks_embedded += 1;
            failures.remove(&(chunk.hash.clone(), chunk.seq));
            Ok(())
        };
        let record_failure = |failures: &mut HashMap<(String, usize), EmbedFailure>,
                              chunk: &ChunkItem,
                              reason: &str| {
            let entry = failures
                .entry((chunk.hash.clone(), chunk.seq))
                .or_insert_with(|| EmbedFailure {
                    path: chunk.path.clone(),
                    hash: chunk.hash.clone(),
                    seq: chunk.seq,
                    attempts: 0,
                    reason: String::new(),
                });
            entry.attempts += 1;
            entry.reason = reason.chars().take(180).collect();
        };

        for chunk_batch in batch_chunks.chunks(BATCH_SIZE) {
            let texts: Vec<String> = chunk_batch
                .iter()
                .map(|c| format_doc_for_embedding(&c.text, Some(&c.title), &model))
                .collect();

            match backend.embed_batch(&texts) {
                Ok(embeddings) => {
                    for (chunk, embedding) in chunk_batch.iter().zip(embeddings) {
                        match embedding {
                            Some(v) => {
                                insert_chunk(chunk, &v, &mut chunks_embedded, &mut failures)?
                            }
                            None => record_failure(
                                &mut failures,
                                chunk,
                                "batch embedding returned no vector",
                            ),
                        }
                    }
                }
                Err(batch_err) => {
                    for chunk in chunk_batch {
                        let text =
                            format_doc_for_embedding(&chunk.text, Some(&chunk.title), &model);
                        match backend.embed(&text) {
                            Ok(Some(v)) => {
                                insert_chunk(chunk, &v, &mut chunks_embedded, &mut failures)?
                            }
                            Ok(None) => record_failure(
                                &mut failures,
                                chunk,
                                &format!(
                                    "batch failed: {batch_err}; single embed returned no vector"
                                ),
                            ),
                            Err(e) => record_failure(&mut failures, chunk, &e.to_string()),
                        }
                    }
                }
            }

            if let Some(on_progress) = &options.on_progress {
                on_progress(&EmbedProgress {
                    chunks_embedded,
                    total_chunks,
                    docs_total,
                    errors: failures.len(),
                });
            }
        }

        let mut retried = true;
        while retried {
            retried = false;
            let retry_keys: Vec<(String, usize)> = failures
                .iter()
                .filter(|(_, f)| f.attempts < MAX_RETRY_ATTEMPTS)
                .map(|(k, _)| k.clone())
                .collect();
            for key in retry_keys {
                let Some(chunk) = batch_chunks
                    .iter()
                    .find(|c| c.hash == key.0 && c.seq == key.1)
                else {
                    continue;
                };
                retried = true;
                let text = format_doc_for_embedding(&chunk.text, Some(&chunk.title), &model);
                match backend.embed(&text) {
                    Ok(Some(v)) => insert_chunk(chunk, &v, &mut chunks_embedded, &mut failures)?,
                    Ok(None) => {
                        record_failure(&mut failures, chunk, "embedding returned no vector")
                    }
                    Err(e) => record_failure(&mut failures, chunk, &e.to_string()),
                }
            }
        }

        let removed = store.remove_incomplete_embeddings(&expected_chunks_by_hash, &model)?;
        chunks_embedded = chunks_embedded.saturating_sub(removed);
    }

    let mut failure_list: Vec<EmbedFailure> = failures.into_values().collect();
    failure_list.sort_by(|a, b| (&a.path, a.seq).cmp(&(&b.path, b.seq)));
    Ok(EmbedResult {
        docs_processed: docs_total,
        chunks_embedded,
        errors: failure_list.len(),
        failures: failure_list,
        duration_ms: started.elapsed().as_millis(),
    })
}
