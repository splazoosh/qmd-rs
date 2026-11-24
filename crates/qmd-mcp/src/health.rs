//! Whether the embedder works, as a long-lived server last saw it.
//!
//! A server can lose the GPU under a loaded model and go on answering
//! everything that does not embed (a driver replaced under a running daemon
//! does this), so the state lives outside the index thread, where `/health`
//! reads it without queueing, and [`Watched`] records it on every embedding.
//!
//! Recovery is not decided here. The embedder a server is given keeps its model
//! in a host process and replaces a host that fails (`qmd::host`), so a failure
//! recorded here is one a new host did not cure; the next query tries again.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use qmd::chunk::Tokenizer;
use qmd::embed::{Embedder, EmbeddingBackend};

/// What this process knows about its embedder. Shared with the index thread.
#[derive(Debug)]
pub struct EmbedderHealth {
    state: Mutex<State>,
}

#[derive(Debug)]
struct State {
    /// Whether this process has embedded anything.
    embedded: bool,
    /// Whether a model is in memory.
    models_loaded: bool,
    /// The last embedding failed with this; cleared by a success.
    failure: Option<String>,
}

impl EmbedderHealth {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(State {
                embedded: false,
                models_loaded: false,
                failure: None,
            }),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `idle` before the first embedding, then `ok` or `failed`.
    pub fn label(&self) -> &'static str {
        let state = self.state();
        match (&state.failure, state.embedded) {
            (Some(_), _) => "failed",
            (None, true) => "ok",
            (None, false) => "idle",
        }
    }

    /// The failure and what to do about it, while the embedder is failing.
    pub fn failure(&self) -> Option<String> {
        self.state().failure.clone()
    }

    /// Whether a model is in memory now: false before the first use, after an
    /// idle unload, and after a job that left none.
    pub fn models_loaded(&self) -> bool {
        self.state().models_loaded
    }

    pub(crate) fn set_models_loaded(&self, loaded: bool) {
        self.state().models_loaded = loaded;
    }

    fn succeeded(&self) {
        let mut state = self.state();
        state.embedded = true;
        state.failure = None;
    }

    /// Record `error`, log it, and return the text a caller should read.
    pub(crate) fn failed(&self, error: &str) -> String {
        let message = format!(
            "embedding failed: {error}. Vector search is unavailable until this clears; the \
             next query tries again. Lexical searches still work"
        );
        eprintln!("qmd: {message}");
        self.state().failure = Some(message.clone());
        message
    }
}

/// An embedding backend that reports each result to [`EmbedderHealth`].
pub(crate) struct Watched {
    pub(crate) inner: Box<dyn EmbeddingBackend>,
    pub(crate) health: Arc<EmbedderHealth>,
}

impl Embedder for Watched {
    fn model_uri(&self) -> &str {
        self.inner.model_uri()
    }

    fn embed_batch(&self, texts: &[String]) -> qmd::Result<Vec<Option<Vec<f32>>>> {
        match self.inner.embed_batch(texts) {
            Ok(vectors) => {
                if vectors.iter().any(Option::is_some) {
                    self.health.succeeded();
                } else if !texts.is_empty() {
                    self.health.failed("no text in the batch produced a vector");
                }
                Ok(vectors)
            }
            Err(e) => Err(qmd::Error::Other(self.health.failed(&e.to_string()))),
        }
    }
}

impl Tokenizer for Watched {
    fn count_tokens(&self, text: &str) -> qmd::Result<usize> {
        self.inner.count_tokens(text)
    }

    fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> qmd::Result<String> {
        self.inner.truncate_to_tokens(text, max_tokens)
    }
}
