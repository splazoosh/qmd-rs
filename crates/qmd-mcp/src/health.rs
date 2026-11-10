//! Whether the embedder still works, as a long-lived server last saw it.
//!
//! A process can lose the GPU under a loaded model and go on answering
//! everything that does not embed (a driver replaced under a running
//! daemon does this), so the state lives outside the index thread, where
//! `/health` reads it without queueing, and [`Watched`] records it on every
//! embedding.
//!
//! The only recovery measured to work is a new process. Freeing the model in
//! place is not attempted: ggml aborts when `cudaFree` fails. A failure in a
//! process that has never embedded is left to the caller, since a new process
//! would be in the same position; a bad load is retried by the next query.

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
    can_relaunch: bool,
    /// The last embedding failed with this; cleared by a success.
    failure: Option<String>,
}

impl EmbedderHealth {
    pub(crate) fn new(can_relaunch: bool) -> Self {
        Self {
            state: Mutex::new(State {
                embedded: false,
                can_relaunch,
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

    fn succeeded(&self) {
        let mut state = self.state();
        state.embedded = true;
        state.failure = None;
    }

    /// Record `error`, log it, and return the text a caller should read.
    pub(crate) fn failed(&self, error: &str) -> String {
        let mut state = self.state();
        let advice = if state.embedded && state.can_relaunch {
            "This process can no longer embed, so the daemon is restarting in a new one: \
             retry in a few seconds"
        } else {
            "Vector search is unavailable until this clears or the server is restarted \
             ('qmd mcp stop', then start it; a stdio server restarts with its client)"
        };
        let message = format!("embedding failed: {error}. {advice}. Lexical searches still work");
        eprintln!("qmd: {message}");
        state.failure = Some(message.clone());
        message
    }

    /// Whether the index thread should hand over to a new process.
    pub(crate) fn relaunch_due(&self) -> bool {
        let state = self.state();
        state.can_relaunch && state.embedded && state.failure.is_some()
    }

    /// The new process could not be started: stay up and say so.
    pub(crate) fn relaunch_failed(&self, error: &anyhow::Error) {
        let mut state = self.state();
        state.can_relaunch = false;
        let message = format!(
            "embedding failed and the daemon could not restart itself ({error:#}). \
             Restart it: 'qmd mcp stop', then start it. Lexical searches still work"
        );
        eprintln!("qmd: {message}");
        state.failure = Some(message);
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
