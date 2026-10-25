//! The thread that owns the index.
//!
//! [`ServerHandler`](rmcp::ServerHandler) requires `Send + Sync`, and neither
//! `rusqlite::Connection` (`Send`, not `Sync`) nor `LlamaEmbedder` (neither) can
//! satisfy it. A dedicated thread owns both and takes closures over an `mpsc`
//! channel; the handler holds only a [`Worker`], which is `Send + Sync`. Nothing
//! in the synchronous library is coloured async, and serializing index access
//! keeps one embedder however many clients are connected.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context as _, Result};
use qmd::config::{self, CollectionConfig};
use qmd::embed::{Embedder, EmbeddingBackend};
use qmd::rerank::Reranker;
use qmd::store::Store;
use tokio::sync::{mpsc, oneshot};

/// Loads the embedding backend for a model URI. The CLI supplies the real
/// llama-backed one; tests supply a mock. Keeping it a callback keeps the
/// `llama` feature (and its native build) off this crate entirely.
pub type EmbedderFactory = Box<dyn Fn(&str) -> Result<Box<dyn EmbeddingBackend>> + Send + 'static>;

/// Loads the reranker for a model URI, on first use.
pub type RerankerFactory = Box<dyn Fn(&str) -> Result<Box<dyn Reranker>> + Send + 'static>;

/// The model loaders the index thread calls lazily.
pub struct Factories {
    pub embedder: EmbedderFactory,
    pub reranker: RerankerFactory,
}

/// The index, plus a lazily loaded embedder. Lives on the worker thread.
pub struct Index {
    pub store: Store,
    pub config: CollectionConfig,
    pub config_path: PathBuf,
    pub db_path: PathBuf,
    /// Embedding model URI; the fingerprint `status` and vector search agree on.
    pub model: String,
    /// Modification time of `config_path` when `config` was read.
    config_mtime: Option<SystemTime>,
    embedder: Option<Box<dyn EmbeddingBackend>>,
    factories: Factories,
    reranker: Option<(String, Box<dyn Reranker>)>,
}

impl Index {
    /// Open the store and mirror the YAML config into it, as the CLI does.
    ///
    /// # Errors
    /// Propagates config-parse and store-open failures.
    pub fn open(
        config_path: &Path,
        db_path: &Path,
        model: String,
        factories: Factories,
    ) -> Result<Self> {
        let config = config::load_config(config_path)?;
        let store = Store::open(db_path)
            .with_context(|| format!("failed to open index at {}", db_path.display()))?;
        store.sync_config_to_db(&config)?;
        Ok(Self {
            store,
            config,
            config_path: config_path.to_owned(),
            db_path: db_path.to_owned(),
            model,
            config_mtime: modified(config_path),
            embedder: None,
            factories,
            reranker: None,
        })
    }

    /// Re-read the YAML config if the file changed since it was read.
    ///
    /// A daemon outlives the `qmd collection add` or `remove` that edits the
    /// config; without this it would validate `collections` against the list
    /// it started with. A config that no longer parses keeps the old one.
    ///
    /// # Errors
    /// Propagates config-parse and store failures.
    pub fn reload_config_if_changed(&mut self) -> Result<()> {
        let mtime = modified(&self.config_path);
        if mtime == self.config_mtime {
            return Ok(());
        }
        let config = config::load_config(&self.config_path)?;
        self.store.sync_config_to_db(&config)?;
        self.config = config;
        self.config_mtime = mtime;
        Ok(())
    }

    /// Load the embedding model if it is not loaded yet.
    ///
    /// An idle server never pays for it; the first `vec`/`hyde` query does.
    /// Separate from [`Self::embedder`] so a caller can hold the store and the
    /// embedder at once, which the search pipeline needs.
    ///
    /// # Errors
    /// Propagates model-load failures.
    pub fn ensure_embedder(&mut self) -> Result<()> {
        if self.embedder.is_none() {
            self.embedder = Some((self.factories.embedder)(&self.model)?);
        }
        Ok(())
    }

    /// The loaded embedder, or `None` if [`Self::ensure_embedder`] has not run.
    pub fn embedder(&self) -> Option<&dyn Embedder> {
        self.embedder.as_deref().map(|e| e as &dyn Embedder)
    }

    /// The reranker for the config's rerank model, loaded on first use and
    /// again when the config names another model.
    pub fn ensure_reranker(&mut self) -> Result<()> {
        let model = qmd::models::resolve_rerank_model(self.config.models.as_ref());
        if self.reranker.as_ref().is_none_or(|(m, _)| *m != model) {
            let reranker = (self.factories.reranker)(&model)?;
            self.reranker = Some((model, reranker));
        }
        Ok(())
    }

    pub fn reranker(&self) -> Option<&dyn Reranker> {
        self.reranker.as_ref().map(|(_, r)| r.as_ref())
    }
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

type Job = Box<dyn FnOnce(&mut Index) + Send + 'static>;

/// A handle to the index thread. `Send + Sync`, cheap to clone.
#[derive(Clone)]
pub struct Worker {
    tx: mpsc::UnboundedSender<Job>,
}

impl Worker {
    /// Spawn the thread, open the index on it, and hand back a handle.
    ///
    /// The index is built on the worker thread because a loaded embedder is
    /// not `Send` and so cannot be moved onto one afterwards. The thread exits
    /// when every handle is dropped, closing the store and unloading the model.
    ///
    /// # Errors
    /// Propagates whatever [`Index::open`] fails with.
    pub fn spawn(
        config_path: &Path,
        db_path: &Path,
        model: String,
        factories: Factories,
    ) -> Result<Self> {
        let (tx, mut rx) = mpsc::unbounded_channel::<Job>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let config_path = config_path.to_owned();
        let db_path = db_path.to_owned();
        std::thread::Builder::new()
            .name("qmd-index".to_owned())
            .spawn(move || {
                let mut index = match Index::open(&config_path, &db_path, model, factories) {
                    Ok(index) => {
                        let _ = ready_tx.send(Ok(()));
                        index
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                while let Some(job) = rx.blocking_recv() {
                    if let Err(e) = index.reload_config_if_changed() {
                        eprintln!("qmd: keeping the previous config: {e:#}");
                    }
                    let run = std::panic::AssertUnwindSafe(|| job(&mut index));
                    if std::panic::catch_unwind(run).is_err() {
                        eprintln!("qmd: a request panicked on the index thread");
                    }
                }
            })
            .context("failed to spawn the qmd index thread")?;
        ready_rx
            .recv()
            .context("the qmd index thread stopped before opening the index")??;
        Ok(Self { tx })
    }

    /// Run `f` on the index thread and await its result.
    ///
    /// # Errors
    /// Returns an error if the index thread is gone, or whatever `f` returns.
    pub async fn run<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Index) -> Result<T> + Send + 'static,
    {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Box::new(move |index| {
                if reply_tx.is_closed() {
                    return;
                }
                let _ = reply_tx.send(f(index));
            }))
            .map_err(|_| anyhow::anyhow!("the qmd index thread has stopped"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("the qmd index thread dropped the request"))?
    }
}
