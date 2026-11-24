//! The thread that owns the index.
//!
//! [`ServerHandler`](rmcp::ServerHandler) requires `Send + Sync`, and neither
//! `rusqlite::Connection` (`Send`, not `Sync`) nor `LlamaEmbedder` (neither) can
//! satisfy it. A dedicated thread owns both and takes closures over an `mpsc`
//! channel; the handler holds only a [`Worker`], which is `Send + Sync`. Nothing
//! in the synchronous library is coloured async, and serializing index access
//! keeps one embedder however many clients are connected.
//!
//! The thread also drops the models once none has been used for the configured
//! idle time ([`qmd::models::resolve_idle_unload`]); the next query that needs
//! one loads it again.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context as _, Result, anyhow};
use qmd::config::{self, CollectionConfig};
use qmd::embed::{Embedder, EmbeddingBackend};
use qmd::models::GpuMode;
use qmd::rerank::Reranker;
use qmd::store::Store;
use tokio::sync::oneshot;

use crate::health::{EmbedderHealth, Watched};

/// Loads the embedding backend for a model URI, to run under a GPU mode. The
/// CLI supplies the real one, in a model host; tests supply a mock. Keeping it
/// a callback keeps the `llama` feature (and its native build) off this crate
/// entirely. Dropping what it returns must give the model's memory back.
pub type EmbedderFactory =
    Box<dyn Fn(&str, GpuMode) -> Result<Box<dyn EmbeddingBackend>> + Send + 'static>;

/// Loads the reranker for a model URI, on first use.
pub type RerankerFactory = Box<dyn Fn(&str, GpuMode) -> Result<Box<dyn Reranker>> + Send + 'static>;

/// The model loaders the index thread calls lazily.
pub struct Factories {
    pub embedder: EmbedderFactory,
    pub reranker: RerankerFactory,
    /// Whether what the loaders returned has a model in memory now. A handle
    /// to a model host has none until the host loads one, nor once the host
    /// has failed.
    pub in_memory: Box<dyn Fn() -> bool + Send + 'static>,
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
    /// The GPU mode the loaded models were asked for under, and when one was
    /// last used; `None` while none is loaded.
    loaded: Option<(GpuMode, Instant)>,
    health: Arc<EmbedderHealth>,
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
        health: Arc<EmbedderHealth>,
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
            loaded: None,
            health,
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
        let mode = self.model_mode()?;
        if self.embedder.is_none() {
            let inner = (self.factories.embedder)(&self.model, mode)
                .map_err(|e| anyhow!(self.health.failed(&format!("{e:#}"))))?;
            self.embedder = Some(Box::new(Watched {
                inner,
                health: Arc::clone(&self.health),
            }));
        }
        self.model_used(mode);
        Ok(())
    }

    /// The loaded embedder, or `None` if [`Self::ensure_embedder`] has not run.
    pub fn embedder(&self) -> Option<&dyn Embedder> {
        self.embedder.as_deref().map(|e| e as &dyn Embedder)
    }

    /// The reranker for the config's rerank model, loaded on first use and
    /// again when the config names another model.
    pub fn ensure_reranker(&mut self) -> Result<()> {
        let mode = self.model_mode()?;
        let model = qmd::models::resolve_rerank_model(self.config.models.as_ref());
        if self.reranker.as_ref().is_none_or(|(m, _)| *m != model) {
            let reranker = (self.factories.reranker)(&model, mode)?;
            self.reranker = Some((model, reranker));
        }
        self.model_used(mode);
        Ok(())
    }

    /// The GPU mode to load a model under. Models loaded under another mode
    /// are dropped first, so a changed `models.gpu` takes effect without a
    /// restart.
    fn model_mode(&mut self) -> Result<GpuMode> {
        let mode = qmd::models::resolve_gpu_mode(self.config.models.as_ref())?;
        if self.loaded.is_some_and(|(loaded, _)| loaded != mode) {
            self.unload_models();
        }
        Ok(mode)
    }

    /// Note that a model loaded under `mode` is about to be used.
    fn model_used(&mut self, mode: GpuMode) {
        self.loaded = Some((mode, Instant::now()));
        self.health.set_models_loaded(true);
    }

    /// Drop the loaded models, which gives their memory back.
    fn unload_models(&mut self) {
        self.embedder = None;
        self.reranker = None;
        self.loaded = None;
        self.health.set_models_loaded(false);
    }

    /// How long the loaded models may still go unused before they are
    /// dropped; `None` when none is loaded or they are kept.
    fn idle_left(&self) -> Option<Duration> {
        let (_, used) = self.loaded?;
        let idle = qmd::models::resolve_idle_unload(self.config.models.as_ref())?;
        Some(idle.saturating_sub(used.elapsed()))
    }

    /// Close the job that began at `started` if it used a model. The idle time
    /// starts again, since a model is in use until the job that asked for it
    /// ends, and `/health` learns whether the job left one in memory.
    fn job_ended(&mut self, started: Instant) {
        if let Some((_, used)) = &mut self.loaded
            && *used >= started
        {
            *used = Instant::now();
            self.health.set_models_loaded((self.factories.in_memory)());
        }
    }

    pub fn reranker(&self) -> Option<&dyn Reranker> {
        self.reranker.as_ref().map(|(_, r)| r.as_ref())
    }

    /// The embedder's state, for `status`.
    pub fn health(&self) -> &EmbedderHealth {
        &self.health
    }
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

type Job = Box<dyn FnOnce(&mut Index) + Send + 'static>;

/// A handle to the index thread. `Send + Sync`, cheap to clone.
#[derive(Clone)]
pub struct Worker {
    tx: mpsc::Sender<Job>,
    health: Arc<EmbedderHealth>,
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
        let (tx, rx) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::channel();
        let config_path = config_path.to_owned();
        let db_path = db_path.to_owned();
        let health = Arc::new(EmbedderHealth::new());
        let index_health = Arc::clone(&health);
        std::thread::Builder::new()
            .name("qmd-index".to_owned())
            .spawn(move || {
                let mut index =
                    match Index::open(&config_path, &db_path, model, factories, index_health) {
                        Ok(index) => {
                            let _ = ready_tx.send(Ok(()));
                            index
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                loop {
                    let job = match index.idle_left() {
                        Some(left) => match rx.recv_timeout(left) {
                            Ok(job) => job,
                            Err(RecvTimeoutError::Timeout) => {
                                index.unload_models();
                                continue;
                            }
                            Err(RecvTimeoutError::Disconnected) => break,
                        },
                        None => match rx.recv() {
                            Ok(job) => job,
                            Err(_) => break,
                        },
                    };
                    if let Err(e) = index.reload_config_if_changed() {
                        eprintln!("qmd: keeping the previous config: {e:#}");
                    }
                    let started = Instant::now();
                    let run = std::panic::AssertUnwindSafe(|| job(&mut index));
                    if std::panic::catch_unwind(run).is_err() {
                        eprintln!("qmd: a request panicked on the index thread");
                    }
                    index.job_ended(started);
                }
            })
            .context("failed to spawn the qmd index thread")?;
        ready_rx
            .recv()
            .context("the qmd index thread stopped before opening the index")??;
        Ok(Self { tx, health })
    }

    /// The embedder's state, readable while the index thread is busy.
    pub fn health(&self) -> &EmbedderHealth {
        &self.health
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

#[cfg(test)]
mod tests {
    use super::*;

    /// An index in `dir` whose loaders return the mocks and report `in_memory`.
    fn index(dir: &Path, in_memory: bool) -> Index {
        let config_path = dir.join("index.yml");
        let config = CollectionConfig {
            models: Some(config::ModelsConfig {
                gpu: Some(GpuMode::Off),
                idle_seconds: Some(60),
                ..Default::default()
            }),
            ..Default::default()
        };
        config::save_config(&config_path, &config).unwrap();
        let factories = Factories {
            embedder: Box::new(|_: &str, _| {
                Ok(Box::new(qmd::embed::MockEmbedder::default()) as Box<dyn EmbeddingBackend>)
            }),
            reranker: Box::new(|_: &str, _| {
                Ok(Box::new(qmd::rerank::MockReranker) as Box<dyn Reranker>)
            }),
            in_memory: Box::new(move || in_memory),
        };
        Index::open(
            &config_path,
            &dir.join("index.sqlite"),
            "mock".to_owned(),
            factories,
            Arc::new(EmbedderHealth::new()),
        )
        .unwrap()
    }

    #[test]
    fn the_idle_time_runs_from_the_end_of_the_job_that_used_a_model() {
        let dir = tempfile::tempdir().unwrap();
        let mut index = index(dir.path(), true);
        let tick = || std::thread::sleep(Duration::from_millis(2));

        let started = Instant::now();
        index.ensure_embedder().unwrap();
        let (_, at_load) = index.loaded.unwrap();
        tick();
        index.job_ended(started);
        let (_, at_end) = index.loaded.unwrap();
        assert!(at_end > at_load);

        tick();
        index.job_ended(Instant::now());
        assert_eq!(index.loaded.unwrap().1, at_end, "the job used no model");
    }

    #[test]
    fn a_job_that_leaves_no_model_in_memory_reads_unloaded() {
        let dir = tempfile::tempdir().unwrap();
        let mut index = index(dir.path(), false);

        let started = Instant::now();
        index.ensure_embedder().unwrap();
        assert!(index.health().models_loaded(), "while the job runs");
        index.job_ended(started);
        assert!(!index.health().models_loaded());
    }
}
