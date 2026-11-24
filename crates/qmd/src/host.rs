//! Models in a child process.
//!
//! A long-lived server keeps its models in a host process and talks to it over
//! the host's stdin and stdout, one JSON line each way. Ending the host is how
//! the memory is given back and how a failure is recovered from, and neither
//! runs llama.cpp's teardown in the server: ggml aborts on a `cudaFree` that
//! fails, which a lost device makes it do.
//!
//! [`serve`] is the host's loop and [`HostClient`] the server's end. The client
//! starts a host on the first request, replaces one that fails
//! ([`Shared::call`] holds the rule), and ends it when the last handle to it is
//! dropped.

use std::fmt;
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde::{Deserialize, Serialize};

use crate::chunk::Tokenizer;
use crate::embed::{Embedder, EmbeddingBackend};
use crate::error::{Error, Result};
use crate::models::GpuMode;
use crate::rerank::Reranker;

/// What the server asks of the host. Each request names its model, so a host
/// loads lazily and one started after a failure needs no replay.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request {
    Embed {
        model: String,
        texts: Vec<String>,
    },
    CountTokens {
        model: String,
        text: String,
    },
    Truncate {
        model: String,
        text: String,
        max_tokens: usize,
    },
    Rerank {
        model: String,
        query: String,
        documents: Vec<String>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Response {
    Vectors(Vec<Option<Vec<f32>>>),
    Count(usize),
    Text(String),
    Scores(Vec<f64>),
    Error(String),
}

/// Loads a model in the host; the error is the text the server reports.
pub type Loader<T> = Box<dyn Fn(&str) -> std::result::Result<Box<T>, String>>;

/// How the host loads each kind of model.
pub struct Loaders {
    pub embedder: Loader<dyn EmbeddingBackend>,
    pub reranker: Loader<dyn Reranker>,
}

impl fmt::Debug for Loaders {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Loaders").finish_non_exhaustive()
    }
}

/// The models a host has loaded: one per role, replaced when a request names
/// another.
#[derive(Default)]
struct Loaded {
    embedder: Option<(String, Box<dyn EmbeddingBackend>)>,
    reranker: Option<(String, Box<dyn Reranker>)>,
}

impl Loaded {
    fn embedder(
        &mut self,
        model: &str,
        loaders: &Loaders,
    ) -> std::result::Result<&dyn EmbeddingBackend, String> {
        if self.embedder.as_ref().is_none_or(|(m, _)| m != model) {
            self.embedder = None;
            self.embedder = Some((model.to_owned(), (loaders.embedder)(model)?));
        }
        Ok(self.embedder.as_ref().expect("just loaded").1.as_ref())
    }

    fn reranker(
        &mut self,
        model: &str,
        loaders: &Loaders,
    ) -> std::result::Result<&dyn Reranker, String> {
        if self.reranker.as_ref().is_none_or(|(m, _)| m != model) {
            self.reranker = None;
            self.reranker = Some((model.to_owned(), (loaders.reranker)(model)?));
        }
        Ok(self.reranker.as_ref().expect("just loaded").1.as_ref())
    }

    fn answer(
        &mut self,
        request: Request,
        loaders: &Loaders,
    ) -> std::result::Result<Response, String> {
        let text = |e: Error| e.to_string();
        Ok(match request {
            Request::Embed { model, texts } => Response::Vectors(
                self.embedder(&model, loaders)?
                    .embed_batch(&texts)
                    .map_err(text)?,
            ),
            Request::CountTokens { model, text: t } => Response::Count(
                self.embedder(&model, loaders)?
                    .count_tokens(&t)
                    .map_err(text)?,
            ),
            Request::Truncate {
                model,
                text: t,
                max_tokens,
            } => Response::Text(
                self.embedder(&model, loaders)?
                    .truncate_to_tokens(&t, max_tokens)
                    .map_err(text)?,
            ),
            Request::Rerank {
                model,
                query,
                documents,
            } => Response::Scores(
                self.reranker(&model, loaders)?
                    .rerank(&query, &documents)
                    .map_err(text)?,
            ),
        })
    }
}

/// Answer requests from `input` on `output` until `input` ends, which is how
/// the server lets a host go when it is not ended outright.
///
/// # Errors
/// Fails when `input` cannot be read or `output` written.
pub fn serve(input: impl BufRead, mut output: impl Write, loaders: &Loaders) -> io::Result<()> {
    let mut loaded = Loaded::default();
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(request) => loaded
                .answer(request, loaders)
                .unwrap_or_else(Response::Error),
            Err(e) => Response::Error(format!("unreadable request: {e}")),
        };
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}

/// One running host, as the client drives it.
trait Link: Send {
    fn exchange(&mut self, request: &Request) -> io::Result<Response>;

    /// The host's process ID, for the log.
    fn pid(&self) -> u32;
}

/// Starts a host that runs its models under a GPU mode.
type Start = Box<dyn Fn(GpuMode) -> io::Result<Box<dyn Link>> + Send>;

/// A host process. Dropping it ends the process without asking it to free
/// anything.
struct Process {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Process {
    fn start(mut command: Command) -> io::Result<Self> {
        command.stdin(Stdio::piped()).stdout(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt as _;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = command.spawn()?;
        let stdin = child.stdin.take().expect("piped above");
        let stdout = BufReader::new(child.stdout.take().expect("piped above"));
        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }
}

impl Link for Process {
    fn exchange(&mut self, request: &Request) -> io::Result<Response> {
        let mut line = serde_json::to_string(request)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.flush()?;
        let mut answer = String::new();
        if self.stdout.read_line(&mut answer)? == 0 {
            let status = self
                .child
                .wait()
                .map_or_else(|e| e.to_string(), |s| s.to_string());
            return Err(io::Error::other(format!(
                "it ended without answering ({status})"
            )));
        }
        Ok(serde_json::from_str(&answer)?)
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A started host and what is known of it.
struct Host {
    link: Box<dyn Link>,
    mode: GpuMode,
    /// Whether it has answered a request with anything but an error.
    served: bool,
}

struct Shared {
    start: Start,
    /// The mode asked for by the handle created last.
    mode: GpuMode,
    /// `auto` met a GPU that a new host could not use and a CPU host answered
    /// instead: hosts run on the CPU until the models are next unloaded.
    cpu_fallback: bool,
    host: Option<Host>,
    handles: usize,
}

impl Shared {
    /// Send `request` to the host, starting one if none is running.
    ///
    /// A host that fails is ended and never asked again, whatever the failure:
    /// a model on a lost device cannot be told from any other, and a new
    /// process is the recovery that works. A host that had served is replaced
    /// and the request repeated. A new host that fails is the GPU itself not
    /// working, so under `auto` the next one runs on the CPU, and hosts stay
    /// there once it has answered; under any other mode, and on the CPU, the
    /// failure is the caller's and was not the GPU's.
    fn call(&mut self, request: &Request) -> Result<Response> {
        let mut try_cpu = false;
        loop {
            let mode = if self.cpu_fallback || try_cpu {
                GpuMode::Off
            } else {
                self.mode
            };
            if self.host.as_ref().is_some_and(|h| h.mode != mode) {
                self.host = None;
            }
            let host = match &mut self.host {
                Some(host) => host,
                None => {
                    let link = (self.start)(mode).map_err(|e| {
                        Error::Other(format!("could not start the model host: {e}"))
                    })?;
                    eprintln!(
                        "qmd: model host started (PID {}, GPU mode {mode})",
                        link.pid()
                    );
                    self.host.insert(Host {
                        link,
                        mode,
                        served: false,
                    })
                }
            };
            let failure = match host.link.exchange(request) {
                Ok(Response::Error(e)) => e,
                Ok(response) => {
                    host.served = true;
                    if try_cpu {
                        eprintln!("qmd: the models run on the CPU until they are next unloaded");
                        self.cpu_fallback = true;
                    }
                    return Ok(response);
                }
                Err(e) => format!("the model host stopped: {e}"),
            };
            let served = host.served;
            self.host = None;
            if served {
                eprintln!("qmd: the model host failed ({failure}); asking a new one");
            } else if self.mode == GpuMode::Auto && mode != GpuMode::Off {
                eprintln!("qmd: a new model host failed ({failure}); asking one on the CPU");
                try_cpu = true;
            } else {
                return Err(Error::Other(failure));
            }
        }
    }
}

/// The server's end of a model host. Clones share one host.
#[derive(Clone)]
pub struct HostClient {
    shared: Arc<Mutex<Shared>>,
}

impl HostClient {
    /// A client whose hosts are the processes `command` describes, one per GPU
    /// mode asked for. The process must run [`serve`] on its stdin and stdout.
    /// None is started before the first request.
    pub fn new(command: impl Fn(GpuMode) -> Command + Send + 'static) -> Self {
        Self::with_start(Box::new(move |mode| {
            Ok(Box::new(Process::start(command(mode))?) as Box<dyn Link>)
        }))
    }

    fn with_start(start: Start) -> Self {
        Self {
            shared: Arc::new(Mutex::new(Shared {
                start,
                mode: GpuMode::Auto,
                cpu_fallback: false,
                host: None,
                handles: 0,
            })),
        }
    }

    /// The embedder for `model` in the host, run under `mode`. The model loads
    /// on first use, and the host ends when the last handle is dropped.
    pub fn embedder(&self, model: &str, mode: GpuMode) -> RemoteEmbedder {
        RemoteEmbedder {
            lease: self.lease(mode),
            model: model.to_owned(),
        }
    }

    /// The reranker for `model` in the host, as [`Self::embedder`].
    pub fn reranker(&self, model: &str, mode: GpuMode) -> RemoteReranker {
        RemoteReranker {
            lease: self.lease(mode),
            model: model.to_owned(),
        }
    }

    /// Whether a host is running, and with it a model in memory: false before
    /// the first request and after a failure no new host cured.
    pub fn is_running(&self) -> bool {
        lock(&self.shared).host.is_some()
    }

    fn lease(&self, mode: GpuMode) -> Lease {
        let mut shared = lock(&self.shared);
        shared.handles += 1;
        if shared.mode != mode {
            shared.mode = mode;
            shared.cpu_fallback = false;
        }
        Lease(Arc::clone(&self.shared))
    }
}

impl fmt::Debug for HostClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostClient").finish_non_exhaustive()
    }
}

fn lock(shared: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    shared.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A handle's share of the host.
struct Lease(Arc<Mutex<Shared>>);

impl Lease {
    fn call(&self, request: &Request) -> Result<Response> {
        lock(&self.0).call(request)
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let mut shared = lock(&self.0);
        shared.handles -= 1;
        if shared.handles == 0 {
            if let Some(host) = shared.host.take() {
                eprintln!(
                    "qmd: model host ended (PID {}): models unloaded",
                    host.link.pid()
                );
            }
            shared.cpu_fallback = false;
        }
    }
}

fn unexpected(response: &Response) -> Error {
    Error::Other(format!("the model host answered out of turn: {response:?}"))
}

/// An embedding model in a host process.
pub struct RemoteEmbedder {
    lease: Lease,
    model: String,
}

impl fmt::Debug for RemoteEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteEmbedder")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl Embedder for RemoteEmbedder {
    fn model_uri(&self) -> &str {
        &self.model
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Option<Vec<f32>>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        match self.lease.call(&Request::Embed {
            model: self.model.clone(),
            texts: texts.to_vec(),
        })? {
            Response::Vectors(vectors) => Ok(vectors),
            other => Err(unexpected(&other)),
        }
    }
}

impl Tokenizer for RemoteEmbedder {
    fn count_tokens(&self, text: &str) -> Result<usize> {
        match self.lease.call(&Request::CountTokens {
            model: self.model.clone(),
            text: text.to_owned(),
        })? {
            Response::Count(count) => Ok(count),
            other => Err(unexpected(&other)),
        }
    }

    fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> Result<String> {
        match self.lease.call(&Request::Truncate {
            model: self.model.clone(),
            text: text.to_owned(),
            max_tokens,
        })? {
            Response::Text(text) => Ok(text),
            other => Err(unexpected(&other)),
        }
    }
}

/// A reranker in a host process.
pub struct RemoteReranker {
    lease: Lease,
    model: String,
}

impl fmt::Debug for RemoteReranker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteReranker")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl Reranker for RemoteReranker {
    fn model_uri(&self) -> &str {
        &self.model
    }

    fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        match self.lease.call(&Request::Rerank {
            model: self.model.clone(),
            query: query.to_owned(),
            documents: documents.to_vec(),
        })? {
            Response::Scores(scores) => Ok(scores),
            other => Err(unexpected(&other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::embed::MockEmbedder;
    use crate::rerank::MockReranker;

    fn mock_loaders(loads: Arc<AtomicUsize>) -> Loaders {
        Loaders {
            embedder: Box::new(move |model| {
                loads.fetch_add(1, Ordering::Relaxed);
                Ok(Box::new(MockEmbedder::new(model, 8)))
            }),
            reranker: Box::new(|_| Ok(Box::new(MockReranker))),
        }
    }

    /// Run `requests` through [`serve`] and return its answers.
    fn served(requests: &[Request], loaders: &Loaders) -> Vec<Response> {
        let input: String = requests
            .iter()
            .map(|r| serde_json::to_string(r).unwrap() + "\n")
            .collect();
        let mut output = Vec::new();
        serve(input.as_bytes(), &mut output, loaders).unwrap();
        String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn the_host_answers_as_the_model_would_and_loads_it_once() {
        let loads = Arc::new(AtomicUsize::new(0));
        let embed = |text: &str| Request::Embed {
            model: "m".to_owned(),
            texts: vec![text.to_owned()],
        };
        let answers = served(
            &[
                embed("alpha"),
                embed("beta"),
                Request::CountTokens {
                    model: "m".to_owned(),
                    text: "12345678".to_owned(),
                },
                Request::Rerank {
                    model: "r".to_owned(),
                    query: "cache policy".to_owned(),
                    documents: vec!["the cache".to_owned()],
                },
            ],
            &mock_loaders(Arc::clone(&loads)),
        );

        let direct = MockEmbedder::new("m", 8).embed("alpha").unwrap();
        assert!(matches!(&answers[0], Response::Vectors(v) if v[0] == direct));
        assert!(matches!(answers[2], Response::Count(2)));
        assert!(matches!(&answers[3], Response::Scores(s) if s == &[0.5]));
        assert_eq!(loads.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_failed_load_and_an_unreadable_request_are_answered_not_fatal() {
        let loaders = Loaders {
            embedder: Box::new(|_| Err("no such model".to_owned())),
            reranker: Box::new(|_| Ok(Box::new(MockReranker))),
        };
        let mut output = Vec::new();
        let input = "{\"op\":\"embed\",\"model\":\"m\",\"texts\":[\"a\"]}\nnot json\n";
        serve(input.as_bytes(), &mut output, &loaders).unwrap();
        let output = String::from_utf8(output).unwrap();
        let mut lines = output.lines();
        assert_eq!(lines.next(), Some(r#"{"error":"no such model"}"#));
        assert!(lines.next().unwrap().contains("unreadable request"));
    }

    /// What a scripted host does with each request it is sent.
    #[derive(Clone, Copy)]
    enum Act {
        Answer,
        Error,
        Die,
    }

    struct Scripted {
        acts: std::vec::IntoIter<Act>,
    }

    impl Link for Scripted {
        fn exchange(&mut self, _request: &Request) -> io::Result<Response> {
            match self.acts.next().unwrap_or(Act::Answer) {
                Act::Answer => Ok(Response::Vectors(vec![Some(vec![1.0])])),
                Act::Error => Ok(Response::Error("device lost".to_owned())),
                Act::Die => Err(io::Error::other("pipe closed")),
            }
        }

        fn pid(&self) -> u32 {
            0
        }
    }

    /// A client whose nth host follows `scripts[n]`, with the modes its hosts
    /// were started under.
    fn scripted(scripts: Vec<Vec<Act>>) -> (HostClient, Arc<Mutex<Vec<GpuMode>>>) {
        let started = Arc::new(Mutex::new(Vec::new()));
        let modes = Arc::clone(&started);
        let scripts = Mutex::new(scripts.into_iter());
        let client = HostClient::with_start(Box::new(move |mode| {
            modes.lock().unwrap().push(mode);
            let acts = scripts.lock().unwrap().next().unwrap_or_default();
            Ok(Box::new(Scripted {
                acts: acts.into_iter(),
            }) as Box<dyn Link>)
        }));
        (client, started)
    }

    fn embed(embedder: &RemoteEmbedder) -> Result<Option<Vec<f32>>> {
        embedder.embed("text")
    }

    #[test]
    fn no_host_starts_before_the_first_request_and_it_ends_with_the_last_handle() {
        let (client, started) = scripted(vec![]);
        let embedder = client.embedder("m", GpuMode::Cuda);
        let reranker = client.reranker("r", GpuMode::Cuda);
        assert!(started.lock().unwrap().is_empty());

        embed(&embedder).unwrap();
        embed(&embedder).unwrap();
        assert_eq!(*started.lock().unwrap(), [GpuMode::Cuda]);

        drop(embedder);
        assert!(lock(&client.shared).host.is_some(), "the reranker holds it");
        drop(reranker);
        assert!(lock(&client.shared).host.is_none());

        embed(&client.embedder("m", GpuMode::Cuda)).unwrap();
        assert_eq!(started.lock().unwrap().len(), 2);
    }

    #[test]
    fn a_host_that_fails_after_serving_is_replaced_and_the_request_answered() {
        for failure in [Act::Error, Act::Die] {
            let (client, started) = scripted(vec![vec![Act::Answer, failure]]);
            let embedder = client.embedder("m", GpuMode::Cuda);
            embed(&embedder).unwrap();

            assert_eq!(embed(&embedder).unwrap(), Some(vec![1.0]));
            assert_eq!(*started.lock().unwrap(), [GpuMode::Cuda; 2]);
        }
    }

    #[test]
    fn under_auto_a_new_host_that_fails_gives_way_to_the_cpu_until_unloaded() {
        let (client, started) = scripted(vec![vec![Act::Error]]);
        let embedder = client.embedder("m", GpuMode::Auto);
        embed(&embedder).unwrap();
        embed(&embedder).unwrap();
        assert_eq!(*started.lock().unwrap(), [GpuMode::Auto, GpuMode::Off]);

        drop(embedder);
        embed(&client.embedder("m", GpuMode::Auto)).unwrap();
        assert_eq!(started.lock().unwrap().last(), Some(&GpuMode::Auto));
    }

    #[test]
    fn a_named_mode_and_the_cpu_report_a_new_hosts_failure() {
        let (client, started) = scripted(vec![vec![Act::Error]]);
        let failure = embed(&client.embedder("m", GpuMode::Cuda)).unwrap_err();
        assert_eq!(failure.to_string(), "device lost");
        assert_eq!(*started.lock().unwrap(), [GpuMode::Cuda]);

        let (client, started) = scripted(vec![vec![Act::Die], vec![Act::Error]]);
        let failure = embed(&client.embedder("m", GpuMode::Auto)).unwrap_err();
        assert_eq!(failure.to_string(), "device lost");
        assert_eq!(*started.lock().unwrap(), [GpuMode::Auto, GpuMode::Off]);
    }

    #[test]
    fn a_request_no_host_answers_ends_after_one_host_per_remedy() {
        let fail = || vec![Act::Error];
        let (client, started) = scripted(vec![vec![Act::Answer, Act::Error], fail(), fail()]);
        let embedder = client.embedder("m", GpuMode::Auto);
        embed(&embedder).unwrap();

        assert_eq!(embed(&embedder).unwrap_err().to_string(), "device lost");
        assert_eq!(
            *started.lock().unwrap(),
            [GpuMode::Auto, GpuMode::Auto, GpuMode::Off]
        );
    }

    #[test]
    fn a_failure_the_cpu_does_not_cure_leaves_the_gpu_in_use() {
        let (client, started) = scripted(vec![vec![Act::Error], vec![Act::Error]]);
        let embedder = client.embedder("m", GpuMode::Auto);
        embed(&embedder).unwrap_err();

        embed(&embedder).unwrap();
        assert_eq!(started.lock().unwrap().last(), Some(&GpuMode::Auto));
    }

    #[test]
    fn a_client_is_running_while_a_host_holds_its_models() {
        let (client, _) = scripted(vec![vec![Act::Answer, Act::Error], vec![Act::Error]]);
        let embedder = client.embedder("m", GpuMode::Cuda);
        assert!(!client.is_running());

        embed(&embedder).unwrap();
        assert!(client.is_running());

        embed(&embedder).unwrap_err();
        assert!(!client.is_running(), "no new host cured the failure");
    }

    #[test]
    fn a_handle_asking_for_another_mode_moves_the_host_to_it() {
        let (client, started) = scripted(vec![]);
        let embedder = client.embedder("m", GpuMode::Cuda);
        embed(&embedder).unwrap();
        let _reranker = client.reranker("r", GpuMode::Off);
        embed(&embedder).unwrap();
        assert_eq!(*started.lock().unwrap(), [GpuMode::Cuda, GpuMode::Off]);
    }
}
