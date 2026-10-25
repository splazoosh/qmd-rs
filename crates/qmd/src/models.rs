//! Model inventory, resolution, and the `pull` download protocol, ported from
//! TS `llm.ts` (`parseHfUri`, `inspectGgufFile`, `pullModels`). See [`hf`] for
//! how a download reaches Hugging Face.

pub mod hf;

use std::io::Read;
use std::path::{Path, PathBuf};

use crate::config::ModelsConfig;
use crate::error::{Error, Result};
use hf::{HfSetup, Route};

/// Default embedding model (768-dim EmbeddingGemma-300M Q8).
pub const DEFAULT_EMBED_MODEL: &str =
    "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";

/// Resolve the embed model URI: explicit config > `QMD_EMBED_MODEL` > default.
pub fn resolve_embed_model(config: Option<&ModelsConfig>) -> String {
    resolve_model(
        config.and_then(|m| m.embed.as_deref()),
        "QMD_EMBED_MODEL",
        DEFAULT_EMBED_MODEL,
    )
}

/// Default reranker (Qwen3-Reranker-0.6B Q8).
pub const DEFAULT_RERANK_MODEL: &str =
    "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";

/// Resolve the rerank model URI: explicit config > `QMD_RERANK_MODEL` > default.
pub fn resolve_rerank_model(config: Option<&ModelsConfig>) -> String {
    resolve_model(
        config.and_then(|m| m.rerank.as_deref()),
        "QMD_RERANK_MODEL",
        DEFAULT_RERANK_MODEL,
    )
}

fn resolve_model(configured: Option<&str>, env_var: &str, default: &str) -> String {
    if let Some(model) = configured {
        return model.to_owned();
    }
    if let Ok(model) = std::env::var(env_var)
        && !model.trim().is_empty()
    {
        return model;
    }
    default.to_owned()
}

/// Model cache directory: `{cache_dir}/models` (respects `XDG_CACHE_HOME`).
pub fn model_cache_dir() -> PathBuf {
    crate::config::cache_dir().join("models")
}

/// A parsed `hf:{user}/{repo}/{file}` reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfRef {
    /// `user/repo`.
    pub repo: String,
    /// File path within the repo.
    pub file: String,
}

impl HfRef {
    /// Download URL on `endpoint` (`https://huggingface.co`, or a mirror
    /// named by `HF_ENDPOINT`).
    pub fn url(&self, endpoint: &str) -> String {
        format!(
            "{}/{}/resolve/main/{}",
            endpoint.trim_end_matches('/'),
            self.repo,
            self.file
        )
    }

    /// The directory the CLI downloads into so that the file lands at
    /// [`model_path`]: `{cache}/user/repo`.
    fn local_dir(&self, cache_dir: &Path) -> PathBuf {
        self.repo
            .split('/')
            .fold(cache_dir.to_path_buf(), |path, part| path.join(part))
    }
}

/// Parse an `hf:user/repo/path/to/file.gguf` URI.
pub fn parse_hf_uri(model: &str) -> Option<HfRef> {
    let rest = model.strip_prefix("hf:")?;
    let parts: Vec<&str> = rest.split('/').collect();
    let unsafe_part = |p: &&str| matches!(*p, "" | "." | "..") || p.contains(['\\', ':']);
    if parts.len() < 3 || parts.iter().any(unsafe_part) {
        return None;
    }
    Some(HfRef {
        repo: parts[..2].join("/"),
        file: parts[2..].join("/"),
    })
}

/// GGUF sniff result kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufKind {
    Missing,
    Gguf,
    Html,
    Invalid,
}

/// Result of sniffing a candidate GGUF file.
#[derive(Debug, Clone)]
pub struct GgufInspection {
    pub kind: GgufKind,
    pub size_bytes: u64,
    pub details: String,
}

impl GgufInspection {
    pub fn is_valid(&self) -> bool {
        self.kind == GgufKind::Gguf
    }
}

/// Inspect a potential GGUF model file without mutating it: check the `GGUF`
/// magic and detect HTML error pages from proxies/captive portals.
pub fn inspect_gguf_file(path: &Path) -> GgufInspection {
    let Ok(metadata) = std::fs::metadata(path) else {
        return GgufInspection {
            kind: GgufKind::Missing,
            size_bytes: 0,
            details: "file does not exist".to_owned(),
        };
    };
    let size_bytes = metadata.len();

    let mut sniff = [0u8; 512];
    let read = std::fs::File::open(path)
        .and_then(|mut f| f.read(&mut sniff))
        .unwrap_or(0);
    let sniff = &sniff[..read];

    if sniff.starts_with(b"GGUF") {
        return GgufInspection {
            kind: GgufKind::Gguf,
            size_bytes,
            details: format!("valid GGUF ({} KB)", size_bytes / 1024),
        };
    }

    let text = String::from_utf8_lossy(sniff).to_lowercase();
    if text.contains("<!doctype") || text.contains("<html") {
        return GgufInspection {
            kind: GgufKind::Html,
            size_bytes,
            details: format!(
                "HTML page, not a GGUF model ({} KB); likely proxy/firewall/captive portal response",
                size_bytes / 1024
            ),
        };
    }

    GgufInspection {
        kind: GgufKind::Invalid,
        size_bytes,
        details: format!("not valid GGUF (bad magic, {} KB)", size_bytes / 1024),
    }
}

/// Validate a downloaded model file; removes invalid files so the next
/// attempt re-downloads.
///
/// # Errors
/// Returns a descriptive error for HTML pages or corrupt files.
fn validate_gguf_file(path: &Path, model_uri: &str) -> Result<()> {
    let inspection = inspect_gguf_file(path);
    if inspection.kind == GgufKind::Missing || inspection.is_valid() {
        return Ok(());
    }
    let _ = std::fs::remove_file(path);
    if inspection.kind == GgufKind::Html {
        return Err(Error::Other(format!(
            "downloaded model file is an HTML page, not a GGUF model.\n\
             Something is intercepting the download from huggingface.co.\n\
             Model: {model_uri}\n Path: {}\n\
             Try a mirror (HF_ENDPOINT=https://hf-mirror.com) or download manually \
             and set QMD_EMBED_MODEL=/path/to/model.gguf",
            path.display()
        )));
    }
    Err(Error::Other(format!(
        "model file is not valid GGUF ({}).\nModel: {model_uri}\nPath: {}\n\
         The file has been removed; run the command again to re-download.",
        inspection.details,
        path.display()
    )))
}

/// Result of pulling one model.
#[derive(Debug, Clone)]
pub struct PullResult {
    pub model: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    /// Whether an existing cached file was replaced.
    pub refreshed: bool,
    /// The route that fetched the file; `None` when the cached one was kept.
    pub route: Option<Route>,
    /// Why the CLI failed, when it did and the built-in download took over.
    pub cli_failure: Option<String>,
}

fn remote_etag(setup: &HfSetup, hf: &HfRef) -> Option<String> {
    let response = hf::call(setup, true, &hf.url(&setup.endpoint)).ok()?;
    if response.status() != 200 {
        return None;
    }
    response
        .headers()
        .get("etag")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
}

/// Local cache path for a model URI: `hf:user/repo/file` lands at
/// `{cache}/user/repo/file`, so two repos shipping the same file name cannot
/// share one file; other values are treated as filesystem paths.
pub fn model_path(model: &str, cache_dir: &Path) -> PathBuf {
    match parse_hf_uri(model) {
        Some(hf) => hf
            .repo
            .split('/')
            .chain(hf.file.split('/'))
            .fold(cache_dir.to_path_buf(), |path, part| path.join(part)),
        None => PathBuf::from(model),
    }
}

/// The `.etag` recorded beside a cached model.
fn etag_path(model_file: &Path) -> PathBuf {
    let mut name = model_file.file_name().unwrap_or_default().to_os_string();
    name.push(".etag");
    model_file.with_file_name(name)
}

/// Move a model cached under the old file-name-only layout (`{cache}/file`)
/// to `path`, but only when its recorded ETag equals `remote`: Hugging Face
/// serves an LFS file's SHA-256 as its ETag, so a match proves it is this
/// repo's file and not a same-named one from elsewhere. Anything else is left
/// alone and the model downloads again.
fn adopt_legacy(cache_dir: &Path, hf: &HfRef, path: &Path, remote: Option<&str>) {
    let Some(remote) = remote else { return };
    let Some(name) = hf.file.rsplit('/').next() else {
        return;
    };
    let legacy = cache_dir.join(name);
    let legacy_etag = etag_path(&legacy);
    let matches = std::fs::read_to_string(&legacy_etag).is_ok_and(|e| e.trim() == remote);
    if path.exists() || !matches || !inspect_gguf_file(&legacy).is_valid() {
        return;
    }
    let Some(parent) = path.parent() else { return };
    if std::fs::create_dir_all(parent).is_ok() && std::fs::rename(&legacy, path).is_ok() {
        let _ = std::fs::rename(&legacy_etag, etag_path(path));
    }
}

fn download(setup: &HfSetup, url: &str, dest: &Path) -> Result<u64> {
    let response = hf::call(setup, false, url)
        .map_err(|e| Error::Other(format!("download failed for {url}: {e}")))?;
    if response.status() != 200 {
        return Err(Error::Other(format!(
            "download failed for {url}: HTTP {}",
            response.status()
        )));
    }

    let expected: Option<u64> = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok());
    let tmp = dest.with_extension("part");
    let mut file = std::fs::File::create(&tmp)?;
    let mut reader = response.into_body().into_reader();
    let bytes = std::io::copy(&mut reader, &mut file)?;
    file.sync_all()?;
    drop(file);
    if let Some(expected) = expected
        && bytes != expected
    {
        let _ = std::fs::remove_file(&tmp);
        return Err(Error::Other(format!(
            "download truncated for {url}: got {bytes} of {expected} bytes"
        )));
    }

    if dest.exists() {
        std::fs::remove_file(dest)?;
    }
    std::fs::rename(&tmp, dest)?;
    Ok(bytes)
}

/// Download a model into the cache, skipping when the stored ETag still
/// matches the remote one. Validates the GGUF magic before accepting.
///
/// The file is fetched by the `hf` CLI when one is on `PATH`, else by qmd's
/// own download, which sends the user's Hugging Face token when one is set
/// (see [`hf`]). A CLI that fails is reported in
/// [`PullResult::cli_failure`] and the built-in download takes over. Either
/// route lands the file at [`model_path`].
///
/// Deviation from TS: when the ETag HEAD request fails (offline) but a valid
/// local file exists, the local file is kept instead of failing.
///
/// # Errors
/// Fails on network errors with no usable local file, or invalid downloads.
pub fn pull_model(model: &str, cache_dir: &Path, refresh: bool) -> Result<PullResult> {
    pull_model_with(&HfSetup::from_env(), model, cache_dir, refresh)
}

/// [`pull_model`] with an explicit Hugging Face setup.
///
/// # Errors
/// As [`pull_model`].
pub fn pull_model_with(
    setup: &HfSetup,
    model: &str,
    cache_dir: &Path,
    refresh: bool,
) -> Result<PullResult> {
    std::fs::create_dir_all(cache_dir)?;
    let path = model_path(model, cache_dir);

    let Some(hf) = parse_hf_uri(model) else {
        validate_gguf_file(&path, model)?;
        if !path.exists() {
            return Err(Error::Other(format!(
                "model file not found: {}",
                path.display()
            )));
        }
        let size_bytes = std::fs::metadata(&path)?.len();
        return Ok(PullResult {
            model: model.to_owned(),
            path,
            size_bytes,
            refreshed: false,
            route: None,
            cli_failure: None,
        });
    };

    let remote = remote_etag(setup, &hf);
    adopt_legacy(cache_dir, &hf, &path, remote.as_deref());
    let etag_path = etag_path(&path);
    let local_etag = std::fs::read_to_string(&etag_path)
        .ok()
        .map(|s| s.trim().to_owned());
    let local_valid = inspect_gguf_file(&path).is_valid();

    let up_to_date = local_valid
        && !refresh
        && match (&remote, &local_etag) {
            (Some(r), Some(l)) => r == l,
            (None, _) => true,
            (Some(_), None) => false,
        };

    if up_to_date {
        let size_bytes = std::fs::metadata(&path)?.len();
        return Ok(PullResult {
            model: model.to_owned(),
            path,
            size_bytes,
            refreshed: false,
            route: None,
            cli_failure: None,
        });
    }

    let refreshed = path.exists();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let (route, cli_failure) = fetch(setup, &hf, cache_dir, &path, refreshed)?;
    validate_gguf_file(&path, model)?;
    if let Some(etag) = remote.or_else(|| remote_etag(setup, &hf)) {
        std::fs::write(&etag_path, format!("{etag}\n"))?;
    }
    let size_bytes = std::fs::metadata(&path)?.len();
    Ok(PullResult {
        model: model.to_owned(),
        path,
        size_bytes,
        refreshed,
        route: Some(route),
        cli_failure,
    })
}

/// Fetch `hf` to `path` through the CLI if there is one, falling back to the
/// built-in download when it fails or leaves no new valid GGUF there.
fn fetch(
    setup: &HfSetup,
    hf: &HfRef,
    cache_dir: &Path,
    path: &Path,
    force: bool,
) -> Result<(Route, Option<String>)> {
    let mut cli_failure = None;
    if let Some(cli) = &setup.cli {
        let local_dir = hf.local_dir(cache_dir);
        let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
        let before = modified(path);
        match hf::cli_download(setup, cli, &hf.repo, &hf.file, &local_dir, force) {
            Ok(()) if inspect_gguf_file(path).is_valid() && modified(path) != before => {
                return Ok((Route::Cli(cli.clone()), None));
            }
            Ok(()) => {
                cli_failure = Some(format!(
                    "{} left no new valid GGUF at {}",
                    cli.display(),
                    path.display()
                ));
            }
            Err(e) => cli_failure = Some(format!("{} failed: {e}", cli.display())),
        }
    }
    download(setup, &hf.url(&setup.endpoint), path)?;
    Ok((setup.builtin_route(), cli_failure))
}

/// Resolve a model to a local file path, downloading it on first use (the
/// lazy path used by `embed`/`vsearch`; `pull` is the eager form).
///
/// # Errors
/// Propagates download and validation failures.
pub fn ensure_model(model: &str) -> Result<PathBuf> {
    let cache_dir = model_cache_dir();
    let path = model_path(model, &cache_dir);
    if inspect_gguf_file(&path).is_valid() {
        return Ok(path);
    }
    let result = pull_model(model, &cache_dir, false)?;
    if let Some(failure) = &result.cli_failure {
        eprintln!("qmd: {failure}; used the built-in download instead");
    }
    Ok(result.path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hf_uris() {
        let hf = parse_hf_uri(DEFAULT_EMBED_MODEL).unwrap();
        assert_eq!(hf.repo, "ggml-org/embeddinggemma-300M-GGUF");
        assert_eq!(hf.file, "embeddinggemma-300M-Q8_0.gguf");
        assert!(hf.url(hf::DEFAULT_ENDPOINT).ends_with(
            "/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf"
        ));

        assert!(parse_hf_uri("hf:only/two").is_none());
        assert!(parse_hf_uri("hf:u/../../escape.gguf").is_none());
        assert!(parse_hf_uri("/local/path.gguf").is_none());
        assert_eq!(
            parse_hf_uri("hf:u/r/nested/dir/f.gguf").unwrap().file,
            "nested/dir/f.gguf"
        );
    }

    #[test]
    fn a_legacy_cached_model_is_adopted_only_when_its_etag_matches() {
        let cache = tempfile::tempdir().unwrap();
        let legacy = cache.path().join("m.gguf");
        std::fs::write(&legacy, b"GGUFdata").unwrap();
        std::fs::write(etag_path(&legacy), "\"sha-a\"\n").unwrap();

        let other = parse_hf_uri("hf:x/other/m.gguf").unwrap();
        let other_path = model_path("hf:x/other/m.gguf", cache.path());
        adopt_legacy(cache.path(), &other, &other_path, Some("\"sha-b\""));
        assert!(
            legacy.exists() && !other_path.exists(),
            "a mismatch moved it"
        );

        let hf = parse_hf_uri("hf:x/repo/m.gguf").unwrap();
        let path = model_path("hf:x/repo/m.gguf", cache.path());
        adopt_legacy(cache.path(), &hf, &path, Some("\"sha-a\""));
        assert!(!legacy.exists() && inspect_gguf_file(&path).is_valid());
        assert_eq!(
            std::fs::read_to_string(etag_path(&path)).unwrap().trim(),
            "\"sha-a\""
        );
    }

    #[test]
    fn model_resolution_order_is_config_env_default() {
        let config = ModelsConfig {
            embed: Some("hf:a/b/c.gguf".to_owned()),
            generate: None,
            rerank: None,
        };
        assert_eq!(resolve_embed_model(Some(&config)), "hf:a/b/c.gguf");
        assert_eq!(resolve_embed_model(None), DEFAULT_EMBED_MODEL);
    }

    #[test]
    fn gguf_inspection_detects_magic_html_and_garbage() {
        let dir = tempfile::tempdir().unwrap();

        let good = dir.path().join("good.gguf");
        std::fs::write(&good, b"GGUFrest-of-model-data").unwrap();
        assert_eq!(inspect_gguf_file(&good).kind, GgufKind::Gguf);

        let html = dir.path().join("html.gguf");
        std::fs::write(&html, b"<!DOCTYPE html><html>blocked</html>").unwrap();
        assert_eq!(inspect_gguf_file(&html).kind, GgufKind::Html);

        let bad = dir.path().join("bad.gguf");
        std::fs::write(&bad, b"not a model").unwrap();
        assert_eq!(inspect_gguf_file(&bad).kind, GgufKind::Invalid);

        assert_eq!(
            inspect_gguf_file(&dir.path().join("missing.gguf")).kind,
            GgufKind::Missing
        );
    }

    mod pulls {
        use super::super::*;
        use hf::{HfToken, TokenSource};
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpListener;
        use std::sync::{Arc, Mutex};

        const TOKEN: &str = "hf_testTOKENvalue";
        const MODEL: &str = "hf:user/repo/m.gguf";

        /// What one request carried: method, path and `Authorization` header.
        type Seen = Arc<Mutex<Vec<(String, String, Option<String>)>>>;

        /// A local HTTP server answering each request with `respond(method,
        /// path)` and recording what it received.
        fn serve(
            respond: impl Fn(&str, &str) -> (u16, Vec<(String, String)>, Vec<u8>) + Send + 'static,
        ) -> (String, Seen) {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let seen: Seen = Arc::default();
            let log = Arc::clone(&seen);
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(mut stream) = stream else { return };
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    let mut parts = line.split_whitespace();
                    let method = parts.next().unwrap_or_default().to_owned();
                    let path = parts.next().unwrap_or_default().to_owned();
                    let mut auth = None;
                    loop {
                        let mut header = String::new();
                        reader.read_line(&mut header).unwrap();
                        if header.trim().is_empty() {
                            break;
                        }
                        if let Some((name, value)) = header.split_once(':')
                            && name.eq_ignore_ascii_case("authorization")
                        {
                            auth = Some(value.trim().to_owned());
                        }
                    }
                    let (status, headers, body) = respond(&method, &path);
                    log.lock().unwrap().push((method.clone(), path, auth));
                    let mut head = format!(
                        "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n",
                        body.len()
                    );
                    for (name, value) in headers {
                        head.push_str(&format!("{name}: {value}\r\n"));
                    }
                    head.push_str("\r\n");
                    let _ = stream.write_all(head.as_bytes());
                    if method != "HEAD" {
                        let _ = stream.write_all(&body);
                    }
                }
            });
            (base, seen)
        }

        /// The endpoint redirects once on its own origin and then to `cdn`,
        /// another origin standing in for the signed-URL CDN.
        fn hub(cdn: &str) -> (String, Seen) {
            let cdn = cdn.to_owned();
            serve(move |_, path| match path {
                "/user/repo/resolve/main/m.gguf" => (
                    302,
                    vec![("Location".into(), "/user/repo/hop".into())],
                    Vec::new(),
                ),
                "/user/repo/hop" => (
                    302,
                    vec![("Location".into(), format!("{cdn}/signed?sig=1"))],
                    Vec::new(),
                ),
                _ => (404, Vec::new(), Vec::new()),
            })
        }

        fn cdn() -> (String, Seen) {
            serve(|_, _| {
                (
                    200,
                    vec![("ETag".into(), "\"sha-1\"".into())],
                    b"GGUFfrom-cdn".to_vec(),
                )
            })
        }

        fn setup(endpoint: &str, cli: Option<PathBuf>, token: bool) -> HfSetup {
            HfSetup {
                endpoint: endpoint.to_owned(),
                cli,
                token: token.then(|| (HfToken::from_test(TOKEN), TokenSource::Env("HF_TOKEN"))),
            }
        }

        /// A stand-in CLI: `ok` writes a GGUF where `--local-dir` and the file
        /// name say; otherwise it prints the token to stderr and fails.
        fn stand_in(dir: &Path, ok: bool) -> PathBuf {
            let (name, script) = match (cfg!(windows), ok) {
                (true, true) => ("hf.cmd", "@echo off\r\necho GGUFstand-in> \"%~5\\%~3\"\r\n"),
                (true, false) => (
                    "hf.cmd",
                    "@echo off\r\necho error: token hf_testTOKENvalue rejected 1>&2\r\nexit /b 1\r\n",
                ),
                (false, true) => ("hf", "#!/bin/sh\nprintf GGUFstand-in > \"$5/$3\"\n"),
                (false, false) => (
                    "hf",
                    "#!/bin/sh\necho 'error: token hf_testTOKENvalue rejected' >&2\nexit 1\n",
                ),
            };
            let path = dir.join(name);
            std::fs::write(&path, script).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            path
        }

        fn auth_of(seen: &Seen) -> Vec<(String, Option<String>)> {
            seen.lock()
                .unwrap()
                .iter()
                .map(|(_, path, auth)| (path.clone(), auth.clone()))
                .collect()
        }

        #[test]
        fn the_token_reaches_the_endpoint_and_not_a_host_it_redirects_to() {
            let (cdn, cdn_seen) = cdn();
            let (hub, hub_seen) = hub(&cdn);
            let cache = tempfile::tempdir().unwrap();

            let result =
                pull_model_with(&setup(&hub, None, true), MODEL, cache.path(), false).unwrap();

            assert_eq!(result.route, Some(Route::Builtin { token: true }));
            assert_eq!(result.path, model_path(MODEL, cache.path()));
            assert!(inspect_gguf_file(&result.path).is_valid());
            let bearer = Some(format!("Bearer {TOKEN}"));
            assert!(
                auth_of(&hub_seen).iter().all(|(_, auth)| *auth == bearer),
                "every endpoint request, both hops included, carries the token"
            );
            assert!(
                auth_of(&cdn_seen).iter().all(|(_, auth)| auth.is_none()),
                "the redirect target never sees it"
            );
            assert!(!format!("{result:?}").contains(TOKEN));
        }

        #[test]
        fn with_no_token_nothing_is_sent() {
            let (cdn, cdn_seen) = cdn();
            let (hub, hub_seen) = hub(&cdn);
            let cache = tempfile::tempdir().unwrap();

            let result =
                pull_model_with(&setup(&hub, None, false), MODEL, cache.path(), false).unwrap();

            assert_eq!(result.route, Some(Route::Builtin { token: false }));
            assert!(auth_of(&hub_seen).iter().all(|(_, a)| a.is_none()));
            assert!(auth_of(&cdn_seen).iter().all(|(_, a)| a.is_none()));
        }

        #[test]
        fn a_cli_pull_lands_at_the_cache_path_and_records_the_etag() {
            let (cdn, _) = cdn();
            let (hub, _) = hub(&cdn);
            let cache = tempfile::tempdir().unwrap();
            let bin = tempfile::tempdir().unwrap();
            let cli = stand_in(bin.path(), true);
            let on_path = HfSetup::from_vars(|name| match name {
                "PATH" => Some(bin.path().as_os_str().to_owned()),
                "PATHEXT" => Some(".exe;.cmd".into()),
                "HF_ENDPOINT" => Some(hub.clone().into()),
                "HF_TOKEN" => Some(TOKEN.into()),
                _ => None,
            });
            assert_eq!(on_path.cli.as_ref(), Some(&cli));

            let result = pull_model_with(&on_path, MODEL, cache.path(), false).unwrap();

            assert_eq!(result.route, Some(Route::Cli(cli)));
            assert!(result.cli_failure.is_none());
            let path = model_path(MODEL, cache.path());
            assert_eq!(result.path, path);
            assert!(std::fs::read(&path).unwrap().starts_with(b"GGUFstand-in"));
            assert_eq!(
                std::fs::read_to_string(etag_path(&path)).unwrap().trim(),
                "\"sha-1\""
            );
        }

        #[test]
        fn a_failing_cli_falls_back_and_its_report_hides_the_token() {
            let (cdn, _) = cdn();
            let (hub, _) = hub(&cdn);
            let cache = tempfile::tempdir().unwrap();
            let bin = tempfile::tempdir().unwrap();
            let cli = stand_in(bin.path(), false);

            let result =
                pull_model_with(&setup(&hub, Some(cli), true), MODEL, cache.path(), false).unwrap();

            assert_eq!(result.route, Some(Route::Builtin { token: true }));
            let failure = result.cli_failure.unwrap();
            assert!(failure.contains("rejected"), "{failure}");
            assert!(!failure.contains(TOKEN), "{failure}");
            assert!(
                std::fs::read(&result.path)
                    .unwrap()
                    .starts_with(b"GGUFfrom-cdn")
            );
        }

        #[test]
        fn a_rejected_token_is_named_but_never_shown() {
            let (hub, _) = serve(|_, _| (401, Vec::new(), Vec::new()));
            let cache = tempfile::tempdir().unwrap();

            let err = pull_model_with(&setup(&hub, None, true), MODEL, cache.path(), false)
                .unwrap_err()
                .to_string();

            assert!(err.contains("token was sent"), "{err}");
            assert!(!err.contains(TOKEN), "{err}");
        }

        #[test]
        fn a_cli_that_exits_0_without_writing_never_stamps_a_stale_file() {
            let (cdn, _) = cdn();
            let (hub, _) = hub(&cdn);
            let cache = tempfile::tempdir().unwrap();
            let path = model_path(MODEL, cache.path());
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"GGUFstale").unwrap();
            std::fs::write(etag_path(&path), "\"old\"\n").unwrap();
            let bin = tempfile::tempdir().unwrap();
            let (name, script) = if cfg!(windows) {
                ("hf.cmd", "@exit /b 0\r\n")
            } else {
                ("hf", "#!/bin/sh\nexit 0\n")
            };
            let cli = bin.path().join(name);
            std::fs::write(&cli, script).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
            }

            let result =
                pull_model_with(&setup(&hub, Some(cli), false), MODEL, cache.path(), false)
                    .unwrap();

            assert_eq!(result.route, Some(Route::Builtin { token: false }));
            assert!(
                result
                    .cli_failure
                    .unwrap()
                    .contains("left no new valid GGUF")
            );
            assert!(std::fs::read(&path).unwrap().starts_with(b"GGUFfrom-cdn"));
            assert_eq!(
                std::fs::read_to_string(etag_path(&path)).unwrap().trim(),
                "\"sha-1\""
            );
        }
    }

    #[test]
    fn model_path_maps_hf_to_cache_and_paths_to_themselves() {
        let cache = Path::new("/cache/models");
        assert_eq!(
            model_path(DEFAULT_EMBED_MODEL, cache),
            cache
                .join("ggml-org")
                .join("embeddinggemma-300M-GGUF")
                .join("embeddinggemma-300M-Q8_0.gguf")
        );
        assert_ne!(
            model_path("hf:a/one/model.gguf", cache),
            model_path("hf:a/two/model.gguf", cache),
            "two repos with one file name must not share a cache file"
        );
        assert_eq!(
            model_path("/somewhere/model.gguf", cache),
            PathBuf::from("/somewhere/model.gguf")
        );
    }
}
