//! Model inventory, resolution, and the `pull` download protocol, ported from
//! TS `llm.ts` (`parseHfUri`, `inspectGgufFile`, `pullModels`).

use std::io::Read;
use std::path::{Path, PathBuf};

use crate::config::ModelsConfig;
use crate::error::{Error, Result};

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
    /// Download URL on huggingface.co (honors `HF_ENDPOINT` for mirrors).
    pub fn url(&self) -> String {
        let endpoint = std::env::var("HF_ENDPOINT")
            .ok()
            .filter(|e| !e.trim().is_empty())
            .unwrap_or_else(|| "https://huggingface.co".to_owned());
        format!(
            "{}/{}/resolve/main/{}",
            endpoint.trim_end_matches('/'),
            self.repo,
            self.file
        )
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
}

fn remote_etag(hf: &HfRef) -> Option<String> {
    let response = ureq::head(hf.url()).call().ok()?;
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

fn download(url: &str, dest: &Path) -> Result<u64> {
    let response = ureq::get(url)
        .call()
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
/// Deviation from TS: when the ETag HEAD request fails (offline) but a valid
/// local file exists, the local file is kept instead of failing.
///
/// # Errors
/// Fails on network errors with no usable local file, or invalid downloads.
pub fn pull_model(model: &str, cache_dir: &Path, refresh: bool) -> Result<PullResult> {
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
        });
    };

    let remote = remote_etag(&hf);
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
        });
    }

    let refreshed = path.exists();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    download(&hf.url(), &path)?;
    validate_gguf_file(&path, model)?;
    if let Some(etag) = remote.or_else(|| remote_etag(&hf)) {
        std::fs::write(&etag_path, format!("{etag}\n"))?;
    }
    let size_bytes = std::fs::metadata(&path)?.len();
    Ok(PullResult {
        model: model.to_owned(),
        path,
        size_bytes,
        refreshed,
    })
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
    Ok(pull_model(model, &cache_dir, false)?.path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hf_uris() {
        let hf = parse_hf_uri(DEFAULT_EMBED_MODEL).unwrap();
        assert_eq!(hf.repo, "ggml-org/embeddinggemma-300M-GGUF");
        assert_eq!(hf.file, "embeddinggemma-300M-Q8_0.gguf");
        assert!(hf.url().ends_with(
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
