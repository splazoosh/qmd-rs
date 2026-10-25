//! How a model pull reaches Hugging Face: the `hf` CLI when one is installed,
//! otherwise qmd's own HTTP download, authenticated with the user's Hugging
//! Face token when one is found.
//!
//! The token is looked for where `huggingface_hub` looks: `HF_TOKEN`, then
//! `HUGGING_FACE_HUB_TOKEN`, then the file at `HF_TOKEN_PATH`, else
//! `$HF_HOME/token` (default `~/.cache/huggingface/token`, which `hf auth
//! login` writes). It is sent only to the endpoint in use (`HF_ENDPOINT` or
//! `https://huggingface.co`) and never to another origin a redirect leads to,
//! such as the CDN that serves file contents through a signed URL.

use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use url::Url;

/// Default Hugging Face endpoint, overridden by `HF_ENDPOINT`.
pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// CLI names, in order of preference: `hf`, then the older `huggingface-cli`.
const CLI_NAMES: [&str; 2] = ["hf", "huggingface-cli"];

/// Redirect hops followed before a request gives up.
const MAX_REDIRECTS: usize = 10;

/// A Hugging Face access token. Its `Debug` output never shows the value.
#[derive(Clone, PartialEq, Eq)]
pub struct HfToken(String);

impl HfToken {
    /// Strip line breaks and surrounding whitespace, as `huggingface_hub`
    /// does; an empty result is no token.
    fn clean(raw: &str) -> Option<Self> {
        let token = raw.replace(['\r', '\n'], "");
        let token = token.trim();
        (!token.is_empty()).then(|| Self(token.to_owned()))
    }

    #[cfg(test)]
    pub(crate) fn from_test(token: &str) -> Self {
        Self(token.to_owned())
    }

    /// Replace every occurrence of the token in `text`.
    pub fn redact(&self, text: &str) -> String {
        text.replace(&self.0, "<redacted>")
    }
}

impl fmt::Debug for HfToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("HfToken(<redacted>)")
    }
}

/// Where a token was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenSource {
    /// An environment variable, by name.
    Env(&'static str),
    /// A token file.
    File(PathBuf),
}

impl fmt::Display for TokenSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Env(name) => f.write_str(name),
            Self::File(path) => write!(f, "{}", path.display()),
        }
    }
}

/// How a pull fetches a model file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    /// The Hugging Face CLI at this path.
    Cli(PathBuf),
    /// qmd's own download; `token` says whether it authenticates.
    Builtin { token: bool },
}

impl fmt::Display for Route {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cli(path) => write!(f, "hf CLI ({})", path.display()),
            Self::Builtin { token: true } => f.write_str("built-in download with token"),
            Self::Builtin { token: false } => f.write_str("built-in download"),
        }
    }
}

/// What a pull would use: the endpoint, the CLI if one is installed, and the
/// token if one is set.
#[derive(Debug, Clone)]
pub struct HfSetup {
    pub endpoint: String,
    pub cli: Option<PathBuf>,
    pub token: Option<(HfToken, TokenSource)>,
}

impl HfSetup {
    /// Read the setup from this process's environment and `PATH`.
    pub fn from_env() -> Self {
        Self::from_vars(|name| std::env::var_os(name))
    }

    /// Read the setup through `var`, an environment lookup.
    pub fn from_vars(var: impl Fn(&str) -> Option<OsString>) -> Self {
        let endpoint = var("HF_ENDPOINT")
            .and_then(|e| e.into_string().ok())
            .filter(|e| !e.trim().is_empty())
            .map_or_else(
                || DEFAULT_ENDPOINT.to_owned(),
                |e| e.trim_end_matches('/').to_owned(),
            );
        Self {
            endpoint,
            cli: find_cli(var("PATH"), var("PATHEXT")),
            token: find_token(&var),
        }
    }

    /// The route the next download takes first.
    pub fn route(&self) -> Route {
        match &self.cli {
            Some(cli) => Route::Cli(cli.clone()),
            None => self.builtin_route(),
        }
    }

    pub(crate) fn builtin_route(&self) -> Route {
        Route::Builtin {
            token: self.token.is_some(),
        }
    }

    /// One line naming where the token came from, never the token.
    pub fn token_status(&self) -> String {
        match &self.token {
            Some((_, source)) => format!("found ({source})"),
            None => "none".to_owned(),
        }
    }

    /// Hide the token in text that may be shown to the user.
    pub(crate) fn redact(&self, text: &str) -> String {
        match &self.token {
            Some((token, _)) => token.redact(text),
            None => text.to_owned(),
        }
    }
}

fn find_token(var: &impl Fn(&str) -> Option<OsString>) -> Option<(HfToken, TokenSource)> {
    for name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Some(token) = var(name).and_then(|v| HfToken::clean(&v.to_string_lossy())) {
            return Some((token, TokenSource::Env(name)));
        }
    }
    let path = match var("HF_TOKEN_PATH").filter(|p| !p.is_empty()) {
        Some(path) => expand_home(PathBuf::from(path), var),
        None => {
            let home = match var("HF_HOME").filter(|p| !p.is_empty()) {
                Some(home) => expand_home(PathBuf::from(home), var),
                None => var("XDG_CACHE_HOME")
                    .filter(|p| !p.is_empty())
                    .map(PathBuf::from)
                    .or_else(|| home_dir(var).map(|h| h.join(".cache")))?
                    .join("huggingface"),
            };
            home.join("token")
        }
    };
    let token = HfToken::clean(&std::fs::read_to_string(&path).ok()?)?;
    Some((token, TokenSource::File(path)))
}

fn home_dir(var: &impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    var("HOME")
        .or_else(|| var("USERPROFILE"))
        .filter(|h| !h.is_empty())
        .map(PathBuf::from)
}

/// Expand a leading `~`, as `huggingface_hub` does for these variables.
fn expand_home(path: PathBuf, var: &impl Fn(&str) -> Option<OsString>) -> PathBuf {
    match (path.strip_prefix("~"), home_dir(var)) {
        (Ok(rest), Some(home)) => home.join(rest),
        _ => path,
    }
}

/// The first `hf` on `path`, else the first `huggingface-cli`. On Windows
/// each name is tried with every `PATHEXT` extension.
fn find_cli(path: Option<OsString>, pathext: Option<OsString>) -> Option<PathBuf> {
    let dirs: Vec<PathBuf> = std::env::split_paths(&path?).collect();
    let exts: Vec<String> = if cfg!(windows) {
        pathext
            .and_then(|e| e.into_string().ok())
            .unwrap_or_else(|| ".COM;.EXE;.BAT;.CMD".to_owned())
            .split(';')
            .filter(|e| !e.is_empty())
            .map(str::to_ascii_lowercase)
            .collect()
    } else {
        vec![String::new()]
    };
    CLI_NAMES.iter().find_map(|name| {
        dirs.iter().find_map(|dir| {
            exts.iter()
                .map(|ext| dir.join(format!("{name}{ext}")))
                .find(|candidate| candidate.is_file())
        })
    })
}

/// Download `repo`'s `file` through the CLI into `local_dir`, where it lands
/// at `local_dir/file`. The CLI reads the token itself; it is never passed on
/// the command line.
///
/// # Errors
/// Returns why the CLI could not run or what it reported, with the token
/// redacted.
pub(crate) fn cli_download(
    setup: &HfSetup,
    cli: &Path,
    repo: &str,
    file: &str,
    local_dir: &Path,
    force: bool,
) -> Result<(), String> {
    let mut command = Command::new(cli);
    command
        .args(["download", repo, file, "--local-dir"])
        .arg(local_dir)
        .env("HF_ENDPOINT", &setup.endpoint)
        .stdin(Stdio::null());
    if force {
        command.arg("--force-download");
    }
    let output = command
        .output()
        .map_err(|e| format!("could not run: {e}"))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let tail: Vec<&str> = stderr.trim().lines().rev().take(3).collect();
    let tail: Vec<&str> = tail.into_iter().rev().collect();
    Err(setup.redact(&format!("{}: {}", output.status, tail.join(" | "))))
}

/// Origin of the endpoint, the only one the token is sent to.
fn endpoint_origin(setup: &HfSetup) -> Option<url::Origin> {
    Url::parse(&setup.endpoint).ok().map(|u| u.origin())
}

/// Send a GET (or HEAD) to `url`, following redirects by hand so the token
/// goes only to the endpoint's origin and is dropped for any other.
///
/// # Errors
/// Returns a transport or HTTP-status message, with the token redacted.
pub(crate) fn call(
    setup: &HfSetup,
    head: bool,
    url: &str,
) -> Result<ureq::http::Response<ureq::Body>, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .max_redirects(0)
        .build()
        .into();
    let origin = endpoint_origin(setup);
    let mut current = Url::parse(url).map_err(|e| format!("bad URL {url}: {e}"))?;
    for _ in 0..=MAX_REDIRECTS {
        let mut request = if head {
            agent.head(current.as_str())
        } else {
            agent.get(current.as_str())
        };
        let mut authenticated = false;
        if let Some((token, _)) = &setup.token
            && origin.as_ref() == Some(&current.origin())
        {
            request = request.header("Authorization", format!("Bearer {}", token.0));
            authenticated = true;
        }
        let response = request.call().map_err(|e| {
            let hint = match e {
                ureq::Error::StatusCode(401 | 403) if authenticated => {
                    " (the Hugging Face token was sent; check that it is valid and has access)"
                }
                _ => "",
            };
            setup.redact(&format!("{e}{hint}"))
        })?;
        if !response.status().is_redirection() {
            return Ok(response);
        }
        let location = response
            .headers()
            .get("location")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| format!("HTTP {} without a Location header", response.status()))?;
        current = current
            .join(location)
            .map_err(|e| format!("bad redirect target: {e}"))?;
    }
    Err(format!("more than {MAX_REDIRECTS} redirects"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn vars(pairs: &[(&str, &OsString)]) -> impl Fn(&str) -> Option<OsString> {
        let map: HashMap<String, OsString> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).clone()))
            .collect();
        move |name| map.get(name).cloned()
    }

    #[test]
    fn the_token_comes_from_hf_token_first_then_the_token_file() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("hf-home");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(home.join("token"), "hf_fromfile\r\n").unwrap();
        let home = home.into_os_string();
        let env_token = OsString::from("  hf_fromenv\n");
        let empty = OsString::new();

        let setup = HfSetup::from_vars(vars(&[("HF_TOKEN", &env_token), ("HF_HOME", &home)]));
        let (token, source) = setup.token.clone().unwrap();
        assert_eq!(token, HfToken("hf_fromenv".to_owned()));
        assert_eq!(source, TokenSource::Env("HF_TOKEN"));

        let setup = HfSetup::from_vars(vars(&[("HF_TOKEN", &empty), ("HF_HOME", &home)]));
        let (token, source) = setup.token.clone().unwrap();
        assert_eq!(token, HfToken("hf_fromfile".to_owned()));
        assert!(matches!(source, TokenSource::File(_)));
        assert!(setup.token_status().starts_with("found ("));
        assert!(!setup.token_status().contains("hf_fromfile"));
    }

    #[test]
    fn hf_token_path_replaces_hf_home_and_the_default_home_is_under_cache() {
        let dir = tempfile::tempdir().unwrap();
        let explicit = dir.path().join("elsewhere");
        std::fs::write(&explicit, "hf_explicit").unwrap();
        let default = dir.path().join(".cache").join("huggingface");
        std::fs::create_dir_all(&default).unwrap();
        std::fs::write(default.join("token"), "hf_default").unwrap();
        let home = dir.path().as_os_str().to_owned();
        let explicit = explicit.into_os_string();

        let setup = HfSetup::from_vars(vars(&[("HF_TOKEN_PATH", &explicit), ("HOME", &home)]));
        assert_eq!(setup.token.unwrap().0, HfToken("hf_explicit".to_owned()));

        let setup = HfSetup::from_vars(vars(&[("HOME", &home)]));
        assert_eq!(
            setup.token.clone().unwrap().0,
            HfToken("hf_default".to_owned())
        );
        assert_eq!(setup.route(), Route::Builtin { token: true });

        let nowhere = dir.path().join("none").into_os_string();
        let setup = HfSetup::from_vars(vars(&[("HOME", &nowhere)]));
        assert!(setup.token.is_none());
        assert_eq!(setup.route(), Route::Builtin { token: false });
        assert_eq!(setup.endpoint, DEFAULT_ENDPOINT);
    }

    #[test]
    fn a_cli_on_path_is_found_preferring_hf() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a"), dir.path().join("b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        let ext = if cfg!(windows) { ".exe" } else { "" };
        std::fs::write(a.join(format!("huggingface-cli{ext}")), "").unwrap();
        let path = std::env::join_paths([&a, &b]).unwrap();
        let empty_home = dir.path().join("home").into_os_string();

        let setup = HfSetup::from_vars(vars(&[("PATH", &path), ("HOME", &empty_home)]));
        assert_eq!(setup.cli, Some(a.join(format!("huggingface-cli{ext}"))));

        std::fs::write(b.join(format!("hf{ext}")), "").unwrap();
        let setup = HfSetup::from_vars(vars(&[("PATH", &path), ("HOME", &empty_home)]));
        let hf = b.join(format!("hf{ext}"));
        assert_eq!(setup.route(), Route::Cli(hf.clone()));
        assert_eq!(setup.cli, Some(hf));
    }

    #[test]
    fn debug_output_never_shows_the_token() {
        let setup = HfSetup {
            endpoint: DEFAULT_ENDPOINT.to_owned(),
            cli: None,
            token: Some((
                HfToken("hf_secret".to_owned()),
                TokenSource::Env("HF_TOKEN"),
            )),
        };
        assert!(!format!("{setup:?}").contains("hf_secret"));
        assert_eq!(setup.redact("bad hf_secret here"), "bad <redacted> here");
    }
}
