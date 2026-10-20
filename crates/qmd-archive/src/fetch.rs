//! Reading a document from a host, under fixed access rules.
//!
//! Each rule shows up here as code rather than as a comment:
//!
//! 1. **Identify honestly.** [`user_agent`] names the read and, when whoever
//!    runs it has configured one, a URL that reaches them. There is no browser
//!    impersonation to turn on, and no option to change the product token.
//! 2. **Prefer the machine interface.** Not enforceable in a fetcher — it is a
//!    choice about which URL is handed in — but the manifest records the URL,
//!    so the choice is visible afterwards.
//! 3. **Render, do not spoof.** There is nothing here whose selling point is
//!    not being detected: no stealth options, no proxy list, no header forgery.
//!    A browser renderer is [not implemented](FetchMethod::Browser).
//! 4. **Stop at a human gate.** [`human_gate`] recognises a challenge and turns
//!    it into a refusal that is recorded, not retried.
//! 5. **One request at a time, with a deliberate delay.** Held by
//!    [`crate::gate`], which persists across processes. A 429 is never retried:
//!    persistence is the harm.
//! 6. **An account is the user's act.** Nothing here authenticates, and no
//!    credential is read from anywhere.

use std::collections::BTreeMap;

use ureq::ResponseExt;

use crate::error::Refusal;

/// The product token. Fixed on purpose — an archive that can be told to call
/// itself something else is one configuration change from impersonation.
pub const PRODUCT: &str = "qmd-archive";

/// Refuse a response larger than this rather than store a truncated one.
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

/// A host that has not answered by then is not going to.
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// The whole request, body included. Long enough for the largest body at a
/// slow line; short enough that a stalled host does not hold a run forever.
const GLOBAL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// The most redirects one archive call follows. Each is a separate request,
/// checked against robots.txt and spaced by the gate like any other.
pub const MAX_REDIRECTS: usize = 5;

/// Response headers worth keeping. Provenance, cache validators, and the two
/// that carry terms.
const KEPT_HEADERS: &[&str] = &[
    "content-type",
    "content-length",
    "content-language",
    "etag",
    "last-modified",
    "date",
    "server",
    "link",
    "x-robots-tag",
    "cf-mitigated",
    "retry-after",
    "location",
];

/// How a document was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FetchMethod {
    /// An ordinary HTTP request.
    Request,
    /// A stock browser engine. **Not implemented**: qmd drives no browser,
    /// and access rule 3 permits only a stock engine, never a patched one. The
    /// variant exists so an item fetched that way later is distinguishable from
    /// one fetched now, rather than being silently conflated with it.
    Browser,
}

/// One response, reduced to what an item records.
#[derive(Debug, Clone)]
pub struct Response {
    /// The URL the bytes actually came from, after any redirects.
    pub final_url: String,
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    pub fn content_type(&self) -> Option<&str> {
        self.header("content-type")
    }

    /// Where a redirect points, resolved against the URL that was asked for.
    /// `None` for anything that is not a redirect.
    pub fn redirect_target(&self, requested: &str) -> Option<String> {
        if !matches!(self.status, 301 | 302 | 303 | 307 | 308) {
            return None;
        }
        let base = url::Url::parse(requested).ok()?;
        Some(base.join(self.header("location")?).ok()?.to_string())
    }
}

/// The seam the tests fetch through. Nothing in the crate reaches the network
/// except an implementation of this trait.
pub trait Fetcher {
    /// Perform one GET. Implementations do not retry, do not follow redirects
    /// (a 3xx is returned as the response), and do not sleep.
    fn get(&self, url: &str, user_agent: &str) -> Result<Response, Refusal>;
}

/// The user-agent sent with every request: what this is, its version, and — if
/// one is configured — a URL that reaches whoever runs it. There is no default
/// contact: a shipped default would name someone else's site.
pub fn user_agent(contact: Option<&str>) -> String {
    let version = env!("CARGO_PKG_VERSION");
    match contact {
        Some(contact) => format!("{PRODUCT}/{version} (+{contact})"),
        None => format!("{PRODUCT}/{version}"),
    }
}

/// A `ureq`-backed fetcher.
#[derive(Debug, Clone, Default)]
pub struct HttpFetcher;

impl Fetcher for HttpFetcher {
    fn get(&self, url: &str, user_agent: &str) -> Result<Response, Refusal> {
        let response = ureq::get(url)
            .header("user-agent", user_agent)
            .header("accept", "text/html,application/xhtml+xml,text/plain,*/*")
            .config()
            .http_status_as_error(false)
            .max_redirects(0)
            .timeout_connect(Some(CONNECT_TIMEOUT))
            .timeout_global(Some(GLOBAL_TIMEOUT))
            .build()
            .call()
            .map_err(|error| Refusal::Transport {
                url: url.to_owned(),
                message: error.to_string(),
            })?;

        let status = response.status().as_u16();
        let final_url = response.get_uri().to_string();

        let headers = response
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                let name = name.as_str().to_ascii_lowercase();
                KEPT_HEADERS
                    .contains(&name.as_str())
                    .then(|| value.to_str().ok().map(|v| (name, v.to_owned())))
                    .flatten()
            })
            .collect::<Vec<_>>();

        let body = response
            .into_body()
            .with_config()
            .limit(MAX_BODY_BYTES as u64)
            .read_to_vec()
            .map_err(|error| Refusal::Transport {
                url: url.to_owned(),
                message: error.to_string(),
            })?;

        Ok(Response {
            final_url,
            status,
            headers,
            body,
        })
    }
}

/// Whether a response is a challenge asking for a person.
///
/// A challenge is not a failure to be worked around; it is the site saying it
/// wants a human. The caller records the evidence and stops.
pub fn human_gate(response: &Response) -> Option<String> {
    if let Some(value) = response.header("cf-mitigated") {
        return Some(format!("cf-mitigated: {value}"));
    }

    let head_len = response.body.len().min(4096);
    let head = String::from_utf8_lossy(&response.body[..head_len]).to_ascii_lowercase();
    const MARKERS: &[&str] = &[
        "just a moment",
        "checking your browser",
        "enable javascript and cookies to continue",
        "captcha",
        "hcaptcha",
        "g-recaptcha",
        "cf-challenge",
        "/cdn-cgi/challenge-platform",
        "are you a robot",
        "unusual traffic from your computer network",
    ];
    let marker = MARKERS.iter().find(|marker| head.contains(**marker))?;

    if response.status == 200 && response.body.len() > 64 * 1024 {
        return None;
    }
    Some(format!("body contains {marker:?}"))
}

/// Parse `Retry-After`, which may be seconds or an HTTP date.
pub fn retry_after_secs(value: &str, now: chrono::DateTime<chrono::Utc>) -> Option<u64> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(seconds);
    }
    let when = chrono::DateTime::parse_from_rfc2822(value).ok()?;
    let delta = when.with_timezone(&chrono::Utc) - now;
    Some(delta.num_seconds().max(0) as u64)
}

/// Turn a rate-limiting response into the refusal it earns. A 429 is the one
/// response where trying again is the harm, so this never suggests one.
pub fn rate_limit_refusal(host: &str, status: u16) -> Option<Refusal> {
    matches!(status, 429 | 503).then(|| Refusal::RateLimited {
        host: host.to_owned(),
        status,
    })
}

/// A canned fetcher for tests: URL to response, and a record of what was asked
/// for. Never touches the network.
#[derive(Debug, Default)]
pub struct StubFetcher {
    responses: std::sync::Mutex<BTreeMap<String, Queue>>,
    pub requests: std::sync::Mutex<Vec<(String, String)>>,
}

/// Responses for one URL, served in the order they were queued. Once the queue
/// runs out the last one repeats, so a test only queues what it cares about.
#[derive(Debug, Default)]
struct Queue {
    responses: Vec<Result<Response, Refusal>>,
    served: usize,
}

impl StubFetcher {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue one response for `url`; successive calls take them in order.
    pub fn push(&self, url: &str, response: Result<Response, Refusal>) {
        self.responses
            .lock()
            .unwrap()
            .entry(url.to_owned())
            .or_default()
            .responses
            .push(response);
    }

    /// Queue an ordinary 200 with a body.
    pub fn push_ok(&self, url: &str, content_type: &str, body: &[u8]) {
        self.push(
            url,
            Ok(Response {
                final_url: url.to_owned(),
                status: 200,
                headers: vec![("content-type".to_owned(), content_type.to_owned())],
                body: body.to_vec(),
            }),
        );
    }

    pub fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}

impl Fetcher for StubFetcher {
    fn get(&self, url: &str, user_agent: &str) -> Result<Response, Refusal> {
        self.requests
            .lock()
            .unwrap()
            .push((url.to_owned(), user_agent.to_owned()));
        let mut responses = self.responses.lock().unwrap();
        let missing = || Refusal::Status {
            url: url.to_owned(),
            status: 404,
        };
        let queue = responses.get_mut(url).ok_or_else(missing)?;
        let index = queue.served.min(queue.responses.len().saturating_sub(1));
        queue.served += 1;
        queue
            .responses
            .get(index)
            .cloned()
            .unwrap_or_else(|| Err(missing()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(status: u16, body: &[u8], headers: &[(&str, &str)]) -> Response {
        Response {
            final_url: "https://example.org/x".to_owned(),
            status,
            headers: headers
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect(),
            body: body.to_vec(),
        }
    }

    #[test]
    fn the_user_agent_names_the_read_and_carries_a_url_not_a_mailbox() {
        let agent = user_agent(Some("https://example.invalid/"));
        assert_eq!(
            agent,
            format!(
                "qmd-archive/{} (+https://example.invalid/)",
                env!("CARGO_PKG_VERSION")
            )
        );
        assert!(
            !agent.contains('@'),
            "a mailbox identifies a person: {agent}"
        );
        assert!(
            !agent.to_ascii_lowercase().contains("mozilla"),
            "never impersonate a browser we are not running: {agent}"
        );
    }

    #[test]
    fn without_a_contact_the_user_agent_names_only_the_read() {
        assert_eq!(
            user_agent(None),
            format!("qmd-archive/{}", env!("CARGO_PKG_VERSION"))
        );
    }

    #[test]
    fn a_challenge_page_is_recognised_as_a_human_gate() {
        let gate = human_gate(&response(
            403,
            b"<html><title>Just a moment...</title></html>",
            &[],
        ));
        assert!(gate.is_some());

        let header_gate = human_gate(&response(403, b"denied", &[("cf-mitigated", "challenge")]));
        assert_eq!(header_gate.as_deref(), Some("cf-mitigated: challenge"));
    }

    #[test]
    fn a_page_about_captchas_is_not_a_captcha() {
        let mut body = b"<html><p>a captcha is an interactive challenge</p>".to_vec();
        body.resize(100 * 1024, b' ');
        assert!(human_gate(&response(200, &body, &[])).is_none());
    }

    #[test]
    fn rate_limiting_earns_a_refusal_and_never_a_retry() {
        assert!(matches!(
            rate_limit_refusal("example.org", 429),
            Some(Refusal::RateLimited { status: 429, .. })
        ));
        assert!(rate_limit_refusal("example.org", 404).is_none());
    }

    #[test]
    fn retry_after_reads_seconds_and_dates() {
        let now = chrono::DateTime::parse_from_rfc2822("Mon, 31 Aug 2026 12:00:00 GMT")
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert_eq!(retry_after_secs("600", now), Some(600));
        assert_eq!(
            retry_after_secs("Mon, 31 Aug 2026 12:10:00 GMT", now),
            Some(600)
        );
        assert_eq!(retry_after_secs("not a date", now), None);
    }
}
