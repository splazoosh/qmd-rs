//! Item identity: the URL a document was served from, and where it lands on
//! disk.
//!
//! The directory name is legible on purpose — `host/slug-abcd1234` — because a
//! search hit names a path and a reader should be able to tell what they are
//! looking at before opening the manifest. The eight hex characters are what
//! actually distinguishes two items; the slug is decoration and is allowed to
//! collide.

use sha2::{Digest, Sha256};
use url::Url;

use crate::error::{Error, Result};

/// A URL accepted for archiving, plus the disk identity derived from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemKey {
    /// The URL as normalised — this is what the manifest records as requested.
    pub url: String,
    /// `http` or `https`, kept so sibling reads of the same host (robots.txt)
    /// go over the scheme the host actually serves.
    pub scheme: String,
    pub host: String,
    /// The port, when it is not the scheme's default.
    pub port: Option<u16>,
    /// `scheme://host[:port]`: what a `robots.txt` governs (RFC 9309 §2.3).
    pub origin: String,
    /// Path with query, used for the robots.txt match.
    pub path_and_query: String,
    slug: String,
    digest: String,
}

impl ItemKey {
    /// Normalise `input` and derive its identity. Fragments are dropped (a
    /// fragment is never sent to the server, so it cannot distinguish two
    /// documents); everything else is left exactly as given, because a query
    /// string routinely selects the document.
    pub fn parse(input: &str) -> Result<Self> {
        let mut url = Url::parse(input).map_err(|e| Error::Url(format!("{input}: {e}")))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(Error::Url(format!("{input}: scheme is not http or https")));
        }
        url.set_fragment(None);
        let scheme = url.scheme().to_owned();
        let host = url
            .host_str()
            .ok_or_else(|| Error::Url(format!("{input}: no host")))?
            .to_ascii_lowercase();
        if host.trim_matches('.').is_empty() {
            return Err(Error::Url(format!("{input}: host is not a name")));
        }
        let port = url.port();
        let origin = url.origin().ascii_serialization();

        let mut path_and_query = url.path().to_owned();
        if let Some(q) = url.query() {
            path_and_query.push('?');
            path_and_query.push_str(q);
        }

        let url = url.to_string();
        let digest = hex::encode(Sha256::digest(url.as_bytes()));
        Ok(Self {
            slug: slugify(&path_and_query),
            digest: digest[..8].to_owned(),
            scheme,
            url,
            host,
            port,
            origin,
            path_and_query,
        })
    }

    /// `host/slug-abcd1234`, the item's path relative to the archive root.
    pub fn dir_name(&self) -> String {
        format!("{}/{}-{}", host_dir(&self.host), self.slug, self.digest)
    }

    /// The directory under `hosts/` holding this origin's `robots.txt`. The
    /// bare host for `https` on its default port, which is what every archive
    /// before per-origin robots used; `+http` and `+<port>` otherwise. `+`
    /// never appears in a [`host_dir`], so the names cannot collide.
    pub fn origin_dir(&self) -> String {
        let mut name = host_dir(&self.host);
        if self.scheme != "https" {
            name.push('+');
            name.push_str(&self.scheme);
        }
        if let Some(port) = self.port {
            name.push_str(&format!("+{port}"));
        }
        name
    }

    /// The URL's SHA-256, first eight hex characters.
    pub fn digest(&self) -> &str {
        &self.digest
    }
}

/// A host as one safe path component. An ordinary DNS name is left exactly as
/// it is, so existing archives keep their layout. Anything else — an IPv6
/// literal's brackets and colons, characters a filesystem refuses, a trailing
/// dot Windows would strip — is percent-encoded, and `%` never appears in a URL
/// host, so an encoded name cannot collide with a real one. A first label that
/// is a Windows device name (`nul.example.org`) has its first character encoded
/// so it names a directory rather than a device.
pub fn host_dir(host: &str) -> String {
    const DEVICES: &[&str] = &["con", "prn", "aux", "nul"];
    let first_label = host.split('.').next().unwrap_or("");
    let numbered = |prefix: &str| {
        first_label
            .strip_prefix(prefix)
            .is_some_and(|n| n.len() == 1 && n.as_bytes()[0].is_ascii_digit())
    };
    let device = DEVICES.contains(&first_label) || numbered("com") || numbered("lpt");

    let mut out = String::with_capacity(host.len());
    for (i, byte) in host.bytes().enumerate() {
        let plain = byte.is_ascii_lowercase()
            || byte.is_ascii_digit()
            || byte == b'-'
            || (byte == b'.' && i + 1 != host.len() && i != 0);
        if plain && !(device && i == 0) {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

/// The last meaningful path segment, reduced to something a filesystem and a
/// human both tolerate. Empty paths become `index`.
fn slugify(path_and_query: &str) -> String {
    let stem = path_and_query
        .split(['?', '#'])
        .next()
        .unwrap_or("")
        .trim_matches('/')
        .rsplit('/')
        .find(|s| !s.is_empty())
        .unwrap_or("index");

    let mut out = String::new();
    let mut last_dash = false;
    for ch in stem.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash && !out.is_empty() {
            out.push('-');
            last_dash = true;
        }
    }
    let trimmed = out.trim_matches('-');
    let capped: String = trimmed.chars().take(48).collect();
    if capped.is_empty() {
        "index".to_owned()
    } else {
        capped.trim_matches('-').to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dir_name_is_legible_and_stable() {
        let key = ItemKey::parse("http://www.zxdesign.info/book/theZXSpectrumULA.shtml").unwrap();
        assert_eq!(key.host, "www.zxdesign.info");
        assert!(
            key.dir_name()
                .starts_with("www.zxdesign.info/thezxspectrumula-shtml-"),
            "{}",
            key.dir_name()
        );
        let again = ItemKey::parse("http://www.zxdesign.info/book/theZXSpectrumULA.shtml").unwrap();
        assert_eq!(key.dir_name(), again.dir_name());
    }

    #[test]
    fn fragment_does_not_split_an_item_but_query_does() {
        let bare = ItemKey::parse("https://example.org/a/b").unwrap();
        let fragment = ItemKey::parse("https://example.org/a/b#section-3").unwrap();
        let query = ItemKey::parse("https://example.org/a/b?format=txt").unwrap();
        assert_eq!(bare.dir_name(), fragment.dir_name());
        assert_ne!(bare.dir_name(), query.dir_name());
        assert_eq!(query.path_and_query, "/a/b?format=txt");
    }

    #[test]
    fn host_case_is_normalised_and_scheme_is_checked() {
        assert_eq!(
            ItemKey::parse("https://EXAMPLE.org/x").unwrap().host,
            "example.org"
        );
        assert!(ItemKey::parse("file:///etc/passwd").is_err());
        assert!(ItemKey::parse("not a url").is_err());
    }

    #[test]
    fn a_host_never_leaves_the_archive_root_or_names_a_device() {
        for url in ["http://../x", "http://./x", "http://.../x"] {
            if let Ok(key) = ItemKey::parse(url) {
                let dir = key.dir_name();
                let first = dir.split('/').next().unwrap();
                assert!(first.trim_matches('.').len() == first.len(), "{url}: {dir}");
            }
        }
        assert_eq!(host_dir("nul.example.org"), "%6Eul.example.org");
        assert_eq!(host_dir("com1"), "%63om1");
        assert_eq!(host_dir("computer.example"), "computer.example");
        assert_eq!(host_dir("[::1]"), "%5B%3A%3A1%5D");
        assert_eq!(host_dir("example.org."), "example.org%2E");
        assert_eq!(host_dir("www.zxdesign.info"), "www.zxdesign.info");
        let v6 = ItemKey::parse("http://[::1]:8080/a").unwrap();
        assert!(!v6.dir_name().contains(':'), "{}", v6.dir_name());
    }

    #[test]
    fn robots_are_kept_per_origin_without_moving_the_https_default() {
        let origin = |url: &str| ItemKey::parse(url).unwrap().origin_dir();
        assert_eq!(origin("https://example.org/a"), "example.org");
        assert_eq!(origin("https://example.org:443/a"), "example.org");
        assert_eq!(origin("http://example.org/a"), "example.org+http");
        assert_eq!(origin("https://example.org:8443/a"), "example.org+8443");
        let key = ItemKey::parse("http://example.org:8080/a").unwrap();
        assert_eq!(key.origin, "http://example.org:8080");
    }

    #[test]
    fn a_bare_host_still_gets_a_directory() {
        let key = ItemKey::parse("https://example.org/").unwrap();
        assert!(key.dir_name().starts_with("example.org/index-"));
    }
}
