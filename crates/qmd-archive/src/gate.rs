//! The one-request-at-a-time gate, kept on disk.
//!
//! Access rule 5 asks for a deliberate delay between reads of a host. A gate
//! held in memory would satisfy that only within a single process, and every
//! `qmd archive` is its own process — two invocations a second apart would each
//! see an empty gate and each believe itself the first. So the last read of
//! each host is written down, and the refusal is a refusal: the gate never
//! sleeps and retries on the caller's behalf.
//!
//! `Retry-After` is recorded the same way. A host that asked to be left alone
//! for ten minutes is left alone for ten minutes, across processes, whether or
//! not anyone remembers.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Refusal, Result};

/// Access rule 5's floor. A host may ask for more; it may not ask for less.
pub const MINIMUM_DELAY_SECS: u64 = 5;

/// The longest a `Retry-After` is held against a host: one week.
pub const MAX_RETRY_AFTER_SECS: u64 = 7 * 24 * 60 * 60;

const STATE_FILE: &str = "hosts.json";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct State {
    #[serde(default)]
    hosts: BTreeMap<String, HostState>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct HostState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_request: Option<DateTime<Utc>>,
    /// Set by a `Retry-After`: no read of this host before this instant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    not_before: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    not_before_reason: Option<String>,
}

/// What the gate says about reading a host right now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Clearance {
    Ready,
    /// The spacing rule 5 asks for has not elapsed. Waiting this out is the
    /// intended behaviour; the caller decides whether the wait is acceptable
    /// and refuses with [`Refusal::RateGate`] when it is not.
    WaitFor {
        seconds: u64,
        host: String,
        elapsed_secs: u64,
    },
}

/// Per-host request spacing, persisted under the archive root.
#[derive(Debug, Clone)]
pub struct Gate {
    path: PathBuf,
    minimum_secs: u64,
}

impl Gate {
    pub fn new(state_dir: &Path) -> Self {
        Self {
            path: state_dir.join(STATE_FILE),
            minimum_secs: MINIMUM_DELAY_SECS,
        }
    }

    /// Raise the floor — never lower it. A `Crawl-delay` longer than the floor
    /// is honoured; one shorter is ignored.
    pub fn with_host_delay(mut self, crawl_delay_secs: Option<u64>) -> Self {
        if let Some(delay) = crawl_delay_secs {
            self.minimum_secs = self.minimum_secs.max(delay);
        }
        self
    }

    pub fn minimum_secs(&self) -> u64 {
        self.minimum_secs
    }

    /// Whether `host` may be read at `now`.
    ///
    /// The two delays this returns are different in kind. Ordinary spacing is
    /// the deliberate delay rule 5 asks for, and waiting it out is the correct
    /// thing to do — [`Clearance::WaitFor`]. A `Retry-After` is a host saying
    /// go away, and waiting it out inside a run would be persistence through a
    /// block, so it is an error rather than a wait.
    pub fn clearance(&self, host: &str, now: DateTime<Utc>) -> Result<Clearance> {
        let state = self.load()?;
        let Some(entry) = state.hosts.get(host) else {
            return Ok(Clearance::Ready);
        };

        if let Some(not_before) = entry.not_before
            && now < not_before
        {
            let remaining = (not_before - now).num_seconds().max(0) as u64;
            return Err(Refusal::RetryAfter {
                host: host.to_owned(),
                retry_after_secs: remaining,
                remaining_secs: remaining,
            }
            .into());
        }

        if let Some(last) = entry.last_request {
            let elapsed = (now - last).num_seconds();
            if elapsed >= 0 && (elapsed as u64) < self.minimum_secs {
                return Ok(Clearance::WaitFor {
                    seconds: self.minimum_secs - elapsed as u64,
                    host: host.to_owned(),
                    elapsed_secs: elapsed as u64,
                });
            }
        }
        Ok(Clearance::Ready)
    }

    /// Record that a request to `host` was made. Called before the response is
    /// known, so a request that fails still counts against the gate.
    pub fn record_request(&self, host: &str, now: DateTime<Utc>) -> Result<()> {
        let mut state = self.load()?;
        state.hosts.entry(host.to_owned()).or_default().last_request = Some(now);
        self.save(&state)
    }

    /// Record a `Retry-After`. The host is not read again before it elapses.
    pub fn record_retry_after(
        &self,
        host: &str,
        now: DateTime<Utc>,
        seconds: u64,
        reason: &str,
    ) -> Result<()> {
        let mut state = self.load()?;
        let entry = state.hosts.entry(host.to_owned()).or_default();
        let seconds = seconds.min(MAX_RETRY_AFTER_SECS) as i64;
        let held = chrono::Duration::try_seconds(seconds)
            .and_then(|delta| now.checked_add_signed(delta))
            .unwrap_or(DateTime::<Utc>::MAX_UTC);
        entry.not_before = Some(held);
        entry.not_before_reason = Some(reason.to_owned());
        self.save(&state)
    }

    /// When `host` may next be read, if it is currently held off.
    pub fn held_until(&self, host: &str) -> Result<Option<DateTime<Utc>>> {
        Ok(self
            .load()?
            .hosts
            .get(host)
            .and_then(|entry| entry.not_before))
    }

    fn load(&self) -> Result<State> {
        match std::fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| Error::Manifest {
                path: self.path.clone(),
                message: e.to_string(),
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(State::default()),
            Err(source) => Err(Error::Io {
                path: self.path.clone(),
                source,
            }),
        }
    }

    fn save(&self, state: &State) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| Error::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let json = serde_json::to_vec_pretty(state).map_err(|e| Error::Other(e.to_string()))?;
        crate::store::write_file(&self.path, &json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Refusal;

    fn at(seconds: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_780_000_000 + seconds, 0).unwrap()
    }

    #[test]
    fn a_fresh_host_may_be_read() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            Gate::new(dir.path())
                .clearance("example.org", at(0))
                .unwrap(),
            Clearance::Ready
        );
    }

    #[test]
    fn the_gate_survives_the_process_that_set_it() {
        let dir = tempfile::tempdir().unwrap();
        Gate::new(dir.path())
            .record_request("example.org", at(0))
            .unwrap();

        let clearance = Gate::new(dir.path())
            .clearance("example.org", at(2))
            .unwrap();
        assert_eq!(
            clearance,
            Clearance::WaitFor {
                seconds: 3,
                host: "example.org".to_owned(),
                elapsed_secs: 2,
            }
        );
        assert_eq!(
            Gate::new(dir.path())
                .clearance("example.org", at(5))
                .unwrap(),
            Clearance::Ready
        );
    }

    #[test]
    fn one_host_does_not_gate_another() {
        let dir = tempfile::tempdir().unwrap();
        let gate = Gate::new(dir.path());
        gate.record_request("a.example", at(0)).unwrap();
        assert_eq!(
            gate.clearance("b.example", at(0)).unwrap(),
            Clearance::Ready
        );
    }

    #[test]
    fn retry_after_is_an_error_rather_than_a_wait() {
        let dir = tempfile::tempdir().unwrap();
        let gate = Gate::new(dir.path());
        gate.record_retry_after("example.org", at(0), 600, "HTTP 429")
            .unwrap();

        let err = gate.clearance("example.org", at(60)).unwrap_err();
        match err {
            Error::Refused(Refusal::RetryAfter { remaining_secs, .. }) => {
                assert_eq!(remaining_secs, 540);
            }
            other => panic!("{other}"),
        }
        assert_eq!(
            gate.clearance("example.org", at(601)).unwrap(),
            Clearance::Ready
        );
        assert_eq!(gate.held_until("example.org").unwrap(), Some(at(600)));
    }

    #[test]
    fn an_absurd_retry_after_is_clamped_rather_than_overflowing() {
        let dir = tempfile::tempdir().unwrap();
        let gate = Gate::new(dir.path());
        gate.record_retry_after("example.org", at(0), u64::MAX, "HTTP 429")
            .unwrap();
        assert_eq!(
            gate.held_until("example.org").unwrap(),
            Some(at(MAX_RETRY_AFTER_SECS as i64))
        );
    }

    #[test]
    fn a_host_may_ask_for_longer_but_not_shorter() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            Gate::new(dir.path())
                .with_host_delay(Some(30))
                .minimum_secs(),
            30
        );
        assert_eq!(
            Gate::new(dir.path())
                .with_host_delay(Some(1))
                .minimum_secs(),
            5
        );
        assert_eq!(
            Gate::new(dir.path()).with_host_delay(None).minimum_secs(),
            5
        );
    }
}
