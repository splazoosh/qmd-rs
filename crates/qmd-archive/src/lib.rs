//! A content store for documents the user has already been served.
//!
//! The load-bearing property is not the copy: it is that an archived item has
//! a hash that reproduces. A hosted page does not — it is re-rendered, wrapped
//! in the fetching machine's locale, or decorated with rotating tokens, so the
//! hash of a response records the fetch and not the document. So an item stores
//! two hashes: the raw response's, explicitly not expected to reproduce, and
//! the hash of an *extract the document determines*, which is the one a
//! citation uses.
//!
//! Nothing here crawls, follows links, refreshes in the background or runs
//! unattended. Archiving is one document, on request. Nothing is published from
//! the store, and an item records the terms it arrived under so that free to
//! read is never mistaken for free to redistribute.
//!
//! Fetching obeys a fixed set of access rules; see [`fetch`] for how each rule
//! is expressed in code, and [`error::Refusal`] for the reads this tool
//! declines to make.

pub mod error;
pub mod extract;
pub mod fetch;
pub mod gate;
pub mod item;
pub mod manifest;
pub mod robots;
pub mod store;
pub mod terms;

pub use error::{Error, Refusal, Result};
pub use store::{Archive, Outcome};
