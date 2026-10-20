//! Shared eval corpus and query set for the evaluation gates, ported from
//! TS `test/eval-bm25.test.ts` / `test/eval.test.ts`.

#![allow(dead_code)]

use std::path::PathBuf;

use qmd::Store;
use qmd::store::{hash_content, now_iso};

pub struct EvalQuery {
    pub query: &'static str,
    pub expected_doc: &'static str,
    pub difficulty: &'static str,
}

const fn q(query: &'static str, expected_doc: &'static str, difficulty: &'static str) -> EvalQuery {
    EvalQuery {
        query,
        expected_doc,
        difficulty,
    }
}

pub const EVAL_QUERIES: &[EvalQuery] = &[
    q("API versioning", "api-design", "easy"),
    q("Series A fundraising", "fundraising", "easy"),
    q("CAP theorem", "distributed-systems", "easy"),
    q("overfitting machine learning", "machine-learning", "easy"),
    q("remote work VPN", "remote-work", "easy"),
    q("Project Phoenix retrospective", "product-launch", "easy"),
    q("how to structure REST endpoints", "api-design", "medium"),
    q("raising money for startup", "fundraising", "medium"),
    q(
        "consistency vs availability tradeoffs",
        "distributed-systems",
        "medium",
    ),
    q(
        "how to prevent models from memorizing data",
        "machine-learning",
        "medium",
    ),
    q("working from home guidelines", "remote-work", "medium"),
    q(
        "what went wrong with the launch",
        "product-launch",
        "medium",
    ),
    q("nouns not verbs", "api-design", "hard"),
    q("Sequoia investor pitch", "fundraising", "hard"),
    q(
        "Raft algorithm leader election",
        "distributed-systems",
        "hard",
    ),
    q("F1 score precision recall", "machine-learning", "hard"),
    q("quarterly team gathering travel", "remote-work", "hard"),
    q("beta program 47 bugs", "product-launch", "hard"),
    q(
        "how much runway before running out of money",
        "fundraising",
        "fusion",
    ),
    q(
        "datacenter replication sync strategy",
        "distributed-systems",
        "fusion",
    ),
    q(
        "splitting data for training and testing",
        "machine-learning",
        "fusion",
    ),
    q("JSON response codes error messages", "api-design", "fusion"),
    q(
        "video calls camera async messaging",
        "remote-work",
        "fusion",
    ),
    q(
        "CI/CD pipeline testing coverage",
        "product-launch",
        "fusion",
    ),
];

pub fn by_difficulty(difficulty: &str) -> Vec<&'static EvalQuery> {
    EVAL_QUERIES
        .iter()
        .filter(|q| q.difficulty == difficulty)
        .collect()
}

pub fn eval_docs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/eval-docs")
        .canonicalize()
        .expect("fixtures/eval-docs must exist")
}

/// Open an in-memory store with the 6-document eval corpus indexed.
pub fn indexed_store() -> Store {
    let store = Store::open_in_memory().expect("open store");
    let dir = eval_docs_dir();
    let now = now_iso();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("read eval-docs")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "md"))
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    assert_eq!(entries.len(), 6, "eval corpus must have 6 documents");

    for entry in entries {
        let file = entry.file_name().to_string_lossy().into_owned();
        let content = std::fs::read_to_string(entry.path()).expect("read doc");
        let title = qmd::store::extract_title(&content, &file);
        let hash = hash_content(&content);
        store.insert_content(&hash, &content, &now).unwrap();
        store
            .insert_document("eval-docs", &file, &title, &hash, &now, &now)
            .unwrap();
    }
    store
}
