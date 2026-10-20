# Provenance

Where every file in this repository, and every data table its source
declares, comes from, and the licence it is under. Nothing listed as not
shipped is in this repository; it is fetched by Cargo or by `qmd` itself.

| Item | Origin | Licence |
|---|---|---|
| `crates/` | Written for this repository. The design, algorithms, SQLite schema and prompt formats are ported from qmd v2.6.3 (https://github.com/tobi/qmd). | MIT (`LICENSE`); the port carries the upstream notice in `LICENSE-UPSTREAM` |
| `EXCLUDE_DIRS` | The directories qmd v2.6.3 never indexes | MIT (`LICENSE-UPSTREAM`) |
| `EVAL_QUERIES` | The queries and expected documents of qmd v2.6.3's evaluation tests | MIT (`LICENSE-UPSTREAM`) |
| `SKIP`, `BLOCK`, `KEPT_HEADERS`, `MARKERS`, `DEVICES`, `OPEN_MARKERS`, `RESTRICTIVE_MARKERS`, `NEGATIONS`, `DOCUMENT_TYPES`, `A`, `B` | Written for this repository | MIT (`LICENSE`) |
| `crates/qmd-archive/fixtures/` | Written for this repository: synthetic pages | MIT (`LICENSE`) |
| `fixtures/eval-docs/`, `fixtures/eval-fixture.json` | Copied unchanged from qmd v2.6.3: `test/eval-docs/` and `src/bench/fixtures/example.json` | MIT (`LICENSE-UPSTREAM`) |
| `Cargo.toml`, `Cargo.lock` | Generated for this repository | MIT (`LICENSE`) |
| `README.md`, `CHANGELOG.md`, `PROVENANCE.md`, `LICENSE`, `.gitignore` | Written for this repository | MIT (`LICENSE`) |
| `LICENSE-UPSTREAM` | qmd's `LICENSE`, unchanged | MIT |
| Rust dependencies (not shipped) | Fetched from crates.io by Cargo, at the versions `Cargo.lock` pins | Each crate's own licence, as its package states |
| llama.cpp (not shipped) | Compiled from source by the `llama-cpp-sys-2` crate at build time | MIT |
| Embedding model (not shipped) | EmbeddingGemma-300M, downloaded by `qmd` on first use from https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF | Gemma Terms of Use, which the user accepts by downloading it; not redistributed here |
