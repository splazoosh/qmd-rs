# Changelog

## 0.3.0

- `query --explain` shows how each result got its rank: its rank, weight,
  score and RRF share in each BM25 and vector list, the fused score and rank,
  and the final score, with the rerank blend when `--rerank` ran. `cli` format
  prints it under each result, `json` adds an `explain` object; the MCP
  `query` tool and `POST /query` take `explain: true`. It changes no ranking
  or score.
- New collection commands: `collection rename <old> <new>` (no re-index, no
  re-embed), `collection show <name>`, `collection update-cmd <name> [cmd]`,
  `collection include <name>` and `collection exclude <name>`.
- `qmd update` runs each collection's `update` command in its directory before
  indexing it, through `sh -c` (`cmd /C` on Windows); a failing command stops
  the update with its status. Under a project-local `.qmd` config it runs only
  with `QMD_TRUST_LOCAL_CONFIG=1`.
- `qmd cleanup [--dry-run]` clears cached rerank scores, deletes vectors no
  active document uses, inactive document rows and unreferenced content, then
  compacts FTS and vacuums. Nothing needs re-embedding afterwards.
- `collection show` and `cleanup --dry-run` are read-only commands for the
  closed-stdout exit status of 0.2.2.

## 0.2.2

- **Narrows 0.2.1:** when the reader of its output goes away, `qmd` exits with
  status 0 only for a read-only command: `status`, `drift`, `search`,
  `vsearch`, `query`, `get`, `ls`, `collection list`, `archive list`,
  `archive show`, `archive verify`, and `mcp` served in the foreground. Every
  other command (`init`, `update`, `embed`, `pull`, `collection add`,
  `collection remove`, `archive <url>`, `archive index`, `mcp --daemon`,
  `mcp stop`) exits 141, as a Unix tool killed by SIGPIPE does, so a pipeline
  can tell that its work stopped partway. No panic message either way.

## 0.2.1

- `qmd` exits quietly with status 0 when the reader of its output goes away
  (`qmd status | head`), instead of panicking in `println!`.
- Two changes shipped in 0.2.0 that its notes left out:
  - The `qmd/prebuilt-ggml` feature links a static CUDA ggml found under
    `CMAKE_PREFIX_PATH` instead of building ggml (README, Build).
  - CI builds and tests on Linux and Windows (CPU build); the lint job runs
    beside it.

## 0.2.0

- **Default changed:** BM25 terms now match if any is present (`--match any`);
  `--match all` or `search.match: all` restores every-term matching, the
  TypeScript behaviour. It applies to `search`, `query` and every `lex:` line.
  No re-embed is needed.
- `query --rerank` reranks hybrid results with Qwen3-Reranker-0.6B (opt-in;
  `search.rerank`, MCP `rerank`); scores are cached per chunk.
- `collection add --chunk-tokens N` / `chunkTokens` sets a collection's chunk
  size (default 900); changing it re-embeds that collection.
- `collection add --chunk-boundary heading` / `chunkBoundary: heading` starts a
  collection's chunks at headings (opt-in; the default `window` is unchanged);
  changing it re-embeds that collection.
- Model downloads use the `hf` CLI when it is on `PATH`, and otherwise send the
  Hugging Face token when one is set; `qmd status` reports which.
- CI adds a lint job: `cargo fmt --check` and `cargo clippy -D warnings`.

## 0.1.0

First release. `qmd` indexes directories of markdown into one SQLite file and
searches them with BM25, vector search over a local EmbeddingGemma model, or
both fused by reciprocal rank fusion, matching qmd v2.6.3's
`query --no-rerank`. It reads and writes the TypeScript implementation's index
schema, serves the index over MCP on stdio or from one shared HTTP process,
and mirrors served documents with their provenance through `qmd archive`.
