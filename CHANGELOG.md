# Changelog

## 0.4.0

- New `qmd context` commands: `context add [PATH] TEXT`, `context list`,
  `context rm PATH` and `context check`. PATH is `/` for the global context,
  `qmd://<collection>/<path>`, or a directory inside a collection; a
  directory two collections share is refused rather than guessed. A prefix
  covers whole path segments only (`/docs` covers `/docs/a.md`, not
  `/docs-old/a.md`). `context check` lists collections with no context and
  the top-level directories none covers. `context list` and `context check`
  are read-only commands for the closed-stdout exit status.
- Search results, `get` and the MCP `query` and `get` tools carry the
  context of every prefix above a document, global first; the MCP `query`
  text summary prints it as a `Context:` line.
- `embed --max-duration <MINUTES>` (default 30, `0` for no cap): `embed`
  starts no new work past the cap and reports how many documents still need
  embedding, for the next run to finish.
- `embed` retries a failed embedding call chunk by chunk, up to three
  attempts per chunk, and stops once more than 80 % of attempted chunks have
  failed. A document that fails to chunk is reported and skipped instead of
  ending the run.
- `embed` first deletes vectors no active document uses, so an edited
  document's old vectors no longer wait for `cleanup`.
- The config file is saved atomically: a failed save leaves `index.yml` as
  it was instead of truncated. The save writes through a symlink rather than
  replacing it, keeps the file's permissions, and names the file in its
  error with the I/O cause.
- On Windows, a collection's `update` command that begins with a double
  quote (`"my tool.exe" sync`) now runs; 0.3.1 still dropped that leading
  quote.
- When `collection rename` cannot save the config and cannot undo the index
  rename either, the error now names both failures and their causes, and
  says that the index and the config disagree.

## 0.3.1

- On Windows, a collection's `update` command keeps its double quotes:
  `cmd /C` now receives it verbatim. In 0.3.0 a command such as
  `git commit -m "a b"` or `echo x>"out file.txt"` was mangled or failed.
- `collection rename` leaves the index untouched when the config cannot be
  saved. In 0.3.0 the index kept the new name while the config kept the old
  one, and the next `qmd update` indexed every file a second time.

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
