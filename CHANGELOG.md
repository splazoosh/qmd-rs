# Changelog

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
