# qmd-rs

On-device hybrid search for markdown and text notes. Directories of documents
are indexed into one SQLite file and queried with BM25 full-text search (FTS5)
fused with vector search (sqlite-vec and a local GGUF embedding model run by
llama.cpp) by reciprocal rank fusion. Nothing leaves the machine and no API key
is needed.

This is a Rust port of [qmd](https://github.com/tobi/qmd) v2.6.3. `qmd query` is its
`--no-rerank` path unless you pass `--rerank` (Qwen3-Reranker-0.6B); LLM query
expansion is not ported. BM25 terms match if any is present; `--match all`
requires every one, as the TypeScript implementation does. A collection's chunk
size is its `chunkTokens` (default 900; `collection add --chunk-tokens`), and
`chunkBoundary: heading` starts its chunks at headings. The
SQLite schema is the TypeScript implementation's, so both can open one index.

`sectionLevel: N` (1–6; `collection add --section-level N`, or
`collection sections <name> [N]` on an existing one; omit N to switch back)
makes each section of a file at heading level N its own document. BM25 ranks
sections, vector search and fusion work over them, one file can give several
results, and each result names its file, heading, line span and an identity
`collection/path#K` that `get` fetches. A section runs from a heading of level N
or shallower to the next one, so deeper headings stay inside it. Text before
the first such heading is a section of its own. A file with no such heading is
one section, the whole file. A heading with nothing under it joins the section
after it. A section longer than the embedder takes in one input (1800 tokens)
is split at paragraph boundaries for embedding only and stays one result.
Switching a collection rebuilds its full-text rows and needs `qmd embed` to
re-embed it. These tables exist only once a collection opts in.

## Build

Requires Rust 1.96 or later and CMake; llama.cpp is compiled from source.

CPU only:

```
cargo build --release --no-default-features --features llama
```

With CUDA offload (the default features), which also needs the CUDA toolkit:

```
cargo build --release
```

The binary is `target/release/qmd`. `qmd status` says whether it was built
with CUDA. llama.cpp's CMake build reads `CMAKE_*` and `GGML_*` environment
variables. The `qmd/prebuilt-ggml` feature skips building ggml: it links a static
CUDA ggml installed under `CMAKE_PREFIX_PATH`, whose `key.txt` must read
`llama-cpp-sys-2-<locked version>-…`.

## Quick start

```
cd my-notes
qmd init                          # project-local index in ./.qmd/
qmd collection add . --name notes # default mask **/*.md
qmd embed                         # downloads the embedding model on first use
qmd query "how do we handle authentication"
```

Without `qmd init`, commands use a global index under `~/.config/qmd` and
`~/.cache/qmd`. A `.qmd/` directory in the current directory or any parent
takes precedence; `--index <name>` selects a named global index. `qmd --help`
lists every command: `search` (BM25), `vsearch` (vector), `query` (hybrid),
`get`, `ls`, `update`, `drift`, `status`, `archive` and `mcp` among them.
`qmd archive` identifies itself as `qmd-archive/<version>`; set
`QMD_ARCHIVE_CONTACT` to a URL that reaches you to add it as `(+<url>)`.

The default embedding model is EmbeddingGemma-300M (Q8_0), downloaded from
Hugging Face (`ggml-org/embeddinggemma-300M-GGUF`) into `~/.cache/qmd/models/`.
It is not part of this repository, and its use is under Google's Gemma Terms
of Use. `QMD_EMBED_MODEL`, or `models.embed` in the index's `index.yml`,
selects another. The download goes through the `hf` CLI when it is on `PATH`;
otherwise qmd fetches the file itself, sending your Hugging Face token
(`HF_TOKEN`, or the file `hf auth login` writes) when one is set. `qmd status`
shows which.

## MCP

`qmd mcp` serves the index to an agent over MCP on stdio, with the tools
`query`, `get`, `multi_get` and `status`:

```json
{ "mcpServers": { "qmd": { "command": "qmd", "args": ["mcp"] } } }
```

`qmd mcp --http --daemon` serves the same tools from one shared process on
`http://127.0.0.1:8181/mcp`; `qmd mcp stop` ends it.

## Licence

MIT, in [LICENSE](LICENSE). The upstream project's notice is in
[LICENSE-UPSTREAM](LICENSE-UPSTREAM), and the origin of every file is in
[PROVENANCE.md](PROVENANCE.md).
