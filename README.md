# qmd-rs

On-device hybrid search for markdown and text notes. Directories of documents
are indexed into one SQLite file and queried with BM25 full-text search (FTS5)
fused with vector search (sqlite-vec and a local GGUF embedding model run by
llama.cpp) by reciprocal rank fusion. Nothing leaves the machine and no API key
is needed.

This is a Rust port of [qmd](https://github.com/tobi/qmd) v2.6.3, equivalent to
`qmd query --no-rerank`: LLM reranking and query expansion are not ported. The
SQLite schema is the TypeScript implementation's, so both can open one index.

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
variables.

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
selects another.

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
