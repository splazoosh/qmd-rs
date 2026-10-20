# Changelog

## 0.1.0

First release. `qmd` indexes directories of markdown into one SQLite file and
searches them with BM25, vector search over a local EmbeddingGemma model, or
both fused by reciprocal rank fusion, matching qmd v2.6.3's
`query --no-rerank`. It reads and writes the TypeScript implementation's index
schema, serves the index over MCP on stdio or from one shared HTTP process,
and mirrors served documents with their provenance through `qmd archive`.
