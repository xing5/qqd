# qqd

`qqd` is a local-first search CLI for Markdown knowledge bases. It keeps a
qmd-compatible SQLite index, supports lexical and vector search, and can expose
the same index through an MCP server for agent use.

## Build

```bash
cargo build --release
```

Use the release binary for benchmarks and regular indexing:

```bash
./target/release/qqd --help
```

`qqd` auto-discovers local GGUF models from `~/.cache/qmd/models`. You can
override them with:

```bash
QQD_EMBED_MODEL=/path/to/embed.gguf
QQD_RERANK_MODEL=/path/to/rerank.gguf
```

## Basic Usage

Create an isolated named index, add a Markdown collection, and build vectors:

```bash
qqd --index notes collection add ~/notes --name notes
qqd --index notes embed
qqd --index notes status
```

Search the index:

```bash
qqd --index notes query "how do we deploy the service?"
qqd --index notes search "exact keyword"
qqd --index notes vsearch "semantic question in natural language"
```

Inspect indexed content:

```bash
qqd --index notes collection list
qqd --index notes collection show notes
qqd --index notes ls notes
qqd --index notes get 'qmd://notes/path/to/file.md'
```

Refresh after files change:

```bash
qqd --index notes update
qqd --index notes embed
```

## Local xkb Example

For the local knowledge base at `~/workspace/xkb/runtime/items`, keep it in a
named index so it does not mix with the default qmd/qqd index:

```bash
cd /home/xing/workspace/qqd
QQD=/home/xing/workspace/qqd/target/release/qqd

$QQD --index xkb collection add ~/workspace/xkb/runtime/items --name xkb-items
$QQD --index xkb embed
$QQD --index xkb status
```

Query it:

```bash
$QQD --index xkb query "how does runtime item scheduling work?"
$QQD --index xkb search "scheduled runtime item"
$QQD --index xkb vsearch "find notes about background jobs"
```

Browse or fetch documents:

```bash
$QQD --index xkb collection list
$QQD --index xkb collection show xkb-items
$QQD --index xkb ls xkb-items
$QQD --index xkb get 'qmd://xkb-items/path/to/file.md'
```

The default collection pattern is `**/*.md`. The xkb directory currently has
Markdown, JSON, images, videos, and PDFs; the command above indexes the Markdown
files only. To also index JSON as raw text, add a second collection:

```bash
$QQD --index xkb collection add ~/workspace/xkb/runtime/items --name xkb-json --mask '**/*.json'
$QQD --index xkb embed
```

## BEIR SciFact Benchmark

Latest measured benchmark: BEIR SciFact, 5,183 documents, 300 scored qrel
queries, top-k 100. `qqd` was run from `target/release/qqd`; `qmd` was the
installed local command. Runs were executed sequentially to avoid resource
contention.

| Metric | qmd | qqd release |
| --- | ---: | ---: |
| Index time | `5.639s` | `1.874s` |
| Embed time | `497.505s` | `389.341s` |
| Vectors/chunks | `5358` | `5358` |
| DB size | `50.2 MB` | `49.5 MB` |

| Surface | qmd p50/p95 | qqd p50/p95 | qmd nDCG@10 | qqd nDCG@10 | qmd Recall@100 | qqd Recall@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lex | `0.404 / 0.455s` | `0.875 / 1.286s` | `0.0467` | `0.0467` | `0.0458` | `0.0458` |
| vec | `1.651 / 1.919s` | `0.950 / 1.366s` | `0.7525` | `0.7531` | `0.9199` | `0.9672` |
| hybrid | `1.702 / 1.968s` | `0.961 / 1.312s` | `0.7570` | `0.7580` | `0.9199` | `0.9672` |

Summary: `qqd` is faster on index, embed, vector query, and hybrid query in
this benchmark. `qmd` remains faster for lex-only query latency.

## Development

Run the standard checks before committing:

```bash
cargo fmt
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```
