#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QMD_UPSTREAM_DIR="${QMD_UPSTREAM_DIR:-/tmp/qmd-upstream}"
QMD_TSX="${QMD_UPSTREAM_DIR}/node_modules/.bin/tsx"
QMD_CLI="${QMD_UPSTREAM_DIR}/src/cli/qmd.ts"
FIXTURE_ROOT="${ROOT_DIR}/tests/fixtures"
CORPUS_DIR="${FIXTURE_ROOT}/qmd-corpus"
DB_PATH="${FIXTURE_ROOT}/qmd-index.sqlite"
CONFIG_DIR="${FIXTURE_ROOT}/qmd-config"
EMBED_MODEL="${QMD_EMBED_MODEL:-${HOME}/.cache/qmd/models/hf_ggml-org_embeddinggemma-300M-Q8_0.gguf}"

if [[ ! -x "${QMD_TSX}" ]]; then
  echo "missing qmd tsx launcher at ${QMD_TSX}" >&2
  exit 1
fi

mkdir -p "${CORPUS_DIR}" "${CONFIG_DIR}"
rm -f "${DB_PATH}"
find "${CORPUS_DIR}" -type f -name '*.md' -delete
printf 'collections: {}\n' > "${CONFIG_DIR}/index.yml"

for idx in $(seq 1 12); do
  slug=$(printf 'doc-%02d' "${idx}")
  title="QMD Proper Port Fixture ${idx}"
  case "${idx}" in
    1) keyword="indexing performance"; body="This document is the canonical reference for indexing performance, corpus build timing, and collection add latency." ;;
    2) keyword="vector retrieval"; body="This document focuses on vector retrieval, cosine distance, nearest-neighbor search, and semantic lookup." ;;
    3) keyword="semantic embeddings"; body="This document explains semantic embeddings, embeddinggemma formatting, and chunk-aware vector generation." ;;
    4) keyword="sqlite vec extension"; body="This document covers sqlite vec extension loading, vec0 virtual tables, and extension lifecycle behavior." ;;
    5) keyword="chunk overlap"; body="This document describes chunk overlap, token windows, code-fence safety, and qmd chunk metadata." ;;
    6) keyword="rerank pipeline"; body="This document is about rerank pipeline behavior, candidate pools, and chunk-level reranking." ;;
    7) keyword="mcp query"; body="This document describes MCP query behavior, stdio transport, and HTTP semantic query flows." ;;
    8) keyword="bm25 search"; body="This document covers BM25 search, FTS matching, and lexical query routing." ;;
    9) keyword="fixture generation"; body="This document describes fixture generation, deterministic corpora, and qmd parity harness inputs." ;;
    10) keyword="title formatting"; body="This document explains title formatting, embedding input templates, and qmd document formatting rules." ;;
    11) keyword="cleanup behavior"; body="This document describes cleanup behavior, orphaned vectors, and vec table deletion ordering." ;;
    12) keyword="hybrid search"; body="This document is the canonical source for hybrid search, query expansion, and vector plus lexical fusion." ;;
  esac
  cat > "${CORPUS_DIR}/${slug}.md" <<EOF
# ${title}

Keyword: ${keyword}

${body}

Reference phrase: ${keyword} should return ${slug}.md as the top relevant fixture document.

Additional context:
- qmd proper port
- qqd parity
- upstream oracle
- repeated relevance context for chunked semantic indexing
- ${keyword} retrieval fidelity matters for chunk-level parity
- this paragraph is intentionally repeated to force multi-chunk embeddings
EOF
  for repeat in $(seq 1 120); do
    cat >> "${CORPUS_DIR}/${slug}.md" <<EOF

Repeat ${repeat}: ${body}
Repeat ${repeat}: ${keyword} should remain the dominant retrieval signal for ${slug}.md.
Repeat ${repeat}: qmd and qqd proper-port verification depends on chunk-level semantic indexing fidelity.
EOF
  done
done

INDEX_PATH="${DB_PATH}" \
QMD_CONFIG_DIR="${CONFIG_DIR}" \
QMD_LLAMA_GPU=false \
"${QMD_TSX}" "${QMD_CLI}" collection add "${CORPUS_DIR}" --name docs

INDEX_PATH="${DB_PATH}" \
QMD_CONFIG_DIR="${CONFIG_DIR}" \
QMD_LLAMA_GPU=false \
QMD_EMBED_MODEL="${EMBED_MODEL}" \
"${QMD_TSX}" "${QMD_CLI}" embed

echo "fixture corpus: ${CORPUS_DIR}"
echo "fixture db: ${DB_PATH}"
