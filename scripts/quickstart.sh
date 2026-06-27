#!/usr/bin/env bash
# Quickstart: install, prepare data, run the offline agent demo, and self-grade.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> install (core + extras)"
pip install -e ".[ml,api,report]"

echo "==> load corpus + build (query, paper) pairs"
scisearch data

echo "==> see the baseline you must beat (BM25)"
scisearch evaluate || true

echo "==> run the agent on sample queries (offline TF-IDF if no torch)"
scisearch demo-agent --tfidf

echo "==> generate report.pdf + slides.pptx and self-grade"
scisearch generate-report
scisearch generate-slides
scisearch grade
