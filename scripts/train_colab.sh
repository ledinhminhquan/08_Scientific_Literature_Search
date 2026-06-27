#!/usr/bin/env bash
# Fine-tune the dense retriever on Colab/GPU. The notebook does this with an auto
# GPU profile; this is the plain CLI equivalent.
set -euo pipefail
cd "$(dirname "$0")/.."

export SCISEARCH_ARTIFACTS_DIR="${SCISEARCH_ARTIFACTS_DIR:-/content/drive/MyDrive/scisearch}"

scisearch data
scisearch --config configs/train.yaml train
scisearch evaluate
scisearch error-analysis
scisearch autopilot --no-train
