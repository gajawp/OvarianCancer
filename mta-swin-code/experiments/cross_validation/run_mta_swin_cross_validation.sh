#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATASET_ROOT="${1:-/projects/insightx-lab/cleaned-brain-tumour-dataset}"
PRETRAINED_WEIGHTS="${2:-${PROJECT_ROOT}/experiments/comparison/external/best_model.pth}"
OUTPUT_DIR="${3:-${PROJECT_ROOT}/experiments/cross_validation/outputs/mta_swin_5fold_seed1}"
SEED="${4:-1}"
FOLDS="${5:-5}"

shift "$(( $# < 5 ? $# : 5 ))"

python "${PROJECT_ROOT}/experiments/cross_validation/run_mta_swin_cross_validation.py" \
  --dataset-root "${DATASET_ROOT}" \
  --pretrained-weights "${PRETRAINED_WEIGHTS}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --folds "${FOLDS}" \
  --save-plots \
  "$@"
