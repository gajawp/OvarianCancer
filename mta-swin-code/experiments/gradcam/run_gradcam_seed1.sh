#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CHECKPOINT="${1:-/projects/insightx-lab/mta-swin/finetuned-weights-seed=1/best_MTA-Swin_pretrained_model.pt}"
DATASET_ROOT="${2:-/projects/insightx-lab/cleaned-brain-tumour-dataset}"
OUTPUT_DIR="${3:-${PROJECT_ROOT}/experiments/gradcam/outputs/seed1_all_stage_4samples}"
SPLIT="${4:-Testing}"

shift "$(( $# < 4 ? $# : 4 ))"

python "${PROJECT_ROOT}/experiments/gradcam/visualize_mta_swin_gradcam.py" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATASET_ROOT}" \
  --split "${SPLIT}" \
  --output-dir "${OUTPUT_DIR}" \
  --samples-per-class 4 \
  --save-heatmaps \
  "$@"
