#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CHECKPOINT="${1:-/projects/insightx-lab/mta-swin/finetuned-weights-seed=1/best_MTA-Swin_pretrained_model.pt}"
DATASET_ROOT="${2:-/projects/insightx-lab/cleaned-brain-tumour-dataset}"
OUTPUT_DIR="${3:-${PROJECT_ROOT}/experiments/gradcam/outputs/seed1_stage4_last_norm2_paper}"

shift "$(( $# < 3 ? $# : 3 ))"

python "${PROJECT_ROOT}/experiments/gradcam/visualize_mta_swin_gradcam.py" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --layers stage4_last_norm2 \
  --image-paths \
    "${DATASET_ROOT}/Testing/glioma/Te-glTr_0006.jpg" \
    "${DATASET_ROOT}/Testing/meningioma/Te-meTr_0006.jpg" \
    "${DATASET_ROOT}/Testing/pituitary/Te-piTr_0004.jpg" \
    "${DATASET_ROOT}/Testing/notumor/Te-no_0015.jpg" \
  --paper-figure \
  --paper-figure-name gradcam_stage4_last_norm2_paper.png \
  "$@"
