#!/usr/bin/env bash
# One-shot nnU-Net v2 SOTA reference on MMOTU OTU_2d (official 1000/469 split):
#   convert -> plan/preprocess -> train -> predict -> evaluate.
#
# Needs a GPU + nnunetv2 installed (`pip install nnunetv2`). Long-running: run it
# inside tmux, or with nohup:  nohup bash nnunet_mmotu/run_nnunet.sh > nnunet.log 2>&1 &
#
# Override defaults via env vars, e.g.:
#   TRAINER=nnUNetTrainer FOLD="0 1 2 3 4" bash nnunet_mmotu/run_nnunet.sh   # full 1000ep, 5 folds
set -euo pipefail

# ---- config (override via env) ----
DATASET_ID="${DATASET_ID:-501}"
DATASET_NAME="${DATASET_NAME:-MMOTU2d}"
CONFIG="${CONFIG:-2d}"
FOLD="${FOLD:-0}"                               # single fold by default; "0 1 2 3 4" for all
TRAINER="${TRAINER:-nnUNetTrainer_250epochs}"   # "" or "nnUNetTrainer" = full 1000 epochs
OTU_ROOT="${OTU_ROOT:-OTU_2d}"

# ---- paths (relative to repo root = parent of this script's dir) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export nnUNet_raw="$SCRIPT_DIR/nnUNet_raw"
export nnUNet_preprocessed="$SCRIPT_DIR/nnUNet_preprocessed"
export nnUNet_results="$SCRIPT_DIR/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

DID3=$(printf "%03d" "$DATASET_ID")
DATADIR="$nnUNet_raw/Dataset${DID3}_${DATASET_NAME}"
PREDS="$SCRIPT_DIR/preds"

TR_ARG=""; [ -n "$TRAINER" ] && TR_ARG="-tr $TRAINER"

if ! command -v nnUNetv2_train >/dev/null 2>&1; then
    echo "ERROR: nnunetv2 not found. Install it first:  pip install nnunetv2"; exit 1
fi

echo "############## 1/5 convert MMOTU -> nnU-Net (official split) ##############"
if [ -f "$DATADIR/dataset.json" ] && [ "${FORCE_CONVERT:-0}" != "1" ]; then
    echo "dataset.json already exists at $DATADIR (set FORCE_CONVERT=1 to redo); skipping."
else
    python nnunet_mmotu/convert_to_nnunet.py --otu-root "$OTU_ROOT" \
        --dataset-id "$DATASET_ID" --dataset-name "$DATASET_NAME"
fi

echo "############## 2/5 plan + preprocess ##############"
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity

echo "############## 3/5 train ($CONFIG, folds: $FOLD, trainer: ${TRAINER:-default-1000ep}) ##############"
for f in $FOLD; do
    echo "---- fold $f ----"
    nnUNetv2_train "$DATASET_ID" "$CONFIG" "$f" $TR_ARG
done

echo "############## 4/5 predict on official 469 test ##############"
FIRST_FOLD=$(echo "$FOLD" | awk '{print $1}')
nnUNetv2_predict -i "$DATADIR/imagesTs" -o "$PREDS" \
    -d "$DATASET_ID" -c "$CONFIG" -f $FOLD $TR_ARG

echo "############## 5/5 evaluate (foreground Dice/IoU + mIoU) ##############"
python nnunet_mmotu/evaluate_preds.py --preds "$PREDS" --gt "$DATADIR/test_gt"

echo "############## DONE. Metrics above; per-image CSV in nnunet_mmotu/. ##############"
