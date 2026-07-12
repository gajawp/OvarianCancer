# MTA-Swin Cross-Validation

This directory contains the revision experiment for reviewer-requested cross-validation.

The script runs only the ImageNet-pretrained MTA-Swin model. It merges the cleaned dataset's
`Training/` and `Testing/` folders into one image pool, then performs stratified K-fold
cross-validation. For each fold, the held-out fold is used only for testing. The remaining
folds are split into train and validation subsets for early stopping.

This is intended as a stability check for MTA-Swin under repeated image-level splits, not as
a new comparison against all baseline models.

## Run

```bash
bash experiments/cross_validation/run_mta_swin_cross_validation.sh \
  /projects/insightx-lab/cleaned-brain-tumour-dataset \
  /path/to/imagenet_pretrained_mta_swin/best_model.pth
```

Optional positional arguments:

```text
1. dataset root
2. ImageNet-pretrained MTA-Swin checkpoint
3. output directory
4. random seed
5. number of folds
```

Extra Python arguments can be appended after the five positional arguments, for example:

```bash
bash experiments/cross_validation/run_mta_swin_cross_validation.sh \
  /projects/insightx-lab/cleaned-brain-tumour-dataset \
  /path/to/best_model.pth \
  experiments/cross_validation/outputs/mta_swin_5fold_seed1 \
  1 \
  5 \
  --max-epochs 120 \
  --early-stopping-patience 15
```

By default, each fold's best weights are stored in memory only and are not written as large
checkpoint files. Add `--keep-checkpoints` if you need to persist fold checkpoints.

## Outputs

- `fold_metrics.csv`: fold-level test metrics
- `cv_summary.csv`: mean, standard deviation, min, and max across folds
- `cv_predictions.pkl`: per-fold predictions and probabilities
- `cv_metadata.json`: protocol and configuration details
- `fold_*/`: per-class metrics, per-fold predictions, optional plots, and checkpoints only when `--keep-checkpoints` is used
