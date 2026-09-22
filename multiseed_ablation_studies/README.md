# Multi-Seed Ablation Studies

This bundle repeats three important configurations using seeds `42`, `123`,
and `2026`:

1. Reference: ROI + mask + augmentation + alpha-weighted focal loss
2. No augmentation
3. ROI only: explicit mask branch removed

All other settings remain fixed. Each run uses separate checkpoint and result
directories, so existing models and different seeds are never overwritten.

## Placement

Place the complete `multiseed_ablation_studies` folder directly inside the
`Ovarian_Cancer` project root.

## Recommended command

Run all configurations and all three seeds:

```bash
caffeinate -i python -m multiseed_ablation_studies.run_seed_studies
```

The runner automatically reuses the existing seed-42 `metric_summary.csv` for
each configuration when it finds the previously completed ablation results.
Therefore, under your current project structure, it should train only seeds
`123` and `2026`: six additional training runs.

If a result already exists, the runner skips it. This allows the command to be
stopped and resumed safely.

## Run one configuration

```bash
caffeinate -i python -m multiseed_ablation_studies.run_seed_studies \
  --experiments reference
```

Valid experiment names are:

```text
reference
no_augmentation
roi_only
```

## Run selected seeds

```bash
caffeinate -i python -m multiseed_ablation_studies.run_seed_studies \
  --experiments reference no_augmentation roi_only \
  --seeds 123 2026
```

## Force a rerun

```bash
caffeinate -i python -m multiseed_ablation_studies.run_seed_studies \
  --experiments reference \
  --seeds 123 \
  --force
```

To retrain seed 42 instead of reusing its completed metrics:

```bash
caffeinate -i python -m multiseed_ablation_studies.run_seed_studies \
  --seeds 42 123 2026 \
  --rerun-seed42
```

## Output structure

```text
multiseed_checkpoints/
  reference/seed_123/checkpoints/
  no_augmentation/seed_123/checkpoints/
  roi_only/seed_123/checkpoints/

classification_results/multiseed_ablation/
  reference/seed_42/
  reference/seed_123/
  reference/seed_2026/
  no_augmentation/...
  roi_only/...
  all_seed_metrics.csv
  mean_std_summary.csv
```

`mean_std_summary.csv` reports the arithmetic mean and sample standard
deviation using the format:

```text
0.5870 ± 0.0410
```

## Regenerate summaries only

```bash
python -m multiseed_ablation_studies.summarize_seed_results
```

The summary includes accuracy, balanced accuracy, malignant precision,
sensitivity, specificity, malignant F1, macro F1, NPV, ROC-AUC, and PR-AUC.
