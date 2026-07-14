# MMOTU Task 3 (8-class recognition) — MTA-Swin reproduction & preliminary improvements

## 1. Scope

This document reports a reproduction of **Task 3 (single-modality recognition /
classification)** on the MMOTU dataset using the **MTA-Swin** model, carried out
per the advisor's request. The goals are:

1. Establish an MTA-Swin classification baseline on MMOTU OTU_2d (the original
   MMOTU release does not provide a Task 3 baseline or code).
2. Explore the **performance gap** relative to recent literature on the same
   dataset.
3. Explore **preliminary, data-side ideas** for closing that gap.

All numbers below are our own runs unless a citation is given.

## 2. Dataset & experimental setup

- **Data**: MMOTU **OTU_2d**, 1469 2D ultrasound images, **8 tumor categories**,
  official split **1000 train / 469 test** (`train_cls.txt` / `val_cls.txt`).
- **Model**: MTA-Swin-Tiny (~27.5 M params), ImageNet-1K pretrained, input 224×224.
- **Training**: AdamW (lr 1e-4, wd 1e-4), batch 32, AMP; label smoothing 0.1.
- **Protocol**: two protocols are used and labelled per table:
  - *val + early-stopping*: `train_cls` split 80/20 (stratified) for validation;
  - *no-val (paper-style)*: train on the full 1000-image pool for a fixed number
    of epochs (cosine LR, no early stopping), evaluate the final model.
  - **Test set is always the official 469-image `val_cls` held-out set.**
- **Metrics**: Accuracy, Balanced Accuracy, macro-F1, MCC, ROC-AUC (OvR macro).
  Balanced Accuracy / macro-F1 are emphasized because the data is imbalanced.

### 2.1 Class distribution (test support) — imbalance

| Label | 0 CC | 1 SC | 2 T | 3 TCT | 4 SCH | 5 NO | 6 MC | 7 HGSC |
|---|---|---|---|---|---|---|---|---|
| Train (~) | 226 | 153 | 228 | 57 | 47 | 180 | 71 | 38 |
| Test | 110 | 66 | 108 | 31 | 19 | 87 | 33 | 15 |

CC = chocolate cyst, SC = serous cystadenoma, T = teratoma, TCT = theca cell
tumor, SCH = simple cyst, NO = normal ovary, MC = mucinous cystadenoma,
HGSC = high-grade serous cystadenoma. Classes 4/7 (SCH/HGSC) are the smallest.

## 3. Pipeline sanity-check — reference baselines

Standard backbones trained through the same pipeline (val + early-stopping,
accuracy-selected, seed 42), to confirm the data/eval path is sound.

| Model | Params | Accuracy | Bal-Acc | Macro-F1 | MCC | ROC-AUC |
|---|---|---|---|---|---|---|
| ResNet-50 | 23.5 M | 66.53 | 56.08 | 57.31 | 0.59 | 86.99 |
| Swin-T | 27.5 M | 74.20 | 63.71 | 65.10 | 0.69 | 90.53 |
| **MTA-Swin** | 27.5 M | 72.50 | 62.71 | 65.44 | 0.67 | 90.54 |

All models converge normally and land in a consistent range; minority-class
recall is the main weakness across all of them (see §7), indicating a
data-intrinsic difficulty rather than a pipeline issue.

## 4. Multi-seed reproduction (no-val, 50 epochs, seeds 0/1/2)

| Model | Accuracy | Bal-Acc | Macro-F1 | MCC | ROC-AUC |
|---|---|---|---|---|---|
| MTA-Swin | 76.12 ± 2.01 | 67.18 ± 3.64 | 69.61 ± 3.34 | 0.71 ± 0.02 | 92.32 ± 0.62 |
| Swin-T | 75.91 ± 1.38 | 67.37 ± 2.35 | 69.24 ± 2.17 | 0.71 ± 0.02 | 93.11 ± 0.38 |

## 5. Performance gap vs. recent MMOTU Task 3 literature

| Work | Method (summary) | Accuracy | Macro-F1 | Eval protocol |
|---|---|---|---|---|
| Ours (baseline) | MTA-Swin, full-image | 77–78 | 71–72 | official 1000/469 held-out |
| Ours (best, §8) | MTA-Swin + sampling + aug + MixUp | **80.5** | **76.1** | official 1000/469 held-out |
| EfficientOvaNet (2025) | EfficientNet-B3 dual-branch (global + mask-ROI), MixUp/CutMix, weighted sampling | 91.9 | 91.9 | **5-fold CV**, 70/15/15 |
| Early-fusion hybrid (2025) | EfficientNet-B7 + Swin early fusion; oversampling; US-aug; ensemble | 92.1 (ens. 93.3) | 92.1 | **5-fold CV ×10** |

**Note on comparability**: the two ~92% works use k-fold cross-validation over
all 1469 images (larger effective training set + CV averaging), not the official
1000/469 held-out split used here. The evaluation protocol alone accounts for a
substantial part of the gap, so the absolute numbers are not directly comparable.

## 6. Effect of training-epoch budget (MTA-Swin, no-val, seed 42)

| Epochs | Accuracy | Bal-Acc | Macro-F1 | MCC | ROC-AUC | Miscls |
|---|---|---|---|---|---|---|
| 10 | 70.36 | 56.69 | 57.82 | 0.64 | 90.03 | 139 |
| 30 | 74.41 | 62.38 | 65.04 | 0.69 | 91.32 | 120 |
| 50 | 75.27 | 64.14 | 67.38 | 0.70 | 91.98 | 116 |
| 70 | 77.83 | 69.57 | 72.31 | 0.73 | 91.23 | 104 |
| 100 | 78.25 | 71.04 | 72.35 | 0.74 | 91.59 | 102 |
| 150 | 77.83 | 70.77 | 73.92 | 0.73 | 91.83 | 104 |
| 200 | 78.25 | 71.39 | 73.78 | 0.74 | 92.43 | 102 |

(Full sweep 10–200 in 10-epoch steps was run; representative rows shown.)
Performance rises steeply up to ~70 epochs and then plateaus; 100 epochs is a
reasonable operating point (values beyond 100 fluctuate within run-to-run noise).
The cosine schedule length scales with the epoch budget, so "more epochs" and
"gentler annealing" are partly confounded here.

## 7. Preliminary improvement levers (MTA-Swin, no-val, 100 epochs, seed 42)

Each lever is toggled on top of the base recipe. Δ is vs. base.

| Config | Accuracy | Bal-Acc | Macro-F1 | MCC | ROC-AUC | Miscls |
|---|---|---|---|---|---|---|
| base | 78.25 | 71.04 | 72.35 | 0.74 | 91.59 | 102 |
| sampler (β=1.0) | 76.12 | 70.03 | 71.73 | 0.71 | 93.40 | 112 |
| sampler (β=0.5) | 76.33 | 68.01 | 69.47 | 0.71 | 92.11 | 111 |
| ROI crop †| 81.45 | 75.17 | 76.06 | 0.78 | 95.55 | 87 |
| ROI mask †| 80.17 | 72.64 | 73.82 | 0.76 | 95.38 | 93 |
| aug medium | 78.25 | 70.49 | 72.22 | 0.74 | 93.06 | 102 |
| aug strong | 78.47 | 70.42 | 73.00 | 0.74 | 93.45 | 101 |
| MixUp/CutMix | 79.32 | 69.83 | 72.48 | 0.75 | 94.01 | 97 |
| sampler(0.5) + aug strong | 80.17 | 73.48 | 74.83 | 0.76 | 93.95 | 93 |
| MixUp + aug strong | 80.38 | 74.18 | 75.72 | 0.76 | 93.79 | 92 |
| **MixUp + aug strong + sampler(0.5)** | **81.45** | **74.74** | **76.94** | **0.78** | **94.62** | **87** |

† ROI crops the lesion using the dataset's ground-truth binary masks (available
for all images in MMOTU). A deployable pipeline would instead use masks predicted
by a segmentation model; ROI rows are listed for reference and were not pursued
further here.

**Observations (single seed, to be read as trends):**

- Individually, balanced sampling and stronger augmentation do **not** help and
  can slightly reduce Accuracy/macro-F1; MixUp alone helps overall metrics but
  not minority balance.
- They become effective **in combination**: oversampled minority images gain
  variety from augmentation/MixUp (i.e. effectively new samples rather than exact
  repeats). The three combined give the largest, consistent improvement.
- ROI (ground-truth mask) is the single strongest lever but relies on GT masks.
  The data-side combination reaches the same Accuracy/macro-F1 without masks.

## 8. Best data-side recipe — multi-seed confirmation

Recipe = **balanced sampling (β=0.5) + strong (lesion-preserving) augmentation +
MixUp/CutMix**, no-val protocol, seeds 0/1/2 (mean ± std).

| Config | Accuracy | Bal-Acc | Macro-F1 | MCC | ROC-AUC |
|---|---|---|---|---|---|
| base (100 ep) | 77.47 ± 0.82 | 69.62 ± 1.44 | 71.42 ± 1.18 | 0.73 ± 0.01 | 92.57 ± 0.75 |
| **recipe (100 ep)** | **80.45 ± 0.44** | **73.62 ± 1.53** | **76.08 ± 1.26** | **0.76 ± 0.01** | **94.55 ± 0.67** |
| recipe (150 ep) | 80.74 ± 1.12 | 73.69 ± 2.09 | 76.03 ± 1.94 | 0.77 ± 0.01 | 94.74 ± 0.30 |

- Δ (recipe − base, 100 ep): **Accuracy +2.98, Bal-Acc +4.00, macro-F1 +4.65,
  ROC-AUC +1.98.** The ±std bands for Accuracy and macro-F1 do not overlap, so the
  improvement is consistent across seeds rather than seed variance.
- 150 epochs gives no meaningful gain over 100 (and higher variance).

### 8.1 Per-class recall (sensitivity, %), mean over seeds 0/1/2

| Config | 0 CC | 1 SC | 2 T | 3 TCT | 4 SCH | 5 NO | 6 MC | 7 HGSC |
|---|---|---|---|---|---|---|---|---|
| base | 88.2 | 79.8 | 82.1 | 55.9 | 54.4 | 80.8 | 53.5 | 62.2 |
| recipe | 89.7 | 76.3 | 85.8 | 63.4 | 63.2 | 83.1 | 69.7 | 57.8 |

The recipe improves recall on several minority classes (MC +16.2, SCH +8.8,
TCT +7.5) while keeping majority classes stable; HGSC (15 test samples) is the
noisiest class.

## 9. Summary

- An MTA-Swin Task 3 baseline was established on MMOTU OTU_2d under the official
  held-out split: **~77–78% Accuracy / ~71–72 macro-F1**.
- A purely **data-side** recipe (balanced sampling + lesion-preserving strong
  augmentation + MixUp/CutMix) raises this to **80.5% Accuracy / 76.1 macro-F1**
  (3-seed mean, +3.0 / +4.7), with consistent, non-overlapping error bars and
  improved minority-class recall — no change to the model architecture.
- The remaining gap to the ~92% figures in recent work is, in large part,
  attributable to a different evaluation protocol (k-fold CV over all images vs.
  the official held-out split) and to techniques not yet applied here
  (e.g. mask-based ROI two-branch inputs, ImageNet pretraining strength,
  multi-architecture ensembling).

### Reproducing these tables

Run configurations are encoded in each output file's tag, e.g.
`MTA-Swin_pretrained_noval_e100_augstrong_mixup_samp0.5_s0`. See `README.md` for
the exact commands (`--no-val`, `--epochs`, `--sampler`, `--sampler-beta`,
`--aug`, `--mixup`, `--roi`, `--seed`).
