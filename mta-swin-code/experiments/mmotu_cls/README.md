# MMOTU 8-class Classification — MTA-Swin (pretrained)

Runs a single model, **MTA-Swin-Tiny (ImageNet-1K pretrained)**, on the MMOTU
`OTU_2d` ovarian-tumor dataset as an **8-class** image classification task.
Adapted from `experiments/comparison/` (same training recipe), with the data
layer swapped to MMOTU's flat images + `*_cls.txt` labels.

## Task setup

- **Classes**: 8 (labels `0-7` in `OTU_2d/train_cls.txt` / `val_cls.txt`).
- **Split**: `train_cls.txt` (1000) is the training pool → stratified **80/20**
  into train/val (**seed 42**) for early stopping + LR scheduling.
  `val_cls.txt` (469) is the fixed **held-out test set** (paper's official split).
- **Recipe** (from the comparison config): 224×224 input, batch 32, AdamW
  `lr=1e-4 wd=1e-4`, CrossEntropy + label smoothing 0.1, `ReduceLROnPlateau`
  on val-acc, early stopping patience 20, up to 200 epochs, AMP on CUDA.
  No class weighting yet (dataset is imbalanced — watch macro-F1 / balanced acc).

## Files

- `config.py` — paths (env-overridable), hyperparameters, MTA stage config, class names.
- `run_mmotu_mta_swin.py` — data loading, training, held-out evaluation, artifact saving.

## Environment (conda, GPU server)

```bash
conda create -n mta python=3.10 -y
conda activate mta

# 1) Install a torch/torchvision build matching the server's CUDA first
#    (from the official PyTorch index), then the rest:
pip install -r experiments/comparison/requirements.txt   # torch, torchvision, timm, sklearn, pandas, seaborn, matplotlib, pillow
```

## Paths / data

Defaults assume the **whole project repo** is copied to the server (so
`OTU_2d/` sits next to `mta-swin-code/`, and `best_model.pth` is in the
`mta-swin-code/` root). If your layout differs on the server, override via
environment variables — no code edits needed:

```bash
export MMOTU_IMAGE_DIR=/path/to/OTU_2d/images
export MMOTU_TRAIN_CLS=/path/to/OTU_2d/train_cls.txt
export MMOTU_VAL_CLS=/path/to/OTU_2d/val_cls.txt
export MTA_PRETRAINED_WEIGHTS=/path/to/best_model.pth
```

The script prints all resolved paths + the pretrained-weight load result
(matched / missing / unexpected tensors) at startup — check that non-head
weights all load (otherwise the checkpoint's MTA config doesn't match).

## Run

```bash
cd mta-swin-code
python experiments/mmotu_cls/run_mmotu_mta_swin.py
# quick smoke test:  python experiments/mmotu_cls/run_mmotu_mta_swin.py --epochs 2
```

### Other comparison baselines (sanity check)

The same pipeline can train any comparison-baseline model via `--model` /
`--mode`, useful to confirm the data/eval path behaves on non-MTA models:

```bash
python experiments/mmotu_cls/run_mmotu_mta_swin.py --model ResNet-50   # torchvision, pretrained
python experiments/mmotu_cls/run_mmotu_mta_swin.py --model Swin-T      # closest ref to MTA-Swin
python experiments/mmotu_cls/run_mmotu_mta_swin.py --model ResNet-50 --mode scratch
```

Accepted `--model`: `MTA-Swin` (default), `ResNet-50`, `EfficientNet-B4`,
`ConvNeXt-T`, `Swin-T`, `DeiT-S/16`, `ViT-S/16`, `mamba`, `maxvit`, `davit`,
`cait`, `inceptionnext`, `swinv2`, `Custom CNN`. `--mode {pretrained,scratch}`
(default `pretrained`). Pretrained weights for these are downloaded on first
use (torchvision / timm) — needs internet or a warm cache on the server.
Each run's outputs/checkpoint are tagged `<model>_<mode>_<selection><_cw>`.

### Imbalance handling (optional)

Defaults reproduce the original run (accuracy-based early stopping, no class
weights). MMOTU is imbalanced, so these switches usually help minority-class
recall / macro-F1:

```bash
# early stopping + LR scheduler track macro-F1 instead of accuracy
python experiments/mmotu_cls/run_mmotu_mta_swin.py --selection-metric macro_f1

# inverse-frequency class-weighted CrossEntropy
python experiments/mmotu_cls/run_mmotu_mta_swin.py --class-weights

# shortcut for both (macro-F1 selection + class weights)
python experiments/mmotu_cls/run_mmotu_mta_swin.py --balanced
```

Each configuration is tagged (`accuracy`, `macro_f1`, `macro_f1_cw`, ...) in its
output filenames and checkpoint, so runs don't overwrite each other and are easy
to compare. The summary CSV records the selection metric and whether class
weights were used.

## Outputs (`experiments/mmotu_cls/outputs/`)

- `summary_<ts>.csv` — accuracy, balanced acc, macro-F1/precision/sensitivity/specificity, MCC, ROC-AUC, params.
- `per_class_<ts>.csv` — per-class precision/sensitivity/specificity/F1 + support.
- `checkpoints/best_mta_swin_pretrained.pt` — best-val-acc weights.
- `plots/` — training curves + confusion matrix.

## Class index mapping (inferred)

`0-CC` chocolate cyst · `1-SC` serous cystadenoma · `2-T` teratoma ·
`3-TCT` theca cell tumor · `4-SCH` simple cyst · `5-NO` normal ovary ·
`6-MC` mucinous cystadenoma · `7-HGSC` high-grade serous cystadenoma.
(Order follows the paper; index 5 = Normal Ovary is corroborated by its
absence from the CEUS split. Used only for display labels.)
