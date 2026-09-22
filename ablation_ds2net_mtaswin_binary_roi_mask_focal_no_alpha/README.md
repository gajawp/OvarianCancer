# Ablation A7: Focal Loss Without Class Alpha Weighting

This experiment isolates the contribution of inverse-frequency class weights
inside focal loss. It retains focal focusing with `gamma=2.0` and changes only:

```python
FOCAL_USE_CLASS_ALPHA = False
```

## Loss comparison

Reference:

```text
Focal loss, gamma=2.0, inverse-frequency alpha enabled
```

A7:

```text
Focal loss, gamma=2.0, alpha disabled
```

Unlike A4, this experiment does not replace focal loss with cross-entropy. It
therefore separates the effect of class weighting from the focal modulation
term.

## Settings retained

- Input size: 224x224
- DS2Net RGB ROI and explicit mask encoder
- No full-image context branch
- Synchronized ROI-and-mask augmentation
- Nearest-neighbor mask resizing
- Batch size: 1
- Gradient accumulation: 4
- Same official split files, random seed, pretrained Swin backbone, optimizer,
  scheduler, early stopping, and checkpoint selection

## Commands

Run from the `Ovarian_Cancer` project root:

```bash
python -m ablation_ds2net_mtaswin_binary_roi_mask_focal_no_alpha.test_model
```

```bash
caffeinate -i python -m ablation_ds2net_mtaswin_binary_roi_mask_focal_no_alpha.train
```

```bash
python -m ablation_ds2net_mtaswin_binary_roi_mask_focal_no_alpha.evaluate
```

## Outputs

```text
ablation_ds2net_mtaswin_binary_roi_mask_focal_no_alpha/checkpoints/
classification_results/ablation_ds2net_mtaswin_binary_roi_mask_focal_no_alpha/
```

Compare against the alpha-weighted focal reference:

- Accuracy: 0.9787
- Balanced accuracy: 0.7634
- Sensitivity: 0.5333
- Malignant F1: 0.6154
- ROC-AUC: 0.8689
- PR-AUC: 0.5510

A reduction in minority-class metrics without alpha supports retaining
inverse-frequency class weighting.
