# Ablation A6: Bilinear Mask Resizing

This experiment measures the effect of mask-resizing interpolation. It uses the
augmented ROI-and-mask reference model and changes only the initial mask resize
operation.

## Single change

Reference:

```python
TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
```

A6:

```python
TF.resize(mask, size, interpolation=InterpolationMode.BILINEAR)
mask_tensor = (TF.to_tensor(mask) >= 0.5).float()
```

Bilinear interpolation creates intermediate grayscale boundary values before
the mask is thresholded. Nearest-neighbor interpolation preserves discrete
class labels directly. Affine augmentation still uses nearest-neighbor mask
interpolation so this experiment changes only the initial resizing operation.

## Settings retained

- DS2Net RGB ROI and explicit mask encoder
- No full-image context branch
- Synchronized ROI-and-mask augmentation enabled
- Binary focal loss with inverse-frequency alpha
- Input size: 224x224
- Batch size: 1
- Gradient accumulation: 4
- Same official split files, random seed, pretrained Swin backbone, optimizer,
  scheduler, early stopping, and checkpoint-selection hierarchy

## Commands

Run from the `Ovarian_Cancer` project root:

```bash
python -m ablation_ds2net_mtaswin_binary_roi_mask_bilinear_resize.test_model
```

```bash
caffeinate -i python -m ablation_ds2net_mtaswin_binary_roi_mask_bilinear_resize.train
```

```bash
python -m ablation_ds2net_mtaswin_binary_roi_mask_bilinear_resize.evaluate
```

## Outputs

```text
ablation_ds2net_mtaswin_binary_roi_mask_bilinear_resize/checkpoints/
classification_results/ablation_ds2net_mtaswin_binary_roi_mask_bilinear_resize/
```

Compare against the nearest-neighbor reference:

- Accuracy: 0.9787
- Balanced accuracy: 0.7634
- Sensitivity: 0.5333
- Malignant F1: 0.6154
- ROC-AUC: 0.8689
- PR-AUC: 0.5510

If bilinear resizing reduces classification performance, the result supports
nearest-neighbor resizing for preserving segmentation-mask boundaries.
