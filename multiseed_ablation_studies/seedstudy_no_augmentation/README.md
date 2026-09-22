# Ablation A3: ROI + Explicit Mask Without Augmentation

This experiment measures the contribution of synchronized ROI-and-mask data
augmentation. It uses the successful ROI-and-mask model without a context
branch as the fixed reference and changes only:

```python
USE_AUGMENTATION = False
```

## Architecture retained

- DS2Net RGB bounding-box ROI -> Swin-T + token-attention pooling
- Explicit DS2Net mask -> CNN mask encoder
- ROI and mask feature concatenation -> binary classifier
- No full-image context branch

## Training settings retained

- Input size: 224x224
- Batch size: 1
- Gradient accumulation: 4
- Focal loss with inverse-frequency alpha
- Nearest-neighbor mask resizing
- Same official split files, random seed, pretrained Swin backbone, optimizer,
  scheduler, early stopping, and checkpoint-selection hierarchy

All horizontal flips, affine transforms, brightness adjustments, and contrast
adjustments are disabled during training. Deterministic resizing and
normalization are still applied.

## Commands

Run from the `Ovarian_Cancer` project root:

```bash
python -m ablation_ds2net_mtaswin_binary_roi_mask_no_augmentation.test_model
```

```bash
caffeinate -i python -m ablation_ds2net_mtaswin_binary_roi_mask_no_augmentation.train
```

```bash
python -m ablation_ds2net_mtaswin_binary_roi_mask_no_augmentation.evaluate
```

## Expected dataset paths

```text
datasets/OTU_2d_ds2net_rgb_mask_padding20/train/images/
datasets/OTU_2d_ds2net_rgb_mask_padding20/train/masks/
datasets/OTU_2d_ds2net_rgb_mask_padding20/val/images/
datasets/OTU_2d_ds2net_rgb_mask_padding20/val/masks/
```

## Outputs

```text
ablation_ds2net_mtaswin_binary_roi_mask_no_augmentation/checkpoints/
classification_results/ablation_ds2net_mtaswin_binary_roi_mask_no_augmentation/
```

Compare the results against the augmented ROI-and-mask reference:

- Accuracy: 0.9787
- Balanced accuracy: 0.7634
- Sensitivity: 0.5333
- Malignant F1: 0.6154
- ROC-AUC: 0.8689
- PR-AUC: 0.5510

A performance reduction without augmentation supports the contribution of
synchronized ROI-and-mask augmentation.
