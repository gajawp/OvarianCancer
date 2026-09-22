# Ablation A2: ROI + Explicit Mask Without Full-Image Context

This is a controlled ablation of the attached `224x224` dual-context binary
model. It removes only the ResNet18 full-image context branch.

## Retained architecture

- DS2Net RGB bounding-box ROI -> Swin-T + token-attention pooling
- Explicit DS2Net mask -> CNN mask encoder
- ROI and mask feature concatenation -> binary classifier

## Controlled settings retained

- Input size: 224x224
- Batch size: 1
- Gradient accumulation: 4
- Focal loss with inverse-frequency alpha
- Synchronized RGB ROI and mask augmentation
- Nearest-neighbor mask resizing
- Same official split files, optimizer, scheduler, pretrained Swin backbone,
  early stopping, and checkpoint-selection hierarchy

## Commands

Run from the `Ovarian_Cancer` project root:

```bash
python -m ablation_ds2net_mtaswin_binary_roi_mask_no_context.test_model
```

```bash
caffeinate -i python -m ablation_ds2net_mtaswin_binary_roi_mask_no_context.train
```

```bash
python -m ablation_ds2net_mtaswin_binary_roi_mask_no_context.evaluate
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
ablation_ds2net_mtaswin_binary_roi_mask_no_context/checkpoints/
classification_results/ablation_ds2net_mtaswin_binary_roi_mask_no_context/
```

Compare these results with the exact attached dual-context mask-aware baseline.
A performance reduction after context removal supports the contribution of
full-image contextual information.
