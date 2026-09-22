# Ablation A1: Dual-Context MTA-Swin Without Explicit Mask

This package is a controlled ablation of the attached `224x224` dual-context
binary model. It removes only the explicit mask encoder and mask-feature input.

## Architecture

- DS2Net RGB bounding-box ROI -> Swin-T + token-attention pooling
- Original full ultrasound image -> ResNet18 context branch
- ROI and context feature concatenation -> binary classifier
- No explicit mask tensor or mask encoder is supplied to the classifier

The DS2Net mask files are still read so the sample list, ROI source,
synchronized preprocessing, and metadata stay identical to the baseline. The
transformed mask is discarded before the batch is returned.

## Controlled settings retained

- Input size: 224x224
- Batch size: 1
- Gradient accumulation: 4
- Focal loss with inverse-frequency alpha
- Same augmentation configuration and official split files
- Same pretrained backbones, optimizer, scheduler, and checkpoint selection
- Checkpoint ranking: malignant F1, sensitivity, then balanced accuracy

## Commands

Run these from the `Ovarian_Cancer` project root:

```bash
python -m ablation_ds2net_mtaswin_binary_context224_no_mask.test_model
```

```bash
caffeinate -i python -m ablation_ds2net_mtaswin_binary_context224_no_mask.train
```

```bash
python -m ablation_ds2net_mtaswin_binary_context224_no_mask.evaluate
```

## Expected dataset paths

```text
datasets/OTU_2d/images/
datasets/OTU_2d_ds2net_rgb_mask_padding20/train/images/
datasets/OTU_2d_ds2net_rgb_mask_padding20/train/masks/
datasets/OTU_2d_ds2net_rgb_mask_padding20/val/images/
datasets/OTU_2d_ds2net_rgb_mask_padding20/val/masks/
```

## Outputs

```text
ablation_ds2net_mtaswin_binary_context224_no_mask/checkpoints/
classification_results/ablation_ds2net_mtaswin_binary_context224_no_mask/
```

Compare the metrics with the exact mask-aware context224 baseline. A reduction
in malignant F1, sensitivity, balanced accuracy, or PR-AUC after removing the
mask indicates that explicit mask features helped.
