# Binary Dual-Context Mask-Aware MTA-Swin at 384×384

## Architecture

- DS2Net RGB ROI -> Swin-T + token-attention pooling
- Original full ultrasound image -> ResNet18 context branch
- Explicit DS2Net mask -> CNN mask encoder
- Feature concatenation -> binary classifier

## Training

- Input size: 384×384
- Batch size: 1
- Gradient accumulation: 4
- Effective batch size: 4
- Focal loss with inverse-frequency alpha
- No balanced sampler
- No offline malignant-only augmentation
- Best checkpoint selected by malignant F1, then sensitivity, then balanced accuracy

## Commands

```bash
python -m integration_ds2net_maskaware_mtaswin_binary_context384.test_model
```

```bash
caffeinate -i python -m integration_ds2net_maskaware_mtaswin_binary_context384.train
```

```bash
python -m integration_ds2net_maskaware_mtaswin_binary_context384.evaluate
```

## Expected project paths

```text
datasets/OTU_2d/images/
datasets/OTU_2d_ds2net_rgb_mask_padding20/train/images/
datasets/OTU_2d_ds2net_rgb_mask_padding20/train/masks/
datasets/OTU_2d_ds2net_rgb_mask_padding20/val/images/
datasets/OTU_2d_ds2net_rgb_mask_padding20/val/masks/
```
