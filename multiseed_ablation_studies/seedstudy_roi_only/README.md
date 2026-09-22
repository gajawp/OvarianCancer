# Ablation A5: RGB ROI Only

This experiment isolates the contribution of the explicit segmentation-mask
encoder in the final no-context architecture. It removes the mask encoder and
feeds only the DS2Net-derived RGB bounding-box ROI to MTA-Swin.

## Architecture

- DS2Net RGB bounding-box ROI -> Swin-T + token-attention pooling
- ROI feature -> binary classifier
- No explicit mask encoder
- No full-image context branch

The DS2Net-derived ROI images and the same split files are retained. Mask names
remain in sample metadata for consistent result CSVs, but mask pixels are not
loaded or supplied to the classifier.

## Settings retained

- Synchronized-reference augmentation probabilities applied to the RGB ROI
- Input size: 224x224
- Batch size: 1
- Gradient accumulation: 4
- Focal loss with inverse-frequency alpha
- Same official split files, random seed, pretrained Swin backbone, optimizer,
  scheduler, early stopping, and checkpoint-selection hierarchy

## Commands

Run from the `Ovarian_Cancer` project root:

```bash
python -m ablation_ds2net_mtaswin_binary_roi_only.test_model
```

```bash
caffeinate -i python -m ablation_ds2net_mtaswin_binary_roi_only.train
```

```bash
python -m ablation_ds2net_mtaswin_binary_roi_only.evaluate
```

## Outputs

```text
ablation_ds2net_mtaswin_binary_roi_only/checkpoints/
classification_results/ablation_ds2net_mtaswin_binary_roi_only/
```

Compare against the ROI-plus-mask reference:

- Accuracy: 0.9787
- Balanced accuracy: 0.7634
- Sensitivity: 0.5333
- Malignant F1: 0.6154
- ROC-AUC: 0.8689
- PR-AUC: 0.5510

A reduction in malignant F1, sensitivity, balanced accuracy, or PR-AUC supports
the contribution of the explicit mask branch.

Note: a balanced-sampler ablation was not created because the supplied
reference already uses `shuffle=True` without a weighted or balanced sampler.
