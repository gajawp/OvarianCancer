# Ablation A4: Focal Loss Replaced With Cross-Entropy

This experiment measures the contribution of focal loss to imbalanced binary
malignant classification. It uses the augmented ROI-and-mask reference model
and changes only the loss function.

## Loss change

Reference:

```python
FocalLoss(gamma=2.0, alpha=inverse_frequency_weights)
```

A4:

```python
nn.CrossEntropyLoss()
```

The cross-entropy loss is intentionally unweighted. Adding class weights would
introduce another imbalance-handling mechanism and would not cleanly isolate
the contribution of focal loss.

## Settings retained

- DS2Net RGB ROI and explicit mask encoder
- No full-image context branch
- Synchronized ROI-and-mask augmentation enabled
- Nearest-neighbor mask resizing
- Input size: 224x224
- Batch size: 1
- Gradient accumulation: 4
- Same official split files, random seed, pretrained Swin backbone, optimizer,
  scheduler, early stopping, and checkpoint-selection hierarchy

## Commands

Run from the `Ovarian_Cancer` project root:

```bash
python -m ablation_ds2net_mtaswin_binary_roi_mask_cross_entropy.test_model
```

```bash
caffeinate -i python -m ablation_ds2net_mtaswin_binary_roi_mask_cross_entropy.train
```

```bash
python -m ablation_ds2net_mtaswin_binary_roi_mask_cross_entropy.evaluate
```

## Outputs

```text
ablation_ds2net_mtaswin_binary_roi_mask_cross_entropy/checkpoints/
classification_results/ablation_ds2net_mtaswin_binary_roi_mask_cross_entropy/
```

Compare against the focal-loss reference:

- Accuracy: 0.9787
- Balanced accuracy: 0.7634
- Sensitivity: 0.5333
- Malignant F1: 0.6154
- ROC-AUC: 0.8689
- PR-AUC: 0.5510

A reduction in malignant F1, sensitivity, balanced accuracy, or PR-AUC with
cross-entropy supports retaining focal loss.
