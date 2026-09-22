# Final Relevant Experiment Summary

Metrics were collected from each experiment's `metric_summary.csv`.

## Multi-seed mean ± standard deviation

| Experiment | Metric | Seeds | Mean ± SD |
| --- | --- | --- | --- |
| reference | accuracy | 3 | 0.9723 ± 0.0077 |
| reference | balanced_accuracy | 3 | 0.7386 ± 0.0365 |
| reference | malignant_precision | 3 | 0.6128 ± 0.1489 |
| reference | sensitivity | 3 | 0.4889 ± 0.0770 |
| reference | specificity | 3 | 0.9883 ± 0.0089 |
| reference | malignant_f1 | 3 | 0.5334 ± 0.0714 |
| reference | macro_f1 | 3 | 0.7596 ± 0.0373 |
| reference | negative_predictive_value | 3 | 0.9832 ± 0.0024 |
| reference | roc_auc | 3 | 0.8688 ± 0.0077 |
| reference | pr_auc | 3 | 0.4855 ± 0.0571 |
| no_augmentation | accuracy | 3 | 0.9730 ± 0.0062 |
| no_augmentation | balanced_accuracy | 3 | 0.6637 ± 0.0295 |
| no_augmentation | malignant_precision | 3 | 0.7650 ± 0.2757 |
| no_augmentation | sensitivity | 3 | 0.3333 ± 0.0667 |
| no_augmentation | specificity | 3 | 0.9941 ± 0.0083 |
| no_augmentation | malignant_f1 | 3 | 0.4419 ± 0.0299 |
| no_augmentation | macro_f1 | 3 | 0.7140 ± 0.0156 |
| no_augmentation | negative_predictive_value | 3 | 0.9783 ± 0.0020 |
| no_augmentation | roc_auc | 3 | 0.8395 ± 0.0088 |
| no_augmentation | pr_auc | 3 | 0.4219 ± 0.0187 |
| roi_only | accuracy | 3 | 0.9723 ± 0.0056 |
| roi_only | balanced_accuracy | 3 | 0.7386 ± 0.0206 |
| roi_only | malignant_precision | 3 | 0.5899 ± 0.1078 |
| roi_only | sensitivity | 3 | 0.4889 ± 0.0385 |
| roi_only | specificity | 3 | 0.9883 ± 0.0051 |
| roi_only | malignant_f1 | 3 | 0.5326 ± 0.0632 |
| roi_only | macro_f1 | 3 | 0.7591 ± 0.0330 |
| roi_only | negative_predictive_value | 3 | 0.9832 ± 0.0013 |
| roi_only | roc_auc | 3 | 0.8687 ± 0.0413 |
| roi_only | pr_auc | 3 | 0.4774 ± 0.0611 |

## Skipped experiments

- `integration_ds2net_maskaware_mtaswin`: config import failed: No module named 'integration_ds2net_maskaware_mtaswin'
- `integration_ds2net_maskaware_mtaswin_aug`: config import failed: No module named 'integration_ds2net_maskaware_mtaswin_aug'
- `integration_ds2net_maskaware_mtaswin_aug_balanced`: config import failed: No module named 'integration_ds2net_maskaware_mtaswin_aug_balanced'
- `integration_ds2net_maskaware_mtaswin_aug_focal`: config import failed: No module named 'integration_ds2net_maskaware_mtaswin_aug_focal'
- `integration_ds2net_maskaware_mtaswin_aug_focal_binary`: config import failed: No module named 'integration_ds2net_maskaware_mtaswin_aug_focal_binary'
- `integration_ds2net_maskaware_mtaswin_aug_focal_binary_malignant_aug`: config import failed: No module named 'integration_ds2net_maskaware_mtaswin_aug_focal_binary_malignant_aug'
- `integration_ds2net_maskaware_mtaswin_binary_context384`: config import failed: No module named 'integration_ds2net_maskaware_mtaswin_binary_context384'
- `integration_ds2net_maskaware_mtaswin_binary_context224`: config import failed: No module named 'integration_ds2net_maskaware_mtaswin_binary_context224'
- `integration_ds2net_maskaware_mtaswin_binary_context224_batch2`: config import failed: No module named 'integration_ds2net_maskaware_mtaswin_binary_context224_batch2'
- `integration_ds2net_maskaware_mtaswin_binary_context224_batch2_accum2`: config import failed: No module named 'integration_ds2net_maskaware_mtaswin_binary_context224_batch2_accum2'
- `integration_ds2net_maskaware_mtaswin_binary_context224_lightcnn`: config import failed: No module named 'integration_ds2net_maskaware_mtaswin_binary_context224_lightcnn'
- `integration_ds2net_maskaware_mtaswin_binary_context224_lightcnn_groupnorm`: config import failed: No module named 'integration_ds2net_maskaware_mtaswin_binary_context224_lightcnn_groupnorm'
- `ablation_ds2net_mtaswin_binary_context224_no_mask`: config import failed: No module named 'ablation_ds2net_mtaswin_binary_context224_no_mask'
- `ablation_ds2net_mtaswin_binary_roi_mask_no_context`: config import failed: No module named 'ablation_ds2net_mtaswin_binary_roi_mask_no_context'
- `ablation_ds2net_mtaswin_binary_roi_mask_no_augmentation`: config import failed: No module named 'ablation_ds2net_mtaswin_binary_roi_mask_no_augmentation'
- `ablation_ds2net_mtaswin_binary_roi_mask_cross_entropy`: config import failed: No module named 'ablation_ds2net_mtaswin_binary_roi_mask_cross_entropy'
- `ablation_ds2net_mtaswin_binary_roi_only`: config import failed: No module named 'ablation_ds2net_mtaswin_binary_roi_only'
- `ablation_ds2net_mtaswin_binary_roi_mask_bilinear_resize`: config import failed: No module named 'ablation_ds2net_mtaswin_binary_roi_mask_bilinear_resize'
- `ablation_ds2net_mtaswin_binary_roi_mask_focal_no_alpha`: config import failed: No module named 'ablation_ds2net_mtaswin_binary_roi_mask_focal_no_alpha'
