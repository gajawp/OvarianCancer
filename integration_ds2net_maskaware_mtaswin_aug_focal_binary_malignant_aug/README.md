# Binary MTA-Swin with Malignant-Only Augmentation

This experiment creates four synchronized RGB-and-mask copies for every
malignant training image.

Expected expansion:
- Original malignant training images: 38
- New malignant images: 152
- Total malignant training images: 190
- Expanded training set: 1,152 images

Run:
1. python -m integration_ds2net_maskaware_mtaswin_aug_focal_binary_malignant_aug.generate_malignant_augmented_dataset
2. python -m integration_ds2net_maskaware_mtaswin_aug_focal_binary_malignant_aug.test_model
3. caffeinate -i python -m integration_ds2net_maskaware_mtaswin_aug_focal_binary_malignant_aug.train
4. python -m integration_ds2net_maskaware_mtaswin_aug_focal_binary_malignant_aug.evaluate
