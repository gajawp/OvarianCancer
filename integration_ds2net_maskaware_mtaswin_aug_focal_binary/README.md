# Binary Augmentation + Focal Loss Experiment

Task:
- Class 0: original OTU labels 0-6, grouped as non-malignant/other
- Class 1: original OTU label 7, high-grade serous carcinoma

Model:
- DS2Net-generated RGB ROI
- Explicit segmentation mask
- Mask-aware MTA-Swin
- Two output logits
- Synchronized RGB/mask augmentation
- Focal loss with inverse-frequency class alpha
- No balanced sampler
- Best checkpoint selected using malignant-class F1

Commands:
python -m integration_ds2net_maskaware_mtaswin_aug_focal_binary.test_model
caffeinate -i python -m integration_ds2net_maskaware_mtaswin_aug_focal_binary.train
python -m integration_ds2net_maskaware_mtaswin_aug_focal_binary.evaluate
