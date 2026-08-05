# nnU-Net SOTA reference on MMOTU OTU_2d (2D lesion segmentation)

Runs the self-configuring **nnU-Net v2** as a SOTA segmentation reference on the
**official 1000/469 patient-level split**, so we know how far our custom models
are from SOTA before building a seg→cls pipeline.

Metric is reported the same way as our benchmark (per-image foreground Dice/IoU)
plus **mIoU (bg+fg)** for comparability with the MMOTU paper / follow-ups.

## 1. Install + env

```bash
pip install nnunetv2
export nnUNet_raw=$PWD/nnunet_mmotu/nnUNet_raw
export nnUNet_preprocessed=$PWD/nnunet_mmotu/nnUNet_preprocessed
export nnUNet_results=$PWD/nnunet_mmotu/nnUNet_results
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
```

## 2. Convert MMOTU -> nnU-Net format (official split)

```bash
python nnunet_mmotu/convert_to_nnunet.py --otu-root OTU_2d
# -> $nnUNet_raw/Dataset501_MMOTU2d/{imagesTr,labelsTr,imagesTs,test_gt,dataset.json}
```

## 3. Plan + preprocess + train (2d)

```bash
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
nnUNetv2_train 501 2d 0                      # fold 0
# faster first estimate (recommended to start):
# nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_250epochs
```

Default trainer is 1000 epochs and a single 2D fold can take a few hours. Start
with `-tr nnUNetTrainer_250epochs` (or `_50epochs`) and fold 0; run all 5 folds
(`-f 0 1 2 3 4`) only if the estimate is promising.

## 4. Predict on the official 469 test + evaluate

```bash
nnUNetv2_predict -i "$nnUNet_raw/Dataset501_MMOTU2d/imagesTs" \
                 -o nnunet_mmotu/preds -d 501 -c 2d -f 0

python nnunet_mmotu/evaluate_preds.py \
    --preds nnunet_mmotu/preds \
    --gt   "$nnUNet_raw/Dataset501_MMOTU2d/test_gt"
```

Prints foreground Dice / IoU / background IoU / mIoU and writes a per-image CSV.

## Notes

- Input is converted to **grayscale, 1 channel** (ultrasound is grayscale);
  switch to RGB later if wanted.
- The 1000 training images are used for nnU-Net's internal cross-validation; the
  469 test images are the held-out official test set (never seen in training).
- To compare with our custom models: their reported numbers are per-image
  **foreground** Dice/IoU; use those columns, and use `mIoU` to line up with the
  literature (~90 mIoU on OTU_2d).
- Not tracked in git: `nnUNet_raw/`, `nnUNet_preprocessed/`, `nnUNet_results/`,
  `preds/` (generated/large).
