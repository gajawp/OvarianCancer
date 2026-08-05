"""
Evaluate nnU-Net predicted masks against MMOTU GT on the official 469 test set.

Reports the SAME foreground metrics as the classmate's benchmark (per-image
foreground Dice / IoU, averaged) PLUS mIoU = mean(background IoU, foreground IoU)
so the number lines up with the MMOTU paper / follow-ups.

Usage:
  python nnunet_mmotu/evaluate_preds.py --preds preds --gt <raw>/Dataset501_MMOTU2d/test_gt
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def load_binary(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Could not read: {path}")
    return (m > 0).astype(np.uint8)


def per_image_metrics(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6):
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    tn = int(np.logical_and(pred == 0, gt == 0).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())
    fg_dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    fg_iou = (tp + smooth) / (tp + fp + fn + smooth)
    bg_iou = (tn + smooth) / (tn + fp + fn + smooth)
    miou = 0.5 * (fg_iou + bg_iou)
    return fg_dice, fg_iou, bg_iou, miou


def main():
    parser = argparse.ArgumentParser(description="Evaluate nnU-Net preds vs MMOTU GT (official test).")
    parser.add_argument("--preds", type=Path, required=True, help="nnUNetv2_predict output dir (<id>.png)")
    parser.add_argument("--gt", type=Path, required=True, help="test_gt dir (<id>.png)")
    parser.add_argument("--out-csv", type=Path, default=Path("nnunet_mmotu/nnunet_test_metrics.csv"))
    args = parser.parse_args()

    gt_files = sorted(args.gt.glob("*.png"))
    if not gt_files:
        raise SystemExit(f"No GT masks found in {args.gt}")

    rows, fgd, fgi, bgi, mi = [], [], [], [], []
    missing = 0
    for gt_path in gt_files:
        pred_path = args.preds / gt_path.name
        if not pred_path.exists():
            missing += 1
            continue
        d, i, b, m = per_image_metrics(load_binary(pred_path), load_binary(gt_path))
        rows.append((gt_path.stem, d, i, b, m))
        fgd.append(d); fgi.append(i); bgi.append(b); mi.append(m)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "fg_dice", "fg_iou", "bg_iou", "miou"])
        w.writerows(rows)

    n = len(rows)
    print(f"Evaluated {n} images (missing preds: {missing})")
    print("-" * 44)
    print(f"Foreground Dice : {np.mean(fgd):.4f}")
    print(f"Foreground IoU  : {np.mean(fgi):.4f}")
    print(f"Background IoU  : {np.mean(bgi):.4f}")
    print(f"mIoU (bg+fg)    : {np.mean(mi):.4f}")
    print("-" * 44)
    print(f"Per-image CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
