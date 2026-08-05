"""
Convert MMOTU OTU_2d (binary lesion segmentation) into nnU-Net v2 raw format,
using the OFFICIAL patient-level split (train.txt=1000 / val.txt=469).

Output layout (under --out, default $nnUNet_raw/Dataset501_MMOTU2d):
  imagesTr/<id>_0000.png     1000 training images (grayscale, 1 channel)
  labelsTr/<id>.png          1000 training masks   (0=background, 1=tumor)
  imagesTs/<id>_0000.png     469 test images (for nnUNetv2_predict)
  test_gt/<id>.png           469 test GT masks (held out, for our evaluation)
  dataset.json

Then:
  export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
  nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
  nnUNetv2_train 501 2d 0            # add -tr nnUNetTrainer_250epochs to go faster
  nnUNetv2_predict -i <raw>/Dataset501_MMOTU2d/imagesTs -o preds -d 501 -c 2d -f 0
  python nnunet_mmotu/evaluate_preds.py --preds preds --gt <raw>/Dataset501_MMOTU2d/test_gt
"""

from __future__ import annotations  # allow "X | None" / list[str] on Python 3.9

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def read_ids(list_file: Path) -> list[str]:
    ids = []
    for line in Path(list_file).read_text().splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if token:
            ids.append(Path(token).stem)  # "658.JPG" or "658" -> "658"
    return ids


def find_file(directory: Path, stem: str, suffixes) -> Path | None:
    for suffix in suffixes:
        p = directory / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


def save_gray_png(src_image: Path, dst: Path) -> None:
    img = cv2.imread(str(src_image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {src_image}")
    cv2.imwrite(str(dst), img)


def save_binary_mask_png(src_mask: Path, dst: Path) -> None:
    mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {src_mask}")
    binary = (mask > 0).astype(np.uint8)  # 0 / 1 class labels
    cv2.imwrite(str(dst), binary)


def main():
    parser = argparse.ArgumentParser(description="MMOTU OTU_2d -> nnU-Net v2 raw format.")
    parser.add_argument("--otu-root", type=Path, default=Path("OTU_2d"), help="Folder with images/ annotations/ train.txt val.txt")
    parser.add_argument("--out", type=Path, default=None, help="Output dataset dir (default $nnUNet_raw/Dataset501_MMOTU2d)")
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", type=str, default="MMOTU2d")
    parser.add_argument("--mask-suffix", type=str, default="_binary.PNG", help="Suffix mapping image id -> mask file")
    args = parser.parse_args()

    image_dir = args.otu_root / "images"
    mask_dir = args.otu_root / "annotations"
    train_ids = read_ids(args.otu_root / "train.txt")
    test_ids = read_ids(args.otu_root / "val.txt")
    print(f"train={len(train_ids)}  test={len(test_ids)}")

    if args.out is not None:
        out = args.out
    else:
        raw = os.environ.get("nnUNet_raw")
        if not raw:
            raise SystemExit("Set --out or the nnUNet_raw environment variable.")
        out = Path(raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"

    for sub in ("imagesTr", "labelsTr", "imagesTs", "test_gt"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    img_suffixes = (".JPG", ".jpg", ".jpeg", ".png", ".PNG")

    def convert(ids, image_out, label_out, label_name):
        missing = 0
        for stem in ids:
            img = find_file(image_dir, stem, img_suffixes)
            msk = find_file(mask_dir, stem, (args.mask_suffix,))
            if img is None or msk is None:
                missing += 1
                print(f"  [skip] {stem}: image={img is not None} mask={msk is not None}")
                continue
            save_gray_png(img, out / image_out / f"{stem}_0000.png")
            save_binary_mask_png(msk, out / label_out / f"{stem}.png")
        return missing

    print("Converting training set...")
    m1 = convert(train_ids, "imagesTr", "labelsTr", "train")
    print("Converting test set...")
    m2 = convert(test_ids, "imagesTs", "test_gt", "test")

    dataset_json = {
        "channel_names": {"0": "ultrasound"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(train_ids) - m1,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (out / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    print(f"\nDone. Dataset at: {out}")
    print(f"  imagesTr/labelsTr: {len(train_ids) - m1}  |  imagesTs/test_gt: {len(test_ids) - m2}")
    if m1 or m2:
        print(f"  [WARNING] skipped {m1} train + {m2} test (missing image/mask).")


if __name__ == "__main__":
    main()
