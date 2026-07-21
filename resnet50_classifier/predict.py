from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from classification_common.config import get_config
from classification_common.dataset import build_eval_transform
from classification_common.roi_utils import (
    crop_roi,
    read_binary_mask,
    read_rgb_image,
)
from resnet50_classifier.model import build_resnet50


def prepare_input_image(
    image_path: Path,
    input_mode: str,
    mask_path: Optional[Path],
    roi_padding: float,
    mask_threshold: float,
    use_masked_roi: bool,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = read_rgb_image(image_path)
    height, width = image.shape[:2]
    full_box = (0, 0, width, height)

    if input_mode == "full_image":
        return image, full_box

    if mask_path is None:
        raise ValueError(
            "A mask path is required for ground_truth_roi and "
            "predicted_roi prediction."
        )

    mask = read_binary_mask(
        mask_path,
        threshold=mask_threshold,
    )

    return crop_roi(
        image=image,
        mask=mask,
        padding=roi_padding,
        use_masked_roi=use_masked_roi,
    )


@torch.no_grad()
def predict_single_image(
    image_path: Path,
    checkpoint_path: Path,
    input_mode: str,
    mask_path: Optional[Path] = None,
    top_k: int = 3,
) -> dict:
    config = get_config()
    config.input_mode = input_mode

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.device,
        weights_only=False,
    )

    checkpoint_mode = checkpoint.get("input_mode")
    if checkpoint_mode and checkpoint_mode != input_mode:
        print(
            "Warning: checkpoint was trained with input mode "
            f"{checkpoint_mode!r}, but prediction uses "
            f"{input_mode!r}."
        )

    num_classes = checkpoint.get(
        "num_classes",
        config.num_classes,
    )
    class_names = checkpoint.get(
        "class_names",
        config.class_names,
    )
    class_names = {
        int(index): name
        for index, name in class_names.items()
    }

    model = build_resnet50(
        num_classes=num_classes,
        pretrained=False,
        dropout=config.dropout,
    ).to(config.device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_array, roi_box = prepare_input_image(
        image_path=image_path,
        input_mode=input_mode,
        mask_path=mask_path,
        roi_padding=config.roi_padding,
        mask_threshold=config.mask_threshold,
        use_masked_roi=config.use_masked_roi,
    )

    transform = build_eval_transform(
        checkpoint.get("image_size", config.image_size)
    )

    tensor = transform(
        Image.fromarray(image_array)
    ).unsqueeze(0).to(config.device)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

    top_k = min(top_k, num_classes)
    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=top_k,
    )

    ranked_predictions = []

    for probability, class_index in zip(
        top_probabilities.cpu().tolist(),
        top_indices.cpu().tolist(),
    ):
        ranked_predictions.append(
            {
                "class_index": int(class_index),
                "class_name": class_names[int(class_index)],
                "probability": float(probability),
                "percentage": float(probability * 100.0),
            }
        )

    result = {
        "image_path": str(image_path),
        "mask_path": str(mask_path) if mask_path else None,
        "input_mode": input_mode,
        "roi_box_xyxy": list(roi_box),
        "prediction": ranked_predictions[0],
        "top_predictions": ranked_predictions,
    }

    return result


def save_roi_preview(
    image_path: Path,
    mask_path: Optional[Path],
    input_mode: str,
    output_path: Path,
) -> None:
    config = get_config()

    roi, _ = prepare_input_image(
        image_path=image_path,
        input_mode=input_mode,
        mask_path=mask_path,
        roi_padding=config.roi_padding,
        mask_threshold=config.mask_threshold,
        use_masked_roi=config.use_masked_roi,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bgr_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_roi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict ovarian tumor class using ResNet50."
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to an ultrasound image.",
    )

    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help=(
            "Path to ground-truth or predicted mask. Required for "
            "ROI input modes."
        ),
    )

    parser.add_argument(
        "--input-mode",
        choices=[
            "full_image",
            "ground_truth_roi",
            "predicted_roi",
        ],
        default="ground_truth_roi",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--save-roi",
        type=Path,
        default=None,
        help="Optional path to save the ROI preview.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()

    checkpoint_path = (
        args.checkpoint
        if args.checkpoint is not None
        else config.best_checkpoint_path
    )

    result = predict_single_image(
        image_path=args.image,
        checkpoint_path=checkpoint_path,
        input_mode=args.input_mode,
        mask_path=args.mask,
        top_k=args.top_k,
    )

    if args.save_roi is not None:
        save_roi_preview(
            image_path=args.image,
            mask_path=args.mask,
            input_mode=args.input_mode,
            output_path=args.save_roi,
        )

    print("=" * 72)
    print("Ovarian Tumor Classification Prediction")
    print("=" * 72)
    print(
        f"Predicted class: "
        f"{result['prediction']['class_name']}"
    )
    print(
        f"Confidence     : "
        f"{result['prediction']['percentage']:.2f}%"
    )

    print("\nTop predictions")
    print("-" * 72)
    for prediction in result["top_predictions"]:
        print(
            f"[{prediction['class_index']}] "
            f"{prediction['class_name']}: "
            f"{prediction['percentage']:.2f}%"
        )

    print("\nComplete result")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
