from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common import config as segmentation_config
from integration_ds2net_maskaware_mtaswin import config
from integration_ds2net_maskaware_mtaswin.load_ds2net import (
    load_trained_ds2net,
)
from integration_ds2net_maskaware_mtaswin.roi_utils import (
    extract_rgb_roi_and_mask,
)


def read_split(split_file: Path) -> list[tuple[str, int]]:
    samples = []

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with split_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid line {line_number} in {split_file}: {line!r}"
                )

            samples.append((parts[0], int(parts[1])))

    return samples


def preprocess_for_ds2net(
    image_bgr: np.ndarray,
) -> torch.Tensor:
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    resized_rgb = cv2.resize(
        image_rgb,
        (
            segmentation_config.IMAGE_SIZE,
            segmentation_config.IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    normalized = resized_rgb.astype(np.float32) / 255.0
    chw = normalized.transpose(2, 0, 1)

    return torch.from_numpy(chw).float().unsqueeze(0)


def extract_final_logits(model_output) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, (tuple, list)):
        if not model_output:
            raise ValueError("DS2Net returned an empty output.")
        return model_output[0]

    if isinstance(model_output, dict):
        for key in ("out", "logits", "prediction", "pred"):
            if key in model_output:
                return model_output[key]

    raise TypeError(
        f"Unsupported DS2Net output type: {type(model_output).__name__}"
    )


@torch.inference_mode()
def predict_ds2net_mask(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    original_height, original_width = image_bgr.shape[:2]

    image_tensor = preprocess_for_ds2net(image_bgr).to(device)
    final_logits = extract_final_logits(model(image_tensor))

    binary_mask = (
        torch.sigmoid(final_logits)
        >= segmentation_config.PREDICTION_THRESHOLD
    ).to(torch.uint8)

    binary_mask = (
        binary_mask.squeeze().cpu().numpy().astype(np.uint8)
    )

    return cv2.resize(
        binary_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )


def process_split(
    split_name: str,
    split_file: Path,
    rgb_dir: Path,
    mask_dir: Path,
    output_list: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    samples = read_split(split_file)

    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    output_list.parent.mkdir(parents=True, exist_ok=True)

    output_lines = []
    valid_masks = 0
    fallbacks = 0
    missing = 0
    failed = 0

    progress = tqdm(
        samples,
        desc=f"Generating {split_name} RGB+mask samples",
    )

    for image_name, label in progress:
        image_path = config.ORIGINAL_IMAGE_DIR / image_name

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            missing += 1
            print(f"\nCould not read image: {image_path}")
            continue

        predicted_mask = predict_ds2net_mask(
            model=model,
            image_bgr=image_bgr,
            device=device,
        )

        rgb_roi, mask_roi, _, mask_found = extract_rgb_roi_and_mask(
            image_bgr=image_bgr,
            predicted_mask=predicted_mask,
            padding=config.ROI_PADDING,
            minimum_area=config.MINIMUM_COMPONENT_AREA,
        )

        if mask_found:
            valid_masks += 1
        else:
            fallbacks += 1

        rgb_path = rgb_dir / image_name
        mask_name = f"{Path(image_name).stem}_mask.png"
        mask_path = mask_dir / mask_name

        rgb_saved = cv2.imwrite(str(rgb_path), rgb_roi)
        mask_saved = cv2.imwrite(
            str(mask_path),
            mask_roi.astype(np.uint8) * 255,
        )

        if not rgb_saved or not mask_saved:
            failed += 1
            print(f"\nCould not save outputs for {image_name}")
            continue

        output_lines.append(f"{image_name} {mask_name} {label}\n")

        progress.set_postfix(
            valid=valid_masks,
            fallback=fallbacks,
        )

    with output_list.open("w", encoding="utf-8") as file:
        file.writelines(output_lines)

    print()
    print("=" * 72)
    print(f"{split_name.upper()} SUMMARY")
    print("=" * 72)
    print("Input samples:", len(samples))
    print("Saved samples:", len(output_lines))
    print("Valid masks:", valid_masks)
    print("Whole-image fallbacks:", fallbacks)
    print("Missing images:", missing)
    print("Failed saves:", failed)
    print("RGB directory:", rgb_dir)
    print("Mask directory:", mask_dir)
    print("Output list:", output_list)


def main() -> None:
    device = torch.device(segmentation_config.DEVICE)

    print("=" * 72)
    print("DS2NET RGB ROI + EXPLICIT MASK DATASET GENERATION")
    print("=" * 72)
    print("Device:", device)
    print("Original images:", config.ORIGINAL_IMAGE_DIR)
    print("Checkpoint:", segmentation_config.DS2NET_MODEL_PATH)
    print("ROI padding:", config.ROI_PADDING)
    print("Output root:", config.MASK_AWARE_DATASET_ROOT)

    model = load_trained_ds2net(
        device=device,
        checkpoint_path=segmentation_config.DS2NET_MODEL_PATH,
    )

    process_split(
        split_name="train",
        split_file=config.TRAIN_CLASSIFICATION_LIST,
        rgb_dir=config.TRAIN_RGB_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        output_list=config.TRAIN_LIST,
        model=model,
        device=device,
    )

    process_split(
        split_name="validation",
        split_file=config.VAL_CLASSIFICATION_LIST,
        rgb_dir=config.VAL_RGB_DIR,
        mask_dir=config.VAL_MASK_DIR,
        output_list=config.VAL_LIST,
        model=model,
        device=device,
    )


if __name__ == "__main__":
    main()
