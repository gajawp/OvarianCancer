# integration_ds2net_MtaSwinn/generate_roi_dataset.py

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common import config as segmentation_config
from integration_ds2net_MtaSwinn import (
    config as integration_config,
)
from integration_ds2net_MtaSwinn.load_ds2net import (
    load_trained_ds2net,
)
from integration_ds2net_MtaSwinn.roi_utils import (
    extract_segmentation_guided_roi,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "OTU_2d"
)

IMAGE_DIRECTORY = (
    DATASET_ROOT
    / "images"
)

TRAIN_SPLIT_FILE = (
    DATASET_ROOT
    / "train_cls.txt"
)

VAL_SPLIT_FILE = (
    DATASET_ROOT
    / "val_cls.txt"
)

ROI_DATASET_ROOT = (
    integration_config.ROI_DATASET_ROOT
)

TRAIN_ROI_DIRECTORY = (
    integration_config.TRAIN_ROI_DIR
)

VAL_ROI_DIRECTORY = (
    integration_config.VAL_ROI_DIR
)

TRAIN_MASK_DIRECTORY = (
    integration_config.TRAIN_MASK_DIR
)

VAL_MASK_DIRECTORY = (
    integration_config.VAL_MASK_DIR
)

TRAIN_OUTPUT_LIST = (
    integration_config.TRAIN_ROI_LIST
)

VAL_OUTPUT_LIST = (
    integration_config.VAL_ROI_LIST
)

ROI_MODE = integration_config.ROI_MODE
ROI_PADDING = integration_config.ROI_PADDING


def read_classification_split(
    split_file: Path,
) -> list[tuple[str, int]]:
    """
    Read lines such as:

        658.JPG 5
        384.JPG 3
    """

    samples = []

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}"
        )

    with split_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid line {line_number} "
                    f"in {split_file}: {line!r}"
                )

            image_name = parts[0]

            try:
                class_label = int(parts[1])
            except ValueError as error:
                raise ValueError(
                    f"Invalid class label on line "
                    f"{line_number}: {line!r}"
                ) from error

            samples.append(
                (
                    image_name,
                    class_label,
                )
            )

    return samples


def preprocess_for_ds2net(
    image_bgr: np.ndarray,
) -> torch.Tensor:
    """
    Match preprocessing in the existing ds2net/predict.py.
    """

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

    normalized_rgb = (
        resized_rgb.astype(np.float32)
        / 255.0
    )

    chw_image = normalized_rgb.transpose(
        2,
        0,
        1,
    )

    image_tensor = (
        torch.from_numpy(chw_image)
        .float()
        .unsqueeze(0)
    )

    return image_tensor


def extract_final_logits(
    model_output,
) -> torch.Tensor:
    """
    Extract the final DS2Net output.

    DS2NetSingleDomain returns:
        final, aux2, aux3, aux4
    """

    if isinstance(
        model_output,
        torch.Tensor,
    ):
        return model_output

    if isinstance(
        model_output,
        (tuple, list),
    ):
        if len(model_output) == 0:
            raise ValueError(
                "DS2Net returned an empty output."
            )

        return model_output[0]

    if isinstance(
        model_output,
        dict,
    ):
        for key in (
            "out",
            "logits",
            "prediction",
            "pred",
        ):
            if key in model_output:
                return model_output[key]

    raise TypeError(
        "Unsupported DS2Net output type: "
        f"{type(model_output).__name__}"
    )


@torch.inference_mode()
def predict_mask(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Generate a binary DS2Net mask and resize it back
    to the original ultrasound-image dimensions.
    """

    original_height, original_width = (
        image_bgr.shape[:2]
    )

    image_tensor = preprocess_for_ds2net(
        image_bgr
    ).to(device)

    model_output = model(
        image_tensor
    )

    final_logits = extract_final_logits(
        model_output
    )

    probabilities = torch.sigmoid(
        final_logits
    )

    binary_mask = (
        probabilities
        >= segmentation_config.PREDICTION_THRESHOLD
    ).float()

    binary_mask = (
        binary_mask
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    binary_mask = cv2.resize(
        binary_mask,
        (
            original_width,
            original_height,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    return binary_mask


def process_split(
    split_name: str,
    split_file: Path,
    roi_directory: Path,
    mask_directory: Path,
    output_split_file: Path,
    model: torch.nn.Module,
    device: torch.device,
):
    samples = read_classification_split(
        split_file
    )

    roi_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_split_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_lines = []

    valid_masks = 0
    fallback_images = 0
    missing_images = 0
    failed_saves = 0

    progress_bar = tqdm(
        samples,
        desc=f"Generating {split_name} ROIs",
    )

    for image_name, class_label in progress_bar:
        image_path = (
            IMAGE_DIRECTORY
            / image_name
        )

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            print(
                f"\nCould not read image: "
                f"{image_path}"
            )

            missing_images += 1
            continue

        predicted_mask = predict_mask(
            model=model,
            image_bgr=image_bgr,
            device=device,
        )

        (
            roi_image,
            cleaned_mask,
            mask_found,
        ) = extract_segmentation_guided_roi(
            image=image_bgr,
            predicted_mask=predicted_mask,
            mode=ROI_MODE,
            padding=ROI_PADDING,
        )

        if mask_found:
            valid_masks += 1
        else:
            fallback_images += 1

        roi_path = (
            roi_directory
            / image_name
        )

        mask_filename = (
            f"{Path(image_name).stem}_mask.png"
        )

        mask_path = (
            mask_directory
            / mask_filename
        )

        mask_to_save = (
            cleaned_mask.astype(np.uint8)
            * 255
        )

        roi_saved = cv2.imwrite(
            str(roi_path),
            roi_image,
        )

        mask_saved = cv2.imwrite(
            str(mask_path),
            mask_to_save,
        )

        if not roi_saved or not mask_saved:
            failed_saves += 1

            print(
                f"\nCould not save outputs for "
                f"{image_name}"
            )

            continue

        output_lines.append(
            f"{image_name} {class_label}\n"
        )

        progress_bar.set_postfix(
            valid=valid_masks,
            fallback=fallback_images,
        )

    with output_split_file.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.writelines(
            output_lines
        )

    print()
    print("=" * 70)
    print(f"{split_name.upper()} SUMMARY")
    print("=" * 70)
    print("Input samples:", len(samples))
    print("Saved samples:", len(output_lines))
    print("Valid masks:", valid_masks)
    print(
        "Whole-image fallbacks:",
        fallback_images,
    )
    print("Missing images:", missing_images)
    print("Failed saves:", failed_saves)
    print("ROI directory:", roi_directory)
    print("Mask directory:", mask_directory)
    print("Output list:", output_split_file)


def main():
    device = torch.device(
        segmentation_config.DEVICE
    )

    print("=" * 70)
    print("DS2NET ROI DATASET GENERATION")
    print("=" * 70)

    print("Device:", device)
    print(
        "Image directory:",
        IMAGE_DIRECTORY,
    )
    print(
        "DS2Net checkpoint:",
        segmentation_config.DS2NET_MODEL_PATH,
    )
    print(
        "Segmentation size:",
        segmentation_config.IMAGE_SIZE,
    )
    print(
        "Prediction threshold:",
        segmentation_config.PREDICTION_THRESHOLD,
    )
    print(
        "ROI dataset root:",
        ROI_DATASET_ROOT,
    )
    print("ROI mode:", ROI_MODE)
    print("ROI padding:", ROI_PADDING)

    model = load_trained_ds2net(
        device=device,
        checkpoint_path=(
            segmentation_config.DS2NET_MODEL_PATH
        ),
    )

    process_split(
        split_name="train",
        split_file=TRAIN_SPLIT_FILE,
        roi_directory=TRAIN_ROI_DIRECTORY,
        mask_directory=TRAIN_MASK_DIRECTORY,
        output_split_file=TRAIN_OUTPUT_LIST,
        model=model,
        device=device,
    )

    process_split(
        split_name="validation",
        split_file=VAL_SPLIT_FILE,
        roi_directory=VAL_ROI_DIRECTORY,
        mask_directory=VAL_MASK_DIRECTORY,
        output_split_file=VAL_OUTPUT_LIST,
        model=model,
        device=device,
    )

    print()
    print(
        "DS2Net ROI generation completed."
    )


if __name__ == "__main__":
    main()