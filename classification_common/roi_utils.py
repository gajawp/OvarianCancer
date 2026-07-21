from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".JPG",
    ".JPEG",
    ".PNG",
)


def find_matching_file(
    directory: Path,
    image_name: str,
    extra_stems: Optional[Iterable[str]] = None,
) -> Path:
    """
    Find an image or mask even when the file extension differs.

    For image '658.JPG', this function can locate files such as:
      - 658.JPG
      - 658.png
      - 658_mask.png
      - 658_label.png
    """
    directory = Path(directory)
    input_path = Path(image_name)
    stem = input_path.stem

    direct_path = directory / image_name
    if direct_path.exists():
        return direct_path

    candidate_stems = [
        stem,
        f"{stem}_mask",
        f"{stem}_label",
        f"{stem}_annotation",
        f"{stem}_segmentation",
    ]

    if extra_stems:
        candidate_stems.extend(extra_stems)

    for candidate_stem in candidate_stems:
        for extension in SUPPORTED_EXTENSIONS:
            candidate = directory / f"{candidate_stem}{extension}"
            if candidate.exists():
                return candidate

    # Case-insensitive fallback.
    stem_lower = stem.lower()
    matches = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in {ext.lower() for ext in SUPPORTED_EXTENSIONS}
        and (
            path.stem.lower() == stem_lower
            or path.stem.lower().startswith(f"{stem_lower}_")
        )
    ]

    if matches:
        return sorted(matches)[0]

    raise FileNotFoundError(
        f"No matching file found for {image_name!r} in {directory}"
    )


def read_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_binary_mask(
    path: Path,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Read a segmentation mask and return a uint8 binary mask.

    Nearest-neighbor interpolation must be used whenever masks are resized.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")

    if threshold <= 1.0:
        threshold_value = int(round(threshold * 255))
    else:
        threshold_value = int(round(threshold))

    return (mask > threshold_value).astype(np.uint8)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest non-background connected component.
    """
    binary = (mask > 0).astype(np.uint8)

    number_of_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if number_of_labels <= 1:
        return binary

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_component_label = int(np.argmax(component_areas)) + 1
    return (labels == largest_component_label).astype(np.uint8)


def bounding_box_from_mask(
    mask: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Return bounding box as (x_min, y_min, x_max, y_max), where x_max and
    y_max are exclusive.
    """
    coordinates = cv2.findNonZero((mask > 0).astype(np.uint8))

    if coordinates is None:
        return None

    x, y, width, height = cv2.boundingRect(coordinates)
    return x, y, x + width, y + height


def expand_bounding_box(
    box: Tuple[int, int, int, int],
    image_shape: Tuple[int, ...],
    padding: float = 0.10,
) -> Tuple[int, int, int, int]:
    """
    Expand a bounding box by a fraction of its width and height.
    """
    x_min, y_min, x_max, y_max = box
    image_height, image_width = image_shape[:2]

    box_width = x_max - x_min
    box_height = y_max - y_min

    pad_x = int(round(box_width * padding))
    pad_y = int(round(box_height * padding))

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(image_width, x_max + pad_x)
    y_max = min(image_height, y_max + pad_y)

    return x_min, y_min, x_max, y_max


def crop_roi(
    image: np.ndarray,
    mask: np.ndarray,
    padding: float = 0.10,
    use_masked_roi: bool = False,
    keep_largest_component: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop the tumor/ovary region from an RGB image using a binary mask.

    If the mask is empty, the complete image is returned.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with shape HxWx3, got {image.shape}"
        )

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    binary_mask = (mask > 0).astype(np.uint8)

    if keep_largest_component:
        binary_mask = largest_connected_component(binary_mask)

    box = bounding_box_from_mask(binary_mask)

    if box is None:
        height, width = image.shape[:2]
        return image.copy(), (0, 0, width, height)

    expanded_box = expand_bounding_box(
        box=box,
        image_shape=image.shape,
        padding=padding,
    )

    x_min, y_min, x_max, y_max = expanded_box
    roi = image[y_min:y_max, x_min:x_max].copy()

    if use_masked_roi:
        roi_mask = binary_mask[y_min:y_max, x_min:x_max]
        roi = roi * roi_mask[..., None]

    return roi, expanded_box
