import cv2
import numpy as np


def keep_largest_component(
    binary_mask: np.ndarray,
    minimum_area: int = 50,
) -> np.ndarray:
    binary_mask = (binary_mask > 0).astype(np.uint8)

    number_of_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    if number_of_labels <= 1:
        return np.zeros_like(binary_mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    largest_area = stats[largest_label, cv2.CC_STAT_AREA]

    if largest_area < minimum_area:
        return np.zeros_like(binary_mask)

    return (labels == largest_label).astype(np.uint8)


def get_bounding_box(
    binary_mask: np.ndarray,
    padding: int,
) -> tuple[int, int, int, int] | None:
    y_coords, x_coords = np.where(binary_mask > 0)

    if len(x_coords) == 0:
        return None

    height, width = binary_mask.shape[:2]

    x_min = max(int(x_coords.min()) - padding, 0)
    x_max = min(int(x_coords.max()) + padding + 1, width)
    y_min = max(int(y_coords.min()) - padding, 0)
    y_max = min(int(y_coords.max()) + padding + 1, height)

    return x_min, y_min, x_max, y_max


def extract_rgb_roi_and_mask(
    image_bgr: np.ndarray,
    predicted_mask: np.ndarray,
    padding: int = 20,
    minimum_area: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    height, width = image_bgr.shape[:2]

    if predicted_mask.shape[:2] != (height, width):
        predicted_mask = cv2.resize(
            predicted_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    cleaned_mask = keep_largest_component(
        predicted_mask,
        minimum_area=minimum_area,
    )

    box = get_bounding_box(
        cleaned_mask,
        padding=padding,
    )

    if box is None:
        return (
            image_bgr.copy(),
            np.zeros((height, width), dtype=np.uint8),
            cleaned_mask,
            False,
        )

    x_min, y_min, x_max, y_max = box

    rgb_roi = image_bgr[y_min:y_max, x_min:x_max].copy()
    mask_roi = cleaned_mask[y_min:y_max, x_min:x_max].copy()

    return rgb_roi, mask_roi, cleaned_mask, True
