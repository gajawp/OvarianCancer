# integrated_ds2net_mtaswin/roi_utils.py

import cv2
import numpy as np


def keep_largest_component(
    binary_mask: np.ndarray,
    minimum_area: int = 50,
) -> np.ndarray:
    """
    Keep the largest connected foreground region.
    """

    binary_mask = (
        binary_mask > 0
    ).astype(np.uint8)

    (
        number_of_labels,
        labels,
        statistics,
        _,
    ) = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    if number_of_labels <= 1:
        return np.zeros_like(
            binary_mask
        )

    foreground_areas = (
        statistics[
            1:,
            cv2.CC_STAT_AREA,
        ]
    )

    largest_label = (
        int(np.argmax(foreground_areas))
        + 1
    )

    largest_area = statistics[
        largest_label,
        cv2.CC_STAT_AREA,
    ]

    if largest_area < minimum_area:
        return np.zeros_like(
            binary_mask
        )

    return (
        labels == largest_label
    ).astype(np.uint8)


def get_bounding_box(
    binary_mask: np.ndarray,
    padding: int = 20,
):
    y_coordinates, x_coordinates = np.where(
        binary_mask > 0
    )

    if len(x_coordinates) == 0:
        return None

    height, width = binary_mask.shape[:2]

    x_min = max(
        int(x_coordinates.min()) - padding,
        0,
    )

    x_max = min(
        int(x_coordinates.max()) + padding + 1,
        width,
    )

    y_min = max(
        int(y_coordinates.min()) - padding,
        0,
    )

    y_max = min(
        int(y_coordinates.max()) + padding + 1,
        height,
    )

    return (
        x_min,
        y_min,
        x_max,
        y_max,
    )


def extract_segmentation_guided_roi(
    image: np.ndarray,
    predicted_mask: np.ndarray,
    mode: str = "bbox",
    padding: int = 20,
):
    """
    Return:
        ROI image,
        cleaned full-size mask,
        whether a valid mask was detected.
    """

    height, width = image.shape[:2]

    if predicted_mask.shape[:2] != (
        height,
        width,
    ):
        predicted_mask = cv2.resize(
            predicted_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    cleaned_mask = keep_largest_component(
        predicted_mask
    )

    bounding_box = get_bounding_box(
        cleaned_mask,
        padding=padding,
    )

    # Empty-mask fallback.
    if bounding_box is None:
        return (
            image.copy(),
            cleaned_mask,
            False,
        )

    (
        x_min,
        y_min,
        x_max,
        y_max,
    ) = bounding_box

    roi_image = image[
        y_min:y_max,
        x_min:x_max,
    ]

    roi_mask = cleaned_mask[
        y_min:y_max,
        x_min:x_max,
    ]

    if mode == "bbox":
        return (
            roi_image,
            cleaned_mask,
            True,
        )

    if mode == "masked":
        masked_roi = (
            roi_image
            * roi_mask[..., None]
        )

        return (
            masked_roi,
            cleaned_mask,
            True,
        )

    raise ValueError(
        f"Unsupported ROI mode: {mode}"
    )