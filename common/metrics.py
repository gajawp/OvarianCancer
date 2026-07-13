
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


# ==========================================================
# PyTorch Metrics
# Used during training and validation
# ==========================================================

def get_binary_prediction(logits, threshold=0.5):
    """
    Converts raw model logits into a binary segmentation mask.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model output with shape:
        [batch_size, 1, height, width]

    threshold : float
        Probability threshold used to convert the sigmoid output
        into a binary mask.

    Returns
    -------
    torch.Tensor
        Binary prediction containing only 0 and 1.
    """

    probabilities = torch.sigmoid(logits)

    prediction = (
        probabilities > threshold
    ).float()

    return prediction


def dice_score(logits, target, threshold=0.5):
    """
    Calculates the mean Dice score for a batch.

    Dice measures the overlap between the predicted mask
    and the ground-truth mask.

    Dice = 2TP / (2TP + FP + FN)

    Parameters
    ----------
    logits : torch.Tensor
        Raw segmentation logits.

    target : torch.Tensor
        Ground-truth binary masks.

    threshold : float
        Prediction threshold.

    Returns
    -------
    torch.Tensor
        Mean Dice score for the batch.
    """

    prediction = get_binary_prediction(
        logits,
        threshold
    )

    intersection = (
        prediction * target
    ).sum(dim=(1, 2, 3))

    prediction_sum = prediction.sum(
        dim=(1, 2, 3)
    )

    target_sum = target.sum(
        dim=(1, 2, 3)
    )

    smooth = 1e-6

    dice = (
        2.0 * intersection + smooth
    ) / (
        prediction_sum
        + target_sum
        + smooth
    )

    return dice.mean()


def iou_score(logits, target, threshold=0.5):
    """
    Calculates the mean Intersection over Union score.

    IoU = TP / (TP + FP + FN)

    Parameters
    ----------
    logits : torch.Tensor
        Raw segmentation logits.

    target : torch.Tensor
        Ground-truth binary masks.

    threshold : float
        Prediction threshold.

    Returns
    -------
    torch.Tensor
        Mean IoU score for the batch.
    """

    prediction = get_binary_prediction(
        logits,
        threshold
    )

    intersection = (
        prediction * target
    ).sum(dim=(1, 2, 3))

    prediction_sum = prediction.sum(
        dim=(1, 2, 3)
    )

    target_sum = target.sum(
        dim=(1, 2, 3)
    )

    union = (
        prediction_sum
        + target_sum
        - intersection
    )

    smooth = 1e-6

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean()


# ==========================================================
# NumPy Metrics
# Used during final evaluation
# ==========================================================

def calculate_numpy_metrics(prediction, target):
    """
    Calculates segmentation metrics for one predicted mask.

    This function is normally used inside evaluate.py after
    converting PyTorch tensors into NumPy arrays.

    Parameters
    ----------
    prediction : numpy.ndarray
        Binary predicted mask containing 0 and 1.

    target : numpy.ndarray
        Binary ground-truth mask containing 0 and 1.

    Returns
    -------
    dict
        Dictionary containing:
        Dice, IoU, precision, recall and specificity.
    """

    prediction = prediction.astype(
        np.uint8
    )

    target = target.astype(
        np.uint8
    )

    true_positive = np.logical_and(
        prediction == 1,
        target == 1
    ).sum()

    true_negative = np.logical_and(
        prediction == 0,
        target == 0
    ).sum()

    false_positive = np.logical_and(
        prediction == 1,
        target == 0
    ).sum()

    false_negative = np.logical_and(
        prediction == 0,
        target == 1
    ).sum()

    smooth = 1e-6

    dice = (
        2 * true_positive + smooth
    ) / (
        2 * true_positive
        + false_positive
        + false_negative
        + smooth
    )

    iou = (
        true_positive + smooth
    ) / (
        true_positive
        + false_positive
        + false_negative
        + smooth
    )

    precision = (
        true_positive + smooth
    ) / (
        true_positive
        + false_positive
        + smooth
    )

    recall = (
        true_positive + smooth
    ) / (
        true_positive
        + false_negative
        + smooth
    )

    specificity = (
        true_negative + smooth
    ) / (
        true_negative
        + false_positive
        + smooth
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity)
    }


def hausdorff_distance(prediction, target):
    """
    Calculates the bidirectional Hausdorff distance.

    Hausdorff distance measures the largest boundary distance
    between the predicted mask and the ground-truth mask.

    Lower values indicate better boundary agreement.

    Parameters
    ----------
    prediction : numpy.ndarray
        Binary predicted mask.

    target : numpy.ndarray
        Binary ground-truth mask.

    Returns
    -------
    float
        Hausdorff distance.

        Returns NaN when either mask contains no foreground
        pixels because the distance cannot be calculated.
    """

    prediction_points = np.argwhere(
        prediction > 0
    )

    target_points = np.argwhere(
        target > 0
    )

    if (
        len(prediction_points) == 0
        or len(target_points) == 0
    ):
        return np.nan

    forward_distance = directed_hausdorff(
        prediction_points,
        target_points
    )[0]

    backward_distance = directed_hausdorff(
        target_points,
        prediction_points
    )[0]

    return float(
        max(
            forward_distance,
            backward_distance
        )
    )

