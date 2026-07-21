from typing import Dict
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


def batch_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> Dict[str, float]:
    predictions = (torch.sigmoid(logits) >= threshold).float()
    targets = (targets >= 0.5).float()
    dims = (1, 2, 3)
    tp = (predictions * targets).sum(dims)
    fp = (predictions * (1 - targets)).sum(dims)
    fn = ((1 - predictions) * targets).sum(dims)
    tn = ((1 - predictions) * (1 - targets)).sum(dims)
    eps = 1e-7
    return {
        "dice": ((2 * tp + eps) / (2 * tp + fp + fn + eps)).mean().item(),
        "iou": ((tp + eps) / (tp + fp + fn + eps)).mean().item(),
        "precision": ((tp + eps) / (tp + fp + eps)).mean().item(),
        "recall": ((tp + eps) / (tp + fn + eps)).mean().item(),
        "specificity": ((tn + eps) / (tn + fp + eps)).mean().item(),
    }


def hausdorff_distance(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_points = np.argwhere(prediction > 0)
    target_points = np.argwhere(target > 0)
    if len(pred_points) == 0 or len(target_points) == 0:
        return float("nan")
    return float(max(
        directed_hausdorff(pred_points, target_points)[0],
        directed_hausdorff(target_points, pred_points)[0],
    ))
