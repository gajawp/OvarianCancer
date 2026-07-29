# integration_ds2net_MtaSwinn/metrics.py

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    roc_auc_ovr_macro: float | None


def calculate_classification_metrics(
    labels: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    probabilities: np.ndarray | None = None,
    num_classes: int = 8,
) -> ClassificationMetrics:
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            labels,
            predictions,
        )
    )

    macro_precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    roc_auc = None

    if probabilities is not None:
        try:
            roc_auc = roc_auc_score(
                labels,
                probabilities,
                labels=list(
                    range(num_classes)
                ),
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            # ROC-AUC may be unavailable if one or more
            # classes are absent from the evaluated split.
            roc_auc = None

    return ClassificationMetrics(
        accuracy=float(accuracy),
        balanced_accuracy=float(
            balanced_accuracy
        ),
        macro_precision=float(
            macro_precision
        ),
        macro_recall=float(
            macro_recall
        ),
        macro_f1=float(
            macro_f1
        ),
        weighted_f1=float(
            weighted_f1
        ),
        roc_auc_ovr_macro=(
            None
            if roc_auc is None
            else float(roc_auc)
        ),
    )


def build_confusion_matrix(
    labels,
    predictions,
    num_classes: int,
    normalize: bool = False,
):
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(
            range(num_classes)
        ),
    )

    if not normalize:
        return matrix

    row_sums = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(
            matrix,
            dtype=np.float64,
        ),
        where=row_sums != 0,
    )

    return normalized