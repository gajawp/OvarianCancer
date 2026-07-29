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
    labels,
    predictions,
    probabilities,
    num_classes: int,
) -> ClassificationMetrics:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    roc_auc = None

    if probabilities is not None:
        try:
            roc_auc = roc_auc_score(
                labels,
                probabilities,
                labels=list(range(num_classes)),
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            roc_auc = None

    return ClassificationMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        balanced_accuracy=float(
            balanced_accuracy_score(labels, predictions)
        ),
        macro_precision=float(
            precision_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        macro_recall=float(
            recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        macro_f1=float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        weighted_f1=float(
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        roc_auc_ovr_macro=roc_auc,
    )


def build_confusion_matrix(
    labels,
    predictions,
    num_classes: int,
    normalize: bool,
):
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(num_classes)),
    )

    if not normalize:
        return matrix

    row_sums = matrix.sum(axis=1, keepdims=True)

    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums != 0,
    )
