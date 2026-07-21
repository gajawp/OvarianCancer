from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def calculate_specificity_per_class(
    confusion: np.ndarray,
) -> List[float]:
    specificities: List[float] = []
    total = confusion.sum()

    for class_index in range(confusion.shape[0]):
        true_positive = confusion[class_index, class_index]
        false_positive = confusion[:, class_index].sum() - true_positive
        false_negative = confusion[class_index, :].sum() - true_positive
        true_negative = (
            total - true_positive - false_positive - false_negative
        )

        denominator = true_negative + false_positive
        specificity = (
            float(true_negative / denominator)
            if denominator > 0
            else float("nan")
        )
        specificities.append(specificity)

    return specificities


def calculate_top_k_accuracy(
    probabilities: np.ndarray,
    targets: np.ndarray,
    k: int,
) -> float:
    k = min(k, probabilities.shape[1])
    top_k_indices = np.argsort(probabilities, axis=1)[:, -k:]
    correct = [
        target in predictions
        for target, predictions in zip(targets, top_k_indices)
    ]
    return float(np.mean(correct))


def calculate_multiclass_metrics(
    targets: Iterable[int],
    predictions: Iterable[int],
    probabilities: Optional[np.ndarray],
    class_names: Dict[int, str],
    top_k_values: Iterable[int] = (1, 3),
) -> Dict:
    targets_array = np.asarray(list(targets), dtype=np.int64)
    predictions_array = np.asarray(list(predictions), dtype=np.int64)

    labels = sorted(class_names.keys())
    target_names = [class_names[index] for index in labels]

    confusion = confusion_matrix(
        targets_array,
        predictions_array,
        labels=labels,
    )

    precision_per_class, recall_per_class, f1_per_class, support = (
        precision_recall_fscore_support(
            targets_array,
            predictions_array,
            labels=labels,
            zero_division=0,
        )
    )

    specificity_per_class = calculate_specificity_per_class(confusion)

    metrics = {
        "accuracy": float(
            accuracy_score(targets_array, predictions_array)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets_array, predictions_array)
        ),
        "macro_precision": float(
            precision_score(
                targets_array,
                predictions_array,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                targets_array,
                predictions_array,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                targets_array,
                predictions_array,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                targets_array,
                predictions_array,
                average="weighted",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion.tolist(),
        "per_class": {},
    }

    for position, class_index in enumerate(labels):
        metrics["per_class"][str(class_index)] = {
            "class_name": class_names[class_index],
            "precision": float(precision_per_class[position]),
            "recall": float(recall_per_class[position]),
            "specificity": float(specificity_per_class[position]),
            "f1": float(f1_per_class[position]),
            "support": int(support[position]),
        }

    report = classification_report(
        targets_array,
        predictions_array,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    metrics["classification_report"] = report

    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64)

        for k in top_k_values:
            metrics[f"top_{k}_accuracy"] = calculate_top_k_accuracy(
                probabilities=probabilities,
                targets=targets_array,
                k=k,
            )

        try:
            binarized_targets = label_binarize(
                targets_array,
                classes=labels,
            )
            metrics["macro_roc_auc_ovr"] = float(
                roc_auc_score(
                    binarized_targets,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                )
            )
            metrics["weighted_roc_auc_ovr"] = float(
                roc_auc_score(
                    binarized_targets,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                )
            )
        except ValueError:
            # ROC-AUC is undefined when one or more classes are absent.
            metrics["macro_roc_auc_ovr"] = None
            metrics["weighted_roc_auc_ovr"] = None

    return metrics


def save_metrics(
    metrics: Dict,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_confusion_matrix(
    confusion: np.ndarray,
    class_names: Dict[int, str],
    output_path: Path,
    normalize: bool = False,
) -> None:
    confusion = np.asarray(confusion, dtype=np.float64)

    if normalize:
        row_sums = confusion.sum(axis=1, keepdims=True)
        confusion = np.divide(
            confusion,
            row_sums,
            out=np.zeros_like(confusion),
            where=row_sums != 0,
        )

    labels = [
        class_names[index]
        for index in sorted(class_names.keys())
    ]

    figure_size = max(10, len(labels) * 1.4)
    plt.figure(figsize=(figure_size, figure_size))

    plt.imshow(confusion, interpolation="nearest")
    plt.title(
        "Normalized Confusion Matrix"
        if normalize
        else "Confusion Matrix"
    )
    plt.colorbar()

    tick_positions = np.arange(len(labels))
    plt.xticks(
        tick_positions,
        labels,
        rotation=45,
        ha="right",
    )
    plt.yticks(tick_positions, labels)

    display_format = ".2f" if normalize else ".0f"
    threshold = (
        confusion.max() / 2.0
        if confusion.size and confusion.max() > 0
        else 0
    )

    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            value = confusion[row, column]
            plt.text(
                column,
                row,
                format(value, display_format),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
