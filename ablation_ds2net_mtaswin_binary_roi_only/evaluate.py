import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ablation_ds2net_mtaswin_binary_roi_only import config
from ablation_ds2net_mtaswin_binary_roi_only.dataset import (
    ROIBinaryDataset,
)
from ablation_ds2net_mtaswin_binary_roi_only.model import (
    create_roi_model,
)


def create_loader():
    dataset = ROIBinaryDataset(
        roi_directory=config.VAL_RGB_DIR,
        mask_directory=config.VAL_MASK_DIR,
        split_file=config.VAL_LIST,
        image_size=config.IMAGE_SIZE,
        training=False,
    )

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )


def load_model(device):
    model = create_roi_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        dropout=config.DROPOUT,
        attention_hidden_dim=config.ATTENTION_HIDDEN_DIM,
    ).to(device)

    checkpoint = torch.load(
        config.BEST_MODEL_PATH,
        map_location=device,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    print(
        "Loaded checkpoint epoch:",
        checkpoint.get("epoch", "unknown"),
    )
    print(
        "Checkpoint malignant F1:",
        checkpoint.get(
            "validation_malignant_f1",
            "unknown",
        ),
    )

    return model


def save_confusion_matrix(
    matrix,
    output_path,
    normalized,
):
    figure = plt.figure(figsize=(8, 7))
    plt.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )
    plt.title(
        "Normalized Binary Confusion Matrix"
        if normalized
        else "Binary Confusion Matrix"
    )
    plt.colorbar()

    labels = [
        config.CLASS_NAMES[0],
        config.CLASS_NAMES[1],
    ]
    ticks = np.arange(2)
    plt.xticks(
        ticks,
        labels,
        rotation=30,
        ha="right",
    )
    plt.yticks(ticks, labels)

    threshold = (
        matrix.max() / 2.0
        if matrix.size
        else 0
    )

    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            plt.text(
                column,
                row,
                (
                    f"{value:.3f}"
                    if normalized
                    else str(int(value))
                ),
                horizontalalignment="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


@torch.inference_mode()
def main():
    device = config.DEVICE
    config.RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    loader = create_loader()
    model = load_model(device)

    labels_all = []
    predictions_all = []
    malignant_probabilities_all = []
    rows = []

    for batch in tqdm(
        loader,
        desc="Evaluating RGB ROI-only ablation",
    ):
        (
            roi_images,
            labels,
            image_names,
            mask_names,
            original_labels,
        ) = batch

        roi_images = roi_images.to(device)

        logits = model(
            roi_images,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )
        predictions = probabilities.argmax(dim=1)

        labels_np = labels.numpy()
        predictions_np = predictions.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        original_labels_np = original_labels.numpy()

        labels_all.extend(labels_np.tolist())
        predictions_all.extend(
            predictions_np.tolist()
        )
        malignant_probabilities_all.extend(
            probabilities_np[:, 1].tolist()
        )

        for index, image_name in enumerate(image_names):
            true_label = int(labels_np[index])
            predicted_label = int(
                predictions_np[index]
            )

            rows.append(
                {
                    "image_name": image_name,
                    "mask_name": mask_names[index],
                    "original_multiclass_label": int(
                        original_labels_np[index]
                    ),
                    "true_binary_label": true_label,
                    "predicted_binary_label": (
                        predicted_label
                    ),
                    "probability_non_malignant": float(
                        probabilities_np[index, 0]
                    ),
                    "probability_malignant": float(
                        probabilities_np[index, 1]
                    ),
                    "correct": int(
                        true_label == predicted_label
                    ),
                }
            )

    labels = np.asarray(labels_all)
    predictions = np.asarray(predictions_all)
    malignant_probabilities = np.asarray(
        malignant_probabilities_all
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )
    tn, fp, fn, tp = matrix.ravel()

    accuracy = accuracy_score(
        labels,
        predictions,
    )
    balanced_accuracy = balanced_accuracy_score(
        labels,
        predictions,
    )
    malignant_precision = precision_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )
    sensitivity = recall_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )
    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )
    malignant_f1 = f1_score(
        labels,
        predictions,
        pos_label=1,
        zero_division=0,
    )
    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    roc_auc = roc_auc_score(
        labels,
        malignant_probabilities,
    )
    pr_auc = average_precision_score(
        labels,
        malignant_probabilities,
    )
    negative_predictive_value = (
        tn / (tn + fn)
        if tn + fn > 0
        else 0.0
    )

    print("Accuracy                :", f"{accuracy:.4f}")
    print(
        "Balanced Accuracy       :",
        f"{balanced_accuracy:.4f}",
    )
    print(
        "Malignant Precision     :",
        f"{malignant_precision:.4f}",
    )
    print(
        "Sensitivity / Recall    :",
        f"{sensitivity:.4f}",
    )
    print(
        "Specificity             :",
        f"{specificity:.4f}",
    )
    print(
        "Malignant F1            :",
        f"{malignant_f1:.4f}",
    )
    print("Macro F1                :", f"{macro_f1:.4f}")
    print(
        "Negative Predictive Val.:",
        f"{negative_predictive_value:.4f}",
    )
    print("ROC-AUC                 :", f"{roc_auc:.4f}")
    print("PR-AUC                  :", f"{pr_auc:.4f}")
    print("True Negatives          :", int(tn))
    print("False Positives         :", int(fp))
    print("False Negatives         :", int(fn))
    print("True Positives          :", int(tp))

    with config.EVALUATION_RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": (
            "DS2Net + ROI Swin-T + Full-image "
            "ResNet18 + Mask Encoder"
        ),
        "input_size": config.IMAGE_SIZE,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "malignant_precision": malignant_precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "malignant_f1": malignant_f1,
        "macro_f1": macro_f1,
        "negative_predictive_value": (
            negative_predictive_value
        ),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    with config.METRIC_SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary.keys()),
        )
        writer.writeheader()
        writer.writerow(summary)

    normalized = matrix.astype(np.float64)
    row_sums = normalized.sum(
        axis=1,
        keepdims=True,
    )
    normalized = np.divide(
        normalized,
        row_sums,
        out=np.zeros_like(normalized),
        where=row_sums != 0,
    )

    save_confusion_matrix(
        matrix,
        config.CONFUSION_MATRIX_PATH,
        False,
    )
    save_confusion_matrix(
        normalized,
        config.NORMALIZED_CONFUSION_MATRIX_PATH,
        True,
    )

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            labels,
            malignant_probabilities,
        )
    )
    figure = plt.figure(figsize=(8, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {roc_auc:.4f}",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Chance",
    )
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Dual-Context ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    figure.savefig(
        config.ROC_CURVE_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    precision, recall, _ = precision_recall_curve(
        labels,
        malignant_probabilities,
    )
    figure = plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.4f}",
    )
    plt.xlabel("Recall / sensitivity")
    plt.ylabel("Precision")
    plt.title("Dual-Context Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    figure.savefig(
        config.PR_CURVE_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
