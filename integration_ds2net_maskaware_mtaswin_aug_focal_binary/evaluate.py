import csv
import shutil

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

from integration_ds2net_maskaware_mtaswin_aug_focal_binary import config
from integration_ds2net_maskaware_mtaswin_aug_focal_binary.dataset import (
    MaskAwareBinaryClassificationDataset,
)
from integration_ds2net_maskaware_mtaswin_aug_focal_binary.model import (
    create_maskaware_mta_swin,
)


def create_loader():
    dataset = MaskAwareBinaryClassificationDataset(
        image_directory=config.VAL_RGB_DIR,
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
    model = create_maskaware_mta_swin(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        dropout=config.DROPOUT,
        attention_hidden_dim=config.ATTENTION_HIDDEN_DIM,
        mask_feature_dim=config.MASK_FEATURE_DIM,
    ).to(device)

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print(
        "Checkpoint malignant F1:",
        checkpoint.get("validation_malignant_f1", "unknown"),
    )
    return model


def save_confusion_matrix(matrix, output_path, normalized):
    figure = plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(
        "Normalized Binary Confusion Matrix"
        if normalized
        else "Binary Confusion Matrix"
    )
    plt.colorbar()

    labels = [config.CLASS_NAMES[0], config.CLASS_NAMES[1]]
    ticks = np.arange(2)
    plt.xticks(ticks, labels, rotation=30, ha="right")
    plt.yticks(ticks, labels)

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            plt.text(
                column,
                row,
                f"{value:.3f}" if normalized else str(int(value)),
                horizontalalignment="center",
                color="white" if value > threshold else "black",
            )

    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_roc_curve(labels, malignant_probabilities):
    false_positive_rate, true_positive_rate, _ = roc_curve(
        labels, malignant_probabilities
    )
    auc_value = roc_auc_score(labels, malignant_probabilities)

    figure = plt.figure(figsize=(8, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {auc_value:.4f}",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("High-Grade Serous Carcinoma ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    figure.savefig(config.ROC_CURVE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_pr_curve(labels, malignant_probabilities):
    precision, recall, _ = precision_recall_curve(
        labels, malignant_probabilities
    )
    average_precision = average_precision_score(
        labels, malignant_probabilities
    )

    figure = plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {average_precision:.4f}")
    plt.xlabel("Recall / sensitivity")
    plt.ylabel("Precision")
    plt.title("High-Grade Serous Carcinoma Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    figure.savefig(config.PR_CURVE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)



def evaluate_threshold(labels, malignant_probabilities, threshold):
    predictions = (
        malignant_probabilities >= threshold
    ).astype(np.int64)

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )
    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )

    sensitivity = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            labels,
            predictions,
        ),
        "malignant_precision": precision_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "malignant_f1": f1_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "true_negatives": int(true_negative),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
    }


@torch.inference_mode()
def main():
    device = config.DEVICE
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    false_negative_dir = config.RESULTS_DIR / "false_negative_cases"
    false_positive_dir = config.RESULTS_DIR / "false_positive_cases"

    if false_negative_dir.exists():
        shutil.rmtree(false_negative_dir)
    if false_positive_dir.exists():
        shutil.rmtree(false_positive_dir)

    false_negative_dir.mkdir(parents=True, exist_ok=True)
    false_positive_dir.mkdir(parents=True, exist_ok=True)

    loader = create_loader()
    model = load_model(device)

    labels_all = []
    malignant_probabilities_all = []
    rows = []

    for (
        rgb_images,
        masks,
        labels,
        image_names,
        mask_names,
        original_labels,
    ) in tqdm(
        loader,
        desc="Evaluating binary augmented focal-loss mask-aware MTA-Swin",
    ):
        rgb_images = rgb_images.to(device)
        masks = masks.to(device)

        logits = model(rgb_images, masks)
        probabilities = torch.softmax(logits, dim=1)
        labels_np = labels.numpy()
        original_labels_np = original_labels.numpy()
        probabilities_np = probabilities.cpu().numpy()

        labels_all.extend(labels_np.tolist())
        malignant_probabilities_all.extend(probabilities_np[:, 1].tolist())

        for index, image_name in enumerate(image_names):
            rows.append(
                {
                    "image_name": image_name,
                    "mask_name": mask_names[index],
                    "original_multiclass_label": int(original_labels_np[index]),
                    "true_binary_label": int(labels_np[index]),
                    "true_class_name": config.CLASS_NAMES[int(labels_np[index])],
                    "probability_non_malignant": float(probabilities_np[index, 0]),
                    "probability_malignant": float(probabilities_np[index, 1]),
                }
            )

    labels = np.asarray(labels_all)
    malignant_probabilities = np.asarray(malignant_probabilities_all)

    thresholds = np.arange(0.05, 0.951, 0.05)
    threshold_results = [
        evaluate_threshold(
            labels,
            malignant_probabilities,
            threshold,
        )
        for threshold in thresholds
    ]

    print("\nThreshold comparison")
    print("-" * 120)
    for result in threshold_results:
        print(
            f"Threshold={result['threshold']:.2f} | "
            f"Balanced Acc={result['balanced_accuracy']:.4f} | "
            f"Precision={result['malignant_precision']:.4f} | "
            f"Sensitivity={result['sensitivity']:.4f} | "
            f"Specificity={result['specificity']:.4f} | "
            f"Malignant F1={result['malignant_f1']:.4f} | "
            f"FP={result['false_positives']} | "
            f"FN={result['false_negatives']}"
        )

    best_result = max(
        threshold_results,
        key=lambda result: (
            result["malignant_f1"],
            result["sensitivity"],
            result["balanced_accuracy"],
        ),
    )
    automatically_selected_threshold = best_result["threshold"]
    best_threshold = 0.50

    print(
        "\nAutomatically selected threshold:",
        f"{automatically_selected_threshold:.2f}",
    )
    print("Final inspection threshold:", f"{best_threshold:.2f}")
    print(
        "Selection criterion: malignant F1, then sensitivity, "
        "then balanced accuracy"
    )

    threshold_results_path = (
        config.RESULTS_DIR / "threshold_comparison.csv"
    )
    with threshold_results_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(threshold_results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(threshold_results)

    predictions = (
        malignant_probabilities >= best_threshold
    ).astype(np.int64)

    for index, row in enumerate(rows):
        predicted_label = int(predictions[index])
        true_label = int(row["true_binary_label"])
        probability_malignant = float(
            row["probability_malignant"]
        )

        row["selected_threshold"] = float(best_threshold)
        row["predicted_binary_label"] = predicted_label
        row["predicted_class_name"] = (
            config.CLASS_NAMES[predicted_label]
        )
        row["confidence"] = (
            probability_malignant
            if predicted_label == 1
            else 1.0 - probability_malignant
        )
        row["correct"] = int(true_label == predicted_label)

    false_negative_rows = []
    false_positive_rows = []

    for row in rows:
        true_label = int(row["true_binary_label"])
        predicted_label = int(row["predicted_binary_label"])

        image_name = row["image_name"]
        mask_name = row["mask_name"]

        source_image = config.VAL_RGB_DIR / image_name
        source_mask = config.VAL_MASK_DIR / mask_name

        if true_label == 1 and predicted_label == 0:
            false_negative_rows.append(row)
            shutil.copy2(
                source_image,
                false_negative_dir / image_name,
            )
            shutil.copy2(
                source_mask,
                false_negative_dir / f"mask_{mask_name}",
            )

        elif true_label == 0 and predicted_label == 1:
            false_positive_rows.append(row)
            shutil.copy2(
                source_image,
                false_positive_dir / image_name,
            )
            shutil.copy2(
                source_mask,
                false_positive_dir / f"mask_{mask_name}",
            )

    false_negative_csv = (
        config.RESULTS_DIR / "false_negative_cases.csv"
    )
    false_positive_csv = (
        config.RESULTS_DIR / "false_positive_cases.csv"
    )

    if false_negative_rows:
        with false_negative_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(false_negative_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(false_negative_rows)

    if false_positive_rows:
        with false_positive_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(false_positive_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(false_positive_rows)

    print(
        "False-negative cases saved to:",
        false_negative_dir,
    )
    print(
        "False-positive cases saved to:",
        false_positive_dir,
    )

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    accuracy = accuracy_score(labels, predictions)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    malignant_precision = precision_score(
        labels, predictions, pos_label=1, zero_division=0
    )
    sensitivity = recall_score(
        labels, predictions, pos_label=1, zero_division=0
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive > 0
        else 0.0
    )
    malignant_f1 = f1_score(
        labels, predictions, pos_label=1, zero_division=0
    )
    macro_f1 = f1_score(
        labels, predictions, average="macro", zero_division=0
    )
    roc_auc = roc_auc_score(labels, malignant_probabilities)
    pr_auc = average_precision_score(labels, malignant_probabilities)
    negative_predictive_value = (
        true_negative / (true_negative + false_negative)
        if true_negative + false_negative > 0
        else 0.0
    )

    print("Selected Threshold      :", f"{best_threshold:.2f}")
    print("Accuracy                :", f"{accuracy:.4f}")
    print("Balanced Accuracy       :", f"{balanced_accuracy:.4f}")
    print("Malignant Precision     :", f"{malignant_precision:.4f}")
    print("Sensitivity / Recall    :", f"{sensitivity:.4f}")
    print("Specificity             :", f"{specificity:.4f}")
    print("Malignant F1            :", f"{malignant_f1:.4f}")
    print("Macro F1                :", f"{macro_f1:.4f}")
    print("Negative Predictive Val.:", f"{negative_predictive_value:.4f}")
    print("ROC-AUC                 :", f"{roc_auc:.4f}")
    print("PR-AUC                  :", f"{pr_auc:.4f}")
    print("True Negatives          :", int(true_negative))
    print("False Positives         :", int(false_positive))
    print("False Negatives         :", int(false_negative))
    print("True Positives          :", int(true_positive))

    with config.EVALUATION_RESULTS_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": "DS2Net + Mask-Aware MTA-Swin + Augmentation + Focal Loss",
        "task": "High-grade serous carcinoma vs non-malignant/other",
        "selected_threshold": best_threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "malignant_precision": malignant_precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "malignant_f1": malignant_f1,
        "macro_f1": macro_f1,
        "negative_predictive_value": negative_predictive_value,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negatives": int(true_negative),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
    }

    with config.METRIC_SUMMARY_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    normalized = matrix.astype(np.float64)
    row_sums = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(
        normalized,
        row_sums,
        out=np.zeros_like(normalized),
        where=row_sums != 0,
    )

    save_confusion_matrix(matrix, config.CONFUSION_MATRIX_PATH, False)
    save_confusion_matrix(
        normalized,
        config.NORMALIZED_CONFUSION_MATRIX_PATH,
        True,
    )
    save_roc_curve(labels, malignant_probabilities)
    save_pr_curve(labels, malignant_probabilities)


if __name__ == "__main__":
    main()