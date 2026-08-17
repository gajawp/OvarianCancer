import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from integration_ds2net_maskaware_mtaswin_aug_balanced import config
from integration_ds2net_maskaware_mtaswin_aug_balanced.dataset import (
    MaskAwareClassificationDataset,
)
from integration_ds2net_maskaware_mtaswin.metrics import (
    build_confusion_matrix,
    calculate_classification_metrics,
)
from integration_ds2net_maskaware_mtaswin_aug_balanced.model import (
    create_maskaware_mta_swin,
)


def create_loader():
    dataset = MaskAwareClassificationDataset(
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

    checkpoint = torch.load(
        config.BEST_MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return model


def save_confusion_matrix(
    matrix,
    output_path,
    normalized: bool,
):
    figure = plt.figure(figsize=(10, 8))

    plt.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    plt.title(
        "Normalized Confusion Matrix"
        if normalized
        else "Confusion Matrix"
    )

    plt.colorbar()

    labels = [
        config.CLASS_NAMES[index]
        for index in range(config.NUM_CLASSES)
    ]

    ticks = np.arange(config.NUM_CLASSES)

    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    threshold = matrix.max() / 2.0 if matrix.size else 0

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]

            plt.text(
                column,
                row,
                f"{value:.2f}" if normalized else str(int(value)),
                horizontalalignment="center",
                color="white" if value > threshold else "black",
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
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    loader = create_loader()
    model = load_model(device)

    labels_all = []
    predictions_all = []
    probabilities_all = []
    rows = []

    for rgb_images, masks, labels, image_names in tqdm(
        loader,
        desc="Evaluating augmented balanced mask-aware MTA-Swin",
    ):
        rgb_images = rgb_images.to(device)
        masks = masks.to(device)

        logits = model(rgb_images, masks)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)

        labels_np = labels.numpy()
        predictions_np = predictions.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        confidence_np = confidence.cpu().numpy()

        labels_all.extend(labels_np.tolist())
        predictions_all.extend(predictions_np.tolist())
        probabilities_all.extend(probabilities_np.tolist())

        for index, image_name in enumerate(image_names):
            true_label = int(labels_np[index])
            predicted_label = int(predictions_np[index])

            row = {
                "image_name": image_name,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": float(confidence_np[index]),
                "correct": int(true_label == predicted_label),
            }

            for class_index in range(config.NUM_CLASSES):
                row[f"probability_class_{class_index}"] = float(
                    probabilities_np[index, class_index]
                )

            rows.append(row)

    metrics = calculate_classification_metrics(
        labels=labels_all,
        predictions=predictions_all,
        probabilities=np.asarray(probabilities_all),
        num_classes=config.NUM_CLASSES,
    )

    print("Accuracy          :", f"{metrics.accuracy:.4f}")
    print(
        "Balanced Accuracy :",
        f"{metrics.balanced_accuracy:.4f}",
    )
    print(
        "Macro Precision   :",
        f"{metrics.macro_precision:.4f}",
    )
    print(
        "Macro Recall      :",
        f"{metrics.macro_recall:.4f}",
    )
    print("Macro F1          :", f"{metrics.macro_f1:.4f}")
    print("Weighted F1       :", f"{metrics.weighted_f1:.4f}")
    print(
        "ROC-AUC OVR Macro :",
        "N/A"
        if metrics.roc_auc_ovr_macro is None
        else f"{metrics.roc_auc_ovr_macro:.4f}",
    )

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
        "model": "DS2Net + Mask-Aware MTA-Swin + Augmentation",
        "input": "RGB ROI plus explicit mask",
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "macro_precision": metrics.macro_precision,
        "macro_recall": metrics.macro_recall,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "roc_auc_ovr_macro": metrics.roc_auc_ovr_macro,
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

    confusion = build_confusion_matrix(
        labels_all,
        predictions_all,
        config.NUM_CLASSES,
        normalize=False,
    )

    normalized = build_confusion_matrix(
        labels_all,
        predictions_all,
        config.NUM_CLASSES,
        normalize=True,
    )

    save_confusion_matrix(
        confusion,
        config.CONFUSION_MATRIX_PATH,
        normalized=False,
    )

    save_confusion_matrix(
        normalized,
        config.NORMALIZED_CONFUSION_MATRIX_PATH,
        normalized=True,
    )


if __name__ == "__main__":
    main()
