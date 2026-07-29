# integration_ds2net_MtaSwinn/evaluate.py

import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from integration_ds2net_MtaSwinn import config
from integration_ds2net_MtaSwinn.dataset import (
    DS2NetROIClassificationDataset,
)
from integration_ds2net_MtaSwinn.metrics import (
    build_confusion_matrix,
    calculate_classification_metrics,
)
from integration_ds2net_MtaSwinn.model import (
    create_mta_swin_model,
)


def create_validation_loader():
    validation_transform = (
        transforms.Compose(
            [
                transforms.Resize(
                    (
                        config.IMAGE_SIZE,
                        config.IMAGE_SIZE,
                    ),
                    interpolation=(
                        transforms
                        .InterpolationMode
                        .BILINEAR
                    ),
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )
    )

    dataset = (
        DS2NetROIClassificationDataset(
            image_directory=(
                config.VAL_ROI_DIR
            ),
            split_file=(
                config.VAL_ROI_LIST
            ),
            transform=validation_transform,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return loader


def load_model(
    device,
):
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "MTA-Swin checkpoint not found: "
            f"{config.BEST_MODEL_PATH}"
        )

    model = create_mta_swin_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        dropout=config.DROPOUT,
        attention_hidden_dim=(
            config.ATTENTION_HIDDEN_DIM
        ),
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
    figure = plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    title = (
        "Normalized Confusion Matrix"
        if normalized
        else "Confusion Matrix"
    )

    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(
        config.NUM_CLASSES
    )

    class_labels = [
        config.CLASS_NAMES[index]
        for index in range(
            config.NUM_CLASSES
        )
    ]

    plt.xticks(
        tick_marks,
        class_labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        tick_marks,
        class_labels,
    )

    threshold = (
        matrix.max() / 2.0
        if matrix.size > 0
        else 0
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            value = matrix[
                row,
                column,
            ]

            text = (
                f"{value:.2f}"
                if normalized
                else str(int(value))
            )

            plt.text(
                column,
                row,
                text,
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

    plt.close(
        figure
    )


@torch.inference_mode()
def main():
    device = config.DEVICE

    config.RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 75)
    print(
        "DS2NET ROI MTA-SWIN EVALUATION"
    )
    print("=" * 75)

    print("Device:", device)
    print(
        "Checkpoint:",
        config.BEST_MODEL_PATH,
    )

    validation_loader = (
        create_validation_loader()
    )

    model = load_model(
        device
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []

    result_rows = []

    progress_bar = tqdm(
        validation_loader,
        desc="Evaluating MTA-Swin",
    )

    for (
        images,
        labels,
        image_names,
    ) in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )

        logits = model(
            images
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence_values, predictions = (
            probabilities.max(
                dim=1
            )
        )

        labels_numpy = labels.numpy()
        predictions_numpy = (
            predictions.cpu().numpy()
        )
        probabilities_numpy = (
            probabilities.cpu().numpy()
        )
        confidence_numpy = (
            confidence_values.cpu().numpy()
        )

        all_labels.extend(
            labels_numpy.tolist()
        )

        all_predictions.extend(
            predictions_numpy.tolist()
        )

        all_probabilities.extend(
            probabilities_numpy.tolist()
        )

        for index, image_name in enumerate(
            image_names
        ):
            true_label = int(
                labels_numpy[index]
            )

            predicted_label = int(
                predictions_numpy[index]
            )

            row = {
                "image_name": image_name,
                "true_label": true_label,
                "true_class": (
                    config.CLASS_NAMES[
                        true_label
                    ]
                ),
                "predicted_label": (
                    predicted_label
                ),
                "predicted_class": (
                    config.CLASS_NAMES[
                        predicted_label
                    ]
                ),
                "confidence": float(
                    confidence_numpy[index]
                ),
                "correct": int(
                    true_label
                    == predicted_label
                ),
            }

            for class_index in range(
                config.NUM_CLASSES
            ):
                row[
                    f"probability_class_"
                    f"{class_index}"
                ] = float(
                    probabilities_numpy[
                        index,
                        class_index,
                    ]
                )

            result_rows.append(
                row
            )

    probability_array = np.asarray(
        all_probabilities,
        dtype=np.float64,
    )

    metrics = (
        calculate_classification_metrics(
            labels=all_labels,
            predictions=all_predictions,
            probabilities=probability_array,
            num_classes=(
                config.NUM_CLASSES
            ),
        )
    )

    print()
    print("Evaluation Results")
    print("-" * 50)
    print(
        f"Accuracy          : "
        f"{metrics.accuracy:.4f}"
    )
    print(
        f"Balanced Accuracy : "
        f"{metrics.balanced_accuracy:.4f}"
    )
    print(
        f"Macro Precision   : "
        f"{metrics.macro_precision:.4f}"
    )
    print(
        f"Macro Recall      : "
        f"{metrics.macro_recall:.4f}"
    )
    print(
        f"Macro F1          : "
        f"{metrics.macro_f1:.4f}"
    )
    print(
        f"Weighted F1       : "
        f"{metrics.weighted_f1:.4f}"
    )

    if (
        metrics.roc_auc_ovr_macro
        is not None
    ):
        print(
            f"ROC-AUC OVR Macro : "
            f"{metrics.roc_auc_ovr_macro:.4f}"
        )
    else:
        print(
            "ROC-AUC OVR Macro : N/A"
        )

    if len(result_rows) > 0:
        with (
            config
            .EVALUATION_RESULTS_PATH
            .open(
                "w",
                newline="",
                encoding="utf-8",
            )
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    result_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                result_rows
            )

    summary_row = {
        "model": "DS2Net + MTA-Swin",
        "input": "DS2Net predicted bounding-box ROI",
        "accuracy": metrics.accuracy,
        "balanced_accuracy": (
            metrics.balanced_accuracy
        ),
        "macro_precision": (
            metrics.macro_precision
        ),
        "macro_recall": (
            metrics.macro_recall
        ),
        "macro_f1": metrics.macro_f1,
        "weighted_f1": (
            metrics.weighted_f1
        ),
        "roc_auc_ovr_macro": (
            metrics.roc_auc_ovr_macro
        ),
    }

    with (
        config.METRIC_SUMMARY_PATH.open(
            "w",
            newline="",
            encoding="utf-8",
        )
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_row.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(
            summary_row
        )

    confusion = build_confusion_matrix(
        labels=all_labels,
        predictions=all_predictions,
        num_classes=config.NUM_CLASSES,
        normalize=False,
    )

    normalized_confusion = (
        build_confusion_matrix(
            labels=all_labels,
            predictions=all_predictions,
            num_classes=(
                config.NUM_CLASSES
            ),
            normalize=True,
        )
    )

    save_confusion_matrix(
        matrix=confusion,
        output_path=(
            config.CONFUSION_MATRIX_PATH
        ),
        normalized=False,
    )

    save_confusion_matrix(
        matrix=normalized_confusion,
        output_path=(
            config
            .NORMALIZED_CONFUSION_MATRIX_PATH
        ),
        normalized=True,
    )

    print()
    print(
        "Detailed results:",
        config.EVALUATION_RESULTS_PATH,
    )
    print(
        "Metric summary:",
        config.METRIC_SUMMARY_PATH,
    )
    print(
        "Confusion matrix:",
        config.CONFUSION_MATRIX_PATH,
    )


if __name__ == "__main__":
    main()