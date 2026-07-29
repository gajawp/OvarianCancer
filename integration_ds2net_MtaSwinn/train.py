# integration_ds2net_MtaSwinn/train.py

import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import (
    compute_class_weight,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from integration_ds2net_MtaSwinn import config
from integration_ds2net_MtaSwinn.dataset import (
    DS2NetROIClassificationDataset,
)
from integration_ds2net_MtaSwinn.metrics import (
    calculate_classification_metrics,
)
from integration_ds2net_MtaSwinn.model import (
    create_mta_swin_model,
)


def set_random_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def create_transforms():
    train_transform = transforms.Compose(
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
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.RandomRotation(
                degrees=10
            ),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
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

    return (
        train_transform,
        validation_transform,
    )


def create_data_loaders():
    (
        train_transform,
        validation_transform,
    ) = create_transforms()

    train_dataset = (
        DS2NetROIClassificationDataset(
            image_directory=(
                config.TRAIN_ROI_DIR
            ),
            split_file=(
                config.TRAIN_ROI_LIST
            ),
            transform=train_transform,
        )
    )

    validation_dataset = (
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return (
        train_loader,
        validation_loader,
        train_dataset,
    )


def calculate_class_weights(
    train_dataset,
    device,
):
    labels = np.asarray(
        [
            label
            for _, label in (
                train_dataset.samples
            )
        ],
        dtype=np.int64,
    )

    present_classes = np.unique(
        labels
    )

    calculated_weights = (
        compute_class_weight(
            class_weight="balanced",
            classes=present_classes,
            y=labels,
        )
    )

    full_weights = np.ones(
        config.NUM_CLASSES,
        dtype=np.float32,
    )

    for class_index, weight in zip(
        present_classes,
        calculated_weights,
    ):
        full_weights[
            int(class_index)
        ] = float(weight)

    return torch.tensor(
        full_weights,
        dtype=torch.float32,
        device=device,
    )


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    epoch,
):
    model.train()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    progress_bar = tqdm(
        data_loader,
        desc=(
            f"Training epoch "
            f"{epoch + 1}/"
            f"{config.NUM_EPOCHS}"
        ),
    )

    for images, labels, _ in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .tolist()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .tolist()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = (
        total_loss
        / len(data_loader.dataset)
    )

    metrics = (
        calculate_classification_metrics(
            labels=all_labels,
            predictions=all_predictions,
            probabilities=None,
            num_classes=(
                config.NUM_CLASSES
            ),
        )
    )

    return average_loss, metrics


@torch.inference_mode()
def validate(
    model,
    data_loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    progress_bar = tqdm(
        data_loader,
        desc="Validation",
    )

    for images, labels, _ in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

    average_loss = (
        total_loss
        / len(data_loader.dataset)
    )

    probability_array = np.asarray(
        all_probabilities
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

    return average_loss, metrics


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch,
    validation_loss,
    validation_metrics,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
        ),
        "validation_loss": (
            validation_loss
        ),
        "validation_accuracy": (
            validation_metrics.accuracy
        ),
        "validation_balanced_accuracy": (
            validation_metrics
            .balanced_accuracy
        ),
        "validation_macro_f1": (
            validation_metrics.macro_f1
        ),
        "num_classes": (
            config.NUM_CLASSES
        ),
        "image_size": (
            config.IMAGE_SIZE
        ),
        "roi_mode": (
            config.ROI_MODE
        ),
        "roi_padding": (
            config.ROI_PADDING
        ),
    }

    torch.save(
        checkpoint,
        path,
    )


def save_history(
    history: list[dict],
):
    config.RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with config.TRAINING_HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                history[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            history
        )


def plot_history(
    history: list[dict],
):
    epochs = [
        row["epoch"]
        for row in history
    ]

    figure = plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        [
            row["train_loss"]
            for row in history
        ],
        label="Training loss",
    )

    plt.plot(
        epochs,
        [
            row["val_loss"]
            for row in history
        ],
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "DS2Net ROI MTA-Swin Loss"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    figure.savefig(
        config.TRAINING_CHART_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    f1_chart_path = (
        config.RESULTS_DIR
        / "macro_f1_curve.png"
    )

    figure = plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        [
            row["train_macro_f1"]
            for row in history
        ],
        label="Training macro F1",
    )

    plt.plot(
        epochs,
        [
            row["val_macro_f1"]
            for row in history
        ],
        label="Validation macro F1",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title(
        "DS2Net ROI MTA-Swin Macro F1"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    figure.savefig(
        f1_chart_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main():
    set_random_seed(
        config.RANDOM_SEED
    )

    device = config.DEVICE

    print("=" * 75)
    print(
        "DS2NET SEGMENTATION-GUIDED "
        "MTA-SWIN TRAINING"
    )
    print("=" * 75)

    print("Device:", device)
    print(
        "Training ROI directory:",
        config.TRAIN_ROI_DIR,
    )
    print(
        "Validation ROI directory:",
        config.VAL_ROI_DIR,
    )
    print(
        "Training list:",
        config.TRAIN_ROI_LIST,
    )
    print(
        "Validation list:",
        config.VAL_ROI_LIST,
    )
    print(
        "Number of classes:",
        config.NUM_CLASSES,
    )

    (
        train_loader,
        validation_loader,
        train_dataset,
    ) = create_data_loaders()

    print(
        "Training samples:",
        len(train_loader.dataset),
    )

    print(
        "Validation samples:",
        len(validation_loader.dataset),
    )

    model = create_mta_swin_model(
        num_classes=config.NUM_CLASSES,
        pretrained=(
            config.USE_PRETRAINED_WEIGHTS
        ),
        dropout=config.DROPOUT,
        attention_hidden_dim=(
            config.ATTENTION_HIDDEN_DIM
        ),
    ).to(device)

    if config.FREEZE_BACKBONE_EPOCHS > 0:
        model.freeze_backbone()

        print(
            "Swin backbone frozen for first",
            config.FREEZE_BACKBONE_EPOCHS,
            "epochs.",
        )

    class_weights = calculate_class_weights(
        train_dataset=train_dataset,
        device=device,
    )

    print(
        "Class weights:",
        class_weights.detach()
        .cpu()
        .numpy(),
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=(
            config.LABEL_SMOOTHING
        ),
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: (
                parameter.requires_grad
            ),
            model.parameters(),
        ),
        lr=config.LEARNING_RATE,
        weight_decay=(
            config.WEIGHT_DECAY
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        )
    )

    best_validation_f1 = -1.0
    epochs_without_improvement = 0

    history = []

    config.CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        config.NUM_EPOCHS
    ):
        epoch_start_time = time.time()

        if (
            epoch
            == config.FREEZE_BACKBONE_EPOCHS
        ):
            model.unfreeze_backbone()

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=(
                    config.LEARNING_RATE
                    * 0.25
                ),
                weight_decay=(
                    config.WEIGHT_DECAY
                ),
            )

            scheduler = (
                torch.optim.lr_scheduler
                .ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                )
            )

            print(
                "\nSwin backbone unfrozen."
            )

        (
            training_loss,
            training_metrics,
        ) = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        (
            validation_loss,
            validation_metrics,
        ) = validate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(
            validation_metrics.macro_f1
        )

        learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        epoch_time = (
            time.time()
            - epoch_start_time
        )

        history_row = {
            "epoch": epoch + 1,
            "train_loss": training_loss,
            "train_accuracy": (
                training_metrics.accuracy
            ),
            "train_macro_f1": (
                training_metrics.macro_f1
            ),
            "val_loss": validation_loss,
            "val_accuracy": (
                validation_metrics.accuracy
            ),
            "val_balanced_accuracy": (
                validation_metrics
                .balanced_accuracy
            ),
            "val_macro_f1": (
                validation_metrics.macro_f1
            ),
            "learning_rate": learning_rate,
            "epoch_time_seconds": (
                epoch_time
            ),
        }

        history.append(
            history_row
        )

        print()
        print(
            f"Epoch {epoch + 1}/"
            f"{config.NUM_EPOCHS}"
        )

        print(
            f"Train loss={training_loss:.4f}, "
            f"accuracy="
            f"{training_metrics.accuracy:.4f}, "
            f"macro F1="
            f"{training_metrics.macro_f1:.4f}"
        )

        print(
            f"Val loss={validation_loss:.4f}, "
            f"accuracy="
            f"{validation_metrics.accuracy:.4f}, "
            f"balanced accuracy="
            f"{validation_metrics.balanced_accuracy:.4f}, "
            f"macro F1="
            f"{validation_metrics.macro_f1:.4f}"
        )

        save_checkpoint(
            path=config.LAST_MODEL_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            validation_loss=validation_loss,
            validation_metrics=(
                validation_metrics
            ),
        )

        if (
            validation_metrics.macro_f1
            > best_validation_f1
        ):
            best_validation_f1 = (
                validation_metrics.macro_f1
            )

            epochs_without_improvement = 0

            save_checkpoint(
                path=config.BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                validation_loss=(
                    validation_loss
                ),
                validation_metrics=(
                    validation_metrics
                ),
            )

            print(
                "Saved best checkpoint: "
                f"macro F1="
                f"{best_validation_f1:.4f}"
            )

        else:
            epochs_without_improvement += 1

            print(
                "Epochs without improvement:",
                epochs_without_improvement,
            )

        save_history(
            history
        )

        plot_history(
            history
        )

        if (
            epochs_without_improvement
            >= config.EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping triggered."
            )
            break

    print()
    print("=" * 75)
    print("MTA-Swin training completed")
    print("=" * 75)

    print(
        "Best validation macro F1:",
        f"{best_validation_f1:.4f}",
    )

    print(
        "Best checkpoint:",
        config.BEST_MODEL_PATH,
    )


if __name__ == "__main__":
    main()