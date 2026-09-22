import csv
import random
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ablation_ds2net_mtaswin_binary_context224_no_mask import config
from ablation_ds2net_mtaswin_binary_context224_no_mask.dataset import (
    DualContextBinaryDataset,
)
from ablation_ds2net_mtaswin_binary_context224_no_mask.model import (
    create_dual_context_model,
)


@dataclass
class BinaryMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    sensitivity: float
    specificity: float
    malignant_f1: float
    macro_f1: float


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_metrics(labels, predictions):
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )
    tn, fp, fn, tp = matrix.ravel()

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    return BinaryMetrics(
        accuracy=accuracy_score(labels, predictions),
        balanced_accuracy=balanced_accuracy_score(
            labels,
            predictions,
        ),
        precision=precision_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        sensitivity=recall_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        specificity=specificity,
        malignant_f1=f1_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        macro_f1=f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    )


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", alpha.float())

    def forward(self, logits, targets):
        log_probabilities = torch.log_softmax(
            logits,
            dim=1,
        )
        probabilities = log_probabilities.exp()

        target_log_probabilities = log_probabilities.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)
        target_probabilities = probabilities.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        loss = -(
            (1.0 - target_probabilities).pow(self.gamma)
            * target_log_probabilities
        )

        if self.alpha is not None:
            loss = loss * self.alpha[targets]

        return loss.mean()


def create_loaders():
    train_dataset = DualContextBinaryDataset(
        roi_directory=config.TRAIN_RGB_DIR,
        mask_directory=config.TRAIN_MASK_DIR,
        full_image_directory=config.FULL_IMAGE_DIR,
        split_file=config.TRAIN_LIST,
        image_size=config.IMAGE_SIZE,
        training=True,
    )
    val_dataset = DualContextBinaryDataset(
        roi_directory=config.VAL_RGB_DIR,
        mask_directory=config.VAL_MASK_DIR,
        full_image_directory=config.FULL_IMAGE_DIR,
        split_file=config.VAL_LIST,
        image_size=config.IMAGE_SIZE,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, train_dataset


def focal_alpha(dataset, device):
    labels = np.asarray(
        [sample[2] for sample in dataset.samples]
    )
    counts = np.bincount(
        labels,
        minlength=config.NUM_CLASSES,
    ).astype(np.float32)

    weights = len(labels) / (
        config.NUM_CLASSES * counts
    )

    print(
        "Training binary class counts:",
        counts.astype(int).tolist(),
    )
    print(
        "Focal alpha:",
        [round(float(value), 4) for value in weights],
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def create_optimizer(model, backbones_unfrozen):
    if not backbones_unfrozen:
        return torch.optim.AdamW(
            filter(
                lambda parameter: parameter.requires_grad,
                model.parameters(),
            ),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

    return torch.optim.AdamW(
        [
            {
                "params": list(
                    model.roi_features.parameters()
                )
                + list(model.roi_norm.parameters()),
                "lr": config.ROI_BACKBONE_LEARNING_RATE,
            },
            {
                "params": model.context_backbone.parameters(),
                "lr": config.CONTEXT_BACKBONE_LEARNING_RATE,
            },
            {
                "params": list(
                    model.roi_attention.parameters()
                )
                + list(model.classifier.parameters()),
                "lr": config.LEARNING_RATE,
            },
        ],
        weight_decay=config.WEIGHT_DECAY,
    )


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training,
    epoch=0,
):
    model.train(training)

    total_loss = 0.0
    labels_all = []
    predictions_all = []

    description = (
        f"Training epoch {epoch + 1}/{config.NUM_EPOCHS}"
        if training
        else "Validation"
    )

    context = (
        torch.enable_grad()
        if training
        else torch.inference_mode()
    )

    if training:
        optimizer.zero_grad(set_to_none=True)

    with context:
        progress = tqdm(loader, desc=description)

        for batch_index, batch in enumerate(progress):
            (
                roi_images,
                full_images,
                labels,
                _,
                _,
                _,
            ) = batch

            roi_images = roi_images.to(device)
            full_images = full_images.to(device)
            labels = labels.to(device)

            logits = model(
                roi_images,
                full_images,
            )

            raw_loss = criterion(logits, labels)
            loss = (
                raw_loss
                / config.GRADIENT_ACCUMULATION_STEPS
                if training
                else raw_loss
            )

            if training:
                loss.backward()

                should_step = (
                    (batch_index + 1)
                    % config.GRADIENT_ACCUMULATION_STEPS
                    == 0
                    or batch_index + 1 == len(loader)
                )

                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            predictions = logits.argmax(dim=1)
            batch_size = roi_images.size(0)

            total_loss += (
                raw_loss.item() * batch_size
            )
            labels_all.extend(
                labels.detach().cpu().tolist()
            )
            predictions_all.extend(
                predictions.detach().cpu().tolist()
            )

            progress.set_postfix(
                loss=f"{raw_loss.item():.4f}"
            )

    return (
        total_loss / len(loader.dataset),
        calculate_metrics(
            labels_all,
            predictions_all,
        ),
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    val_loss,
    metrics,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "validation_loss": val_loss,
            "validation_malignant_f1": metrics.malignant_f1,
            "validation_sensitivity": metrics.sensitivity,
            "validation_balanced_accuracy": metrics.balanced_accuracy,
        },
        path,
    )


def save_history(history):
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
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)


def plot_history(history):
    epochs = [row["epoch"] for row in history]

    figure = plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        [row["train_loss"] for row in history],
        label="Training loss",
    )
    plt.plot(
        epochs,
        [row["val_loss"] for row in history],
        label="Validation loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Dual-Context Binary Model Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    figure.savefig(
        config.TRAINING_CURVES_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure = plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        [row["train_malignant_f1"] for row in history],
        label="Training malignant F1",
    )
    plt.plot(
        epochs,
        [row["val_malignant_f1"] for row in history],
        label="Validation malignant F1",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Malignant F1")
    plt.title("Dual-Context Malignant F1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    figure.savefig(
        config.MALIGNANT_F1_CURVE_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    set_random_seed(config.RANDOM_SEED)
    device = config.DEVICE

    print("=" * 88)
    print(
        "ABLATION A1: DUAL-CONTEXT MTA-SWIN WITHOUT EXPLICIT MASK "
        f"AT {config.IMAGE_SIZE}x{config.IMAGE_SIZE}"
    )
    print("=" * 88)

    train_loader, val_loader, train_dataset = (
        create_loaders()
    )

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Batch size:", config.BATCH_SIZE)
    print(
        "Gradient accumulation:",
        config.GRADIENT_ACCUMULATION_STEPS,
    )

    model = create_dual_context_model(
        num_classes=config.NUM_CLASSES,
        pretrained=config.USE_PRETRAINED_WEIGHTS,
        dropout=config.DROPOUT,
        attention_hidden_dim=config.ATTENTION_HIDDEN_DIM,
    ).to(device)

    if config.FREEZE_BACKBONES_EPOCHS > 0:
        model.freeze_backbones()

    alpha = (
        focal_alpha(train_dataset, device)
        if config.FOCAL_USE_CLASS_ALPHA
        else None
    )

    criterion = FocalLoss(
        gamma=config.FOCAL_GAMMA,
        alpha=alpha,
    ).to(device)

    optimizer = create_optimizer(
        model,
        backbones_unfrozen=False,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        )
    )

    config.CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    config.RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_f1 = -1.0
    best_sensitivity = -1.0
    best_balanced_accuracy = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(config.NUM_EPOCHS):
        start = time.time()

        if epoch == config.FREEZE_BACKBONES_EPOCHS:
            model.unfreeze_backbones()

            optimizer = create_optimizer(
                model,
                backbones_unfrozen=True,
            )

            scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                )
            )

            print("\nROI Swin and context ResNet unfrozen.")

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
            epoch=epoch,
        )

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            training=False,
        )

        scheduler.step(val_metrics.malignant_f1)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_metrics.accuracy,
                "train_balanced_accuracy": (
                    train_metrics.balanced_accuracy
                ),
                "train_malignant_f1": (
                    train_metrics.malignant_f1
                ),
                "val_loss": val_loss,
                "val_accuracy": val_metrics.accuracy,
                "val_balanced_accuracy": (
                    val_metrics.balanced_accuracy
                ),
                "val_malignant_precision": (
                    val_metrics.precision
                ),
                "val_sensitivity": (
                    val_metrics.sensitivity
                ),
                "val_specificity": (
                    val_metrics.specificity
                ),
                "val_malignant_f1": (
                    val_metrics.malignant_f1
                ),
                "learning_rate": (
                    optimizer.param_groups[0]["lr"]
                ),
                "epoch_time_seconds": (
                    time.time() - start
                ),
            }
        )

        print(
            f"Epoch {epoch + 1}/{config.NUM_EPOCHS}\n"
            f"Train loss={train_loss:.4f}, "
            f"accuracy={train_metrics.accuracy:.4f}, "
            f"malignant F1={train_metrics.malignant_f1:.4f}\n"
            f"Val loss={val_loss:.4f}, "
            f"accuracy={val_metrics.accuracy:.4f}, "
            f"balanced accuracy="
            f"{val_metrics.balanced_accuracy:.4f}, "
            f"sensitivity={val_metrics.sensitivity:.4f}, "
            f"specificity={val_metrics.specificity:.4f}, "
            f"malignant F1={val_metrics.malignant_f1:.4f}"
        )

        save_checkpoint(
            config.LAST_MODEL_PATH,
            model,
            optimizer,
            scheduler,
            epoch,
            val_loss,
            val_metrics,
        )

        improved = (
            val_metrics.malignant_f1 > best_f1
            or (
                np.isclose(
                    val_metrics.malignant_f1,
                    best_f1,
                )
                and val_metrics.sensitivity
                > best_sensitivity
            )
            or (
                np.isclose(
                    val_metrics.malignant_f1,
                    best_f1,
                )
                and np.isclose(
                    val_metrics.sensitivity,
                    best_sensitivity,
                )
                and val_metrics.balanced_accuracy
                > best_balanced_accuracy
            )
        )

        if improved:
            best_f1 = val_metrics.malignant_f1
            best_sensitivity = val_metrics.sensitivity
            best_balanced_accuracy = (
                val_metrics.balanced_accuracy
            )
            epochs_without_improvement = 0

            save_checkpoint(
                config.BEST_MODEL_PATH,
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                val_metrics,
            )

            print(
                "Saved best checkpoint: "
                f"malignant F1={best_f1:.4f}, "
                f"sensitivity={best_sensitivity:.4f}, "
                f"balanced accuracy="
                f"{best_balanced_accuracy:.4f}"
            )
        else:
            epochs_without_improvement += 1

        save_history(history)
        plot_history(history)

        if (
            epochs_without_improvement
            >= config.EARLY_STOPPING_PATIENCE
        ):
            print("Early stopping triggered.")
            break

    print("Best malignant F1:", f"{best_f1:.4f}")
    print("Best sensitivity:", f"{best_sensitivity:.4f}")
    print(
        "Best balanced accuracy:",
        f"{best_balanced_accuracy:.4f}",
    )
    print("Best checkpoint:", config.BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
