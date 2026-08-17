import csv
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from integration_ds2net_maskaware_mtaswin_aug_focal import config
from integration_ds2net_maskaware_mtaswin_aug_focal.dataset import (
    MaskAwareClassificationDataset,
)
from integration_ds2net_maskaware_mtaswin.metrics import (
    calculate_classification_metrics,
)
from integration_ds2net_maskaware_mtaswin_aug_focal.model import (
    create_maskaware_mta_swin,
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_data_loaders():
    train_dataset = MaskAwareClassificationDataset(
        image_directory=config.TRAIN_RGB_DIR,
        mask_directory=config.TRAIN_MASK_DIR,
        split_file=config.TRAIN_LIST,
        image_size=config.IMAGE_SIZE,
        training=True,
    )

    val_dataset = MaskAwareClassificationDataset(
        image_directory=config.VAL_RGB_DIR,
        mask_directory=config.VAL_MASK_DIR,
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


class MulticlassFocalLoss(nn.Module):
    """Focal loss for single-label multiclass classification."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma = gamma

        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", alpha.float())

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        log_probabilities = torch.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()

        target_log_probabilities = log_probabilities.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        target_probabilities = probabilities.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        focal_factor = (1.0 - target_probabilities).pow(self.gamma)
        loss = -focal_factor * target_log_probabilities

        if self.alpha is not None:
            loss = loss * self.alpha[targets]

        return loss.mean()



def create_optimizer(model, backbone_unfrozen: bool):
    if not backbone_unfrozen:
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
                "params": list(model.rgb_features.parameters())
                + list(model.rgb_norm.parameters()),
                "lr": config.BACKBONE_LEARNING_RATE,
            },
            {
                "params": list(model.token_attention.parameters())
                + list(model.mask_encoder.parameters())
                + list(model.fusion_classifier.parameters()),
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
    training: bool,
    epoch: int = 0,
):
    model.train(training)

    total_loss = 0.0
    labels_all = []
    predictions_all = []
    probabilities_all = []

    description = (
        f"Training epoch {epoch + 1}/{config.NUM_EPOCHS}"
        if training
        else "Validation"
    )

    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        progress = tqdm(loader, desc=description)

        for rgb_images, masks, labels, _ in progress:
            rgb_images = rgb_images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(rgb_images, masks)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            batch_size = rgb_images.size(0)
            total_loss += loss.item() * batch_size

            labels_all.extend(labels.detach().cpu().tolist())
            predictions_all.extend(
                predictions.detach().cpu().tolist()
            )

            if not training:
                probabilities_all.extend(
                    probabilities.detach().cpu().numpy().tolist()
                )

            progress.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = total_loss / len(loader.dataset)

    metrics = calculate_classification_metrics(
        labels=labels_all,
        predictions=predictions_all,
        probabilities=(
            None
            if training
            else np.asarray(probabilities_all)
        ),
        num_classes=config.NUM_CLASSES,
    )

    return average_loss, metrics


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    validation_loss,
    validation_metrics,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "validation_loss": validation_loss,
            "validation_macro_f1": validation_metrics.macro_f1,
        },
        path,
    )


def save_history(history):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    plt.title("Augmented Focal-Loss Mask-Aware MTA-Swin Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    figure.savefig(
        config.TRAINING_LOSS_CHART_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure = plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        [row["train_macro_f1"] for row in history],
        label="Training macro F1",
    )
    plt.plot(
        epochs,
        [row["val_macro_f1"] for row in history],
        label="Validation macro F1",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Augmented Focal-Loss Mask-Aware MTA-Swin Macro F1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    figure.savefig(
        config.MACRO_F1_CHART_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    set_random_seed(config.RANDOM_SEED)
    device = config.DEVICE

    print("=" * 78)
    print("DS2NET RGB ROI + EXPLICIT MASK MTA-SWIN AUGMENTED FOCAL-LOSS TRAINING")
    print("=" * 78)

    train_loader, val_loader, _ = create_data_loaders()

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))

    model = create_maskaware_mta_swin(
        num_classes=config.NUM_CLASSES,
        pretrained=config.USE_PRETRAINED_WEIGHTS,
        dropout=config.DROPOUT,
        attention_hidden_dim=config.ATTENTION_HIDDEN_DIM,
        mask_feature_dim=config.MASK_FEATURE_DIM,
    ).to(device)

    if config.FREEZE_BACKBONE_EPOCHS > 0:
        model.freeze_rgb_backbone()

    criterion = MulticlassFocalLoss(
        gamma=config.FOCAL_GAMMA,
        alpha=None,
    ).to(device)

    print("Loss function: multiclass focal loss")
    print("Focal gamma:", config.FOCAL_GAMMA)
    print("Class alpha enabled:", config.FOCAL_USE_CLASS_ALPHA)
    print("Balanced sampler enabled: False")

    optimizer = create_optimizer(
        model,
        backbone_unfrozen=False,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
    )

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(config.NUM_EPOCHS):
        start = time.time()

        if epoch == config.FREEZE_BACKBONE_EPOCHS:
            model.unfreeze_rgb_backbone()

            optimizer = create_optimizer(
                model,
                backbone_unfrozen=True,
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
            )

            print("\nRGB Swin backbone unfrozen.")

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

        scheduler.step(val_metrics.macro_f1)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_metrics.accuracy,
                "train_macro_f1": train_metrics.macro_f1,
                "val_loss": val_loss,
                "val_accuracy": val_metrics.accuracy,
                "val_balanced_accuracy": val_metrics.balanced_accuracy,
                "val_macro_f1": val_metrics.macro_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_time_seconds": time.time() - start,
            }
        )

        print(
            f"Epoch {epoch + 1}/{config.NUM_EPOCHS}\n"
            f"Train loss={train_loss:.4f}, "
            f"accuracy={train_metrics.accuracy:.4f}, "
            f"macro F1={train_metrics.macro_f1:.4f}\n"
            f"Val loss={val_loss:.4f}, "
            f"accuracy={val_metrics.accuracy:.4f}, "
            f"balanced accuracy={val_metrics.balanced_accuracy:.4f}, "
            f"macro F1={val_metrics.macro_f1:.4f}"
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

        if val_metrics.macro_f1 > best_f1:
            best_f1 = val_metrics.macro_f1
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
                f"Saved best checkpoint: macro F1={best_f1:.4f}"
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

    print("Best validation macro F1:", f"{best_f1:.4f}")
    print("Best checkpoint:", config.BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
