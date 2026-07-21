from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from classification_common.config import get_config
from classification_common.dataset import (
    calculate_class_weights,
    create_dataloaders,
)
from classification_common.metrics import (
    calculate_multiclass_metrics,
    save_metrics,
)
from resnet50_classifier.model import build_resnet50


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    progress = tqdm(
        loader,
        desc="Training" if is_training else "Validating",
        leave=False,
    )

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, targets)

            if is_training:
                loss.backward()
                optimizer.step()

        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        all_targets.extend(targets.detach().cpu().numpy())
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_probabilities.extend(probabilities.detach().cpu().numpy())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = total_loss / len(loader.dataset)

    return (
        average_loss,
        np.asarray(all_targets),
        np.asarray(all_predictions),
        np.asarray(all_probabilities),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_macro_f1: float,
    config,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_macro_f1": best_macro_f1,
        "num_classes": config.num_classes,
        "class_names": config.class_names,
        "input_mode": config.input_mode,
        "image_size": config.image_size,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main() -> None:
    config = get_config()

    # Change this when running a different experiment:
    config.input_mode = "ground_truth_roi"
    # config.input_mode = "full_image"
    # config.input_mode = "predicted_roi"
    # config.predicted_mask_dir = (
    #     config.project_root / "classification_results" / "predicted_masks"
    # )

    config.create_output_directories()
    config.validate_paths(require_masks=True)

    set_seed(config.random_seed)

    print("=" * 72)
    print("ResNet50 Ovarian Tumor Classification Training")
    print("=" * 72)
    print(f"Device       : {config.device}")
    print(f"Input mode   : {config.input_mode}")
    print(f"Image size   : {config.image_size}")
    print(f"Batch size   : {config.batch_size}")
    print(f"Epochs       : {config.epochs}")
    print(f"Train split  : {config.train_list}")
    print(f"Val split    : {config.val_list}")

    (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
    ) = create_dataloaders(config)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")
    print(f"Train counts : {train_dataset.class_counts()}")
    print(f"Val counts   : {val_dataset.class_counts()}")

    model = build_resnet50(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        dropout=config.dropout,
    ).to(config.device)

    if config.use_class_weights:
        class_weights = calculate_class_weights(
            labels=train_dataset.labels,
            num_classes=config.num_classes,
        ).to(config.device)

        print(
            "Class weights: "
            f"{class_weights.detach().cpu().numpy().round(4).tolist()}"
        )
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        print("-" * 72)

        (
            train_loss,
            train_targets,
            train_predictions,
            train_probabilities,
        ) = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=config.device,
            optimizer=optimizer,
        )

        (
            val_loss,
            val_targets,
            val_predictions,
            val_probabilities,
        ) = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=config.device,
            optimizer=None,
        )

        train_metrics = calculate_multiclass_metrics(
            targets=train_targets,
            predictions=train_predictions,
            probabilities=train_probabilities,
            class_names=config.class_names,
            top_k_values=config.top_k,
        )

        val_metrics = calculate_multiclass_metrics(
            targets=val_targets,
            predictions=val_predictions,
            probabilities=val_probabilities,
            class_names=config.class_names,
            top_k_values=config.top_k,
        )

        current_macro_f1 = val_metrics["macro_f1"]
        scheduler.step(current_macro_f1)

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record: Dict = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics[
                "balanced_accuracy"
            ],
            "val_macro_f1": current_macro_f1,
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(epoch_record)

        print(
            f"Train loss={train_loss:.4f} | "
            f"accuracy={train_metrics['accuracy']:.4f} | "
            f"macro F1={train_metrics['macro_f1']:.4f}"
        )
        print(
            f"Val   loss={val_loss:.4f} | "
            f"accuracy={val_metrics['accuracy']:.4f} | "
            f"balanced accuracy="
            f"{val_metrics['balanced_accuracy']:.4f} | "
            f"macro F1={current_macro_f1:.4f}"
        )
        print(f"Learning rate: {current_lr:.8f}")

        save_checkpoint(
            path=config.last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_macro_f1=max(best_macro_f1, current_macro_f1),
            config=config,
        )

        if current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            epochs_without_improvement = 0

            save_checkpoint(
                path=config.best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_macro_f1=best_macro_f1,
                config=config,
            )

            save_metrics(
                metrics=val_metrics,
                output_path=(
                    config.metrics_dir
                    / "resnet50_best_validation_metrics.json"
                ),
            )

            print(
                f"Saved new best model: "
                f"macro F1={best_macro_f1:.4f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                "No macro-F1 improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )

        history_path = (
            config.metrics_dir / "resnet50_training_history.json"
        )
        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        if (
            epochs_without_improvement
            >= config.early_stopping_patience
        ):
            print(
                "\nEarly stopping triggered after "
                f"{epoch} epochs."
            )
            break

    print("\nTraining complete.")
    print(f"Best validation macro F1: {best_macro_f1:.4f}")
    print(f"Best checkpoint: {config.best_checkpoint_path}")


if __name__ == "__main__":
    main()
