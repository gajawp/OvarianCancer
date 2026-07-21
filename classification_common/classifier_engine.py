from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification_common.dataset import (
    OvarianClassificationDataset,
    build_eval_transform,
    calculate_class_weights,
    create_dataloaders,
)
from classification_common.metrics import (
    calculate_multiclass_metrics,
    save_confusion_matrix,
    save_metrics,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_targets, all_predictions, all_probabilities = [], [], []

    for images, targets in tqdm(
        loader,
        desc="Training" if training else "Validating",
        leave=False,
    ):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        all_targets.extend(targets.detach().cpu().numpy())
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_probabilities.extend(probabilities.detach().cpu().numpy())

    return (
        total_loss / len(loader.dataset),
        np.asarray(all_targets),
        np.asarray(all_predictions),
        np.asarray(all_probabilities),
    )


def train_classifier(config, model_name: str, build_model: Callable):
    config.create_output_directories()
    config.validate_paths(require_masks=True)
    set_seed(config.random_seed)

    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(config)

    model = build_model(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        dropout=config.dropout,
    ).to(config.device)

    class_weights = None
    if config.use_class_weights:
        class_weights = calculate_class_weights(
            train_dataset.labels,
            config.num_classes,
        ).to(config.device)

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

    best_f1 = -1.0
    stale_epochs = 0
    history = []

    best_path = config.checkpoint_dir / f"{model_name}_best.pth"
    last_path = config.checkpoint_dir / f"{model_name}_last.pth"

    print("=" * 72)
    print(f"{model_name} classification training")
    print("=" * 72)
    print(f"Device: {config.device}")
    print(f"Input mode: {config.input_mode}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")

        train_loss, y_train, p_train, prob_train = _run_epoch(
            model, train_loader, criterion, config.device, optimizer
        )
        val_loss, y_val, p_val, prob_val = _run_epoch(
            model, val_loader, criterion, config.device
        )

        train_metrics = calculate_multiclass_metrics(
            y_train, p_train, prob_train, config.class_names, config.top_k
        )
        val_metrics = calculate_multiclass_metrics(
            y_val, p_val, prob_val, config.class_names, config.top_k
        )

        current_f1 = val_metrics["macro_f1"]
        scheduler.step(current_f1)

        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": current_f1,
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(record)

        print(
            f"Train loss={train_loss:.4f}, "
            f"accuracy={train_metrics['accuracy']:.4f}, "
            f"macro F1={train_metrics['macro_f1']:.4f}"
        )
        print(
            f"Val loss={val_loss:.4f}, "
            f"accuracy={val_metrics['accuracy']:.4f}, "
            f"balanced accuracy={val_metrics['balanced_accuracy']:.4f}, "
            f"macro F1={current_f1:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_macro_f1": max(best_f1, current_f1),
            "num_classes": config.num_classes,
            "class_names": config.class_names,
            "input_mode": config.input_mode,
            "image_size": config.image_size,
            "dropout": config.dropout,
        }
        torch.save(checkpoint, last_path)

        if current_f1 > best_f1:
            best_f1 = current_f1
            stale_epochs = 0
            torch.save(checkpoint, best_path)
            save_metrics(
                val_metrics,
                config.metrics_dir / f"{model_name}_best_validation_metrics.json",
            )
            print(f"Saved best checkpoint: macro F1={best_f1:.4f}")
        else:
            stale_epochs += 1

        with (
            config.metrics_dir / f"{model_name}_training_history.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        if stale_epochs >= config.early_stopping_patience:
            print("Early stopping triggered.")
            break

    print(f"Best validation macro F1: {best_f1:.4f}")
    print(f"Best checkpoint: {best_path}")


@torch.no_grad()
def evaluate_classifier(config, model_name: str, build_model: Callable):
    config.create_output_directories()
    config.validate_paths(require_masks=True)

    checkpoint_path = config.checkpoint_dir / f"{model_name}_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = OvarianClassificationDataset(
        image_dir=config.image_dir,
        list_path=config.val_list,
        num_classes=config.num_classes,
        transform=build_eval_transform(config.image_size),
        input_mode=config.input_mode,
        mask_dir=config.mask_dir,
        predicted_mask_dir=config.predicted_mask_dir,
        roi_padding=config.roi_padding,
        mask_threshold=config.mask_threshold,
        use_masked_roi=config.use_masked_roi,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.device,
        weights_only=False,
    )
    model = build_model(
        num_classes=checkpoint.get("num_classes", config.num_classes),
        pretrained=False,
        dropout=checkpoint.get("dropout", config.dropout),
    ).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    targets_all, predictions_all, probabilities_all = [], [], []

    for images, targets in tqdm(loader, desc="Evaluating"):
        logits = model(images.to(config.device, non_blocking=True))
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        targets_all.extend(targets.numpy())
        predictions_all.extend(predictions.cpu().numpy())
        probabilities_all.extend(probabilities.cpu().numpy())

    metrics = calculate_multiclass_metrics(
        targets_all,
        predictions_all,
        np.asarray(probabilities_all),
        config.class_names,
        config.top_k,
    )

    save_metrics(
        metrics,
        config.metrics_dir / f"{model_name}_evaluation_metrics.json",
    )

    confusion = np.asarray(metrics["confusion_matrix"])
    save_confusion_matrix(
        confusion,
        config.class_names,
        config.confusion_matrix_dir / f"{model_name}_confusion_matrix.png",
        normalize=False,
    )
    save_confusion_matrix(
        confusion,
        config.class_names,
        config.confusion_matrix_dir
        / f"{model_name}_confusion_matrix_normalized.png",
        normalize=True,
    )

    print("=" * 72)
    print(f"{model_name} evaluation")
    print("=" * 72)
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"Macro precision   : {metrics['macro_precision']:.4f}")
    print(f"Macro recall      : {metrics['macro_recall']:.4f}")
    print(f"Macro F1          : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1       : {metrics['weighted_f1']:.4f}")
    print(f"Top-3 accuracy    : {metrics.get('top_3_accuracy', float('nan')):.4f}")

    return metrics
