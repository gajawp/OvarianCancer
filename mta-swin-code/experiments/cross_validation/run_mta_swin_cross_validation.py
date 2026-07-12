from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_DIR = PROJECT_ROOT / "experiments" / "comparison"
for path in (PROJECT_ROOT, COMPARISON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import DEFAULT_CONFIG, ComparisonConfig
from run_seed_comparison import (
    ImageDataset,
    MTASwinModel,
    build_per_class_metrics_table,
    build_transforms,
    create_grad_scaler,
    evaluate_predictions,
    sanitize_name,
    save_confusion_matrix,
    save_training_curves,
    set_all_seeds,
    train_one_epoch,
    validate,
)


DEFAULT_DATASET_ROOT = Path("/projects/insightx-lab/cleaned-brain-tumour-dataset")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "cross_validation" / "outputs" / "mta_swin_5fold_seed1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run stratified K-fold cross-validation for ImageNet-pretrained MTA-Swin. "
            "The cleaned dataset's Training and Testing folders are merged before splitting."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Cleaned dataset root.")
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=DEFAULT_CONFIG.pretrained_weights_path,
        help="ImageNet-pretrained MTA-Swin checkpoint used for fold initialization.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--folds", type=int, default=5, help="Number of stratified folds.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for K-fold shuffling and inner validation split.")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_CONFIG.num_epochs, help="Maximum epochs per fold.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_CONFIG.early_stopping_patience,
        help="Early stopping patience based on inner validation accuracy.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG.num_workers, help="DataLoader workers.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_CONFIG.validation_split,
        help="Validation fraction carved from the training folds for early stopping.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save training curves, confusion matrices, and per-class metrics for each fold.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Persist each fold's best checkpoint to disk. By default, best weights are kept in memory only.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    return parser.parse_args()


def build_cv_config(args: argparse.Namespace) -> ComparisonConfig:
    return replace(
        DEFAULT_CONFIG,
        image_path=args.dataset_root,
        pretrained_weights_path=args.pretrained_weights,
        batch_size=args.batch_size,
        num_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        validation_split=args.validation_split,
        num_workers=args.num_workers,
    )


def create_df_from_split(dataset_root: Path, split: str, label_map: dict[str, int]) -> pd.DataFrame:
    records: list[tuple[str, int, str, str]] = []
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Expected split folder not found: {split_dir}")

    for label_name, label in label_map.items():
        class_dir = split_dir / label_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append((str(image_path), label, label_name, split))

    return pd.DataFrame(records, columns=["file_path", "label", "label_name", "source_split"])


def load_cv_dataframe(dataset_root: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    train_dir = dataset_root / "Training"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Expected Training folder not found: {train_dir}")

    class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    if not class_names:
        raise FileNotFoundError(f"No class folders found under {train_dir}")
    label_map = {class_name: idx for idx, class_name in enumerate(class_names)}

    frames = [
        create_df_from_split(dataset_root, split="Training", label_map=label_map),
        create_df_from_split(dataset_root, split="Testing", label_map=label_map),
    ]
    df = pd.concat(frames, ignore_index=True)
    print("\n=== Cross-validation dataset ===")
    print(f"Dataset root: {dataset_root}")
    print(f"Total samples: {len(df)}")
    print("Label map:", label_map)
    print("Class distribution:")
    print(df["label_name"].value_counts().sort_index())
    print("Source split distribution:")
    print(df["source_split"].value_counts().sort_index())
    return df, label_map


def make_fold_loaders(
    fold_seed: int,
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform,
    eval_transform,
    config: ComparisonConfig,
):
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config.validation_split,
        random_state=fold_seed,
        stratify=train_val_df["label"],
    )

    train_loader = DataLoader(
        ImageDataset(train_df, transform=train_transform),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        ImageDataset(val_df, transform=eval_transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        ImageDataset(test_df, transform=eval_transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return train_loader, val_loader, test_loader, train_df, val_df


class InMemoryEarlyStopping:
    def __init__(self, patience: int, verbose: bool = False, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_value = float("-inf")
        self.should_stop = False
        self.best_state_dict: dict[str, torch.Tensor] | None = None

    def __call__(self, value: float, model: nn.Module):
        if self.best_score is None:
            self.best_score = value
            self._save_checkpoint(value, model)
            return

        if value < self.best_score + self.delta:
            self.counter += 1
            if self.counter == 1:
                print(f"EarlyStopping counter starts: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
            return

        self.best_score = value
        self.counter = 0
        self._save_checkpoint(value, model)

    def _save_checkpoint(self, value: float, model: nn.Module):
        if self.verbose:
            print(
                f"Validation accuracy increased ({self.best_value:.6f} --> {value:.6f}). "
                "Saving best weights in memory..."
            )
        self.best_state_dict = {
            key: tensor.detach().cpu().clone()
            for key, tensor in model.state_dict().items()
        }
        self.best_value = value


class DiskEarlyStopping(InMemoryEarlyStopping):
    def __init__(self, patience: int, path: Path, verbose: bool = False, delta: float = 0.0):
        super().__init__(patience=patience, verbose=verbose, delta=delta)
        self.path = path

    def _save_checkpoint(self, value: float, model: nn.Module):
        if self.verbose:
            print(f"Validation accuracy increased ({self.best_value:.6f} --> {value:.6f}). Saving model...")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.best_value = value


def train_fold(
    fold_index: int,
    train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_map: dict[str, int],
    config: ComparisonConfig,
    device: torch.device,
    train_transform,
    eval_transform,
    output_dir: Path,
    seed: int,
    save_plots: bool,
    keep_checkpoints: bool,
) -> tuple[dict, dict]:
    fold_number = fold_index + 1
    fold_seed = seed + fold_index
    fold_dir = output_dir / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 100)
    print(f"FOLD {fold_number}")
    print("#" * 100)
    set_all_seeds(fold_seed)

    train_loader, val_loader, test_loader, train_df, val_df = make_fold_loaders(
        fold_seed=fold_seed,
        train_val_df=train_val_df,
        test_df=test_df,
        train_transform=train_transform,
        eval_transform=eval_transform,
        config=config,
    )
    print(f"Fold split: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")
    print("Fold test class distribution:")
    print(test_df["label_name"].value_counts().sort_index())

    model = MTASwinModel(config=config, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.reduce_factor,
        patience=config.reduce_patience,
        min_lr=config.min_lr,
    )
    scaler = create_grad_scaler(device)

    checkpoint_path = fold_dir / "best_MTA-Swin_pretrained.pt"
    if keep_checkpoints:
        early_stopping = DiskEarlyStopping(
            patience=config.early_stopping_patience,
            path=checkpoint_path,
            verbose=True,
            delta=0.0,
        )
    else:
        early_stopping = InMemoryEarlyStopping(
            patience=config.early_stopping_patience,
            verbose=True,
            delta=0.0,
        )

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []
    epoch_times: list[float] = []

    for epoch in range(config.num_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        early_stopping(val_acc, model)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{config.num_epochs}] | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"time={epoch_time:.2f}s"
            )

        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    if keep_checkpoints:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        if early_stopping.best_state_dict is None:
            raise RuntimeError("No best model state was captured during training.")
        model.load_state_dict(early_stopping.best_state_dict)

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    predictions, true_labels, probabilities = evaluate_predictions(model, test_loader, device)

    class_names = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]
    cm, per_class_df, summary_metrics = build_per_class_metrics_table(true_labels, predictions, class_names)

    bal_acc = balanced_accuracy_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predictions)
    try:
        roc_auc = roc_auc_score(
            true_labels,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=list(range(len(class_names))),
        )
    except Exception as exc:
        print(f"ROC-AUC failed for fold {fold_number}: {exc!r}")
        roc_auc = float("nan")

    fold_metrics = {
        "Fold": fold_number,
        "Train Samples": int(len(train_df)),
        "Validation Samples": int(len(val_df)),
        "Test Samples": int(len(test_df)),
        "Epochs Trained": int(len(epoch_times)),
        "Best Validation Accuracy (%)": float(early_stopping.best_value * 100.0),
        "Test Loss": float(test_loss),
        "Accuracy (%)": float(summary_metrics["accuracy"] * 100.0),
        "Balanced Accuracy (%)": float(bal_acc * 100.0),
        "MCC": float(mcc),
        "ROC-AUC (%)": float(roc_auc * 100.0),
        "Macro F1 (%)": float(summary_metrics["macro_f1"] * 100.0),
        "Precision (%)": float(summary_metrics["avg_precision"] * 100.0),
        "Sensitivity (%)": float(summary_metrics["avg_sensitivity"] * 100.0),
        "Specificity (%)": float(summary_metrics["avg_specificity"] * 100.0),
        "Misclassified (n/total)": f"{int(summary_metrics['total_misclassified'])}/{len(true_labels)}",
        "Checkpoint": str(checkpoint_path) if keep_checkpoints else "",
        "Best State Storage": "disk" if keep_checkpoints else "memory",
    }

    per_class_df.to_csv(fold_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame([fold_metrics]).to_csv(fold_dir / "fold_metrics.csv", index=False)
    with (fold_dir / "predictions.pkl").open("wb") as handle:
        pickle.dump(
            {
                "fold": fold_number,
                "test_file_paths": test_df["file_path"].tolist(),
                "y_true": np.asarray(true_labels, dtype=np.int64),
                "y_pred": np.asarray(predictions, dtype=np.int64),
                "probabilities": np.asarray(probabilities, dtype=np.float32),
            },
            handle,
        )

    if save_plots:
        save_training_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accuracies=train_accuracies,
            val_accuracies=val_accuracies,
            title=f"MTA-Swin Cross-Validation Fold {fold_number}",
            output_path=fold_dir / "training_curves.png",
        )
        save_confusion_matrix(
            cm=cm,
            class_names=class_names,
            title=f"MTA-Swin Cross-Validation Fold {fold_number}",
            output_path=fold_dir / "confusion_matrix.png",
        )

    prediction_record = {
        "Fold": fold_number,
        "y_true": np.asarray(true_labels, dtype=np.int64),
        "y_pred": np.asarray(predictions, dtype=np.int64),
        "probabilities": np.asarray(probabilities, dtype=np.float32),
        "test_file_paths": test_df["file_path"].tolist(),
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Fold summary:")
    print(pd.DataFrame([fold_metrics]).to_string(index=False))
    return fold_metrics, prediction_record


def summarize_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "Accuracy (%)",
        "Balanced Accuracy (%)",
        "MCC",
        "ROC-AUC (%)",
        "Macro F1 (%)",
        "Precision (%)",
        "Sensitivity (%)",
        "Specificity (%)",
    ]
    records = []
    for column in metric_columns:
        values = fold_df[column].astype(float)
        records.append(
            {
                "Metric": column,
                "Mean": float(values.mean()),
                "Std": float(values.std(ddof=1)),
                "Min": float(values.min()),
                "Max": float(values.max()),
            }
        )
    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if not args.pretrained_weights.exists():
        raise FileNotFoundError(f"Pretrained MTA-Swin checkpoint not found: {args.pretrained_weights}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_cv_config(args)
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Using pretrained weights: {config.pretrained_weights_path}")

    full_df, label_map = load_cv_dataframe(config.image_path)
    if args.folds > int(full_df["label"].value_counts().min()):
        raise ValueError("Number of folds exceeds the smallest class count.")

    train_transform, eval_transform = build_transforms(config)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_metrics: list[dict] = []
    prediction_records: list[dict] = []
    for fold_index, (train_val_index, test_index) in enumerate(splitter.split(full_df, full_df["label"])):
        train_val_df = full_df.iloc[train_val_index].reset_index(drop=True)
        test_df = full_df.iloc[test_index].reset_index(drop=True)
        metrics, predictions = train_fold(
            fold_index=fold_index,
            train_val_df=train_val_df,
            test_df=test_df,
            label_map=label_map,
            config=config,
            device=device,
            train_transform=train_transform,
            eval_transform=eval_transform,
            output_dir=output_dir,
            seed=args.seed,
            save_plots=args.save_plots,
            keep_checkpoints=args.keep_checkpoints,
        )
        fold_metrics.append(metrics)
        prediction_records.append(predictions)

    fold_df = pd.DataFrame(fold_metrics)
    summary_df = summarize_folds(fold_df)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fold_metrics_path = output_dir / "fold_metrics.csv"
    summary_path = output_dir / "cv_summary.csv"
    predictions_path = output_dir / "cv_predictions.pkl"
    metadata_path = output_dir / "cv_metadata.json"

    fold_df.to_csv(fold_metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with predictions_path.open("wb") as handle:
        pickle.dump(prediction_records, handle)

    metadata = {
        "timestamp": timestamp,
        "dataset_root": str(config.image_path),
        "pretrained_weights": str(config.pretrained_weights_path),
        "output_dir": str(output_dir),
        "folds": args.folds,
        "seed": args.seed,
        "keep_checkpoints": args.keep_checkpoints,
        "model": "MTA-Swin pretrained",
        "protocol": (
            "Training and Testing folders are merged into one cleaned image pool. "
            "A stratified K-fold split creates a held-out test fold for each run; "
            "the remaining folds are split again into train and validation subsets for early stopping."
        ),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "label_map": label_map,
        "total_samples": int(len(full_df)),
        "class_distribution": {
            str(key): int(value) for key, value in full_df["label_name"].value_counts().sort_index().items()
        },
        "source_split_distribution": {
            str(key): int(value) for key, value in full_df["source_split"].value_counts().sort_index().items()
        },
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\n=== Cross-validation fold metrics ===")
    print(fold_df.to_string(index=False))
    print("\n=== Cross-validation summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved {fold_metrics_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {predictions_path}")
    print(f"Saved {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
