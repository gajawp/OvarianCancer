from __future__ import annotations

import argparse
import os
import pickle
import random
import re
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from config import (
    DEFAULT_CONFIG,
    IMAGE_PATH_ENV_VAR,
    PRETRAINED_WEIGHTS_ENV_VAR,
    PROJECT_ROOT,
    ComparisonConfig,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import create_model


try:
    from torch.amp import GradScaler as AmpGradScaler

    def create_grad_scaler(device: torch.device):
        return AmpGradScaler(device.type, enabled=(device.type == "cuda"))

    def autocast_context(device: torch.device):
        return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))

except ImportError:
    from torch.cuda.amp import GradScaler as AmpGradScaler
    from torch.cuda.amp import autocast as cuda_autocast

    def create_grad_scaler(device: torch.device):
        return AmpGradScaler(enabled=(device.type == "cuda"))

    def autocast_context(device: torch.device):
        return cuda_autocast() if device.type == "cuda" else nullcontext()


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def format_model_display_name(model_name: str, mode: str) -> str:
    return f"{model_name} ({mode})" if mode == "scratch" else model_name


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["file_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(1024, 2048)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(2048, 512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class MTASwinModel(nn.Module):
    def __init__(self, config: ComparisonConfig, pretrained: bool):
        super().__init__()

        self.model = create_model(
            model_name="mta_swin_tiny",
            num_classes=1000,
            img_size=config.target_size[0],
            stage_qk_conv=list(config.mta_stage_qk_conv),
            stage_head_mixing=list(config.mta_stage_head_mixing),
            stage_group_norm=list(config.mta_stage_group_norm),
            cq=config.mta_cq,
            ck=config.mta_ck,
            ch=config.mta_ch,
            drop_path_rate=config.mta_drop_path_rate,
        )

        if pretrained:
            if not config.pretrained_weights_path.exists():
                raise FileNotFoundError(
                    "MTA-Swin pretrained weights were requested, but no checkpoint was found at "
                    f"{config.pretrained_weights_path}. Set {PRETRAINED_WEIGHTS_ENV_VAR} or update "
                    "experiments/comparison/config.py."
                )

            checkpoint = torch.load(config.pretrained_weights_path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

            filtered_state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("head.")
            }
            self.model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Loaded pretrained weights from: {config.pretrained_weights_path}")

        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, config.num_classes)

    def forward(self, x):
        return self.model(x)


def create_df_from_folder(folder_path: Path, label_map: dict[str, int]) -> pd.DataFrame:
    records: list[tuple[str, int, str]] = []
    print(f"Processing folder: {folder_path}")
    for label_name in label_map:
        label_path = folder_path / label_name
        if not label_path.exists():
            print(f"  {label_name}: folder not found")
            continue

        count = 0
        # Preserve the notebook's original directory traversal behavior.
        # The train/val split is created from the resulting dataframe order,
        # so changing the file iteration order changes the split even with the same seed.
        for filename in os.listdir(label_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = label_path / filename
                records.append((str(image_path), label_map[label_name], label_name))
                count += 1
        print(f"  {label_name}: {count} images")

    df = pd.DataFrame(records, columns=["file_path", "label", "label_name"])
    print(f"Total samples in folder: {len(df)}")
    if not df.empty:
        print("Class distribution:")
        print(df["label_name"].value_counts())
    print("-" * 40)
    return df


def build_transforms(config: ComparisonConfig):
    train_transform = transforms.Compose(
        [
            transforms.Resize(config.target_size, antialias=True),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 1.0))],
                p=0.3,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(config.target_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def load_dataset_frames(config: ComparisonConfig):
    train_dir = config.image_path / "Training"
    test_dir = config.image_path / "Testing"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            "Expected dataset folders were not found under "
            f"{config.image_path}. Set {IMAGE_PATH_ENV_VAR} or update experiments/comparison/config.py. "
            "The dataset root must contain Training/ and Testing/ subdirectories."
        )
    categories = sorted(
        category.name for category in train_dir.iterdir() if category.is_dir()
    )
    label_map = {category: idx for idx, category in enumerate(categories)}
    print(f"Label map: {label_map}")

    full_train_df = create_df_from_folder(train_dir, label_map)
    test_df = create_df_from_folder(test_dir, label_map)
    print("\n=== Dataset Loaded ===")
    print(f"Full training pool: {len(full_train_df)} samples")
    print(f"Test set (fixed):   {len(test_df)} samples")
    return full_train_df, test_df, label_map


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders_for_seed(
    seed: int,
    full_train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform,
    eval_transform,
    config: ComparisonConfig,
):
    train_df, val_df = train_test_split(
        full_train_df,
        test_size=config.validation_split,
        random_state=seed,
        stratify=full_train_df["label"],
    )

    train_dataset = ImageDataset(train_df, transform=train_transform)
    val_dataset = ImageDataset(val_df, transform=eval_transform)
    test_dataset = ImageDataset(test_df, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, test_loader, train_df, val_df


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def create_torchvision_model(model_name: str, num_classes: int, pretrained: bool):
    if model_name == "ResNet-50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "EfficientNet-B4":
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b4(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name == "ConvNeXt-T":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if model_name == "Swin-T":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    if model_name == "DeiT-S/16":
        return timm.create_model("deit_small_patch16_224", pretrained=pretrained, num_classes=num_classes)

    if model_name == "ViT-S/16":
        return timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=num_classes)

    if model_name.lower() == "mamba":
        timm_name = "mambaout_tiny.in1k"
    elif model_name.lower() in {"maxvit", "maxivt"}:
        timm_name = "maxvit_tiny_rw_224.sw_in1k"
    elif model_name.lower() == "davit":
        timm_name = "davit_tiny.msft_in1k"
    elif model_name.lower() == "cait":
        timm_name = "cait_s24_224.fb_dist_in1k"
    elif model_name.lower() == "inceptionnext":
        timm_name = "inception_next_tiny.sail_in1k"
    elif model_name.lower() == "swinv2":
        timm_name = "swinv2_cr_tiny_ns_224.sw_in1k"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if pretrained:
        try:
            return timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pretrained weights for {model_name} "
                f"(timm id: {timm_name}). Comparison runs do not fall back to scratch automatically. "
                "Ensure the pretrained weights are available in the current environment, "
                "or run the scratch variant explicitly."
            ) from exc
    return timm.create_model(timm_name, pretrained=False, num_classes=num_classes)


def build_model(model_name: str, mode: str, config: ComparisonConfig):
    if model_name == "Custom CNN":
        return CustomCNN(num_classes=config.num_classes)
    if model_name == "MTA-Swin":
        return MTASwinModel(config=config, pretrained=(mode == "pretrained"))
    return create_torchvision_model(model_name, num_classes=config.num_classes, pretrained=(mode == "pretrained"))


class EarlyStopping:
    def __init__(self, patience: int, path: Path, verbose: bool = False, delta: float = 0.0):
        self.patience = patience
        self.path = path
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_value = float("-inf")
        self.should_stop = False

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
            print(f"Validation accuracy increased ({self.best_value:.6f} --> {value:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.best_value = value


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast_context(device):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return running_loss / total, correct / total


def evaluate_predictions(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return (
        np.asarray(all_predictions, dtype=np.int64),
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_probabilities, dtype=np.float32),
    )


def build_per_class_metrics_table(y_true, y_pred, class_names):
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    specificities = []
    misclassified = []
    for idx in range(num_classes):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
        total_class_samples = cm[idx, :].sum()
        misclassified.append(f"{total_class_samples - tp}/{total_class_samples}")

    overall_accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
    total_samples = len(y_true)
    total_correct = int(np.sum(y_true == y_pred))
    total_misclassified = total_samples - total_correct

    metrics_df = pd.DataFrame(
        {
            "Class": class_names,
            "Precision": [f"{value * 100:.2f}%" for value in precision],
            "Sensitivity": [f"{value * 100:.2f}%" for value in recall],
            "Specificity": [f"{value * 100:.2f}%" for value in specificities],
            "F1-Score": [f"{value * 100:.2f}%" for value in f1],
            "Accuracy": [""] * num_classes,
            "Misclassified": misclassified,
        }
    )

    metrics_df = pd.concat(
        [
            metrics_df,
            pd.DataFrame(
                [
                    {
                        "Class": "Macro Average",
                        "Precision": f"{np.mean(precision) * 100:.2f}%",
                        "Sensitivity": f"{np.mean(recall) * 100:.2f}%",
                        "Specificity": f"{np.mean(specificities) * 100:.2f}%",
                        "F1-Score": f"{np.mean(f1) * 100:.2f}%",
                        "Accuracy": "",
                        "Misclassified": "",
                    },
                    {
                        "Class": "Overall",
                        "Precision": "",
                        "Sensitivity": "",
                        "Specificity": "",
                        "F1-Score": "",
                        "Accuracy": f"{overall_accuracy * 100:.2f}%",
                        "Misclassified": f"{total_misclassified}/{total_samples}",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    summary = {
        "accuracy": float(overall_accuracy),
        "macro_f1": float(macro_f1),
        "avg_precision": float(np.mean(precision)),
        "avg_sensitivity": float(np.mean(recall)),
        "avg_specificity": float(np.mean(specificities)),
        "total_misclassified": int(total_misclassified),
    }
    return cm, metrics_df, summary


def save_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, title: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))
    plt.suptitle(title, fontsize=18, fontweight="bold")

    plt.subplot(2, 2, 1)
    plt.plot(train_losses, color="blue", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, color="blue", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training accuracy")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(val_losses, color="red", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(val_accuracies, color="red", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation accuracy")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(cm, class_names, title: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 20},
    )
    plt.title(title, fontsize=18, fontweight="bold", pad=16)
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def estimate_flops_g(model: nn.Module, input_res: int) -> float:
    model = model.to("cpu").eval()
    dummy_input = torch.zeros(1, 3, input_res, input_res)

    try:
        from fvcore.nn import FlopCountAnalysis

        with torch.no_grad():
            return float(FlopCountAnalysis(model, dummy_input).total()) / 1e9
    except Exception:
        pass

    try:
        from thop import profile

        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return float(macs) * 2.0 / 1e9
    except Exception:
        print("FLOPs not computed; install fvcore or thop to enable this metric.")
        return float("nan")


@torch.no_grad()
def measure_inference_throughput_and_peak_mb(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    input_res: int,
    warmup: int = 10,
    iters: int = 50,
) -> tuple[float, float]:
    model.eval()
    x = torch.randn(batch_size, 3, input_res, input_res, device=device, dtype=torch.float32)

    try:
        if device.type == "cuda":
            for _ in range(warmup):
                _ = model(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

            t0 = time.time()
            for _ in range(iters):
                _ = model(x)
            torch.cuda.synchronize()
            t1 = time.time()

            imgs_per_s = batch_size * iters / max(t1 - t0, 1e-9)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            return float(imgs_per_s), float(peak_mb)

        for _ in range(warmup):
            _ = model(x)
        t0 = time.time()
        for _ in range(iters):
            _ = model(x)
        t1 = time.time()
        imgs_per_s = batch_size * iters / max(t1 - t0, 1e-9)
        return float(imgs_per_s), float("nan")
    except RuntimeError as exc:
        print(f"Throughput or peak-memory measurement failed: {exc!r}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return float("nan"), float("nan")


def save_seed_artifacts(seed: int, seed_df: pd.DataFrame, pred_store: list[dict], artifact_dir: Path):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_path = artifact_dir / f"seed{seed}_metrics_{timestamp}.csv"
    preds_path = artifact_dir / f"seed{seed}_preds_{timestamp}.pkl"

    seed_df.to_csv(metrics_path, index=False)
    with preds_path.open("wb") as handle:
        pickle.dump(pred_store, handle)

    print(f"Saved metrics CSV: {metrics_path}")
    print(f"Saved predictions PKL: {preds_path} (runs={len(pred_store)})")
    return metrics_path, preds_path, timestamp


def run_seed(
    seed: int,
    config: ComparisonConfig,
    device: torch.device,
    full_train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_map: dict[str, int],
    train_transform,
    eval_transform,
    checkpoint_root: Path,
    save_plots: bool,
    plot_root: Path | None,
    skip_efficiency: bool,
):
    print("\n" + "#" * 110)
    print(f"RUN SEED = {seed}")
    print("#" * 110)

    set_all_seeds(seed)
    train_loader, val_loader, test_loader, train_df, val_df = make_loaders_for_seed(
        seed=seed,
        full_train_df=full_train_df,
        test_df=test_df,
        train_transform=train_transform,
        eval_transform=eval_transform,
        config=config,
    )
    print(f"Dataset split @ seed={seed}: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

    checkpoint_dir = checkpoint_root / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    class_names = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]
    efficiency_cache: dict[str, dict[str, float]] = {}
    pred_store: list[dict] = []
    seed_results: list[dict] = []

    for model_name, mode in config.model_specs:
        display_name = format_model_display_name(model_name, mode)
        print("\n" + "=" * 80)
        print(f"TRAINING MODEL: {display_name} | seed={seed}")
        print("=" * 80)

        set_all_seeds(seed)
        model = build_model(model_name, mode, config)
        if not skip_efficiency:
            efficiency_cache.setdefault(display_name, {})
            if "FLOPs (G)" not in efficiency_cache[display_name]:
                efficiency_cache[display_name]["FLOPs (G)"] = estimate_flops_g(
                    model,
                    input_res=config.target_size[0],
                )

        num_params = count_trainable_parameters(model)
        print(f"Model parameters: {num_params:,}")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.reduce_factor,
            patience=config.reduce_patience,
            min_lr=config.min_lr,
        )
        scaler = create_grad_scaler(device)

        checkpoint_path = checkpoint_dir / f"best_{sanitize_name(model_name)}_{mode}.pt"
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            path=checkpoint_path,
            verbose=True,
            delta=0.0,
        )

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        epoch_times = []

        print(f"\nStarting {display_name} Training | seed={seed}")
        print(
            f"Initial LR: {config.learning_rate:.2e} | Reduce factor: {config.reduce_factor} | "
            f"Patience: {config.reduce_patience} | Min LR: {config.min_lr:.2e}"
        )
        print("-" * 60)

        for epoch in range(config.num_epochs):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step(val_acc)

            dt = time.time() - t0
            epoch_times.append(dt)

            early_stopping(val_acc, model)
            if early_stopping.should_stop:
                print("Early stopping triggered")
                break

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            if epoch == 0 or (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                avg_time = sum(epoch_times) / len(epoch_times)
                print(
                    f"\nEpoch [{epoch + 1}/{config.num_epochs}] - Time: {dt:.2f}s | "
                    f"Avg: {avg_time:.2f}s | LR: {current_lr:.2e}"
                )
                print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
                print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"\nBest model weights restored for {display_name} | seed={seed}")

        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

        predictions, true_labels, probabilities = evaluate_predictions(model, test_loader, device)
        cm, metrics_df, summary_metrics = build_per_class_metrics_table(true_labels, predictions, class_names)

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
            print(f"ROC-AUC failed for {display_name} seed={seed}: {exc!r}")
            roc_auc = float("nan")

        pred_store.append(
            {
                "Model": display_name,
                "Seed": seed,
                "y_true": np.asarray(true_labels, dtype=np.int64),
                "y_pred": np.asarray(predictions, dtype=np.int64),
            }
        )

        eff = efficiency_cache.get(display_name, {})
        if not skip_efficiency and (
            "Peak GPU Mem (MB)" not in eff or "Throughput (imgs/s@bs=32)" not in eff
        ):
            throughput, peak_mb = measure_inference_throughput_and_peak_mb(
                model=model,
                device=device,
                batch_size=config.batch_size,
                input_res=config.target_size[0],
            )
            eff["Peak GPU Mem (MB)"] = peak_mb
            eff["Throughput (imgs/s@bs=32)"] = throughput
            efficiency_cache[display_name] = eff

        avg_time_per_epoch = float(sum(epoch_times) / len(epoch_times)) if epoch_times else float("nan")
        total_test = int(len(true_labels))
        misclassified = int(summary_metrics["total_misclassified"])

        seed_results.append(
            {
                "Seed": seed,
                "Model": display_name,
                "Accuracy (%)": float(summary_metrics["accuracy"] * 100.0),
                "Balanced Accuracy (%)": float(bal_acc * 100.0),
                "MCC": float(mcc),
                "ROC-AUC (%)": float(roc_auc * 100.0),
                "Macro F1 (%)": float(summary_metrics["macro_f1"] * 100.0),
                "Precision (%)": float(summary_metrics["avg_precision"] * 100.0),
                "Sensitivity (%)": float(summary_metrics["avg_sensitivity"] * 100.0),
                "Specificity (%)": float(summary_metrics["avg_specificity"] * 100.0),
                "FLOPs (G)": _safe_float(eff.get("FLOPs (G)", float("nan"))),
                "Peak GPU Mem (MB)": _safe_float(eff.get("Peak GPU Mem (MB)", float("nan"))),
                "Throughput (imgs/s@bs=32)": _safe_float(eff.get("Throughput (imgs/s@bs=32)", float("nan"))),
                "Parameters": int(num_params),
                "Time/Epoch (s)": avg_time_per_epoch,
                "Misclassified (n/total)": f"{misclassified}/{total_test}",
            }
        )

        if save_plots and plot_root is not None:
            model_plot_dir = plot_root / f"seed_{seed}" / sanitize_name(display_name)
            save_training_curves(
                train_losses=train_losses,
                val_losses=val_losses,
                train_accuracies=train_accuracies,
                val_accuracies=val_accuracies,
                title=f"Training Curves - {display_name} | seed={seed}",
                output_path=model_plot_dir / "training_curves.png",
            )
            save_confusion_matrix(
                cm=cm,
                class_names=class_names,
                title=f"Confusion Matrix - {display_name} | seed={seed}",
                output_path=model_plot_dir / "confusion_matrix.png",
            )
            metrics_df.to_csv(model_plot_dir / "per_class_metrics.csv", index=False)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    seed_df = pd.DataFrame(seed_results)
    rounded = seed_df.copy()
    for column in [
        "Accuracy (%)",
        "Balanced Accuracy (%)",
        "ROC-AUC (%)",
        "Macro F1 (%)",
        "Precision (%)",
        "Sensitivity (%)",
        "Specificity (%)",
        "FLOPs (G)",
        "Peak GPU Mem (MB)",
        "Throughput (imgs/s@bs=32)",
        "Time/Epoch (s)",
    ]:
        if column in rounded.columns:
            rounded[column] = rounded[column].map(lambda value: round(float(value), 3) if pd.notnull(value) else value)

    print("\nPer-seed summary:")
    print(rounded.to_string(index=False))
    return seed_df, pred_store


def parse_args():
    parser = argparse.ArgumentParser(description="Run one seed of the comparison experiment.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for the train/val split.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_CONFIG.seed_run_dir,
        help="Directory for per-seed CSV and PKL outputs.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CONFIG.checkpoint_root,
        help="Root directory for model checkpoints.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save training curves, confusion matrices, and per-class metrics.",
    )
    parser.add_argument(
        "--plot-root",
        type=Path,
        default=DEFAULT_CONFIG.plot_root,
        help="Root directory for saved plots and diagnostics.",
    )
    parser.add_argument(
        "--skip-efficiency",
        action="store_true",
        help="Skip FLOPs, throughput, and peak-memory measurements.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = DEFAULT_CONFIG

    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_train_df, test_df, label_map = load_dataset_frames(config)
    train_transform, eval_transform = build_transforms(config)

    seed_df, pred_store = run_seed(
        seed=args.seed,
        config=config,
        device=device,
        full_train_df=full_train_df,
        test_df=test_df,
        label_map=label_map,
        train_transform=train_transform,
        eval_transform=eval_transform,
        checkpoint_root=args.checkpoint_root,
        save_plots=args.save_plots,
        plot_root=args.plot_root,
        skip_efficiency=args.skip_efficiency,
    )

    save_seed_artifacts(
        seed=args.seed,
        seed_df=seed_df,
        pred_store=pred_store,
        artifact_dir=args.artifact_dir,
    )


if __name__ == "__main__":
    main()
