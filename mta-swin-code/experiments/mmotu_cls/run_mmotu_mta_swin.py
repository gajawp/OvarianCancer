"""
Train and evaluate MTA-Swin (ImageNet-1K pretrained) on MMOTU OTU_2d
8-class ovarian-tumor classification.

Data: MMOTU provides a flat image folder plus label files
    train_cls.txt / val_cls.txt   (each line:  "<name>.JPG  <label 0-7>")
Split: train_cls.txt is the training pool -> stratified 80/20 into train/val
       (seed 42) for early stopping / LR scheduling; val_cls.txt is the fixed
       held-out test set (this is the paper's official 1000/469 split).

Model: MTA-Swin (pretrained) by default; --model / --mode select any of the
comparison baselines (ResNet-50, Swin-T, ...) for pipeline sanity checks. Run:
    python experiments/mmotu_cls/run_mmotu_mta_swin.py
    python experiments/mmotu_cls/run_mmotu_mta_swin.py --model ResNet-50
"""

from __future__ import annotations

import argparse
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

from config import CLASS_NAMES, DEFAULT_CONFIG, MMOTUConfig, MTA_ROOT, PRETRAINED_WEIGHTS_ENV_VAR

# Make src/ importable (mta-swin-code root)
if str(MTA_ROOT) not in sys.path:
    sys.path.insert(0, str(MTA_ROOT))

from src.models import create_model


# AMP helpers (enabled only on CUDA)
try:
    from torch.amp import GradScaler as AmpGradScaler

    def create_grad_scaler(device: torch.device):
        return AmpGradScaler(device.type, enabled=(device.type == "cuda"))

    def autocast_context(device: torch.device):
        return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))

except ImportError:  # older torch
    from torch.cuda.amp import GradScaler as AmpGradScaler
    from torch.cuda.amp import autocast as cuda_autocast

    def create_grad_scaler(device: torch.device):
        return AmpGradScaler(enabled=(device.type == "cuda"))

    def autocast_context(device: torch.device):
        return cuda_autocast() if device.type == "cuda" else nullcontext()


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


class MTASwinModel(nn.Module):
    """MTA-Swin-Tiny built with the pretraining config, then head-swapped to
    num_classes. Loads ImageNet-1K weights (head stripped)."""

    def __init__(self, config: MMOTUConfig, pretrained: bool = True):
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
                    "MTA-Swin pretrained weights were requested but not found at "
                    f"{config.pretrained_weights_path}. Set {PRETRAINED_WEIGHTS_ENV_VAR} "
                    "or drop best_model.pth into the mta-swin-code root."
                )

            checkpoint = torch.load(config.pretrained_weights_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

            filtered_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith("head.")
            }
            incompatible = self.model.load_state_dict(filtered_state_dict, strict=False)

            # strict=False can silently skip everything on an arch mismatch, so
            # surface the load result explicitly. A tensor from the checkpoint is
            # "matched" if it is not reported as unexpected; head.* keys are the
            # only ones expected to be missing (we stripped them and re-init below).
            matched = len(filtered_state_dict) - len(incompatible.unexpected_keys)
            print(f"Loaded pretrained weights from: {config.pretrained_weights_path}")
            print(
                f"  matched tensors: {matched}/{len(filtered_state_dict)} | "
                f"missing: {len(incompatible.missing_keys)} | unexpected: {len(incompatible.unexpected_keys)}"
            )
            non_head_missing = [k for k in incompatible.missing_keys if not k.startswith("head.")]
            if non_head_missing:
                print(
                    f"  [WARNING] {len(non_head_missing)} non-head weights did NOT load "
                    "-- check that the checkpoint matches the MTA config. e.g.:"
                )
                for k in non_head_missing[:5]:
                    print(f"    missing: {k}")

        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, config.num_classes)

    def forward(self, x):
        return self.model(x)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


class CustomCNN(nn.Module):
    """Simple CNN baseline from the comparison experiment."""

    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, 7, stride=2, padding=3), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(128, 256, 5, stride=1, padding=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(1024, 2048), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_torchvision_model(model_name: str, num_classes: int, pretrained: bool):
    """Build a comparison-baseline model (torchvision or timm) with a fresh head."""
    if model_name == "ResNet-50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "EfficientNet-B4":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "ConvNeXt-T":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    if model_name == "Swin-T":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    if model_name == "DeiT-S/16":
        return timm.create_model("deit_small_patch16_224", pretrained=pretrained, num_classes=num_classes)
    if model_name == "ViT-S/16":
        return timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=num_classes)

    timm_ids = {
        "mamba": "mambaout_tiny.in1k",
        "maxvit": "maxvit_tiny_rw_224.sw_in1k",
        "davit": "davit_tiny.msft_in1k",
        "cait": "cait_s24_224.fb_dist_in1k",
        "inceptionnext": "inception_next_tiny.sail_in1k",
        "swinv2": "swinv2_cr_tiny_ns_224.sw_in1k",
    }
    key = model_name.lower()
    if key in timm_ids:
        return timm.create_model(timm_ids[key], pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unknown model name: {model_name}")


# Comparison model names accepted by --model (MTA-Swin is the default).
AVAILABLE_MODELS = (
    "MTA-Swin", "ResNet-50", "EfficientNet-B4", "ConvNeXt-T", "Swin-T",
    "DeiT-S/16", "ViT-S/16", "mamba", "maxvit", "davit", "cait",
    "inceptionnext", "swinv2", "Custom CNN",
)


def build_model(model_name: str, mode: str, config: MMOTUConfig) -> nn.Module:
    pretrained = mode == "pretrained"
    if model_name == "MTA-Swin":
        return MTASwinModel(config, pretrained=pretrained)
    if model_name == "Custom CNN":
        return CustomCNN(num_classes=config.num_classes)  # always from scratch
    return create_torchvision_model(model_name, config.num_classes, pretrained=pretrained)


def read_cls_dataframe(cls_path: Path, image_dir: Path) -> pd.DataFrame:
    """Parse a MMOTU *_cls.txt file into a dataframe of (file_path, label, label_name)."""
    records: list[tuple[str, int, str]] = []
    with open(cls_path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, label = parts[0], int(parts[1])
            file_path = image_dir / name
            label_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label)
            records.append((str(file_path), label, label_name))

    df = pd.DataFrame(records, columns=["file_path", "label", "label_name"])
    print(f"{cls_path.name}: {len(df)} samples")
    print(df["label_name"].value_counts().sort_index().to_string())
    print("-" * 40)
    return df


def build_transforms(config: MMOTUConfig):
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


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
            if self.counter >= self.patience:
                self.should_stop = True
            return
        self.best_score = value
        self.counter = 0
        self._save_checkpoint(value, model)

    def _save_checkpoint(self, value: float, model: nn.Module):
        if self.verbose:
            print(f"  monitored score improved ({self.best_value:.4f} -> {value:.4f}); saving checkpoint")
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
        total += labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Returns (loss, accuracy, y_true, y_pred) so callers can also compute
    imbalance-aware selection metrics without a second forward pass."""
    model.eval()
    running_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        targets = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
        y_true.extend(labels.numpy())
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    accuracy = float((y_true_arr == y_pred_arr).mean()) if total else 0.0
    return running_loss / total, accuracy, y_true_arr, y_pred_arr


def selection_score(y_true, y_pred, metric: str) -> float:
    """Higher-is-better score used for early stopping / LR scheduling."""
    if metric == "macro_f1":
        return float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2])
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    return float(accuracy_score(y_true, y_pred))  # "accuracy"


@torch.no_grad()
def evaluate_predictions(model, dataloader, device):
    model.eval()
    preds, labels_all, probs = [], [], []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        preds.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.numpy())
        probs.extend(probabilities.cpu().numpy())
    return (
        np.asarray(preds, dtype=np.int64),
        np.asarray(labels_all, dtype=np.int64),
        np.asarray(probs, dtype=np.float32),
    )


def build_per_class_metrics_table(y_true, y_pred, class_names):
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    specificities = []
    misclassified = []
    for idx in range(num_classes):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        total_class = cm[idx, :].sum()
        misclassified.append(f"{total_class - tp}/{total_class}")

    overall_accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
    total_misclassified = int(len(y_true) - np.sum(y_true == y_pred))

    rows = [
        {
            "Class": class_names[idx],
            "Support": int(support[idx]),
            "Precision": f"{precision[idx] * 100:.2f}%",
            "Sensitivity": f"{recall[idx] * 100:.2f}%",
            "Specificity": f"{specificities[idx] * 100:.2f}%",
            "F1-Score": f"{f1[idx] * 100:.2f}%",
            "Misclassified": misclassified[idx],
        }
        for idx in range(num_classes)
    ]
    rows.append(
        {
            "Class": "Macro Average",
            "Support": int(np.sum(support)),
            "Precision": f"{np.mean(precision) * 100:.2f}%",
            "Sensitivity": f"{np.mean(recall) * 100:.2f}%",
            "Specificity": f"{np.mean(specificities) * 100:.2f}%",
            "F1-Score": f"{np.mean(f1) * 100:.2f}%",
            "Misclassified": f"{total_misclassified}/{len(y_true)}",
        }
    )
    metrics_df = pd.DataFrame(rows)

    summary = {
        "accuracy": float(overall_accuracy),
        "macro_f1": float(macro_f1),
        "avg_precision": float(np.mean(precision)),
        "avg_sensitivity": float(np.mean(recall)),
        "avg_specificity": float(np.mean(specificities)),
        "total_misclassified": total_misclassified,
    }
    return cm, metrics_df, summary


def save_training_curves(train_losses, val_losses, train_accs, val_accs, title, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))
    plt.suptitle(title, fontsize=16, fontweight="bold")
    for i, (data, name, color) in enumerate(
        [
            (train_losses, "Training loss", "blue"),
            (train_accs, "Training accuracy", "blue"),
            (val_losses, "Validation loss", "red"),
            (val_accs, "Validation accuracy", "red"),
        ]
    ):
        plt.subplot(2, 2, i + 1)
        plt.plot(data, color=color, linewidth=2)
        plt.xlabel("Epoch")
        plt.title(name)
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(cm, class_names, title, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=15, fontweight="bold", pad=14)
    plt.ylabel("True Label", fontweight="bold")
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="MTA-Swin (pretrained) on MMOTU OTU_2d 8-class classification.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs (default: config).")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving training curves / confusion matrix.")
    parser.add_argument(
        "--model",
        choices=AVAILABLE_MODELS,
        default="MTA-Swin",
        help="Which comparison model to train (default: MTA-Swin).",
    )
    parser.add_argument(
        "--mode",
        choices=["pretrained", "scratch"],
        default="pretrained",
        help="Use ImageNet-pretrained weights or train from scratch (default: pretrained).",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["accuracy", "macro_f1", "balanced_accuracy"],
        default=None,
        help="Val metric that early stopping + LR scheduler track (default: config / accuracy).",
    )
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Use inverse-frequency class weights in CrossEntropyLoss.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Shortcut for imbalance handling: macro_f1 selection + class weights.",
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    num_epochs = args.epochs if args.epochs is not None else config.num_epochs

    # Resolve imbalance-handling switches (CLI overrides config; --balanced is a shortcut).
    selection_metric = args.selection_metric or ("macro_f1" if args.balanced else config.selection_metric)
    use_class_weights = args.class_weights or args.balanced or config.use_class_weights
    run_tag = (
        f"{sanitize_name(args.model)}_{args.mode}_{selection_metric}" + ("_cw" if use_class_weights else "")
    )
    model_display = f"{args.model} ({args.mode})"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model:            {model_display}")
    print(f"Image dir:        {config.image_dir}")
    print(f"Train cls file:   {config.train_cls_path}")
    print(f"Val (test) cls:   {config.val_cls_path}")
    if args.model == "MTA-Swin" and args.mode == "pretrained":
        print(f"Pretrained wts:   {config.pretrained_weights_path}")
    print(f"Seed:             {config.seed}")
    print(f"Selection metric: {selection_metric}")
    print(f"Class weights:    {use_class_weights}")
    print(f"Run tag:          {run_tag}\n")

    set_all_seeds(config.seed)

    # ---- data ----
    full_train_df = read_cls_dataframe(config.train_cls_path, config.image_dir)
    test_df = read_cls_dataframe(config.val_cls_path, config.image_dir)

    train_df, val_df = train_test_split(
        full_train_df,
        test_size=config.validation_split,
        random_state=config.seed,
        stratify=full_train_df["label"],
    )
    print(f"Split: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}\n")

    train_transform, eval_transform = build_transforms(config)
    train_loader = DataLoader(
        ImageDataset(train_df, train_transform), batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        ImageDataset(val_df, eval_transform), batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        ImageDataset(test_df, eval_transform), batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )

    # ---- model ----
    set_all_seeds(config.seed)
    model = build_model(args.model, args.mode, config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}\n")

    if use_class_weights:
        # Inverse-frequency ("balanced") weights from the TRAIN split only.
        counts = (
            train_df["label"].value_counts().reindex(range(config.num_classes), fill_value=0).sort_index().to_numpy()
        )
        counts = np.maximum(counts, 1)
        weights = counts.sum() / (config.num_classes * counts)
        class_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=config.label_smoothing)
        print("Class weights (per label 0..7):", np.round(weights, 3).tolist())
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.reduce_factor,
        patience=config.reduce_patience, min_lr=config.min_lr,
    )
    scaler = create_grad_scaler(device)

    checkpoint_path = config.checkpoint_dir / f"best_mta_swin_pretrained_{run_tag}.pt"
    early_stopping = EarlyStopping(config.early_stopping_patience, checkpoint_path, verbose=True)

    # ---- train ----
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    print("Starting training\n" + "-" * 60)
    for epoch in range(num_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_true, val_pred = validate(model, val_loader, criterion, device)
        val_macro_f1 = float(precision_recall_fscore_support(val_true, val_pred, average="macro", zero_division=0)[2])
        val_score = selection_score(val_true, val_pred, selection_metric)
        scheduler.step(val_score)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] {time.time() - t0:.1f}s | LR {lr:.2e} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_macro_f1:.4f} | "
            f"[{selection_metric}] {val_score:.4f} best {early_stopping.best_value:.4f}"
        )

        early_stopping(val_score, model)
        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch + 1} (best {selection_metric} {early_stopping.best_value:.4f})")
            break

    # ---- evaluate on held-out test (val_cls.txt) ----
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("\nRestored best checkpoint; evaluating on held-out test set (val_cls.txt)")

    test_loss, test_acc, _, _ = validate(model, test_loader, criterion, device)
    predictions, true_labels, probabilities = evaluate_predictions(model, test_loader, device)
    cm, metrics_df, summary = build_per_class_metrics_table(true_labels, predictions, list(CLASS_NAMES))

    bal_acc = balanced_accuracy_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predictions)
    try:
        roc_auc = roc_auc_score(
            true_labels, probabilities, multi_class="ovr", average="macro",
            labels=list(range(len(CLASS_NAMES))),
        )
    except Exception as exc:
        print(f"ROC-AUC failed: {exc!r}")
        roc_auc = float("nan")

    print("\n" + "=" * 60)
    print(f"TEST RESULTS ({model_display}, MMOTU OTU_2d, 8-class)")
    print("=" * 60)
    print(f"Accuracy           : {summary['accuracy'] * 100:.2f}%")
    print(f"Balanced Accuracy  : {bal_acc * 100:.2f}%")
    print(f"Macro F1           : {summary['macro_f1'] * 100:.2f}%")
    print(f"Macro Precision    : {summary['avg_precision'] * 100:.2f}%")
    print(f"Macro Sensitivity  : {summary['avg_sensitivity'] * 100:.2f}%")
    print(f"Macro Specificity  : {summary['avg_specificity'] * 100:.2f}%")
    print(f"MCC                : {mcc:.4f}")
    print(f"ROC-AUC (ovr macro): {roc_auc * 100:.2f}%")
    print(f"Test loss          : {test_loss:.4f}")
    print("\nPer-class metrics:")
    print(metrics_df.to_string(index=False))

    # ---- save artifacts ----
    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_row = {
        "Model": model_display,
        "Seed": config.seed,
        "Selection Metric": selection_metric,
        "Class Weights": use_class_weights,
        "Accuracy (%)": round(summary["accuracy"] * 100, 3),
        "Balanced Accuracy (%)": round(bal_acc * 100, 3),
        "Macro F1 (%)": round(summary["macro_f1"] * 100, 3),
        "Macro Precision (%)": round(summary["avg_precision"] * 100, 3),
        "Macro Sensitivity (%)": round(summary["avg_sensitivity"] * 100, 3),
        "Macro Specificity (%)": round(summary["avg_specificity"] * 100, 3),
        "MCC": round(mcc, 4),
        "ROC-AUC (%)": round(roc_auc * 100, 3),
        "Parameters": int(num_params),
        "Misclassified (n/total)": f"{summary['total_misclassified']}/{len(true_labels)}",
    }
    pd.DataFrame([summary_row]).to_csv(config.output_dir / f"summary_{run_tag}_{timestamp}.csv", index=False)
    metrics_df.to_csv(config.output_dir / f"per_class_{run_tag}_{timestamp}.csv", index=False)
    print(f"\nSaved summary + per-class CSVs to {config.output_dir}")

    if not args.no_plots:
        save_training_curves(
            train_losses, val_losses, train_accs, val_accs,
            title=f"{model_display} - MMOTU OTU_2d [{run_tag}]",
            output_path=config.plot_dir / f"training_curves_{run_tag}_{timestamp}.png",
        )
        save_confusion_matrix(
            cm, list(CLASS_NAMES),
            title=f"Confusion Matrix - {model_display} - MMOTU OTU_2d [{run_tag}]",
            output_path=config.plot_dir / f"confusion_matrix_{run_tag}_{timestamp}.png",
        )
        print(f"Saved plots to {config.plot_dir}")


if __name__ == "__main__":
    main()
