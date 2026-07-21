from pathlib import Path
from typing import Dict
import csv
import math
import torch
import torch.nn.functional as F

from .config import CFG, get_device
from .dataset import build_loaders
from .losses import DiceBCELoss
from .metrics import batch_metrics, hausdorff_distance


def ensure_size(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def train_model(model, model_name: str, optimizer, output_dir: Path) -> Path:
    device = get_device()
    model.to(device)
    train_loader, val_loader = build_loaders()
    criterion = DiceBCELoss()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_model.pth"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    best_dice = -1.0
    stale_epochs = 0

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = ensure_size(model(images), masks)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_dice = 0.0
        count = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = ensure_size(model(images), masks)
                metrics = batch_metrics(logits, masks, CFG.threshold)
                val_dice += metrics["dice"] * images.size(0)
                count += images.size(0)
        val_dice /= max(count, 1)
        scheduler.step(val_dice)
        average_loss = train_loss / len(train_loader.dataset)
        print(f"{model_name} | epoch {epoch:03d} | train loss {average_loss:.4f} | val dice {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "best_val_dice": best_dice}, checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= CFG.patience:
                print("Early stopping.")
                break
    return checkpoint


def evaluate_model(model, checkpoint: Path, model_name: str, output_dir: Path) -> Dict[str, float]:
    device = get_device()
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"] if "model_state_dict" in payload else payload)
    model.to(device).eval()
    _, val_loader = build_loaders()
    totals = {key: 0.0 for key in ["dice", "iou", "precision", "recall", "specificity"]}
    hausdorff_values = []
    count = 0

    with torch.no_grad():
        for images, masks, _ in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits = ensure_size(model(images), masks)
            values = batch_metrics(logits, masks, CFG.threshold)
            batch_size = images.size(0)
            for key, value in values.items():
                totals[key] += value * batch_size
            predictions = (torch.sigmoid(logits) >= CFG.threshold).cpu().numpy()
            targets = masks.cpu().numpy()
            for pred, target in zip(predictions, targets):
                hd = hausdorff_distance(pred[0], target[0])
                if not math.isnan(hd):
                    hausdorff_values.append(hd)
            count += batch_size

    results = {key: value / max(count, 1) for key, value in totals.items()}
    results["hausdorff"] = sum(hausdorff_values) / len(hausdorff_values) if hausdorff_values else float("nan")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *results.keys()])
        writer.writeheader()
        writer.writerow({"model": model_name, **results})
    print(f"\n{model_name} evaluation")
    for key, value in results.items():
        print(f"{key.capitalize():12s}: {value:.4f}")
    print(f"Saved: {csv_path}")
    return results
