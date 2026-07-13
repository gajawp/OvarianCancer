import time

import torch
import torch.nn as nn
from tqdm import tqdm

from common import config
from common.metrics import dice_score, iou_score
from common.utils import create_data_loaders, set_random_seed
from segformer.model import SegFormer


class DiceLoss(nn.Module):
    def forward(self, logits, target):
        probabilities = torch.sigmoid(logits).flatten(1)
        target = target.flatten(1)
        intersection = (probabilities * target).sum(dim=1)
        smooth = 1e-6
        dice = (2.0 * intersection + smooth) / (
            probabilities.sum(dim=1) + target.sum(dim=1) + smooth
        )
        return 1.0 - dice.mean()


BCE_LOSS = nn.BCEWithLogitsLoss()
DICE_LOSS = DiceLoss()


def segmentation_loss(logits, target):
    return BCE_LOSS(logits, target) + DICE_LOSS(logits, target)


def validate_model(model, loader, device):
    model.eval()
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            total_dice += dice_score(
                logits,
                masks,
                threshold=config.PREDICTION_THRESHOLD,
            ).item()
            total_iou += iou_score(
                logits,
                masks,
                threshold=config.PREDICTION_THRESHOLD,
            ).item()

    return total_dice / len(loader), total_iou / len(loader)


def main():
    set_random_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)
    train_loader, validation_loader = create_data_loaders(config)

    model = SegFormer(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
    )

    checkpoint_path = config.SEGFORMER_MODEL_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_dice = -1.0
    epochs_without_improvement = 0

    print("Training SegFormer")
    print("------------------")
    print("Device:", device)
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(validation_loader.dataset))

    for epoch in range(config.EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}")
        for images, masks in progress:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        validation_dice, validation_iou = validate_model(
            model,
            validation_loader,
            device,
        )
        scheduler.step(validation_dice)

        print(
            f"Epoch [{epoch + 1}/{config.EPOCHS}] "
            f"Loss: {total_loss / len(train_loader):.4f} "
            f"Dice: {validation_dice:.4f} "
            f"IoU: {validation_iou:.4f} "
            f"LR: {optimizer.param_groups[0]['lr']:.7f} "
            f"Time: {time.time() - start_time:.1f}s"
        )

        if validation_dice > best_dice:
            best_dice = validation_dice
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved: {checkpoint_path} (best Dice: {best_dice:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print("\nTraining completed.")
    print(f"Best validation Dice: {best_dice:.4f}")
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
