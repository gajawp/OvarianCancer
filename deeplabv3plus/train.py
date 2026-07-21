import time

import torch
import torch.nn as nn
from tqdm import tqdm

from common import config
from common.metrics import dice_score, iou_score
from common.utils import (
    create_data_loaders,
    set_random_seed,
)
from deeplabv3plus.model import DeepLabV3Plus


# ==========================================================
# Dice Loss
# ==========================================================

class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    The model outputs raw logits. Sigmoid is applied inside
    the loss function before calculating Dice overlap.
    """

    def forward(self, logits, target):
        probabilities = torch.sigmoid(
            logits
        ).flatten(1)

        target = target.flatten(1)

        intersection = (
            probabilities * target
        ).sum(dim=1)

        smooth = 1e-6

        dice = (
            2.0 * intersection + smooth
        ) / (
            probabilities.sum(dim=1)
            + target.sum(dim=1)
            + smooth
        )

        return 1.0 - dice.mean()


# ==========================================================
# Segmentation Loss
# ==========================================================

BCE_LOSS = nn.BCEWithLogitsLoss()

DICE_LOSS = DiceLoss()


def segmentation_loss(logits, target):
    """
    Combined BCE and Dice loss.

    BCE improves pixel-level classification, while Dice
    directly encourages foreground-mask overlap.
    """

    binary_cross_entropy = BCE_LOSS(
        logits,
        target,
    )

    dice = DICE_LOSS(
        logits,
        target,
    )

    return (
        binary_cross_entropy
        + dice
    )


# ==========================================================
# Training Epoch
# ==========================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
):
    """
    Train the model for one epoch.

    Returns
    -------
    tuple
        average_loss, average_dice, average_iou
    """

    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    progress_bar = tqdm(
        train_loader,
        desc=(
            f"Epoch {epoch + 1}/"
            f"{config.EPOCHS}"
        ),
    )

    for batch in progress_bar:
        if len(batch) < 2:
            raise ValueError(
                "The training DataLoader must return "
                "an image tensor and a mask tensor."
            )

        images = batch[0].to(
            device,
            non_blocking=True,
        )

        masks = batch[1].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        loss = segmentation_loss(
            logits,
            masks,
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        batch_dice = dice_score(
            logits.detach(),
            masks,
            threshold=(
                config.PREDICTION_THRESHOLD
            ),
        ).item()

        batch_iou = iou_score(
            logits.detach(),
            masks,
            threshold=(
                config.PREDICTION_THRESHOLD
            ),
        ).item()

        total_loss += (
            loss.item()
            * batch_size
        )

        total_dice += (
            batch_dice
            * batch_size
        )

        total_iou += (
            batch_iou
            * batch_size
        )

        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{batch_dice:.4f}",
            iou=f"{batch_iou:.4f}",
        )

    if total_samples == 0:
        raise RuntimeError(
            "The training DataLoader contained no samples."
        )

    average_loss = (
        total_loss
        / total_samples
    )

    average_dice = (
        total_dice
        / total_samples
    )

    average_iou = (
        total_iou
        / total_samples
    )

    return (
        average_loss,
        average_dice,
        average_iou,
    )


# ==========================================================
# Checkpoint Saving
# ==========================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    training_loss,
    training_dice,
    training_iou,
    checkpoint_path,
):
    """
    Save model and training state.

    A dictionary checkpoint is used so training metadata is
    retained along with the model weights.
    """

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "training_loss": training_loss,
        "training_dice": training_dice,
        "training_iou": training_iou,
        "random_seed": (
            config.RANDOM_SEED
        ),
        "training_list": str(
            config.TRAIN_LIST_PATH
        ),
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


# ==========================================================
# Main Training Function
# ==========================================================

def main():
    """
    Train DeepLabV3+ using only the official training split.

    Important:
    - train_cls.txt is used for model training.
    - val_cls.txt is not used during training.
    - The official test set is evaluated separately by
      evaluate.py.
    """

    set_random_seed(
        config.RANDOM_SEED
    )

    device = torch.device(
        config.DEVICE
    )

    print("=" * 70)
    print("DeepLabV3+ Segmentation Training")
    print("=" * 70)
    print("Device:", device)
    print(
        "Official training list:",
        config.TRAIN_LIST_PATH,
    )
    print(
        "Official test list:",
        config.TEST_LIST_PATH,
    )

    # ------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------

    train_loader, _ = create_data_loaders(
        config
    )

    print(
        "Training images:",
        len(train_loader.dataset),
    )

    print(
        "Official test images are not used during training."
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = DeepLabV3Plus(
        in_channels=3,
        out_channels=1,
    ).to(device)

    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )

    # The scheduler uses training loss rather than official
    # test performance to avoid test-set leakage.
    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
        )
    )

    # ------------------------------------------------------
    # Checkpoint Path
    # ------------------------------------------------------

    checkpoint_path = (
        config.DEEPLABV3PLUS_MODEL_PATH
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Training State
    # ------------------------------------------------------

    best_training_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    # ------------------------------------------------------
    # Epoch Loop
    # ------------------------------------------------------

    for epoch in range(
        config.EPOCHS
    ):
        start_time = time.time()

        (
            training_loss,
            training_dice,
            training_iou,
        ) = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        scheduler.step(
            training_loss
        )

        current_learning_rate = (
            optimizer
            .param_groups[0]["lr"]
        )

        elapsed_time = (
            time.time()
            - start_time
        )

        print(
            f"\nEpoch [{epoch + 1}/"
            f"{config.EPOCHS}]"
        )

        print(
            f"Training Loss : "
            f"{training_loss:.4f}"
        )

        print(
            f"Training Dice : "
            f"{training_dice:.4f}"
        )

        print(
            f"Training IoU  : "
            f"{training_iou:.4f}"
        )

        print(
            f"Learning Rate : "
            f"{current_learning_rate:.7f}"
        )

        print(
            f"Epoch Time    : "
            f"{elapsed_time:.1f} seconds"
        )

        # --------------------------------------------------
        # Save Best Model
        # --------------------------------------------------

        if (
            training_loss
            < best_training_loss
        ):
            best_training_loss = (
                training_loss
            )

            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                training_loss=training_loss,
                training_dice=training_dice,
                training_iou=training_iou,
                checkpoint_path=checkpoint_path,
            )

            print(
                "Best model saved:",
                checkpoint_path,
            )

            print(
                "Best training loss:",
                f"{best_training_loss:.4f}",
            )

        else:
            epochs_without_improvement += 1

            print(
                "Epochs without improvement:",
                epochs_without_improvement,
            )

        # --------------------------------------------------
        # Early Stopping
        # --------------------------------------------------

        if (
            epochs_without_improvement
            >= config.EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping triggered."
            )

            break

    # ------------------------------------------------------
    # Final Information
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("Training completed")
    print("=" * 70)

    print(
        "Best training loss:",
        f"{best_training_loss:.4f}",
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "\nRun evaluate.py to calculate final metrics "
        "on the official test set."
    )


if __name__ == "__main__":
    main()