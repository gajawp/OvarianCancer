import time

import torch
from tqdm import tqdm

from common import config
from common.loss import deep_supervision_loss
from common.metrics import dice_score, iou_score
from common.utils import (
    create_data_loaders,
    set_random_seed,
)
from ds2net.model import DS2NetSingleDomain


# ==========================================================
# Model Output Helper
# ==========================================================

def extract_final_logits(model_output):
    """
    Extract the final segmentation logits from DS2Net output.

    DS2Net may return:
    - a single tensor;
    - a tuple or list containing the final output and
      auxiliary deep-supervision outputs;
    - a dictionary containing the final output.
    """

    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, (tuple, list)):
        if len(model_output) == 0:
            raise ValueError(
                "DS2Net returned an empty tuple or list."
            )

        return model_output[0]

    if isinstance(model_output, dict):
        possible_keys = [
            "out",
            "logits",
            "prediction",
            "pred",
        ]

        for key in possible_keys:
            if key in model_output:
                return model_output[key]

        raise KeyError(
            "Could not find segmentation logits in the "
            "DS2Net output dictionary."
        )

    raise TypeError(
        "Unsupported DS2Net output type: "
        f"{type(model_output).__name__}"
    )


# ==========================================================
# Train One Epoch
# ==========================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
):
    """
    Train DS2Net for one epoch.

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
                "an image and a mask."
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

        outputs = model(images)

        loss = deep_supervision_loss(
            outputs,
            masks,
        )

        loss.backward()

        optimizer.step()

        final_logits = extract_final_logits(
            outputs
        )

        batch_size = images.size(0)

        batch_dice = dice_score(
            final_logits.detach(),
            masks,
            threshold=(
                config.PREDICTION_THRESHOLD
            ),
        ).item()

        batch_iou = iou_score(
            final_logits.detach(),
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
            "The DS2Net training DataLoader contained "
            "no samples."
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
    Save the DS2Net weights and training metadata.
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
        "random_seed": config.RANDOM_SEED,
        "training_list": str(
            config.TRAIN_LIST_PATH
        ),
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


# ==========================================================
# Main
# ==========================================================

def main():
    """
    Train DS2Net using only the official training split.

    train_cls.txt:
        Used for model training.

    val_cls.txt:
        Not used during training. It is evaluated separately
        by ds2net/evaluate.py.
    """

    set_random_seed(
        config.RANDOM_SEED
    )

    device = torch.device(
        config.DEVICE
    )

    print("=" * 70)
    print("DS2Net Segmentation Training")
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
    # Training DataLoader
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

    print(
        "Image size:",
        config.IMAGE_SIZE,
    )

    print(
        "Batch size:",
        config.BATCH_SIZE,
    )

    print(
        "Epochs:",
        config.EPOCHS,
    )

    print(
        "Learning rate:",
        config.LEARNING_RATE,
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = DS2NetSingleDomain(
        in_channels=3,
        out_channels=1,
    ).to(device)

    # ------------------------------------------------------
    # Optimizer and Scheduler
    # ------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )

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
    # Checkpoint
    # ------------------------------------------------------

    checkpoint_path = (
        config.DS2NET_MODEL_PATH
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_training_loss = float("inf")

    epochs_without_improvement = 0

    # ------------------------------------------------------
    # Epoch Loop
    # ------------------------------------------------------

    for epoch in range(
        config.EPOCHS
    ):
        epoch_start_time = time.time()

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
            - epoch_start_time
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
        # Best Checkpoint
        # --------------------------------------------------

        if training_loss < best_training_loss:
            best_training_loss = training_loss

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

    print("\n" + "=" * 70)
    print("DS2Net training completed")
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
        "\nRun ds2net.evaluate to calculate final metrics "
        "on the official test set."
    )


if __name__ == "__main__":
    main()