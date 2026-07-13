import time

import torch
from tqdm import tqdm

from common import config
from common.loss import deep_supervision_loss
from common.metrics import dice_score, iou_score
from common.utils import create_data_loaders, set_random_seed
from attention_unet.model import AttentionUNet


def validate_model(model, validation_loader, device):
    model.eval()

    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in validation_loader:
            images = images.to(device)
            masks = masks.to(device)

            final_logits = model(images)[0]

            total_dice += dice_score(
                final_logits,
                masks,
                threshold=config.PREDICTION_THRESHOLD,
            ).item()

            total_iou += iou_score(
                final_logits,
                masks,
                threshold=config.PREDICTION_THRESHOLD,
            ).item()

    number_of_batches = len(validation_loader)

    return (
        total_dice / number_of_batches,
        total_iou / number_of_batches,
    )


def main():
    set_random_seed(config.RANDOM_SEED)

    device = torch.device(config.DEVICE)

    train_loader, validation_loader = create_data_loaders(config)

    model = AttentionUNet(
        in_channels=3,
        out_channels=1,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
    )

    checkpoint_path = config.ATTENTION_UNET_MODEL_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_dice = -1.0
    epochs_without_improvement = 0

    print("Training Attention U-Net")
    print("------------------------")
    print("Device:", device)
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(validation_loader.dataset))
    print("Image size:", config.IMAGE_SIZE)
    print("Batch size:", config.BATCH_SIZE)
    print("Epochs:", config.EPOCHS)
    print("Learning rate:", config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()

        model.train()
        total_training_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.EPOCHS}",
        )

        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = deep_supervision_loss(
                outputs,
                masks,
            )

            loss.backward()
            optimizer.step()

            total_training_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        average_training_loss = (
            total_training_loss / len(train_loader)
        )

        validation_dice, validation_iou = validate_model(
            model,
            validation_loader,
            device,
        )

        scheduler.step(validation_dice)

        current_learning_rate = optimizer.param_groups[0]["lr"]
        elapsed_time = time.time() - epoch_start_time

        print(
            f"Epoch [{epoch + 1}/{config.EPOCHS}] "
            f"Loss: {average_training_loss:.4f} "
            f"Dice: {validation_dice:.4f} "
            f"IoU: {validation_iou:.4f} "
            f"LR: {current_learning_rate:.7f} "
            f"Time: {elapsed_time:.1f}s"
        )

        if validation_dice > best_dice:
            best_dice = validation_dice
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print(
                f"Model saved: {checkpoint_path} "
                f"(best Dice: {best_dice:.4f})"
            )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= config.EARLY_STOPPING_PATIENCE
        ):
            print(
                "Early stopping triggered after "
                f"{config.EARLY_STOPPING_PATIENCE} "
                "epochs without improvement."
            )
            break

    print("\nTraining completed.")
    print(f"Best validation Dice: {best_dice:.4f}")
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
