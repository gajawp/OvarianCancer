import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from common import config
from common.metrics import (
    calculate_numpy_metrics,
    hausdorff_distance,
)
from common.utils import (
    append_result,
    create_validation_loader,
    load_model_checkpoint,
)
from dsnet.model import DSNet


def save_qualitative_result(
    image,
    target_mask,
    predicted_mask,
    index,
    save_directory,
):
    """Save input, ground truth, prediction, and overlay."""

    image = image.permute(1, 2, 0).cpu().numpy()
    target_mask = target_mask.squeeze().cpu().numpy()

    figure = plt.figure(figsize=(15, 4))

    plt.subplot(1, 4, 1)
    plt.title("Input Ultrasound")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Ground Truth")
    plt.imshow(target_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Prediction")
    plt.imshow(predicted_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Prediction Overlay")
    plt.imshow(image)
    plt.imshow(
        predicted_mask,
        cmap="jet",
        alpha=0.35,
    )
    plt.axis("off")

    plt.tight_layout()

    output_path = save_directory / f"dsnet_result_{index:03d}.png"

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    device = torch.device(config.DEVICE)

    validation_loader = create_validation_loader(
        config,
        batch_size=1,
    )

    model = DSNet(
        in_channels=3,
        out_channels=1,
    ).to(device)

    model = load_model_checkpoint(
        model=model,
        checkpoint_path=config.DSNET_MODEL_PATH,
        device=device,
    )

    model.eval()

    save_directory = config.DSNET_RESULTS_DIR
    save_directory.mkdir(parents=True, exist_ok=True)

    metric_values = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "hausdorff_distance": [],
    }

    with torch.no_grad():
        for index, (image, mask) in enumerate(
            tqdm(
                validation_loader,
                desc="Evaluating DSNet",
            )
        ):
            image = image.to(device)
            mask = mask.to(device)

            final_logits = model(image)[0]

            probabilities = torch.sigmoid(final_logits)
            prediction = (
                probabilities
                > config.PREDICTION_THRESHOLD
            ).float()

            prediction_numpy = (
                prediction.cpu().numpy().squeeze()
            )
            mask_numpy = (
                mask.cpu().numpy().squeeze()
            )

            sample_metrics = calculate_numpy_metrics(
                prediction_numpy,
                mask_numpy,
            )

            for metric_name, metric_value in sample_metrics.items():
                metric_values[metric_name].append(metric_value)

            metric_values["hausdorff_distance"].append(
                hausdorff_distance(
                    prediction_numpy,
                    mask_numpy,
                )
            )

            if index < 30:
                save_qualitative_result(
                    image=image.cpu().squeeze(0),
                    target_mask=mask.cpu().squeeze(0),
                    predicted_mask=prediction_numpy,
                    index=index,
                    save_directory=save_directory,
                )

    average_metrics = {
        metric_name: float(np.nanmean(values))
        for metric_name, values in metric_values.items()
    }

    print("\nDSNet Evaluation Results")
    print("------------------------")
    print(f"Dice Score      : {average_metrics['dice']:.4f}")
    print(f"IoU Score       : {average_metrics['iou']:.4f}")
    print(f"Precision       : {average_metrics['precision']:.4f}")
    print(f"Recall          : {average_metrics['recall']:.4f}")
    print(f"Specificity     : {average_metrics['specificity']:.4f}")
    print(
        "Hausdorff Dist. : "
        f"{average_metrics['hausdorff_distance']:.4f}"
    )

    results_path = config.MODEL_COMPARISON_PATH

    append_result(
        csv_path=results_path,
        model_name="DSNet",
        metrics=average_metrics,
    )

    print("\nComparison table updated:", results_path)
    print("Qualitative results saved in:", save_directory)


if __name__ == "__main__":
    main()
