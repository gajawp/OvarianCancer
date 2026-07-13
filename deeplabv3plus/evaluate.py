from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from common import config
from common.metrics import hausdorff_distance
from common.utils import (
    append_result,
    create_validation_loader,
    evaluate_batch_per_image,
    get_mean_metrics,
    initialize_metric_storage,
    load_model_checkpoint,
    print_metric_summary,
    save_evaluation_results,
    summarize_metrics,
)
from deeplabv3plus.model import DeepLabV3Plus


# ==========================================================
# Model Output Helper
# ==========================================================

def extract_final_logits(model_output):
    """
    Extract segmentation logits from the model output.

    Supports:
    - Tensor output
    - Tuple or list output
    - Dictionary output
    """

    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, (tuple, list)):
        if len(model_output) == 0:
            raise ValueError(
                "The model returned an empty tuple or list."
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
            "model output dictionary."
        )

    raise TypeError(
        "Unsupported model output type: "
        f"{type(model_output).__name__}"
    )


# ==========================================================
# Qualitative Result
# ==========================================================

def save_qualitative_result(
    image,
    target_mask,
    predicted_mask,
    index,
    save_directory,
):
    """
    Save:
    1. Input ultrasound image
    2. Ground-truth mask
    3. Predicted mask
    4. Prediction overlay
    """

    save_directory = Path(save_directory)

    save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_numpy = (
        image.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    target_numpy = (
        target_mask.detach()
        .cpu()
        .squeeze()
        .numpy()
    )

    image_numpy = np.clip(
        image_numpy,
        0.0,
        1.0,
    )

    figure = plt.figure(
        figsize=(15, 4)
    )

    plt.subplot(1, 4, 1)
    plt.title("Input Ultrasound")
    plt.imshow(image_numpy)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Ground Truth")
    plt.imshow(
        target_numpy,
        cmap="gray",
    )
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Prediction")
    plt.imshow(
        predicted_mask,
        cmap="gray",
    )
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Prediction Overlay")
    plt.imshow(image_numpy)
    plt.imshow(
        predicted_mask,
        cmap="jet",
        alpha=0.35,
    )
    plt.axis("off")

    plt.tight_layout()

    output_path = (
        save_directory
        / f"deeplabv3plus_result_{index:03d}.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ==========================================================
# Evaluation
# ==========================================================

def main():
    """
    Evaluate the trained DeepLabV3+ model.

    This script:
    - Loads the validation dataset
    - Loads the saved checkpoint
    - Calculates metrics per image
    - Reports mean and standard deviation
    - Saves qualitative results
    - Saves per-image metrics
    - Saves summary metrics
    - Updates the global comparison CSV
    """

    device = torch.device(
        config.DEVICE
    )

    print("Device:", device)

    # ------------------------------------------------------
    # Validation DataLoader
    # ------------------------------------------------------

    validation_loader = create_validation_loader(
        config=config,
        batch_size=1,
    )

    print(
        "Validation samples:",
        len(validation_loader.dataset),
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = DeepLabV3Plus(
        in_channels=3,
        out_channels=1,
    ).to(device)

    model = load_model_checkpoint(
        model=model,
        checkpoint_path=(
            config.DEEPLABV3PLUS_MODEL_PATH
        ),
        device=device,
    )

    model.eval()

    # ------------------------------------------------------
    # Output Paths
    # ------------------------------------------------------

    qualitative_directory = Path(
        config.DEEPLABV3PLUS_RESULTS_DIR
    )

    qualitative_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_csv_path = Path(
        config.MODEL_COMPARISON_PATH
    )

    results_root = (
        comparison_csv_path.parent
    )

    # ------------------------------------------------------
    # Metric Storage
    # ------------------------------------------------------

    metric_storage = (
        initialize_metric_storage()
    )

    result_rows = []

    # Retain ordinary Hausdorff Distance for compatibility
    # with your previous results.
    hausdorff_values = []

    # ------------------------------------------------------
    # Evaluation Loop
    # ------------------------------------------------------

    with torch.no_grad():
        progress_bar = tqdm(
            validation_loader,
            desc="Evaluating DeepLabV3+",
        )

        for index, batch in enumerate(
            progress_bar
        ):
            if len(batch) < 2:
                raise ValueError(
                    "The validation DataLoader must return "
                    "an image and a mask."
                )

            image = batch[0].to(
                device,
                non_blocking=True,
            )

            mask = batch[1].to(
                device,
                non_blocking=True,
            )

            model_output = model(image)

            logits = extract_final_logits(
                model_output
            )

            # Calculate all segmentation metrics per image.
            evaluate_batch_per_image(
                outputs=logits,
                masks=mask,
                metric_storage=metric_storage,
                result_rows=result_rows,
                threshold=(
                    config.PREDICTION_THRESHOLD
                ),
            )

            prediction_probability = torch.sigmoid(
                logits
            )

            prediction = (
                prediction_probability
                >= config.PREDICTION_THRESHOLD
            ).float()

            prediction_numpy = (
                prediction.detach()
                .cpu()
                .squeeze()
                .numpy()
            )

            mask_numpy = (
                mask.detach()
                .cpu()
                .squeeze()
                .numpy()
            )

            # Ordinary Hausdorff Distance.
            sample_hausdorff = hausdorff_distance(
                prediction_numpy,
                mask_numpy,
            )

            hausdorff_values.append(
                sample_hausdorff
            )

            result_rows[-1][
                "hausdorff_distance"
            ] = sample_hausdorff

            # Save the first 30 qualitative examples.
            if index < 30:
                save_qualitative_result(
                    image=image[0],
                    target_mask=mask[0],
                    predicted_mask=prediction_numpy,
                    index=index,
                    save_directory=(
                        qualitative_directory
                    ),
                )

    # ------------------------------------------------------
    # Summary Statistics
    # ------------------------------------------------------

    summary = summarize_metrics(
        metric_storage=metric_storage,
        sample_standard_deviation=False,
    )

    print_metric_summary(
        summary
    )

    valid_hausdorff_values = np.asarray(
        hausdorff_values,
        dtype=np.float64,
    )

    valid_hausdorff_values = (
        valid_hausdorff_values[
            np.isfinite(
                valid_hausdorff_values
            )
        ]
    )

    if len(valid_hausdorff_values) > 0:
        average_hausdorff = float(
            np.mean(
                valid_hausdorff_values
            )
        )

        std_hausdorff = float(
            np.std(
                valid_hausdorff_values
            )
        )
    else:
        average_hausdorff = np.nan
        std_hausdorff = np.nan

    print(
        f"{'hausdorff_distance':<22}: "
        f"{average_hausdorff:.4f} ± "
        f"{std_hausdorff:.4f}"
    )

    # ------------------------------------------------------
    # Save Detailed Results
    # ------------------------------------------------------

    save_evaluation_results(
        model_name="deeplabv3plus",
        result_rows=result_rows,
        summary=summary,
        results_root=results_root,
    )

    # ------------------------------------------------------
    # Update Global Comparison CSV
    # ------------------------------------------------------

    average_metrics = get_mean_metrics(
        summary
    )

    average_metrics[
        "hausdorff_distance"
    ] = average_hausdorff

    append_result(
        csv_path=comparison_csv_path,
        model_name="DeepLabV3+",
        metrics=average_metrics,
    )

    # ------------------------------------------------------
    # Final Messages
    # ------------------------------------------------------

    print(
        "\nDeepLabV3+ evaluation completed."
    )

    print(
        "Comparison table updated:",
        comparison_csv_path,
    )

    print(
        "Qualitative results saved in:",
        qualitative_directory,
    )

    print(
        "Detailed metric results saved in:",
        results_root / "deeplabv3plus",
    )


if __name__ == "__main__":
    main()