import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import config
from common.utils import (
    load_model_checkpoint,
)
from transunet.model import TransUNet


def preprocess_image(image_path):
    image_path = Path(
        image_path
    )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: "
            f"{image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    original_image = image.copy()

    resized_image = cv2.resize(
        image,
        (
            config.IMAGE_SIZE,
            config.IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    normalized_image = (
        resized_image.astype(
            np.float32
        ) / 255.0
    ).transpose(
        2,
        0,
        1,
    )

    image_tensor = torch.from_numpy(
        normalized_image
    ).float().unsqueeze(0)

    return (
        original_image,
        resized_image,
        image_tensor,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a TransUNet "
            "segmentation mask."
        )
    )

    parser.add_argument(
        "image_path",
        help=(
            "Path to the input "
            "ultrasound image."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "transunet_prediction.png"
        ),
    )

    arguments = parser.parse_args()

    device = torch.device(
        getattr(
            config,
            "TRANSUNET_DEVICE",
            config.DEVICE,
        )
    )

    model = TransUNet(
        in_channels=3,
        out_channels=1,
    ).to(device)

    model = load_model_checkpoint(
        model=model,
        checkpoint_path=(
            config.TRANSUNET_MODEL_PATH
        ),
        device=device,
    )

    model.eval()

    (
        original_image,
        resized_image,
        image_tensor,
    ) = preprocess_image(
        arguments.image_path
    )

    with torch.no_grad():
        logits = model(
            image_tensor.to(device)
        )

        predicted_mask = (
            torch.sigmoid(logits)
            > config.PREDICTION_THRESHOLD
        ).float()

    predicted_mask = (
        predicted_mask.cpu()
        .squeeze()
        .numpy()
    )

    figure = plt.figure(
        figsize=(15, 4)
    )

    plt.subplot(1, 4, 1)
    plt.title("Original Ultrasound")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Resized Input")
    plt.imshow(resized_image)
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Predicted Mask")
    plt.imshow(
        predicted_mask,
        cmap="gray",
    )
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Prediction Overlay")
    plt.imshow(resized_image)
    plt.imshow(
        predicted_mask,
        cmap="jet",
        alpha=0.35,
    )
    plt.axis("off")

    plt.tight_layout()

    output_path = Path(
        arguments.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        "Prediction saved to:",
        output_path,
    )


if __name__ == "__main__":
    main()
