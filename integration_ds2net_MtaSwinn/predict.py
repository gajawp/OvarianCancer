# integration_ds2net_MtaSwinn/predict.py

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from common import config as segmentation_config
from integration_ds2net_MtaSwinn import config
from integration_ds2net_MtaSwinn.generate_roi_dataset import (
    predict_mask,
)
from integration_ds2net_MtaSwinn.load_ds2net import (
    load_trained_ds2net,
)
from integration_ds2net_MtaSwinn.model import (
    create_mta_swin_model,
)
from integration_ds2net_MtaSwinn.roi_utils import (
    extract_segmentation_guided_roi,
)


def create_classifier_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (
                    config.IMAGE_SIZE,
                    config.IMAGE_SIZE,
                ),
                interpolation=(
                    transforms
                    .InterpolationMode
                    .BILINEAR
                ),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


def load_classifier(
    device,
):
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "MTA-Swin checkpoint not found: "
            f"{config.BEST_MODEL_PATH}"
        )

    model = create_mta_swin_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        dropout=config.DROPOUT,
        attention_hidden_dim=(
            config.ATTENTION_HIDDEN_DIM
        ),
    ).to(device)

    checkpoint = torch.load(
        config.BEST_MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


@torch.inference_mode()
def perform_end_to_end_prediction(
    image_path: Path,
    output_path: Path,
):
    device = config.DEVICE

    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    original_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    ds2net = load_trained_ds2net(
        device=device,
        checkpoint_path=(
            segmentation_config
            .DS2NET_MODEL_PATH
        ),
    )

    predicted_mask = predict_mask(
        model=ds2net,
        image_bgr=image_bgr,
        device=device,
    )

    (
        roi_bgr,
        cleaned_mask,
        mask_found,
    ) = extract_segmentation_guided_roi(
        image=image_bgr,
        predicted_mask=predicted_mask,
        mode=config.ROI_MODE,
        padding=config.ROI_PADDING,
    )

    roi_rgb = cv2.cvtColor(
        roi_bgr,
        cv2.COLOR_BGR2RGB,
    )

    classifier_transform = (
        create_classifier_transform()
    )

    classifier_input = (
        classifier_transform(
            Image.fromarray(
                roi_rgb
            )
        )
        .unsqueeze(0)
        .to(device)
    )

    classifier = load_classifier(
        device
    )

    logits = classifier(
        classifier_input
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    confidence, predicted_class = (
        probabilities.max(
            dim=1
        )
    )

    predicted_index = int(
        predicted_class.item()
    )

    confidence_value = float(
        confidence.item()
    )

    predicted_name = (
        config.CLASS_NAMES[
            predicted_index
        ]
    )

    mask_overlay = (
        original_rgb.copy()
    )

    colored_mask = np.zeros_like(
        original_rgb
    )

    colored_mask[
        cleaned_mask > 0
    ] = [255, 0, 0]

    mask_overlay = cv2.addWeighted(
        mask_overlay,
        0.70,
        colored_mask,
        0.30,
        0,
    )

    figure = plt.figure(
        figsize=(16, 4)
    )

    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(
        original_rgb
    )
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("DS2Net Mask")
    plt.imshow(
        cleaned_mask,
        cmap="gray",
    )
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("DS2Net Overlay")
    plt.imshow(
        mask_overlay
    )
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title(
        f"MTA-Swin ROI\n"
        f"{predicted_name} "
        f"({confidence_value:.3f})"
    )
    plt.imshow(
        roi_rgb
    )
    plt.axis("off")

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print("=" * 70)
    print("END-TO-END PREDICTION")
    print("=" * 70)
    print("Input image:", image_path)
    print("DS2Net mask found:", mask_found)
    print(
        "Predicted class index:",
        predicted_index,
    )
    print(
        "Predicted class:",
        predicted_name,
    )
    print(
        "Confidence:",
        f"{confidence_value:.4f}",
    )

    print("\nClass probabilities")

    for class_index, probability in enumerate(
        probabilities.squeeze(0)
        .cpu()
        .tolist()
    ):
        print(
            f"  {config.CLASS_NAMES[class_index]}: "
            f"{probability:.4f}"
        )

    print(
        "\nVisualization saved to:",
        output_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run DS2Net segmentation followed "
            "by MTA-Swin classification."
        )
    )

    parser.add_argument(
        "image_path",
        type=Path,
        help=(
            "Path to an ovarian ultrasound image."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            config.RESULTS_DIR
            / "end_to_end_prediction.png"
        ),
        help=(
            "Path for saving the prediction figure."
        ),
    )

    arguments = parser.parse_args()

    perform_end_to_end_prediction(
        image_path=arguments.image_path,
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()