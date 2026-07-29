import argparse
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

from common import config as segmentation_config
from integration_ds2net_maskaware_mtaswin import config
from integration_ds2net_maskaware_mtaswin.generate_rgb_mask_dataset import (
    predict_ds2net_mask,
)
from integration_ds2net_maskaware_mtaswin.load_ds2net import (
    load_trained_ds2net,
)
from integration_ds2net_maskaware_mtaswin.model import (
    create_maskaware_mta_swin,
)
from integration_ds2net_maskaware_mtaswin.roi_utils import (
    extract_rgb_roi_and_mask,
)


def transform_inputs(rgb_roi_bgr, mask_roi):
    rgb_roi = cv2.cvtColor(
        rgb_roi_bgr,
        cv2.COLOR_BGR2RGB,
    )

    image = Image.fromarray(rgb_roi)
    mask = Image.fromarray(mask_roi * 255).convert("L")

    image = TF.resize(
        image,
        [config.IMAGE_SIZE, config.IMAGE_SIZE],
        interpolation=InterpolationMode.BILINEAR,
    )

    mask = TF.resize(
        mask,
        [config.IMAGE_SIZE, config.IMAGE_SIZE],
        interpolation=InterpolationMode.NEAREST,
    )

    image_tensor = TF.normalize(
        TF.to_tensor(image),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    mask_tensor = (TF.to_tensor(mask) >= 0.5).float()

    return image_tensor, mask_tensor


def load_classifier(device):
    model = create_maskaware_mta_swin(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        dropout=config.DROPOUT,
        attention_hidden_dim=config.ATTENTION_HIDDEN_DIM,
        mask_feature_dim=config.MASK_FEATURE_DIM,
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
def predict_one(image_path: Path):
    device = config.DEVICE

    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    ds2net = load_trained_ds2net(
        device=device,
        checkpoint_path=segmentation_config.DS2NET_MODEL_PATH,
    )

    predicted_mask = predict_ds2net_mask(
        ds2net,
        image_bgr,
        device,
    )

    rgb_roi, mask_roi, _, mask_found = extract_rgb_roi_and_mask(
        image_bgr,
        predicted_mask,
        padding=config.ROI_PADDING,
        minimum_area=config.MINIMUM_COMPONENT_AREA,
    )

    rgb_tensor, mask_tensor = transform_inputs(
        rgb_roi,
        mask_roi,
    )

    classifier = load_classifier(device)

    logits = classifier(
        rgb_tensor.unsqueeze(0).to(device),
        mask_tensor.unsqueeze(0).to(device),
    )

    probabilities = torch.softmax(logits, dim=1)
    confidence, predicted = probabilities.max(dim=1)

    predicted_index = int(predicted.item())

    print("Mask found:", mask_found)
    print("Predicted label:", predicted_index)
    print(
        "Predicted class:",
        config.CLASS_NAMES[predicted_index],
    )
    print("Confidence:", f"{float(confidence.item()):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    arguments = parser.parse_args()

    predict_one(arguments.image_path)


if __name__ == "__main__":
    main()
