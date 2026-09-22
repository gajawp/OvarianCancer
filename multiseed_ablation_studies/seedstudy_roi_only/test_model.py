import torch

from multiseed_ablation_studies.seedstudy_roi_only import config
from multiseed_ablation_studies.seedstudy_roi_only.model import (
    create_roi_model,
)


def main():
    device = config.DEVICE

    model = create_roi_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        dropout=config.DROPOUT,
        attention_hidden_dim=config.ATTENTION_HIDDEN_DIM,
    ).to(device)

    roi = torch.randn(
        1,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=device,
    )
    model.eval()

    with torch.no_grad():
        output = model(
            roi,
            return_features=True,
        )

    assert output["logits"].shape == (
        1,
        config.NUM_CLASSES,
    )

    print("RGB ROI-only ablation test passed.")
    print("Device:", device)
    print("ROI input:", tuple(roi.shape))
    print("Logits:", tuple(output["logits"].shape))
    print(
        "ROI features:",
        tuple(output["roi_features"].shape),
    )
    print(
        "Fused features:",
        tuple(output["fused_features"].shape),
    )


if __name__ == "__main__":
    main()
