import torch

from ablation_ds2net_mtaswin_binary_roi_mask_cross_entropy import config
from ablation_ds2net_mtaswin_binary_roi_mask_cross_entropy.model import (
    create_roi_mask_model,
)


def main():
    device = config.DEVICE

    model = create_roi_mask_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        dropout=config.DROPOUT,
        attention_hidden_dim=config.ATTENTION_HIDDEN_DIM,
        mask_feature_dim=config.MASK_FEATURE_DIM,
    ).to(device)

    roi = torch.randn(
        1,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=device,
    )
    mask = torch.randint(
        0,
        2,
        (
            1,
            1,
            config.IMAGE_SIZE,
            config.IMAGE_SIZE,
        ),
        device=device,
    ).float()

    model.eval()

    with torch.no_grad():
        output = model(
            roi,
            mask,
            return_features=True,
        )

    assert output["logits"].shape == (
        1,
        config.NUM_CLASSES,
    )

    print("ROI-and-mask cross-entropy ablation test passed.")
    print("Device:", device)
    print("ROI input:", tuple(roi.shape))
    print("Mask input:", tuple(mask.shape))
    print("Logits:", tuple(output["logits"].shape))
    print(
        "ROI features:",
        tuple(output["roi_features"].shape),
    )
    print(
        "Mask features:",
        tuple(output["mask_features"].shape),
    )
    print(
        "Fused features:",
        tuple(output["fused_features"].shape),
    )


if __name__ == "__main__":
    main()
