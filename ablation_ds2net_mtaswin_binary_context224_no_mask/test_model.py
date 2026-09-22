import torch

from ablation_ds2net_mtaswin_binary_context224_no_mask import config
from ablation_ds2net_mtaswin_binary_context224_no_mask.model import (
    create_dual_context_model,
)


def main():
    device = config.DEVICE

    model = create_dual_context_model(
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
    full_image = torch.randn(
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
            full_image,
            return_features=True,
        )

    assert output["logits"].shape == (
        1,
        config.NUM_CLASSES,
    )

    print("Dual-context no-mask ablation test passed.")
    print("Device:", device)
    print("ROI input:", tuple(roi.shape))
    print("Full-image input:", tuple(full_image.shape))
    print("Logits:", tuple(output["logits"].shape))
    print(
        "ROI features:",
        tuple(output["roi_features"].shape),
    )
    print(
        "Context features:",
        tuple(output["context_features"].shape),
    )
    print(
        "Fused features:",
        tuple(output["fused_features"].shape),
    )


if __name__ == "__main__":
    main()
