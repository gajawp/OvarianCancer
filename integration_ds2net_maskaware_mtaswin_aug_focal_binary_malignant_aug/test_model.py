import torch

from integration_ds2net_maskaware_mtaswin_aug_focal_binary_malignant_aug import config
from integration_ds2net_maskaware_mtaswin_aug_focal_binary_malignant_aug.model import (
    create_maskaware_mta_swin,
)


def main():
    device = config.DEVICE
    model = create_maskaware_mta_swin(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        dropout=config.DROPOUT,
        attention_hidden_dim=config.ATTENTION_HIDDEN_DIM,
        mask_feature_dim=config.MASK_FEATURE_DIM,
    ).to(device)

    rgb = torch.randn(
        2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device
    )
    mask = torch.randint(
        0,
        2,
        (2, 1, config.IMAGE_SIZE, config.IMAGE_SIZE),
        device=device,
    ).float()

    model.eval()
    with torch.no_grad():
        output = model(rgb, mask, return_attention=True)

    assert output["logits"].shape == (2, config.NUM_CLASSES)
    assert output["mask_features"].shape == (2, config.MASK_FEATURE_DIM)

    print("Binary model test passed.")
    print("Device:", device)
    print("RGB input:", tuple(rgb.shape))
    print("Mask input:", tuple(mask.shape))
    print("Logits:", tuple(output["logits"].shape))
    print("Classes:", config.CLASS_NAMES)
    print("Attention:", tuple(output["attention_weights"].shape))
    print("Mask features:", tuple(output["mask_features"].shape))


if __name__ == "__main__":
    main()
