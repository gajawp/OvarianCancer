# integration_ds2net_MtaSwinn/test_model.py

import torch

from integration_ds2net_MtaSwinn import config
from integration_ds2net_MtaSwinn.model import (
    create_mta_swin_model,
)


def main():
    device = config.DEVICE

    print("=" * 70)
    print("MTA-SWIN MODEL TEST")
    print("=" * 70)

    model = create_mta_swin_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        dropout=config.DROPOUT,
        attention_hidden_dim=(
            config.ATTENTION_HIDDEN_DIM
        ),
    ).to(device)

    sample_input = torch.randn(
        2,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        output = model(
            sample_input,
            return_attention=True,
        )

    logits = output["logits"]

    attention_weights = (
        output["attention_weights"]
    )

    expected_logits_shape = (
        2,
        config.NUM_CLASSES,
    )

    assert tuple(
        logits.shape
    ) == expected_logits_shape

    assert attention_weights.ndim == 3

    attention_sum = (
        attention_weights.sum(
            dim=1
        )
    )

    expected_sum = torch.ones_like(
        attention_sum
    )

    assert torch.allclose(
        attention_sum,
        expected_sum,
        atol=1e-5,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Model test passed.")
    print("Device:", device)
    print(
        "Input shape:",
        tuple(sample_input.shape),
    )
    print(
        "Logit shape:",
        tuple(logits.shape),
    )
    print(
        "Attention shape:",
        tuple(
            attention_weights.shape
        ),
    )
    print(
        "Total parameters:",
        f"{parameter_count:,}",
    )
    print(
        "Trainable parameters:",
        f"{trainable_parameter_count:,}",
    )


if __name__ == "__main__":
    main()