import torch

from attention_unet.model import AttentionUNet


def main():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = AttentionUNet(
        in_channels=3,
        out_channels=1,
    ).to(device)

    sample_input = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        final, aux2, aux3, aux4 = model(
            sample_input
        )

    expected_shape = (2, 1, 256, 256)

    assert tuple(final.shape) == expected_shape
    assert tuple(aux2.shape) == expected_shape
    assert tuple(aux3.shape) == expected_shape
    assert tuple(aux4.shape) == expected_shape

    print("Attention U-Net test passed.")
    print("Device:", device)
    print("Final output:", tuple(final.shape))
    print("Auxiliary output 2:", tuple(aux2.shape))
    print("Auxiliary output 3:", tuple(aux3.shape))
    print("Auxiliary output 4:", tuple(aux4.shape))


if __name__ == "__main__":
    main()
