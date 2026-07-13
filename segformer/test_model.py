import torch

from segformer.model import SegFormer


def main():
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SegFormer(in_channels=3, out_channels=1).to(device)
    sample_input = torch.randn(2, 3, 256, 256, device=device)

    model.eval()

    with torch.no_grad():
        output = model(sample_input)

    expected_shape = (2, 1, 256, 256)
    assert tuple(output.shape) == expected_shape

    print("SegFormer test passed.")
    print("Device:", device)
    print("Output:", tuple(output.shape))


if __name__ == "__main__":
    main()
