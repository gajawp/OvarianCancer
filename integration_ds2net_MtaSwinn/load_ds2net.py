# integrated_ds2net_mtaswin/load_ds2net.py

from pathlib import Path

import torch

from common import config
from common.utils import load_model_checkpoint
from ds2net.model import DS2NetSingleDomain


def load_trained_ds2net(
    device: torch.device,
    checkpoint_path: str | Path | None = None,
) -> torch.nn.Module:
    """
    Load the already-trained DS2Net segmentation model.

    The architecture and checkpoint-loading process match
    the existing ds2net training and evaluation files.
    """

    if checkpoint_path is None:
        checkpoint_path = config.DS2NET_MODEL_PATH

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DS2Net checkpoint not found: {checkpoint_path}"
        )

    model = DS2NetSingleDomain(
        in_channels=3,
        out_channels=1,
    ).to(device)

    model = load_model_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model.eval()

    # DS2Net is only used to generate ROIs.
    # It should not be updated while training MTA-Swin.
    for parameter in model.parameters():
        parameter.requires_grad = False

    print("Loaded DS2Net checkpoint:", checkpoint_path)

    return model