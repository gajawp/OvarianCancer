from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass(frozen=True)
class Config:
    project_root: Path = Path(__file__).resolve().parents[2]
    image_dir: Path = project_root / "datasets" / "OTU_2d" / "images"
    mask_dir: Path = project_root / "datasets" / "OTU_2d" / "annotations"
    image_size: int = 256
    batch_size: int = 4
    epochs: int = 50
    warmup_epochs: int = 5
    encoder_lr: float = 1e-5
    decoder_lr: float = 1e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.20
    seed: int = 42
    threshold: float = 0.50
    patience: int = 12
    num_workers: int = 0
    mask_suffix: str = "_binary.PNG"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


CFG = Config()
