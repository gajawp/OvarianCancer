"""
Configuration for the MMOTU 8-class classification experiment with MTA-Swin.

This mirrors experiments/comparison/config.py but targets the MMOTU OTU_2d
dataset (flat images + *_cls.txt label files) and runs a single model:
MTA-Swin (ImageNet-1K pretrained). Large assets (dataset, pretrained weights)
are read through environment variables so the repo stays portable; sensible
repo-relative defaults are provided for local use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


MMOTU_DIR = Path(__file__).resolve().parent          # experiments/mmotu_cls
MTA_ROOT = MMOTU_DIR.parents[1]                       # mta-swin-code (has src/, best_model.pth)
REPO_ROOT = MMOTU_DIR.parents[2]                      # project root (has OTU_2d/)

IMAGE_DIR_ENV_VAR = "MMOTU_IMAGE_DIR"
TRAIN_CLS_ENV_VAR = "MMOTU_TRAIN_CLS"
VAL_CLS_ENV_VAR = "MMOTU_VAL_CLS"
PRETRAINED_WEIGHTS_ENV_VAR = "MTA_PRETRAINED_WEIGHTS"

DEFAULT_IMAGE_DIR = REPO_ROOT / "OTU_2d" / "images"
DEFAULT_TRAIN_CLS = REPO_ROOT / "OTU_2d" / "train_cls.txt"
DEFAULT_VAL_CLS = REPO_ROOT / "OTU_2d" / "val_cls.txt"
DEFAULT_PRETRAINED_WEIGHTS = MTA_ROOT / "best_model.pth"

# Inferred 0-7 -> category mapping (paper category order; index 5 = Normal
# Ovary is corroborated by its absence from the CEUS/OTU_3d split). Used only
# for confusion-matrix / per-class table labels, not for any logic.
CLASS_NAMES: tuple[str, ...] = (
    "0-CC",    # chocolate cyst
    "1-SC",    # serous cystadenoma
    "2-T",     # teratoma
    "3-TCT",   # theca cell tumor
    "4-SCH",   # simple cyst
    "5-NO",    # normal ovary
    "6-MC",    # mucinous cystadenoma
    "7-HGSC",  # high-grade serous cystadenoma
)


def _path_from_env(env_var: str, default: Path) -> Path:
    value = os.environ.get(env_var)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class MMOTUConfig:
    image_dir: Path = field(default_factory=lambda: _path_from_env(IMAGE_DIR_ENV_VAR, DEFAULT_IMAGE_DIR))
    train_cls_path: Path = field(default_factory=lambda: _path_from_env(TRAIN_CLS_ENV_VAR, DEFAULT_TRAIN_CLS))
    val_cls_path: Path = field(default_factory=lambda: _path_from_env(VAL_CLS_ENV_VAR, DEFAULT_VAL_CLS))
    pretrained_weights_path: Path = field(
        default_factory=lambda: _path_from_env(PRETRAINED_WEIGHTS_ENV_VAR, DEFAULT_PRETRAINED_WEIGHTS)
    )

    # Training hyperparameters (identical to the comparison experiment,
    # except num_classes 4 -> 8).
    batch_size: int = 32
    num_epochs: int = 200
    learning_rate: float = 1e-4
    min_lr: float = 1e-7
    num_classes: int = 8
    target_size: tuple[int, int] = (224, 224)
    label_smoothing: float = 0.1
    weight_decay: float = 1e-4
    reduce_factor: float = 0.5
    reduce_patience: int = 7
    early_stopping_patience: int = 20
    validation_split: float = 0.2
    num_workers: int = 8
    pin_memory: bool = True

    # Fixed seed for the train/val split and all RNGs (per request).
    seed: int = 42

    # MTA-Swin stage configuration -- must match how best_model.pth was
    # pretrained (this is the comparison-config setup).
    mta_stage_qk_conv: tuple[bool, bool, bool, bool] = (True, True, False, False)
    mta_stage_head_mixing: tuple[bool, bool, bool, bool] = (False, False, True, True)
    mta_stage_group_norm: tuple[bool, bool, bool, bool] = (True, True, False, False)
    mta_cq: int = 5
    mta_ck: int = 11
    mta_ch: int = 2
    mta_drop_path_rate: float = 0.1

    # Outputs
    output_dir: Path = MMOTU_DIR / "outputs"
    checkpoint_dir: Path = MMOTU_DIR / "outputs" / "checkpoints"
    plot_dir: Path = MMOTU_DIR / "outputs" / "plots"


DEFAULT_CONFIG = MMOTUConfig()
