from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


COMPARISON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = COMPARISON_DIR.parents[1]
IMAGE_PATH_ENV_VAR = "MTA_COMPARISON_IMAGE_PATH"
PRETRAINED_WEIGHTS_ENV_VAR = "MTA_COMPARISON_PRETRAINED_WEIGHTS"
DEFAULT_IMAGE_PATH = COMPARISON_DIR / "external" / "cleaned-brain-tumour-dataset"
DEFAULT_PRETRAINED_WEIGHTS_PATH = COMPARISON_DIR / "external" / "best_model.pth"


ModelSpec = tuple[str, str]


def _path_from_env(env_var: str, default: Path) -> Path:
    value = os.environ.get(env_var)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class ComparisonConfig:
    image_path: Path = field(default_factory=lambda: _path_from_env(IMAGE_PATH_ENV_VAR, DEFAULT_IMAGE_PATH))
    pretrained_weights_path: Path = field(
        default_factory=lambda: _path_from_env(PRETRAINED_WEIGHTS_ENV_VAR, DEFAULT_PRETRAINED_WEIGHTS_PATH)
    )

    batch_size: int = 32
    num_epochs: int = 200
    learning_rate: float = 1e-4
    min_lr: float = 1e-7
    num_classes: int = 4
    target_size: tuple[int, int] = (224, 224)
    label_smoothing: float = 0.1
    weight_decay: float = 1e-4
    reduce_factor: float = 0.5
    reduce_patience: int = 7
    early_stopping_patience: int = 20
    validation_split: float = 0.2
    num_workers: int = 8
    pin_memory: bool = True

    default_seeds: tuple[int, int, int] = (0, 1, 2)
    seed_run_dir: Path = COMPARISON_DIR / "mta_seed_runs"
    final_table_dir: Path = COMPARISON_DIR / "final_tables"
    checkpoint_root: Path = COMPARISON_DIR / "checkpoints"
    plot_root: Path = COMPARISON_DIR / "plots"

    bootstrap_iterations: int = 2000
    confidence_interval: float = 95.0
    bootstrap_seed: int = 0

    mta_stage_qk_conv: tuple[bool, bool, bool, bool] = (True, True, False, False)
    mta_stage_head_mixing: tuple[bool, bool, bool, bool] = (False, False, True, True)
    mta_stage_group_norm: tuple[bool, bool, bool, bool] = (True, True, False, False)
    mta_cq: int = 5
    mta_ck: int = 11
    mta_ch: int = 2
    mta_drop_path_rate: float = 0.1

    model_specs: tuple[ModelSpec, ...] = (
        ("Custom CNN", "scratch"),
        ("ResNet-50", "pretrained"),
        ("EfficientNet-B4", "pretrained"),
        ("ConvNeXt-T", "pretrained"),
        ("DeiT-S/16", "pretrained"),
        ("ViT-S/16", "pretrained"),
        ("Swin-T", "scratch"),
        ("Swin-T", "pretrained"),
        ("mamba", "pretrained"),
        ("maxivt", "pretrained"),
        ("davit", "pretrained"),
        ("cait", "pretrained"),
        ("inceptionnext", "pretrained"),
        ("swinv2", "pretrained"),
        ("MTA-Swin", "scratch"),
        ("MTA-Swin", "pretrained"),
    )


DEFAULT_CONFIG = ComparisonConfig()
