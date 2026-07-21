from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import torch


@dataclass
class ClassificationConfig:
    """
    Central configuration for ovarian tumor classification.

    Expected dataset structure
    --------------------------
    datasets/OTU_2d/
    ├── images/
    ├── annotations/
    ├── train_cls.txt
    └── val_cls.txt

    Split-file format
    -----------------
    658.JPG 5
    384.JPG 3
    """

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )

    dataset_dir: Path | None = None
    image_dir: Path | None = None
    mask_dir: Path | None = None
    train_list: Path | None = None
    val_list: Path | None = None

    results_dir: Path | None = None
    checkpoint_dir: Path | None = None
    metrics_dir: Path | None = None
    prediction_dir: Path | None = None
    confusion_matrix_dir: Path | None = None

    # Dataset
    num_classes: int = 8
    image_size: int = 224
    input_mode: str = "ground_truth_roi"
    roi_padding: float = 0.10
    mask_threshold: float = 0.5
    use_masked_roi: bool = False

    # Training
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    label_smoothing: float = 0.0
    use_class_weights: bool = True
    pretrained: bool = True
    dropout: float = 0.30
    random_seed: int = 42

    # Evaluation
    top_k: Tuple[int, ...] = (1, 3)

    # Predicted-mask mode
    predicted_mask_dir: Path | None = None

    class_names: Dict[int, str] = field(
        default_factory=lambda: {
            0: "Chocolate cyst",
            1: "Serous cystadenoma",
            2: "Teratoma",
            3: "Theca cell tumor",
            4: "Simple cyst",
            5: "Normal ovary",
            6: "Mucinous cystadenoma",
            7: "High-grade serous carcinoma",
        }
    )

    def __post_init__(self) -> None:
        if self.dataset_dir is None:
            self.dataset_dir = self.project_root / "datasets" / "OTU_2d"

        if self.image_dir is None:
            self.image_dir = self.dataset_dir / "images"

        if self.mask_dir is None:
            self.mask_dir = self.dataset_dir / "annotations"

        if self.train_list is None:
            self.train_list = self.dataset_dir / "train_cls.txt"

        if self.val_list is None:
            self.val_list = self.dataset_dir / "val_cls.txt"

        if self.results_dir is None:
            self.results_dir = self.project_root / "classification_results"

        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.results_dir / "checkpoints"

        if self.metrics_dir is None:
            self.metrics_dir = self.results_dir / "metrics"

        if self.prediction_dir is None:
            self.prediction_dir = self.results_dir / "predictions"

        if self.confusion_matrix_dir is None:
            self.confusion_matrix_dir = (
                self.results_dir / "confusion_matrices"
            )

        valid_modes = {
            "full_image",
            "ground_truth_roi",
            "predicted_roi",
        }
        if self.input_mode not in valid_modes:
            raise ValueError(
                f"input_mode must be one of {sorted(valid_modes)}, "
                f"but received {self.input_mode!r}"
            )

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "resnet50_best.pth"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "resnet50_last.pth"

    def create_output_directories(self) -> None:
        for directory in (
            self.results_dir,
            self.checkpoint_dir,
            self.metrics_dir,
            self.prediction_dir,
            self.confusion_matrix_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def validate_paths(self, require_masks: bool = True) -> None:
        required = [
            self.image_dir,
            self.train_list,
            self.val_list,
        ]

        if require_masks and self.input_mode == "ground_truth_roi":
            required.append(self.mask_dir)

        if self.input_mode == "predicted_roi":
            if self.predicted_mask_dir is None:
                raise ValueError(
                    "predicted_mask_dir must be set when "
                    "input_mode='predicted_roi'."
                )
            required.append(self.predicted_mask_dir)

        missing = [str(path) for path in required if not path.exists()]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "The following required paths do not exist:\n"
                f"{formatted}"
            )


def get_config() -> ClassificationConfig:
    """
    Return the default classification configuration.

    Modify the returned object in train/evaluate/predict when running a
    different experiment.
    """
    config = ClassificationConfig()
    config.create_output_directories()
    return config
