from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import ClassificationConfig
from .roi_utils import (
    crop_roi,
    find_matching_file,
    read_binary_mask,
    read_rgb_image,
)


def parse_classification_list(
    list_path: Path,
) -> List[Tuple[str, int]]:
    """
    Parse a classification split file.

    Supported line formats:
        658.JPG 5
        658.JPG,5
        658.JPG\t5
    """
    samples: List[Tuple[str, int]] = []

    with Path(list_path).open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            normalized = line.replace(",", " ")
            parts = normalized.split()

            if len(parts) < 2:
                raise ValueError(
                    f"Invalid line {line_number} in {list_path}: "
                    f"{raw_line!r}"
                )

            image_name = parts[0]

            try:
                label = int(parts[-1])
            except ValueError as error:
                raise ValueError(
                    f"Invalid class label on line {line_number} in "
                    f"{list_path}: {parts[-1]!r}"
                ) from error

            samples.append((image_name, label))

    if not samples:
        raise ValueError(f"No samples found in split file: {list_path}")

    return samples


def build_train_transform(
    image_size: int,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.90, 1.10),
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_eval_transform(
    image_size: int,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class OvarianClassificationDataset(Dataset):
    """
    Dataset supporting three image-input modes:

    full_image:
        Use the entire ultrasound image.

    ground_truth_roi:
        Crop the image using its ground-truth segmentation mask.

    predicted_roi:
        Crop the image using a predicted segmentation mask.
    """

    def __init__(
        self,
        image_dir: Path,
        list_path: Path,
        num_classes: int,
        transform: Optional[Callable] = None,
        input_mode: str = "full_image",
        mask_dir: Optional[Path] = None,
        predicted_mask_dir: Optional[Path] = None,
        roi_padding: float = 0.10,
        mask_threshold: float = 0.5,
        use_masked_roi: bool = False,
        return_metadata: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.list_path = Path(list_path)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.predicted_mask_dir = (
            Path(predicted_mask_dir)
            if predicted_mask_dir is not None
            else None
        )

        self.num_classes = num_classes
        self.transform = transform
        self.input_mode = input_mode
        self.roi_padding = roi_padding
        self.mask_threshold = mask_threshold
        self.use_masked_roi = use_masked_roi
        self.return_metadata = return_metadata

        valid_modes = {
            "full_image",
            "ground_truth_roi",
            "predicted_roi",
        }
        if input_mode not in valid_modes:
            raise ValueError(
                f"input_mode must be one of {sorted(valid_modes)}"
            )

        if input_mode == "ground_truth_roi" and self.mask_dir is None:
            raise ValueError(
                "mask_dir is required for ground_truth_roi mode."
            )

        if input_mode == "predicted_roi" and self.predicted_mask_dir is None:
            raise ValueError(
                "predicted_mask_dir is required for predicted_roi mode."
            )

        self.samples = parse_classification_list(self.list_path)

        invalid_labels = [
            label
            for _, label in self.samples
            if label < 0 or label >= self.num_classes
        ]
        if invalid_labels:
            raise ValueError(
                f"Found invalid labels: {sorted(set(invalid_labels))}. "
                f"Expected labels from 0 to {self.num_classes - 1}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> List[int]:
        return [label for _, label in self.samples]

    def class_counts(self) -> Dict[int, int]:
        counter = Counter(self.labels)
        return {
            class_index: counter.get(class_index, 0)
            for class_index in range(self.num_classes)
        }

    def _load_image_and_roi(
        self,
        image_name: str,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        image_path = find_matching_file(self.image_dir, image_name)
        image = read_rgb_image(image_path)

        height, width = image.shape[:2]
        full_box = (0, 0, width, height)

        if self.input_mode == "full_image":
            return image, full_box

        selected_mask_dir = (
            self.mask_dir
            if self.input_mode == "ground_truth_roi"
            else self.predicted_mask_dir
        )

        assert selected_mask_dir is not None

        mask_path = find_matching_file(
            selected_mask_dir,
            image_name,
        )
        mask = read_binary_mask(
            mask_path,
            threshold=self.mask_threshold,
        )

        roi, box = crop_roi(
            image=image,
            mask=mask,
            padding=self.roi_padding,
            use_masked_roi=self.use_masked_roi,
        )
        return roi, box

    def __getitem__(self, index: int):
        image_name, label = self.samples[index]
        image_array, roi_box = self._load_image_and_roi(image_name)

        image = Image.fromarray(image_array)

        if self.transform is not None:
            image = self.transform(image)

        if self.return_metadata:
            metadata = {
                "image_name": image_name,
                "roi_box": torch.tensor(roi_box, dtype=torch.int64),
            }
            return image, label, metadata

        return image, label


def calculate_class_weights(
    labels: Sequence[int],
    num_classes: int,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.

    weight_c = N / (C * count_c)

    Classes absent from the training split receive weight 0.
    """
    counts = np.bincount(
        np.asarray(labels, dtype=np.int64),
        minlength=num_classes,
    )
    total_samples = int(counts.sum())

    weights = np.zeros(num_classes, dtype=np.float32)

    for class_index, count in enumerate(counts):
        if count > 0:
            weights[class_index] = (
                total_samples / (num_classes * int(count))
            )

    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    config: ClassificationConfig,
) -> Tuple[DataLoader, DataLoader, OvarianClassificationDataset,
           OvarianClassificationDataset]:
    train_dataset = OvarianClassificationDataset(
        image_dir=config.image_dir,
        list_path=config.train_list,
        num_classes=config.num_classes,
        transform=build_train_transform(config.image_size),
        input_mode=config.input_mode,
        mask_dir=config.mask_dir,
        predicted_mask_dir=config.predicted_mask_dir,
        roi_padding=config.roi_padding,
        mask_threshold=config.mask_threshold,
        use_masked_roi=config.use_masked_roi,
        return_metadata=False,
    )

    val_dataset = OvarianClassificationDataset(
        image_dir=config.image_dir,
        list_path=config.val_list,
        num_classes=config.num_classes,
        transform=build_eval_transform(config.image_size),
        input_mode=config.input_mode,
        mask_dir=config.mask_dir,
        predicted_mask_dir=config.predicted_mask_dir,
        roi_padding=config.roi_padding,
        mask_threshold=config.mask_threshold,
        use_masked_roi=config.use_masked_roi,
        return_metadata=False,
    )

    persistent_workers = config.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, train_dataset, val_dataset
