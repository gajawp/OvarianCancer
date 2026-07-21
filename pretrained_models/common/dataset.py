from pathlib import Path
from typing import List, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, Subset

from .config import CFG


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_images(image_dir: Path) -> List[Path]:
    files = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return files


def make_transforms(train: bool) -> A.Compose:
    operations = [A.Resize(CFG.image_size, CFG.image_size)]
    if train:
        operations.extend([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ])
    operations.extend([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    return A.Compose(operations)


class OvarianDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], mask_dir: Path, train: bool):
        self.image_paths = list(image_paths)
        self.mask_dir = Path(mask_dir)
        self.transform = make_transforms(train)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _mask_path(self, image_path: Path) -> Path:
        return self.mask_dir / f"{image_path.stem}{CFG.mask_suffix}"

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        image_path = self.image_paths[index]
        mask_path = self._mask_path(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 0).astype(np.float32)
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"].float()
        mask_tensor = transformed["mask"].float().unsqueeze(0)
        return image_tensor, mask_tensor, image_path.name


def split_indices(length: int) -> Tuple[List[int], List[int]]:
    generator = torch.Generator().manual_seed(CFG.seed)
    permutation = torch.randperm(length, generator=generator).tolist()
    val_count = max(1, int(round(length * CFG.val_fraction)))
    return permutation[val_count:], permutation[:val_count]


def build_loaders() -> Tuple[DataLoader, DataLoader]:
    image_paths = list_images(CFG.image_dir)
    train_indices, val_indices = split_indices(len(image_paths))
    train_dataset = OvarianDataset(image_paths, CFG.mask_dir, train=True)
    val_dataset = OvarianDataset(image_paths, CFG.mask_dir, train=False)

    train_loader = DataLoader(
        Subset(train_dataset, train_indices), batch_size=CFG.batch_size,
        shuffle=True, num_workers=CFG.num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices), batch_size=CFG.batch_size,
        shuffle=False, num_workers=CFG.num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
