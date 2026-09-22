import random
from pathlib import Path

import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from multiseed_ablation_studies.seedstudy_reference import config


class ROIMaskBinaryDataset(Dataset):
    def __init__(
        self,
        roi_directory: str | Path,
        mask_directory: str | Path,
        split_file: str | Path,
        image_size: int,
        training: bool,
    ):
        self.roi_directory = Path(roi_directory)
        self.mask_directory = Path(mask_directory)
        self.split_file = Path(split_file)
        self.image_size = image_size
        self.training = training

        for path in (
            self.roi_directory,
            self.mask_directory,
            self.split_file,
        ):
            if not path.exists():
                raise FileNotFoundError(f"Required path not found: {path}")

        self.samples = self._read_split_file()

    def _read_split_file(self):
        samples = []

        with self.split_file.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 3:
                    raise ValueError(
                        f"Expected image mask label on line "
                        f"{line_number}: {line!r}"
                    )

                image_name, mask_name, label_text = parts
                original_label = int(label_text)

                if original_label not in config.BINARY_LABEL_MAP:
                    raise ValueError(
                        f"Unsupported original class {original_label}"
                    )

                roi_path = self.roi_directory / image_name
                mask_path = self.mask_directory / mask_name

                if not roi_path.exists():
                    raise FileNotFoundError(f"ROI not found: {roi_path}")
                if not mask_path.exists():
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                samples.append(
                    (
                        image_name,
                        mask_name,
                        config.BINARY_LABEL_MAP[original_label],
                        original_label,
                    )
                )

        if not samples:
            raise ValueError(f"No samples found in {self.split_file}")

        return samples

    def __len__(self):
        return len(self.samples)

    def _resize_all(self, roi, mask):
        roi = TF.resize(
            roi,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        return roi, mask

    def _augment(self, roi, mask):
        if random.random() < config.HORIZONTAL_FLIP_PROBABILITY:
            roi = TF.hflip(roi)
            mask = TF.hflip(mask)

        if random.random() < config.AFFINE_PROBABILITY:
            angle = random.uniform(
                -config.ROTATION_LIMIT_DEGREES,
                config.ROTATION_LIMIT_DEGREES,
            )
            max_shift = int(
                self.image_size
                * config.TRANSLATION_LIMIT_FRACTION
            )
            translate = [
                random.randint(-max_shift, max_shift),
                random.randint(-max_shift, max_shift),
            ]
            scale = random.uniform(
                config.SCALE_MIN,
                config.SCALE_MAX,
            )

            roi = TF.affine(
                roi,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        if random.random() < config.BRIGHTNESS_PROBABILITY:
            roi = TF.adjust_brightness(
                roi,
                random.uniform(
                    config.BRIGHTNESS_MIN,
                    config.BRIGHTNESS_MAX,
                ),
            )

        if random.random() < config.CONTRAST_PROBABILITY:
            roi = TF.adjust_contrast(
                roi,
                random.uniform(
                    config.CONTRAST_MIN,
                    config.CONTRAST_MAX,
                ),
            )
        return roi, mask

    @staticmethod
    def _to_normalized_tensor(image):
        tensor = TF.to_tensor(image)
        return TF.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __getitem__(self, index):
        image_name, mask_name, binary_label, original_label = (
            self.samples[index]
        )

        with Image.open(
            self.roi_directory / image_name
        ) as image_file:
            roi = image_file.convert("RGB")

        with Image.open(
            self.mask_directory / mask_name
        ) as mask_file:
            mask = mask_file.convert("L")

        roi, mask = self._resize_all(
            roi,
            mask,
        )

        if self.training and config.USE_AUGMENTATION:
            roi, mask = self._augment(
                roi,
                mask,
            )

        roi_tensor = self._to_normalized_tensor(roi)
        mask_tensor = (TF.to_tensor(mask) >= 0.5).float()

        return (
            roi_tensor,
            mask_tensor,
            binary_label,
            image_name,
            mask_name,
            original_label,
        )
