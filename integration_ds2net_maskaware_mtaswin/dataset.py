import random
from pathlib import Path

import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


class MaskAwareClassificationDataset(Dataset):
    def __init__(
        self,
        image_directory: str | Path,
        mask_directory: str | Path,
        split_file: str | Path,
        image_size: int,
        training: bool,
    ):
        self.image_directory = Path(image_directory)
        self.mask_directory = Path(mask_directory)
        self.split_file = Path(split_file)
        self.image_size = image_size
        self.training = training

        for path in (
            self.image_directory,
            self.mask_directory,
            self.split_file,
        ):
            if not path.exists():
                raise FileNotFoundError(f"Required path not found: {path}")

        self.samples = self._read_split_file()

    def _read_split_file(self) -> list[tuple[str, str, int]]:
        samples = []

        with self.split_file.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(
                        f"Expected image mask label on line "
                        f"{line_number}: {line!r}"
                    )

                image_name, mask_name, label_text = parts
                label = int(label_text)

                image_path = self.image_directory / image_name
                mask_path = self.mask_directory / mask_name

                if not image_path.exists():
                    raise FileNotFoundError(f"RGB ROI not found: {image_path}")
                if not mask_path.exists():
                    raise FileNotFoundError(f"Mask ROI not found: {mask_path}")

                samples.append((image_name, mask_name, label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _transform(
        self,
        image: Image.Image,
        mask: Image.Image,
    ):
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        if self.training:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            angle = random.uniform(-10.0, 10.0)

            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )

            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

            image = TF.adjust_brightness(
                image,
                random.uniform(0.9, 1.1),
            )

            image = TF.adjust_contrast(
                image,
                random.uniform(0.9, 1.1),
            )

        image_tensor = TF.to_tensor(image)

        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        mask_tensor = (TF.to_tensor(mask) >= 0.5).float()

        return image_tensor, mask_tensor

    def __getitem__(self, index: int):
        image_name, mask_name, label = self.samples[index]

        image = Image.open(
            self.image_directory / image_name
        ).convert("RGB")

        mask = Image.open(
            self.mask_directory / mask_name
        ).convert("L")

        image_tensor, mask_tensor = self._transform(
            image,
            mask,
        )

        return image_tensor, mask_tensor, label, image_name
