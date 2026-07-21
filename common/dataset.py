import os

import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset


class OvarianDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        split_file,
        image_size=256,
        transform=None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.split_file = split_file
        self.image_size = image_size
        self.transform = transform

        self.images = self._read_split_file(
            split_file
        )

        if len(self.images) == 0:
            raise ValueError(
                f"No images found in split file: "
                f"{split_file}"
            )

    @staticmethod
    def _read_split_file(split_file):
        images = []

        with open(
            split_file,
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) < 1:
                    raise ValueError(
                        f"Invalid line {line_number} "
                        f"in {split_file}"
                    )

                # Ignore the classification label.
                image_name = parts[0]

                images.append(image_name)

        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name,
        )

        base_name = os.path.splitext(
            image_name
        )[0]

        mask_name = (
            base_name
            + "_binary.PNG"
        )

        mask_path = os.path.join(
            self.mask_dir,
            mask_name,
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: "
                f"{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise FileNotFoundError(
                f"Could not read mask: "
                f"{mask_path}"
            )

        image = cv2.resize(
            image,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        mask = (
            mask > 0
        ).astype("float32")

        if self.transform:
            augmented = self.transform(
                image=image,
                mask=mask,
            )

            image = augmented["image"]
            mask = augmented["mask"]

        image = (
            image.astype("float32")
            / 255.0
        )

        image = image.transpose(
            2,
            0,
            1,
        )

        mask = mask[
            None,
            :,
            :,
        ]

        return (
            torch.tensor(
                image,
                dtype=torch.float32,
            ),
            torch.tensor(
                mask,
                dtype=torch.float32,
            ),
            image_name,
        )


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(
            limit=20,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            p=0.3,
        ),
    ])