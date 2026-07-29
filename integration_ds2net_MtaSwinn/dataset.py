# integration_ds2net_MtaSwinn/dataset.py

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


class DS2NetROIClassificationDataset(Dataset):
    """
    Classification dataset using DS2Net-generated ROI images.

    Expected split-file format:

        658.JPG 5
        384.JPG 3
        367.JPG 3
    """

    def __init__(
        self,
        image_directory: str | Path,
        split_file: str | Path,
        transform: Callable | None = None,
    ):
        self.image_directory = Path(
            image_directory
        )

        self.split_file = Path(
            split_file
        )

        self.transform = transform

        if not self.image_directory.exists():
            raise FileNotFoundError(
                "ROI image directory does not exist: "
                f"{self.image_directory}"
            )

        if not self.split_file.exists():
            raise FileNotFoundError(
                "ROI split file does not exist: "
                f"{self.split_file}"
            )

        self.samples = self._read_split_file()

        if len(self.samples) == 0:
            raise RuntimeError(
                "No classification samples were found in: "
                f"{self.split_file}"
            )

    def _read_split_file(
        self,
    ) -> list[tuple[str, int]]:
        samples = []

        with self.split_file.open(
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

                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid line {line_number} in "
                        f"{self.split_file}: {line!r}"
                    )

                image_name = parts[0]

                try:
                    label = int(parts[1])
                except ValueError as error:
                    raise ValueError(
                        f"Invalid class label on line "
                        f"{line_number}: {line!r}"
                    ) from error

                image_path = (
                    self.image_directory
                    / image_name
                )

                if not image_path.exists():
                    raise FileNotFoundError(
                        f"ROI listed in split file was "
                        f"not found: {image_path}"
                    )

                samples.append(
                    (
                        image_name,
                        label,
                    )
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ):
        image_name, label = (
            self.samples[index]
        )

        image_path = (
            self.image_directory
            / image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(
                image
            )

        return (
            image,
            label,
            image_name,
        )