import random
import shutil
from pathlib import Path

from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from integration_ds2net_maskaware_mtaswin_aug_focal_binary_malignant_aug import config


def augment_pair(image, mask, rng):
    if rng.random() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    angle = rng.uniform(-config.ROTATION_LIMIT_DEGREES, config.ROTATION_LIMIT_DEGREES)
    max_shift = int(config.IMAGE_SIZE * config.TRANSLATION_LIMIT_FRACTION)
    translate = [rng.randint(-max_shift, max_shift), rng.randint(-max_shift, max_shift)]
    scale = rng.uniform(config.SCALE_MIN, config.SCALE_MAX)

    image = TF.affine(
        image, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR, fill=0,
    )
    mask = TF.affine(
        mask, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
        interpolation=InterpolationMode.NEAREST, fill=0,
    )

    image = TF.adjust_brightness(
        image, rng.uniform(config.BRIGHTNESS_MIN, config.BRIGHTNESS_MAX)
    )
    image = TF.adjust_contrast(
        image, rng.uniform(config.CONTRAST_MIN, config.CONTRAST_MAX)
    )
    return image, mask


def read_samples():
    samples = []
    with config.ORIGINAL_TRAIN_LIST.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(f"Invalid line {line_number}: {line!r}")

            image_name, mask_name, label_text = parts
            label = int(label_text)

            image_path = config.ORIGINAL_TRAIN_RGB_DIR / image_name
            mask_path = config.ORIGINAL_TRAIN_MASK_DIR / mask_name
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)

            samples.append((image_name, mask_name, label))
    return samples


def main():
    rng = random.Random(config.MALIGNANT_AUGMENTATION_SEED)
    samples = read_samples()

    if config.MALIGNANT_AUG_DATASET_ROOT.exists():
        shutil.rmtree(config.MALIGNANT_AUG_DATASET_ROOT)

    config.TRAIN_RGB_DIR.mkdir(parents=True, exist_ok=True)
    config.TRAIN_MASK_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    malignant_originals = 0
    augmented_created = 0

    for image_name, mask_name, label in samples:
        source_image = config.ORIGINAL_TRAIN_RGB_DIR / image_name
        source_mask = config.ORIGINAL_TRAIN_MASK_DIR / mask_name

        shutil.copy2(source_image, config.TRAIN_RGB_DIR / image_name)
        shutil.copy2(source_mask, config.TRAIN_MASK_DIR / mask_name)
        rows.append(f"{image_name} {mask_name} {label}")

        if label != config.MALIGNANT_ORIGINAL_LABEL:
            continue

        malignant_originals += 1
        with Image.open(source_image) as image_file:
            image = image_file.convert("RGB")
        with Image.open(source_mask) as mask_file:
            mask = mask_file.convert("L")

        image_stem, image_suffix = Path(image_name).stem, Path(image_name).suffix
        mask_stem, mask_suffix = Path(mask_name).stem, Path(mask_name).suffix

        for copy_index in range(1, config.MALIGNANT_AUGMENTED_COPIES + 1):
            aug_image, aug_mask = augment_pair(image.copy(), mask.copy(), rng)
            new_image_name = f"{image_stem}_malaug{copy_index}{image_suffix}"
            new_mask_name = f"{mask_stem}_malaug{copy_index}{mask_suffix}"

            aug_image.save(config.TRAIN_RGB_DIR / new_image_name)
            aug_mask.save(config.TRAIN_MASK_DIR / new_mask_name)
            rows.append(f"{new_image_name} {new_mask_name} {label}")
            augmented_created += 1

    config.TRAIN_LIST.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print("=" * 78)
    print("MALIGNANT-ONLY OFFLINE AUGMENTATION COMPLETE")
    print("=" * 78)
    print("Original training samples :", len(samples))
    print("Original malignant samples:", malignant_originals)
    print("Copies per malignant image:", config.MALIGNANT_AUGMENTED_COPIES)
    print("New malignant images      :", augmented_created)
    print("Expanded training samples :", len(rows))
    print("Expanded malignant samples:", malignant_originals + augmented_created)
    print("Output dataset:", config.MALIGNANT_AUG_DATASET_ROOT)
    print("New training list:", config.TRAIN_LIST)


if __name__ == "__main__":
    main()
